"""
NCCS Full Historical Downloader
=================================
Downloads ALL available NCCS Core Series files (Form 990 / 990-EZ / 990-PF)
from 1989 to 2025 for all US states.

Strategy:
1. Scan the S3 bucket to build a live manifest of available files
2. Download each discovered file (Parquet or CSV → Parquet)
3. Build a DuckDB catalogue for fast querying

This script can be interrupted and re-run safely (skips already-downloaded files).

Usage:
    python full_download.py                    # Download everything
    python full_download.py --scan-only        # Just scan and print manifest
    python full_download.py --year 2010 2011   # Specific years only
    python full_download.py --workers 4        # Parallel downloads
"""

import argparse
import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import duckdb
import pandas as pd
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.client import Config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUCKET = "nccsdata"
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = Path(__file__).parent / "nccs990.duckdb"
MANIFEST_PATH = DATA_DIR / "s3_manifest.json"

FORM_TYPES = ["PC", "EZ", "PF"]
TARGET_YEARS = list(range(1989, 2026))

# S3 top-level prefixes to scan
SCAN_PREFIXES = [
    "harmonized/core/",
    "processed_merged/core/",
    "raw/core/",
    "legacy/",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR.parent / "download.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def make_s3():
    return boto3.client(
        "s3", region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )


def list_all_keys(s3, prefix: str) -> list[dict]:
    """Return list of {Key, Size} for all objects under prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    results = []
    try:
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                results.append({"key": obj["Key"], "size": obj["Size"]})
    except Exception as exc:
        log.warning(f"Could not list {prefix}: {exc}")
    return results


def scan_all_prefixes() -> list[dict]:
    """Scan all SCAN_PREFIXES and return combined list of objects."""
    s3 = make_s3()
    all_objects = []
    for prefix in SCAN_PREFIXES:
        log.info(f"Scanning s3://{BUCKET}/{prefix}")
        objs = list_all_keys(s3, prefix)
        log.info(f"  → {len(objs)} objects")
        all_objects.extend(objs)

    # Save manifest
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(all_objects, indent=2))
    log.info(f"Manifest saved: {MANIFEST_PATH} ({len(all_objects)} total objects)")
    return all_objects


def filter_core_data_files(objects: list[dict]) -> list[dict]:
    """
    Filter to files that are likely 990 core data files.
    Accepts Parquet and CSV files; excludes tiny index/metadata files.
    """
    SKIP_EXTENSIONS = {".json", ".md", ".txt", ".html", ".zip"}
    MIN_SIZE = 1_000  # bytes – skip trivially small files

    core = []
    for obj in objects:
        key = obj["key"]
        size = obj["size"]
        if size < MIN_SIZE:
            continue
        ext = Path(key).suffix.lower()
        if ext in SKIP_EXTENSIONS:
            continue
        if ext not in {".parquet", ".csv"}:
            continue
        # Must mention a form type somewhere
        upper = key.upper()
        is_core = any(f"-{ft}-" in upper or f"/{ft}/" in upper or f"_{ft}_" in upper for ft in FORM_TYPES)
        if is_core:
            core.append(obj)
    return core


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def local_path_for_key(key: str) -> Path:
    """Determine local path for an S3 key (always .parquet)."""
    fname = Path(key).name
    if fname.endswith(".csv"):
        fname = fname[:-4] + ".parquet"
    return DATA_DIR / fname


def download_one(key: str, size: int) -> dict:
    """Download a single S3 object; convert CSV to Parquet."""
    local = local_path_for_key(key)
    # Skip if already downloaded
    if local.exists():
        return {"key": key, "status": "skipped", "rows": None}

    s3 = make_s3()
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        data = obj["Body"].read()

        local.parent.mkdir(parents=True, exist_ok=True)

        if key.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data), low_memory=False)
            df.to_parquet(local, index=False)
            rows = len(df)
        else:
            local.write_bytes(data)
            try:
                meta = pq.read_metadata(local)
                rows = meta.num_rows
            except Exception:
                rows = "?"

        log.info(f"✓ {Path(key).name}  ({rows} rows, {size/1e6:.1f} MB)")
        return {"key": key, "status": "ok", "rows": rows}

    except Exception as exc:
        log.error(f"✗ {key}: {exc}")
        return {"key": key, "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# DuckDB catalogue
# ---------------------------------------------------------------------------

def build_duckdb_catalogue():
    """
    Scan all local Parquet files and create a DuckDB database with:
      - file_catalogue table: metadata about each file
      - all_filings view: unified lazy union of all files
    """
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        log.warning("No Parquet files found – skipping DuckDB build.")
        return

    log.info(f"Building DuckDB catalogue from {len(parquet_files)} files…")

    rows = []
    for pf in sorted(parquet_files):
        fname = pf.stem  # e.g. CORE-PC-BMF-NONPROFIT-RGN-2022
        parts = fname.upper().split("-")
        year = next((int(p) for p in parts if p.isdigit() and len(p) == 4), None)
        form = next((p for p in parts if p in FORM_TYPES), "UNKNOWN")
        try:
            meta = pq.read_metadata(pf)
            num_rows = meta.num_rows
            num_cols = meta.num_columns
        except Exception:
            num_rows = num_cols = None
        rows.append({
            "filename": pf.name,
            "path": str(pf),
            "form_type": form,
            "year": year,
            "num_rows": num_rows,
            "num_columns": num_cols,
        })

    cat_df = pd.DataFrame(rows)

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE file_catalogue AS SELECT * FROM cat_df")
    log.info("Created table: file_catalogue")

    # Build UNION ALL view
    union_parts = []
    for _, row in cat_df.iterrows():
        escaped_path = row["path"].replace("'", "''")
        union_parts.append(
            f"SELECT *, '{row['form_type']}' AS _form_type, {row['year']} AS _filing_year "
            f"FROM parquet_scan('{escaped_path}')"
        )

    if union_parts:
        view_sql = (
            "CREATE OR REPLACE VIEW all_filings AS\n"
            + "\nUNION ALL\n".join(union_parts)
        )
        con.execute(view_sql)
        log.info("Created view: all_filings")

    # Summary stats
    summary = con.execute(
        "SELECT _form_type, _filing_year, COUNT(*) AS n FROM all_filings GROUP BY 1,2 ORDER BY 2,1"
    ).df()
    summary_path = DATA_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Summary saved to {summary_path}")
    print("\nData Summary:")
    print(summary.to_string(index=False))

    con.close()
    log.info(f"DuckDB catalogue: {DB_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NCCS Full Historical 990 Downloader")
    parser.add_argument("--scan-only", action="store_true", help="Scan bucket and exit")
    parser.add_argument("--year", type=int, nargs="+", help="Limit to specific years")
    parser.add_argument("--form", choices=FORM_TYPES, nargs="+", help="Limit to specific form types")
    parser.add_argument("--workers", type=int, default=2, help="Parallel download threads (default: 2)")
    parser.add_argument("--rebuild-db", action="store_true", help="Rebuild DuckDB catalogue only")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_db:
        build_duckdb_catalogue()
        return

    # Scan
    if MANIFEST_PATH.exists():
        log.info(f"Loading existing manifest: {MANIFEST_PATH}")
        all_objects = json.loads(MANIFEST_PATH.read_text())
    else:
        all_objects = scan_all_prefixes()

    if args.scan_only:
        core = filter_core_data_files(all_objects)
        print(f"\nFound {len(core)} core data files:")
        for obj in core:
            print(f"  {obj['key']}  ({obj['size']/1e6:.1f} MB)")
        return

    # Filter
    core_files = filter_core_data_files(all_objects)
    log.info(f"Core data files found: {len(core_files)}")

    if args.year:
        core_files = [o for o in core_files if any(str(y) in o["key"] for y in args.year)]
    if args.form:
        core_files = [
            o for o in core_files
            if any(f"-{f}-" in o["key"].upper() or f"/{f}/" in o["key"].upper() for f in args.form)
        ]

    log.info(f"Files to download after filtering: {len(core_files)}")

    if not core_files:
        log.warning("No files match – try running with --scan-only to see what's available.")
        return

    # Download (parallel)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, o["key"], o["size"]): o for o in core_files}
        for future in as_completed(futures):
            results.append(future.result())

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    log.info(f"Download complete — OK: {ok}, Skipped: {skipped}, Errors: {errors}")

    # Build catalogue
    build_duckdb_catalogue()


if __name__ == "__main__":
    main()

