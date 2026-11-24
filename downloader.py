"""
NCCS 990 Dataset Downloader
============================
Downloads NCCS Core Series (Form 990 / 990-EZ / 990-PF) from the public
AWS S3 bucket (nccsdata) for all available years (1989–2025).

Data is saved as Parquet files locally and catalogued in a DuckDB database
so the Streamlit app can query it efficiently without loading everything into RAM.

Usage:
    python downloader.py               # download all years & form types
    python downloader.py --year 2022   # specific year only
    python downloader.py --dry-run     # list what would be downloaded
"""

import argparse
import io
import json
import logging
import os
import time
from pathlib import Path

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from botocore import UNSIGNED
from botocore.client import Config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BUCKET = "nccsdata"
# Known prefixes for the NCCS processed / harmonised core files.
# The bucket layout is: raw/core/{year}/{FORMTYPE}/
# and harmonised: harmonized/core/CORE-{FORMTYPE}-BMF-NONPROFIT-RGN/

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = Path(__file__).parent / "nccs990.duckdb"

# NCCS currently publishes core files for these form variants:
FORM_TYPES = ["PC", "EZ", "PF"]   # PC = 990 (public charity), EZ = 990-EZ, PF = 990-PF

# Filing years available (NCCS core: 1989–2022 paper-scanned; e-file from ~2012)
YEARS = list(range(1989, 2026))

# S3 path patterns tried in order (the bucket layout has changed over time)
S3_PATTERNS = [
    # Harmonised parquet (most recent layout)
    "harmonized/core/CORE-{form}-BMF-NONPROFIT-RGN/CORE-{form}-BMF-NONPROFIT-RGN-{year}.parquet",
    # Processed merged
    "processed_merged/core/{year}/{form}/CORE-{form}-{year}.parquet",
    # Legacy CSV
    "raw/core/{year}/{form}/CORE-{form}-{year}.csv",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def get_s3_client():
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )


def list_all_keys(s3, prefix=""):
    """List every key under a prefix in the public bucket."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def discover_available_files(s3):
    """
    Walk the bucket to discover all available core data files.
    Returns a list of dicts: {year, form, s3_key, size_bytes}
    """
    log.info("Scanning S3 bucket for available files (this may take ~30 s)…")
    prefixes_to_scan = [
        "harmonized/core/",
        "processed_merged/core/",
        "raw/core/",
    ]
    all_keys = []
    for prefix in prefixes_to_scan:
        try:
            keys = list_all_keys(s3, prefix)
            all_keys.extend(keys)
            log.info(f"  {prefix}: {len(keys)} objects")
        except Exception as exc:
            log.warning(f"  Could not scan {prefix}: {exc}")

    # Save manifest for later reference
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(all_keys, indent=2))
    log.info(f"Manifest saved to {manifest_path}")
    return all_keys


def download_file(s3, key: str, local_path: Path) -> bool:
    """Download a single S3 object; returns True on success."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        data = obj["Body"].read()

        # If it's CSV, convert to Parquet for uniform downstream handling
        if key.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data), low_memory=False)
            parquet_path = local_path.with_suffix(".parquet")
            df.to_parquet(parquet_path, index=False)
            log.info(f"  CSV→Parquet saved: {parquet_path.name}  ({len(df):,} rows)")
        else:
            local_path.write_bytes(data)
            # Quick row-count via pyarrow metadata
            try:
                meta = pq.read_metadata(local_path)
                log.info(f"  Saved: {local_path.name}  ({meta.num_rows:,} rows)")
            except Exception:
                log.info(f"  Saved: {local_path.name}")
        return True
    except Exception as exc:
        log.error(f"  Failed {key}: {exc}")
        return False


def build_catalogue(data_dir: Path, db_path: Path):
    """
    Create / refresh the DuckDB catalogue by scanning all local Parquet files.
    Creates a single virtual view that unions everything for fast ad-hoc queries.
    """
    parquet_files = list(data_dir.rglob("*.parquet"))
    if not parquet_files:
        log.warning("No Parquet files found – catalogue not built.")
        return

    log.info(f"Building DuckDB catalogue from {len(parquet_files)} files…")
    con = duckdb.connect(str(db_path))

    # Register each file as a named view/table
    rows = []
    for pf in parquet_files:
        rel_path = str(pf.relative_to(data_dir))
        # Parse year / form from path or filename
        parts = pf.stem.split("-")  # e.g. CORE-PC-BMF-NONPROFIT-RGN-2022
        year = None
        form = None
        for p in parts:
            if p.isdigit() and len(p) == 4:
                year = int(p)
            elif p in FORM_TYPES:
                form = p
        rows.append({"year": year, "form": form, "path": str(pf)})

    catalogue_df = pd.DataFrame(rows)
    con.execute("CREATE OR REPLACE TABLE file_catalogue AS SELECT * FROM catalogue_df")

    # Create a combined view (lazy – uses parquet_scan per file)
    # We build a UNION ALL query referencing all discovered files
    union_parts = []
    for _, row in catalogue_df.iterrows():
        escaped = row["path"].replace("'", "''")
        union_parts.append(
            f"SELECT *, {row['year']} AS _year, '{row['form']}' AS _form "
            f"FROM parquet_scan('{escaped}')"
        )

    if union_parts:
        view_sql = "CREATE OR REPLACE VIEW all_filings AS\n" + "\nUNION ALL\n".join(union_parts)
        con.execute(view_sql)
        log.info("DuckDB view 'all_filings' created.")

    con.close()
    log.info(f"Catalogue written to {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Download NCCS 990 Core Data")
    parser.add_argument("--year", type=int, help="Download a single year only")
    parser.add_argument("--form", choices=FORM_TYPES, help="Form type (PC/EZ/PF)")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    parser.add_argument("--rebuild-db", action="store_true", help="Rebuild DuckDB catalogue only")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_db:
        build_catalogue(DATA_DIR, DB_PATH)
        return

    s3 = get_s3_client()

    # First discover what's actually in the bucket
    all_keys = discover_available_files(s3)

    # Filter to core data files (Parquet or CSV, ignore small index files)
    core_keys = [
        k for k in all_keys
        if ("CORE-" in k or "/core/" in k.lower())
        and (k.endswith(".parquet") or k.endswith(".csv"))
        and not k.endswith("index.parquet")
    ]
    log.info(f"Found {len(core_keys)} core data files in S3.")

    # Apply year / form filters
    if args.year:
        core_keys = [k for k in core_keys if str(args.year) in k]
    if args.form:
        core_keys = [k for k in core_keys if f"-{args.form}-" in k or f"/{args.form}/" in k]

    if args.dry_run:
        for k in core_keys:
            print(k)
        print(f"\n{len(core_keys)} files would be downloaded.")
        return

    # Download each file
    success, failure = 0, 0
    for key in core_keys:
        filename = Path(key).name
        local_path = DATA_DIR / filename
        if local_path.exists() or local_path.with_suffix(".parquet").exists():
            log.info(f"  Skip (exists): {filename}")
            success += 1
            continue
        log.info(f"Downloading: {key}")
        ok = download_file(s3, key, local_path)
        if ok:
            success += 1
        else:
            failure += 1
        time.sleep(0.1)  # polite throttle

    log.info(f"Done. {success} succeeded, {failure} failed.")

    # Rebuild the DuckDB catalogue
    build_catalogue(DATA_DIR, DB_PATH)


if __name__ == "__main__":
    main()

