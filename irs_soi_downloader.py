"""
NCCS / IRS SOI 990 Data Downloader
=====================================
Downloads IRS Statistics of Income (SOI) extract ZIP files for Form 990,
990-EZ, and 990-PF from the official IRS.gov public server.

Coverage:
  - SOI current series:  2012–2024 (IRS.gov, free, no auth)
  - Legacy NCCS series:  1989–2011 (requires NCCS S3 access — see note below)

The IRS SOI files are the same underlying data the NCCS harmonises into their
Core Series. Each zip contains a space-delimited (.dat) or CSV file with
~100–400 columns of financial and descriptive data for every nonprofit that
filed that form in that tax year.

Usage:
    python irs_soi_downloader.py                 # Download all available years & forms
    python irs_soi_downloader.py --year 2022     # Specific year
    python irs_soi_downloader.py --form 990      # Specific form
    python irs_soi_downloader.py --dry-run       # List URLs without downloading
    python irs_soi_downloader.py --rebuild-db    # Rebuild DuckDB catalogue only
    python irs_soi_downloader.py --workers 3     # Parallel downloads
"""

import argparse
import io
import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration — exact URL stems extracted from UrbanInstitute/nccs-data-core
# ---------------------------------------------------------------------------

IRS_BASE = "https://www.irs.gov/pub/irs-soi"

# (year, form) → filename stem (append .zip for download URL)
# NA/None = not published by IRS for that year
SOI_STEMS = {
    "990": {
        2012: "12eofinextract990",
        2013: "13eofinextract990",
        2014: "14eofinextract990",
        2015: "15eofinextract990",
        2016: "16eofinextract990",
        2017: "17eofinextract990",
        2018: "18eoextract990",
        2019: "19eoextract990",
        2020: "20eoextract990",
        2021: "21eoextract990",
        2022: "22eoextract990",
        2023: "23eoextract990",
        2024: "24eoextract990",
    },
    "990ez": {
        2012: "12eofinextract990ez",
        2013: "13eofinextract990ez",
        2014: "14eofinextract990ez",
        2015: "15eofinextractEZ",
        2016: "16eofinextractez",
        2017: "17eofinextractEZ",
        2018: "18eoextractez",
        2019: "19eoextractez",
        2020: "20eoextractez",
        2021: "21eoextractez",
        2022: "22eoextractez",
        2023: "23eoextractez",
        2024: "24eoextract990EZ",
    },
    "990pf": {
        2012: "12eofinextract990pf",
        2013: "13eofinextract990pf",
        2014: "14eofinextract990pf",
        2015: "15eofinextract990pf",
        2016: "16eofinextract990pf",
        2017: None,   # not published
        2018: None,   # not published
        2019: None,   # not published
        2020: "20eoextract990pf",
        2021: "21eoextract990pf",
        2022: "22eoextract990pf",
        2023: "23eoextract990pf",
        2024: "24eoextract990pf",
    },
}

FORMS = list(SOI_STEMS.keys())
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = Path(__file__).parent / "nccs990.duckdb"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "download.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def build_url(form: str, year: int):
    stem = SOI_STEMS.get(form, {}).get(year)
    return f"{IRS_BASE}/{stem}.zip" if stem else None


def detect_delimiter(raw: bytes) -> str:
    """Auto-detect delimiter for IRS SOI files (CSV or space-delimited .dat)."""
    sample = raw[:2048].decode("utf-8", errors="replace")
    return "," if "," in sample else " "


def parse_dat_file(raw: bytes) -> pd.DataFrame:
    """Parse a space-delimited .dat file (2012–2017 format)."""
    text = raw.decode("latin-1", errors="replace")
    return pd.read_csv(
        io.StringIO(text),
        sep=r"\s+",
        engine="python",
        dtype=str,
    )


def download_and_extract(form: str, year: int) -> dict:
    """Download a single IRS SOI ZIP, extract the data file, save as Parquet."""
    url = build_url(form, year)
    if not url:
        return {"form": form, "year": year, "status": "skipped_no_url"}

    out_name = f"irs_soi_{form}_{year}.parquet"
    out_path = DATA_DIR / out_name

    if out_path.exists():
        log.info(f"  Skip (exists): {out_name}")
        return {"form": form, "year": year, "status": "skipped_exists", "path": str(out_path)}

    log.info(f"  Downloading: {url}")
    try:
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()

        # Read ZIP content
        content = b"".join(resp.iter_content(chunk_size=1 << 20))
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            # Find the data file (largest file inside the zip)
            names = z.namelist()
            data_files = [n for n in names if n.lower().endswith((".csv", ".dat", ".txt"))
                         and not n.startswith("__")]
            if not data_files:
                log.warning(f"  No data file found in {url}: {names}")
                return {"form": form, "year": year, "status": "error_no_data_file"}

            # Pick the largest file
            data_file = max(data_files, key=lambda n: z.getinfo(n).file_size)
            raw = z.read(data_file)

        # Parse
        if data_file.lower().endswith(".dat"):
            df = parse_dat_file(raw)
        else:
            # Try CSV; use latin-1 for safety
            text = raw.decode("latin-1", errors="replace")
            df = pd.read_csv(io.StringIO(text), low_memory=False, dtype=str)

        # Normalise column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]

        # Add metadata columns
        df["_form_type"] = form
        df["_tax_year"] = year
        df["_source"] = f"IRS SOI {data_file}"

        # Save as Parquet
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

        log.info(f"  Saved: {out_name}  ({len(df):,} rows × {len(df.columns)} cols)")
        return {"form": form, "year": year, "status": "ok", "rows": len(df), "path": str(out_path)}

    except requests.HTTPError as e:
        log.error(f"  HTTP error {url}: {e}")
        return {"form": form, "year": year, "status": f"http_error_{e.response.status_code}"}
    except Exception as e:
        log.error(f"  Failed {url}: {e}")
        return {"form": form, "year": year, "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# DuckDB catalogue
# ---------------------------------------------------------------------------

def build_duckdb_catalogue():
    """Build / refresh DuckDB catalogue from local Parquet files."""
    parquet_files = sorted(
        list(DATA_DIR.glob("irs_soi_*.parquet")) + list(DATA_DIR.glob("legacy_core_*.parquet"))
    )
    if not parquet_files:
        log.warning("No Parquet files found – skipping DuckDB build.")
        return

    log.info(f"Building DuckDB catalogue from {len(parquet_files)} files…")

    rows = []
    for pf in parquet_files:
        fname = pf.stem  # e.g. irs_soi_990_2022 or legacy_core_990combined_1989
        parts = fname.split("_")
        form = None
        year = None
        for p in parts:
            if p.isdigit() and len(p) == 4:
                year = int(p)
            elif p in FORMS or p in ("990", "990ez", "990pf", "990combined"):
                form = p

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

    # Build a lazy UNION ALL view across all files
    union_parts = []
    for _, row in cat_df.iterrows():
        escaped = row["path"].replace("'", "''")
        union_parts.append(
            f"SELECT *, '{row['form_type']}' AS _form, {row['year']} AS _year "
            f"FROM parquet_scan('{escaped}')"
        )

    if union_parts:
        con.execute(
            "CREATE OR REPLACE VIEW all_filings AS\n"
            + "\nUNION ALL BY NAME\n".join(union_parts)
        )

    # Print summary
    try:
        summary = con.execute(
            "SELECT _form, _year, COUNT(*) n FROM all_filings GROUP BY 1,2 ORDER BY 2,1"
        ).df()
        print("\n--- Data Summary ---")
        print(summary.to_string(index=False))
        print(f"\nTotal records: {summary['n'].sum():,}")

        summary.to_csv(DATA_DIR / "summary.csv", index=False)
    except Exception as e:
        log.warning(f"Could not build summary: {e}")

    con.close()
    log.info(f"DuckDB catalogue: {DB_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IRS SOI 990 Downloader (2012-2024)")
    parser.add_argument("--year", type=int, nargs="+",
                        help="Year(s) to download, e.g. 2022 or 2018 2019 2020")
    parser.add_argument("--form", choices=FORMS, nargs="+",
                        help="Form type(s): 990 / 990ez / 990pf")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print URLs without downloading")
    parser.add_argument("--rebuild-db", action="store_true",
                        help="Rebuild DuckDB catalogue from existing Parquet files")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel download threads (default: 2; be polite to IRS servers)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_db:
        build_duckdb_catalogue()
        return

    # Build work list
    forms = args.form if args.form else FORMS
    years = args.year if args.year else list(range(2012, 2025))

    tasks = []
    for form in forms:
        for year in years:
            url = build_url(form, year)
            tasks.append((form, year, url))

    if args.dry_run:
        print(f"{'Form':<8} {'Year':<6} {'URL'}")
        print("-" * 80)
        total_avail = 0
        for form, year, url in tasks:
            status = url if url else "(not published)"
            print(f"{form:<8} {year:<6} {status}")
            if url:
                total_avail += 1
        print(f"\n{total_avail} files available for download.")
        return

    available_tasks = [(f, y) for f, y, url in tasks if url]
    log.info(f"Tasks: {len(available_tasks)} available out of {len(tasks)} combinations")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_and_extract, f, y): (f, y) for f, y in available_tasks}
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                pbar.update(1)
                pbar.set_postfix({"last": f"{r['form']}_{r['year']}_{r['status']}"})

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if "skipped" in r["status"])
    errors = sum(1 for r in results if "error" in r["status"])
    log.info(f"Complete — OK: {ok}, Skipped: {skipped}, Errors: {errors}")

    build_duckdb_catalogue()


if __name__ == "__main__":
    main()
