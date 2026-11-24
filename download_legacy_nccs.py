"""
NCCS Legacy Core Downloader (1989–2011)
=======================================
Downloads the Urban Institute / NCCS harmonized Legacy Core files from the public S3 archive.
Covers:
  - Tax years: 1989–2011
  - Forms:
      - 990combined: Harmonized Form 990 + Form 990-EZ (PZ)
      - 990pf: Form 990-PF (Private Foundations)

Source endpoint:
  https://nccsdata.s3.amazonaws.com/processed_legacy/core/{year}/{form}/core_{year}_{form}.parquet

Usage:
    python download_legacy_nccs.py                    # Download all 1989-2011 files
    python download_legacy_nccs.py --year 1989 1990   # Specific years
    python download_legacy_nccs.py --dry-run          # Print URLs without downloading
"""

import argparse
import logging
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
BASE_URL = "https://nccsdata.s3.amazonaws.com/processed_legacy/core"

LEGACY_YEARS = list(range(1989, 2012))  # 1989 to 2011 inclusive
LEGACY_FORMS = ["990combined", "990pf"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "download_legacy.log"),
    ],
)
log = logging.getLogger(__name__)


def build_legacy_url(year: int, form: str) -> str:
    return f"{BASE_URL}/{year}/{form}/core_{year}_{form}.parquet"


def download_legacy_file(year: int, form: str) -> dict:
    url = build_legacy_url(year, form)
    out_name = f"legacy_core_{form}_{year}.parquet"
    out_path = DATA_DIR / out_name

    if out_path.exists() and out_path.stat().st_size > 1000:
        log.info(f"  Skip (exists): {out_name}")
        return {"year": year, "form": form, "status": "skipped_exists", "path": str(out_path)}

    log.info(f"  Downloading: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NCCS990-Curator/1.0 (Research Archive Tool)"}
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        log.info(f"  Saved: {out_name} ({len(data)/1e6:.1f} MB)")
        return {"year": year, "form": form, "status": "ok", "size_mb": len(data)/1e6, "path": str(out_path)}
    except Exception as e:
        log.error(f"  Failed {url}: {e}")
        return {"year": year, "form": form, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Download NCCS Legacy Core Data (1989-2011)")
    parser.add_argument("--year", type=int, nargs="+", help="Specific year(s) e.g. 1989 1995")
    parser.add_argument("--form", choices=LEGACY_FORMS, nargs="+", help="990combined or 990pf")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without downloading")
    parser.add_argument("--workers", type=int, default=3, help="Parallel download threads")
    args = parser.parse_args()

    years = args.year if args.year else LEGACY_YEARS
    forms = args.form if args.form else LEGACY_FORMS

    tasks = [(y, f) for y in years for f in forms]

    if args.dry_run:
        print(f"Discovered {len(tasks)} legacy files to download:")
        for y, f in tasks:
            print(f"  {y} | {f:<12} -> {build_legacy_url(y, f)}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {len(tasks)} legacy partitions with {args.workers} workers…")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_legacy_file, y, f): (y, f) for y, f in tasks}
        with tqdm(total=len(futures), desc="Legacy Downloads") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                pbar.update(1)
                pbar.set_postfix({"last": f"{res['form']}_{res['year']}_{res['status']}"})

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped_exists")
    errs = sum(1 for r in results if r["status"] == "error")
    log.info(f"Done — OK: {ok}, Skipped: {skipped}, Errors: {errs}")

    # Rebuild DuckDB catalogue if available
    try:
        from irs_soi_downloader import build_duckdb_catalogue
        build_duckdb_catalogue()
    except Exception as e:
        log.warning(f"Could not automatically refresh DuckDB: {e}")


if __name__ == "__main__":
    main()
