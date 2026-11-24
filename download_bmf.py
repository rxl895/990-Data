"""
IRS Exempt Organizations Business Master File (BMF) Downloader
===============================================================
Downloads the IRS EO Business Master File, which contains org name, state,
NTEE code, and other metadata for every registered nonprofit (linked by EIN).

The BMF is split into 4 regional CSVs:
  eo1.csv  — Regions 1-4 (New England, Mid-Atlantic, South, Midwest)
  eo2.csv  — Regions 5-8
  eo3.csv  — Regions 9-12
  eo4.csv  — Foreign / miscellaneous

These are joined to the SOI filing data via EIN to add state, org name, NTEE, etc.

Usage:
    python download_bmf.py              # Download and merge BMF
    python download_bmf.py --info-only  # Print BMF column info only
"""

import io
import sys
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
BMF_URLS = [
    "https://www.irs.gov/pub/irs-soi/eo1.csv",
    "https://www.irs.gov/pub/irs-soi/eo2.csv",
    "https://www.irs.gov/pub/irs-soi/eo3.csv",
    "https://www.irs.gov/pub/irs-soi/eo4.csv",
]
BMF_PATH = DATA_DIR / "bmf_all.parquet"


def download_bmf() -> pd.DataFrame:
    """Download all 4 BMF regional CSVs and concatenate them."""
    dfs = []
    for url in BMF_URLS:
        print(f"Downloading {url}…", end=" ", flush=True)
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content), dtype=str, low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]
            dfs.append(df)
            print(f"{len(df):,} rows, cols: {list(df.columns[:8])}")
        except Exception as e:
            print(f"FAILED: {e}")

    if not dfs:
        raise RuntimeError("Could not download any BMF files")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal BMF records: {len(combined):,}")
    print(f"Columns: {list(combined.columns)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(BMF_PATH, index=False)
    print(f"Saved to {BMF_PATH}")
    return combined


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-only", action="store_true")
    args = parser.parse_args()

    if BMF_PATH.exists() and not args.info_only:
        print(f"BMF already exists: {BMF_PATH}")
        bmf = pd.read_parquet(BMF_PATH)
    else:
        bmf = download_bmf()

    if args.info_only or True:
        print("\nBMF Column Summary:")
        print(bmf.dtypes.to_string())
        print("\nSample (5 rows):")
        print(bmf.head(5).to_string())
        if "state" in bmf.columns:
            print("\nTop states:")
            print(bmf["state"].value_counts().head(15))
        elif "st" in bmf.columns:
            print("\nTop states (st):")
            print(bmf["st"].value_counts().head(15))


if __name__ == "__main__":
    main()

