"""
NCCS 990 Quick-Start Data Fetcher
===================================
Downloads a sample year from the NCCS public S3 bucket to verify connectivity
and demonstrate the data structure before running a full multi-year download.

Usage:
    python fetch_sample.py --year 2022
"""

import argparse
import io
import os
import sys
from pathlib import Path

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.client import Config

DATA_DIR = Path(__file__).parent / "data"

BUCKET = "nccsdata"

# Known working keys discovered by bucket scan (update if bucket layout changes)
KNOWN_PATTERNS = [
    # Harmonised series (newest)
    "harmonized/core/CORE-{form}-BMF-NONPROFIT-RGN/CORE-{form}-BMF-NONPROFIT-RGN-{year}.parquet",
    # Processed merged
    "processed_merged/core/{year}/{form}/CORE-{form}-{year}.parquet",
]

FORMS = ["PC", "EZ", "PF"]


def s3_client():
    return boto3.client("s3", region_name="us-east-1",
                        config=Config(signature_version=UNSIGNED))


def try_download(s3, year: int, form: str):
    for pattern in KNOWN_PATTERNS:
        key = pattern.format(year=year, form=form)
        try:
            print(f"  Trying s3://{BUCKET}/{key}")
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            data = obj["Body"].read()
            if key.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(data))
            else:
                df = pd.read_csv(io.BytesIO(data), low_memory=False)
            print(f"  ✓ Success! {len(df):,} rows, {len(df.columns)} columns")
            return df, key
        except s3.exceptions.NoSuchKey:
            continue
        except Exception as e:
            print(f"  ✗ {e}")
            continue
    return None, None


def scan_bucket_for_core(s3, prefix: str = ""):
    """List actual keys in bucket."""
    print(f"\nScanning bucket prefix: '{prefix}'")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            print(f"  DIR: {cp['Prefix']}")
        for obj in page.get("Contents", []):
            print(f"  FILE: {obj['Key']}  ({obj['Size']:,} bytes)")
            keys.append(obj["Key"])
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--form", choices=FORMS + ["ALL"], default="ALL")
    parser.add_argument("--scan", action="store_true", help="Scan bucket structure")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    s3 = s3_client()

    if args.scan:
        # Walk top-level dirs
        scan_bucket_for_core(s3, "")
        scan_bucket_for_core(s3, "harmonized/")
        scan_bucket_for_core(s3, "processed_merged/")
        scan_bucket_for_core(s3, "raw/")
        return

    forms = FORMS if args.form == "ALL" else [args.form]

    for form in forms:
        print(f"\n--- Fetching {form} for {args.year} ---")
        df, key = try_download(s3, args.year, form)
        if df is not None:
            out_path = DATA_DIR / f"CORE-{form}-{args.year}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  Saved to {out_path}")
            print(f"  Columns: {list(df.columns[:10])} ...")
        else:
            print(f"  Could not find {form} data for {args.year}")


if __name__ == "__main__":
    main()
