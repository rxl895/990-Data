# NCCS 990 Nonprofit Filing Explorer

A complete Python pipeline + Streamlit web app to **curate and explore IRS Form 990 data** for every US nonprofit from **1989 to 2025**, covering all 50 states + DC.

## Project Structure

```
nccs990/
├── app.py                  # Streamlit web app (5 interactive tabs)
├── irs_soi_downloader.py   # IRS SOI data downloader (2012–2024, all 36 URLs)
├── download_bmf.py         # IRS BMF downloader (org names, states, NTEE)
├── full_download.py        # Extended downloader with parallel workers
├── downloader.py           # Legacy NCCS S3 downloader (requires auth)
├── fetch_sample.py         # Quick single-year sample fetcher
├── requirements.txt        # Python dependencies
├── data/                   # Downloaded Parquet files (auto-created)
│   ├── irs_soi_990_2022.parquet
│   ├── irs_soi_990ez_2022.parquet
│   ├── irs_soi_990pf_2022.parquet
│   ├── bmf_all.parquet     # 1.97M orgs with state/NTEE/name
│   └── summary.csv
├── nccs990.duckdb          # DuckDB catalogue (auto-built)
└── download.log            # Download progress log
```

## Data Sources

| Era | Source | URL | Coverage |
|---|---|---|---|
| **2012–2024** | IRS Statistics of Income (SOI) | `https://www.irs.gov/pub/irs-soi/` | Free, public |
| **1989–2011** | NCCS Legacy (Urban Institute) | `s3://nccsdata/legacy/core/` | Requires NCCS S3 access |
| **Org Info** | IRS Exempt Organizations BMF | `https://www.irs.gov/pub/irs-soi/eo{1-4}.csv` | Free, public |

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Data (Step by Step)

```bash
# STEP 1: See all 36 available SOI files (dry run)
python irs_soi_downloader.py --dry-run

# STEP 2: Download a quick sample (Form 990-EZ, 2022 — 231k rows, ~5s)
python irs_soi_downloader.py --year 2022 --form 990ez

# STEP 3: Download all 3 form types for a year range (recommended start)
python irs_soi_downloader.py --year 2018 2019 2020 2021 2022 2023 2024 --workers 3

# STEP 4: Download ALL years (2012–2024, all forms) — ~1 hr, ~5 GB
python irs_soi_downloader.py --workers 3

# STEP 5: Download BMF (org names, states, NTEE codes) — 1.97M orgs, ~1 min
python download_bmf.py

# STEP 6: Rebuild DuckDB catalogue (auto-runs after any download)
python irs_soi_downloader.py --rebuild-db
```

### Launch the App
```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

## Streamlit App Features

| Tab | Content |
|---|---|
| **🏠 Overview** | KPI metrics (filings, revenue, assets), filings-by-year bar chart, top-states bar + choropleth map, form-type pie chart |
| **💰 Financials** | Revenue/expenses/assets trend over time, revenue distribution histogram |
| **🏷️ NTEE Sectors** | Filings by NTEE major category (Arts, Education, Health, etc.), category pie chart |
| **🗺️ State Deep Dive** | Select any state → filings trend, revenue trend, top 20 orgs by revenue |
| **📋 Browse Data** | Human-readable curated columns, Plain-English toggle, numeric currency formatting, search, column customizer, built-in IRS Field Dictionary, and CSV export |

### Sidebar Filters (all tabs)
- Year range slider
- Form type selector (990 / 990-EZ / 990-PF / All)  
- State selector (All states / individual state)
- Max display rows

## Data Volumes

| Form | Years Available | Approx Rows/Year | Approx File Size |
|---|---|---|---|
| Form 990 | 2012–2024 | ~350,000 | 50–55 MB/zip |
| Form 990-EZ | 2012–2024 | ~230,000 | 10–12 MB/zip |
| Form 990-PF | 2012–2024 (gaps 2017–19) | ~110,000 | 12–14 MB/zip |
| **BMF** | Current snapshot | **1,975,975** orgs | ~65 MB |

## Key IRS SOI Column Reference

| Column | Meaning |
|---|---|
| `ein` | Employer Identification Number (links to BMF) |
| `taxpd` | Tax period (YYYYMM format) |
| `totrevnue` | Total Revenue (990-EZ) |
| `totexpns` | Total Expenses (990-EZ) |
| `totassetsend` | Total Assets, end of year |
| `subseccd` | IRS Subsection Code (3 = 501c3, etc.) |
| `_tax_year` | Added by pipeline — calendar year |
| `_form_type` | Added by pipeline — 990 / 990ez / 990pf |

## BMF Column Reference

| Column | Meaning |
|---|---|
| `name` | Organization name |
| `state` | State abbreviation |
| `city` | City |
| `nteecc` / `ntee_cd` | NTEE Classification Code |
| `subsection` | IRS subsection (501c type) |
| `deductibility` | 1 = tax-deductible contributions |
| `ruling` | YYYYMM of IRS recognition |
| `asset_amt` | Asset amount (BMF snapshot) |

## Notes

- Files from 2012–2017 are space-delimited `.dat` format; 2018+ are CSV. The downloader handles both automatically.
- 990-PF was not published by IRS for 2017–2019.
- The BMF is a **current snapshot** of registered nonprofits; historical org info is in the SOI files themselves.
- DuckDB catalogue is rebuilt automatically after each download run, enabling SQL queries across all years.
- Financial figures are in **nominal USD** (not inflation-adjusted).
