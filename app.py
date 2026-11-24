"""
NCCS / IRS SOI 990 Explorer – Streamlit App
=============================================
Interactive dashboard for exploring IRS Form 990 / 990-EZ / 990-PF
filings for US nonprofits (2012–2024, all 50 states).

Architecture: DuckDB-native — data stays on disk, only aggregated
results (~KB) are pulled into pandas for plotting. Handles 1.8 GB+
of parquet files without OOM.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DB_PATH  = Path(__file__).parent / "nccs990.duckdb"
BMF_PATH = DATA_DIR / "bmf_all.parquet"
ZIP_PATH = DATA_DIR / "zip_coordinates.parquet"

FORM_LABELS = {
    "990":    "Form 990 (Public Charities)",
    "990ez":  "Form 990-EZ (Small Orgs)",
    "990pf":  "Form 990-PF (Private Foundations)",
    "ALL":    "All Forms",
}

US_STATES = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas",
    "CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware",
    "FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho",
    "IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas",
    "KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada",
    "NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York",
    "NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma",
    "OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina",
    "SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah",
    "VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia",
    "WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia",
    "PR":"Puerto Rico","VI":"Virgin Islands","GU":"Guam",
}

NTEE_MAJOR = {
    "A":"Arts, Culture & Humanities","B":"Education","C":"Environment",
    "D":"Animal-Related","E":"Health Care","F":"Mental Health & Crisis",
    "G":"Disease & Disorders","H":"Medical Research","I":"Crime & Legal",
    "J":"Employment","K":"Food, Agriculture","L":"Housing & Shelter",
    "M":"Public Safety","N":"Recreation & Sports","O":"Youth Development",
    "P":"Human Services","Q":"International","R":"Civil Rights",
    "S":"Community Improvement","T":"Philanthropy","U":"Science & Technology",
    "V":"Social Science","W":"Public & Societal Benefit","X":"Religion",
    "Y":"Mutual Benefit","Z":"Unknown",
}

# IRS SOI column name aliases → semantic name
COL_ALIASES = {
    "total_revenue": ["totrevnue","totrevenue","p1totrev","total_revenue","tot_revenue"],
    "total_expenses": ["totexpns","totexps","totexpense","total_expenses"],
    "total_assets": ["totassetsend","totassets","total_assets_eoy","totnetassetsend"],
    "ein": ["ein","ein9"],
    "org_name": ["name","orgname","organization_name"],
    "state": ["state","st","statecd","addr_state"],
    "ntee": ["nteecc","ntee_cd","ntee_cc","ntee1","ntee"],
    "tax_year": ["_tax_year","taxyear","fisyr","tax_prd_yr","taxpd"],
    "form_type": ["_form_type","formtype","form_type"],
    "subsection": ["subseccd","subsec_cd","subsection"],
    "city": ["city","citycd"],
    "zip": ["zip5","zip","zipcd"],
}

COLUMN_METADATA = {
    "name": {"label": "Organization Name", "category": "Identity", "desc": "Legal name of the nonprofit organization"},
    "ein": {"label": "EIN", "category": "Identity", "desc": "IRS Employer Identification Number (9 digits)"},
    "state": {"label": "State", "category": "Location", "desc": "Two-letter US state or territory postal code"},
    "city": {"label": "City", "category": "Location", "desc": "City of organization headquarters"},
    "zip": {"label": "ZIP Code", "category": "Location", "desc": "Postal ZIP code"},
    "_tax_year": {"label": "Tax Year", "category": "Filing Info", "desc": "Calendar tax year of the filing"},
    "tax_year": {"label": "Tax Year", "category": "Filing Info", "desc": "Calendar tax year of the filing"},
    "_form_type": {"label": "Form Type", "category": "Filing Info", "desc": "Form filed (990, 990-EZ, 990-PF)"},
    "form_type": {"label": "Form Type", "category": "Filing Info", "desc": "Form filed (990, 990-EZ, 990-PF)"},
    "taxpd": {"label": "Tax Period", "category": "Filing Info", "desc": "Accounting tax period end date (YYYYMM)"},
    "subseccd": {"label": "IRC Subsection", "category": "Tax Status", "desc": "IRS 501(c) subsection code (e.g. 03 = 501(c)(3))"},
    "subsection": {"label": "IRC Subsection", "category": "Tax Status", "desc": "IRS 501(c) subsection code (e.g. 03 = 501(c)(3))"},
    "nteecc": {"label": "NTEE Code", "category": "Sector", "desc": "National Taxonomy of Exempt Entities classification code"},
    "ntee_cd": {"label": "NTEE Code", "category": "Sector", "desc": "National Taxonomy of Exempt Entities classification code"},
    "ntee_label": {"label": "NTEE Sector", "category": "Sector", "desc": "Human-readable NTEE major category name"},
    "totrevnue": {"label": "Total Revenue ($)", "category": "Revenue", "desc": "Total revenue from all sources during the tax year"},
    "totrevenue": {"label": "Total Revenue ($)", "category": "Revenue", "desc": "Total revenue from all sources during the tax year"},
    "total_revenue": {"label": "Total Revenue ($)", "category": "Revenue", "desc": "Total revenue from all sources during the tax year"},
    "totcntrbs": {"label": "Contributions & Grants ($)", "category": "Revenue", "desc": "Contributions, gifts, grants, and similar amounts received"},
    "prgmservrev": {"label": "Program Service Revenue ($)", "category": "Revenue", "desc": "Revenue from activities related directly to exempt purpose"},
    "duesassesmnts": {"label": "Membership Dues ($)", "category": "Revenue", "desc": "Membership dues and assessments received"},
    "othrinvstinc": {"label": "Investment Income ($)", "category": "Revenue", "desc": "Interest, dividends, and other investment income"},
    "totexpns": {"label": "Total Expenses ($)", "category": "Expenses", "desc": "Total operating and program expenses incurred during year"},
    "totexps": {"label": "Total Expenses ($)", "category": "Expenses", "desc": "Total operating and program expenses incurred during year"},
    "total_expenses": {"label": "Total Expenses ($)", "category": "Expenses", "desc": "Total operating and program expenses incurred during year"},
    "totexcessyr": {"label": "Net Surplus / Deficit ($)", "category": "Profitability", "desc": "Excess (or deficit) for the year: Revenue minus Expenses"},
    "totassetsend": {"label": "Total Assets ($)", "category": "Balance Sheet", "desc": "Total assets at the end of the tax year"},
    "totassets": {"label": "Total Assets ($)", "category": "Balance Sheet", "desc": "Total assets at the end of the tax year"},
    "total_assets": {"label": "Total Assets ($)", "category": "Balance Sheet", "desc": "Total assets at the end of the tax year"},
    "totliabend": {"label": "Total Liabilities ($)", "category": "Balance Sheet", "desc": "Total liabilities at the end of the tax year"},
    "totnetassetsend": {"label": "Net Assets ($)", "category": "Balance Sheet", "desc": "Total net assets or fund balances at end of year"},
    "subseccd": {"label": "IRC Subsection", "category": "Tax Status", "desc": "IRS 501(c) subsection code"},
    "ruling": {"label": "IRS Ruling Date", "category": "IRS Master File", "desc": "Year & month exemption was officially granted by IRS (YYYYMM)"},
    "deductibility": {"label": "Deductibility Code", "category": "IRS Master File", "desc": "1 = Contributions are tax-deductible to donors"},
    "_source": {"label": "Source File", "category": "System", "desc": "Source data archive or extract filename"},
}

CURATED_COLUMNS_PRIORITY = [
    ("name", "Organization Name"),
    ("ein", "EIN"),
    ("state", "State"),
    ("city", "City"),
    ("_tax_year", "Tax Year"),
    ("_form_type", "Form Type"),
    ("subseccd", "IRC Subsection"),
    ("subsection", "IRC Subsection"),
    ("ntee_label", "NTEE Sector"),
    ("nteecc", "NTEE Code"),
    ("totrevnue", "Total Revenue ($)"),
    ("totrevenue", "Total Revenue ($)"),
    ("total_revenue", "Total Revenue ($)"),
    ("totexpns", "Total Expenses ($)"),
    ("totexps", "Total Expenses ($)"),
    ("total_expenses", "Total Expenses ($)"),
    ("totexcessyr", "Net Surplus / Deficit ($)"),
    ("totassetsend", "Total Assets ($)"),
    ("totassets", "Total Assets ($)"),
    ("total_assets", "Total Assets ($)"),
    ("totliabend", "Total Liabilities ($)"),
    ("totnetassetsend", "Net Assets ($)"),
    ("totcntrbs", "Contributions & Grants ($)"),
    ("prgmservrev", "Program Service Revenue ($)"),
    ("zip", "ZIP Code"),
]

# ---------------------------------------------------------------------------
# DuckDB connection — one persistent in-process connection per session
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db() -> duckdb.DuckDBPyConnection:
    """Return a single shared DuckDB in-process connection (no file needed)."""
    return duckdb.connect()


def parse_year(fname: str):
    m = re.search(r'(\d{4})', fname)
    return int(m.group(1)) if m else None


def parse_form(fname: str) -> str:
    for f in ("990pf","990ez","990"):
        if f in fname.lower():
            return f
    return "unknown"


# ---------------------------------------------------------------------------
# Schema introspection — sniff column names from one parquet file
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def sniff_columns(parquet_path: str) -> list[str]:
    con = duckdb.connect()
    cols = con.execute(f"SELECT * FROM read_parquet('{parquet_path}') LIMIT 0").description
    con.close()
    # Strip BOM, whitespace, and normalise to lowercase
    return [c[0].strip().lower().lstrip('\ufeff').replace('ï»¿', '') for c in cols]


def find_col(columns: list[str], aliases: list[str]) -> str | None:
    for a in aliases:
        if a in columns:
            return a
    return None


# ---------------------------------------------------------------------------
# DuckDB query builder
# ---------------------------------------------------------------------------

def build_view_sql(files: list[str]) -> str:
    """Read one or more parquet files, unifying schemas by column name.
    Uses DuckDB's native list form with union_by_name=True so files with
    different column sets (990 vs 990ez vs 990pf) can be combined safely.
    """
    file_list = ", ".join(f"'{f}'" for f in files)
    return f"SELECT * FROM read_parquet([{file_list}], union_by_name=true)"


def where_clause(state: str, state_col: str | None,
                 yr_min: int, yr_max: int, yr_col: str | None) -> str:
    clauses = []
    if yr_col:
        clauses.append(f"TRY_CAST({yr_col} AS INTEGER) BETWEEN {yr_min} AND {yr_max}")
    if state != "ALL" and state_col:
        clauses.append(f"UPPER(TRIM(CAST({state_col} AS VARCHAR))) = '{state}'")
    return ("WHERE " + " AND ".join(clauses)) if clauses else ""


# ---------------------------------------------------------------------------
# Aggregation queries — only small result sets enter Python RAM
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Counting filings…")
def query_kpis(files: tuple, yr_min: int, yr_max: int,
               form: str, state: str) -> dict:
    db = get_db()
    cols = sniff_columns(files[0]) if files else []
    yr_col  = find_col(cols, COL_ALIASES["tax_year"])
    st_col  = find_col(cols, COL_ALIASES["state"])
    ein_col = find_col(cols, COL_ALIASES["ein"]) or "ein"
    base    = build_view_sql(list(files))

    matched_rev = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_revenue"] if c in cols]
    rev_expr = f"COALESCE({', '.join(matched_rev)}, 0)" if matched_rev else None

    matched_ast = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_assets"] if c in cols]
    ast_expr = f"COALESCE({', '.join(matched_ast)}, 0)" if matched_ast else None

    matched_exp = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_expenses"] if c in cols]
    exp_expr = f"COALESCE({', '.join(matched_exp)}, 0)" if matched_exp else None

    agg_parts = ["COUNT(*) AS n_filings"]
    if rev_expr: agg_parts.append(f"SUM({rev_expr}) AS total_rev")
    if ast_expr: agg_parts.append(f"SUM({ast_expr}) AS total_ast")
    if exp_expr: agg_parts.append(f"SUM({exp_expr}) AS total_exp")

    # If state filter is selected and filings do not have state column, join BMF
    if state != "ALL" and not st_col and BMF_PATH.exists():
        yr_filter = f"TRY_CAST(t.{yr_col} AS INTEGER) BETWEEN {yr_min} AND {yr_max}" if yr_col else ""
        st_filter = f"UPPER(TRIM(b.state)) = '{state}'"
        filters = [st_filter]
        if yr_filter: filters.append(yr_filter)
        wh = "WHERE " + " AND ".join(filters)
        sql = f"""
            SELECT {', '.join(agg_parts)}
            FROM ({base}) t
            LEFT JOIN read_parquet('{BMF_PATH}') b
                ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
            {wh}
        """
    else:
        wh = where_clause(state, st_col, yr_min, yr_max, yr_col)
        sql = f"SELECT {', '.join(agg_parts)} FROM ({base}) t {wh}"

    row = db.execute(sql).fetchone()
    keys = [d[0] for d in db.execute(sql).description]
    return dict(zip(keys, row))


@st.cache_data(ttl=600, show_spinner="Building charts…")
def query_filings_by_year(files: tuple, yr_min: int, yr_max: int,
                          form: str, state: str) -> pd.DataFrame:
    db = get_db()
    cols   = sniff_columns(files[0]) if files else []
    yr_col = find_col(cols, COL_ALIASES["tax_year"])
    st_col = find_col(cols, COL_ALIASES["state"])
    ein_col = find_col(cols, COL_ALIASES["ein"]) or "ein"
    if not yr_col:
        return pd.DataFrame()
    base = build_view_sql(list(files))

    if state != "ALL" and not st_col and BMF_PATH.exists():
        filters = [
            f"UPPER(TRIM(b.state)) = '{state}'",
            f"TRY_CAST(t.{yr_col} AS INTEGER) BETWEEN {yr_min} AND {yr_max}",
        ]
        wh = "WHERE " + " AND ".join(filters)
        sql = f"""
            SELECT TRY_CAST(t.{yr_col} AS INTEGER) AS Year, COUNT(*) AS Filings
            FROM ({base}) t
            LEFT JOIN read_parquet('{BMF_PATH}') b
                ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
            {wh}
            GROUP BY 1 ORDER BY 1
        """
    else:
        wh = where_clause(state, st_col, yr_min, yr_max, yr_col)
        sql = f"""
            SELECT TRY_CAST(t.{yr_col} AS INTEGER) AS Year, COUNT(*) AS Filings
            FROM ({base}) t {wh}
            GROUP BY 1 ORDER BY 1
        """
    return db.execute(sql).df()


@st.cache_data(ttl=600, show_spinner="Building state rankings…")
def query_by_state(files: tuple, yr_min: int, yr_max: int,
                   form: str, state: str, bmf_exists: bool = True) -> pd.DataFrame:
    db = get_db()
    cols   = sniff_columns(files[0]) if files else []
    yr_col = find_col(cols, COL_ALIASES["tax_year"])
    st_col = find_col(cols, COL_ALIASES["state"])
    ein_col = find_col(cols, COL_ALIASES["ein"]) or "ein"
    base = build_view_sql(list(files))

    if st_col:
        wh = where_clause(state, st_col, yr_min, yr_max, yr_col)
        sql = f"""
            SELECT UPPER(TRIM(CAST(t.{st_col} AS VARCHAR))) AS State, COUNT(*) AS Filings
            FROM ({base}) t {wh}
            GROUP BY 1 ORDER BY 2 DESC LIMIT 55
        """
    elif bmf_exists:
        filters = ["b.state IS NOT NULL", "b.state != ''"]
        if yr_col:
            filters.append(f"TRY_CAST(t.{yr_col} AS INTEGER) BETWEEN {yr_min} AND {yr_max}")
        if state != "ALL":
            filters.append(f"UPPER(TRIM(b.state)) = '{state}'")
        wh = "WHERE " + " AND ".join(filters)
        sql = f"""
            SELECT UPPER(TRIM(CAST(b.state AS VARCHAR))) AS State, COUNT(*) AS Filings
            FROM ({base}) t
            LEFT JOIN read_parquet('{BMF_PATH}') b
                ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
            {wh}
            GROUP BY 1 ORDER BY 2 DESC LIMIT 55
        """
    else:
        return pd.DataFrame()
    return db.execute(sql).df()


@st.cache_data(ttl=600, show_spinner="Loading comprehensive map data…")
def query_map_data(files: tuple, yr_min: int, yr_max: int, form: str, bmf_exists: bool) -> pd.DataFrame:
    db = get_db()
    cols = sniff_columns(files[0]) if files else []
    yr_col = find_col(cols, COL_ALIASES["tax_year"])
    st_col = find_col(cols, COL_ALIASES["state"])
    ein_col = find_col(cols, COL_ALIASES["ein"]) or "ein"
    base = build_view_sql(list(files))

    matched_rev = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_revenue"] if c in cols]
    rev_expr = f"COALESCE({', '.join(matched_rev)}, 0)" if matched_rev else "0"

    matched_ast = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_assets"] if c in cols]
    ast_expr = f"COALESCE({', '.join(matched_ast)}, 0)" if matched_ast else "0"

    matched_exp = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_expenses"] if c in cols]
    exp_expr = f"COALESCE({', '.join(matched_exp)}, 0)" if matched_exp else "0"

    yr_filter = f"TRY_CAST(t.{yr_col} AS INTEGER) BETWEEN {yr_min} AND {yr_max}" if yr_col else ""

    if st_col:
        wh = f"WHERE {yr_filter}" if yr_filter else ""
        sql = f"""
            SELECT
                UPPER(TRIM(CAST(t.{st_col} AS VARCHAR))) AS state,
                COUNT(*) AS filings,
                SUM({rev_expr}) AS total_revenue,
                SUM({ast_expr}) AS total_assets,
                SUM({exp_expr}) AS total_expenses,
                ROUND(SUM({rev_expr}) / NULLIF(COUNT(*), 0), 0) AS avg_revenue,
                ROUND(SUM({ast_expr}) / NULLIF(COUNT(*), 0), 0) AS avg_assets
            FROM ({base}) t
            {wh}
            GROUP BY 1
            HAVING state IS NOT NULL AND state != ''
            ORDER BY filings DESC
        """
    elif bmf_exists:
        filters = ["b.state IS NOT NULL", "b.state != ''"]
        if yr_filter:
            filters.append(yr_filter)
        wh = "WHERE " + " AND ".join(filters)
        sql = f"""
            SELECT
                UPPER(TRIM(CAST(b.state AS VARCHAR))) AS state,
                COUNT(*) AS filings,
                SUM({rev_expr}) AS total_revenue,
                SUM({ast_expr}) AS total_assets,
                SUM({exp_expr}) AS total_expenses,
                ROUND(SUM({rev_expr}) / NULLIF(COUNT(*), 0), 0) AS avg_revenue,
                ROUND(SUM({ast_expr}) / NULLIF(COUNT(*), 0), 0) AS avg_assets
            FROM ({base}) t
            LEFT JOIN read_parquet('{BMF_PATH}') b
                ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
            {wh}
            GROUP BY 1
            ORDER BY filings DESC
        """
    else:
        return pd.DataFrame()

    df = db.execute(sql).df()
    if not df.empty:
        df["state_name"] = df["state"].map(US_STATES).fillna(df["state"])
    return df


@st.cache_data(ttl=600, show_spinner="Loading sector treemap…")
def query_treemap_data(files: tuple, yr_min: int, yr_max: int, form: str, bmf_exists: bool) -> pd.DataFrame:
    if not bmf_exists:
        return pd.DataFrame()
    db = get_db()
    cols = sniff_columns(files[0]) if files else []
    yr_col = find_col(cols, COL_ALIASES["tax_year"])
    ein_col = find_col(cols, COL_ALIASES["ein"]) or "ein"
    base = build_view_sql(list(files))

    matched_rev = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_revenue"] if c in cols]
    rev_expr = f"COALESCE({', '.join(matched_rev)}, 0)" if matched_rev else "0"

    filters = ["b.ntee_cd IS NOT NULL", "b.ntee_cd != ''", "b.state IS NOT NULL", "b.state != ''"]
    if yr_col:
        filters.append(f"TRY_CAST(t.{yr_col} AS INTEGER) BETWEEN {yr_min} AND {yr_max}")
    wh = "WHERE " + " AND ".join(filters)

    sql = f"""
        SELECT
            UPPER(LEFT(TRIM(b.ntee_cd), 1)) AS ntee_major,
            UPPER(TRIM(b.state)) AS state,
            COUNT(*) AS filings,
            SUM({rev_expr}) AS revenue
        FROM ({base}) t
        LEFT JOIN read_parquet('{BMF_PATH}') b
            ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
        {wh}
        GROUP BY 1, 2
        HAVING filings > 5
        ORDER BY revenue DESC
    """
    df = db.execute(sql).df()
    if not df.empty:
        df["sector"] = df["ntee_major"].map(NTEE_MAJOR).fillna("Other")
        df["state_name"] = df["state"].map(US_STATES).fillna(df["state"])
        df = df[df["ntee_major"].ne("N")]
    return df


@st.cache_data(ttl=600, show_spinner="Loading organization point coordinates…")
def query_org_points(files: tuple, yr_min: int, yr_max: int, form: str,
                     state_filter: str, sector_filter: str, search_query: str,
                     sort_metric: str, limit: int, bmf_exists: bool) -> pd.DataFrame:
    """Query individual nonprofit organizations joined with ZIP centroid coordinates for map plotting."""
    if not bmf_exists or not ZIP_PATH.exists():
        return pd.DataFrame()
    db = get_db()
    cols = sniff_columns(files[0]) if files else []
    yr_col  = find_col(cols, COL_ALIASES["tax_year"])
    ein_col = find_col(cols, COL_ALIASES["ein"]) or "ein"
    base    = build_view_sql(list(files))

    matched_rev = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_revenue"] if c in cols]
    rev_expr = f"COALESCE({', '.join(matched_rev)}, 0)" if matched_rev else "0"

    matched_ast = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_assets"] if c in cols]
    ast_expr = f"COALESCE({', '.join(matched_ast)}, 0)" if matched_ast else "0"

    filters = [
        "b.name IS NOT NULL",
        "z.latitude IS NOT NULL",
        "z.longitude IS NOT NULL",
    ]
    if yr_col:
        filters.append(f"TRY_CAST(t.{yr_col} AS INTEGER) BETWEEN {yr_min} AND {yr_max}")
    if state_filter != "ALL":
        filters.append(f"UPPER(TRIM(b.state)) = '{state_filter}'")
    if sector_filter != "ALL":
        filters.append(f"UPPER(LEFT(TRIM(b.ntee_cd), 1)) = '{sector_filter}'")
    if search_query:
        clean_s = search_query.strip().replace("'", "''")
        filters.append(f"(b.name ILIKE '%{clean_s}%' OR b.city ILIKE '%{clean_s}%' OR LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') ILIKE '%{clean_s}%')")

    wh = "WHERE " + " AND ".join(filters)

    order_clause = "ORDER BY total_revenue DESC"
    if sort_metric == "Assets":
        order_clause = "ORDER BY total_assets DESC"
    elif sort_metric == "Random Sample":
        order_clause = "ORDER BY RANDOM()"

    sql = f"""
        SELECT
            LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') AS ein,
            b.name AS org_name,
            b.city,
            UPPER(TRIM(b.state)) AS state,
            LEFT(REPLACE(b.zip, '-', ''), 5) AS zip5,
            z.latitude,
            z.longitude,
            UPPER(LEFT(TRIM(b.ntee_cd), 1)) AS ntee_major,
            b.subsection,
            SUM({rev_expr}) AS total_revenue,
            MAX({ast_expr}) AS total_assets
        FROM ({base}) t
        LEFT JOIN read_parquet('{BMF_PATH}') b
            ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
        INNER JOIN read_parquet('{ZIP_PATH}') z
            ON LEFT(REPLACE(b.zip, '-', ''), 5) = z.zip
        {wh}
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
        {order_clause}
        LIMIT {limit}
    """
    df = db.execute(sql).df()
    if not df.empty:
        df["sector"] = df["ntee_major"].map(NTEE_MAJOR).fillna("Other / Unclassified")
        df["state_name"] = df["state"].map(US_STATES).fillna(df["state"])
    return df


@st.cache_data(ttl=600, show_spinner="Building charts…")
def query_form_type(files: tuple, yr_min: int, yr_max: int,
                    form: str, state: str) -> pd.DataFrame:
    db = get_db()
    cols   = sniff_columns(files[0]) if files else []
    yr_col = find_col(cols, COL_ALIASES["tax_year"])
    st_col = find_col(cols, COL_ALIASES["state"])
    ft_col = find_col(cols, COL_ALIASES["form_type"])
    if not ft_col:
        return pd.DataFrame()
    base = build_view_sql(list(files))
    wh   = where_clause(state, st_col, yr_min, yr_max, yr_col)
    sql  = f"""
        SELECT CAST({ft_col} AS VARCHAR) AS Form, COUNT(*) AS Filings
        FROM ({base}) t {wh}
        GROUP BY 1 ORDER BY 2 DESC
    """
    return db.execute(sql).df()


@st.cache_data(ttl=600, show_spinner="Building financial trends…")
def query_financial_trends(files: tuple, yr_min: int, yr_max: int,
                           form: str, state: str) -> pd.DataFrame:
    db = get_db()
    cols    = sniff_columns(files[0]) if files else []
    yr_col  = find_col(cols, COL_ALIASES["tax_year"])
    st_col  = find_col(cols, COL_ALIASES["state"])
    rev_col = find_col(cols, COL_ALIASES["total_revenue"])
    exp_col = find_col(cols, COL_ALIASES["total_expenses"])
    ast_col = find_col(cols, COL_ALIASES["total_assets"])
    if not yr_col:
        return pd.DataFrame()
    agg_parts = []
    labels = []
    if rev_col: agg_parts.append(f"SUM(TRY_CAST({rev_col} AS DOUBLE)) AS Revenue"); labels.append("Revenue")
    if exp_col: agg_parts.append(f"SUM(TRY_CAST({exp_col} AS DOUBLE)) AS Expenses"); labels.append("Expenses")
    if ast_col: agg_parts.append(f"SUM(TRY_CAST({ast_col} AS DOUBLE)) AS Assets"); labels.append("Assets")
    if not agg_parts:
        return pd.DataFrame()
    base = build_view_sql(list(files))
    wh   = where_clause(state, st_col, yr_min, yr_max, yr_col)
    sql  = f"""
        SELECT TRY_CAST({yr_col} AS INTEGER) AS Year, {', '.join(agg_parts)}
        FROM ({base}) t {wh}
        GROUP BY 1 ORDER BY 1
    """
    return db.execute(sql).df()


@st.cache_data(ttl=600, show_spinner="Building revenue histogram…")
def query_revenue_histogram(files: tuple, yr_min: int, yr_max: int,
                            form: str, state: str, n_bins: int = 80) -> pd.DataFrame:
    """Compute histogram buckets entirely in DuckDB — avoids pulling raw rows."""
    db = get_db()
    cols    = sniff_columns(files[0]) if files else []
    yr_col  = find_col(cols, COL_ALIASES["tax_year"])
    st_col  = find_col(cols, COL_ALIASES["state"])
    rev_col = find_col(cols, COL_ALIASES["total_revenue"])
    if not rev_col:
        return pd.DataFrame()
    base = build_view_sql(list(files))
    wh   = where_clause(state, st_col, yr_min, yr_max, yr_col)
    # Append revenue > 0 filter using AND if WHERE already exists, else new WHERE
    rev_filter = f"TRY_CAST({rev_col} AS DOUBLE) > 0"
    if wh:
        wh_full = wh + f" AND {rev_filter}"
    else:
        wh_full = f"WHERE {rev_filter}"
    # Clip at 99th percentile, then bucket
    sql = f"""
        WITH vals AS (
            SELECT TRY_CAST({rev_col} AS DOUBLE) AS v
            FROM ({base}) t {wh_full}
        ),
        pct AS (SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY v) AS p99 FROM vals),
        clipped AS (SELECT LEAST(v, (SELECT p99 FROM pct)) AS v FROM vals),
        bounds AS (SELECT MIN(v) AS lo, MAX(v) AS hi FROM clipped),
        bucketed AS (
            SELECT
                FLOOR((v - lo) / NULLIF(hi - lo, 0) * {n_bins}) AS bucket,
                lo, hi
            FROM clipped, bounds
        )
        SELECT
            bucket,
            lo + bucket * (hi - lo) / {n_bins} AS bin_start,
            COUNT(*) AS count
        FROM bucketed
        GROUP BY bucket, lo, hi
        ORDER BY bucket
    """
    return db.execute(sql).df()


@st.cache_data(ttl=600, show_spinner="Loading NTEE data…")
def query_ntee(files: tuple, yr_min: int, yr_max: int,
               form: str, state: str, bmf_exists: bool) -> pd.DataFrame:
    db = get_db()
    cols     = sniff_columns(files[0]) if files else []
    yr_col   = find_col(cols, COL_ALIASES["tax_year"])
    st_col   = find_col(cols, COL_ALIASES["state"])
    ntee_col = find_col(cols, COL_ALIASES["ntee"])
    # Try BMF join if no native NTEE col
    if not ntee_col and bmf_exists:
        ntee_col = "b.ntee_cd"  # in bmf_all.parquet the column is ntee_cd
    if not ntee_col:
        return pd.DataFrame()
    base = build_view_sql(list(files))
    wh   = where_clause(state, st_col, yr_min, yr_max, yr_col)
    if ntee_col == "b.ntee_cd":
        ein_col = find_col(cols, COL_ALIASES["ein"]) or "ein"
        ntee_filter = "b.ntee_cd IS NOT NULL AND TRIM(CAST(b.ntee_cd AS VARCHAR)) != ''"
        full_wh = (wh + f" AND {ntee_filter}") if wh else f"WHERE {ntee_filter}"
        sql = f"""
            SELECT UPPER(LEFT(TRIM(CAST(b.ntee_cd AS VARCHAR)), 1)) AS ntee_major, COUNT(*) AS count
            FROM ({base}) t
            LEFT JOIN read_parquet('{BMF_PATH}') b
                ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
            {full_wh}
            GROUP BY 1 ORDER BY 2 DESC
        """
    else:
        ntee_filter = f"{ntee_col} IS NOT NULL AND CAST({ntee_col} AS VARCHAR) != ''"
        full_wh = (wh + f" AND {ntee_filter}") if wh else f"WHERE {ntee_filter}"
        sql = f"""
            SELECT UPPER(LEFT(TRIM(CAST({ntee_col} AS VARCHAR)), 1)) AS ntee_major, COUNT(*) AS count
            FROM ({base}) t {full_wh}
            GROUP BY 1 ORDER BY 2 DESC
        """
    df = db.execute(sql).df()
    if df.empty:
        return df
    df["ntee_label"] = df["ntee_major"].map(NTEE_MAJOR).fillna("Other / Unclassified")
    return df[df["ntee_major"].ne("N")]


@st.cache_data(ttl=600, show_spinner="Loading state data…")
def query_state_detail(files: tuple, yr_min: int, yr_max: int,
                       form: str, target_state: str, bmf_exists: bool = True) -> dict:
    """Return filings-by-year and revenue-by-year for a single state."""
    db = get_db()
    cols    = sniff_columns(files[0]) if files else []
    yr_col  = find_col(cols, COL_ALIASES["tax_year"])
    st_col  = find_col(cols, COL_ALIASES["state"])
    ein_col = find_col(cols, COL_ALIASES["ein"]) or "ein"
    base    = build_view_sql(list(files))

    matched_rev = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_revenue"] if c in cols]
    rev_expr = f"COALESCE({', '.join(matched_rev)}, 0)" if matched_rev else "0"

    result = {}
    if st_col:
        wh = where_clause(target_state, st_col, yr_min, yr_max, yr_col)
        if yr_col:
            sql = f"""
                SELECT TRY_CAST(t.{yr_col} AS INTEGER) AS Year, COUNT(*) AS Filings
                FROM ({base}) t {wh}
                GROUP BY 1 ORDER BY 1
            """
            result["by_year"] = db.execute(sql).df()
            if matched_rev:
                sql2 = f"""
                    SELECT TRY_CAST(t.{yr_col} AS INTEGER) AS Year,
                           SUM({rev_expr}) AS "Total Revenue"
                    FROM ({base}) t {wh}
                    GROUP BY 1 ORDER BY 1
                """
                result["rev_trend"] = db.execute(sql2).df()
    elif bmf_exists:
        filters = [f"UPPER(TRIM(b.state)) = '{target_state}'"]
        if yr_col:
            filters.append(f"TRY_CAST(t.{yr_col} AS INTEGER) BETWEEN {yr_min} AND {yr_max}")
        wh = "WHERE " + " AND ".join(filters)
        if yr_col:
            sql = f"""
                SELECT TRY_CAST(t.{yr_col} AS INTEGER) AS Year, COUNT(*) AS Filings
                FROM ({base}) t
                LEFT JOIN read_parquet('{BMF_PATH}') b
                    ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
                {wh}
                GROUP BY 1 ORDER BY 1
            """
            result["by_year"] = db.execute(sql).df()
            if matched_rev:
                sql2 = f"""
                    SELECT TRY_CAST(t.{yr_col} AS INTEGER) AS Year,
                           SUM({rev_expr}) AS "Total Revenue"
                    FROM ({base}) t
                    LEFT JOIN read_parquet('{BMF_PATH}') b
                        ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
                    {wh}
                    GROUP BY 1 ORDER BY 1
                """
                result["rev_trend"] = db.execute(sql2).df()
    return result


@st.cache_data(ttl=600, show_spinner="Loading top orgs…")
def query_top_orgs(files: tuple, yr_min: int, yr_max: int,
                   form: str, target_state: str, bmf_exists: bool) -> pd.DataFrame:
    db = get_db()
    cols    = sniff_columns(files[0]) if files else []
    yr_col  = find_col(cols, COL_ALIASES["tax_year"])
    st_col  = find_col(cols, COL_ALIASES["state"])
    ein_col = find_col(cols, COL_ALIASES["ein"]) or "ein"
    base    = build_view_sql(list(files))

    matched_rev = [f"TRY_CAST(t.{c} AS DOUBLE)" for c in COL_ALIASES["total_revenue"] if c in cols]
    if not matched_rev:
        return pd.DataFrame()
    rev_expr = f"COALESCE({', '.join(matched_rev)}, 0)"

    if bmf_exists:
        filters = [f"UPPER(TRIM(b.state)) = '{target_state}'"]
        if yr_col:
            filters.append(f"TRY_CAST(t.{yr_col} AS INTEGER) BETWEEN {yr_min} AND {yr_max}")
        wh = "WHERE " + " AND ".join(filters)
        sql = f"""
            SELECT b.name AS "Org Name",
                   LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') AS EIN,
                   SUM({rev_expr}) AS "Total Revenue"
            FROM ({base}) t
            LEFT JOIN read_parquet('{BMF_PATH}') b
                ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
            {wh}
            GROUP BY 1, 2 ORDER BY 3 DESC NULLS LAST LIMIT 20
        """
    else:
        wh = where_clause(target_state, st_col, yr_min, yr_max, yr_col)
        sql = f"""
            SELECT LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') AS EIN,
                   SUM({rev_expr}) AS "Total Revenue"
            FROM ({base}) t {wh}
            GROUP BY 1 ORDER BY 2 DESC NULLS LAST LIMIT 20
        """
    return db.execute(sql).df()


@st.cache_data(ttl=600, show_spinner="Loading browse data…")
def query_browse(files: tuple, yr_min: int, yr_max: int, form: str,
                 state: str, row_limit: int, search: str,
                 bmf_exists: bool) -> pd.DataFrame:
    """Pull only row_limit rows of curated columns for the Browse tab."""
    db = get_db()
    cols    = sniff_columns(files[0]) if files else []
    yr_col  = find_col(cols, COL_ALIASES["tax_year"])
    st_col  = find_col(cols, COL_ALIASES["state"])
    rev_col = find_col(cols, COL_ALIASES["total_revenue"])
    exp_col = find_col(cols, COL_ALIASES["total_expenses"])
    ast_col = find_col(cols, COL_ALIASES["total_assets"])
    ein_col = find_col(cols, COL_ALIASES["ein"]) or "ein"
    ft_col  = find_col(cols, COL_ALIASES["form_type"])
    ntee_col= find_col(cols, COL_ALIASES["ntee"])

    base = build_view_sql(list(files))
    wh   = where_clause(state, st_col, yr_min, yr_max, yr_col)

    # Build SELECT list
    select_parts = [f"LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') AS ein"]
    if yr_col:  select_parts.append(f"TRY_CAST(t.{yr_col} AS INTEGER) AS tax_year")
    if ft_col:  select_parts.append(f"CAST(t.{ft_col} AS VARCHAR) AS form_type")
    if st_col and st_col != "state":
        select_parts.append(f"UPPER(TRIM(CAST(t.{st_col} AS VARCHAR))) AS state")
    elif st_col:
        select_parts.append(f"UPPER(TRIM(CAST(t.state AS VARCHAR))) AS state")
    if rev_col: select_parts.append(f"TRY_CAST(t.{rev_col} AS DOUBLE) AS total_revenue")
    if exp_col: select_parts.append(f"TRY_CAST(t.{exp_col} AS DOUBLE) AS total_expenses")
    if ast_col: select_parts.append(f"TRY_CAST(t.{ast_col} AS DOUBLE) AS total_assets")
    if ntee_col:select_parts.append(f"CAST(t.{ntee_col} AS VARCHAR) AS ntee_code")

    search_filter = ""
    if search:
        s = search.strip().replace("'", "''")
        search_parts = [f"CAST(t.{ein_col} AS VARCHAR) ILIKE '%{s}%'"]
        if st_col: search_parts.append(f"CAST(t.{st_col} AS VARCHAR) ILIKE '%{s}%'")
        if bmf_exists: search_parts += [f"b.name ILIKE '%{s}%'", f"b.city ILIKE '%{s}%'"]
        search_filter = "(" + " OR ".join(search_parts) + ")"

    # Combine wh + search_filter into a single WHERE block
    filters_parts = []
    if wh:  # already starts with "WHERE ..."
        filters_parts.append(wh[len("WHERE "):])  # strip the leading "WHERE "
    if search_filter:
        filters_parts.append(search_filter)
    combined_wh = ("WHERE " + " AND ".join(filters_parts)) if filters_parts else ""

    if bmf_exists:
        select_parts_bmf = [
            "b.name AS org_name", "b.city", "b.zip",
            "b.ntee_cd AS ntee_code_bmf", "b.subsection",
        ]
        sql = f"""
            SELECT {', '.join(select_parts)}, {', '.join(select_parts_bmf)}
            FROM ({base}) t
            LEFT JOIN read_parquet('{BMF_PATH}') b
                ON LPAD(CAST(t.{ein_col} AS VARCHAR), 9, '0') = LPAD(CAST(b.ein AS VARCHAR), 9, '0')
            {combined_wh}
            LIMIT {row_limit}
        """
    else:
        sql = f"""
            SELECT {', '.join(select_parts)}
            FROM ({base}) t {combined_wh}
            LIMIT {row_limit}
        """

    df = db.execute(sql).df()

    # Add NTEE label
    ntee_src = "ntee_code_bmf" if "ntee_code_bmf" in df.columns else "ntee_code" if "ntee_code" in df.columns else None
    if ntee_src:
        df["ntee_label"] = df[ntee_src].astype(str).str.strip().str[:1].str.upper().map(NTEE_MAJOR).fillna("Other")

    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(files):
    st.sidebar.title("🔍 Filters")

    years_avail = sorted({parse_year(f.name) for f in files if parse_year(f.name)})
    if len(years_avail) > 1:
        yr_min, yr_max = min(years_avail), max(years_avail)
        default_start = max(yr_min, yr_max - 2)
        year_range = st.sidebar.slider("Tax Year Range", yr_min, yr_max,
                                       value=(default_start, yr_max))
    elif len(years_avail) == 1:
        yr = years_avail[0]
        st.sidebar.markdown(f"**Tax Year:** `{yr}` *(download more years to enable range slider)*")
        year_range = (yr, yr)
    else:
        year_range = (2012, 2024)

    form_sel = st.sidebar.selectbox(
        "Form Type", ["ALL","990","990ez","990pf"],
        format_func=lambda x: FORM_LABELS.get(x,x))

    state_sel = st.sidebar.selectbox(
        "State", ["ALL"] + sorted(US_STATES.keys()),
        format_func=lambda x: "All States" if x=="ALL" else f"{x} – {US_STATES.get(x,x)}")

    row_limit = st.sidebar.select_slider(
        "Max display rows (Browse tab)", [500,1_000,5_000,10_000,50_000], value=5_000)

    return {"year_range": year_range, "form": form_sel,
            "state": state_sel, "row_limit": row_limit}


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

def tab_overview(files, filters, bmf_exists):
    yr_min, yr_max = filters["year_range"]
    form  = filters["form"]
    state = filters["state"]
    ftup  = tuple(files)

    kpis = query_kpis(ftup, yr_min, yr_max, form, state)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Filings", f"{kpis.get('n_filings', 0):,}")
    if "total_rev" in kpis and kpis["total_rev"]:
        k2.metric("Total Revenue", f"${kpis['total_rev']/1e9:.2f}B")
    if "total_ast" in kpis and kpis["total_ast"]:
        k3.metric("Total Assets", f"${kpis['total_ast']/1e9:.2f}B")
    if "total_exp" in kpis and kpis["total_exp"]:
        k4.metric("Total Expenses", f"${kpis['total_exp']/1e9:.2f}B")

    st.markdown("---")

    yr_df = query_filings_by_year(ftup, yr_min, yr_max, form, state)
    if not yr_df.empty:
        fig = px.bar(yr_df, x="Year", y="Filings",
                     title="📅 Filings by Tax Year", color="Filings",
                     color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    st_df = query_by_state(ftup, yr_min, yr_max, form, state, bmf_exists)
    if not st_df.empty:
        with col_a:
            top20 = st_df.head(20)
            fig2 = px.bar(top20, x="State", y="Filings",
                          title="🗺️ Top 20 States by Filings", color="Filings",
                          color_continuous_scale="Teal")
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            map_df = st_df.rename(columns={"State":"state","Filings":"count"})
            fig3 = px.choropleth(map_df, locations="state", locationmode="USA-states",
                                 color="count", scope="usa",
                                 color_continuous_scale="YlOrRd",
                                 title="🗺️ Filings by State (Choropleth)")
            st.plotly_chart(fig3, use_container_width=True)

    ft_df = query_form_type(ftup, yr_min, yr_max, form, state)
    if not ft_df.empty:
        fig4 = px.pie(ft_df, names="Form", values="Filings",
                      title="📋 Filings by Form Type")
        st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Map & Treemap Explorer
# ---------------------------------------------------------------------------

def tab_map_explorer(files, filters, bmf_exists):
    yr_min, yr_max = filters["year_range"]
    form  = filters["form"]
    ftup  = tuple(files)

    st.subheader("🗺️ Geographic Map & Treemap Explorer")
    st.caption("Interactive spatial distribution and hierarchical sector structures across US nonprofits.")

    view_type = st.radio(
        "Map Structure View",
        [
            "📍 Organization Point Map (Interactive Dots)",
            "🇺🇸 US Choropleth Map (State Aggregates)",
            "🌲 Sector & State Treemap (Hierarchical)",
            "📊 State Rankings & Matrix",
        ],
        horizontal=True,
    )

    if view_type == "📍 Organization Point Map (Interactive Dots)":
        st.markdown("#### 📍 Individual Nonprofit Organizations Plotted on Map")
        st.caption("Each point represents an organization positioned at its ZIP code centroid. Hover over dots to inspect financials, sector, and location.")

        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
        with f_col1:
            state_options = ["ALL"] + sorted(US_STATES.keys())
            curr_st_idx = 0
            if filters["state"] in state_options:
                curr_st_idx = state_options.index(filters["state"])
            map_state_sel = st.selectbox(
                "Filter State",
                state_options,
                index=curr_st_idx,
                format_func=lambda x: "All States (National)" if x == "ALL" else f"{x} – {US_STATES.get(x, x)}",
                key="point_map_state"
            )
        with f_col2:
            sector_options = [("ALL", "All Sectors")] + [(k, f"{k} – {v}") for k, v in sorted(NTEE_MAJOR.items())]
            map_sector_sel = st.selectbox(
                "Filter Sector (NTEE)",
                sector_options,
                format_func=lambda x: x[1],
                key="point_map_sector"
            )[0]
        with f_col3:
            map_search = st.text_input(
                "Quick Org / City Search",
                placeholder="e.g. Red Cross, Hospital, Boston...",
                key="point_map_search"
            )

        c_col1, c_col2, c_col3 = st.columns([1, 1, 1])
        with c_col1:
            point_limit = st.select_slider(
                "Number of Orgs to Plot",
                options=[50, 100, 250, 500, 1000],
                value=250,
                key="point_map_limit"
            )
        with c_col2:
            sort_metric = st.selectbox(
                "Select Organizations By",
                ["Top by Revenue ($)", "Top by Assets ($)", "Random Sample"],
                key="point_map_sort"
            )
        with c_col3:
            size_mode = st.selectbox(
                "Bubble Sizing",
                ["Proportional to Revenue", "Proportional to Assets", "Uniform Size"],
                key="point_map_size"
            )

        sort_key = "Revenue" if "Revenue" in sort_metric else "Assets" if "Assets" in sort_metric else "Random Sample"
        df_pts = query_org_points(
            ftup, yr_min, yr_max, form,
            state_filter=map_state_sel,
            sector_filter=map_sector_sel,
            search_query=map_search,
            sort_metric=sort_key,
            limit=point_limit,
            bmf_exists=bmf_exists
        )

        if df_pts.empty:
            st.warning("No organizations match the selected filters with valid coordinates. Ensure BMF and ZIP coordinates are present.")
        else:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Organizations Plotted", f"{len(df_pts):,}")
            total_rev_pts = df_pts["total_revenue"].sum()
            p2.metric("Total Plotted Revenue", f"${total_rev_pts/1e9:.2f}B" if total_rev_pts >= 1e9 else f"${total_rev_pts/1e6:.1f}M")
            top_org = df_pts.sort_values("total_revenue", ascending=False).iloc[0]
            org_short = top_org["org_name"][:26] + "..." if len(top_org["org_name"]) > 26 else top_org["org_name"]
            p3.metric("Top Revenue Org", org_short)
            p4.metric("Top Org Revenue", f"${top_org['total_revenue']/1e9:.2f}B" if top_org['total_revenue'] >= 1e9 else f"${top_org['total_revenue']/1e6:.1f}M")

            st.markdown("---")

            if size_mode == "Proportional to Revenue":
                df_pts["plot_size"] = np.sqrt(df_pts["total_revenue"].clip(lower=0)) + 1
                size_col = "plot_size"
                size_max = 24
            elif size_mode == "Proportional to Assets":
                df_pts["plot_size"] = np.sqrt(df_pts["total_assets"].clip(lower=0)) + 1
                size_col = "plot_size"
                size_max = 24
            else:
                df_pts["plot_size"] = 8
                size_col = "plot_size"
                size_max = 8

            map_title = (
                f"📍 Top {len(df_pts):,} Organizations in {US_STATES.get(map_state_sel, map_state_sel)} ({map_state_sel})"
                if map_state_sel != "ALL"
                else f"📍 Top {len(df_pts):,} Organizations Plotted Nationwide (US)"
            )

            fig_pts = px.scatter_geo(
                df_pts,
                lat="latitude",
                lon="longitude",
                hover_name="org_name",
                hover_data={
                    "city": True,
                    "state": True,
                    "zip5": True,
                    "sector": True,
                    "total_revenue": ":$,.0f",
                    "total_assets": ":$,.0f",
                    "latitude": False,
                    "longitude": False,
                    "plot_size": False,
                },
                color="sector",
                size=size_col,
                size_max=size_max,
                scope="usa",
                title=map_title,
            )

            geo_layout = dict(
                bgcolor="rgba(0,0,0,0)",
                lakecolor="rgba(255,255,255,0.1)",
                showsubunits=True,
                subunitcolor="rgba(120, 120, 120, 0.4)",
            )
            # If a specific state is chosen, auto-zoom directly into that state boundary
            if map_state_sel != "ALL":
                geo_layout["fitbounds"] = "locations"
                geo_layout["visible"] = True

            fig_pts.update_layout(
                margin={"r":0,"t":40,"l":0,"b":0},
                geo=geo_layout,
                legend=dict(title="Sector", orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                height=650,
            )
            st.plotly_chart(fig_pts, use_container_width=True)

            # State-wise City Distribution Breakdown
            if map_state_sel != "ALL" and "city" in df_pts.columns and not df_pts.empty:
                st.markdown(f"##### 🏙️ Nonprofit Concentrations by City in {US_STATES.get(map_state_sel, map_state_sel)}")
                c_bar1, c_bar2 = st.columns(2)
                with c_bar1:
                    city_rev = df_pts.groupby("city")["total_revenue"].sum().reset_index().sort_values("total_revenue", ascending=False).head(10)
                    fig_city_rev = px.bar(
                        city_rev, x="city", y="total_revenue",
                        title=f"Top Cities by Plotted Revenue — {map_state_sel}",
                        color="total_revenue", color_continuous_scale="Teal",
                        labels={"city": "City", "total_revenue": "Revenue ($)"}
                    )
                    fig_city_rev.update_layout(height=320, xaxis_title="City")
                    st.plotly_chart(fig_city_rev, use_container_width=True)
                with c_bar2:
                    city_cnt = df_pts.groupby("city").size().reset_index(name="Orgs").sort_values("Orgs", ascending=False).head(10)
                    fig_city_cnt = px.bar(
                        city_cnt, x="city", y="Orgs",
                        title=f"Top Cities by Plotted Organizations — {map_state_sel}",
                        color="Orgs", color_continuous_scale="Blues",
                        labels={"city": "City", "Orgs": "# Orgs"}
                    )
                    fig_city_cnt.update_layout(height=320, xaxis_title="City")
                    st.plotly_chart(fig_city_cnt, use_container_width=True)

            with st.expander(f"📋 Plotted Organizations Directory ({len(df_pts)} orgs)", expanded=False):
                tbl_cols = ["org_name", "ein", "city", "state", "zip5", "sector", "total_revenue", "total_assets"]
                tbl_display = df_pts[tbl_cols].rename(columns={
                    "org_name": "Organization Name", "ein": "EIN", "city": "City",
                    "state": "State", "zip5": "ZIP", "sector": "NTEE Sector",
                    "total_revenue": "Total Revenue ($)", "total_assets": "Total Assets ($)"
                })
                col_cfg = {
                    "Total Revenue ($)": st.column_config.NumberColumn(format="$%d"),
                    "Total Assets ($)": st.column_config.NumberColumn(format="$%d"),
                    "EIN": st.column_config.TextColumn(),
                }
                st.dataframe(tbl_display, column_config=col_cfg, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download Plotted Orgs CSV",
                    tbl_display.to_csv(index=False),
                    "nccs990_plotted_orgs.csv",
                    "text/csv"
                )

    elif view_type == "🇺🇸 US Choropleth Map (State Aggregates)":
        df_map = query_map_data(ftup, yr_min, yr_max, form, bmf_exists)
        if df_map.empty:
            st.warning("No geographic data available. Ensure BMF is downloaded.")
            return

        df_us = df_map[df_map["state"].isin(US_STATES.keys())].copy()
        c1, c2 = st.columns([2, 1])
        with c1:
            metric_choice = st.selectbox(
                "Map Metric",
                [
                    ("filings", "📋 Total Filings"),
                    ("total_revenue", "💰 Total Revenue ($)"),
                    ("total_assets", "🏦 Total Assets ($)"),
                    ("avg_revenue", "📊 Average Revenue ($/org)"),
                    ("avg_assets", "📈 Average Assets ($/org)"),
                ],
                format_func=lambda x: x[1],
            )
        with c2:
            colorscale = st.selectbox(
                "Color Theme",
                ["YlOrRd", "Viridis", "Plasma", "Blues", "Teal", "Turbo", "Sunset"],
            )

        metric_col, metric_label = metric_choice

        # Top summary KPIs for map
        top_state_row = df_us.sort_values(metric_col, ascending=False).iloc[0]
        national_total = df_us[metric_col].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("States Represented", f"{len(df_us):,}")
        if "filings" in metric_col:
            m2.metric("Total US Filings", f"{int(national_total):,}")
        elif "avg" in metric_col:
            m2.metric("National Average", f"${national_total / len(df_us):,.0f}")
        else:
            m2.metric("Total National Sum", f"${national_total/1e9:.2f}B" if national_total >= 1e9 else f"${national_total/1e6:.1f}M")
        m3.metric(f"Top State ({top_state_row['state']})", top_state_row["state_name"])
        top_val = top_state_row[metric_col]
        m4.metric(
            f"{top_state_row['state']} Value",
            f"${top_val/1e9:.2f}B" if top_val >= 1e9 and "filings" not in metric_col else f"{int(top_val):,}" if "filings" in metric_col else f"${top_val:,.0f}"
        )

        st.markdown("---")

        fig_map = px.choropleth(
            df_us,
            locations="state",
            locationmode="USA-states",
            color=metric_col,
            scope="usa",
            color_continuous_scale=colorscale,
            labels={metric_col: metric_label},
            hover_name="state_name",
            hover_data={
                "state": True,
                "filings": ":,d",
                "total_revenue": ":$,.0f",
                "total_assets": ":$,.0f",
                "avg_revenue": ":$,.0f",
            },
            title=f"🗺️ US Geographic Distribution — {metric_label}",
        )
        fig_map.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(255,255,255,0.1)"),
            height=580,
        )
        st.plotly_chart(fig_map, use_container_width=True)

        top15 = df_us.sort_values(metric_col, ascending=False).head(15)
        fig_bar = px.bar(
            top15,
            x="state",
            y=metric_col,
            color=metric_col,
            color_continuous_scale=colorscale,
            labels={metric_col: metric_label, "state": "State"},
            title=f"🏆 Top 15 States by {metric_label}",
            hover_name="state_name",
        )
        fig_bar.update_layout(xaxis_title="State", height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    elif view_type == "🌲 Sector & State Treemap (Hierarchical)":
        st.markdown("#### 🌲 Hierarchical Treemap Structure")
        st.caption("Treemaps represent nested rectangles where area corresponds to relative economic size.")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tree_metric = st.selectbox("Treemap Size Metric", ["Revenue ($)", "Filings Count"])
        with col_t2:
            tree_hierarchy = st.selectbox(
                "Hierarchy Organization",
                ["Sector ➔ State (Explore how each sector is distributed across states)",
                 "State ➔ Sector (Explore sector makeup within each state)"]
            )

        df_tree = query_treemap_data(ftup, yr_min, yr_max, form, bmf_exists)
        if df_tree.empty:
            st.info("Treemap requires NTEE sector classifications from BMF.")
        else:
            val_col = "revenue" if "Revenue" in tree_metric else "filings"
            if "Sector ➔ State" in tree_hierarchy:
                path = ["sector", "state"]
                title_text = f"🌲 Nonprofit Sector ➔ State Treemap (Sized by {tree_metric})"
            else:
                top_states_list = df_us.head(20)["state"].tolist()
                df_tree = df_tree[df_tree["state"].isin(top_states_list)]
                path = ["state_name", "sector"]
                title_text = f"🌲 State ➔ Sector Treemap (Sized by {tree_metric}, Top 20 States)"

            fig_tree = px.treemap(
                df_tree,
                path=path,
                values=val_col,
                color=val_col,
                color_continuous_scale="Viridis",
                title=title_text,
                hover_data={"filings": ":,d", "revenue": ":$,.0f"}
            )
            fig_tree.update_layout(height=650, margin=dict(t=50, l=10, r=10, b=10))
            st.plotly_chart(fig_tree, use_container_width=True)
            st.caption("💡 *Tip: Click on any sector or state rectangle to zoom in. Click the top header to zoom back out.*")

    else:  # State Rankings Matrix
        st.markdown("#### 📊 Comprehensive State Comparison Matrix")
        df_map = query_map_data(ftup, yr_min, yr_max, form, bmf_exists)
        if df_map.empty:
            st.warning("No geographic data available. Ensure BMF is downloaded.")
            return
        df_us = df_map[df_map["state"].isin(US_STATES.keys())].copy()

        ranking_df = df_us.copy().sort_values("filings", ascending=False).reset_index(drop=True)
        ranking_df.index = ranking_df.index + 1
        ranking_df.reset_index(inplace=True)
        ranking_df.rename(columns={"index": "Rank"}, inplace=True)

        tot_filings = ranking_df["filings"].sum()
        tot_rev = ranking_df["total_revenue"].sum()
        ranking_df["% US Filings"] = (ranking_df["filings"] / tot_filings * 100).round(1)
        ranking_df["% US Revenue"] = (ranking_df["total_revenue"] / (tot_rev if tot_rev > 0 else 1) * 100).round(1)

        display_cols = [
            "Rank", "state", "state_name", "filings", "% US Filings",
            "total_revenue", "% US Revenue", "total_assets", "avg_revenue"
        ]
        col_names_friendly = {
            "state": "State",
            "state_name": "State Name",
            "filings": "Filings",
            "total_revenue": "Total Revenue ($)",
            "total_assets": "Total Assets ($)",
            "avg_revenue": "Avg Revenue ($)",
        }
        ranking_display = ranking_df[display_cols].rename(columns=col_names_friendly)

        col_configs = {
            "Filings": st.column_config.NumberColumn(format="%,d"),
            "Total Revenue ($)": st.column_config.NumberColumn(format="$%d"),
            "Total Assets ($)": st.column_config.NumberColumn(format="$%d"),
            "Avg Revenue ($)": st.column_config.NumberColumn(format="$%d"),
            "% US Filings": st.column_config.NumberColumn(format="%.1f%%"),
            "% US Revenue": st.column_config.NumberColumn(format="%.1f%%"),
        }

        st.dataframe(ranking_display, column_config=col_configs, use_container_width=True, height=520, hide_index=True)

        csv_data = ranking_display.to_csv(index=False)
        st.download_button(
            "⬇️ Download State Rankings CSV",
            csv_data,
            "nccs990_state_rankings.csv",
            "text/csv",
        )


# ---------------------------------------------------------------------------
# Tab: Financials
# ---------------------------------------------------------------------------

def tab_financials(files, filters, bmf_exists):
    yr_min, yr_max = filters["year_range"]
    form  = filters["form"]
    state = filters["state"]
    ftup  = tuple(files)

    trend = query_financial_trends(ftup, yr_min, yr_max, form, state)
    if trend.empty:
        st.info("No financial columns detected.")
        return

    value_cols = [c for c in trend.columns if c != "Year"]
    if value_cols:
        fig = px.line(trend, x="Year", y=value_cols,
                      title="💰 Financial Trends Over Time")
        st.plotly_chart(fig, use_container_width=True)

    hist_df = query_revenue_histogram(ftup, yr_min, yr_max, form, state)
    if not hist_df.empty:
        fig2 = px.bar(hist_df, x="bin_start", y="count",
                      title="📊 Revenue Distribution (99th pct clipped)",
                      labels={"bin_start": "Total Revenue ($)", "count": "# Filings"})
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: NTEE Sectors
# ---------------------------------------------------------------------------

def tab_ntee(files, filters, bmf_exists):
    yr_min, yr_max = filters["year_range"]
    ftup = tuple(files)
    df = query_ntee(ftup, yr_min, yr_max, filters["form"], filters["state"], bmf_exists)
    if df.empty:
        st.info("No NTEE column detected. Download the BMF to add NTEE classifications.")
        return

    fig = px.bar(df, x="ntee_label", y="count",
                 title="🏷️ Filings by NTEE Category",
                 color="count", color_continuous_scale="Viridis",
                 labels={"ntee_label":"Category","count":"Filings"})
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.pie(df, names="ntee_label", values="count",
                  title="🥧 NTEE Category Share")
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: State Deep Dive
# ---------------------------------------------------------------------------

def tab_state_deep_dive(files, filters, bmf_exists):
    yr_min, yr_max = filters["year_range"]
    ftup = tuple(files)

    states = sorted(US_STATES.keys())
    sel = st.selectbox("Select a state", states,
                       format_func=lambda x: f"{x} – {US_STATES.get(x,x)}")

    detail = query_state_detail(ftup, yr_min, yr_max, filters["form"], sel, bmf_exists)
    n_filings = detail.get("by_year", pd.DataFrame())
    total = int(n_filings["Filings"].sum()) if not n_filings.empty else 0
    st.metric(f"Filings in {sel} ({US_STATES.get(sel, sel)})", f"{total:,}")

    col_a, col_b = st.columns(2)
    if "by_year" in detail and not detail["by_year"].empty:
        with col_a:
            fig = px.line(detail["by_year"], x="Year", y="Filings",
                          title=f"📅 Filings Over Time — {sel}", markers=True)
            st.plotly_chart(fig, use_container_width=True)
    if "rev_trend" in detail and not detail["rev_trend"].empty:
        with col_b:
            fig2 = px.area(detail["rev_trend"], x="Year", y="Total Revenue",
                           title=f"💰 Total Revenue Over Time — {sel}")
            st.plotly_chart(fig2, use_container_width=True)

    # State-wise Organization Point Map
    st.subheader(f"📍 Geographic Distribution of Organizations in {US_STATES.get(sel, sel)}")
    df_state_pts = query_org_points(
        ftup, yr_min, yr_max, filters["form"],
        state_filter=sel, sector_filter="ALL", search_query="",
        sort_metric="Revenue", limit=150, bmf_exists=bmf_exists
    )
    if not df_state_pts.empty:
        df_state_pts["plot_size"] = np.sqrt(df_state_pts["total_revenue"].clip(lower=0)) + 1
        fig_st_map = px.scatter_geo(
            df_state_pts,
            lat="latitude",
            lon="longitude",
            hover_name="org_name",
            hover_data={
                "city": True,
                "zip5": True,
                "sector": True,
                "total_revenue": ":$,.0f",
                "total_assets": ":$,.0f",
                "latitude": False,
                "longitude": False,
                "plot_size": False,
            },
            color="sector",
            size="plot_size",
            size_max=22,
            scope="usa",
            title=f"📍 Top Organizations in {US_STATES.get(sel, sel)} ({len(df_state_pts)} Plotted)",
        )
        fig_st_map.update_geos(
            fitbounds="locations",
            visible=True,
            showsubunits=True,
            subunitcolor="rgba(120, 120, 120, 0.4)",
        )
        fig_st_map.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            geo=dict(bgcolor="rgba(0,0,0,0)"),
            legend=dict(title="Sector", orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=520,
        )
        st.plotly_chart(fig_st_map, use_container_width=True)

    top_orgs = query_top_orgs(ftup, yr_min, yr_max, filters["form"], sel, bmf_exists)
    if not top_orgs.empty:
        st.subheader(f"🏆 Top 20 Organizations in {sel} by Revenue")
        st.dataframe(top_orgs, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Browse
# ---------------------------------------------------------------------------

def tab_browse(files, filters, bmf_exists):
    st.subheader("📋 Nonprofit Data Browser")
    st.caption("Inspect, search, and download granular Form 990 filings.")

    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search = st.text_input("🔍 Quick Search", placeholder="Search by org name, EIN, city, state…")

    yr_min, yr_max = filters["year_range"]
    df = query_browse(tuple(files), yr_min, yr_max, filters["form"],
                      filters["state"], filters["row_limit"], search, bmf_exists)

    if df.empty:
        st.info("No results found.")
        return

    # Friendly column rename
    rename_map = {
        "org_name": "Organization Name", "ein": "EIN", "state": "State",
        "city": "City", "zip": "ZIP", "tax_year": "Tax Year",
        "form_type": "Form Type", "total_revenue": "Total Revenue ($)",
        "total_expenses": "Total Expenses ($)", "total_assets": "Total Assets ($)",
        "ntee_code": "NTEE Code", "ntee_code_bmf": "NTEE Code",
        "ntee_label": "NTEE Sector", "subsection": "IRC Subsection",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Drop duplicate label columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows in View", f"{len(df):,}")
    if "Total Revenue ($)" in df.columns:
        rev = pd.to_numeric(df["Total Revenue ($)"], errors="coerce")
        m2.metric("Median Revenue", f"${rev.median():,.0f}")
        m3.metric("Max Revenue", f"${rev.max():,.0f}")
    if "Total Assets ($)" in df.columns:
        ast = pd.to_numeric(df["Total Assets ($)"], errors="coerce")
        m4.metric("Total Assets", f"${ast.sum()/1e6:,.1f}M")

    # Column configs
    col_configs = {}
    for col in df.columns:
        if any(kw in col for kw in ["($)", "Revenue", "Expense", "Asset", "Surplus"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            col_configs[col] = st.column_config.NumberColumn(col, format="$%d")
        elif col == "EIN":
            col_configs[col] = st.column_config.TextColumn(col)

    st.dataframe(df, column_config=col_configs, use_container_width=True,
                 height=520, hide_index=True)

    d1, d2 = st.columns([1, 4])
    with d1:
        st.download_button("⬇️ Download CSV", df.to_csv(index=False),
                           "nccs990_data.csv", "text/csv")
    with d2:
        st.caption(f"Showing **{len(df):,}** rows (capped at {filters['row_limit']:,}).")

    with st.expander("📖 IRS Field Reference", expanded=False):
        dict_rows = [{"Plain English": m["label"], "IRS Code": k,
                      "Category": m["category"], "Definition": m["desc"]}
                     for k, m in sorted(COLUMN_METADATA.items(), key=lambda x: x[1]["category"])]
        dict_df = pd.DataFrame(dict_rows)
        ds = st.text_input("Filter Dictionary", placeholder="Search…")
        if ds:
            dict_df = dict_df[dict_df.apply(lambda r: r.astype(str).str.contains(ds, case=False).any(), axis=1)]
        st.dataframe(dict_df, use_container_width=True, height=350, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Downloading NCCS 990 dataset from Hugging Face (first boot only)…")
def ensure_hf_data(repo_id: str) -> None:
    """Download Parquet files from a Hugging Face Dataset repo into DATA_DIR."""
    from huggingface_hub import snapshot_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=["*.parquet"],
    )


def main():
    st.set_page_config(
        page_title="NCCS 990 Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 NCCS / IRS 990 Nonprofit Filing Explorer")
    st.caption(
        "IRS Statistics of Income · Form 990, 990-EZ & 990-PF · "
        "All 50 US states · 2012–2024 (IRS SOI) · 1989–2011 (Legacy, via NCCS)"
    )

    # --- Check data ---
    soi_files    = sorted(DATA_DIR.glob("irs_soi_*.parquet")) if DATA_DIR.exists() else []
    legacy_files = sorted(DATA_DIR.glob("legacy_*.parquet")) if DATA_DIR.exists() else []
    all_files    = soi_files + legacy_files

    hf_dataset_repo = os.environ.get("HF_DATASET_REPO", "").strip()
    if not all_files and hf_dataset_repo:
        ensure_hf_data(hf_dataset_repo)
        soi_files    = sorted(DATA_DIR.glob("irs_soi_*.parquet")) if DATA_DIR.exists() else []
        legacy_files = sorted(DATA_DIR.glob("legacy_*.parquet")) if DATA_DIR.exists() else []
        all_files    = soi_files + legacy_files

    if not all_files:
        st.error("⚠️ No data files found in `./data/`.")
        st.markdown("""
        ### Quick Start
        ```bash
        # 1. Download a single year (fast, ~11 MB, ~230k rows)
        python irs_soi_downloader.py --year 2022 --form 990ez

        # 2. Download all years for all 3 form types (2012–2024)
        python irs_soi_downloader.py --workers 3

        # 3. Download BMF (org names, states, NTEE codes)
        python download_bmf.py

        # 4. Launch app
        streamlit run app.py
        ```
        """)
        st.stop()

    bmf_exists = BMF_PATH.exists()

    # --- Sidebar ---
    filters = render_sidebar(all_files)
    yr_min, yr_max = filters["year_range"]

    # Select matching files
    sel_files = []
    for f in all_files:
        yr  = parse_year(f.name)
        frm = parse_form(f.name)
        if yr and yr_min <= yr <= yr_max:
            if filters["form"] == "ALL" or frm == filters["form"]:
                sel_files.append(str(f))

    if not sel_files:
        st.warning("No files match the current filters.")
        st.stop()

    st.sidebar.info(f"**{len(sel_files)}** file(s) selected")
    if not bmf_exists:
        st.sidebar.warning("BMF not loaded → no org names.\nRun: `python download_bmf.py`")

    # Count total rows (cheap — DuckDB metadata)
    try:
        db = get_db()
        n = db.execute(
            f"SELECT COUNT(*) FROM ({build_view_sql(sel_files)}) t"
        ).fetchone()[0]
        st.sidebar.success(f"**{n:,}** total filings")
    except Exception:
        pass

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Overview", "🗺️ Map & Treemap Explorer", "💰 Financials",
        "🏷️ NTEE Sectors", "🏛️ State Deep Dive", "📋 Browse Data",
    ])
    with tab1: tab_overview(sel_files, filters, bmf_exists)
    with tab2: tab_map_explorer(sel_files, filters, bmf_exists)
    with tab3: tab_financials(sel_files, filters, bmf_exists)
    with tab4: tab_ntee(sel_files, filters, bmf_exists)
    with tab5: tab_state_deep_dive(sel_files, filters, bmf_exists)
    with tab6: tab_browse(sel_files, filters, bmf_exists)


if __name__ == "__main__":
    main()
