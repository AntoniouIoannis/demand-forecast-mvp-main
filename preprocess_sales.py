#!/usr/bin/env python3
"""
Preprocessing pipeline for 2017–2019 sales Excel files.

What it does
------------
1) Reads the three yearly sales Excel files.
2) Standardizes columns and parses dates.
3) Builds a transaction-level table (line items).
4) Aggregates to product-month level.
5) Completes a full product-month panel (fills missing months with zeros).
6) Creates an 80/20 chronological split (default: train through 2019-04, test from 2019-05).
7) Engineers no-leakage rolling statistics over ordered demand (using lagged demand).

Outputs (default folder: /mnt/data)
-----------------------------------
- sales_transactions_2017_2019.csv.gz
- item_month_agg_2017_2019.csv
- train_item_month_2017_2019_cutoff_2019-04.csv
- test_item_month_2017_2019_start_2019-05.csv
- item_month_agg_2017_2019_with_roll.csv
- train_item_month_2017_2019_cutoff_2019-04_with_roll.csv
- test_item_month_2017_2019_start_2019-05_with_roll.csv

Usage
-----
python preprocess_sales.py \
  --sales2017 "/mnt/data/sales 2017.xlsx" \
  --sales2018 "/mnt/data/sales 2018.xlsx" \
  --sales2019 "/mnt/data/sales 2019.xlsx" \
  --outdir "/mnt/data" \
  --train_end "2019-04-01" \
  --test_start "2019-05-01"

Notes
-----
- Target semantics: ordered_qty is treated as demand proxy.
- Date-of-record: uses `shipped` date to assign months, because order-date is not reliably present.
- Leakage control: rolling stats are computed over ordered_lag1 (i.e., history up to t-1).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LOG = logging.getLogger("preprocess_sales")


# -----------------------------
# Helpers
# -----------------------------
def _std_colname(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("\n", " ").replace("\t", " ")
    s = "_".join(s.split())
    s = s.replace("__", "_")
    return s


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _choose_sheet(xl: pd.ExcelFile, preferred: Optional[str] = None) -> str:
    if preferred and preferred in xl.sheet_names:
        return preferred
    # Otherwise choose the first sheet
    return xl.sheet_names[0]


def _build_rename_map(df: pd.DataFrame) -> Dict[str, str]:
    """Build a mapping of column names to standardized names."""
    rename_map: Dict[str, str] = {}

    # Product ID
    for c in ["product_id", "item", "item_no", "item_number", "product", "sku"]:
        if c in df.columns:
            rename_map[c] = "product_id"
            break

    # Customer identifier/name
    for c in ["customer", "customer_name", "customer_num", "name", "sold_to", "bill_to"]:
        if c in df.columns:
            rename_map[c] = "customer"
            break

    # Shipped date
    for c in ["shipped", "ship_date", "shipped_date", "date_shipped"]:
        if c in df.columns:
            rename_map[c] = "shipped"
            break

    # Ordered quantity
    for c in ["ordered_qty", "qty_ordered", "ordered", "order_qty", "qty_ord"]:
        if c in df.columns:
            rename_map[c] = "ordered_qty"
            break

    # Shipped quantity
    for c in ["ship_qty", "qty_shipped", "shipped_qty", "shipped_quantity", "qty_ship"]:
        if c in df.columns:
            rename_map[c] = "ship_qty"
            break

    return rename_map


def read_year_sales(
    filepath: str | Path,
    sheet_preference: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a yearly sales Excel file and return a standardized transaction-level dataframe.

    Expected core fields (post-standardization):
      - product_id (str)
      - customer (str)
      - shipped_dt (datetime64)
      - ordered_qty (float)
      - ship_qty (float)
    """
    filepath = Path(filepath)
    LOG.info("Reading %s", filepath)

    xl = pd.ExcelFile(filepath)
    sheet = _choose_sheet(xl, sheet_preference)
    df = pd.read_excel(filepath, sheet_name=sheet)

    # Standardize column names
    df.columns = [_std_colname(c) for c in df.columns]

    # Heuristic mapping (covers the 2017–2019 sales files observed in this project)
    rename_map = _build_rename_map(df)
    df = df.rename(columns=rename_map)

    required = ["product_id", "customer", "shipped", "ordered_qty", "ship_qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {filepath.name}. "
            f"Available columns: {list(df.columns)}"
        )

    # Parse dates and quantities
    df["shipped_dt"] = pd.to_datetime(df["shipped"], errors="coerce")
    df = df.dropna(subset=["shipped_dt"]).copy()

    df["ordered_qty"] = _coerce_numeric(df["ordered_qty"])
    df["ship_qty"] = _coerce_numeric(df["ship_qty"])

    # Keep only required columns to minimize RAM
    out = df[["product_id", "customer", "shipped_dt", "ordered_qty", "ship_qty"]].copy()

    # Normalize types
    out["product_id"] = out["product_id"].astype(str).str.strip()
    out["customer"] = out["customer"].astype(str).str.strip()

    return out


def build_transactions(
    sales_paths: List[Tuple[str | Path, Optional[str]]],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p, sheet in sales_paths:
        frames.append(read_year_sales(p, sheet_preference=sheet))

    tx = pd.concat(frames, axis=0, ignore_index=True)

    # Month bucket using shipped_dt
    tx["month"] = tx["shipped_dt"].dt.to_period("M").dt.to_timestamp()

    # Light RAM optimization
    tx["product_id"] = tx["product_id"].astype("category")
    tx["customer"] = tx["customer"].astype("category")

    return tx


def aggregate_item_month(tx: pd.DataFrame) -> pd.DataFrame:
    # Ensure minimal columns
    use = tx[["product_id", "customer", "month", "ordered_qty", "ship_qty"]].copy()
    use["n_lines"] = 1

    g = use.groupby(["product_id", "month"], observed=True)
    agg = g.agg(
        ordered_qty=("ordered_qty", "sum"),
        ship_qty=("ship_qty", "sum"),
        n_lines=("n_lines", "sum"),
        n_customers=("customer", "nunique"),
    ).reset_index()

    # Sort for later groupby operations
    agg = agg.sort_values(["product_id", "month"]).reset_index(drop=True)
    return agg


def complete_panel(item_month: pd.DataFrame) -> pd.DataFrame:
    """
    Create a full product-month panel from min to max month and fill missing with zeros.
    """
    item_month = item_month.copy()
    item_month["product_id"] = item_month["product_id"].astype("category")

    min_m = item_month["month"].min()
    max_m = item_month["month"].max()
    months = pd.date_range(min_m, max_m, freq="MS")

    products = item_month["product_id"].cat.categories

    full_index = pd.MultiIndex.from_product(
        [products, months],
        names=["product_id", "month"],
    )

    full = pd.DataFrame(index=full_index).reset_index()

    merged = full.merge(
        item_month,
        on=["product_id", "month"],
        how="left",
        copy=False,
    )

    for c in ["ordered_qty", "ship_qty", "n_lines", "n_customers"]:
        merged[c] = merged[c].fillna(0.0)

    merged = merged.sort_values(["product_id", "month"]).reset_index(drop=True)
    return merged


def add_ordered_rolling_features(
    panel: pd.DataFrame,
    windows: Iterable[int] = (3, 6, 12),
) -> pd.DataFrame:
    """
    Adds lag1 of ordered_qty and rolling mean/std windows over lag1 (history up to t-1).
    """
    df = panel.copy()
    df = df.sort_values(["product_id", "month"]).reset_index(drop=True)

    # Lag1
    df["ordered_lag1"] = df.groupby("product_id", observed=True)["ordered_qty"].shift(1).fillna(0.0)

    # Rolling stats computed on ordered_lag1 (no leakage)
    gb = df.groupby("product_id", observed=True)["ordered_lag1"]

    for w in windows:
        df[f"ordered_roll_mean_{w}"] = gb.transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"ordered_roll_std_{w}"] = gb.transform(lambda s: s.rolling(w, min_periods=1).std(ddof=0))

    # Ensure numeric and fill any remaining NaNs
    roll_cols = ["ordered_lag1"] + [f"ordered_roll_mean_{w}" for w in windows] + [f"ordered_roll_std_{w}" for w in windows]
    df[roll_cols] = df[roll_cols].fillna(0.0)

    return df


def split_train_test(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    train = df[df["month"] <= train_end_ts].copy()
    test = df[df["month"] >= test_start_ts].copy()

    return train, test


def compute_basic_diagnostics(tx: pd.DataFrame, item_month: pd.DataFrame) -> Dict[str, float]:
    # How often ordered != shipped
    tx_diff = (tx["ordered_qty"] != tx["ship_qty"]).mean()
    im_diff = (item_month["ordered_qty"] != item_month["ship_qty"]).mean()

    # Correlation at item-month level (where there is any activity)
    corr = np.nan
    try:
        corr = float(item_month[["ordered_qty", "ship_qty"]].corr().iloc[0, 1])
    except Exception:
        pass

    return {
        "tx_share_ordered_ne_ship": float(tx_diff),
        "item_month_share_ordered_ne_ship": float(im_diff),
        "item_month_corr_ordered_ship": float(corr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales2017", type=str, required=True)
    parser.add_argument("--sales2018", type=str, required=True)
    parser.add_argument("--sales2019", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--sheet2017", type=str, default=None)
    parser.add_argument("--sheet2018", type=str, default=None)
    parser.add_argument("--sheet2019", type=str, default=None)
    parser.add_argument("--train_end", type=str, default="2019-04-01")
    parser.add_argument("--test_start", type=str, default="2019-05-01")
    parser.add_argument("--windows", type=int, nargs="*", default=[3, 6, 12])
    parser.add_argument("--loglevel", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Known sheet names (optional); you can override via CLI args
    sales_paths = [
        (args.sales2017, args.sheet2017),
        (args.sales2018, args.sheet2018),
        (args.sales2019, args.sheet2019),
    ]

    tx = build_transactions(sales_paths)
    LOG.info("Transactions shape: %s", tx.shape)

    item_month = aggregate_item_month(tx)
    LOG.info("Item-month agg shape (observed only): %s", item_month.shape)

    diag = compute_basic_diagnostics(tx, item_month)
    LOG.info("Diagnostics: %s", diag)

    # Save transactions and base aggregation
    tx_out = outdir / "sales_transactions_2017_2019.csv.gz"
    tx.to_csv(tx_out, index=False, compression="gzip")
    LOG.info("Wrote %s", tx_out)

    im_out = outdir / "item_month_agg_2017_2019.csv"
    item_month.to_csv(im_out, index=False)
    LOG.info("Wrote %s", im_out)

    # Split on observed-only aggregation (optional, kept to match earlier artifacts)
    train_base, test_base = split_train_test(item_month, args.train_end, args.test_start)

    train_base_out = outdir / f"train_item_month_2017_2019_cutoff_{pd.Timestamp(args.train_end).strftime('%Y-%m')}.csv"
    test_base_out = outdir / f"test_item_month_2017_2019_start_{pd.Timestamp(args.test_start).strftime('%Y-%m')}.csv"
    train_base.to_csv(train_base_out, index=False)
    test_base.to_csv(test_base_out, index=False)
    LOG.info("Wrote %s", train_base_out)
    LOG.info("Wrote %s", test_base_out)

    # Panel completion and rolling features
    panel = complete_panel(item_month)
    LOG.info("Full panel shape (completed grid): %s", panel.shape)

    panel_with_roll = add_ordered_rolling_features(panel, windows=args.windows)

    panel_out = outdir / "item_month_agg_2017_2019_with_roll.csv"
    panel_with_roll.to_csv(panel_out, index=False)
    LOG.info("Wrote %s", panel_out)

    train_roll, test_roll = split_train_test(panel_with_roll, args.train_end, args.test_start)

    train_roll_out = outdir / f"train_item_month_2017_2019_cutoff_{pd.Timestamp(args.train_end).strftime('%Y-%m')}_with_roll.csv"
    test_roll_out = outdir / f"test_item_month_2017_2019_start_{pd.Timestamp(args.test_start).strftime('%Y-%m')}_with_roll.csv"
    train_roll.to_csv(train_roll_out, index=False)
    test_roll.to_csv(test_roll_out, index=False)
    LOG.info("Wrote %s", train_roll_out)
    LOG.info("Wrote %s", test_roll_out)

    LOG.info("Done.")


if __name__ == "__main__":
    main()
