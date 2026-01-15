#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_hat) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_hat)))

def smape(y_true, y_hat, eps=1e-9) -> float:
    denom = np.maximum((np.abs(y_true) + np.abs(y_hat)) / 2.0, eps)
    return float(np.mean(np.abs(y_hat - y_true) / denom))

def wape(y_true, y_hat, eps=1e-9) -> float:
    denom = max(float(np.sum(np.abs(y_true))), eps)
    return float(np.sum(np.abs(y_hat - y_true)) / denom)

def bias_ratio(y_true, y_hat, eps=1e-9) -> float:
    denom = max(float(np.sum(np.abs(y_true))), eps)
    return float(np.sum(y_hat - y_true) / denom)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="Path to predictions_test.csv")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--last_n_months", type=int, default=6, help="How many last months to include")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv, parse_dates=["month"])
    pred = pred.sort_values(["month", "product_id"]).reset_index(drop=True)

    # Determine last N months within the file (robust to any date range)
    months = sorted(pred["month"].unique())
    if len(months) < args.last_n_months:
        raise ValueError(f"predictions only has {len(months)} months; cannot take last {args.last_n_months}")
    last_months = months[-args.last_n_months:]

    pred_last = pred[pred["month"].isin(last_months)].copy()

    # Product-level summary metrics over last N months
    rows = []
    for pid, sub in pred_last.groupby("product_id", sort=False):
        yt = sub["y_true"].to_numpy(float)
        yp = sub["y_pred"].to_numpy(float)
        rows.append({
            "product_id": pid,
            "months_included": int(sub["month"].nunique()),
            "n_rows": int(len(sub)),
            "actual_total": float(yt.sum()),
            "pred_total": float(yp.sum()),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": rmse(yt, yp),
            "smape": smape(yt, yp),
            "wape": wape(yt, yp),
            "bias_ratio": bias_ratio(yt, yp),
        })

    report = pd.DataFrame(rows)

    # Optional: sort by business importance then error
    report = report.sort_values(["actual_total", "wape"], ascending=[False, True]).reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save (long format) product summary
    out_summary = outdir / f"metrics_by_product_last{args.last_n_months}m.csv"
    report.to_csv(out_summary, index=False)

    # Save (product x month) results table for inspection (wide format)
    pivot_true = pred_last.pivot_table(index="product_id", columns="month", values="y_true", aggfunc="sum", fill_value=0.0)
    pivot_pred = pred_last.pivot_table(index="product_id", columns="month", values="y_pred", aggfunc="sum", fill_value=0.0)

    # Flatten columns for CSV readability
    pivot_true.columns = [f"y_true_{pd.Timestamp(c).strftime('%Y-%m')}" for c in pivot_true.columns]
    pivot_pred.columns = [f"y_pred_{pd.Timestamp(c).strftime('%Y-%m')}" for c in pivot_pred.columns]

    wide = pivot_true.join(pivot_pred, how="outer").reset_index()
    out_wide = outdir / f"results_by_product_month_last{args.last_n_months}m.csv"
    wide.to_csv(out_wide, index=False)

    print("Wrote:")
    print(f" - {out_summary}")
    print(f" - {out_wide}")
    print(f"Months included: {[pd.Timestamp(m).strftime('%Y-%m') for m in last_months]}")

if __name__ == "__main__":
    main()
