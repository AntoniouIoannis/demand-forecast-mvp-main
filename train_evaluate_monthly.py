#!/usr/bin/env python3
"""
Train a demand model and produce MONTHLY metrics (timerange=month
on the test set.
Inputs: the *_with_roll.csv outputs from preprocess_sales.py
Outputs:
  - predictions_test.csv
  - metrics_by_month.csv
  - metrics_overall.json
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def add_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds only leakage-safe features using info up to t-1.
    Assumes df already has ordered_lag1 and ordered rolling
    stats from preprocess.
    """
    df = df.sort_values(["product_id", "month"]).reset_index(drop=True)
    g = df.groupby("product_id", sort=False)

    # Operational lags (safe)
    df["ship_lag1"] = g["ship_qty"].shift(1).fillna(0.0)
    df["customers_lag1"] = g["n_customers"].shift(1).fillna(0.0)
    df["lines_lag1"] = g["n_lines"].shift(1).fillna(0.0)

    # Fast rolling means on lag1 (safe)
    for w in (3, 6, 12):
        df[f"ship_roll_mean_{w}"] = (
            df.groupby("product_id", sort=False)["ship_lag1"]
              .rolling(w, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
        df[f"customers_roll_mean_{w}"] = (
            df.groupby("product_id", sort=False)["customers_lag1"]
              .rolling(w, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

    # Calendar seasonality
    mnum = df["month"].dt.month.astype(int)
    df["sin_moy"] = np.sin(2 * np.pi * mnum / 12.0)
    df["cos_moy"] = np.cos(2 * np.pi * mnum / 12.0)

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv, parse_dates=["month"])
    test = pd.read_csv(args.test_csv, parse_dates=["month"])

    # Concatenate so lags for early test months use prior months
    # (valid, since they are in the past)
    df = pd.concat([train, test], ignore_index=True)
    df["product_id"] = df["product_id"].astype(str)
    df = add_safe_features(df)

    # Re-split by month boundaries already embedded in the input files
    train_df = df[df["month"] <= train["month"].max()].copy()
    test_df = df[df["month"] >= test["month"].min()].copy()

    y_col = "ordered_qty"

    feature_cols = [
        "ordered_lag1",
        "ordered_roll_mean_3", "ordered_roll_mean_6", "ordered_roll_mean_12",
        "ordered_roll_std_3",  "ordered_roll_std_6",  "ordered_roll_std_12",
        "ship_lag1", "ship_roll_mean_3", "ship_roll_mean_6",
        "ship_roll_mean_12",
        "customers_lag1", "customers_roll_mean_3",
        "customers_roll_mean_6", "customers_roll_mean_12",
        "lines_lag1",
        "sin_moy", "cos_moy",
    ]

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[y_col].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[y_col].to_numpy(dtype=float)

    # Model: Poisson boosting preferred for non-negative demand
    model_name = None
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.08,
            max_depth=6,
            max_iter=200,
            random_state=42,
        )
        model.fit(X_train, y_train)
        model_name = "HistGradientBoostingRegressor(loss='poisson')"
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        model_name = "GradientBoostingRegressor"

    y_pred = np.maximum(model.predict(X_test), 0.0)

    # Predictions file
    pred = test_df[["month", "product_id", y_col]].copy()
    pred = pred.rename(columns={y_col: "y_true"})
    pred["y_pred"] = y_pred
    pred_out = outdir / "predictions_test.csv"
    pred.to_csv(pred_out, index=False)

    # Metrics by month
    rows = []
    for m, sub in pred.groupby("month"):
        yt = sub["y_true"].to_numpy(float)
        yp = sub["y_pred"].to_numpy(float)
        rows.append({
            "month": m,
            "n_rows": int(len(sub)),
            "actual_total": float(yt.sum()),
            "pred_total": float(yp.sum()),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": rmse(yt, yp),
            "smape": smape(yt, yp),
            "wape": wape(yt, yp),
            "bias_ratio": bias_ratio(yt, yp),
        })

    metrics_by_month = (
        pd.DataFrame(rows)
        .sort_values("month")
        .reset_index(drop=True)
    )
    mbm_out = outdir / "metrics_by_month.csv"
    metrics_by_month.to_csv(mbm_out, index=False)

    # Overall metrics
    overall = {
        "model": model_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_months": int(train_df["month"].nunique()),
        "test_months": int(test_df["month"].nunique()),
        "actual_total": float(y_test.sum()),
        "pred_total": float(y_pred.sum()),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": rmse(y_test, y_pred),
        "smape": smape(y_test, y_pred),
        "wape": wape(y_test, y_pred),
        "bias_ratio": bias_ratio(y_test, y_pred),
        "r2": float(r2_score(y_test, y_pred)),
    }
    overall_out = outdir / "metrics_overall.json"
    overall_out.write_text(json.dumps(overall, indent=2))

    print("Wrote:")
    print(f" - {pred_out}")
    print(f" - {mbm_out}")
    print(f" - {overall_out}")


if __name__ == "__main__":
    main()
