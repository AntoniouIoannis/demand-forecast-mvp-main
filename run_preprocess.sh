#!/usr/bin/env bash
set -euo pipefail

# Runner for preprocess_sales.py
# Adjust paths below if your files live elsewhere.

python preprocess_sales.py   --sales2017 "/mnt/data/sales 2017.xlsx"   --sales2018 "/mnt/data/sales 2018.xlsx"   --sales2019 "/mnt/data/sales 2019.xlsx"   --outdir "/mnt/data"   --train_end "2019-04-01"   --test_start "2019-05-01"
