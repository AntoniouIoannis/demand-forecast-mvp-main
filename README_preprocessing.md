# Sales preprocessing package (2017–2019)

This package reads the 2017–2019 sales Excel files, standardizes them, aggregates to product-month,
completes the product-month grid, performs an 80/20 chronological split, and creates rolling statistics
(no leakage) over ordered demand.

## Files
- `preprocess_sales.py` — main pipeline script
- `requirements.txt` — minimal Python dependencies
- `run_preprocess.sh` — convenience runner (edit input paths if needed)
- `README_preprocessing.md` — this guide

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
bash run_preprocess.sh
```

## Outputs (written to `--outdir`, default `/mnt/data`)
- `sales_transactions_2017_2019.csv.gz`
- `item_month_agg_2017_2019.csv`
- `train_item_month_2017_2019_cutoff_2019-04.csv`
- `test_item_month_2017_2019_start_2019-05.csv`
- `item_month_agg_2017_2019_with_roll.csv`
- `train_item_month_2017_2019_cutoff_2019-04_with_roll.csv`
- `test_item_month_2017_2019_start_2019-05_with_roll.csv`
