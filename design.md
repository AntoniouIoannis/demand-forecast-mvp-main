# Design Document - Demand Forecasting System

**Project:** Demand Forecasting using Economic Indicators  
**Created:** 2025-11-08  
**Status:** Phase 1 Complete, Phase 2 Design  
**Version:** 1.0

---

## Executive Summary

This document outlines the technical design for a demand forecasting system that predicts retail sales using historical economic indicators. The system follows a time series forecasting approach with supervised machine learning, respecting temporal constraints to prevent data leakage. The MVP focuses on 1-month ahead forecasting using lag-based features.

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEMAND FORECASTING SYSTEM                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Data Sources   │────▶│  Data Pipeline   │────▶│   Feature    │
│  (FRED CSV)     │     │  (ETL Process)   │     │  Engineering │
└─────────────────┘     └──────────────────┘     └──────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Predictions &  │◀────│  ML Models       │◀────│  Training    │
│  Visualizations │     │  (Inference)     │     │  Pipeline    │
└─────────────────┘     └──────────────────┘     └──────────────┘
                                                         │
                                                         ▼
                                                  ┌──────────────┐
                                                  │  Validation  │
                                                  │  Framework   │
                                                  └──────────────┘
```

### Component Breakdown

#### 1. Data Layer
- **Input:** CSV files from FRED (Federal Reserve Economic Data)
- **Processing:** pandas DataFrames for in-memory manipulation
- **Storage:** Local file system (CSV format)
- **Output:** Merged, preprocessed, feature-engineered dataset

#### 2. Feature Engineering Layer
- **Lag Features:** Time-shifted variables (1-12 months)
- **Rolling Statistics:** Moving averages, exponential moving averages
- **Time Features:** Month, quarter, year, cyclical encodings
- **Domain Features:** Growth rates, economic stress indicators

#### 3. Model Layer
- **Baseline:** Simple lag-based linear regression
- **Advanced:** Ensemble tree-based models (XGBoost, LightGBM, Random Forest)
- **Future:** Deep learning (LSTM, Transformer) for complex patterns

#### 4. Evaluation Layer
- **Metrics:** MAE, RMSE, MAPE, R²
- **Validation:** Time series cross-validation (expanding window)
- **Visualization:** Actual vs. predicted plots, residual analysis

---

## Data Flow Diagram

### Phase 1: Data Preparation (IMPLEMENTED)

```
┌──────────────┐
│ retail_sales │─┐
└──────────────┘ │
┌──────────────┐ │
│consumer_credit│├──▶ ┌─────────────┐      ┌──────────────┐
└──────────────┘ │    │   Load &    │      │   Rename &   │
┌──────────────┐ │    │   Parse     │─────▶│   Align      │
│    PCE       │─┤    │   Dates     │      │   Dates      │
└──────────────┘ │    └─────────────┘      └──────────────┘
┌──────────────┐ │                                │
│ unemployment │─┤                                ▼
└──────────────┘ │                         ┌──────────────┐
┌──────────────┐ │                         │    Merge     │
│     GDP      │─┘                         │  (Inner      │
└──────────────┘                           │   Join)      │
                                           └──────────────┘
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │  Resample    │
                                           │  GDP to      │
                                           │  Monthly     │
                                           └──────────────┘
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │   Merged     │
                                           │   Dataset    │
                                           └──────────────┘
```

### Phase 2: Feature Engineering (IMPLEMENTED)

```
┌───────────────────┐
│  Merged Dataset   │
│  (Date Index)     │
└───────────────────┘
         │
         ├──────────────────────────────────────────────┐
         ▼                                              ▼
  ┌─────────────┐                              ┌─────────────┐
  │   Create    │                              │   Create    │
  │   Retail    │                              │  Economic   │
  │   Sales     │                              │  Indicator  │
  │   Lags      │                              │    Lags     │
  │  (1-12 mo)  │                              │  (1-6 mo)   │
  └─────────────┘                              └─────────────┘
         │                                              │
         └──────────────┬───────────────────────────────┘
                        ▼
                 ┌─────────────┐
                 │  Drop NaN   │
                 │  (from lags)│
                 └─────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │ Correlation │
                 │  Analysis   │
                 └─────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │   Feature   │
                 │  Selection  │
                 └─────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Training   │
                 │   Ready     │
                 │  Dataset    │
                 └─────────────┘
```

### Phase 3: Model Training & Prediction (DESIGN PHASE)

```
┌─────────────────┐
│  Training       │
│  Dataset        │
└─────────────────┘
         │
         ├──────────────────────────────────────────────┐
         ▼                                              ▼
  ┌─────────────┐                              ┌─────────────┐
  │  Temporal   │                              │   Feature   │
  │   Split     │                              │   Scaling   │
  │ (No Shuffle)│                              │ (Optional)  │
  └─────────────┘                              └─────────────┘
         │                                              │
         ├──────────────┬───────────────────────────────┘
         │              │
         ▼              ▼
  ┌──────────┐   ┌──────────┐
  │  Train   │   │   Valid  │
  │   Set    │   │   Set    │
  └──────────┘   └──────────┘
         │              │
         ▼              │
  ┌──────────┐         │
  │  Train   │         │
  │  Models  │         │
  │ (XGBoost,│         │
  │ LightGBM,│         │
  │   RF)    │         │
  └──────────┘         │
         │              │
         └──────┬───────┘
                ▼
         ┌──────────┐
         │ Evaluate │
         │  Models  │
         └──────────┘
                │
                ├──────────────┬──────────────┐
                ▼              ▼              ▼
         ┌──────────┐   ┌──────────┐  ┌──────────┐
         │   MAE    │   │  RMSE    │  │    R²    │
         │  MAPE    │   │ Residual │  │  Feature │
         │          │   │ Analysis │  │ Importance│
         └──────────┘   └──────────┘  └──────────┘
                │
                ▼
         ┌──────────┐
         │  Select  │
         │   Best   │
         │  Model   │
         └──────────┘
                │
                ▼
         ┌──────────┐
         │ Generate │
         │Predictions│
         └──────────┘
```

---

## Data Models

### Input Data Schema

#### Economic Indicator Schema (All Datasets)
```python
{
    "observation_date": datetime64,  # Primary temporal key
    "value": float64                 # Indicator value (varies by source)
}
```

**Datasets:**
- `retail_sales.csv`: Monthly retail sales (millions USD)
- `consumer_credit.csv`: Total consumer credit (billions USD)
- `personal_consumption.csv`: PCE (billions USD)
- `unemployment_rate.csv`: Unemployment rate (percentage)
- `gdp.csv`: GDP (billions USD, quarterly)

### Merged Dataset Schema

```python
merged_data = {
    "date": DatetimeIndex,                # Primary index
    "retail_sales": float64,              # Target variable
    "consumer_credit": float64,           # Exogenous feature
    "personal_consumption": float64,      # Exogenous feature
    "unemployment_rate": float64,         # Exogenous feature
    "gdp": float64                        # Exogenous feature (resampled to monthly)
}
```

**Constraints:**
- Date must be first day of month (monthly start frequency: 'MS')
- All values must be numeric (no strings, no categorical)
- Inner join ensures all indicators available for each date

### Feature-Engineered Dataset Schema

```python
feature_data = {
    # Target
    "retail_sales": float64,
    
    # Retail Sales Lags (Autocorrelation)
    "retail_sales_lag_1": float64,
    "retail_sales_lag_2": float64,
    # ... up to lag_12
    
    # Consumer Credit Lags
    "consumer_credit_lag_1": float64,
    "consumer_credit_lag_2": float64,
    # ... up to lag_6
    
    # Personal Consumption Lags (starts at lag_2 due to reporting delay)
    "pce_lag_2": float64,
    "pce_lag_3": float64,
    # ... up to lag_6
    
    # Unemployment Rate Lags
    "unemployment_lag_1": float64,
    "unemployment_lag_2": float64,
    # ... up to lag_6
    
    # GDP Lags (quarterly in monthly format)
    "gdp_lag_3m": float64,   # 1 quarter
    "gdp_lag_6m": float64,   # 2 quarters
    "gdp_lag_9m": float64,   # 3 quarters
    "gdp_lag_12m": float64,  # 4 quarters
    
    # Future: Rolling Statistics
    # "retail_sales_ma_3": float64,
    # "retail_sales_ema_6": float64,
    # "retail_sales_std_3": float64,
    
    # Future: Time Features
    # "month": int64,
    # "quarter": int64,
    # "year": int64,
    # "month_sin": float64,
    # "month_cos": float64,
    
    # Future: Growth Rates
    # "retail_sales_pct_change_1m": float64,
    # "consumer_credit_growth_rate": float64,
}
```

**Constraints:**
- All lag features must respect temporal causality (no data leakage)
- Rows with NaN (from lagging) are dropped before modeling
- Feature count: ~45 features after full engineering

---

## Interfaces

### Forecasting Function Interface (Conceptual - Phase 2)

```python
def forecast_retail_sales(
    historical_data: pd.DataFrame,
    forecast_horizon: int = 1,
    model_type: str = "xgboost",
    include_confidence_interval: bool = False
) -> pd.DataFrame:
    """
    Forecast retail sales for future months.
    
    Parameters
    ----------
    historical_data : pd.DataFrame
        DataFrame with columns ['retail_sales', 'consumer_credit', 
        'personal_consumption', 'unemployment_rate', 'gdp']
        and DatetimeIndex.
    
    forecast_horizon : int, default=1
        Number of months to forecast ahead. Default is 1 (next month).
    
    model_type : str, default="xgboost"
        Model to use: "xgboost", "lightgbm", "random_forest", "baseline"
    
    include_confidence_interval : bool, default=False
        Whether to include prediction intervals (requires quantile regression).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date', 'predicted_sales', 'lower_bound', 
        'upper_bound'] for each forecasted month.
    
    Raises
    ------
    ValueError
        If historical_data has insufficient history (< 12 months).
    ValueError
        If required columns are missing.
    
    Notes
    -----
    - Only uses features available at forecast time (respects lags).
    - For multi-step forecasting (horizon > 1), uses recursive prediction.
    - Model must be pre-trained; loads from `models/{model_type}_model.pkl`.
    """
    pass
```

### Feature Engineering Interface (Conceptual - Phase 2)

```python
def engineer_features(
    df: pd.DataFrame,
    target_col: str = "retail_sales",
    lag_periods: dict = None,
    include_time_features: bool = True,
    include_rolling_stats: bool = True
) -> pd.DataFrame:
    """
    Engineer lag-based and time-based features for forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with economic indicators and DatetimeIndex.
    
    target_col : str, default="retail_sales"
        Name of the target variable column.
    
    lag_periods : dict, optional
        Dictionary specifying lag periods for each column.
        Example: {"retail_sales": [1, 2, 3, 6, 12],
                  "consumer_credit": [1, 2, 3]}
        If None, uses default lag configuration.
    
    include_time_features : bool, default=True
        Whether to create month, quarter, year features.
    
    include_rolling_stats : bool, default=True
        Whether to compute moving averages and rolling std dev.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus engineered features.
        Rows with NaN (from lagging) are NOT dropped.
    
    Notes
    -----
    - Respects temporal constraints: all features use only past data.
    - Caller is responsible for handling NaN values.
    """
    pass
```

---

## Implementation Details

### Technology Stack

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Programming Language | Python | 3.8+ | Industry standard for data science |
| Data Manipulation | pandas | 2.0+ | Efficient time series operations |
| Numerical Computing | NumPy | 1.24+ | Fast array operations |
| Visualization | matplotlib, seaborn | 3.7+, 0.12+ | Rich plotting capabilities |
| Machine Learning | scikit-learn | 1.3+ | Standard ML library |
| Gradient Boosting | XGBoost, LightGBM | 1.7+, 4.0+ | State-of-the-art tabular data models |
| Time Series | statsmodels | 0.14+ | ARIMA, seasonal decomposition |
| Development Environment | Jupyter Notebook | 1.0+ | Interactive exploratory analysis |
| Version Control | Git | - | Code versioning |
| Dependency Management | pip, requirements.txt | - | Python package management |

### Directory Structure

```
Demand Forecast/
│
├── .github/
│   └── copilot-instructions.md      # Specification-driven workflow guide
│
├── data/
│   ├── retail_sales.csv              # Raw: Retail sales time series
│   ├── consumer_credit.csv           # Raw: Consumer credit time series
│   ├── personal_consumption.csv      # Raw: PCE time series
│   ├── unemployment_rate.csv         # Raw: Unemployment rate time series
│   ├── gdp.csv                       # Raw: GDP time series (quarterly)
│   └── merged_economic_data.csv      # Processed: Merged dataset
│
├── documents/
│   └── mvp_model_ntb1.ipynb          # Phase 1: Data exploration & feature eng.
│
├── models/                           # (Future) Trained model artifacts
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   └── model_metadata.json
│
├── notebooks/                        # (Future) Additional analysis notebooks
│
├── src/                              # (Future) Production code modules
│   ├── __init__.py
│   ├── data_loader.py                # Data loading utilities
│   ├── feature_engineering.py        # Feature creation functions
│   ├── models.py                     # Model training and inference
│   ├── evaluation.py                 # Metrics and validation
│   └── utils.py                      # Helper functions
│
├── tests/                            # (Future) Unit and integration tests
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   └── test_models.py
│
├── requirements.txt                  # Python dependencies
├── requirements.md                   # Requirements documentation (EARS)
├── design.md                         # This document
├── tasks.md                          # (To be created) Implementation tasks
└── README.md                         # (Future) Project overview
```

### Temporal Constraint Management

**Critical Design Principle:** No Data Leakage

To ensure realistic forecasting, the system enforces temporal constraints:

#### Data Availability Matrix (at Forecast Time)

| Indicator | Available at Time t? | Usable Lag | Rationale |
|-----------|---------------------|------------|-----------|
| Retail Sales (Target) | ❌ NO | t-1 | Unknown - this is what we predict |
| Consumer Credit | ✅ YES | t-1 | Published ~5 days after month end |
| Personal Consumption | ⚠️ DELAYED | t-2 | Published ~1 month after month end |
| Unemployment Rate | ✅ YES | t-1 | Published ~1 week after month end |
| GDP | ⚠️ DELAYED | t-1 to t-3 | Quarterly data with ~1 month lag |

#### Implementation: Lag Feature Creation

```python
# Example: Creating time-respecting lag features

def create_lag_features(df, column, lags):
    """
    Create lag features that respect temporal constraints.
    
    Note: .shift(n) moves data DOWN by n rows, making it "past" data.
    """
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

# For retail sales (autocorrelation)
df = create_lag_features(df, 'retail_sales', lags=range(1, 13))

# For consumer credit (reporting lag ~5 days, safe to use lag_1)
df = create_lag_features(df, 'consumer_credit', lags=range(1, 7))

# For PCE (reporting lag ~1 month, start at lag_2)
df = create_lag_features(df, 'personal_consumption', lags=range(2, 7))
```

**Validation:** Before model training, verify no data leakage:
```python
# Ensure lag features are truly from the past
assert df['retail_sales_lag_1'].shift(-1).equals(df['retail_sales'])
```

---

## Model Design

### Baseline Model (Phase 2 - First Implementation)

**Type:** Simple Persistence Model (Naïve Forecast)

**Approach:**
- Forecast: `retail_sales(t) = retail_sales(t-1)`
- Purpose: Establish performance floor
- Expected R²: ~0.90 (due to high autocorrelation)

### Advanced Models (Phase 2 - Primary Implementation)

#### Model 1: XGBoost Regressor

**Rationale:**
- Excellent for tabular data with complex interactions
- Handles multicollinearity well
- Built-in regularization prevents overfitting
- Fast training and inference

**Hyperparameters (Initial):**
```python
xgb_params = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'early_stopping_rounds': 50
}
```

#### Model 2: LightGBM Regressor

**Rationale:**
- Faster than XGBoost for large datasets
- Better handling of categorical features (if added later)
- Lower memory usage

**Hyperparameters (Initial):**
```python
lgbm_params = {
    'n_estimators': 500,
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'objective': 'regression',
    'metric': 'rmse'
}
```

#### Model 3: Random Forest Regressor

**Rationale:**
- Robust to outliers
- Minimal hyperparameter tuning required
- Good baseline for comparison

**Hyperparameters (Initial):**
```python
rf_params = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42
}
```

### Model Selection Criteria

**Primary Metric:** RMSE (aligns with business cost of forecast error)

**Secondary Metrics:**
- R² (proportion of variance explained)
- MAPE (percentage error for interpretability)
- MAE (absolute error magnitude)

**Selection Process:**
1. Train all models on training set
2. Evaluate on validation set using time series cross-validation
3. Select model with lowest RMSE
4. Validate selected model on held-out test set
5. Perform residual analysis for bias detection

---

## Error Handling Strategy

### Error Categories & Responses

| Error Type | Example | System Response | User Action |
|------------|---------|-----------------|-------------|
| **File Not Found** | `data/retail_sales.csv` missing | Raise `FileNotFoundError` with expected path | Check file location, verify path |
| **Date Parsing Error** | Invalid date format in CSV | Raise `ValueError` with problematic row | Correct date format to YYYY-MM-DD |
| **Missing Values** | NaN in input data | Log warning, report count and location | Review data quality, impute or exclude |
| **Merge Failure** | No overlapping dates | Raise `ValueError` with date ranges | Check date alignment across datasets |
| **Insufficient Data** | < 12 months for lagging | Raise `ValueError` with required length | Provide more historical data |
| **Model Load Error** | Trained model file missing | Raise `FileNotFoundError` | Train model first before prediction |
| **Feature Mismatch** | Prediction data missing features | Raise `KeyError` with missing features | Ensure consistent feature engineering |

### Error Handling Implementation

```python
# Example: Data loading with error handling
def load_economic_data(file_path, indicator_name):
    """
    Load economic indicator data with comprehensive error handling.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['observation_date'])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data file not found: {file_path}. "
            f"Please ensure {indicator_name} data is downloaded and placed in the correct directory."
        )
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Failed to parse {indicator_name} data. "
            f"Check CSV format and date column. Error: {str(e)}"
        )
    
    # Validate data structure
    if df.shape[1] != 2:
        raise ValueError(
            f"{indicator_name} data must have exactly 2 columns: date and value. "
            f"Found {df.shape[1]} columns."
        )
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️  Warning: {indicator_name} has {missing_count} missing values.")
    
    return df
```

---

## Testing Strategy

### Unit Testing (Future Phase 3)

**Coverage Target:** >80% for production modules

**Test Categories:**

1. **Data Loading Tests**
   - Valid CSV parsing
   - Error handling for missing files
   - Date format validation

2. **Feature Engineering Tests**
   - Lag feature correctness (no data leakage)
   - Rolling statistics computation
   - NaN handling
   - Time feature encoding

3. **Model Tests**
   - Training without errors
   - Prediction output shape and dtype
   - Model persistence (save/load)

4. **Evaluation Tests**
   - Metric calculation accuracy
   - Time series split logic
   - Residual computation

### Integration Testing (Future Phase 3)

**Scenarios:**

1. **End-to-End Pipeline**
   - Load raw data → Engineer features → Train model → Generate prediction
   - Expected outcome: Prediction within reasonable range

2. **Temporal Integrity**
   - Verify no future data leaks into training
   - Expected outcome: Features only use past data

3. **Model Reproducibility**
   - Same data + same seed = same results
   - Expected outcome: Deterministic outputs

### Validation Testing (Phase 2)

**Approach:** Time Series Cross-Validation (Expanding Window)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f"Fold {fold+1} RMSE: {rmse:.2f}")
```

**Rationale:**
- Respects temporal ordering (no random shuffling)
- Expanding window simulates real-world forecasting (more data over time)
- Multiple folds reduce variance in performance estimates

---

## Performance Considerations

### Computational Complexity

| Operation | Time Complexity | Space Complexity | Optimization |
|-----------|----------------|------------------|--------------|
| Data Loading | O(n) | O(n) | Use `usecols` to load only needed columns |
| Lag Feature Creation | O(n * k) | O(n * k) | Vectorized pandas operations (already fast) |
| Correlation Matrix | O(n * m²) | O(m²) | Parallelizable with NumPy |
| Model Training (XGBoost) | O(n * m * log(n) * trees) | O(n * m) | Use GPU acceleration if available |
| Prediction | O(m * trees) | O(m) | Batch predictions instead of row-by-row |

Where:
- n = number of observations (rows)
- m = number of features (columns)
- k = number of lag periods
- trees = number of trees in ensemble

### Expected Performance

**Dataset Size:** ~300-500 monthly observations per indicator

**Hardware:** Standard laptop (8GB RAM, 4-core CPU)

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| Data loading (all 5 datasets) | < 2 seconds | CSV parsing is fast for small files |
| Feature engineering (45 features) | < 5 seconds | Vectorized pandas operations |
| Correlation analysis | < 1 second | Matrix operations on small data |
| Model training (XGBoost) | 10-30 seconds | Depends on hyperparameters |
| Prediction (single forecast) | < 1 second | Tree traversal is very fast |
| Full notebook execution | < 2 minutes | End-to-end analysis |

### Scalability Considerations

**Current Design (MVP):**
- ✅ Handles ~500 observations efficiently
- ✅ Works with 5-10 indicators
- ✅ Generates 50-100 features

**Future Scaling (if needed):**
- 🔄 For 10,000+ observations: Consider chunked processing
- 🔄 For 100+ indicators: Use dimensionality reduction (PCA, feature selection)
- 🔄 For real-time predictions: Pre-compute features, cache models
- 🔄 For distributed training: Use Dask, Ray, or Spark

---

## Security Considerations

### Data Privacy

**Current Status:** No sensitive data (public economic indicators)

**Future Considerations:**
- If incorporating proprietary company data, implement data anonymization
- Ensure compliance with data retention policies
- Encrypt sensitive data at rest and in transit

### Model Security

**Risks:**
- Model theft (trained models are valuable IP)
- Adversarial attacks (manipulated inputs to generate bad forecasts)

**Mitigations:**
- Store trained models in secure locations with access controls
- Implement input validation to detect anomalous feature values
- Log all predictions for audit trails

### Code Security

**Best Practices:**
- No hard-coded credentials or API keys (use environment variables)
- Validate all file paths to prevent directory traversal attacks
- Sanitize any user inputs if system becomes interactive

---

## Deployment Strategy (Future Phase 3)

### Option 1: Batch Forecasting (Scheduled Jobs)

**Architecture:**
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Scheduler  │────▶│   Forecast   │────▶│   Database   │
│   (Cron)     │     │   Script     │     │   (Results)  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Email/Alert │
                     │  to Analysts │
                     └──────────────┘
```

**Use Case:** Monthly forecast updates

**Implementation:**
- Python script triggered by cron job (Linux) or Task Scheduler (Windows)
- Loads latest data, generates forecast, saves to database
- Sends email notification with forecast summary

### Option 2: API Service (On-Demand Forecasting)

**Architecture:**
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client     │────▶│   FastAPI    │────▶│   Model      │
│  (Web App)   │     │   Endpoint   │     │   Inference  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Response   │
                     │   (JSON)     │
                     └──────────────┘
```

**Use Case:** Real-time forecast requests from business applications

**Implementation:**
- FastAPI or Flask REST API
- Endpoint: `POST /forecast` with JSON payload of historical data
- Returns forecast as JSON response
- Dockerized for easy deployment

### Option 3: Notebook Dashboard (Interactive Exploration)

**Architecture:**
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Voilà or   │────▶│   Jupyter    │────▶│   Widgets    │
│   Streamlit  │     │   Notebook   │     │  (Interactive)│
└──────────────┘     └──────────────┘     └──────────────┘
```

**Use Case:** Data scientist or analyst exploring forecasts interactively

**Implementation:**
- Convert notebook to web app using Voilà or Streamlit
- Add interactive widgets for horizon selection, model choice
- Host on internal server for team access

---

## Monitoring & Maintenance (Future Phase 3)

### Model Performance Monitoring

**Metrics to Track:**
- Forecast accuracy over time (rolling RMSE, MAPE)
- Residual patterns (detecting bias or drift)
- Feature importance changes (indicating structural changes)

**Alerting Thresholds:**
- If RMSE increases by >20% for 2 consecutive months: Trigger retraining
- If residuals show systematic bias (mean ≠ 0): Investigate and retrain
- If actuals fall outside 95% prediction interval >10% of the time: Model degradation

### Data Quality Monitoring

**Checks:**
- Missing data rates per indicator
- Outliers or anomalies (beyond 3 standard deviations)
- Delayed data updates (reporting lag exceeds expectations)

**Automation:**
- Scheduled data validation script runs before each forecast
- Logs data quality issues to monitoring dashboard
- Sends alerts if critical issues detected

### Retraining Strategy

**Frequency:** Quarterly (or when performance degrades)

**Process:**
1. Ingest latest historical data (last 3-6 months)
2. Re-engineer features with updated data
3. Retrain all models with expanded history
4. Validate on most recent months
5. Compare performance to previous model
6. Deploy if improvement observed (or if previous model degraded)

**Version Control:**
- Tag models with version and training date
- Keep last 3 model versions for rollback capability
- Document model lineage and performance metrics

---

## Technical Debt & Future Refactoring

### Current Technical Debt

**TD-1: Notebook-Only Implementation**
- **Issue:** All logic in Jupyter notebook, not modular or testable
- **Impact:** Difficult to reuse, maintain, and test
- **Remediation:** Extract functions to `src/` modules (Phase 3)

**TD-2: Manual Feature Engineering**
- **Issue:** Lag features created with repeated code
- **Impact:** Error-prone, hard to extend
- **Remediation:** Create `FeatureEngineer` class with configurable pipelines

**TD-3: No Model Versioning**
- **Issue:** Trained models not saved or versioned
- **Impact:** Can't reproduce results or rollback
- **Remediation:** Implement model registry (MLflow or custom)

**TD-4: Limited Error Handling**
- **Issue:** Basic try-except blocks, no comprehensive validation
- **Impact:** Silent failures or cryptic errors
- **Remediation:** Add custom exception classes and validation decorators

**TD-5: No Logging**
- **Issue:** Print statements instead of structured logging
- **Impact:** Difficult to debug production issues
- **Remediation:** Implement Python `logging` module with levels

### Planned Refactoring (Phase 3)

**Priority 1: Modularization**
- Move notebook code to reusable Python modules
- Implement class-based design for models and pipelines
- Create CLI interface for training and prediction

**Priority 2: Configuration Management**
- Externalize hyperparameters to YAML config files
- Implement environment-based settings (dev, staging, prod)
- Use `pydantic` for configuration validation

**Priority 3: Testing Infrastructure**
- Set up pytest framework
- Achieve 80% code coverage
- Implement CI/CD pipeline (GitHub Actions)

---

## Dependencies & Constraints

### Software Dependencies

**Core Libraries (from requirements.txt):**
```
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
matplotlib>=3.7.0      # Plotting
seaborn>=0.12.0        # Statistical visualization
scikit-learn>=1.3.0    # ML algorithms
xgboost>=1.7.0         # Gradient boosting
lightgbm>=4.0.0        # Gradient boosting (alternative)
statsmodels>=0.14.0    # Time series analysis
```

**Development Dependencies (future):**
```
pytest>=7.0.0          # Testing framework
black>=22.0.0          # Code formatting
flake8>=5.0.0          # Linting
mypy>=1.0.0            # Type checking
jupyter>=1.0.0         # Notebook environment
```

### External Constraints

**Data Source Dependency:**
- System relies on FRED data availability and format consistency
- If FRED changes schema or access method, pipeline will break
- **Mitigation:** Implement data validation layer, consider API access

**Temporal Constraints:**
- Forecasting horizon limited by data reporting lags
- Economic indicators not available in real-time
- **Mitigation:** Document assumptions clearly, design for lag-tolerant features

**Computational Constraints:**
- Training time increases linearly with data size and model complexity
- Inference must be fast (<1 second) for user-facing applications
- **Mitigation:** Profile code, optimize bottlenecks, consider caching

---

## Decision Records

### Decision 1 - 2025-11-08

**Decision:** Use lag-based features instead of ARIMA/SARIMA for MVP

**Context:** Need to choose between traditional time series models (ARIMA) and machine learning with lag features.

**Options:**
1. **ARIMA/SARIMA:** Classic time series approach, interpretable, handles seasonality
2. **ML with Lag Features:** Flexible, handles multiple exogenous variables, ensemble models
3. **Deep Learning (LSTM):** Captures complex patterns, but requires more data and tuning

**Rationale:**
- ARIMA is limited to univariate or few exogenous variables
- ML with lags can leverage all 5 economic indicators simultaneously
- Lag-based features are intuitive and interpretable (business-friendly)
- Ensemble models (XGBoost) are state-of-the-art for tabular data
- Deep learning overkill for current data size (~500 observations)

**Impact:**
- Need to carefully engineer lag features (more upfront work)
- Better scalability to additional indicators
- Easier to explain feature importance to stakeholders
- Potential for higher accuracy with proper tuning

**Review:** After Phase 2 modeling, compare performance to ARIMA baseline

---

### Decision 2 - 2025-11-08

**Decision:** Use inner join when merging datasets, accepting data loss

**Context:** Different indicators have different date ranges. Must decide on join strategy.

**Options:**
1. **Inner Join:** Keep only dates with all indicators available (lose some data)
2. **Outer Join:** Keep all dates, fill missing values (introduces imputation complexity)
3. **Left Join (on retail sales):** Prioritize target variable coverage

**Rationale:**
- Inner join ensures data quality (no imputed values for training)
- Missing a few early/late observations is acceptable (still ~300+ months)
- Simplifies feature engineering (no NaN handling for different indicators)
- Aligns with principle: only use data that's actually available

**Impact:**
- Lose ~10-20% of observations (varies by indicator date ranges)
- Simplifies pipeline (no imputation logic needed)
- Maintains data integrity for modeling
- May revisit if data coverage becomes insufficient

**Review:** Monitor final dataset size; if <200 months, reconsider outer join with imputation

---

### Decision 3 - 2025-11-08

**Decision:** Start PCE lags at lag_2 instead of lag_1

**Context:** Personal Consumption Expenditures (PCE) has a ~1 month reporting lag, meaning PCE for month M is not available until ~30 days into month M+1.

**Options:**
1. **Use lag_1 (t-1):** Assume data is available (unrealistic for real-time forecasting)
2. **Use lag_2 (t-2):** Account for reporting delay (more realistic)
3. **Dynamic lag:** Use lag_1 if forecasting well after the month, lag_2 otherwise

**Rationale:**
- Realistic forecasting scenario: If forecasting for December on Nov 15, PCE for October is available, but not November
- Lag_2 ensures no data leakage in production deployment
- Better to be conservative and slightly underutilize data than to overfit on unrealistic features
- Maintains temporal integrity of the forecasting system

**Impact:**
- Lose one month of PCE predictive power (use t-2 instead of t-1)
- Slight reduction in model accuracy (likely <1% given high retail sales autocorrelation)
- Increased confidence in production deployment (no surprises from unavailable data)
- Clearer documentation of assumptions for business stakeholders

**Review:** If production deployment shows PCE data is consistently available earlier, revert to lag_1

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **ARIMA** | AutoRegressive Integrated Moving Average - classical time series model |
| **Autocorrelation** | Correlation of a time series with its own lagged values |
| **EARS Notation** | Easy Approach to Requirements Syntax - structured requirement format |
| **Exogenous Variables** | External variables that influence the target but are not influenced by it |
| **Feature Engineering** | Creating new predictive variables from raw data |
| **FRED** | Federal Reserve Economic Data - public economic time series repository |
| **Lag Feature** | Time-shifted version of a variable (e.g., lag_1 = previous month's value) |
| **MAPE** | Mean Absolute Percentage Error - accuracy metric in percentage terms |
| **MAE** | Mean Absolute Error - average magnitude of forecast errors |
| **MVP** | Minimum Viable Product - simplest version that delivers value |
| **NaN** | Not a Number - missing value indicator in pandas |
| **PCE** | Personal Consumption Expenditures - measure of consumer spending |
| **R²** | Coefficient of determination - proportion of variance explained by model |
| **RMSE** | Root Mean Squared Error - standard deviation of forecast errors |
| **Time Series CV** | Cross-validation that respects temporal ordering of data |
| **XGBoost** | eXtreme Gradient Boosting - popular ensemble learning algorithm |

---

## Document Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-08 | AI Co-Developer | Initial design document created based on Phase 1 implementation analysis |

---

**Document Status:** Ready for Review  
**Next Steps:** Review with data scientist, refine model design, create tasks.md  
**Approver:** Lead Data Scientist

---

**Confidence Score for Phase 2 Implementation:** 78% (Medium-High)

**Rationale:**
- ✅ Clear understanding of data structure and feature engineering
- ✅ Strong theoretical foundation (lag-based forecasting is well-established)
- ✅ Phase 1 implementation provides solid base
- ⚠️ Model hyperparameter tuning may require iteration
- ⚠️ Actual forecast accuracy unknown until validation
- ⚠️ Production deployment challenges not yet fully scoped

**Recommendation:** Proceed with MVP model development (Phase 2) using PoC/MVP approach with clear success criteria.
