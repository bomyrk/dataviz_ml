# Rain Tomorrow — Classification Project

## 1. Objective

Binary classification: predict whether it will rain tomorrow (≥ 1 mm) at an
Australian weather station, given that day's meteorological observations.

**Dataset:** [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
(~145 K rows, 22 features, 49 stations, Nov 2007 – Jun 2017).

---

## 2. EDA Summary

- Target is imbalanced: **77 % No / 23 % Yes** → PR-AUC used as primary metric.
- Missingness is MNAR (sensor degradation by station); long blocks of missing
  data indicate station outages.
- `Rainfall` and `Evaporation` are highly right-skewed (skew ≈ 10, 4).
- `RainToday` is the strongest single predictor of `RainTomorrow`
  (Cramér's V ≈ 0.55).
- Strong collinearity within feature groups (Temp↔Temp, Humidity↔Cloud↔Sunshine).

---

## 3. Methodology

### 3.1 Split (chronological)

| Set | Period | Rows |
|---|---|---|
| Train | Nov 2007 – Jun 2014 | ~101 000 |
| Val | Jul 2014 – Dec 2015 | ~27 000 |
| Test | Jan 2016 – Jun 2017 | ~27 000 |

No shuffling. `TimeSeriesSplit(n_splits=5)` for internal CV on train only.

### 3.2 Preprocessing (fit on train, transform on val/test)

| Group | Columns | Transform |
|---|---|---|
| Skewed | Rainfall, Evaporation, WindGustSpeed, WindSpeed9am, WindSpeed3pm | median impute → log1p |
| Symmetric | MinTemp, MaxTemp, Temp9am/3pm, Pressure9am/3pm, Humidity9am/3pm, Sunshine | median impute |
| Ordinal | Cloud9am, Cloud3pm | mode impute |
| Categorical | Location, WindGustDir, WindDir9am, WindDir3pm | 'Unknown' fill → one-hot |
| Binary | RainToday | 'Unknown' fill → label encode (1/0/−1) |
| Date | Month, DayOfYear | passthrough |

No rows removed for outliers (zero-inflation is physical, not error).

### 3.3 Models

| Model | Scaling | Notes |
|---|---|---|
| Logistic Regression | StandardScaler | `class_weight='balanced'` |
| LinearSVC | StandardScaler | `class_weight='balanced'` |
| XGBoost | None | `scale_pos_weight=3.5` |

### 3.4 Hyperparameter tuning (Optuna, 100 trials, 5-fold CV)

XGBoost best params: `n_estimators=878, max_depth=8, lr=0.028,
min_child_weight=9, subsample=0.835, colsample_bytree=0.605`.

---

## 4. Results

### 4.1 Cross-validation (5 chronological folds, mean ± std)

| Model | PR-AUC | F1 |
|---|---|---|
| Logistic Regression | 0.702 ± 0.020 | 0.630 ± 0.015 |
| LinearSVC | 0.702 ± 0.019 | 0.581 ± 0.020 |
| **XGBoost (tuned)** | **0.759 ± 0.015** | **0.662 ± 0.017** |

### 4.2 Test set (one-time evaluation)

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| **XGBoost (tuned)** | **0.7416** | **0.8863** | **0.6619** |

### 4.3 Top features (XGBoost importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | RainToday | 0.057 |
| 2 | Humidity3pm | 0.054 |
| 3 | Sunshine | 0.025 |
| 4 | Rainfall | 0.022 |
| 5 | Cloud3pm | 0.016 |
| 6 | WindGustSpeed | 0.015 |

### 4.4 Figures

| Figure | File |
|---|---|
| Confusion matrix | `reports/figures/confusion_matrix_xgboost.png` |
| ROC + PR curves | `reports/figures/roc_pr_xgboost.png` |
| Feature importance | `reports/figures/importance_xgboost.png` |
| Optuna importance | `reports/figures/importance_xgboost.png` (tuning) |

### 4.5 Regression: MaxTemp

Simple linear regression predicting `MaxTemp` from 4 features
(MinTemp, Humidity3pm, Sunshine, Pressure3pm).

| Metric | Value |
|---|---|
| R² | 0.8156 |
| MAE | 2.33 °C |
| RMSE | 3.00 °C |

**Coefficients:**

| Feature | Coefficient | Interpretation |
|---|---|---|
| MinTemp | +0.76 | Primary driver (same-day thermal range) |
| Sunshine | +0.30 | Solar heating increases peak temp |
| Humidity3pm | −0.14 | Cloud/humidity limits peak temp |
| Pressure3pm | −0.09 | Low pressure → warm air mass |

All signs are physically consistent. `MinTemp` + `Sunshine` together
explain the majority of variance. See `reports/figures/regression_maxtemp.png`.


---

## 5. Discussion

- XGBoost outperforms linear models by ~0.05 PR-AUC, confirming that
  non-linear interactions (humidity × pressure × wind) carry signal that a
  linear boundary cannot capture.
- `RainToday` dominates, as expected (autocorrelation of precipitation).
- The test PR-AUC (0.742) is slightly below the CV mean (0.759), consistent
  with mild distribution drift in 2016–2017 vs. the training period.
- Elaborate encodings (target encoding, circular sin/cos, MissingIndicator)
  provided no measurable gain over the baseline pipeline — the signal is
  concentrated in a few strong features.

---

## 6. Reproduction

```bash
pip install -r requirements.txt

# 1. Download data (first run only)
python3 -m src.modeling.create_folds

# 2. Train (all models, 5 folds)
python3 -m src.modeling.train

# 3. Tune XGBoost
python3 -m src.modeling.tune --model xgboost --n-trials 100

# 4. Final evaluation on test
python3 -m src.modeling.evaluate --model xgboost

# 5. Dashboard
streamlit run dashboard/app.py   
