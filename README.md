# Leaderboard

Leaderboard for chosen models: 30 specifications from the model files in this repo, evaluated on
one panel with five error measures and shown in an interactive HTML matrix (pick 3 of 30).

## Run

```bash
pip install numpy pandas scikit-learn linearmodels
python evaluate_models.py           # full run (~30 min on 2 cores); downloads data on first run
python evaluate_models.py --quick   # 3 origins per sample, smoke test
```

Open `results/leaderboard.html` in a browser.

## What it does

| Step | Detail |
|---|---|
| Data | `support/fetch_data.py`: World Bank WDI (CPI, inflation, real GDP, M2, USD FX, current account, nominal GDP), IMF WEO (unemployment, government debt), FRED (OECD 3-month and 10-year rates, BIS REER). 20 countries, from 1960 where available, cached in `data_cache/`. |
| Same sample | Every model gets the same panel and the same origins, re-estimated at each origin with data ≤ origin only. |
| In-sample | origins 1990–2014, forecasts h = 1, 2 |
| Out-of-sample | origins 2015–2024, scored against outturns up to the **2025 cut-off** |
| Final forecast | fitted on all data to 2025 → `results/final_forecasts.csv` (2026, 2027) |
| Measures | RMSE, MAE, MAPE, Theil U2 (vs no-change), directional accuracy |
| Transformations | FRED-MD codes in `support/var_classification.py` (5 = log diff for price, GDP, money, FX; 2 = diff for rates, unemployment, debt; 1 = level for inflation, current account) |

Outputs in `results/`: `forecasts.csv`, `forecast_errors.csv`, `scores.csv`, `model_run_log.csv`,
`final_forecasts.csv`, `leaderboard.html`.

## Notes (11.11.2026)

### Ad code cleaning

Each model file was restructured into the same steps: description, feature matrix, target
variable, estimator, forecasting, and hyperparameters.

Every model had a similar data-leakage issue. `cutoff_year` was not properly embedded in the
`fit()` function, so the panel data was never sliced correctly. Without that cut, the model was
trained on the whole data sample it was given, including the years it was supposed to forecast.
After the adjustment, each model trains only on data up to the cut-off, plus an optional
rolling training window:

```python
# 1. Prevent data leakage by enforcing cutoff_year
panel = panel[panel["year"] <= cutoff_year]
if panel.empty:
    return None

# 2. Implement train_window_years filtering
if self.train_window_years is not None:
    min_allowed_year = cutoff_year - self.train_window_years + 1
    panel = panel[panel["year"] >= min_allowed_year]
```
