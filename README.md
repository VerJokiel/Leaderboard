# Leaderboard

Leaderboard for chosen models: 30 specifications from the model files in this repo (currently 16), evaluated on
one panel with five error measures and shown in an interactive HTML matrix.

## Run

```bash
pip install numpy pandas scikit-learn linearmodels
python evaluate_models.py           # full run (~30 min on 2 cores); downloads data on first run
python evaluate_models.py --quick   # 3 origins per sample, smoke test
```

Open `(https://verjokiel.github.io/Leaderboard/)` in a browser.

## What it does

| Step | Detail |
|---|---|
| Data | `support/fetch_data.py`: World Bank WDI (CPI, inflation, real GDP, M2, USD FX, current account, nominal GDP), IMF WEO (unemployment, government debt), FRED (OECD 3-month and 10-year rates, BIS REER). 20 countries, from 1960 where available, cached in `data_cache/`. |
| Same sample | Every model gets the same panel and the same origins, re-estimated at each origin with data ≤ origin only. |
| In-sample | origins 1990–2014, forecasts h = 1, 2 |
| Out-of-sample | origins 2015–2024, scored against outturns up to the **2025 cut-off** |
| Final forecast | fitted on all data to 2025 → `results/final_forecasts.csv` (2026, 2027) |
| Measures | 
| RMSE,|
MAE, 
MAPE, 
Theil U2 (vs no-change), 
directional accuracy: how often a forecast correctly predicts the direction of change—up or down—rather than the magnitude, with common metrics including Mean Directional Accuracy (MDA)) |
| Transformations | FRED-MD codes in `support/var_classification.py` (5 = log diff for price, GDP, money, FX; 2 = diff for rates, unemployment, debt; 1 = level for inflation, current account) |

Outputs in `results/`: `forecasts.csv`, `forecast_errors.csv`, `scores.csv`, `model_run_log.csv`,
`final_forecasts.csv`, `leaderboard.html`.

## Fixes made to the model files

- All panel models: a country whose data ended before the cutoff got its forecasts labelled with the wrong year (ragged edge). Forecasts now count steps from the last observation.
