# Leaderboard

Leaderboard for chosen models: 30 specifications from the model files in this repo (currently 16), evaluated on
one panel with five error measures and shown in an interactive HTML matrix.

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

## Fixes made to the model files

- Syntax / indentation errors in 5 files; `panel_random_forest (1).py` renamed to `panel_random_forest.py`.
- `multivar_dfm_v2.py`: estimated on the full panel instead of data ≤ cutoff (look-ahead); called a non-existent `_ar_fit`.
- `pca_country_v2.py`: `_prepare_series`, `_prepare_aggregates`, `_candidate_list`, `_aggregate_for` were missing (restored from `pca_country_v3.py`).
- `pca_panel_v2.py`: `_ar_fit`/`_ar_sim` lacked `self`/`@staticmethod`; `tstat_screen` in the grid but not in `__init__`; `col_sd` not stored.
- `panel_ar_country_fe.py`: column `t` vs `year` mismatch.
- `panel_ridge_multivar.py`, `panel_random_forest.py`: `cutoff_year` not passed to the feature builder; NaN rows reached the estimator.
- `panel_lasso_targeted.py`: training window computed but not used; `np.mean` on series with gaps gave all-NaN features.
- All panel models: a country whose data ended before the cutoff got its forecasts labelled with the wrong year (ragged edge). Forecasts now count steps from the last observation.
