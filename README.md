# Leaderboard

Leaderboard for chosen models: 30 specifications from the model files in this repo, evaluated on
one panel with five error measures and shown in an interactive HTML matrix (pick 3 of 30).

## Run

```bash
pip install numpy pandas scikit-learn linearmodels
python evaluate_models.py           # full run (~30 min on 2 cores); downloads data on first run
python evaluate_models.py --quick   # 3 origins per sample, smoke test
```

Open `[results/leaderboard.html](https://verjokiel.github.io/Leaderboard/)` in a browser.

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
