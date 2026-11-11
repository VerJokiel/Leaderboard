""" Model description: AR(1) per (ISO3, variable) — the canonical benchmark, used as the universal floor of every per-country ensemble.

Implementation note: this model IS the benchmark. It applies the per-variable canonical transformation (log_diff / diff / as_is from var_classification)
before fitting and inverts after predicting, so its forecasts exactly match the cached benchmark predictions in data/forecasts/benchmarks/ar1.parquet.

Therefore its skill vs the benchmark is exactly 0 by construction. The scorecard admits it unconditionally — it is the floor every other model must beat.

When no other model wins for a given (ISO3, variable, horizon), the ensemble assembly uses just this model's forecast (= the AR(1) benchmark).
"""

from __future__ import annotations
                import pandas as pd
from benchmarks import benchmark_forecasts
from var_classification import family

class AR1PerSeries:
    name = "ar1_per_series"  family = "AR"  segmentation = "series" min_history = 5  variables = None  version = "0.3"  
inputs = (
        "Per (country i, variable v): apply variable v's canonical "
        "transformation T to the series y_{i,v,t}. Then fit AR(1) on "
        "the transformed series via OLS, forecast h-step ahead, and "
        "back-transform to original scale.")

    def __init__(self): self._series: dict[tuple[str, str], pd.Series] = {}

    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None: self._series.clear()
        for (iso, var), g in panel.sort_values("year").groupby( ["ISO3", "variable"], observed=True):
            if family(var) == "skip":
                continue
                
            gg = g.sort_values("year").dropna(subset=["value"])
            if len(gg) >= self.min_history: self._series[(iso, var)] = pd.Series( gg["value"].astype(float).values, index=gg["year"].astype(int).values)

 def predict(self, cutoff_year: int, horizons: list[int]) -> pd.DataFrame: empty_df = pd.DataFrame( columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"])
        if not self._series: return empty_df
        
        rows = []
        for (iso, var), s in self._series.items():
            preds = benchmark_forecasts(s, var, horizons, method="ar1")
            for h, p in preds.items():
                if pd.isna(p):
                    continue
                rows.append((iso, cutoff_year + h, var, h, float(p), float("nan"), float("nan")))
        return pd.DataFrame( rows, columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"] )
