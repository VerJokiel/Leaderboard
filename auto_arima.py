""" Auto-ARIMA per (country, variable).

Selects (p, d, q) by BIC-minimization over a small grid. Uses statsmodels' ARIMA (no external pmdarima dependency).

For each (ISO3, variable):
  1. Determine integration order d via Augmented Dickey-Fuller test
     (d=0 if reject unit root, else d=1, capped at 2).
  2. For (p, q) in {0,1,2} × {0,1,2}, fit ARIMA(p, d, q) on the
     canonical-transformed series, select min-BIC.
  3. Forecast h-step ahead in transformed space, back-transform to level.

Why this is a useful addition: AR(1) and panel_ar fix d=0; auto_arima 
adds differencing as a selectable hyperparameter and joins MA terms
which capture short-run noise correlation that pure AR doesn't.

PARAM_GRID sweeps over the search-grid size (cheap vs full). """

from __future__ import logging
                import itertools
                import warnings 
from dataclasses import dataclass
                import numpy as np
                import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from var_classification import family, transformation
from transformations import forward as _forward, inverse as _inverse

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class AutoArimaPerSeries:
    family = "ARIMA"  segmentation = "series" variables = None version = "0.2" # C1 adjustemnt to 0.2
    train_window_years = None

    PARAM_GRID = {
        "max_pq":   [2, 3],   # search (p, q) in {0..max_pq}
        "use_adf":  [True, False],  # ADF for d-selection vs fixed d=0
    }

    def __init__(self, max_pq: int = 2, use_adf: bool = True, min_history: int = 12): # C3 (set min hist as an argument)
        self.max_pq = int(max_pq)
        self.use_adf = bool(use_adf)
        self.min_history = int(min_history) # according to C3
        adf_tag = "adf" if self.use_adf else "nodiff"
        self.name = f"auto_arima_pq{self.max_pq}_{'adf' if self.use_adf else 'nodiff'}"
        self.inputs = (
            f"Per (country i, variable v) in v's canonical transformed space T: "
            f"select ARIMA(p, d, q) by BIC over (p, q) in {{0..{self.max_pq}}}² "
            f"with d ∈ "
            + ("{0, 1} chosen by ADF unit-root test on T(y)" if self.use_adf else "{0}")
            + ". Fit via statsmodels.ARIMA on the transformed series. "
            f"h-step forecast from selected model, back-transform to level. "
            f"Differencing inside ARIMA handles trends that AR(p)-on-as_is can't. "
            f"Implementation: statsmodels.tsa.arima.model.ARIMA + statsmodels.tsa."
            f"stattools.adfuller for d selection. "
            f"Reference: Hyndman-Khandakar (2008, J. Statistical Software) "
            f"'Automatic Time Series Forecasting: the forecast package for R' — "
            f"this Python port of their auto.arima."
        )
      # AD futher clean

    def _select_d(self, y: np.ndarray) -> int:
        if not self.use_adf:
            return 0
        if len(y) < 12:
            return 0
        try:
            stat, pvalue, *_ = adfuller(y, autolag="BIC")
        except Exception:
            return 0
        # p < 0.05: reject unit root → d=0. Else d=1.
        return 0 if pvalue < 0.05 else 1

    def _select_pq(self, y: np.ndarray, d: int) -> tuple[int, int, float] | None:
        """Grid search (p, q) by BIC. Returns (p, q, bic) or None."""
        best = None
        for p in range(0, self.max_pq + 1):
            for q in range(0, self.max_pq + 1):
                if p == 0 and q == 0 and d == 0:
                    continue
                try: 
                    fit = ARIMA(y, order=(p, d, q), trend="c" if d == 0 else "t" ).fit(
                        method_kwargs={"warn_convergence": False, "maxiter": 50}
                    )
                    bic = float(fit.bic)
                    if not np.isfinite(bic):
                        continue
                    if best is None or bic < best[2]:
                        best = (p, q, bic, fit)
                except Exception:
                    continue
        if best is None:
            return None
        return best  # (p, q, bic, fit)

    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None:
        self._models: dict[tuple[str, str], object] = {}
        self._ctx: dict[tuple[str, str], dict] = {}
        for (iso, var), g in panel.groupby(["ISO3", "variable"], observed=True):
            if family(var) == "skip":
                continue
            vals = g.sort_values("year")["value"].dropna().astype(float).values
            if len(vals) < self.min_history:
                continue
            tmode = transformation(var)
            tvals, ctx = _forward(vals, tmode)
            if len(tvals) < self.min_history - 1:
                continue
            d = self._select_d(tvals)
            res = self._select_pq(tvals, d)
            if res is None:
                continue
            _, _, _, fit_obj = res
            self._models[(iso, var)] = fit_obj
            self._ctx[(iso, var)] = ctx

    def predict(self, panel: pd.DataFrame, cutoff_year: int, horizons: list[int]) -> pd.DataFrame:
        rows = []
        h_max = max(horizons)
        for (iso, var), fit_obj in self._models.items():
            tmode = transformation(var)
            try:
                pred = fit_obj.get_forecast(steps=h_max)
                fc = pred.predicted_mean
                ci = pred.conf_int(alpha=0.32)   # 68% interval -> ~ +/- 1 sigma
            except Exception:
                continue
            t_steps = list(np.asarray(fc).flatten())
            t_lo = list(np.asarray(ci.iloc[:, 0] if hasattr(ci, "iloc") else ci[:, 0]).flatten())
            t_hi = list(np.asarray(ci.iloc[:, 1] if hasattr(ci, "iloc") else ci[:, 1]).flatten())
            ctx = self._ctx[(iso, var)]
            lvl = _inverse(t_steps, ctx, mode=tmode)
            lvl_lo = _inverse(t_lo, ctx, mode=tmode)
            lvl_hi = _inverse(t_hi, ctx, mode=tmode)
            for h in horizons:
                if h - 1 >= len(lvl):
                    continue
                yhat = lvl[h - 1]
                if np.isfinite(yhat):
                    rows.append((iso, cutoff_year + h, var, h, float(yhat),
                                 float(min(lvl_lo[h - 1], lvl_hi[h - 1])),
                                 float(max(lvl_lo[h - 1], lvl_hi[h - 1]))))
        return pd.DataFrame(
            rows, columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
        )


def model_instances() -> list[AutoArimaPerSeries]:
    keys = list(AutoArimaPerSeries.PARAM_GRID.keys())
    return [
        AutoArimaPerSeries(**dict(zip(keys, combo)))
        for combo in itertools.product(*[AutoArimaPerSeries.PARAM_GRID[k] for k in keys])
    ]
