"""
Pooled AR(p) with Country Fixed Effects (Panel AR).

Description:
    Estimates a univariate pooled autoregressive model of order p for each macro variable 
    independently across a multi-country panel. It captures country-specific baselines 
    via fixed effects while sharing autoregressive dynamics across the panel.

Estimation:
    Fitted using `linearmodels.panel.PanelOLS` with `entity_effects=True`, which 
    analytically absorbs country fixed effects (alpha_i). Fixed effects can be 
    recovered from the estimated_effects attribute.

Feature Matrix (X_{i,t}) per country i at time t:
    - Country Fixed Effects: Absorbed via entity dummy structure.
    - Autoregressive Lags: p columns representing T(v)_{i, t-1} ... T(v)_{i, t-p}.

Target Variable (y):
    - T(v)_{i, t} in the variable's canonical transformed space (e.g., log_diff, diff, as_is). 
      Predictions are back-transformed to original levels using transformations.inverse.

Forecasting (h-step):
    - Performs recursive multi-step forecasting by feeding predicted values back 
      into the own-lag sequence iteratively in transformed space.

Hyperparameters (PARAM_GRID):
    - p: Autoregressive lag order (swept by the evaluation harness).
"""

from __future__ import annotations
import itertools
import warnings
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from var_classification import family, transformation
from transformations import forward as _forward, inverse as _inverse


class PanelAR:
    family = "AR"
    segmentation = "panel_global"
    variables = None
    version = "0.3.1"   # transformation-aware

    # Hyperparameter sweep: lag order × training window
    PARAM_GRID = {
        "p": [1, 2, 3, 5],
        "train_window_years": [None, 50, 20, 10],
    }

    def __init__(self, p: int = 2, train_window_years: int | None = None):
        self.p = int(p)
        self.train_window_years = train_window_years
        suffix = "" if train_window_years is None else f"_w{train_window_years}"
        self.name = f"panel_ar_p{self.p}{suffix}"
        self.min_history = self.p + 10   # need lags + ~10 obs to fit
        self.inputs = (
            f"Per variable v, in v's canonical transformed space T (log_diff / "
            f"diff / as_is): T(y)_{{i,v,t}} = alpha_{{i,v}} + "
            f"sum_{{k=1..{self.p}}} beta_k * T(y)_{{i,v,t-k}} + eps_{{i,v,t}}. "
            f"Within-FE estimator on the pooled stack of all countries (per "
            f"variable). h-step forecast = iterate the difference equation in "
            f"transformed space, then back-transform to level. Country fixed "
            f"effects alpha_{{i,v}} let each country have its own mean growth "
            f"rate / mean level. NO cross-variable features. "
            f"Implementation: linearmodels.panel.PanelOLS with entity_effects=True."
        )

    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None:
        self._coef: dict[str, dict] = {}
        self._last_t: dict[tuple[str, str], list[float]] = {}  # last p transformed obs
        self._ctx: dict[tuple[str, str], dict] = {}             # transformation ctx

        self._last_year: dict[tuple[str, str], int] = {}
        # 1. Enforce cutoff_year globally to prevent data leakage
        panel = panel[panel["year"] <= cutoff_year]

        for var, dfv in panel.groupby("variable", observed=True):
            fam = family(var)
            if fam == "skip":
                continue
            tmode = transformation(var)
            dfv = dfv.sort_values(["ISO3", "year"])

            rows = []
            for iso, g in dfv.groupby("ISO3", observed=True):
                g_sub = g.dropna(subset=["value"])
                # 2. Optional rolling training window
                if self.train_window_years is not None:
                    g_sub = g_sub[g_sub["year"] > cutoff_year - self.train_window_years]
                if len(g_sub) < self.p + 3:
                    continue
                vals = g_sub["value"].astype(float).values
                years_arr = g_sub["year"].astype(int).values

                # Transform per country; forward() may trim leading elements
                tvals, ctx = _forward(vals, tmode)
                tvals = np.asarray(tvals, dtype=float)
                if len(tvals) < self.p + 2:
                    continue
                years_aligned = years_arr[-len(tvals):]
                for t in range(self.p, len(tvals)):
                    rows.append([iso, int(years_aligned[t]), tvals[t]]
                                + [tvals[t - k] for k in range(1, self.p + 1)])
                # Last p transformed values (most recent first) + ctx
                self._last_t[(iso, var)] = tvals[-self.p:][::-1].tolist()
                self._ctx[(iso, var)] = ctx
                self._last_year[(iso, var)] = int(years_arr[-1])

            if not rows:
                continue
            lag_cols = [f"l{k}" for k in range(1, self.p + 1)]
            df = pd.DataFrame(rows, columns=["ISO3", "year", "y"] + lag_cols)
    
            # Drop duplicates on MultiIndex just in case of overlapping data
            df = df.drop_duplicates(subset=["ISO3", "year"]).set_index(["ISO3", "year"])
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = PanelOLS(df["y"], df[lag_cols], entity_effects=True).fit()
            except Exception:
                continue
            beta = results.params.values.astype(float)
            # Recover per-country fixed effects from estimated_effects (constant
            # within each entity); take the mean per ISO3 as a robust scalar.
            eff = results.estimated_effects
            eff_col = eff.columns[0] if hasattr(eff, "columns") else None
            eff_series = eff[eff_col] if eff_col is not None else eff.iloc[:, 0]
            fe = {iso: float(v) for iso, v in eff_series.groupby(level="ISO3").mean().items()}
            self._coef[var] = {"beta": beta, "fe": fe}

    def predict(self, panel: pd.DataFrame, cutoff_year: int, horizons: list[int]) -> pd.DataFrame:
        rows = []
        h_max = max(horizons)
        for (iso, var), last_t in self._last_t.items():
            if var not in self._coef:
                continue
            tmode = transformation(var)
            c = self._coef[var]
            beta = c["beta"]
            alpha = c["fe"].get(iso)
            if alpha is None:
                continue
            # Iterate in transformed space
            t_steps: list[float] = []
            lags = list(last_t)  # [t-1, t-2, ..., t-p]  in transformed space
            offset = cutoff_year - self._last_year[(iso, var)]   # ragged edge
            if offset > 3:
                continue                                          # too stale
            for _h in range(h_max + offset):
                t_hat = alpha + float(np.dot(beta, lags))
                t_steps.append(t_hat)
                lags = [t_hat] + lags[:-1]
            # Back-transform to original scale
            level_steps = _inverse(t_steps, self._ctx[(iso, var)], mode=tmode)
            for h in horizons:
                if h - 1 + offset >= len(level_steps):
                    continue
                yhat = level_steps[h - 1 + offset]
                if np.isfinite(yhat):
                    rows.append((iso, cutoff_year + h, var, h, float(yhat),
                                 float("nan"), float("nan")))
        return pd.DataFrame(
            rows, columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
        )

def model_instances() -> list[PanelAR]:
    keys = list(PanelAR.PARAM_GRID.keys())
    return [
        PanelAR(**dict(zip(keys, combo)))
        for combo in itertools.product(*[PanelAR.PARAM_GRID[k] for k in keys])
    ]
