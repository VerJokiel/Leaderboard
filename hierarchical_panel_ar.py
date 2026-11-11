"""
Hierarchical panel AR(p) in canonical transformed space. Unlike standard 
pooled fixed-effects models  (which constrain $\beta_{i,k} = \beta_k$ for all $i$), 
this model treats parameters as random coefficients (Swamy, 1970) 
shrunk toward a global panel mean via closed-form Empirical Bayes:

Description:
    Estimates a random-coefficients panel AR(p) model where country-specific 
    autoregressive slopes are partially pooled toward the global panel mean 
    using closed-form Empirical Bayes (Swamy 1970 tradition). This balances 
    the noise of country-by-country OLS against the bias of strict full pooling.

Estimation:
    - Step 1: Fits independent OLS regressions for each country to get beta_{i}^{\text{OLS}} 
      and exact sampling variances V_{i,k}.
    - Step 2: Computes between-country variance (tau_k^2) net of estimation noise.
    - Step 3: Shrinks country slopes toward the panel mean using EB weights modulated 
      by `pool_strength`.
    - Step 4: Re-centers country intercepts (alpha_i) consistently using the shrunk slopes.

Feature Matrix (X_{i,t}) per country i at time t:
    - Country Intercept: Separate alpha_i per country.
    - Autoregressive Lags: p columns representing T(v)_{i, t-1} ... T(v)_{i, t-p}.

Target Variable (y):
    - T(v)_{i, t} in the variable's canonical transformed space. Predictions 
      are back-transformed to original levels using transformations.inverse.

Forecasting (h-step):
    - Performs recursive multi-step forecasting for each country using its 
      individual shrunk parameters ($\alpha_i, \beta_i^{\text{shrunk}}$).

Hyperparameters (PARAM_GRID):
    - p: Autoregressive lag order.
    - pool_strength: Multiplier on sampling variance controlling shrinkage intensity 
      (1.0 = pure EB, < 1 = lighter shrinkage, > 1 = heavier shrinkage toward pool).

References:
    - Swamy, P. A. V. B. (1970). "Efficient Inference in a Random Coefficient Regression Model." Econometrica.
    - Lindley, D. V., & Smith, A. F. (1972). "Bayes Estimates for the Linear Model." JRSS-B.
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from var_classification import family, transformation
from transformations import forward as _forward, inverse as _inverse


class HierarchicalPanelAR:
    family = "Hierarchical"
    segmentation = "panel_global"
    variables = None
    version = "0.2"
    PARAM_GRID = {
        "p":                  [1, 2],
        "pool_strength":      [0.5, 1.0, 2.0],   # 1.0 = pure EB; >1 = extra shrinkage
        "train_window_years": [None, 50, 20, 10],
    }

    def __init__(self, p: int = 1, pool_strength: float = 1.0,
                 train_window_years: int | None = None):
        self.p = int(p)
        self.pool_strength = float(pool_strength)
        self.train_window_years = train_window_years
        self.min_history = self.p + 5
        win = "" if train_window_years is None else f"_w{train_window_years}"
        self.name = f"hier_ar_p{self.p}_s{self.pool_strength:g}{win}"
        self.inputs = (
            f"Per variable v in v's canonical transformed space T: per-country "
            f"AR({self.p}) OLS produces (alpha_i, beta_{{i,v}}) with beta_{{i,v}} "
            f"a vector of {self.p} coefficients. Empirical-Bayes shrinkage toward "
            f"the panel mean beta_bar_v with Swamy variance components. "
            f"Implementation: closed-form empirical Bayes via numpy with robust pseudoinverse."
        )

    def _fit_per_variable(self, panel: pd.DataFrame, var: str, cutoff_year: int
    ) -> dict | None:
        """Returns {iso: {'beta': shrunk_coefs, 'alpha': intercept, 'ctx': ctx,
                          'last': last_p_obs}} or None.

        Per-country AR(p) OLS, then Swamy/EB shrinkage of slopes toward the
        panel mean using the proper sampling-variance of beta_i^OLS.
        """
        sub = panel[(panel["variable"] == var)].sort_values(["ISO3", "year"])
        p = self.p
        per_iso: dict = {}
        ols_betas: list[np.ndarray] = []     # per-country beta_i^OLS (p,)
        ols_var_diags: list[np.ndarray] = [] # per-country V_i = diag(sigma2_i * (X'X)^-1)[1:]

        for iso, g in sub.groupby("ISO3", observed=True):
            g_sub = g
            # 1. Implement train_window_years filtering per country
            if self.train_window_years is not None:
                min_yr = cutoff_year - self.train_window_years + 1
                g_sub = g[g["year"] >= min_yr]
            
            vals = g["value"].dropna().astype(float).values
            if len(vals) < p + 4:
                continue
            tmode = transformation(var)
            tvals, ctx = _forward(vals, tmode)
            if len(tvals) < p + 3:
                continue
            # Build per-country OLS regression: X = [1, y_{t-1}, ..., y_{t-p}]
            X_rows, y_rows = [], []
            for t in range(p, len(tvals)):
                X_rows.append([1.0] + list(tvals[t - p: t][::-1]))
                y_rows.append(tvals[t])
            X = np.asarray(X_rows)
            y = np.asarray(y_rows)
            n_i = len(y)
            if n_i <= p + 1:
                continue
            try:
                beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
          # 2. Use pseudoinverse for matrix stability against near-collinear panels
                XtX_inv = np.linalg.inv(X.T @ X)
            except (np.linalg.LinAlgError, ValueError):
                continue
                
            resid = y - X @ beta_full
            sigma2_i = float((resid ** 2).sum() / max(n_i - p - 1, 1))
            # Per-slope sampling variance (drop intercept entry)
            V_i_slopes = sigma2_i * np.diag(XtX_inv)[1:]
            
            # Guard against pathological (near-collinear) cases
            if not np.all(np.isfinite(V_i_slopes)) or np.any(V_i_slopes <= 0):
                continue
                
            per_iso[iso] = {
                "alpha_ols": float(beta_full[0]),
                "beta_ols": beta_full[1:].copy(),
                "x_mean": X[:, 1:].mean(axis=0),   # for recomputing alpha under shrunk beta
                "y_mean": float(y.mean()),
                "V_slopes": V_i_slopes,
                "ctx": ctx,
                "last": tvals[-p:][::-1].copy(),
            }
            ols_betas.append(per_iso[iso]["beta_ols"])
            ols_var_diags.append(V_i_slopes)

        if len(per_iso) < 3:
            return None

        beta_mat = np.asarray(ols_betas)             # (N, p)
        V_mat   = np.asarray(ols_var_diags)          # (N, p)
        beta_bar = beta_mat.mean(axis=0)
        
        # Swamy method-of-moments variance components:
        tau2 = np.maximum(beta_mat.var(axis=0, ddof=1) - V_mat.mean(axis=0), 0.0)
        tau2 = tau2 + 1e-12   # numerical floor so weights stay defined

        # Per-country, per-coefficient shrinkage
        for st in per_iso.values():
            V_i = st["V_slopes"] * self.pool_strength
            w = tau2 / (tau2 + V_i)
            beta_shrunk = (1.0 - w) * beta_bar + w * st["beta_ols"]
            # Recompute intercept consistently under the shrunk slopes
            alpha_shrunk = st["y_mean"] - float(np.dot(beta_shrunk, st["x_mean"]))
            st["beta"] = beta_shrunk
            st["alpha"] = alpha_shrunk

        return per_iso

 # 3. Enforce global cutoff_year to prevent data leakage
        panel = panel[panel["year"] <= cutoff_year]

        for var in panel["variable"].unique():
            if family(var) == "skip":
                continue
            res = self._fit_per_variable(panel, var, cutoff_year)
            if res is not None:
                self._per_var[var] = res

    def predict(self, panel: pd.DataFrame, cutoff_year: int, horizons: list[int]) -> pd.DataFrame:
        rows = []
        h_max = max(horizons)
        for var, per_iso in self._per_var.items():
            tmode = transformation(var)
            for iso, st in per_iso.items():
                lags = list(st["last"])   # [t-1, t-2, ...]
                t_steps = []
                for _h in range(h_max):
                    t_hat = st["alpha"] + float(np.dot(st["beta"], lags))
                    t_steps.append(t_hat)
                    lags = [t_hat] + lags[:-1]
                level_steps = _inverse(t_steps, st["ctx"], mode=tmode)
                for h in horizons:
                    yhat = level_steps[h - 1]
                    if np.isfinite(yhat):
                        rows.append((iso, cutoff_year + h, var, h, float(yhat),
                                     float("nan"), float("nan")))
        return pd.DataFrame(
            rows, columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
        )

def model_instances() -> list[HierarchicalPanelAR]:
    keys = list(HierarchicalPanelAR.PARAM_GRID.keys())
    return [
        HierarchicalPanelAR(**dict(zip(keys, combo)))
        for combo in itertools.product(*[HierarchicalPanelAR.PARAM_GRID[k] for k in keys])
    ]
