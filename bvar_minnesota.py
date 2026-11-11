""" Bayesian VAR per country with Minnesota prior — Doan-Litterman-Sims dummy-observations form.
References
----------
- Doan, T., Litterman, R. and Sims, C. (1984). "Forecasting and Conditional
  Projection Using Realistic Prior Distributions." Econometric Reviews
  3(1), 1-100. (Original Minnesota prior + dummy-observation form.)
- Banbura, M., Giannone, D. and Reichlin, L. (2010). "Large Bayesian
  Vector Auto Regressions." Journal of Applied Econometrics 25, 71-92.
- Giannone, D., Lenza, M. and Primiceri, G. E. (2015). "Prior Selection
  for Vector Autoregressions." Review of Economics and Statistics 97(2),
  436-451. (Hierarchical hyperparameter selection — we use fixed
  hyperparameters from the PARAM_GRID rather than maximising the
  marginal likelihood.)
"""

from __future__ import annotations
import itertools
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from var_classification import family, transformation
from transformations import forward as _forward, inverse as _inverse


_BLOCK_SMALL = ("rGDP", "CPI", "strate")
_BLOCK_MID   = ("rGDP", "CPI", "strate", "USDfx")
_BLOCK_BIG   = ("rGDP", "CPI", "strate", "USDfx", "unemp")


class BVARMinnesota:
    family = "BVAR"
    segmentation = "country_multivar"
    version = "0.2"
    train_window_years = None

    PARAM_GRID = {
        "p":                  [1, 2],
        "lambda1":            [0.2, 0.5],
        "lambda2":            [0.5, 1.0],
        "block":              [_BLOCK_SMALL, _BLOCK_MID, _BLOCK_BIG],
        "train_window_years": [None, 50, 20, 10],
    }

def __init__(
        self,
        p: int = 1,
        lambda1: float = 0.2,
        lambda2: float = 0.5,
        block: tuple = _BLOCK_MID,
        train_window_years: int | None = None,
    ):
                     
        self.p = int(p)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.block = tuple(block)
        self.train_window_years = train_window_years
        self.variables = list(self.block)
        self.min_history = self.p + 5
        block_tag = "_".join(self.block)
        win = "" if train_window_years is None else f"_w{train_window_years}"
        self.name = f"bvar_min_p{self.p}_l1_{self.lambda1:g}_l2_{self.lambda2:g}_{block_tag}{win}"
        self.inputs = (
            f"Bayesian VAR(p={self.p}) per country on the joint vector "
            f"Y_{{i,t}} = [T(v)_{{i,t}} for v in {self.block}] in canonical "
            f"transformed space. Y_{{i,t}} = c_i + sum_{{k=1..{self.p}}} A_k "
            f"Y_{{i,t-k}} + eps. Minnesota prior on A_k: own-lag-1 coefficient "
            f"shrunk to 0 (we're in transformed/stationary space), cross-lag "
            f"coefficients shrunk to 0 with cross-equation factor lambda2="
            f"{self.lambda2:g}, higher-lag coefficients shrunk to 0 with "
            f"strength scaling 1/k^2. Overall prior tightness lambda1="
            f"{self.lambda1:g}. Posterior mean computed via Theil mixed "
            f"estimation: augment OLS with Minnesota dummy observations, then "
            f"fit equation-by-equation with statsmodels.OLS. h-step forecast "
            f"iterates the VAR in transformed space, then back-transforms. "
            f"Implementation: statsmodels OLS on stacked actual + dummy rows. "
            f"References: Doan, Litterman, Sims (1984, Econometric Reviews) "
            f"'Forecasting and Conditional Projection Using Realistic Prior "
            f"Distributions' for the Minnesota prior and the dummy-observations "
            f"representation; Banbura, Giannone, Reichlin (2010, JAE) for the "
            f"large-BVAR formulation; Giannone, Lenza, Primiceri (2015, REStat) "
            f"'Prior Selection for Vector Autoregressions' for hierarchical "
            f"hyperparameter selection (we use fixed hyperparameters here)."
        )

    # ------------------------------------------------------------------
    # data assembly
    # ------------------------------------------------------------------
    def _build_country_block(self, panel: pd.DataFrame, iso: str
    ) -> tuple[np.ndarray, list[int], dict[str, dict]] | None:
        sub = panel[(panel["ISO3"] == iso) & (panel["variable"].isin(self.block))]
        wide = sub.pivot_table(
            index="year", columns="variable", values="value", aggfunc="last"
        ).reindex(columns=list(self.block))
        wide = wide.dropna()
        if len(wide) < self.p + 5:
            return None
        years = wide.index.tolist()
        T_cols = []
        ctx_per_var = {}
        for v in self.block:
            tvals, ctx = _forward(wide[v].values, transformation(v))
            T_cols.append(tvals)
            ctx_per_var[v] = ctx
        min_len = min(len(c) for c in T_cols)
        T = np.column_stack([c[-min_len:] for c in T_cols])
        years = years[-min_len:]
        return T, years, ctx_per_var

    # ------------------------------------------------------------------
    # Minnesota dummy observations (Doan-Litterman-Sims 1984)
    # ------------------------------------------------------------------
    def _univariate_ar_sigma(self, T: np.ndarray) -> np.ndarray:
        """AR(p) residual std per variable — used as the sigma_v scale
        in the Minnesota prior. Mirrors the canonical pre-processing
        described in Banbura-Giannone-Reichlin (2010) Appendix A."""
        n, k = T.shape
        sigmas = np.empty(k)
        p = self.p
        for j in range(k):
            y = T[:, j]
            # Build AR(p) design X for this column
            X_rows = []
            y_rows = []
            for t in range(p, n):
                X_rows.append([1.0] + [y[t - lag] for lag in range(1, p + 1)])
                y_rows.append(y[t])
            X_ar = np.asarray(X_rows)
            y_ar = np.asarray(y_rows)
            try:
                # plain OLS for the scale
                beta, *_ = np.linalg.lstsq(X_ar, y_ar, rcond=None)
                resid = y_ar - X_ar @ beta
                # use sample std of residuals; floor to avoid divide-by-zero
                s = float(np.std(resid, ddof=max(1, X_ar.shape[1])))
            except np.linalg.LinAlgError:
                s = float(np.std(y_ar, ddof=1))
            sigmas[j] = max(s, 1e-8)
        return sigmas

    def _fit_equation(self, X_act: np.ndarray, y_act: np.ndarray,
                      eq: int, k: int, sigmas: np.ndarray
                      ) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Fit a single equation of the VAR by OLS on stacked actual +
        Minnesota dummy rows. `eq` is the index of the LHS variable in
        the block; `k` is the block size; `sigmas` are univariate AR
        residual SDs.
        """
        p = self.p
        n_cols = 1 + k * p

        # Per-equation dummy rows.
        # Row 0: intercept — very flat (sd 1e3 -> regressor entry 1e-3).
        rows_X = [np.zeros(n_cols)]
        rows_X[0][0] = 1e-3
        rows_y = [0.0]

        for L in range(1, p + 1):
            for v_prime in range(k):
                col = 1 + (L - 1) * k + v_prime
                if v_prime == eq:
                    sd = self.lambda1 / (L ** 2)
                else:
                    sd = (self.lambda1 * self.lambda2 / (L ** 2)
                          * sigmas[eq] / sigmas[v_prime])
                sd = max(sd, 1e-10)
                r = np.zeros(n_cols)
                r[col] = 1.0 / sd
                rows_X.append(r)
                rows_y.append(0.0)  # prior mean is 0 in transformed space

        X_dum = np.asarray(rows_X)
        y_dum = np.asarray(rows_y)

        X_full = np.vstack([X_act, X_dum])
        y_full = np.concatenate([y_act, y_dum])

        # statsmodels.OLS with hasconst=True so the intercept column (col 0)
        # is recognised in diagnostics.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y_full, X_full, hasconst=True)
            res = model.fit()
        return res

    def _fit_country(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Fit BVAR(p) on transformed block T (n_obs, k). Returns (c, A_list)."""
        n, k = T.shape
        p = self.p
        if n < p + 3:
            return None

        # Actual data design [intercept, lag1 vars, lag2 vars, ...]
        X_rows = []
        for t in range(p, n):
            r = [1.0]
            for lag in range(1, p + 1):
                r.extend(T[t - lag, :])
            X_rows.append(r)
        X_act = np.asarray(X_rows)       # (n-p, 1 + k*p)
        Y_act = T[p:, :]                 # (n-p, k)

        sigmas = self._univariate_ar_sigma(T)

        n_cols = 1 + k * p
        coef = np.zeros((n_cols, k))
        for eq in range(k):
            res = self._fit_equation(X_act, Y_act[:, eq], eq, k, sigmas)
            coef[:, eq] = res.params

        c_vec = coef[0, :]
        A_stack = coef[1:, :]
        A_list = []
        for lag in range(p):
            A_lag = A_stack[lag * k: (lag + 1) * k, :].T   # (k, k)
            A_list.append(A_lag)
        return c_vec, A_list

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None:
        self._coef: dict[str, dict] = {}
        self._last_t: dict[str, np.ndarray] = {}
        self._ctx: dict[str, dict[str, dict]] = {}
        sub = panel[panel["variable"].isin(self.block) & (panel["year"] <= cutoff_year)]
        for iso in sub["ISO3"].unique():
            built = self._build_country_block(sub, iso)
            if built is None:
                continue
            T, years, ctx_per_var = built
            res = self._fit_country(T)
            if res is None:
                continue
            c_vec, A_list = res
            self._coef[iso] = {"c": c_vec, "A": A_list}
            self._last_t[iso] = T[-self.p:][::-1].copy()
            self._ctx[iso] = ctx_per_var

    def predict(self, panel: pd.DataFrame, cutoff_year: int, horizons: list[int]) -> pd.DataFrame:
        rows = []
        h_max = max(horizons)
        for iso, coef in self._coef.items():
            c = coef["c"]
            A_list = coef["A"]
            lags = self._last_t[iso].copy()
            t_steps_by_var = []
            for _h in range(h_max):
                y_hat = c.copy()
                for lag in range(self.p):
                    y_hat = y_hat + A_list[lag] @ lags[lag, :]
                t_steps_by_var.append(y_hat)
                lags = np.vstack([y_hat[None, :], lags[:-1, :]])
            t_steps_array = np.array(t_steps_by_var)
            for j, v in enumerate(self.block):
                level_steps = _inverse(
                    t_steps_array[:, j].tolist(),
                    self._ctx[iso][v],
                    mode=transformation(v),
                )
                for h in horizons:
                    yhat = level_steps[h - 1]
                    if np.isfinite(yhat):
                        rows.append((iso, cutoff_year + h, v, h, float(yhat),
                                     float("nan"), float("nan")))
        return pd.DataFrame(
            rows, columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
        )


def model_instances() -> list[BVARMinnesota]:
    keys = list(BVARMinnesota.PARAM_GRID.keys())
    return [
        BVARMinnesota(**dict(zip(keys, combo)))
        for combo in itertools.product(*[BVARMinnesota.PARAM_GRID[k] for k in keys])
    ]
