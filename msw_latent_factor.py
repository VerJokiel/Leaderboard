"""
Müller-Stock-Watson (REStat 2022) — simplified cross-country latent factor model.

Description:
    Per-variable dynamic factor model across countries. Each country's
    transformed series is decomposed into a country mean, a common
    component driven by K global factors, and a persistent idiosyncratic
    component. Captures shared global shocks (e.g. world growth cycles)
    that single-country models miss. Point-forecast simplification of
    MSW: PCA + OLS instead of hierarchical Bayesian MCMC; no cluster
    factors, no low-frequency filtering.

Target variable (y):
    x_{i,t} = T_v(y)_{i,t}, canonical transformed space; forecasts are
    back-transformed with transformations.inverse.

Feature matrix:
    Z (T years × N countries), each column demeaned by its country mean
    mu_i; missing cells set to 0 (= mean imputation). Data up to cutoff only.

Model:
    x_{i,t} = mu_i + lambda_i' F_t + e_{i,t}
    F_{k,t} = sum_{l=1..p_F} phi_{k,l} F_{k,t-l} + eta_{k,t}
    e_{i,t} = c_i + rho_i e_{i,t-1} + eps_{i,t}

Estimation:
    1. Demean per country, impute missing with 0 → Z.
    2. PCA (SVD) on Z → factors F (T×K), loadings Lambda (N×K).
    3. OLS AR(p_F) on each factor separately.
    4. Idiosyncratic residual e_{i,t} = x_{i,t} - mu_i - lambda_i' F_t
       (observed cells only); OLS AR(1) with intercept per country.

Forecast (h-step, iterated):
    F_{T+h} from the factor ARs, e_{i,T+h} from the idiosyncratic AR(1),
    x_{i,T+h} = mu_i + lambda_i' F_{T+h} + e_{i,T+h}, then back-transform.

Hyperparameters (PARAM_GRID):
    K    : number of global factors.
    p_F  : AR order on each factor.
    Implicit: idiosyncratic AR order = 1, mean imputation, no column scaling.

Reference:
    Müller, U. K., Stock, J. H., Watson, M. W. (2022). "An Econometric Model
    of International Long-Run Growth Dynamics for Long-Horizon Forecasting."
    Review of Economics and Statistics.
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from var_classification import family, transformation
from transformations import forward as _forward, inverse as _inverse

def _contiguous_blocks(s: pd.Series) -> list[pd.Series]:
    """Split a year-indexed series (no NaN) into runs of consecutive years."""
    yrs = s.index.to_numpy()
    breaks = np.where(np.diff(yrs) != 1)[0] + 1
    starts, ends = np.r_[0, breaks], np.r_[breaks, len(s)]
    return [s.iloc[a:b] for a, b in zip(starts, ends)]

class MSWLatentFactor:
    family = "DFM"
    segmentation = "panel_global"
    variables = None
    version = "0.1"
    PARAM_GRID = {
        "K":                  [1, 2, 3],
        "p_factor":           [1, 2],
        "train_window_years": [None, 50, 20, 10],
    }

    MIN_COUNTRIES = 5      # countries needed per year / per variable
    MAX_STALE_YEARS = 5    # skip countries whose last obs is older than cutoff - this

    def __init__(self, K: int = 2, p_factor: int = 1, train_window_years: int | None = None):
        self.K = int(K)
        self.p_factor = int(p_factor)
        self.train_window_years = train_window_years
        self.min_history = max(10, self.p_factor + 5)
        win = "" if train_window_years is None else f"_w{train_window_years}"
        self.name = f"msw_K{self.K}_pF{self.p_factor}{win}"
        self.inputs = (
            f"Per variable v in T_v space: wide matrix (years x countries), demeaned "
            f"per country, missing -> 0; PCA -> {self.K} global factors; AR({self.p_factor}) "
            f"per factor; AR(1) per-country idiosyncratic. Simplified MSW (REStat 2022)."
        )

    # ---- AR helpers ----

    def _ar_fit(self, y: np.ndarray, p: int) -> tuple[float, np.ndarray] | None:
        """Fit AR(p) via OLS. Returns (intercept, coef_vec) or None."""
        y = np.asarray(y, dtype=float)
        X_rows, t_rows = [], []
        for t in range(p, len(y)):
            if np.all(np.isfinite(y[t - p: t + 1])):
                X_rows.append([1.0] + list(y[t - p: t][::-1]))
                t_rows.append(y[t])
        if len(y) < p + 3:
            return None
            
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        except Exception:
            return None
        return float(coef[0]), coef[1:].astype(float)
    @staticmethod
    def _ar_forecast(self, intercept: float, ar_coef: np.ndarray,
                     init: list[float], h_max: int) -> list[float]:
        """h-step recursive forecast given last p observations (most recent first)."""
       lags, out = list(init), []
        for _ in range(h_max):
            y_hat = intercept + float(np.dot(ar_coef, lags[:p]))
            out.append(y_hat)
            lags = [y_hat] + lags[:-1]
        return np.asarray(out)

   def fit(self, panel: pd.DataFrame, cutoff_year: int) -> "MSWLatentFactor":
        self._per_var = {}
        panel = panel[panel["year"] <= cutoff_year]                    # no look-ahead
        if self.train_window_years is not None:
            panel = panel[panel["year"] >= cutoff_year - self.train_window_years + 1]
        for var in panel["variable"].unique():
            if family(var) == "skip":
                continue
            res = self._fit_one_variable(panel, var, cutoff_year)
            if res is not None:
                self._per_var[var] = res
        return self

    def _fit_one_variable(self, panel: pd.DataFrame, var: str) -> dict | None:
        tmode = transformation(var)
        sub = panel[panel["variable"] == var]
        if sub.empty:
            return None

        # Build wide transformed matrix: rows = year, cols = country
        wide_levels = sub.pivot_table(
            index="year", columns="ISO3", values="value", aggfunc="last"
        )

        # Transform each country's column independently
    t_cols, ctx, last_year = {}, {}, {}
        for iso in wide.columns:
            s = wide[iso].dropna().astype(float)
            if len(s) < self.min_history:
                continue
            if int(s.index[-1]) < cutoff_year - self.MAX_STALE_YEARS:
                continue
            pieces, c_last = [], None
            for b in _contiguous_blocks(s):
                tv, c = _forward(b.to_numpy(), tmode)
                tv = np.asarray(tv, dtype=float)
                if len(tv) == 0:
                    continue
                pieces.append(pd.Series(tv, index=b.index[-len(tv):]))
                c_last = c
            if not pieces:
                continue
            ts = pd.concat(pieces)
            if ts.notna().sum() < self.min_history - 1:
                continue
            t_cols[iso], ctx[iso], last_year[iso] = ts, c_last, int(ts.index[-1])

        if len(t_cols) < self.MIN_COUNTRIES
            return None

        # 2. wide matrix on a contiguous year range; trim leading thin years only
        T_wide = pd.DataFrame(t_cols).sort_index()
        T_wide = T_wide.reindex(range(int(T_wide.index.min()), int(T_wide.index.max()) + 1))
        ok = T_wide.notna().sum(axis=1) >= self.MIN_COUNTRIES
        if not ok.any():
            return None
        T_wide = T_wide.loc[ok.idxmax():]
        T_wide = T_wide.loc[:, T_wide.notna().sum() >= 3]
        if len(T_wide) < self.min_history or T_wide.shape[1] < self.MIN_COUNTRIES:
            return None
 
        mu = T_wide.mean()
        D = T_wide - mu                                  # NaN where missing
        Z = D.fillna(0.0).to_numpy()
 
        # 3. PCA via SVD
        K_eff = int(min(self.K, Z.shape[1], Z.shape[0] - 1))
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        F = U[:, :K_eff] * S[:K_eff]                     # (years, K)
        L = Vt[:K_eff].T                                 # (countries, K)
        F_df = pd.DataFrame(F, index=T_wide.index)
 
        # 4a. factor AR(p_F)
        factor_models = []
        for k in range(K_eff):
            res = self._ar_fit(F[:, k], self.p_factor)
            factor_models.append(res if res is not None
                                 else (float(F[:, k].mean()), np.zeros(self.p_factor)))

        # Per-country idiosyncratic series: e = T - mean - L F
        e_wide = T_wide.sub(col_means, axis=1) - (F @ L.T)   # (years × countries)

         # 4b. idiosyncratic AR(1) per country
        E = D - pd.DataFrame(F @ L.T, index=T_wide.index, columns=T_wide.columns)
        idio_models, last_e = {}, {}
        for iso in T_wide.columns:
            idio_models[iso] = self._ar_fit(E[iso].to_numpy(), 1)
            e_last = E.at[last_year[iso], iso]
            last_e[iso] = float(e_last) if np.isfinite(e_last) else 0.0
 
        return {
            "tmode": tmode,
            "K": K_eff,
            "F": F_df,
            "T_end": int(T_wide.index[-1]),
            "factor_models": factor_models,
            "mu": mu.to_dict(),
            "loadings": {iso: L[j] for j, iso in enumerate(T_wide.columns)},
            "idio_models": idio_models,
            "last_e": last_e,
            "last_year": {iso: last_year[iso] for iso in T_wide.columns},
            "ctx": {iso: ctx[iso] for iso in T_wide.columns},
        }

# -------------------------------------------------------------- predict 
    def predict(self, panel: pd.DataFrame, cutoff_year: int, horizons: list[int]) -> pd.DataFrame:
        if not self._per_var:
            raise RuntimeError("Call fit() before predict() (or nothing could be fitted).")
        rows, h_max = [], max(horizons)
        for var, state in self._per_var.items():
            # factor path: in-sample up to T_end, AR forecasts until cutoff + h_max
            F_in, T_end = st["F"], st["T_end"]
            n_fc = cutoff_year + h_max - T_end
            if n_fc > 0:
                fc_cols = []
                for k, (c, coef) in enumerate(st["factor_models"]):
                    init = list(F_in[k].to_numpy()[-self.p_factor:][::-1])
                    fc_cols.append(self._ar_forecast(c, coef, init, n_fc))
                F_fc = pd.DataFrame(np.column_stack(fc_cols),
                                    index=range(T_end + 1, T_end + 1 + n_fc))
                F_all = pd.concat([F_in, F_fc])
            else:
                F_all = F_in
 
            for iso, lam in st["loadings"].items():
                y0 = st["last_year"][iso]
                years = list(range(y0 + 1, cutoff_year + h_max + 1))
                idio = st["idio_models"][iso]
                e_path = (np.zeros(len(years)) if idio is None
                          else self._ar_forecast(idio[0], idio[1], [st["last_e"][iso]], len(years)))
                t_steps = st["mu"][iso] + F_all.loc[years].to_numpy() @ lam + e_path
                levels = np.asarray(_inverse(list(t_steps), st["ctx"][iso], mode=st["tmode"]),
                                    dtype=float)
                for h in horizons:
                    yhat = levels[cutoff_year + h - y0 - 1]
                    if np.isfinite(yhat):
                        rows.append((iso, cutoff_year + h, var, h, float(yhat), np.nan, np.nan))
        return pd.DataFrame(
            rows, columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
        )
 
 
def model_instances() -> list[MSWLatentFactor]:
    keys = list(MSWLatentFactor.PARAM_GRID)
    return [MSWLatentFactor(**dict(zip(keys, combo)))
            for combo in itertools.product(*MSWLatentFactor.PARAM_GRID.values())]
