"""
Multi-variable Dynamic Factor Model (DFM) — full design cross-product.

Description:
Diffusion-index DFM (Stock & Watson, 2002) with joint factors extracted
from a stacked cross-country x cross-variable panel. It extends
multivar_dfm.py by enumerating the same design dimensions as
pca_country_v2.py, except t-stat screening, which is infeasible with
thousands of panel-wide candidates per target.

Model (transformed space, cell (i,v), year t):
    x_{i,v,t} = mu_{i,v} + lambda_{i,v}' F_t + e_{i,v,t}
    F_{k,t}   = c_k + phi_k F_{k,t-1} + eta_{k,t}          k = 1..K
    e_{i,v,t} = rho_{i,v} e_{i,v,t-1} + eps_{i,v,t}
    [optional] G_t = c_G + phi_G G_{t-1} + xi_t            aggregate of target var

Design: grid (2 x 5 x 2 x 1 = 20 instances):
    input_mode   in {curated, all}
                   curated: rGDP, CPI, strate, USDfx, unemp
                   all:     every canonical indicator with adequate
                            coverage up to the cutoff
    factor_aug   in {none, region_eq, region_gdp, global_eq, global_gdp}
    K            in {1, 2}
    tstat_screen in {False}

Target variable:
y_{i,v*,t}: the target variable v* for country i, modelled in its
canonical FRED-MD-style transformed space x = T_{v*}(y)
(level / diff / log-diff, according to the variable's tcode).
Forecasts are reported in [transformed | level] space; for log-diff
targets, levels are recovered as
y_{T+h} = y_T * exp(sum_{s<=h} xhat_{T+s}).

Estimation: (all steps use data t <= cutoff T only)
1. Transform: x_{i,v,t} = T_v(y_{i,v,t}).
2. Panel:     wide X (T x N|V|); cell means mu_{i,v} from observed data;
              missing -> mu; centre: Xtilde = X - mu.
3. Factors:   PCA/SVD of Xtilde -> F = sqrt(T) U_{1:K}, normalised
              F'F/T = I_K.
4. Loadings:  Lambda = Xtilde' F / T  (OLS of each cell on F).
5. Idio:      e_{i,v,t} = x_{i,v,t} - mu_{i,v} - lambda_{i,v}' F_t.
6. Dynamics:  OLS AR(1) per factor F_k (with intercept) and per
              idiosyncratic e_{i,v} (no intercept); |phi|, |rho| < 1.
7. Aggregate (factor_aug != none), via global_features.precompute:
              G_t = sum_{j in S} w_{j,t} x_{j,v*,t}
              S = region(i) or all countries;
              w = 1/|S| (eq) or lagged GDP shares (gdp);
              OLS AR(1) on G_t.

Forecast:  (h = 1..H, iterated from origin T)
    Fhat_{k,T+h} = c_k (1 - phi_k^h)/(1 - phi_k) + phi_k^h F_{k,T}
    ehat_{T+h}   = rho^h e_T
    delta_{T+h}  = Ghat_{T+h} - mean(G)        (0 if factor_aug = none)
    xhat_{i,v*,T+h} = mu_{i,v*} + delta_{T+h}
                      + lambda_{i,v*}' Fhat_{T+h} + ehat_{i,v*,T+h}
then back-transform with T_{v*}^{-1}.

REFERENCES
----------
Stock, J. & Watson, M. (2002). Macroeconomic forecasting using
    diffusion indexes. JASA 97(458).
Bai, J. & Ng, S. (2002). Determining the number of factors in
    approximate factor models. Econometrica 70(1).
McCracken, M. & Ng, S. (2016). FRED-MD. JBES 34(4)  [transformation codes].
"""

from __future__ import annotations
import itertools
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from var_classification import family, transformation
from transformations import forward as _forward, inverse as _inverse
from input_registry import all_macro_inputs
from global_features import precompute as _precompute_factors, feature_columns


log = logging.getLogger(__name__)

_CURATED_BLOCK = ("rGDP", "CPI", "strate", "USDfx", "unemp")

_FACTOR_AUG_KIND = {
    "none":       None,
    "region_eq":  "region_avg",
    "region_gdp": "region_wavg",
    "global_eq":  "global_avg",
    "global_gdp": "global_wavg",
}

_OUT_COLS = ["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]


# --------------------------------------------------------------------------
# AR helpers
# --------------------------------------------------------------------------
def _ols_ar1(y: np.ndarray, x_exog: np.ndarray | None = None,
             max_root: float = 0.98):
    """OLS of y_t on [1, y_{t-1}, (x_{t-1})]. Returns (a, rho, gamma) or None."""
    y = np.asarray(y, float)
    if x_exog is None:
        X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    else:
        X = np.column_stack([np.ones(len(y) - 1), y[:-1], np.asarray(x_exog, float)[:-1]])
    yy = y[1:]
    ok = np.isfinite(yy) & np.isfinite(X).all(axis=1)
    if ok.sum() < X.shape[1] + 3:
        return None
    coef, *_ = np.linalg.lstsq(X[ok], yy[ok], rcond=None)
    a, rho = float(coef[0]), float(np.clip(coef[1], -max_root, max_root))
    gamma = float(coef[2]) if x_exog is not None else 0.0
    return a, rho, gamma
 
 
def _iterate_ar1(a: float, rho: float, last: float, steps: int) -> np.ndarray:
    out, cur = np.empty(steps), last
    for s in range(steps):
        cur = a + rho * cur
        out[s] = cur
    return out


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class MultivarDFMv2:
    family = "DFM"
    segmentation = "panel_global"
    variables = None
    version = "0.3"

    PARAM_GRID = {
        "input_mode":         ["curated", "all"],
        "factor_aug":         list(_FACTOR_AUG_KIND),
        "K":                  [1, 2],
        "tstat_screen":       [False],
        "train_window_years": [None, 50, 20, 10],
    }

    def __init__(self, input_mode: str = "curated", factor_aug: str = "none",
                 K: int = 1, tstat_screen: bool = False,
                 train_window_years: int | None = None,
                 standardize: bool = True,
                 min_cells_per_year: int = 20,
                 min_col_coverage: float = 0.5,
                 max_ar_root: float = 0.98):
        if input_mode not in ("curated", "all"):
            raise ValueError(f"input_mode must be 'curated' or 'all', got {input_mode!r}")
        if factor_aug not in _FACTOR_AUG_KIND:
            raise ValueError(f"unknown factor_aug {factor_aug!r}")
        if tstat_screen:
            raise NotImplementedError("tstat_screen is not supported for the panel-wide DFM")
            
        self.input_mode = input_mode
        self.factor_aug = factor_aug
        self.K = int(K)
        self.tstat_screen = False
        self.train_window_years = train_window_years
        self.standardize = bool(standardize)
        self.min_cells_per_year = int(min_cells_per_year)
        self.min_col_coverage = float(min_col_coverage)
        self.max_ar_root = float(max_ar_root)
        self.min_history = max(10, self.K + 5)
        self._state: dict | None = None
 
        win = "" if train_window_years is None else f"_w{train_window_years}"
        self.name = (f"multivar_dfm_v2_in_{input_mode}_fac_{factor_aug}"
                     f"_nots_K{self.K}{win}")
        self.inputs = (
            f"Joint PCA factors over (country x variable) cells "
            f"[{'curated core block' if input_mode == 'curated' else 'all macro indicators'}], "
            f"K={self.K}, standardize={self.standardize}, "
            f"window={train_window_years or 'expanding'}; factor AR(1); "
            f"idiosyncratic {'ARX(1) on lagged ' + factor_aug + ' aggregate' if factor_aug != 'none' else 'AR(1)'}. "
            f"Refs: Stock-Watson (2002), Bai-Ng (2002)."
        )
    # ---- data preparation ----
    def _variable_block(self, panel: pd.DataFrame) -> list[str]:
        block = _CURATED_BLOCK if self.input_mode == "curated" else all_macro_inputs()
        present = set(panel["variable"].unique())
        return [v for v in block if v in present and family(v) != "skip"]
 
    def _restrict(self, panel: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
        p = panel[panel["year"].astype(int) <= cutoff_year]          # no look-ahead
        if self.train_window_years is not None:
            p = p[p["year"].astype(int) > cutoff_year - self.train_window_years]
        return p
 
    def _agg_series(self, factors_df: pd.DataFrame, var: str) -> pd.Series | None:
        kind = _FACTOR_AUG_KIND[self.factor_aug]
        if kind is None or factors_df is None or factors_df.empty:
            return None
        for c in feature_columns(factors_df, kind):
            if c.endswith(f"_{var}"):
                s = factors_df[c].astype(float)
                return s if s.notna().sum() >= 6 else None
        return None

    # ---- main API ----

    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None:
        self._state = None
        self._cutoff = int(cutoff_year)
        block = self._variable_block(panel)
        if not block:
            log.warning("%s: no variables of the block present", self.name)
            return
        panel_c = self._restrict(panel, cutoff_year)
        sub = panel_c[panel_c["variable"].isin(block)]   # data <= cutoff only
        if sub.empty:
            return

        # 1) transform each cell
        t_cols, ctx = {}, {}
        for (iso, v), g in sub.groupby(["ISO3", "variable"], observed=True):
            g = g.dropna(subset=["value"]).sort_values("year")
            if len(g) < self.min_history:
                continue
            try:
                tvals, c = _forward(g["value"].astype(float).values, transformation(v))
            except Exception as exc:                          # noqa: BLE001
                log.debug("transform failed for %s/%s: %s", iso, v, exc)
                continue
            tvals = np.asarray(tvals, float)
            yrs = g["year"].astype(int).values[-len(tvals):]
            t_cols[(iso, v)] = pd.Series(tvals, index=yrs)
            ctx[(iso, v)] = c
        if len(t_cols) < 10:
            return

        # 2) wide panel with coverage filters
        wide = pd.concat(t_cols, axis=1).sort_index()
        wide = wide.replace([np.inf, -np.inf], np.nan)
        wide = wide.loc[wide.notna().sum(axis=1) >= self.min_cells_per_year]
        if len(wide) < self.min_history:
            return
        wide = wide.loc[:, wide.notna().mean(axis=0) >= self.min_col_coverage]
        if wide.shape[1] < 10:
            return
 
        # 3) impute, scale, PCA
        mu = wide.mean(axis=0)
        sd = wide.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0) if self.standardize \
            else pd.Series(1.0, index=wide.columns)
        Z = ((wide - mu) / sd).fillna(0.0).values
        K_eff = min(self.K, *Z.shape)
        pca = PCA(n_components=K_eff)
        F = pca.fit_transform(Z)                               # (T, K)
        L = pca.components_.T                                  # (cells, K)
 
        common = pd.DataFrame((F @ L.T) * sd.values, index=wide.index, columns=wide.columns)
        e_wide = wide - mu - common                            # NaN where unobserved
 
        # 4) Factor dynamics AR(1)
        factor_models = []
        for k in range(K_eff):
            res = _ols_ar1(F[:, k], max_root=self.max_ar_root)
            factor_models.append(res[:2] if res else (0.0, 0.0))

        # 5) aggregate factors (optional)
        agg_by_var: dict[str, pd.Series] = {}
        agg_models: dict[str, tuple[float, float]] = {}
        if self.factor_aug != "none":
            try:
                fdf = _precompute_factors(panel_c, cutoff_year,
                                          variables=list(block) + ["nGDP_USD"])
            except Exception as exc:                          # noqa: BLE001
                log.warning("%s: aggregate precompute failed (%s)", self.name, exc)
                fdf = pd.DataFrame()
            for v in block:
                s = self._agg_series(fdf, v)
                if s is None:
                    continue
                s = s[s.index.astype(int) <= cutoff_year].dropna()
                res = _ols_ar1(s.values, max_root=self.max_ar_root)
                if res:
                    agg_by_var[v] = s
                    agg_models[v] = res[:2]
                    
        # 6) idiosyncratic AR(1) / ARX(1) per cell (on contiguous years)
        idio = {}
        for key in wide.columns:
            e = e_wide[key].dropna()
            if len(e) < 5:
                continue
            e = e.reindex(range(e.index.min(), e.index.max() + 1))
            x = agg_by_var.get(key[1])
            x = x.reindex(e.index).values if x is not None else None
            res = _ols_ar1(e.values, x, self.max_ar_root) if x is not None else None
            res = res or _ols_ar1(e.values, None, self.max_ar_root)
            if res is None:
                continue
            last_year = int(e.last_valid_index())
            idio[key] = {"a": res[0], "rho": res[1], "gamma": res[2],
                         "last_e": float(e[last_year]), "last_year": last_year}
 
        self._state = {
            "K": K_eff, "cols": list(wide.columns), "years": wide.index.values,
            "F": F, "L": dict(zip(wide.columns, L)), "mu": mu.to_dict(),
            "sd": sd.to_dict(), "factor_models": factor_models,
            "idio": idio, "ctx": ctx,
            "agg": agg_by_var, "agg_models": agg_models,
            "obs_last": {k: int(wide[k].last_valid_index()) for k in wide.columns},
        }
 
# predict
    def predict(self, panel: pd.DataFrame, cutoff_year: int,
                horizons: list[int]) -> pd.DataFrame:
        st = self._state
        if self._state is None:
            return pd.DataFrame(columns=_OUT_COLS)
        cutoff_year = int(cutoff_year)
        h_max = int(max(horizons))
        T_end = int(st["years"][-1])
        last_target = cutoff_year + h_max
 
        # factor values by year: in-sample up to T_end, AR(1) forecasts after
        n_fc = last_target - T_end
        F_by_year = {int(y): st["F"][i] for i, y in enumerate(st["years"])}
        if n_fc > 0:
            paths = np.column_stack([
                _iterate_ar1(a, r, st["F"][-1, k], n_fc)
                for k, (a, r) in enumerate(st["factor_models"])
            ])
            for s in range(n_fc):
                F_by_year[T_end + s + 1] = paths[s]
 
        # aggregate values by year (observed, then AR(1) forecast)
        agg_by_year: dict[str, dict[int, float]] = {}
        for v, s in st["agg"].items():
            d = {int(y): float(val) for y, val in s.items()}
            a, r = st["agg_models"][v]
            last_y = max(d)
            for i, val in enumerate(_iterate_ar1(a, r, d[last_y], max(0, last_target - last_y))):
                d[last_y + i + 1] = float(val)
            agg_by_year[v] = d
 
        rows = []
        for key in st["cols"]:
            iso, var = key
            if key not in st["ctx"]:
                continue
            last_obs = st["obs_last"][key]
            if last_obs > cutoff_year or cutoff_year - last_obs > 3:
                continue
            years = range(last_obs + 1, last_target + 1)
            mu, sd, lam = st["mu"][key], st["sd"][key], st["L"][key]
 
            # idiosyncratic path from the cell's own last residual
            m = st["idio"].get(key)
            e_cur = m["last_e"] if m else 0.0
            e_year = m["last_year"] if m else last_obs
            agg = agg_by_year.get(var, {})
            e_path = {}
            for y in range(e_year + 1, last_target + 1):
                if m:
                    e_cur = m["a"] + m["rho"] * e_cur + m["gamma"] * agg.get(y - 1, 0.0)
                e_path[y] = e_cur
 
            t_steps = [mu + sd * float(lam @ F_by_year[y]) + e_path.get(y, 0.0)
                       for y in years]
            try:
                levels = _inverse(t_steps, st["ctx"][key], mode=transformation(var))
            except Exception as exc:                          # noqa: BLE001
                log.debug("inverse failed for %s/%s: %s", iso, var, exc)
                continue
            offset = cutoff_year - last_obs                   # steps before origin
            for h in horizons:
                yhat = float(levels[offset + h - 1])
                if np.isfinite(yhat):
                    rows.append((iso, cutoff_year + h, var, h, yhat, np.nan, np.nan))
        return pd.DataFrame(rows, columns=_OUT_COLS)


def model_instances() -> list[MultivarDFMv2]:
    keys = list(MultivarDFMv2.PARAM_GRID.keys())
    return [
        MultivarDFMv2(**dict(zip(keys, combo)))
        for combo in itertools.product(*[MultivarDFMv2.PARAM_GRID[k] for k in keys])
    ]
