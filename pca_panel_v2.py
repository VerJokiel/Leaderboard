""" Panel-wide PCA dynamic factor model on diffusion indexes (Stock-Watson 2002, JASA).

world-business cycles that move all countries and all macro variables together:
forecast = series mean + loading × (factor forecast) +  own idiosyncratic AR(1) forecast 
            + optional regional/global shift

Design:
  1. Variable choice: (input_mode) 
     ='curated' uses (rGDP, CPI, strate, USDfx, unemp, M2, infl, REER); 
     ='all' uses macro indicator present in (that input_registry.all_macro_inputs().

  2. Transform each (country, variable) series with (transformations.forward),
     e.g. log-differences for GDP and levels for rates.
     
  3. Build wide matrix: M= [year × ISO3 variable], e.g. (DEU,infl),
     Drop if less than 50% coverage.
  
  4. Run Principal Component Aanalysis 
     Fill missing values with the column mean, then extract K = 1 or 2 factors. 
     Each column gets a loading on each factor.
  
  5. Fit the Dynamics
     AR(1) for each factor and an AR(1) on each column's residual (idiosyncratic part). 
  
  6. Optional aggregate augmentation (factor_aug)
     Regress each residual on the lagged regional or global average of the same variable, 
     equal-weighted or weighted by nominal GDP in USD. This gives one coefficient γ per series:
     factor_aug ∈ {none, region_eq, region_gdp, global_eq, global_gdp}
  
  7. Predict, iterate AR(1)s h steps ahead, reconstruct the transformed value and back-transform to levels.
     Outpput: ISo3, year, variable, horizon, point, lower, upper

     model_instances() for the full grid:
     2 input modes x 5 augmentations x 2 values of K x training windows = 80 variants

  Cross-product:
  input_mode   ∈ {curated, all}                                         2 options
  factor_aug   ∈ {none, region_eq, region_gdp, global_eq, global_gdp}   5 options
  K            ∈ {1, 2}                                                 2 options
  tstat_screen ∈ {False}     (panel-wide screening impractical)         1 option

                                                                      = 20 instances.
"""


# Size issue in construction (15-20 countries):
# The 12-country × 4-variable World Bank sample in the toolkit has only 48 columns. 
# It can never pass the 50-cells-per-year filter, so the model silently returns no forecasts.
# FRED-MD is US-only and monthly, so it does not fit this model directly. 
# It would have to be annualised with ISO3 = "USA"; it suits the country-level FactorDI model better.


# back-test on real data (worldbank, imf, fred. The best setting of your model is the 20-year window, 
# which beats the no-change forecast for most countries on inflation, CPI, money and real GDP. 
# It does not beat it on exchange rates, interest rates or unemployment.

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from var_classification import family, transformation # variables names must match these known under it 
from transformations import forward as _forward, inverse as _inverse # variables names must match these known under it 
from input_registry import all_macro_inputs
from global_features import precompute as _precompute_factors, feature_columns

log = logging.getLogger(__name__)

# Curated-mode block: a wider economically motivated set than
# multivar_dfm_v2's core 5 (panel-wide PCA can afford more dimensions).
_CURATED_BLOCK = ("rGDP", "CPI", "infl", "strate", "ltrate", "USDfx",
                  "unemp", "M2", "REER")

_FACTOR_AUG_KIND = {
    "none":       None,
    "region_eq":  "region_avg",
    "region_gdp": "region_wavg",
    "global_eq":  "global_avg",
    "global_gdp": "global_wavg",   }

_OUT_COLS = ["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]

def _em_pca(Z: np.ndarray, K: int, n_iter: int = 100, tol: float = 1e-6):

  """PCA on a standardised matrix with NaNs (EM algorithm).
    Missing cells start at 0 (the column mean after standardising) and are
    replaced by the rank-K fit until convergence. Observed cells stay fixed.
    Returns factors F (T x K) and loadings L (N x K).
  """
    mask = np.isnan(Z)
    X = np.where(mask, 0.0, Z)
    F = L = None
    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        F, L = U[:, :K] * s[:K], Vt[:K].T
        if not mask.any():
            break
        X_new = np.where(mask, F @ L.T, Z)
        done = np.max(np.abs(X_new - X)) < tol
        X = X_new
        if done:
            break
    return F, L

class PanelPCAv2:
    family = "DFM"
    segmentation = "panel_global"
    variables = None
    version = "0.3"

    PARAM_GRID = {
        "input_mode":         ["curated", "all"],
        "factor_aug":         ["none", "region_eq", "region_gdp", "global_eq", "global_gdp"],
        "K":                  [1, 2],
        "tstat_screen":       [False],
        "train_window_years": [None, 50, 20, 10],
    }

  def __init__(
        self,
        input_mode: str = "curated",
        factor_aug: str = "none",
        K: int = 1,
        train_window_years: int | None = None,
        min_cells_per_year: int | float = 0.5,
        min_col_coverage: float = 0.5,
        min_cols: int = 10,
        n_sims: int = 200,
        max_staleness: int = 3,
        seed: int = 0,
    ):
        if factor_aug not in _FACTOR_AUG_KIND:
            raise ValueError(f"unknown factor_aug {factor_aug!r}")
        self.input_mode = input_mode
        self.factor_aug = factor_aug
        self.K = int(K)
        self.train_window_years = (None if train_window_years is None
                                   else int(train_window_years))
        self.min_cells_per_year = min_cells_per_year
        self.min_col_coverage = float(min_col_coverage)
        self.min_cols = int(min_cols)
        self.n_sims = int(n_sims)
        self.max_staleness = int(max_staleness)
        self.seed = int(seed)
 
        # A short window must not be blocked by the history requirement.
        cap = 15 if self.train_window_years is None else min(15, self.train_window_years - 2)
        self.min_history = max(self.K + 6, cap)
 
        win_tag = "" if self.train_window_years is None else f"_w{self.train_window_years}"
        self.name = (f"pca_panel_v2_in_{self.input_mode}_fac_{self.factor_aug}"
                     f"_nots_K{self.K}{win_tag}")
        self.inputs = (
            f"Panel-wide PCA. Stack (ISO3, variable) columns where variable "
            f"ranges over input_mode='{self.input_mode}' "
            f"({'curated wider block' if self.input_mode == 'curated' else 'all macro indicators'}) "
            f"and ISO3 spans every country with adequate coverage. Data <= cutoff"
            f"{'' if self.train_window_years is None else f', last {self.train_window_years} years'}. "
            f"Canonical transformed space, standardised, EM-PCA → {self.K} factor(s). "
            f"AR(1) on each factor + idiosyncratic AR(1) per (i, v). "
            f"factor_aug='{self.factor_aug}': lag-1 aggregate of each variable "
            f"(kind='{_FACTOR_AUG_KIND[self.factor_aug] or 'none'}'), forecast by AR(1). "
            f"Forecasts aligned by calendar year; 80% intervals from {self.n_sims} simulations. "
            f"References: Stock-Watson (2002, JASA); Bai-Ng (2002, Econometrica)."
        )
        self._state = None
        self.fit_status = "not fitted"

    # ---- helpers ----
   def _fail(self, reason: str) -> None:
        self._state = None
        self.fit_status = reason
        log.warning("%s: no forecasts (%s)", self.name, reason) # if forecast was unsuccessful 

    """OLS AR(p) with intercept → (intercept, coefs, residual sd) or None."""
    def _ar_fit(y, p: int = 1):
        y = np.asarray(y, float)
        if len(y) < p + 3:
            return None
        X = np.array([[1.0] + list(y[t - p:t][::-1]) for t in range(p, len(y))])
        yv = y[p:]
        try:
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        except np.linalg.LinAlgError:
            return None
        resid = yv - X @ coef
        sigma = float(np.sqrt(resid @ resid / max(len(yv) - X.shape[1], 1)))
        return float(coef[0]), coef[1:].astype(float), sigma

    def _ar_forecast(self, intercept, ar_coef, init, n):
        out, lags, p = [], list(init), len(ar_coef)
        for _ in range(n):
            y_hat = intercept + float(np.dot(ar_coef, lags[:p]))
            out.append(y_hat); lags = [y_hat] + lags[:-1]
        return out

        """Simulated AR(1) paths, shape (n_sims, n)."""
    def _ar_sim(intercept, ar_coef, init, n, sigma, rng, n_sims):
        out = np.empty((n_sims, n))
        lag = np.full(n_sims, float(init[0]))
        a = float(ar_coef[0]) if len(ar_coef) else 0.0
        for t in range(n):
            lag = intercept + a * lag + rng.normal(0.0, sigma, n_sims)
            out[:, t] = lag
        return out

    def _variable_set(self, panel: pd.DataFrame) -> list[str]:
        if self.input_mode == "curated":
            block = _CURATED_BLOCK
        else:
            block = all_macro_inputs()
        present = set(panel["variable"].unique())
        return [v for v in block if v in present and family(v) != "skip"]

 def _variable_set(self, panel: pd.DataFrame) -> list[str]:
        block = _CURATED_BLOCK if self.input_mode == "curated" else all_macro_inputs()
        present = set(panel["variable"].unique())
        return [v for v in block if v in present and family(v) != "skip"]
 
    def _agg_col_for_var(self, factors_df: pd.DataFrame, var: str) -> str | None:
        kind = _FACTOR_AUG_KIND.get(self.factor_aug)
        if kind is None or factors_df.empty:
            return None
        for c in feature_columns(factors_df, kind):
            if c.endswith(f"_{var}"):
                return c
        return None

    # ---- main API ----

    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None:
        self._state = None
        variables_in = self._variable_set(panel)
       if not variables_in:
            return self._fail("none of the input variables is in the panel")
                   
 # Only information available at the cutoff (no look-ahead).
        past = panel[panel["year"] <= cutoff_year]
        sub = past[past["variable"].isin(variables_in)]
        if self.train_window_years is not None:
            sub = sub[sub["year"] > cutoff_year - self.train_window_years]
        if sub.empty:
            return self._fail("no data in the training window")

        # Per (ISO3, var) transform
        t_cols: dict[tuple[str, str], pd.Series] = {}
        ctx_per_pair: dict[tuple[str, str], dict] = {}
        for (iso, v), g in sub.groupby(["ISO3", "variable"], observed=True):
            vals = g.sort_values("year")["value"].dropna().astype(float).values
            yrs = g.sort_values("year")["year"].astype(int).values
            if len(vals) < self.min_history:
                continue           
            if cutoff_year - yrs[-1] > self.max_staleness:
                continue                      # too old to forecast from
            tvals, ctx = _forward(vals, transformation(v))
            tvals = np.asarray(tvals, float)
            if len(tvals) == 0:
                continue
            t_cols[(iso, v)] = pd.Series(tvals, index=yrs[-len(tvals):])
            ctx_per_pair[(iso, v)] = ctx
        if len(t_cols) < self.min_cols:
          return self._fail(f"{len(t_cols)} usable series, need {self.min_cols}")
                    
        wide = pd.concat(t_cols, axis=1).sort_index()
        thr = self.min_cells_per_year
        if isinstance(thr, float) and thr < 1:
            thr = int(np.ceil(thr * wide.shape[1]))
        wide = wide.loc[wide.notna().sum(axis=1) >= thr]
        wide = wide.loc[:, wide.notna().mean(axis=0) >= self.min_col_coverage]
        if len(wide) < self.min_history:
          return self._fail(f"{len(wide)} years with >= {thr} observed cells, "
                              f"need {self.min_history}")
        if wide.shape[1] < self.min_cols:
            return self._fail(f"{wide.shape[1]} columns after filters, need {self.min_cols}")       

           # Standardise + EM-PCA.
        col_means = wide.mean(axis=0)
        col_sd = wide.std(axis=0, ddof=0).fillna(1.0).replace(0.0, 1.0)
        Z = ((wide - col_means) / col_sd).values
        try:
            F, L = _em_pca(Z, min(self.K, *Z.shape))
        except (np.linalg.LinAlgError, ValueError) as e:
            return self._fail(f"EM-PCA failed: {e}")
        K_eff = F.shape[1]
        col_keys = list(wide.columns)

        factor_models = []
        for k in range(K_eff):
            res = self._ar_fit(F[:, k], 1)
            if res is None:
                factor_models.append((float(F[:, k].mean()), np.zeros(1)))
            else:
                factor_models.append(res)
                        
             common = pd.DataFrame(F @ L.T, index=wide.index, columns=wide.columns) * col_sd
        e_wide = wide - col_means - common          # residuals in original units

                idio_models, last_idio = {}, {}
        for key in col_keys:
            e = e_wide[key].dropna()
            idio_models[key] = self._ar_fit(e.values, 1) if len(e) >= 5 else None
            if len(e):
                last_idio[key] = (int(e.index[-1]), float(e.iloc[-1]))
            else:
                last_idio[key] = (int(wide.index.max()), 0.0)
                        
 
        # Aggregate augmentation: gamma per series on the lag-1 aggregate of
        # the same variable; the aggregate itself gets an AR(1).
        gamma = {key: 0.0 for key in col_keys}
        agg_xmean = {key: 0.0 for key in col_keys}   # gamma is a slope: apply it to (x - mean x)
        agg_info: dict[str, tuple] = {}
        if self.factor_aug != "none":
            factors_df = _precompute_factors(
                past, cutoff_year, variables=list(variables_in) + ["nGDP_USD"])
            if factors_df is not None and not factors_df.empty:
                for v in variables_in:
                    ac = self._agg_col_for_var(factors_df, v)
                    if ac is None:
                        continue
                    hist = factors_df[ac].dropna()
                    hist = hist[hist.index <= cutoff_year]
                    if self.train_window_years is not None:
                        hist = hist[hist.index > cutoff_year - self.train_window_years]
                    if len(hist) < 5:
                        continue
                    hist.index = hist.index.astype(int)
                    agg_info[v] = (hist.to_dict(), int(hist.index[-1]),
                                   float(hist.iloc[-1]), self._ar_fit(hist.values, 1))
                for key in col_keys:
                    v = key[1]
                    if v not in agg_info:
                        continue
                    e = e_wide[key].dropna()
                    hist_d = agg_info[v][0]
                    x = np.array([hist_d.get(int(y) - 1, np.nan) for y in e.index])
                    ok = np.isfinite(x)
                    if ok.sum() < 5:
                        continue
                    x, y_ = x[ok], e.values[ok]
                    sxx = float(((x - x.mean()) ** 2).sum())
                    if sxx > 0:
                        gamma[key] = float(((x - x.mean()) * (y_ - y_.mean())).sum() / sxx)
                        agg_xmean[key] = float(x.mean())

        self._state = {
            "K": K_eff,
            "col_keys": col_keys,
            "col_means": col_means.to_dict(),
            "loadings": {k: L[j, :] for j, k in enumerate(col_keys)},
            "factor_models": factor_models,
            "F_by_year": pd.DataFrame(F, index=wide.index),
            "last_year": int(wide.index.max()),
            "last_factor": [[float(F[-1, k])] for k in range(K_eff)],
            "last_obs_year": {k: int(t_cols[k].index.max()) for k in col_keys},
            "idio_models": idio_models,
            "last_idio": last_idio,
            "ctx": ctx_per_pair,
            "gamma": gamma,
            "agg_xmean": agg_xmean,
            "agg_info": agg_info,
        }
        self.fit_status = "ok"

 def predict(self, panel: pd.DataFrame, cutoff_year: int,
                horizons: list[int]) -> pd.DataFrame:
        if self._state is None:
            return pd.DataFrame(columns=_OUT_COLS)
        st = self._state
        horizons = sorted({int(h) for h in horizons})
        end_year = cutoff_year + max(horizons)
        last_year, K = st["last_year"], st["K"]
        n_fc = max(end_year - last_year, 1)
        zero = np.zeros(K)
 
        # Factor paths by calendar year: in-sample where known, AR forecast after.
        paths = np.array([self._ar_forecast(c, a, st["last_factor"][k], n_fc)
                          for k, (c, a, _) in enumerate(st["factor_models"])]).T
        fac = {int(y): row.values for y, row in st["F_by_year"].iterrows()}
        fac.update({last_year + i + 1: paths[i] for i in range(n_fc)})
 
        rng = np.random.default_rng(self.seed)
        S = self.n_sims
        if S > 0:
            f_sims = np.stack(
                [self._ar_sim(c, a, st["last_factor"][k], n_fc, s, rng, S)
                 for k, (c, a, s) in enumerate(st["factor_models"])], axis=2)
 
        # Aggregate paths by calendar year (observed, then AR(1) forecast).
        agg_by_year = {}
        for v, (hist_d, y_a, a_last, res) in st["agg_info"].items():
            d = dict(hist_d)
            n = end_year - 1 - y_a
            if n > 0:
                fc = (self._ar_forecast(res[0], res[1], [a_last], n) if res
                      else [a_last] * n)
                d.update({y_a + i + 1: val for i, val in enumerate(fc)})
            agg_by_year[v] = (d, a_last)
 
        def agg_term(key, y):
            g = st["gamma"].get(key, 0.0)
            if g == 0.0 or key[1] not in agg_by_year:
                return 0.0
            d, a_last = agg_by_year[key[1]]
            # the residual has mean zero, so the fitted relation is
            # e = gamma * (x - mean x); without centring, gamma * mean(x) is added
            return g * (d.get(y - 1, a_last) - st["agg_xmean"].get(key, 0.0))
 
        rows = []
        for key in st["col_keys"]:
            iso, var = key
            ctx = st["ctx"].get(key)
            if ctx is None:
                continue
            tmode = transformation(var)
            L_iv = st["loadings"][key]
            mean_iv = float(st["col_means"].get(key, 0.0))
            sd_iv = float(st["col_sd"].get(key, 1.0))
            if not np.isfinite(mean_iv):
                mean_iv = 0.0
 
            y0 = st["last_obs_year"][key]            # anchor of the inverse transform
            years = np.arange(y0 + 1, end_year + 1)
            n = len(years)
            fac_mat = np.array([fac.get(int(y), zero) for y in years])      # (n, K)
            agg_vec = np.array([agg_term(key, int(y)) for y in years])
 
            idio = st["idio_models"].get(key)
            e_year, e_last = st["last_idio"][key]
            n_e = end_year - e_year
            e_idx = years - e_year - 1
            if idio is not None and n_e > 0:
                e_full = np.array(self._ar_forecast(idio[0], idio[1], [e_last], n_e))
                e_path = e_full[e_idx]
            else:
                e_path = np.zeros(n)
 
            t_point = mean_iv + sd_iv * (fac_mat @ L_iv) + e_path + agg_vec
            level = np.asarray(_inverse(list(t_point), ctx, mode=tmode), float)
 
            lev_sims = None
            if S > 0:
                F_s = np.empty((S, n, K))
                for i, y in enumerate(years):
                    F_s[:, i, :] = (f_sims[:, y - last_year - 1, :] if y > last_year
                                    else fac.get(int(y), zero))
                if idio is not None and n_e > 0:
                    e_s = self._ar_sim(idio[0], idio[1], [e_last], n_e,
                                       idio[2], rng, S)[:, e_idx]
                else:
                    e_s = np.zeros((S, n))
                T = mean_iv + sd_iv * (F_s @ L_iv) + e_s + agg_vec
                lev_sims = np.array([_inverse(list(r), ctx, mode=tmode) for r in T], float)
 
            for h in horizons:
                j = cutoff_year + h - y0 - 1         # index of year cutoff_year + h
                if j < 0 or j >= n:
                    continue
                point = float(level[j])
                if not np.isfinite(point):
                    continue
                lo = hi = float("nan")
                if lev_sims is not None:
                    col = lev_sims[:, j]
                    col = col[np.isfinite(col)]
                    if len(col):
                        lo, hi = np.percentile(col, [10, 90])
                        lo, hi = float(min(lo, point)), float(max(hi, point))
                rows.append((iso, cutoff_year + h, var, h, point, lo, hi))
        return pd.DataFrame(rows, columns=_OUT_COLS)
 
 
def model_instances() -> list[PanelPCAv2]:
    keys = list(PanelPCAv2.PARAM_GRID.keys())
    return [
        PanelPCAv2(**dict(zip(keys, combo)))
        for combo in itertools.product(*[PanelPCAv2.PARAM_GRID[k] for k in keys])
    ]
