"""
Per-country PCA factor model with FULL cross-product of design choices.
Stock-Watson (2002, JASA) on diffusion-index Dunamic Factor Models.

Forecasts variable v in country i by (e.g. inflation in Germany) from a few common "factors" that summarise that country's 
other macro variables, plus the variable's own past, and optionally, the regional or world average.
Each (country, target variable) pairs gets its own small model.

Design:
1.   Variable choice (input_mode)
     ='curated' uses the variables economically linked to the target,
                from input_registry.resolve_inputs(target, "curated");
     ='all'     uses every indicator in input_registry.all_macro_inputs().
     The target itself is never a predictor; its own lag enters in step 7.

  2. Data window
     Only data with year <= cutoff_year (no look-ahead).
     train_window_years = N keeps only the last N years (None = all history).
     Each series keeps its latest run of consecutive years (no gaps).
     
  3. Transform each (country, variable) series with transformations.forward,
     e.g. log-differences for for GDP and levels for rates. 
     
  4. Optimal screening(tstat_screen): Keep a predictor x only if it helps forecast the target next year,
     beyond the target's own past: y(t+1) = a + rho*y(t) = b*x(t) + error,
     
  5. Factors by Principal Component Analysis
     build the matrix M = [year x predictor] for this country,
     e.g. for target DEU-infl: columns (DEU,rGDP), (DEU,unemp), (DEU,strate) ...
     Drop predictors with less than 50% coverage and years where fewer than
     half of the predictors are observed.
     Standardise each column (mean 0, sd 1), then extract K = 1 or 2 factors
     F(t) by EM-PCA: missing cells are filled iteratively from the factor fit.
  
  
  6. Optional aggregate augmentation (factor_aug)
     Add agg(t) = regional or global average of the SAME variable,
     equal-weighted or weighted by nominal GDP in USD:
         factor_aug in {none, region_eq, region_gdp, global_eq, global_gdp}
     Regional options need the ISO3 -> region mapping `regions`.

  7. Forecast equation (method)
     ='direct'   (default) one regression per forecast lead l = 1, 2, ...:
                    y(t+l) = c + rho * y(t) + beta * F(t) + gamma * agg(t) + error
                 Uses only what is known at the forecast origin t.
     ='iterated' one regression for next year,
                    y(t) = c + rho * y(t-1) + beta * F(t-1) + gamma * agg(t-1) + error,
                 rolled forward step by step; F and agg are forecast by AR(1).

  8. Predict
     Forecast origin = last year in which target, factors and aggregate are all
     observed. Forecast the path in transformed space, back-transform to levels.
     Horizon h always means the year cutoff_year + h.
     80% interval (lower, upper):
       direct   -> the model's own past forecast errors added to the forecast;
       iterated -> 200 simulated paths with random shocks.
     Output: ISO3, year, variable, horizon, point, lower, upper

  9. When something is missing
     No predictor survives screening -> model without factors (own lag + aggregate).
     Series that cannot be fitted are skipped; reasons and counts are in
     fit_report after fit() (fit_status = "ok" or the reason for no output).

  model_instances() for the full grid:
     2 input modes x 5 augmentations x 2 screening x 2 values of K
     x 4 training windows (None, 50, 20, 10) x 2 methods = 320 variants

NOTE: Web-scrapp test was run for 30 countries, 10 variables, 1954–2025:
IMF WEO: unemployment from 1980 (28 countries).
FRED: OECD 3-month and 10-year rates; BIS real effective exchange rates.
World Bank: GDP, CPI and inflation, broad money and the USD exchange rate.

 Extreemly time expensive in coputing, more than 30 min, with scoring. 

The score is each model’s error divided by the no-change forecast’s error (median across countries); below 1 is better.

Model	   1 year  	2 years	5 years
v3 baseline	1.00	     1.02	     1.06
AR(1)	     1.00	     1.01	     1.03
Panel PCA, 
20-year window	1.04	     0.97	     0.96
Old v2         1.00	     0.98	     1.01

"""

from __future__ import annotations
import itertools

import logging
from collections import Counter 
import numpy as np
import pandas as pd


from var_classification import family, transformation # variables names must match these known under it 
from transformations import forward as _forward, inverse as _inverse # variables names must match these known under it
from input_registry import resolve_inputs, all_macro_inputs
from global_features import precompute as _precompute_factors, feature_columns

log = logging.getLogger(__name__)

# Map factor_aug name → feature_columns kind
_FACTOR_AUG_KIND = {
    "none":       None,
    "region_eq":  "region_avg",
    "region_gdp": "region_wavg",
    "global_eq":  "global_avg",
    "global_gdp": "global_wavg",
}

_METHODS = ("direct", "iterated")
_OUT_COLS = ["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
_MAX_AR = 0.98          # cap on AR coefficients in the iterated recursion


class PCACountryV2:
    family = "DFM"
    segmentation = "country_multivar"
    variables = None
    version = "0.2"

    PARAM_GRID = {
        "input_mode":         ["curated", "all"],
        "factor_aug":         ["none", "region_eq", "region_gdp", "global_eq", "global_gdp"],
        "tstat_screen":       [True, False],
        "K":                  [1, 2],
        "train_window_years": [None, 50, 20, 10],
    }


  # ---------------------------------------------------------------- helpers
def _ols(X: np.ndarray, y: np.ndarray):
    """OLS -> (coef, residuals, standard errors) or None."""
    n, p = X.shape
    if n <= p:
        return None
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        xtx_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    resid = y - X @ coef
    s2 = float(resid @ resid) / (n - p)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * s2, 0.0, None))
    return coef, resid, se


def _em_pca(Z: np.ndarray, K: int, n_iter: int = 100, tol: float = 1e-6):
    """PCA of a standardised matrix with NaNs (EM). Returns F (T x K), L (N x K)."""
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


def _ar1(s: pd.Series):
    """AR(1) on a year-indexed series using consecutive-year pairs only.
    Returns (intercept, slope, residual sd)."""
    prev = s.reindex(s.index - 1).to_numpy(float)
    cur = s.to_numpy(float)
    ok = np.isfinite(prev) & np.isfinite(cur)
    if ok.sum() >= 5:
        res = _ols(np.column_stack([np.ones(ok.sum()), prev[ok]]), cur[ok])
        if res is not None:
            coef, resid, _ = res
            sd = float(np.sqrt(resid @ resid / max(ok.sum() - 2, 1)))
            return float(coef[0]), float(coef[1]), sd
    return float(np.nanmean(cur)), 0.0, float(np.nanstd(cur))


# ---------------------------------------------------------------- model
class PCACountryV3:
    family = "DFM"
    segmentation = "country_multivar"
    variables = None
    version = "0.3"

    PARAM_GRID = {
        "input_mode":         ["curated", "all"],
        "factor_aug":         ["none", "region_eq", "region_gdp", "global_eq", "global_gdp"],
        "tstat_screen":       [True, False],
        "K":                  [1, 2],
        "train_window_years": [None, 50, 20, 10],
        "method":             ["direct", "iterated"],
    }

    def __init__(
        self,
        input_mode: str = "curated",
        factor_aug: str = "none",
        tstat_screen: bool = False,
        K: int = 1,
        train_window_years: int | None = None,
        method: str = "direct",
        *,
        t_threshold: float = 2.0,
        min_feat_share: float = 0.5,
        min_dof: int = 4,
        max_horizon: int = 5,
        max_staleness: int = 3,
        fallback_ar: bool = True,
        regions: dict[str, str] | None = None,
        min_boot: int = 8,
        n_sims: int = 200,
        seed: int = 0,
    ):
        if input_mode not in ("curated", "all"):
            raise ValueError(f"unknown input_mode {input_mode!r}")
        if factor_aug not in _FACTOR_AUG_KIND:
            raise ValueError(f"unknown factor_aug {factor_aug!r}")
        if method not in _METHODS:
            raise ValueError(f"unknown method {method!r}")
        if int(K) < 1:
            raise ValueError("K must be >= 1")
        self.input_mode = input_mode
        self.factor_aug = factor_aug
        self.tstat_screen = bool(tstat_screen)
        self.K = int(K)
        self.train_window_years = None if train_window_years is None else int(train_window_years)
        self.method = method
        self.t_threshold = float(t_threshold)
        self.min_feat_share = float(min_feat_share)
        self.min_dof = int(min_dof)
        self.max_horizon = int(max_horizon)
        self.max_staleness = int(max_staleness)
        self.fallback_ar = bool(fallback_ar)
        self.regions = dict(regions) if regions else {}
        self.min_boot = int(min_boot)
        self.n_sims = int(n_sims)
        self.seed = int(seed)

        ts_tag = "ts" if self.tstat_screen else "nots"
        win_tag = "" if self.train_window_years is None else f"_w{self.train_window_years}"
        m_tag = "dir" if method == "direct" else "it"
        self.name = (f"pca_country_v3_in_{input_mode}_fac_{factor_aug}"
                     f"_{ts_tag}_K{self.K}{win_tag}_{m_tag}")
        self.inputs = (
            f"Per (country, target): predictors = lags of '{input_mode}' variables"
            f"{f', screened |t| > {self.t_threshold:g} with own lag controlled' if self.tstat_screen else ''}; "
            f"standardised, EM-PCA -> {self.K} factor(s); aggregate '{factor_aug}'; "
            f"data <= cutoff{'' if self.train_window_years is None else f', last {self.train_window_years} years'}; "
            f"{'direct h-step regressions on origin-date information' if method == 'direct' else 'iterated AR with lagged factors, factor and aggregate AR(1)'}; "
            f"80% intervals from {'historical residual vectors' if method == 'direct' else f'{self.n_sims} simulations'}. "
            f"References: Stock-Watson (2002); Bai-Ng (2008)."
        )
        self._models: dict[tuple[str, str], dict] = {}
        self.cutoff_year: int | None = None
        self.fit_report: dict = {}
        self.fit_status = "not fitted"

    # ------------------------------------------------------------ data prep
    def _prepare_series(self, panel: pd.DataFrame, cutoff_year: int):
        """Transformed series per (ISO3, variable), indexed by year.
        Returns ({key: Series}, {key: (ctx, last_observed_year)})."""
        p = panel[panel["year"] <= cutoff_year]
        if self.train_window_years is not None:
            p = p[p["year"] > cutoff_year - self.train_window_years]
        p = p.dropna(subset=["value"])
        series, ctx = {}, {}
        for (iso, v), g in p.groupby(["ISO3", "variable"], observed=True, sort=False):
            if family(v) == "skip":
                continue
            g = g.drop_duplicates("year", keep="last").sort_values("year")
            yrs = g["year"].astype(int).to_numpy()
            vals = g["value"].astype(float).to_numpy()
            gaps = np.where(np.diff(yrs) != 1)[0]
            if len(gaps):                              # keep the latest consecutive run
                yrs, vals = yrs[gaps[-1] + 1:], vals[gaps[-1] + 1:]
            if len(vals) < 3:
                continue
            t, c = _forward(vals, transformation(v))
            t = np.asarray(t, float)
            bad = np.where(~np.isfinite(t))[0]
            if len(bad):                               # e.g. log of a non-positive value
                t = t[bad[-1] + 1:]
            if len(t) < 3:
                continue
            series[(iso, v)] = pd.Series(t, index=yrs[-len(t):])
            ctx[(iso, v)] = (c, int(yrs[-1]))
        return series, ctx

    def _prepare_aggregates(self, panel: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
        if self.factor_aug == "none":
            return pd.DataFrame()
        past = panel[panel["year"] <= cutoff_year]
        df = _precompute_factors(past, cutoff_year,
                                 variables=list(all_macro_inputs()) + ["nGDP_USD"],
                                 regions=self.regions)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = df.index.astype(int)
        df = df[df.index <= cutoff_year]
        if self.train_window_years is not None:
            df = df[df.index > cutoff_year - self.train_window_years]
        return df

    def _candidate_list(self, target: str) -> list[str]:
        pool = all_macro_inputs() if self.input_mode == "all" else resolve_inputs(target, "curated")
        return [v for v in dict.fromkeys(pool) if v != target and family(v) != "skip"]

    def _aggregate_for(self, iso: str, target: str, aggs: pd.DataFrame):
        """The aggregate series of the target variable for this country, or (None, reason)."""
        kind = _FACTOR_AUG_KIND[self.factor_aug]
        if kind is None:
            return None, None
        if aggs.empty:
            return None, "no aggregates computed"
        cols = [c for c in feature_columns(aggs, kind) if c.endswith(f"_{target}")]
        if kind.startswith("region"):
            region = self.regions.get(iso)
            if region is None:
                return None, "no region mapping for country"
            cols = [c for c in cols if f"_{region}_" in c]
        elif f"{kind}_{target}" in cols:
            cols = [f"{kind}_{target}"]
        if len(cols) != 1:
            return None, f"{len(cols)} matching aggregate columns"
        s = aggs[cols[0]].replace([np.inf, -np.inf], np.nan).dropna()
        return (s if len(s) else None), (None if len(s) else "aggregate empty")

    # ------------------------------------------------------------ building blocks
    def _screen(self, y: pd.Series, cands: dict[str, pd.Series]) -> list[str]:
        """Targeted predictors: |t| of x_t in y_{t+1} = a + rho*y_t + b*x_t + e."""
        if not self.tstat_screen:
            return list(cands)
        base = pd.DataFrame({"lead": y.reindex(y.index + 1).to_numpy(), "y": y.to_numpy()},
                            index=y.index)
        kept = []
        for v, x in cands.items():
            d = base.assign(x=x.reindex(base.index)).dropna()
            if len(d) < 10:
                continue
            res = _ols(np.column_stack([np.ones(len(d)), d["y"], d["x"]]), d["lead"].to_numpy())
            if res is None:
                continue
            coef, _, se = res
            if se[2] > 0 and abs(coef[2] / se[2]) > self.t_threshold:
                kept.append(v)
        return kept

    def _factors(self, years: pd.Index, cands: dict[str, pd.Series]) -> pd.DataFrame | None:
        """Standardised EM-PCA factors on the target's years, or None."""
        if not cands:
            return None
        X = pd.DataFrame({v: s.reindex(years) for v, s in cands.items()})
        X = X.loc[:, X.notna().mean() >= 0.5]
        X = X.loc[:, X.std(ddof=0) > 0]
        if X.shape[1] == 0:
            return None
        need = max(1, int(np.ceil(self.min_feat_share * X.shape[1])))
        X = X.loc[X.notna().sum(axis=1) >= need]
        k = min(self.K, X.shape[1])
        if len(X) < k + 5:
            return None
        Z = ((X - X.mean()) / X.std(ddof=0)).to_numpy()
        try:
            F, _ = _em_pca(Z, k)
        except np.linalg.LinAlgError:
            return None
        return pd.DataFrame(F, index=X.index, columns=[f"F{j + 1}" for j in range(k)])

    def _lead_regression(self, D: pd.DataFrame, valid: np.ndarray, lead: int):
        """y_{t+lead} on [1, y_t, F_t, agg_t] over origins t."""
        idx = D.index.to_numpy()
        target = D["y"].reindex(idx + lead).to_numpy(float)
        rows = valid & np.isfinite(target)
        X = np.column_stack([np.ones(int(rows.sum())), D.to_numpy(float)[rows]])
        n, p = X.shape
        if n < p + self.min_dof:
            return None
        res = _ols(X, target[rows])
        if res is None:
            return None
        coef, resid, _ = res
        return {"coef": coef, "resid": pd.Series(resid, index=idx[rows]),
                "scale": float(np.sqrt(n / (n - p))), "n": n, "p": p}

    def _clip(self, a: float) -> float:
        if abs(a) > _MAX_AR:
            self._counts["ar_coef_clipped"] += 1
            return float(np.sign(a) * _MAX_AR)
        return float(a)

    # ------------------------------------------------------------ main API
    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None:
        self.cutoff_year = int(cutoff_year)
        self._models = {}
        self._counts = Counter()
        skipped = Counter()

        series, ctx = self._prepare_series(panel, self.cutoff_year)
        aggs = self._prepare_aggregates(panel, self.cutoff_year)
        targets = sorted({v for (_, v) in series})
        cand_lists = {t: self._candidate_list(t) for t in targets}

        for (iso, target) in sorted(series):
            reason = self._fit_one(iso, target, series, ctx, aggs, cand_lists[target])
            if reason:
                skipped[reason] += 1

        self.fit_report = {"fitted": len(self._models), **dict(self._counts),
                           "skipped": dict(skipped)}
        if self._models:
            self.fit_status = "ok"
            log.info("%s cutoff %d: %s", self.name, self.cutoff_year, self.fit_report)
        else:
            self.fit_status = f"no series fitted ({dict(skipped)})"
            log.warning("%s cutoff %d: %s", self.name, self.cutoff_year, self.fit_status)

    def _fit_one(self, iso, target, series, ctx, aggs, cand_list) -> str | None:
        """Fit one (country, target). Returns a skip reason or None."""
        y = series[(iso, target)]
        c_t, last_year = ctx[(iso, target)]
        if self.cutoff_year - last_year > self.max_staleness:
            return "target too stale"

        cands = {v: series[(iso, v)] for v in cand_list if (iso, v) in series}
        kept = self._screen(y, cands)
        F = self._factors(y.index, {v: cands[v] for v in kept})
        if F is None:
            if not self.fallback_ar:
                return "no usable predictors"
            self._counts["fallback_no_factors"] += 1
            F = pd.DataFrame(index=y.index)

        D = pd.DataFrame({"y": y}).join(F, how="left")
        agg, why = self._aggregate_for(iso, target, aggs)
        if why:
            self._counts[f"aggregate dropped: {why}"] += 1
        if agg is not None:
            D["agg"] = agg.reindex(D.index)

        core_ok = D.drop(columns="agg", errors="ignore").notna().all(axis=1).to_numpy()
        if not core_ok.any():
            return "no year with target and factors"
        anchor = int(D.index[core_ok][-1])
        if "agg" in D and pd.isna(D.at[anchor, "agg"]):
            D = D.drop(columns="agg")
            self._counts["aggregate dropped: missing at origin"] += 1
        if self.cutoff_year - anchor > self.max_staleness:
            return "predictors too stale"
        valid = D.notna().all(axis=1).to_numpy()
        has_agg = "agg" in D
        fcols = [c for c in D.columns if c.startswith("F")]

        model = {"method": self.method, "anchor": anchor, "last_year": last_year,
                 "ctx": c_t, "tmode": transformation(target)}

        if self.method == "direct":
            first = last_year + 1 - anchor                # first lead the path needs
            last = self.cutoff_year + self.max_horizon - anchor
            regs = {}
            for lead in range(first, last + 1):
                r = self._lead_regression(D, valid, lead)
                if r is None:
                    break
                regs[lead] = r
            if not regs:
                return "too few observations"
            model["regs"] = regs
            model["x_anchor"] = np.concatenate([[1.0], D.loc[anchor].to_numpy(float)])
        else:
            r = self._lead_regression(D, valid, 1)
            if r is None:
                return "too few observations"
            coef, k = r["coef"], len(fcols)
            model["c"] = float(coef[0])
            model["rho"] = self._clip(float(coef[1]))
            model["beta"] = np.asarray(coef[2:2 + k], float)
            model["gamma"] = float(coef[2 + k]) if has_agg else 0.0
            model["sig_y"] = float(np.sqrt(r["resid"] @ r["resid"] / (r["n"] - r["p"])))
            fa = [_ar1(D.loc[D[c].notna(), c]) for c in fcols]
            model["f_c"] = np.array([a[0] for a in fa])
            model["f_a"] = np.array([self._clip(a[1]) for a in fa])
            model["f_sd"] = np.array([a[2] for a in fa])
            model["f_anchor"] = D.loc[anchor, fcols].to_numpy(float)
            model["y_anchor"] = float(D.at[anchor, "y"])
            model["y_known"] = {int(t): float(v) for t, v in y.items() if t > anchor}
            if has_agg:
                a_c, a_a, a_sd = _ar1(agg)
                model["agg"] = (a_c, self._clip(a_a), a_sd)
                model["agg_known"] = {int(t): float(v) for t, v in agg.items() if t > anchor}
                model["a_anchor"] = float(D.at[anchor, "agg"])
        self._models[(iso, target)] = model
        return None

    # ------------------------------------------------------------ forecasting
    def _direct_path(self, m: dict, years: np.ndarray):
        pts, leads = [], []
        for t in years:
            lead = int(t - m["anchor"])
            r = m["regs"].get(lead)
            if r is None:
                break
            pts.append(float(m["x_anchor"] @ r["coef"]))
            leads.append(lead)
        if not pts:
            return np.array([]), None
        R = pd.concat({lead: m["regs"][lead]["resid"] * m["regs"][lead]["scale"] for lead in leads},
                      axis=1).dropna()
        sims = np.asarray(pts) + R.to_numpy() if len(R) >= self.min_boot else None
        return np.asarray(pts), sims

    def _iterated_path(self, m: dict, years: np.ndarray, rng: np.random.Generator):
        S, k = self.n_sims, len(m["beta"])
        has_agg = "agg" in m
        y_p, f_p = m["y_anchor"], m["f_anchor"].copy()
        a_p = m["a_anchor"] if has_agg else 0.0
        y_s, f_s, a_s = np.full(S, y_p), np.tile(f_p, (S, 1)), np.full(S, a_p)
        pts, sims = [], []
        for t in range(m["anchor"] + 1, int(years[-1]) + 1):
            y_new = m["c"] + m["rho"] * y_p + f_p @ m["beta"] + m["gamma"] * a_p
            y_new_s = (m["c"] + m["rho"] * y_s + f_s @ m["beta"] + m["gamma"] * a_s
                       + rng.normal(0.0, m["sig_y"], S))
            if t in m["y_known"]:                       # observed after the origin
                y_new = m["y_known"][t]
                y_new_s = np.full(S, y_new)
            f_p = m["f_c"] + m["f_a"] * f_p
            f_s = m["f_c"] + m["f_a"] * f_s + rng.normal(0.0, 1.0, (S, k)) * m["f_sd"]
            if has_agg:
                a_c, a_a, a_sd = m["agg"]
                if t in m["agg_known"]:
                    a_p = m["agg_known"][t]
                    a_s = np.full(S, a_p)
                else:
                    a_p = a_c + a_a * a_p
                    a_s = a_c + a_a * a_s + rng.normal(0.0, a_sd, S)
            y_p, y_s = y_new, y_new_s
            if t > m["last_year"]:
                pts.append(y_p)
                sims.append(y_s.copy())
        if not pts:
            return np.array([]), None
        return np.asarray(pts), (np.column_stack(sims) if S > 0 else None)

    def predict(self, panel: pd.DataFrame, cutoff_year: int, horizons: list[int]) -> pd.DataFrame:
        if not self._models:
            return pd.DataFrame(columns=_OUT_COLS)
        cutoff_year = int(cutoff_year)
        if cutoff_year != self.cutoff_year:
            log.warning("%s: predict cutoff %d differs from fit cutoff %d",
                        self.name, cutoff_year, self.cutoff_year)
        hs = sorted({int(h) for h in horizons if 1 <= int(h) <= self.max_horizon})
        if len(hs) < len(set(horizons)):
            log.warning("%s: horizons above max_horizon=%d are not produced",
                        self.name, self.max_horizon)
        if not hs:
            return pd.DataFrame(columns=_OUT_COLS)
        end_year = cutoff_year + max(hs)
        rng = np.random.default_rng(self.seed)

        rows = []
        for (iso, target), m in self._models.items():
            years = np.arange(m["last_year"] + 1, end_year + 1)
            if len(years) == 0:
                continue
            if m["method"] == "direct":
                pts, sims = self._direct_path(m, years)
            else:
                pts, sims = self._iterated_path(m, years, rng)
            if len(pts) == 0:
                continue
            level = np.asarray(_inverse(list(pts), m["ctx"], mode=m["tmode"]), float)
            lev_s = None
            if sims is not None:
                lev_s = np.array([_inverse(list(r), m["ctx"], mode=m["tmode"]) for r in sims],
                                 float)
            for h in hs:
                j = cutoff_year + h - m["last_year"] - 1
                if j < 0 or j >= len(level) or not np.isfinite(level[j]):
                    continue
                point, lo, hi = float(level[j]), float("nan"), float("nan")
                if lev_s is not None:
                    col = lev_s[:, j][np.isfinite(lev_s[:, j])]
                    if len(col):
                        lo, hi = np.percentile(col, [10, 90])
                        lo, hi = float(min(lo, point)), float(max(hi, point))
                rows.append((iso, cutoff_year + h, target, h, point, lo, hi))
        return pd.DataFrame(rows, columns=_OUT_COLS)


def model_instances(regions: dict[str, str] | None = None) -> list[PCACountryV3]:
    keys = list(PCACountryV3.PARAM_GRID.keys())
    return [
        PCACountryV3(**dict(zip(keys, combo)), regions=regions)
        for combo in itertools.product(*[PCACountryV3.PARAM_GRID[k] for k in keys])
    ]
