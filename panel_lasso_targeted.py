"""
Panel LASSO with Factor Augmentation (FAVAR).
It expresses the canonical transformed target variable for country i 
at time t as a linear function penalised by an L1norm(LASSO).

Description:
    Estimates a panel LASSO model featuring country fixed effects, own lags, cross-variable auxiliary lags,
    and low-dimensional principle components extracted from global/regional macro indicators. When factor augmentation
    is active, raw collinear columns are compressed via PCA to 3 orthagonal components, preventing multicollinearity and convergense issues.

Estimation:
    Solves the L1-regularized optimization problem:
        min ||y - X*beta||^2 + alpha * ||beta||_1
    The LASSO penalty automatically selects relevant predictors by shrinking 
    irrelevant coefficients to exact zeros. Country fixed effects are absorbed 
    via one-hot dummy columns (fit_intercept=False).

Feature Matrix (X_{i,t}) per country i at time t:
    - Country FE Dummies: n_iso columns (1 if row is country i, else 0).
    - Target Lags: L columns representing T(v_target)_{i, t-1} ... T(v_target)_{i, t-L}.
    - Auxiliary Lags: L columns per auxiliary variable in aux_block.
    - Factor Principal Components: 3 columns representing PC_1, PC_2, PC_3 
      extracted from the global/regional factor block at time t-1.

Target Variable (y):
    - T(v_target)_{i, t} in v_target's canonical transformed space. 
      Predictions are inverted back to original units at inference time.

Forecasting (h-step):
    - Performs recursive multi-step forecasting for the target variable.
    - Auxiliary variables and macroeconomic factors are held fixed at their 
      latest observed values beyond the cutoff (conditional forecasting pattern).

Hyperparameters (PARAM_GRID):
    - L: Lag count
    - alpha: LASSO penalty strength
    - aux_block: Economically motivated subset of auxiliary variables
    - factor_mode: Inclusion rules for global/regional factors
    - train_window_years: Optional rolling training window constraint

References:
    - Tibshirani (1996, JRSS-B) on LASSO regression.
    - Bai & Ng (2008, J. Econometrics) on targeted predictor selection.
    - Bernanke, Boivin & Eliasz (2005, QJE) on FAVAR factor augmentation.
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from var_classification import family, transformation
from transformations import forward as _forward, inverse as _inverse
from input_registry import resolve_inputs, all_macro_inputs
from global_features import precompute as _precompute_factors, feature_columns


# Number of principal components extracted from the raw factor block.
# 3 is the standard FAVAR default (Bernanke-Boivin-Eliasz 2005).
PCA_K = 3
import logging
log = logging.getLogger(__name__)


class PanelLassoTargeted:
    family = "LASSO"
    segmentation = "panel_global"
    variables = None
    version = "0.3.1"

    PARAM_GRID = {
        "L": [1, 3],
        "alpha": [0.01, 0.1, 1.0],
        "input_mode": ["target_only", "curated", "all"],
        "factor_mode": ["none", "us_anchor", "global_wavg"],
        "train_window_years": [None, 50, 20, 10],
    }

    def __init__(
        self,
        L: int = 1,
        alpha: float = 0.01,
        input_mode: str = "curated",
        factor_mode: str = "none",
        train_window_years: int | None = None,
    ):
        self.L = int(L)
        self.alpha = float(alpha)
        self.input_mode = input_mode
        self.factor_mode = factor_mode
        self.train_window_years = train_window_years
        self.min_history = self.L + 10
        win_tag = "" if train_window_years is None else f"_w{train_window_years}"
        self.name = (
            f"panel_lasso_L{self.L}_a{self.alpha:g}"
            f"_in_{self.input_mode}_fac_{self.factor_mode}{win_tag}"
        )
        self.inputs = (
            f"Per target variable v_target: features = [country FE dummies | "
            f"T_target lags 1..{self.L} | T_v lags 1..{self.L} for each v in "
            f"resolved_inputs({self.input_mode}) | PC_1..PC_{PCA_K} of factors]. "
            f"LASSO (alpha={self.alpha:g}), no extra intercept. FAVAR style."
        )

    def _select_factor_cols(self, factors_df: pd.DataFrame) -> list[str]:
        if self.factor_mode == "none":
            return []
        if self.factor_mode == "us_anchor":
            return feature_columns(factors_df, "us")
        if self.factor_mode == "global_wavg":
            return feature_columns(factors_df, "global_wavg")
        if self.factor_mode == "global_plus_region":
            return (feature_columns(factors_df, "global_wavg")
                    + feature_columns(factors_df, "region_wavg"))
        if self.factor_mode == "region_wavg":
            return feature_columns(factors_df, "region_wavg")
        return []

    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None:
        self._models: dict[str, dict] = {}
        self._factors: pd.DataFrame = pd.DataFrame()
        self._factors_pcs: pd.DataFrame = pd.DataFrame()
        self._pca_imputer: SimpleImputer | None = None
        self._pca: PCA | None = None

        # 1. Enforce cutoff_year on panel to prevent data leakage globally
        panel = panel[panel["year"] <= cutoff_year]

        # Precompute global / regional factors once (if needed)
        if self.factor_mode != "none":
            self._factors = _precompute_factors(
                panel, cutoff_year, variables=list(all_macro_inputs()) + ["nGDP_USD"]
            )
            cols = self._select_factor_cols(self._factors)
            if cols and not self._factors.empty:
                F = self._factors.loc[
                    self._factors.index <= cutoff_year, cols
                ].astype(float)
                if len(F) >= PCA_K + 2:
                    self._pca_imputer = SimpleImputer(strategy="mean")
                    F_imp = self._pca_imputer.fit_transform(F.values)
                    n_components = min(PCA_K, F_imp.shape[1])
                    self._pca = PCA(n_components=n_components).fit(F_imp)
                    pcs = self._pca.transform(F_imp)
                    self._factors_pcs = pd.DataFrame(
                        pcs,
                        index=F.index,
                        columns=[f"PC{j+1}" for j in range(n_components)],
                    )

        for target in panel["variable"].unique():
            if family(target) == "skip":
                continue
            self._fit_one(panel, target, cutoff_year)

    def _fit_one(self, panel: pd.DataFrame, target: str, cutoff_year: int) -> None:
        #2. Implement train_window_years filter per target
        target_panel = panel
        if self.train_window_years is not None:
            min_allowed_year = cutoff_year - self.train_window_years + 1
            target_panel = panel[panel["year"] >= min_allowed_year]

        t_target = transformation(target)
        aux_vars = list(resolve_inputs(target, self.input_mode))
        relevant_vars = [target] + [v for v in aux_vars if v != target]
        sub = target_panel[target_panel["variable"].isin(relevant_vars)]
        if sub.empty:
            return

        wide = sub.pivot_table(
            index=["ISO3", "year"], columns="variable", values="value", aggfunc="last"
        ).reindex(columns=relevant_vars).dropna(subset=[target])
        if wide.empty:
            return
            
        factor_cols = (
            list(self._factors_pcs.columns) if not self._factors_pcs.empty else []
        )

        iso_idx: dict[str, int] = {}
        x_rows: list[list[float]] = []
        y_rows: list[float] = []
        per_iso_last: dict[str, dict] = {}

        for iso, g in wide.groupby(level="ISO3", observed=True):
            vals_t = g[target].values
            if len(vals_t) < self.min_history:
                continue
            t_target_arr, ctx_t = _forward(vals_t, t_target)
            aux_t = {}
            valid_iso = True
            for v in aux_vars:
                if v == target:
                    continue
                aux_vals = g[v].ffill().values
                if len(aux_vals) == 0 or np.isnan(aux_vals).all():
                    valid_iso = False
                    break
                t_arr, _ = _forward(aux_vals, transformation(v))
                aux_t[v] = t_arr
            if not valid_iso:
                continue

            min_len = min([len(t_target_arr)] + [len(a) for a in aux_t.values()] or [len(t_target_arr)])
            if min_len < self.L + 2:
                continue
            t_target_arr = t_target_arr[-min_len:]
            aux_t = {k: a[-min_len:] for k, a in aux_t.items()}

            # Year alignment: get the actual year sequence after transformation
            years = g.index.get_level_values("year").values
            years_aligned = years[-min_len:]

            mu = float(np.nanmean(t_target_arr))
            sd = float(np.nanstd(t_target_arr))
            if not np.isfinite(sd) or sd < 1e-8:
                sd = 1.0
            z_target = (t_target_arr - mu) / sd
            z_aux: dict[str, np.ndarray] = {}
            for v, a in aux_t.items():
                amu = float(np.nanmean(a))
                asd = float(np.nanstd(a))
                if not np.isfinite(asd) or asd < 1e-8:
                    asd = 1.0
                z_aux[v] = (a - amu) / asd

            if iso not in iso_idx:
                iso_idx[iso] = len(iso_idx)
            idx = iso_idx[iso]
            
            for t_pos in range(self.L, len(z_target)):
                year_t = int(years_aligned[t_pos])
                feat = [idx]   # placeholder for country idx
                feat.extend(z_target[t_pos - self.L: t_pos][::-1])
                for v in aux_vars:
                    if v == target:
                        continue
                    feat.extend(z_aux[v][t_pos - self.L: t_pos][::-1])
                # Factor features (PCA scores at lag-1)
                f_year = year_t - 1
                for fc in factor_cols:
                    try:
                        val = float(self._factors_pcs.loc[f_year, fc])
                    except (KeyError, TypeError):
                        val = np.nan
                    feat.append(val)
                # Skip rows with NaN factors
                
                if any(not np.isfinite(x) for x in feat[1:]) or not np.isfinite(z_target[t_pos]):
                    continue
                x_rows.append(feat)
                y_rows.append(float(z_target[t_pos]))
            # Stash last state for prediction (lags in z-space; mu/sd to invert)
            per_iso_last[iso] = {
                "target_lags": z_target[-self.L:][::-1].tolist(),
                "aux_lags": {v: z_aux[v][-self.L:][::-1].tolist()
                             for v in aux_vars if v != target},
                "last_year": int(years_aligned[-1]),
                "ctx_target": ctx_t,
                "mu": mu, "sd": sd,
            }

        if not x_rows:
            return
        n_iso = len(iso_idx)
        X_raw = np.array(x_rows, dtype=float)
        codes = X_raw[:, 0].astype(int)
        feat_cols = X_raw[:, 1:]
        onehot = np.zeros((len(codes), n_iso))
        onehot[np.arange(len(codes)), codes] = 1.0
        X = np.hstack([onehot, feat_cols])
        y = np.array(y_rows, dtype=float)
        try:
            mdl = Lasso(alpha=self.alpha, fit_intercept=False,
                        max_iter=5000).fit(X, y)
        except Exception as e:
            log.warning(f"Failed fitting Lasso for {target}: {e}")
            return
        self._models[target] = {
            "model": mdl, "n_iso": n_iso, "iso_idx": iso_idx,
            "last": per_iso_last, "aux_vars": aux_vars, "factor_cols": factor_cols,
        }

    def predict(self, panel: pd.DataFrame, cutoff_year: int, horizons: list[int]) -> pd.DataFrame:
        rows = []
        h_max = max(horizons)
        for target, mdef in self._models.items():
            mdl = mdef["model"]
            n_iso = mdef["n_iso"]
            iso_idx = mdef["iso_idx"]
            aux_vars = mdef["aux_vars"]
            factor_cols = mdef["factor_cols"]
            for iso, idx in iso_idx.items():
                if iso not in mdef["last"]:
                    continue
                state = mdef["last"][iso]
                tgt_lags = list(state["target_lags"])
                aux_lags = {k: list(v) for k, v in state["aux_lags"].items()}
                ctx_t = state["ctx_target"]
                mu, sd = state["mu"], state["sd"]
            
                offset = cutoff_year - state["last_year"]   # ragged edge
                if offset > 3:
                    continue
                t_steps = []  # z-space; de-standardized below
                for h in range(1, h_max + offset + 1):
                    feat = []
                    onehot = np.zeros(n_iso)
                    onehot[idx] = 1.0
                    feat.extend(onehot)
                    feat.extend(tgt_lags)
                    for v in aux_vars:
                        if v == target:
                            continue
                        feat.extend(aux_lags[v])
                        
                    # 3. Anchor factor PC scores at cutoff_year (safe global reference) instead of ragged last_year
                    
                    f_year = cutoff_year
                    skip = False
                    for fc in factor_cols:
                        try:
                            val = float(self._factors_pcs.loc[f_year, fc])
                        except (KeyError, TypeError):
                            val = np.nan
                        if not np.isfinite(val):
                            skip = True
                            break
                        feat.append(val)
                    if skip:
                        break
                    X = np.array(feat, dtype=float).reshape(1, -1)
                    if not np.all(np.isfinite(X)):        # missing input at the origin
                        break
                    t_hat = float(mdl.predict(X)[0])
                    t_steps.append(t_hat)
                    tgt_lags = [t_hat] + tgt_lags[:-1]
                if not t_steps:
                    continue
                # De-standardize z-space predictions back to the target's
                # transformed space, then invert the transform to levels.
                t_steps = [z * sd + mu for z in t_steps]
                level_steps = _inverse(t_steps, ctx_t, mode=transformation(target))
                for h in horizons:
                    if h - 1 + offset >= len(level_steps):
                        continue
                    yhat = level_steps[h - 1 + offset]
                    if np.isfinite(yhat):
                        rows.append((iso, cutoff_year + h, target, h, float(yhat),
                                     float("nan"), float("nan")))
        return pd.DataFrame(
            rows, columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
        )


def model_instances() -> list[PanelLassoTargeted]:
    keys = list(PanelLassoTargeted.PARAM_GRID.keys())
    return [
        PanelLassoTargeted(**dict(zip(keys, combo)))
        for combo in itertools.product(*[PanelLassoTargeted.PARAM_GRID[k] for k in keys])
    ]
