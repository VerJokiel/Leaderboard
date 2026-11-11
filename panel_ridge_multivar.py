"""
Panel Ridge regression with Cross-Variable Lag features.

Overview:
    Fits a panel ridge regression model for a target variable using its own past lags and
    cross-variable lags from a selected block of auxilary macro predictors.

Feature Matrix ( X_{i,t}) per (country i, time t) is build as:
    X_{i,t} = [
 
    - Country Fixed Effect Dummies: n_iso columns (1 if row is country `i`, else 0)
    - Target Lags: L columns representing T(v_target)_{i, t-1} ... T(v_target)_{i, t-L}
    - Auxiliary Lags: L columns per auxiliary variable representing T(v_aux)_{i, t-1} ... T(v_aux)_{i, t-L}
    
Target variable: T(v_target)_{i, t}, where T() is v_target's canonical transformation.

Estimation:
- solves the Ridge regression optimisation problem:
    min ||y -X*beta||^2 + alpha * ||beta||^2.

Forecasting (h-step):
- Perform a recursive forecast for the target variable only in transformed space.
- Auxiliary variables are held fixed at their latest observed values beyond the cutoff 
   (standard conditional forecasting pattern for non-VAR models)

Hyperparameters PARAM_GRID:
- L: Lag count,
- alpha: Ridge penalty
- aux_block: economically motivated subset of auxiliary variables
- train_window_years: optimal rolling training window constraint
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from var_classification import family, transformation # variables names must match these known under it 
from transformations import forward as _forward, inverse as _inverse # variables names must match these known under it

log = logging.getLogger(__name__)

# Auxiliary variable blocks — economically motivated subsets of the macro core.
# These are the standard "small open economy" indicator blocks: output, prices,
# interest rates, exchange rate, fiscal stance, external balance, employment.
_AUX_REAL   = ("rGDP", "CPI", "unemp")                              # real-side core
_AUX_FIN    = ("rGDP", "CPI", "strate", "ltrate", "USDfx")          # financial-side core
_AUX_FULL   = ("rGDP", "CPI", "strate", "USDfx", "unemp", "M2",
               "gen_govdebt_GDP", "CA_GDP")                         # everything


class PanelRidgeMultivar:
    family = "Ridge"
    segmentation = "panel_global"
    variables = None
    version = "0.2.1"

    PARAM_GRID = {
        "L": [1, 3],
        "alpha": [0.1, 1.0, 10.0],
        "aux_block": [_AUX_REAL, _AUX_FIN, _AUX_FULL],
        "train_window_years": [None, 50, 20, 10],
    }

    def __init__(self, L: int = 3, alpha: float = 1.0, aux_block: tuple = _AUX_REAL,
                 train_window_years: int | None = None):
        self.L = int(L)
        self.alpha = float(alpha)
        self.aux_block = tuple(aux_block)
        self.train_window_years = train_window_years
        self.min_history = self.L + 10
        aux_tag = "_".join(self.aux_block) if self.aux_block else "none"
        window_tag = "" if train_window_years is None else f"_w{train_window_years}"
        self.name = f"panel_ridge_mv_L{self.L}_a{self.alpha:g}_{aux_tag}{window_tag}"
        self.inputs = (
            f"Per target variable v_target, in v_target's canonical transformed "
            f"space T_target: features = [country FE dummies | T_target lags "
            f"1..{self.L} of v_target | T_aux lags 1..{self.L} of each "
            f"v_aux in {self.aux_block} (each in its own canonical transform)]. "
            f"Target = T_target(v_target)_{{i,t}}. Ridge with alpha={self.alpha:g}, "
            f"no extra intercept (FE dummies absorb it). h-step forecast iterates "
            f"the target variable's equation recursively in transformed space; "
            f"auxiliary variables are held at their last observed value (conditional "
            f"forecast). NB: this is the first model with CROSS-VARIABLE features. "
            f"Implementation: sklearn.linear_model.Ridge. "
            f"Reference: Hoerl & Kennard (1970, Technometrics) on ridge regression."
        )

    def _build_pooled(self, panel: pd.DataFrame, var_target: str, cutoff_year: int):
        """Build the pooled (X, y) for one target variable up to cutoff_year."""
        if family(var_target) == "skip":
            return None, None, None, None
            
        # Enforce cutoff_year to prevent data leakage
        panel = panel[panel["year"] <= cutoff_year]
        if panel.empty:
            return None, None, None, None

        # Apply train_window_years filter if specified
        if self.train_window_years is not None:
            min_allowed_year = cutoff_year - self.train_window_years + 1
            panel = panel[panel["year"] >= min_allowed_year]  
            
        t_target = transformation(var_target)
        # Wide-format slice: rows=(iso, year), cols=variable
        relevant_vars = (var_target,) + tuple(v for v in self.aux_block if v != var_target)
        sub = panel[panel["variable"].isin(relevant_vars)]
        if sub.empty:
            return None, None, None, None
            
        wide = sub.pivot_table(
            index=["ISO3", "year"], columns="variable", values="value", aggfunc="last"
        ).reindex(columns=list(relevant_vars))

        # Drop rows where target is missing (aux variables can optionally be interpolated or handled separately)
        wide = wide.dropna(subset=[var_target])

        iso_idx: dict[str, int] = {}
        x_rows, y_rows = [], []
        per_iso_last: dict[str, dict] = {}

        for iso, g in wide.groupby(level="ISO3", observed=True):
            vals_target = g[var_target].values
            if len(vals_target) < self.min_history:
                continue
            t_target_arr, ctx_target = _forward(vals_target, t_target)
            
            aux_t = {}
            valid_iso = True
            for v in self.aux_block:
                if v == var_target:
                    continue
                    
        # Forward fill or drop missing in aux series to avoid total row dropping
                aux_vals = g[v].ffill().values 
                if len(aux_vals) == 0 or np.isnan(aux_vals).all():
                    valid_iso = False
                    break
                t_arr, _ = _forward(g[v].values, transformation(v))
                aux_t[v] = t_arr

            if not valid_iso:
                continue
            
            # Align lengths: take the shortest available transformed length
            
            min_len = min([len(t_target_arr)] + [len(v) for v in aux_t.values()] or [len(t_target_arr)])
            t_target_arr = t_target_arr[-min_len:]
            aux_t = {k: v[-min_len:] for k, v in aux_t.items()}
            
            if len(t_target_arr) < self.L + 2:
                continue
            if iso not in iso_idx:
                iso_idx[iso] = len(iso_idx)
            idx = iso_idx[iso]
            
            for t in range(self.L, len(t_target_arr)):
                feat = list(t_target_arr[t - self.L:t][::-1])
                for v in self.aux_block:
                    if v == var_target:
                        continue
                    feat.extend(aux_t[v][t - self.L:t][::-1])
                x_rows.append([idx] + feat)
                y_rows.append(t_target_arr[t])

            
            # Stash last state for this iso, target var
            per_iso_last[iso] = {
                "target_lags": t_target_arr[-self.L:][::-1].tolist(),
                "aux_lags": {v: aux_t[v][-self.L:][::-1].tolist()
                              for v in self.aux_block if v != var_target},
                "ctx_target": ctx_target,
            }

        if not x_rows:
            return None, None, None, None

        
        n_iso = len(iso_idx)
        X_raw = np.array(x_rows, dtype=float)
        codes = X_raw[:, 0].astype(int)
        feat_cols = X_raw[:, 1:]
        onehot = np.zeros((len(codes), n_iso))
        onehot[np.arange(len(codes)), codes] = 1.0
        X = np.hstack([onehot, feat_cols])
        y = np.array(y_rows, dtype=float)
        return X, y, iso_idx, per_iso_last

    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None:
        self._models: dict[str, dict] = {}   # var -> {model, n_iso, iso_idx, per_iso_last}

        for var in panel["variable"].unique():
            if family(var) == "skip":
                continue
            built = self._build_pooled(panel, var)
            X, y, iso_idx, per_iso_last = built
            if X is None:
                continue
            try:
                mdl = Ridge(alpha=self.alpha, fit_intercept=False).fit(X, y)
            except Exception as e:
                log.warning(f"Failed to fit Ridge for {var}: {e}")
                continue
                
            self._models[var] = {
                "model": mdl,
                "n_iso": len(iso_idx),
                "iso_idx": iso_idx,
                "last": per_iso_last,
            }

    def predict(self, panel: pd.DataFrame, cutoff_year: int, horizons: list[int]) -> pd.DataFrame:
        rows = []
        h_max = max(horizons)
        for var, mdef in self._models.items():
            mdl = mdef["model"]
            n_iso = mdef["n_iso"]
            iso_idx = mdef["iso_idx"]
            for iso, idx in iso_idx.items():
                state = mdef["last"][iso]
                target_lags = list(state["target_lags"])  # [t-1, t-2, ...]
                aux_lags = {k: list(v) for k, v in state["aux_lags"].items()}
                ctx_target = state["ctx_target"]
                t_steps = []

                for _h in range(h_max):
                    onehot = np.zeros(n_iso)
                    onehot[idx] = 1.0
                    feat = list(target_lags)
                    for v in self.aux_block:
                        if v == var:
                            continue
                        feat.extend(aux_lags[v])
                    X = np.concatenate([onehot, np.array(feat)])
                    t_hat = float(mdl.predict(X.reshape(1, -1))[0])
                    t_steps.append(t_hat)
                    target_lags = [t_hat] + target_lags[:-1]
                    
                    # Aux lags: hold at last observed (conditional forecast)
                level_steps = _inverse(t_steps, ctx_target, mode=transformation(var))
                for h in horizons:
                    yhat = level_steps[h - 1]
                    if np.isfinite(yhat):
                        rows.append((iso, cutoff_year + h, var, h, float(yhat),
                                     float("nan"), float("nan")))
        return pd.DataFrame(
            rows, columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
        )


def model_instances() -> list[PanelRidgeMultivar]:
    keys = list(PanelRidgeMultivar.PARAM_GRID.keys())
    return [
        PanelRidgeMultivar(**dict(zip(keys, combo)))
        for combo in itertools.product(*[PanelRidgeMultivar.PARAM_GRID[k] for k in keys])
    ]
