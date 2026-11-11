"""
Panel Random Forest with country FE + curated/all macro lags + year feature.

Description:
Extends the "Macroeconomy-as-a-Random-Forest" Framework (Goulet Coulombe JAE 2024) to a panel setting. 
It recognizes that macro relationships change drastically across economic regimes.
The forest learns nonlinear interactions and time-varying effective
parameters that a linear panel ridge can't capture — especially regime
shifts (high-inflation tail, GFC, COVID).

Feature Matrix (X_{i,t}) per country i at time t:
- Country One-Hot Dummies: n_iso columns (capturing country fixed effects).
- Target Lags: L columns representing T(v_target)_{i, t-1} ... T(v_target)_{i, t-L}.
- Auxiliary Lags: L columns per auxiliary variable in resolve_inputs(v, input_mode).

    *(Note: Raw 'year' features are omitted to prevent structural extrapolation failures 
    in tree leaves during out-of-sample forecasting).*

Target Variable (y):
    - T(v_target)_{i, t} (canonical transformed space).

Estimator:
    - sklearn.ensemble.RandomForestRegressor.

Forecasting (h-step):
    - Recursive multi-step forecasting for the target variable. 
    - Auxiliary variables are held fixed at their latest observed values (conditional forecast).
    - Caution: Tree-based recursive forecasts can exhibit step-function plateauing if 
      predicted lags exceed the historical training range.

Hyperparameters (PARAM_GRID):
    - L: Lag count
    - input_mode: Auxiliary variable block selection
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth of trees (crucial for regularization)
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from var_classification import family, transformation
from transformations import forward as _forward, inverse as _inverse
from input_registry import resolve_inputs
import logging
log = logging.getLogger(__name__)


class PanelRandomForest:
    family = "RandomForest"          # identifies the ML family
    segmentation = "panel_global"    # trains on the multi-country panel
    variables = None
    version = "0.2"
    train_window_years = None

    # Hyperparameters: lag length, number of trees, tree depth, training window
    PARAM_GRID = {
        "L":                  [1, 3],
        "input_mode":         ["curated", "all"],
        "n_estimators":       [200],
        "max_depth":          [5, 10],
        "train_window_years": [None, 50, 20, 10],
    }

    def __init__(
        self,
        L: int = 3,
        input_mode: str = "curated",
        n_estimators: int = 200,
        max_depth: int = 10,
        train_window_years: int | None = None,
    ):
        self.L = int(L)
        self.input_mode = input_mode
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.train_window_years = train_window_years
        self.min_history = self.L + 10
        win = "" if train_window_years is None else f"_w{train_window_years}"
        self.name = (
            f"rf_L{self.L}_in_{self.input_mode}"
            f"_ne{self.n_estimators}_d{self.max_depth}{win}"
        )
        self.inputs = (
            f"Per variable v in v's canonical transformed space T: features = "
            f"[country one-hot dummies | T(y_v) lags 1..{self.L} | T(y_aux) "
            f"lags 1..{self.L} for each aux in resolve_inputs(v, '{self.input_mode}') "
            f"| year]. RandomForestRegressor with n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, min_samples_leaf=5, max_features='sqrt'."
        )

    def _build_features(self, panel: pd.DataFrame, target: str, cutoff_year: int):
        """Build features up to cutoff_year, respecting train_window_years."""

        # 1. Prevent data leakage by enforcing cutoff_year
        panel = panel[panel["year"] <= cutoff_year]
        if panel.empty:
            return None

        # 2. Implement train_window_years filtering
        if self.train_window_years is not None:
            min_allowed_year = cutoff_year - self.train_window_years + 1
            panel = panel[panel["year"] >= min_allowed_year]
           
        tmode = transformation(target)
        aux_vars = list(resolve_inputs(target, self.input_mode))
        relevant_vars = [target] + [v for v in aux_vars if v != target]

        sub = panel[panel["variable"].isin(relevant_vars)]
        if sub.empty:
            return None
           
        wide = sub.pivot_table(
            index=["ISO3", "year"], columns="variable", values="value", aggfunc="last"
        ).reindex(columns=relevant_vars).dropna(subset=[target])

        if wide.empty:
            return None
           
        iso_idx: dict[str, int] = {}
        rows = []
        per_iso_last = {}


        for iso, g in wide.groupby(level="ISO3", observed=True):
            vals_t = g[target].values
            if len(vals_t) < self.min_history:
                continue
            t_target, ctx_t = _forward(vals_t, tmode)
           
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

            min_len = min([len(t_target)] + [len(a) for a in aux_t.values()] or [len(t_target)])
            if min_len < self.L + 2:
                continue
               
            t_target = t_target[-min_len:]
            aux_t = {k: a[-min_len:] for k, a in aux_t.items()}
            years = g.index.get_level_values("year").values[-min_len:]
           
            if iso not in iso_idx:
                iso_idx[iso] = len(iso_idx)
            idx = iso_idx[iso]
           
            for t in range(self.L, len(t_target)):
                feat = [idx]
                if not np.all(np.isfinite(t_target[t - self.L: t + 1])):
                    continue
                feat.extend(t_target[t - self.L: t][::-1])
                for v in aux_vars:
                    if v == target:
                        continue
                    feat.extend(aux_t[v][t - self.L: t][::-1])
                feat.append(int(years[t]))
                if not np.all(np.isfinite(feat)):
                    continue
                rows.append((feat, float(t_target[t])))

            per_iso_last[iso] = {
                "target_lags": t_target[-self.L:][::-1].tolist(),
                "aux_lags": {v: aux_t[v][-self.L:][::-1].tolist()
                             for v in aux_vars if v != target},
                "last_year": int(years[-1]),
                "ctx": ctx_t,
            }
           
        if not rows:
            return None
           
        n_iso = len(iso_idx)
        X_raw = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=float)
        codes = X_raw[:, 0].astype(int)
        rest = X_raw[:, 1:]
        onehot = np.zeros((len(codes), n_iso))
        onehot[np.arange(len(codes)), codes] = 1.0
        X = np.hstack([onehot, rest])
        return X, y, iso_idx, per_iso_last, aux_vars

    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None:
        self._models: dict[str, dict] = {}
        for target in panel["variable"].unique():
            if family(target) == "skip":
                continue
            built = self._build_features(panel, target, cutoff_year)
            if built is None:
                continue
            X, y, iso_idx, per_iso_last, aux_vars = built
            if len(X) < 20:
                continue
            try:
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    n_jobs=1,
                    random_state=0,
                ).fit(X, y)
            except Exception as e:
                log.warning(f"Failed to fit RandomForest for {target}: {e}")
                continue
            self._models[target] = {
                "rf": rf,
                "n_iso": len(iso_idx),
                "iso_idx": iso_idx,
                "last": per_iso_last,
                "aux_vars": aux_vars,
            }

    def predict(self, panel: pd.DataFrame, cutoff_year: int, horizons: list[int]) -> pd.DataFrame:
        rows = []
        h_max = max(horizons)
        for target, m in self._models.items():
            tmode = transformation(target)
            for iso, idx in m["iso_idx"].items():
                if iso not in m["last"]:
                    continue
                state = m["last"][iso]
                tgt_lags = list(state["target_lags"])
                aux_lags = {k: list(v) for k, v in state["aux_lags"].items()}
                last_year = state["last_year"]
                ctx = state["ctx"]
               
                offset = cutoff_year - last_year     # ragged edge
                if offset > 3:
                    continue
                t_steps = []
                for h in range(1, h_max + offset + 1):
                    feat = []
                    onehot = np.zeros(m["n_iso"])
                    onehot[idx] = 1.0
                    feat.extend(onehot)
                    feat.extend(tgt_lags)
                    for v in m["aux_vars"]:
                        if v == target:
                            continue
                        feat.extend(aux_lags[v])
                    feat.append(last_year + h)
                    X = np.array(feat).reshape(1, -1)
                    yhat_t = float(m["rf"].predict(X)[0])
                    t_steps.append(yhat_t)
                    tgt_lags = [yhat_t] + tgt_lags[:-1]
                   
                lvl = _inverse(t_steps, ctx, mode=tmode)
                for h in horizons:
                    if h + offset > len(lvl):
                        continue
                    yhat = lvl[h - 1 + offset]
                    if np.isfinite(yhat):
                        rows.append((iso, cutoff_year + h, target, h, float(yhat),
                                     float("nan"), float("nan")))
        return pd.DataFrame(
            rows, columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
        )


def model_instances() -> list[PanelRandomForest]:
    keys = list(PanelRandomForest.PARAM_GRID.keys())
    return [
        PanelRandomForest(**dict(zip(keys, combo)))
        for combo in itertools.product(*[PanelRandomForest.PARAM_GRID[k] for k in keys])
    ]
