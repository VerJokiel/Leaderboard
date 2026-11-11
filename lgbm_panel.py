"""
Panel LightGBM with quantile loss — economically rich feature set.

This model combines gradient-boosted decision trees (LightGBM) with panel data 
structures and quantile regression loss, enabling it to capture complex 
non-linearities, secular trends, and empirical prediction intervals simultaneously.

Description:
    Estimates a non-linear panel gradient boosting model (LightGBM) trained with quantile 
    regression loss. It captures complex interactions, threshold effects, secular time trends, 
    and macroeconomic regime shifts across countries. For each target, three models are trained 
    simultaneously.
    
Estimation:
    Minimizes the pinball loss (quantile loss) via gradient boosted decision trees using 
    LightGBM. This avoids the strict linearity and normality assumptions of classical models 
    and handles outliers robustly.

Feature Matrix (X_{i,t}) per country i at time t:
    - Country One-Hot Dummies: n_iso columns capturing country fixed effects.
    - Target Lags: L columns representing T(v)_{i, t-1} ... T(v)_{i, t-L}.
    - Curated Auxiliary Lags: L columns per auxiliary variable in curated_inputs(v).
    - Secular Time Trend: Raw integer `year` capturing continuous long-term macro trends.
    - Regime Indicator: Coarse categorical `decade` (floor(year / 10) * 10) capturing structural eras.

Target Variable (y):
    - T(v)_{i, t} in the variable's canonical transformed space. Predictions and intervals 
      are back-transformed to original levels using transformations.inverse.

Forecasting (h-step):
    - Performs recursive multi-step forecasting by feeding predicted target values 
      back into the own-lag sequence.
    - Auxiliary variables are held fixed at their last observed values, while time 
      features (year/decade) advance deterministically.

Hyperparameters (PARAM_GRID):
    - L: Lag count
    - learning_rate: Boosting step size
    - num_leaves: Maximum tree leaf nodes controlling model complexity
    - input_mode: Auxiliary variable block selection

References:
    - Ke et al. (2017, NeurIPS) on LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
    - IMF Working Papers (e.g., 2025/252) on Machine Learning applications to macroeconomic forecasting.
"""

from __future__ import annotations
import itertools
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from var_classification import family, transformation
from transformations import forward as _forward, inverse as _inverse
from input_registry import resolve_inputs

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


class PanelLGBMQuantile:
    family = "GBM"
    segmentation = "panel_global"
    variables = None
    version = "0.2"

    PARAM_GRID = {
        "L":                  [1, 3],
        "input_mode":         ["curated", "all"],
        "num_leaves":         [15, 31],
        "learning_rate":      [0.05],
        "n_estimators":       [200],
        "train_window_years": [None, 50, 20, 10],
    }

    def __init__(
        self,
        L: int = 3,
        input_mode: str = "curated",
        num_leaves: int = 15,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        train_window_years: int | None = None,
    ):
        self.L = int(L)
        self.input_mode = input_mode
        self.num_leaves = int(num_leaves)
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.train_window_years = train_window_years
        self.min_history = self.L + 10
        win = "" if train_window_years is None else f"_w{train_window_years}"
        self.name = (
            f"lgbm_q_L{self.L}_in_{self.input_mode}"
            f"_lvs{self.num_leaves}_lr{self.learning_rate:g}{win}"
        )
        self.inputs = (
            f"Per variable v in v's canonical transformed space T: features = "
            f"[country one-hot dummies | T(y_v) lags 1..{self.L} | T(y_aux) "
            f"lags 1..{self.L} for each aux in resolve_inputs(v, '{self.input_mode}') "
            f"| year]. Three LightGBM quantile regressors per target with tau in {{0.1, 0.5, 0.9}}."
        )

    def _build_features(
        self, panel: pd.DataFrame, target: str
    ) -> tuple[np.ndarray, np.ndarray, dict, dict, list[str]] | None:
        """Returns (X, y, iso_idx, per_iso_last, aux_vars) or None."""
       # 1. Prevent data leakage by enforcing cutoff_year globally
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
        ).reindex(columns=relevant_vars).dropna()
        
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
                t_arr, _ = _forward(g[v].values, transformation(v))
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
                feat.extend(t_target[t - self.L: t][::-1])
                for v in aux_vars:
                    if v == target:
                        continue
                    feat.extend(aux_t[v][t - self.L: t][::-1])
                feat.append(int(years[t]))    # year as feature
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
            built = self._build_features(panel, target)
            if built is None:
                continue
            X, y, iso_idx, per_iso_last, aux_vars = built
            if len(X) < 20:
                continue
            try:
                m50 = lgb.LGBMRegressor(
                    objective="quantile", alpha=0.5,
                    num_leaves=self.num_leaves, learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators, min_data_in_leaf=5,
                    verbose=-1,
                ).fit(X, y)
                m10 = lgb.LGBMRegressor(
                    objective="quantile", alpha=0.1,
                    num_leaves=self.num_leaves, learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators, min_data_in_leaf=5,
                    verbose=-1,
                ).fit(X, y)
                m90 = lgb.LGBMRegressor(
                    objective="quantile", alpha=0.9,
                    num_leaves=self.num_leaves, learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators, min_data_in_leaf=5,
                    verbose=-1,
                ).fit(X, y)
            except Exception:
                continue
            self._models[target] = {
                "m50": m50, "m10": m10, "m90": m90,
                "n_iso": len(iso_idx), "iso_idx": iso_idx,
                "last": per_iso_last, "aux_vars": aux_vars,
            }

    def predict(self, panel: pd.DataFrame, cutoff_year: int, horizons: list[int]) -> pd.DataFrame:
        rows = []
        h_max = max(horizons)
        for target, m in self._models.items():
            tmode = transformation(target)
            for iso, idx in m["iso_idx"].items():
                state = m["last"][iso]
                tgt_lags = list(state["target_lags"])
                aux_lags = {k: list(v) for k, v in state["aux_lags"].items()}
                last_year = state["last_year"]
                ctx = state["ctx"]
                
                t_steps_50, t_steps_10, t_steps_90 = [], [], []
                for h in range(1, h_max + 1):
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
                    p50 = float(m["m50"].predict(X)[0])
                    p10 = float(m["m10"].predict(X)[0])
                    p90 = float(m["m90"].predict(X)[0])
                    t_steps_50.append(p50)
                    t_steps_10.append(p10)
                    t_steps_90.append(p90)
                    tgt_lags = [p50] + tgt_lags[:-1]
                    
                lvl_50 = _inverse(t_steps_50, ctx, mode=tmode)
                lvl_10 = _inverse(t_steps_10, ctx, mode=tmode)
                lvl_90 = _inverse(t_steps_90, ctx, mode=tmode)
                
                for h in horizons:
                    yhat = lvl_50[h - 1]
                    lo = lvl_10[h - 1]
                    hi = lvl_90[h - 1]
                    if np.isfinite(yhat):
                        rows.append((iso, cutoff_year + h, target, h, float(yhat),
                                     float(lo) if np.isfinite(lo) else float("nan"),
                                     float(hi) if np.isfinite(hi) else float("nan")))
        return pd.DataFrame(
            rows, columns=["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
        )


def model_instances() -> list[PanelLGBMQuantile]:
    keys = list(PanelLGBMQuantile.PARAM_GRID.keys())
    return [
        PanelLGBMQuantile(**dict(zip(keys, combo)))
        for combo in itertools.product(*[PanelLGBMQuantile.PARAM_GRID[k] for k in keys])
    ]
