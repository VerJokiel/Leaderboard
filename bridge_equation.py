""" Bridge equations — per-country regression of a target on a few related indicators, with the indicators extrapolated by AR(1) to the forecast horizon.

Ported in spirit from the ECB Nowcasting Toolbox (Meunier & Linzenich 2024,
ECB WP 3004) bridge-equation method (Banbura et al. 2023). 

For each (country, target) with transform T:
    T(target)_t = a + sum_v b_v * T(indicator_v)_{t-lag} + e_t       (OLS)
indicators = resolve_inputs(target, input_mode). Forecast: AR(1) each
indicator's transformed series to t+h, plug into the fitted bridge, then invert
the transform. Optionally include a lag of the target itself (bridge + AR).

References:  Banbura, Bel & Delle Chiaie (2023) / Banbura et al. on bridge-equation nowcasting; Baffigi, Golinelli & Parigi (2004, IJF) on bridge models. """

from __future__ import annotations
                import itertools
                import numpy as np
                import pandas as pd

from var_classification import family, transformation
from transformations import forward as _forward, inverse as _inverse
from input_registry import resolve_inputs

class BridgeEquation:
    family = "Other" segmentation = "country_multivar"  variables = None  version = "0.2"

    PARAM_GRID = { "input_mode": ["curated", "all"], "ind_lag": [0, 1], "use_target_lag": [False, True], }

    def __init__(self, input_mode: str = "curated", ind_lag: int = 0,
                 use_target_lag: bool = False):
        self.input_mode = input_mode
        self.ind_lag = int(ind_lag)
        self.use_target_lag = bool(use_target_lag)
        self.min_history = 12
        self.name = (f"bridge_in_{self.input_mode}_lag{self.ind_lag}"
                     f"{'_artarget' if self.use_target_lag else ''}")
        self.inputs = (
            f"Per-country bridge equation: OLS of T(target)_t on a constant, "
            f"the lag-{self.ind_lag} transformed indicators "
            f"resolve_inputs(target, '{self.input_mode}')"
            f"{' and the lag-1 target' if self.use_target_lag else ''}. "
            f"Indicators are AR(1)-extrapolated to t+h, then the fitted bridge "
            f"is evaluated and back-transformed. Banbura et al. (2023) bridge "
            f"equations (ECB Nowcasting Toolbox), annual single-frequency form; "
            f"Baffigi-Golinelli-Parigi (2004, IJF)." )

    def _ar1_path(self, y, h_max):
        """AR(1) forecast path of length h_max from series y (transformed)."""
        y = np.asarray(y, float)
        if len(y) < 4:  last = float(y[-1]) if len(y) else 0.0
            return [last] * h_max 
        
        y0, y1 = y[:-1], y[1:]
        v = float(np.var(y0))
        if v <= 0:
            return [float(y[-1])] * h_max
            
        b = float(np.cov(y0, y1, ddof=0)[0, 1] / v)
        a = float(np.mean(y1) - b * np.mean(y0))
        out, last = [], float(y[-1])
        
        for _ in range(h_max):
            last = a + b * last
            out.append(last)
        return out

    def fit(self, panel: pd.DataFrame, cutoff_year: int) -> None:
        self._models = {}
        self._cutoff_year = int(cutoff_year)
        
        # wide per country: {ISO3: DataFrame(index=year, columns=variable) in T-space}
        for iso, g_iso in panel.groupby("ISO3", observed=True):
            series, ctx_map = {}, {}
            
            for v, g in g_iso.groupby("variable", observed=True):
                if family(v) == "skip":
                    continue
                vals = g.sort_values("year")["value"].dropna().astype(float).values
                yrs = g.sort_values("year")["year"].astype(int).values
                if len(vals) < self.min_history:
                    continue
                tvals, ctx = _forward(vals, transformation(v))
                series[v] = pd.Series(tvals, index=yrs[-len(tvals):])
                ctx_map[v] = ctx
                
            if not series:
                continue
                
            wide = pd.concat(series, axis=1)
            for target in series:
                inds = [v for v in resolve_inputs(target, self.input_mode)
                        if v in series and v != target]
                if not inds:
                    continue
                    
                # Build design: target_t ~ 1 + indicators_{t-lag} (+ target_{t-1})
                y = wide[target]
                cols = {v: wide[v].shift(self.ind_lag) for v in inds}
                if self.use_target_lag:
                    cols["__ytm1"] = wide[target].shift(1)
                    
                dsn = pd.concat({"__y": y, **cols}, axis=1).dropna()
                if len(dsn) < max(self.min_history, len(cols) + 4):
                    continue
                    
                Y = dsn["__y"].to_numpy(float)
                feat_names = [c for c in dsn.columns if c != "__y"]
                Xd = np.hstack([np.ones((len(dsn), 1)),
                                dsn[feat_names].to_numpy(float)])
                try:
                    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                    
                self._models[(iso, target)] = {
                    "beta": beta, "inds": inds, "feat_names": feat_names,
                    "series": {v: series[v] for v in inds},
                    "target_series": series[target],
                    "ctx": ctx_map[target],
                }

    def predict(self, panel, cutoff_year, horizons):
        cols_out = ["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]
        rows = []
        h_max = max(horizons)
        
        for (iso, target), m in self._models.items():
            # Forecast each indicator's transformed path to t+h_max.
            ind_paths = {v: self._ar1_path(m["series"][v].to_numpy(), h_max)
                         for v in m["inds"]}
            # Recent observed values to seed lagged terms / target AR.
            ind_last = {v: float(m["series"][v].to_numpy()[-1]) for v in m["inds"]}
            ytm1 = float(m["target_series"].to_numpy()[-1])
            t_steps = []
            beta = m["beta"]
            
            for h in range(1, h_max + 1):
                feat = [1.0]
                for fn in m["feat_names"]:
                    if fn == "__ytm1":
                        feat.append(t_steps[-1] if t_steps else ytm1)
                    else:
                        # indicator at t+h-ind_lag (lag shifts which step we read)
                        idx = h - 1 - self.ind_lag
                        if idx < 0:
                            feat.append(ind_last[fn])
                        else:
                            feat.append(ind_paths[fn][idx])
                t_steps.append(float(np.dot(beta, feat)))
                
            level = _inverse(t_steps, m["ctx"], mode=transformation(target))
            for h in horizons:
                yhat = level[h - 1]
                if np.isfinite(yhat):
                    rows.append((iso, cutoff_year + h, target, h, float(yhat),
                                 float("nan"), float("nan")))
        
        return pd.DataFrame(rows, columns=cols_out)


def model_instances() -> list[BridgeEquation]:
    keys = list(BridgeEquation.PARAM_GRID.keys())
    return [BridgeEquation(**dict(zip(keys, combo)))
            for combo in itertools.product(*[BridgeEquation.PARAM_GRID[k] for k in keys])]
