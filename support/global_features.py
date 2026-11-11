"""Cross-country aggregates of each variable, in its transformed (stationary) space.

Columns: global_avg_<v>, global_wavg_<v> (nominal-USD-GDP weights), us_<v>,
         region_avg_<REG>_<v>, region_wavg_<REG>_<v>.   Index = year (<= cutoff).
"""
import numpy as np
import pandas as pd
from var_classification import transformation
from transformations import forward

KINDS = ("global_avg", "global_wavg", "us", "region_avg", "region_wavg")


def _transformed_wide(panel, v):
    out = {}
    for iso, g in panel[panel["variable"] == v].groupby("ISO3"):
        g = g.dropna(subset=["value"]).sort_values("year")
        if len(g) < 3:
            continue
        t, _ = forward(g["value"].values, transformation(v))
        out[iso] = pd.Series(t, index=g["year"].astype(int).values[-len(t):])
    return pd.DataFrame(out)


def precompute(panel, cutoff_year, variables, regions=None):
    panel = panel[panel["year"] <= cutoff_year]
    gdp = panel[panel["variable"] == "nGDP_USD"].pivot_table(index="year", columns="ISO3", values="value")
    cols = {}
    for v in variables:
        if v == "nGDP_USD":
            continue
        W = _transformed_wide(panel, v).replace([np.inf, -np.inf], np.nan)
        if W.empty:
            continue
        w = gdp.reindex(index=W.index, columns=W.columns)
        cols[f"global_avg_{v}"] = W.mean(axis=1)
        cols[f"global_wavg_{v}"] = (W * w).sum(axis=1, min_count=1) / w.where(W.notna()).sum(axis=1)
        if "USA" in W:
            cols[f"us_{v}"] = W["USA"]
        for reg in sorted(set((regions or {}).values())):
            members = [c for c in W.columns if (regions or {}).get(c) == reg]
            if not members:
                continue
            cols[f"region_avg_{reg}_{v}"] = W[members].mean(axis=1)
            wm = w[members]
            cols[f"region_wavg_{reg}_{v}"] = ((W[members] * wm).sum(axis=1, min_count=1)
                                              / wm.where(W[members].notna()).sum(axis=1))
    df = pd.DataFrame(cols).sort_index()
    df.index = df.index.astype(int)
    return df


def feature_columns(df, kind):
    return [c for c in df.columns if c.startswith(kind + "_")]
