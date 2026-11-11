"""Evaluate 30 forecasting model specifications from this repository on one macro panel.

Pipeline
  1. DATA      World Bank WDI, IMF WEO and FRED (OECD/BIS), 20 countries, from 1960 where
               available (support/fetch_data.py). Data are truncated at the cut-off 2025;
               IMF values after 2025 are WEO projections and are never used as actuals.
  2. SAME SAMPLE  every model sees exactly the same panel and the same forecast origins.
               At each origin T a model is re-estimated with data <= T only (no look-ahead)
               and forecasts T+1 and T+2.
  3. SAMPLES   in-sample   = estimation period: origins 1990-2014 (targets 1991-2016)
               out-of-sample = hold-out: origins 2015-2024 (targets 2016-2025, cut-off 2025)
               final forecast = models fitted on all data up to 2025 -> 2026 and 2027
  4. ERRORS    level variables (CPI, rGDP, M2, USDfx, REER): 100*log(forecast/actual) (~% error)
               rates (infl, unemp, strate, ltrate, CA_GDP, debt): forecast - actual (pp)
  5. MEASURES  the five most common in macro forecast evaluation
               RMSE      root mean squared error
               MAE       mean absolute error
               MAPE      mean absolute percentage error, |f - a| / |a| (rates: only |a| >= 0.5)
               Theil U2  RMSE(model) / RMSE(no-change forecast) on the same cases; < 1 beats it
               DA        directional accuracy: share of forecasts with the right sign of change
  6. OUTPUT    results/forecasts.csv, results/scores.csv, results/final_forecasts.csv,
               results/leaderboard.html (pick 3 of the 30 models, see their scores side by side)

Run:  python evaluate_models.py            (full run; uses all CPU cores)
      python evaluate_models.py --quick    (3 origins per sample, for a smoke test)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "support")]

from fetch_data import build_panel, REGIONS, SOURCE                     # noqa: E402
from var_classification import LEVEL_VARS, transformation, TCODE      # noqa: E402

CUTOFF = 2025
IN_SAMPLE = range(1990, 2015)          # origins
OUT_SAMPLE = range(2015, 2025)         # origins; targets up to the cut-off 2025
HORIZONS = [1, 2]
TARGETS = ["infl", "CPI", "rGDP", "unemp", "strate", "ltrate", "USDfx", "REER", "M2", "CA_GDP",
           "gen_govdebt_GDP"]
OUT = os.path.join(HERE, "results")

REAL, FIN, FULL = (("rGDP", "CPI", "unemp"), ("rGDP", "CPI", "strate", "ltrate", "USDfx"),
                   ("rGDP", "CPI", "strate", "USDfx", "unemp", "M2", "gen_govdebt_GDP", "CA_GDP"))

# 30 specifications: (module, class, kwargs, short label). Chosen to span each file's
# PARAM_GRID along the dimensions that matter (lags, penalty, inputs, factors, window).
SPECS = [
    ("panel_ar_country_fe", "PanelAR", dict(p=1), "Panel AR(1) + country FE"),
    ("panel_ar_country_fe", "PanelAR", dict(p=2), "Panel AR(2) + country FE"),
    ("panel_ar_country_fe", "PanelAR", dict(p=3), "Panel AR(3) + country FE"),
    ("panel_ar_country_fe", "PanelAR", dict(p=1, train_window_years=20), "Panel AR(1), 20y window"),
    ("panel_ar_country_fe", "PanelAR", dict(p=2, train_window_years=20), "Panel AR(2), 20y window"),
    ("panel_ridge_multivar", "PanelRidgeMultivar", dict(L=1, alpha=1.0, aux_block=REAL), "Ridge L1, real block"),
    ("panel_ridge_multivar", "PanelRidgeMultivar", dict(L=3, alpha=1.0, aux_block=REAL), "Ridge L3, real block"),
    ("panel_ridge_multivar", "PanelRidgeMultivar", dict(L=1, alpha=10.0, aux_block=FIN), "Ridge L1 a10, financial block"),
    ("panel_ridge_multivar", "PanelRidgeMultivar", dict(L=1, alpha=1.0, aux_block=FULL), "Ridge L1, full block"),
    ("panel_ridge_multivar", "PanelRidgeMultivar", dict(L=1, alpha=1.0, aux_block=REAL, train_window_years=20), "Ridge L1, real, 20y window"),
    ("panel_lasso_targeted", "PanelLassoTargeted", dict(L=1, alpha=0.01, input_mode="target_only"), "LASSO L1, own lags only"),
    ("panel_lasso_targeted", "PanelLassoTargeted", dict(L=1, alpha=0.01, input_mode="curated"), "LASSO L1, curated inputs"),
    ("panel_lasso_targeted", "PanelLassoTargeted", dict(L=1, alpha=0.1, input_mode="all"), "LASSO L1 a0.1, all inputs"),
    ("panel_lasso_targeted", "PanelLassoTargeted", dict(L=1, alpha=0.01, input_mode="curated", factor_mode="global_wavg"), "LASSO curated + global factors"),
    ("panel_lasso_targeted", "PanelLassoTargeted", dict(L=3, alpha=0.01, input_mode="curated", factor_mode="us_anchor"), "LASSO L3 curated + US factors"),
    ("panel_random_forest", "PanelRandomForest", dict(L=1, input_mode="curated", max_depth=5), "Random forest L1, depth 5"),
    ("panel_random_forest", "PanelRandomForest", dict(L=3, input_mode="curated", max_depth=5), "Random forest L3, depth 5"),
    ("panel_random_forest", "PanelRandomForest", dict(L=1, input_mode="all", max_depth=10), "Random forest L1 all, depth 10"),
    ("panel_random_forest", "PanelRandomForest", dict(L=1, input_mode="curated", max_depth=5, train_window_years=20), "Random forest L1, 20y window"),
    ("pca_panel_v2", "PanelPCAv2", dict(input_mode="curated", K=1), "Panel PCA K=1, curated"),
    ("pca_panel_v2", "PanelPCAv2", dict(input_mode="curated", K=2), "Panel PCA K=2, curated"),
    ("pca_panel_v2", "PanelPCAv2", dict(input_mode="all", K=1), "Panel PCA K=1, all"),
    ("pca_panel_v2", "PanelPCAv2", dict(input_mode="curated", K=1, factor_aug="global_gdp"), "Panel PCA K=1 + global agg."),
    ("pca_panel_v2", "PanelPCAv2", dict(input_mode="curated", K=1, train_window_years=20), "Panel PCA K=1, 20y window"),
    ("multivar_dfm_v2", "MultivarDFMv2", dict(input_mode="curated", K=1), "DFM K=1, core block"),
    ("multivar_dfm_v2", "MultivarDFMv2", dict(input_mode="all", K=2), "DFM K=2, all"),
    ("multivar_dfm_v2", "MultivarDFMv2", dict(input_mode="curated", K=1, factor_aug="global_gdp"), "DFM K=1 + global agg."),
    ("pca_country_v2", "PCACountryV3", dict(input_mode="curated", K=1, method="direct"), "Country PCA K=1, direct"),
    ("pca_country_v2", "PCACountryV3", dict(input_mode="curated", K=1, tstat_screen=True, factor_aug="region_gdp", method="direct"), "Country PCA screened + region"),
    ("pca_country_v2", "PCACountryV3", dict(input_mode="all", K=2, method="iterated"), "Country PCA K=2 all, iterated"),
]
RW = "No-change (random walk)"
ALL_C, ALL_VARS = "All countries", "All variables (median)"
COLS = ["ISO3", "year", "variable", "horizon", "point", "lower", "upper"]


def make_model(spec):
    import importlib
    mod, cls, kw, label = spec
    kw = dict(kw)
    if cls == "PCACountryV3":
        kw["regions"] = REGIONS
    m = getattr(importlib.import_module(mod), cls)(**kw)
    return m


def load_panel() -> pd.DataFrame:
    p = build_panel()
    p = p[(p.year <= CUTOFF) & p.value.notna()].copy()          # cut-off: nothing after 2025
    p["year"] = p.year.astype(int)
    return p.reset_index(drop=True)


def no_change(panel, origin):
    """Random-walk benchmark: last value observed at the origin (stale <= 3 years)."""
    last = panel[panel.year <= origin].sort_values("year").groupby(["ISO3", "variable"]).tail(1)
    last = last[(origin - last.year <= 3) & last.variable.isin(TARGETS)]
    return pd.DataFrame([(r.ISO3, origin + h, r.variable, h, r.value, np.nan, np.nan)
                         for r in last.itertuples() for h in HORIZONS], columns=COLS)


def run_spec(i, origins, panel):
    """Fit + forecast one specification at every origin. Runs in a worker process."""
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    spec = SPECS[i]
    out, fails, t0 = [], [], time.time()
    for T in origins:
        try:
            m = make_model(spec)
            m.fit(panel, T)
            fc = m.predict(panel, T, HORIZONS)
            if len(fc):
                fc = fc[COLS].copy()
                fc["origin"] = T
                out.append(fc)
        except Exception as e:                                    # noqa: BLE001
            fails.append(f"{T}: {type(e).__name__}: {e}")
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLS + ["origin"])
    df["model"] = spec[3]
    return df, fails, time.time() - t0, getattr(make_model(spec), "name", spec[1])


def add_errors(fc, panel):
    act = panel.set_index(["ISO3", "variable", "year"]).value
    act = act[~act.index.duplicated()]
    fc = fc[fc.variable.isin(TARGETS)].copy()
    fc["actual"] = act.reindex(pd.MultiIndex.from_frame(fc[["ISO3", "variable", "year"]])).to_numpy()
    last = {}
    for T in fc.origin.unique():
        lp = panel[panel.year <= T].sort_values("year").groupby(["ISO3", "variable"]).value.last()
        last.update({(i, v, T): x for (i, v), x in lp.items()})
    fc["last_obs"] = [last.get(k, np.nan) for k in zip(fc.ISO3, fc.variable, fc.origin)]
    fc = fc.dropna(subset=["actual", "point"])
    lvl = fc.variable.isin(LEVEL_VARS)
    fc = fc[~lvl | ((fc.point > 0) & (fc.actual > 0))].copy()
    lvl = fc.variable.isin(LEVEL_VARS)
    with np.errstate(divide="ignore", invalid="ignore"):
        fc["error"] = np.where(lvl, 100 * np.log(fc.point / fc.actual), fc.point - fc.actual)
        ape = 100 * (fc.point - fc.actual).abs() / fc.actual.abs()
    fc["ape"] = np.where(lvl | (fc.actual.abs() >= 0.5), ape, np.nan)
    fc["dir_hit"] = (np.sign(fc.point - fc.last_obs) == np.sign(fc.actual - fc.last_obs)).astype(float)
    fc.loc[fc.model == RW, "dir_hit"] = np.nan                       # undefined for no-change
    return fc[np.isfinite(fc.error)]


def _agg(g):
    e, r = g.error.to_numpy(), g.rw_error.to_numpy()
    rmse, rmse_rw = np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean(r ** 2))
    return pd.Series({"RMSE": rmse, "MAE": np.mean(np.abs(e)), "MAPE": g.ape.median(),
                      "TheilU": rmse / rmse_rw if rmse_rw > 0 else np.nan,
                      "DA": g.dir_hit.mean(), "n": len(g)})


def _pool_vars(by, keys):
    """'All variables' rows: units differ, so take the median of per-variable scores."""
    return (by.groupby(keys)
              .agg(RMSE=("RMSE", "median"), MAE=("MAE", "median"), MAPE=("MAPE", "median"),
                   TheilU=("TheilU", "median"), DA=("DA", "mean"), n=("n", "sum"))
              .reset_index().assign(variable=ALL_VARS))


def score(fc):
    """Five measures per sample x model x country x variable x horizon.
    ISO3 = 'All countries' pools the errors of every country for that variable.
    Theil U2 and all measures use only cases the random walk also forecasts."""
    key = ["sample", "ISO3", "variable", "origin", "horizon"]
    rw = fc[fc.model == RW].set_index(key).error.rename("rw_error")
    f = fc.join(rw, on=key).dropna(subset=["rw_error"])
    g = ["sample", "model", "variable", "horizon"]
    pooled = f.groupby(g).apply(_agg, include_groups=False).reset_index().assign(ISO3=ALL_C)
    per_c = f.groupby(g + ["ISO3"]).apply(_agg, include_groups=False).reset_index()
    per_c = per_c[per_c.n >= 3]                              # too few cases to score
    out = [pooled, per_c,
           _pool_vars(pooled, ["sample", "model", "horizon"]).assign(ISO3=ALL_C),
           _pool_vars(per_c, ["sample", "model", "horizon", "ISO3"])]
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 origins per sample (smoke test)")
    ap.add_argument("--rescore", action="store_true",
                    help="skip estimation: rescore results/forecasts.csv and rebuild the page")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    panel = load_panel()
    ins, oos = list(IN_SAMPLE), list(OUT_SAMPLE)
    if args.quick:
        ins, oos = ins[-3:], oos[-3:]
    origins = ins + oos + [CUTOFF]
    print(f"panel: {panel.ISO3.nunique()} countries, {panel.variable.nunique()} variables, "
          f"{panel.year.min()}-{panel.year.max()}; {len(SPECS)} models x {len(origins)} origins")

    if args.rescore:
        fc = pd.read_csv(os.path.join(OUT, "forecasts.csv"))
        ins = sorted(fc.loc[fc["sample"] == "In-sample", "origin"].unique())
        oos = sorted(fc.loc[fc["sample"] == "Out-of-sample", "origin"].unique())
        log_rows = pd.read_csv(os.path.join(OUT, "model_run_log.csv")).to_dict("records")
        return finish(fc, panel, ins, oos, log_rows)

    parts, log_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_spec, i, origins, panel): i for i in range(len(SPECS))}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                df, fails, secs, name = fut.result()
            except Exception:                                    # noqa: BLE001
                df, fails, secs, name = pd.DataFrame(), [traceback.format_exc(limit=2)], 0, "?"
            parts.append(df)
            log_rows.append({"model": SPECS[i][3], "file": SPECS[i][0] + ".py", "class": SPECS[i][1],
                             "spec_name": name, "params": json.dumps(SPECS[i][2], default=str),
                             "forecasts": len(df), "failed_origins": len(fails),
                             "first_error": fails[0] if fails else "", "seconds": round(secs, 1)})
            print(f"  [{len(log_rows):2d}/{len(SPECS)}] {SPECS[i][3]:<34} {len(df):6d} forecasts, "
                  f"{len(fails)} failed origins, {secs:5.0f}s")
    rw = pd.concat([no_change(panel, T).assign(origin=T) for T in origins], ignore_index=True)
    fc = pd.concat(parts + [rw.assign(model=RW)], ignore_index=True)
    fc["sample"] = np.where(fc.origin.isin(ins), "In-sample", np.where(fc.origin.isin(oos),
                            "Out-of-sample", "Final forecast"))
    fc.to_csv(os.path.join(OUT, "forecasts.csv"), index=False)
    finish(fc, panel, ins, oos, log_rows)


def finish(fc, panel, ins, oos, log_rows):
    final = fc[fc["sample"] == "Final forecast"]
    final.to_csv(os.path.join(OUT, "final_forecasts.csv"), index=False)

    ev = add_errors(fc[fc["sample"] != "Final forecast"], panel)
    ev.to_csv(os.path.join(OUT, "forecast_errors.csv"), index=False)
    sc = score(ev)
    sc.to_csv(os.path.join(OUT, "scores.csv"), index=False)
    meta = pd.DataFrame(log_rows).sort_values("model")
    meta.to_csv(os.path.join(OUT, "model_run_log.csv"), index=False)

    from build_html import build_html
    build_html(sc, meta, panel, final, ev, os.path.join(OUT, "leaderboard.html"),
               ins=ins, oos=oos, sources=SOURCE, tcodes={v: TCODE[transformation(v)] for v in TARGETS})
    o = sc[(sc["sample"] == "Out-of-sample") & (sc.variable == ALL_VARS) & (sc.ISO3 == ALL_C) & (sc.horizon == 1)]
    print(o.sort_values("TheilU")[["model", "RMSE", "MAE", "MAPE", "TheilU", "DA", "n"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
