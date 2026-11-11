"""Render results/leaderboard.html.

Pick a country, a variable and 3 of the 30 models. The page shows
  - the error matrix: 5 measures x (in-sample | out-of-sample) x 3 models + no-change benchmark
  - forecasts against outturns for that country and variable, 1990-2027
  - the ranking of all 30 models for that country and variable
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

LEVELS = {"CPI", "rGDP", "M2", "USDfx", "REER"}
ALL_C, ALL_V = "All countries", "All variables (median)"
VAR_LABEL = {
    ALL_V: "All variables (median)",
    "infl": "Inflation, CPI % y/y", "CPI": "Consumer price index", "rGDP": "Real GDP",
    "unemp": "Unemployment rate", "strate": "3-month interest rate", "ltrate": "10-year bond yield",
    "USDfx": "Exchange rate, LCU per USD", "REER": "Real effective exchange rate",
    "M2": "Broad money", "CA_GDP": "Current account, % GDP", "gen_govdebt_GDP": "Govt debt, % GDP",
}
UNIT = {"infl": "%", "unemp": "%", "strate": "%", "ltrate": "%", "CA_GDP": "% of GDP",
        "gen_govdebt_GDP": "% of GDP", "CPI": "index", "rGDP": "constant LCU", "M2": "LCU",
        "USDfx": "LCU per USD", "REER": "index"}
COUNTRY = {
    "USA": "United States", "CAN": "Canada", "MEX": "Mexico", "BRA": "Brazil", "DEU": "Germany",
    "FRA": "France", "ITA": "Italy", "ESP": "Spain", "NLD": "Netherlands", "GBR": "United Kingdom",
    "SWE": "Sweden", "NOR": "Norway", "CHE": "Switzerland", "DNK": "Denmark", "JPN": "Japan",
    "KOR": "South Korea", "AUS": "Australia", "NZL": "New Zealand", "IND": "India",
    "ZAF": "South Africa",
}


def _sig(x, k=5):
    if x is None or not np.isfinite(x):
        return None
    return float(f"{x:.{k}g}")


def build_html(scores: pd.DataFrame, meta: pd.DataFrame, panel: pd.DataFrame, final: pd.DataFrame,
               errors: pd.DataFrame, path: str, ins, oos, sources: dict, tcodes: dict) -> None:
    rw = "No-change (random walk)"
    models = sorted(m for m in scores.model.unique() if m != rw)
    mi = {m: i for i, m in enumerate(models + [rw])}

    # scores: key "iso|var|h" -> {sample: {model_index: [RMSE, MAE, MAPE, TheilU, DA, n]}}
    S: dict = {}
    for r in scores.itertuples(index=False):
        k = f"{r.ISO3}|{r.variable}|{r.horizon}"
        S.setdefault(k, {}).setdefault(r.sample, {})[mi[r.model]] = [
            _sig(r.RMSE, 4), _sig(r.MAE, 4), _sig(r.MAPE, 4), _sig(r.TheilU, 4), _sig(r.DA, 3), int(r.n)]

    # actual series and forecasts by target year, for the chart
    y0 = min(ins) - 5
    act = {}
    for (iso, v), g in panel[panel.year >= y0].groupby(["ISO3", "variable"]):
        if v in tcodes:
            act[f"{iso}|{v}"] = {int(y): _sig(x) for y, x in zip(g.year, g.value)}
    fc_all = pd.concat([errors[["model", "ISO3", "variable", "horizon", "year", "point"]],
                        final[["model", "ISO3", "variable", "horizon", "year", "point"]]])
    F: dict = {}
    for r in fc_all.itertuples(index=False):
        F.setdefault(f"{r.ISO3}|{r.variable}|{r.horizon}", {}).setdefault(mi[r.model], {})[int(r.year)] = _sig(r.point)

    # fan chart: quantiles of each model's out-of-sample errors (all countries pooled),
    # per variable and horizon. Errors are % for level variables, pp for rates.
    QS = [5, 15, 25, 35, 50, 65, 75, 85, 95]
    Q = {}
    oe = errors[errors["sample"] == "Out-of-sample"]
    for (m, v, h), g in oe.groupby(["model", "variable", "horizon"]):
        if len(g) >= 20:
            Q[f"{mi[m]}|{v}|{int(h)}"] = [_sig(x, 4) for x in np.percentile(g.error, QS)]

    o = scores[(scores["sample"] == "Out-of-sample") & (scores.variable == ALL_V)
               & (scores.ISO3 == ALL_C) & (scores.horizon == 1) & (scores.model != rw)]
    default = list(o.sort_values("TheilU").model.head(3))
    info = {r["model"]: {"file": r["file"], "params": r["params"]} for r in meta.to_dict("records")}
    variables = [ALL_V] + [v for v in VAR_LABEL if v in tcodes]
    countries = [ALL_C] + sorted({c for c in scores.ISO3 if c != ALL_C}, key=lambda c: COUNTRY.get(c, c))
    cover = (panel.groupby("variable").agg(first=("year", "min"), last=("year", "max"),
                                           countries=("ISO3", "nunique")).reset_index())
    data_rows = [{"label": VAR_LABEL[v], "source": sources.get(v, ""), "tcode": tcodes.get(v),
                  "first": int(r["first"]), "last": int(r["last"]), "countries": int(r["countries"])}
                 for v in variables[1:] for r in cover[cover.variable == v].to_dict("records")]
    payload = {"S": S, "A": act, "F": F, "models": models + [rw], "nModels": len(models),
               "default": default, "info": info, "variables": variables, "vlabel": VAR_LABEL,
               "unit": UNIT, "countries": countries, "cname": {**COUNTRY, ALL_C: ALL_C},
               "data": data_rows, "ins": [int(min(ins)), int(max(ins))],
               "oos": [int(min(oos)), int(max(oos))], "cutoff": int(max(oos)) + 1,
               "Q": Q, "levels": sorted(LEVELS)}
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


TEMPLATE = r"""<title>Forecast Factory Leaderboard</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&display=swap">
<style>
/* Layout: one working column. Selectors -> error matrix -> forecast chart -> ranking -> data notes */
:root{
  --bg:#f5f7f8; --panel:#ffffff; --fg:#17212b; --muted:#5b6876; --line:#d9e0e6; --grid:#e8edf1;
  --accent:#0d6b73; --best:#dff1ec; --bestfg:#0b5c45; --oos:#eef3f6;
  --m1:#0d6b73; --m2:#b4532a; --m3:#5a4fb3; --rw:#8a96a3; --act:#17212b;
  --sans:"IBM Plex Sans",system-ui,sans-serif; --cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#11171c; --panel:#182128; --fg:#e4eaef; --muted:#93a1ae; --line:#2b3741; --grid:#222d36;
  --accent:#4fb8bf; --best:#173b33; --bestfg:#7fdcbc; --oos:#1c2730;
  --m1:#4fb8bf; --m2:#e5875c; --m3:#a39af0; --rw:#7d8a96; --act:#e4eaef; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#11171c; --panel:#182128; --fg:#e4eaef; --muted:#93a1ae; --line:#2b3741; --grid:#222d36;
  --accent:#4fb8bf; --best:#173b33; --bestfg:#7fdcbc; --oos:#1c2730;
  --m1:#4fb8bf; --m2:#e5875c; --m3:#a39af0; --rw:#7d8a96; --act:#e4eaef; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:14px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-block:28px 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:28px}
h1{font-family:var(--cond);font-size:28px;margin:0;letter-spacing:.01em;text-wrap:balance}
h2{font-family:var(--cond);font-size:18px;margin:0 0 4px;text-wrap:balance}
.lede{color:var(--muted);max-width:76ch;margin:6px 0 0}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.controls{display:grid;gap:14px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}
.row{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.pick{display:grid;gap:4px;min-width:0;align-content:start}
.pick label{font-size:12px;font-weight:600;display:flex;align-items:center;gap:6px}
.dot{width:10px;height:10px;border-radius:50%;flex:none}
select{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);width:100%;min-width:0}
select:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.meta{font-family:var(--mono);font-size:11px;color:var(--muted);overflow-wrap:anywhere}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{font:inherit;font-size:13px;border:0;background:var(--panel);color:var(--fg);padding:6px 12px;cursor:pointer}
.seg button+button{border-left:1px solid var(--line)}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--bg);font-weight:600}
.seglabel{font-size:12px;color:var(--muted);margin-right:6px}
.toggles{display:flex;flex-wrap:wrap;gap:16px;align-items:center}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font-size:12px;font-weight:600;color:var(--muted);vertical-align:bottom}
thead tr.grp th{font-family:var(--cond);font-size:13px;color:var(--fg);text-align:center;border-bottom:2px solid var(--line)}
td.num{font-family:var(--mono);font-size:13px}
td.best{background:var(--best);color:var(--bestfg);font-weight:500}
td.rw{color:var(--muted)}
.colA{box-shadow:inset 0 3px 0 var(--m1)} .colB{box-shadow:inset 0 3px 0 var(--m2)} .colC{box-shadow:inset 0 3px 0 var(--m3)} .colR{box-shadow:inset 0 3px 0 var(--rw)}
.gstart{border-left:1px solid var(--line)}
td.mname b{font-family:var(--mono);font-weight:500}
td.mname span{display:block;font-size:12px;color:var(--muted)}
.na{color:var(--muted)}
.note{color:var(--muted);font-size:13px;max-width:82ch;margin:0 0 10px}
.chartwrap{padding:12px 12px 4px}
svg text{font-family:var(--mono);font-size:11px;fill:var(--muted)}
.keys{display:flex;flex-wrap:wrap;gap:6px 18px;padding:4px 14px 14px;font-size:12px}
.keys span{display:inline-flex;align-items:center;gap:6px}
.sw{width:16px;height:3px;border-radius:2px;display:inline-block}
.rank td:nth-child(2),.rank th:nth-child(2){text-align:left}
.sel{color:var(--accent);font-weight:600}
.bar{display:inline-block;height:6px;border-radius:3px;background:var(--accent);opacity:.55;vertical-align:middle;margin-left:8px}
.empty{padding:28px 16px;color:var(--muted);text-align:center}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow" id="eyebrow"></div>
    <h1>Forecast Factory Leaderboard</h1>
    <p class="lede" id="lede"></p>
  </header>

  <section class="controls" aria-label="Choose country, variable and models">
    <div class="row">
      <div class="pick"><label for="country">Country</label><select id="country"></select></div>
      <div class="pick"><label for="variable">Variable</label><select id="variable"></select></div>
      <div class="pick"><label>Horizon</label><span class="seg" id="horizon"></span></div>
    </div>
    <div class="row">
      <div class="pick"><label for="mA"><span class="dot" style="background:var(--m1)"></span>Model A</label><select id="mA"></select><div class="meta" id="iA"></div></div>
      <div class="pick"><label for="mB"><span class="dot" style="background:var(--m2)"></span>Model B</label><select id="mB"></select><div class="meta" id="iB"></div></div>
      <div class="pick"><label for="mC"><span class="dot" style="background:var(--m3)"></span>Model C</label><select id="mC"></select><div class="meta" id="iC"></div></div>
    </div>
  </section>

  <section>
    <h2 id="mtitle">Error matrix</h2>
    <p class="note" id="mnote"></p>
    <div class="card scroll"><table id="matrix"></table></div>
  </section>

  <section>
    <h2 id="ctitle">Forecasts against outturns</h2>
    <p class="note" id="cnote"></p>
    <div class="toggles" style="margin-bottom:10px">
      <div><span class="seglabel">Period</span><span class="seg" id="period"></span></div>
      <div id="scalebox"><span class="seglabel">Scale</span><span class="seg" id="scale"></span></div>
    </div>
    <div class="card"><div class="chartwrap scroll" id="chart"></div><div class="keys" id="keys"></div></div>
  </section>

  <section>
    <h2 id="ftitle">Fan chart</h2>
    <p class="note" id="fnote"></p>
    <div class="toggles" style="margin-bottom:10px"><div><span class="seglabel">Model</span><span class="seg" id="fanmodel"></span></div></div>
    <div class="card"><div class="chartwrap scroll" id="fan"></div><div class="keys" id="fkeys"></div></div>
  </section>

  <section>
    <h2 id="rtitle">All 30 models</h2>
    <div class="toggles" style="margin-bottom:10px"><div><span class="seglabel">Rank on</span><span class="seg" id="sample"></span></div></div>
    <div class="card scroll"><table id="rank" class="rank"></table></div>
  </section>

  <section class="card" style="padding:14px 16px">
    <h2>How to read the measures</h2>
    <p class="note"><b>RMSE</b> root mean squared error and <b>MAE</b> mean absolute error: level variables in % (100·log of forecast over outturn), rates in percentage points. <b>MAPE</b> median absolute percentage error (rates only where |outturn| ≥ 0.5). <b>Theil U2</b> RMSE relative to the no-change forecast on the same cases; below 1 beats it. <b>DA</b> directional accuracy, the share of forecasts that get the sign of the change right; 0.5 is a coin flip. Shaded cell = best of the three models. Lower is better except for DA. Single-country scores rest on few forecasts (about 10 out-of-sample per horizon), so read them with care.</p>
  </section>

  <section>
    <h2>Data</h2>
    <p class="note">Transformation codes follow FRED-MD (McCracken &amp; Ng 2016): 1 = level, 2 = first difference, 5 = log difference. Models forecast the transformed series and map forecasts back to original units before scoring.</p>
    <div class="card scroll"><table id="data"></table></div>
  </section>
</div>

<script>
const D = /*__DATA__*/null;
const MEAS = [["RMSE",-1],["MAE",-1],["MAPE %",-1],["Theil U2",-1],["DA",1]];
const RWI = D.models.length - 1;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let st = {c:"All countries", v:"All variables (median)", h:1, sample:"Out-of-sample", period:"recent", scale:"linear", fan:0,
          picks:D.default.map(m => D.models.indexOf(m))};
try{const s=JSON.parse(localStorage.getItem("ffl3")||"null");
  if(s && s.picks && s.picks.every(i => i>=0 && i<D.nModels) && D.countries.includes(s.c) && D.variables.includes(s.v)) st = s;}catch(e){}

document.getElementById("eyebrow").textContent = `Forecast Factory · model evaluation · cut-off ${D.cutoff}`;
document.getElementById("lede").textContent =
 `${D.nModels} model specifications, estimated on the same panel of ${D.countries.length-1} countries (World Bank, IMF, FRED; from 1960 where available). Each model is re-estimated at every forecast origin with data up to that year only. In-sample: origins ${D.ins[0]}–${D.ins[1]}. Out-of-sample: origins ${D.oos[0]}–${D.oos[1]}, scored against outturns up to ${D.cutoff}.`;

const fill = (id, opts, val) => { const s=document.getElementById(id);
  s.innerHTML = opts.map(([v,l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join(""); s.value = val; return s; };
fill("country", D.countries.map(c => [c, D.cname[c]||c]), st.c).addEventListener("change", e => {st.c=e.target.value; render();});
fill("variable", D.variables.map(v => [v, D.vlabel[v]||v]), st.v).addEventListener("change", e => {st.v=e.target.value; render();});
["A","B","C"].forEach((k,i) => fill("m"+k, D.models.slice(0,D.nModels).map((m,j) => [j,m]), st.picks[i])
  .addEventListener("change", e => {st.picks[i]=+e.target.value; render();}));
function seg(id, opts, key){ const el=document.getElementById(id);
  el.innerHTML = opts.map(([v,l]) => `<button type="button" data-v="${v}">${l}</button>`).join("");
  el.addEventListener("click", e => {const b=e.target.closest("button"); if(!b) return;
    st[key] = (key==="h"||key==="fan") ? +b.dataset.v : b.dataset.v; render();}); }
seg("horizon", [[1,"1 year"],[2,"2 years"]], "h");
seg("sample", [["In-sample","In-sample"],["Out-of-sample","Out-of-sample"]], "sample");
seg("period", [["recent",`Since ${D.oos[0]-10}`],["all",`Since ${D.ins[0]}`]], "period");
seg("scale", [["linear","Linear"],["log","Log"]], "scale");
seg("fanmodel", [[0,"A"],[1,"B"],[2,"C"]], "fan");

const score = (sample, mi) => ((D.S[`${st.c}|${st.v}|${st.h}`]||{})[sample]||{})[mi];
function fmt(x, k){ if(x==null) return '<span class="na">–</span>';
  return (k===3||k===4) ? x.toFixed(2) : Math.abs(x)>=100 ? x.toFixed(0) : x.toFixed(2); }

function matrix(){
  const P = st.picks, cols = [...P, RWI], cls = ["colA","colB","colC","colR"], S = ["In-sample","Out-of-sample"];
  const any = S.some(s => cols.some(m => score(s,m)));
  const t = document.getElementById("matrix");
  if(!any){ t.innerHTML = `<tbody><tr><td class="empty">No forecasts to score for ${esc(D.vlabel[st.v])} in ${esc(D.cname[st.c])}. Pick another variable or country.</td></tr></tbody>`; return; }
  let h = `<thead><tr class="grp"><th></th>${S.map(s => `<th colspan="4" class="gstart">${s}</th>`).join("")}</tr><tr><th>Measure</th>`;
  S.forEach(() => cols.forEach((m,i) => h += `<th class="${cls[i]}${i===0?" gstart":""}" title="${esc(D.models[m])}">${i<3?"ABC"[i]:"No-change"}</th>`));
  h += "</tr></thead><tbody>";
  MEAS.forEach(([name,dir],k) => {
    h += `<tr><td>${name}</td>`;
    S.forEach(s => { const vals = P.map(m => (score(s,m)||[])[k] ?? null), ok = vals.filter(x => x!=null);
      const best = ok.length>1 ? (dir<0 ? Math.min(...ok) : Math.max(...ok)) : null;
      vals.forEach((x,i) => h += `<td class="num${i===0?" gstart":""}${x!=null&&x===best?" best":""}">${fmt(x,k)}</td>`);
      const r = (score(s,RWI)||[])[k]; h += `<td class="num rw">${k===4 ? "–" : fmt(r ?? null,k)}</td>`; });
    h += "</tr>"; });
  h += `<tr><td>Forecasts scored</td>`;
  S.forEach(s => cols.forEach((m,i) => h += `<td class="num rw${i===0?" gstart":""}">${(score(s,m)||[])[5] ?? "–"}</td>`));
  t.innerHTML = h + "</tr></tbody>";
}

// ---------- shared chart helpers
const isLevel = v => D.levels.includes(v);
function niceStep(range, n){ const r = range/n, p = Math.pow(10, Math.floor(Math.log10(r))), m = r/p;
  return (m<1.5?1:m<3.5?2:m<7.5?5:10)*p; }
function fmtTick(v, step){                        // compact, unit-free tick label
  const a = Math.abs(v);
  const sc = a>=1e12?[1e12,"tn"]:a>=1e9?[1e9,"bn"]:a>=1e6?[1e6,"M"]:a>=1e4?[1e3,"k"]:[1,""];
  const st = step/sc[0], d = st>=1 ? 0 : Math.min(3, Math.ceil(-Math.log10(st)));
  return (v/sc[0]).toFixed(d) + sc[1]; }
function axes(lo, hi, log, x0, x1, W, H, L, R, T, B, unit){
  const f = log ? Math.log : (x => x);
  const sx = y => L + (y-x0)/(x1-x0)*(W-L-R), sy = v => T + (f(hi)-f(Math.max(v, log?1e-12:-Infinity)))/(f(hi)-f(lo))*(H-T-B);
  let ticks = [];
  if(log){ const e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi));
    for(let e=e0; e<=e1; e++) for(const m of (e1-e0>3?[1]:[1,2,5])){ const v = m*Math.pow(10,e); if(v>=lo && v<=hi) ticks.push([v, fmtTick(v, v/10)]); }
    if(ticks.length < 3){ const st = niceStep(hi-lo, 5); ticks = [];          // narrow range: linear ticks on the log axis
      for(let v=Math.ceil(lo/st)*st; v<=hi; v+=st) ticks.push([v, fmtTick(v, st)]); } }
  else { const st = niceStep(hi-lo, 5); for(let v=Math.ceil(lo/st)*st; v<=hi+1e-9*st; v+=st) ticks.push([v, fmtTick(Math.abs(v)<st/2?0:v, st)]); }
  let s = "";
  ticks.forEach(([v,l]) => { s += `<line x1="${L}" x2="${W-R}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/><text x="${L-8}" y="${sy(v)+4}" text-anchor="end">${l}</text>`; });
  const yStep = (x1-x0) > 25 ? 5 : (x1-x0) > 10 ? 2 : 1;
  for(let y = Math.ceil(x0/yStep)*yStep; y <= x1; y += yStep) s += `<text x="${sx(y)}" y="${H-8}" text-anchor="middle">${y}</text>`;
  s += `<text transform="translate(14 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle">${esc(unit)}${log?" · log scale":""}</text>`;
  s += `<line x1="${L}" x2="${L}" y1="${T}" y2="${H-B}" stroke="var(--line)"/>`;
  return {sx, sy, s};
}
// y-range driven by the outturn; forecasts may widen it by at most half its span
function yRange(actual, other, log){
  const amin = Math.min(...actual), amax = Math.max(...actual);
  if(log){ const lo0 = Math.max(Math.min(amin, ...other.filter(v=>v>0)), amin/3), hi0 = Math.min(Math.max(amax, ...other), amax*3);
    return [lo0/1.15, hi0*1.15]; }
  const span = (amax-amin) || Math.abs(amax) || 1;
  let lo = Math.max(Math.min(amin, ...other), amin - span/2), hi = Math.min(Math.max(amax, ...other), amax + span/2);
  const pad = (hi-lo)*0.08; return [lo-pad, hi+pad];
}
const series = (d, x0, x1) => Object.entries(d||{}).map(([y,v]) => [+y, v]).filter(([y,v]) => y>=x0 && y<=x1 && v!=null).sort((a,b)=>a[0]-b[0]);

function chart(){
  const box = document.getElementById("chart"), keys = document.getElementById("keys");
  const ok = st.c !== "All countries" && st.v !== "All variables (median)";
  document.getElementById("ctitle").textContent = ok ? `${D.vlabel[st.v]}, ${D.cname[st.c]}` : "Forecasts against outturns";
  document.querySelectorAll("#period button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v===st.period));
  document.querySelectorAll("#scale button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v===st.scale));
  document.getElementById("scalebox").hidden = !ok || !isLevel(st.v);
  if(!ok){ box.innerHTML = `<div class="empty">Choose one country and one variable to see the forecasts against the outturns.</div>`;
    keys.innerHTML=""; document.getElementById("cnote").textContent=""; return; }
  const log = isLevel(st.v) && st.scale === "log";
  const x0 = st.period === "all" ? D.ins[0] : D.oos[0]-10, x1 = D.cutoff + 2;
  const act = D.A[`${st.c}|${st.v}`] || {}, F = D.F[`${st.c}|${st.v}|${st.h}`] || {};
  const defs = [["Outturn","var(--act)",act,2.6,""]];
  st.picks.forEach((m,i) => defs.push([`${"ABC"[i]}: ${D.models[m]}`, `var(--m${i+1})`, F[m], 1.7, ""]));
  defs.push(["No-change", "var(--rw)", F[RWI], 1.4, "4 3"]);
  const pts = defs.map(d => series(d[2], x0, x1).filter(([,v]) => !log || v>0));
  if(!pts[0].length){ box.innerHTML = `<div class="empty">No outturn data for this country and variable.</div>`; keys.innerHTML=""; return; }
  const [lo, hi] = yRange(pts[0].map(p=>p[1]), pts.slice(1).flat().map(p=>p[1]), log);
  const W = 860, H = 340, L = 66, R = 14, T = 14, B = 30;
  const {sx, sy, s: ax} = axes(lo, hi, log, x0, x1, W, H, L, R, T, B, D.unit[st.v]||"");
  const clip = v => Math.max(lo, Math.min(hi, v));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="min-width:560px" role="img" aria-label="Forecasts against outturns">`;
  s += `<rect x="${sx(D.oos[0]+1)}" y="${T}" width="${sx(D.cutoff)-sx(D.oos[0]+1)}" height="${H-T-B}" fill="var(--oos)"/>`;
  s += `<rect x="${sx(D.cutoff)}" y="${T}" width="${sx(x1)-sx(D.cutoff)}" height="${H-T-B}" fill="var(--grid)" opacity=".7"/>`;
  s += ax;
  s += `<text x="${(sx(D.oos[0]+1)+sx(D.cutoff))/2}" y="${T+14}" text-anchor="middle">out-of-sample</text>`;
  s += `<text x="${(sx(D.cutoff)+sx(x1))/2}" y="${T+14}" text-anchor="middle">forecast</text>`;
  let clipped = 0;
  pts.forEach((p,i) => { if(!p.length) return; const [, col, , w, dash] = defs[i];
    clipped += p.filter(([,v]) => v<lo || v>hi).length;
    s += `<polyline fill="none" stroke="${col}" stroke-width="${w}" stroke-linejoin="round" ${dash?`stroke-dasharray="${dash}"`:""} points="${p.map(([y,v])=>`${sx(y).toFixed(1)},${sy(clip(v)).toFixed(1)}`).join(" ")}"/>`;
    if(i===0) p.forEach(([y,v]) => s += `<circle cx="${sx(y).toFixed(1)}" cy="${sy(v).toFixed(1)}" r="2.2" fill="${col}"/>`); });
  box.innerHTML = s + "</svg>";
  keys.innerHTML = defs.map(([n,c,,,dash]) => `<span><i class="sw" style="background:${c}${dash?";opacity:.7":""}"></i>${esc(n)}</span>`).join("");
  document.getElementById("cnote").textContent =
   `${st.h}-year-ahead forecasts, each plotted at the year it targets. Lightly shaded: out-of-sample years ${D.oos[0]+1}–${D.cutoff}. Darker band: forecasts beyond the ${D.cutoff} cut-off, not yet scored.` +
   (clipped ? ` ${clipped} forecast value${clipped>1?"s lie":" lies"} far outside the outturn range and ${clipped>1?"are":"is"} pinned to the chart edge so the outturn stays readable.` : "");
}

// Fan chart: point path of one model from the last outturn, with bands built from the
// quantiles of that model's out-of-sample errors (pooled over countries) at h = 1 and 2.
const BANDS = [[0,8,"90%",.16],[1,7,"70%",.24],[2,6,"50%",.34],[3,5,"30%",.46]];
function fan(){
  const box = document.getElementById("fan"), keys = document.getElementById("fkeys");
  document.querySelectorAll("#fanmodel button").forEach(b => b.setAttribute("aria-pressed", +b.dataset.v===st.fan));
  const ok = st.c !== "All countries" && st.v !== "All variables (median)";
  const mi = st.picks[st.fan], col = `var(--m${st.fan+1})`;
  document.getElementById("ftitle").textContent = ok ? `Fan chart · ${D.vlabel[st.v]}, ${D.cname[st.c]}` : "Fan chart";
  const empty = msg => { box.innerHTML = `<div class="empty">${msg}</div>`; keys.innerHTML = ""; document.getElementById("fnote").textContent = ""; };
  if(!ok) return empty("Choose one country and one variable to see the fan chart.");
  const act = series(D.A[`${st.c}|${st.v}`], D.cutoff-15, D.cutoff);
  if(!act.length) return empty("No outturn data for this country and variable.");
  const [ya, va] = act[act.length-1], lvl = isLevel(st.v), log = lvl && st.scale === "log";
  // central path and bands for h = 1, 2 (target years cutoff+1, cutoff+2)
  const path = [[ya, va]], bands = BANDS.map(() => [[ya, va, va]]);
  for(const h of [1,2]){
    const y = D.cutoff + h, f = ((D.F[`${st.c}|${st.v}|${h}`]||{})[mi]||{})[y], q = D.Q[`${mi}|${st.v}|${h}`];
    if(f==null || !q) continue;
    path.push([y, f]);
    // error e = forecast vs outturn  ->  outturn = f*exp(-e/100) (levels) or f - e (rates)
    const out = e => lvl ? f*Math.exp(-e/100) : f - e;
    BANDS.forEach(([a,b],k) => { const u = out(q[a]), w = out(q[b]); bands[k].push([y, Math.min(u,w), Math.max(u,w)]); });
  }
  if(path.length < 2) return empty(`${esc(D.models[mi])} has no ${D.cutoff+1}–${D.cutoff+2} forecast for this country and variable. Try another model.`);
  const x0 = D.cutoff-15, x1 = D.cutoff+2;
  const allv = [...bands.flat().flatMap(r => [r[1], r[2]]), ...path.map(p=>p[1])];
  const [lo, hi] = yRange(act.map(p=>p[1]), allv.filter(v => !log || v>0), log);
  const W = 860, H = 340, L = 66, R = 70, T = 14, B = 30;
  const {sx, sy, s: ax} = axes(lo, hi, log, x0, x1, W, H, L, R, T, B, D.unit[st.v]||"");
  const clip = v => Math.max(lo, Math.min(hi, v));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="min-width:560px" role="img" aria-label="Fan chart">`;
  s += `<rect x="${sx(ya)}" y="${T}" width="${sx(x1)-sx(ya)}" height="${H-T-B}" fill="var(--oos)"/>` + ax;
  BANDS.forEach(([, , lab, op], k) => { const b = bands[k]; if(b.length < 2) return;
    const up = b.map(([y,,u]) => `${sx(y).toFixed(1)},${sy(clip(u)).toFixed(1)}`), dn = b.slice().reverse().map(([y,l]) => `${sx(y).toFixed(1)},${sy(clip(l)).toFixed(1)}`);
    s += `<polygon points="${up.concat(dn).join(" ")}" fill="${col}" fill-opacity="${op}" stroke="none"/>`;
    const [yl, l, u] = b[b.length-1];
    if(k===0 || k===2){ s += `<text x="${sx(yl)+6}" y="${sy(clip(u))+4}">${lab}</text>`; } });
  s += `<polyline fill="none" stroke="var(--act)" stroke-width="2.6" stroke-linejoin="round" points="${act.map(([y,v]) => `${sx(y).toFixed(1)},${sy(v).toFixed(1)}`).join(" ")}"/>`;
  act.forEach(([y,v]) => s += `<circle cx="${sx(y).toFixed(1)}" cy="${sy(v).toFixed(1)}" r="2.2" fill="var(--act)"/>`);
  s += `<polyline fill="none" stroke="${col}" stroke-width="2.2" stroke-dasharray="5 3" points="${path.map(([y,v]) => `${sx(y).toFixed(1)},${sy(clip(v)).toFixed(1)}`).join(" ")}"/>`;
  path.slice(1).forEach(([y,v]) => { s += `<circle cx="${sx(y)}" cy="${sy(clip(v))}" r="3.5" fill="${col}"/><text x="${sx(y)+6}" y="${sy(clip(v))-6}" style="fill:var(--fg)">${fmtTick(v, Math.abs(v)>=100?1:0.01)}</text>`; });
  box.innerHTML = s + "</svg>";
  keys.innerHTML = `<span><i class="sw" style="background:var(--act)"></i>Outturn</span><span><i class="sw" style="background:${col}"></i>${esc(D.models[mi])}, central forecast</span>` +
    BANDS.slice().reverse().map(([, , lab, op]) => `<span><i class="sw" style="height:10px;background:${col};opacity:${op+.1}"></i>${lab} band</span>`).join("");
  const nq = [1,2].filter(h => D.Q[`${mi}|${st.v}|${h}`]).length;
  document.getElementById("fnote").textContent =
   `Central path: ${D.models[mi]}'s forecasts for ${D.cutoff+1} and ${D.cutoff+2}, made from data up to ${D.cutoff}. Bands: the range of this model's own out-of-sample errors for ${D.vlabel[st.v].toLowerCase()} (${D.oos[0]+1}–${D.cutoff}, all countries pooled), applied to the central path. The 90% band covers the outturn in 9 out of 10 past cases.` +
   (nq < 2 ? " One horizon has too few past errors for bands." : "");
}

function rank(){
  document.querySelectorAll("#sample button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v===st.sample));
  document.getElementById("rtitle").textContent = `All ${D.nModels} models · ${D.vlabel[st.v]} · ${D.cname[st.c]}`;
  const rk = D.models.map((m,i) => ({m,i,r:score(st.sample,i)})).filter(x => x.r && x.r[3]!=null).sort((a,b) => a.r[3]-b.r[3]);
  const t = document.getElementById("rank");
  if(!rk.length){ t.innerHTML = `<tbody><tr><td class="empty">No scores for this selection.</td></tr></tbody>`; return; }
  const mx = Math.max(...rk.map(x => Math.min(x.r[3],3)), 1);
  let h = `<thead><tr><th>#</th><th>Model</th><th>Theil U2</th><th>RMSE</th><th>MAE</th><th>MAPE %</th><th>DA</th><th>n</th></tr></thead><tbody>`;
  rk.forEach((x,j) => { const sel = st.picks.includes(x.i), inf = D.info[x.m]||{};
    h += `<tr><td class="num">${j+1}</td><td class="mname"><b class="${sel?"sel":""}">${esc(x.m)}</b><span>${esc(inf.file||"benchmark")}</span></td>
      <td class="num">${fmt(x.r[3],3)}<span class="bar" style="width:${Math.round(60*Math.min(x.r[3],3)/mx)}px"></span></td>
      <td class="num">${fmt(x.r[0],0)}</td><td class="num">${fmt(x.r[1],1)}</td><td class="num">${fmt(x.r[2],2)}</td><td class="num">${fmt(x.r[4],4)}</td><td class="num">${x.r[5]}</td></tr>`; });
  t.innerHTML = h + "</tbody>";
}

function render(){
  try{localStorage.setItem("ffl3", JSON.stringify(st));}catch(e){}
  document.querySelectorAll("#horizon button").forEach(b => b.setAttribute("aria-pressed", +b.dataset.v===st.h));
  ["A","B","C"].forEach((k,i) => { const inf = D.info[D.models[st.picks[i]]]||{};
    document.getElementById("i"+k).textContent = `${inf.file||""} · ${inf.params||""}`; });
  document.getElementById("mtitle").textContent = `Error matrix · ${D.vlabel[st.v]} · ${D.cname[st.c]}`;
  document.getElementById("mnote").textContent = `${st.h}-year-ahead forecasts. Columns A, B and C are your three models; the no-change forecast is the benchmark that Theil U2 divides by.${st.c==="All countries" ? " All countries pools every country's forecast errors." : ""}`;
  matrix(); chart(); fan(); rank();
}
document.getElementById("data").innerHTML = `<thead><tr><th>Variable</th><th style="text-align:left">Source</th><th>FRED-MD tcode</th><th>From</th><th>To</th><th>Countries</th></tr></thead><tbody>` +
  D.data.map(r => `<tr><td>${esc(r.label)}</td><td style="text-align:left">${esc(r.source)}</td><td class="num">${r.tcode}</td><td class="num">${r.first}</td><td class="num">${r.last}</td><td class="num">${r.countries}</td></tr>`).join("") + "</tbody>";
render();
</script>
"""
