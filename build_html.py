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
               "oos": [int(min(oos)), int(max(oos))], "cutoff": int(max(oos)) + 1}
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
    <div class="card"><div class="chartwrap scroll" id="chart"></div><div class="keys" id="keys"></div></div>
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
let st = {c:"All countries", v:"All variables (median)", h:1, sample:"Out-of-sample",
          picks:D.default.map(m => D.models.indexOf(m))};
try{const s=JSON.parse(localStorage.getItem("ffl2")||"null");
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
    st[key] = key==="h" ? +b.dataset.v : b.dataset.v; render();}); }
seg("horizon", [[1,"1 year"],[2,"2 years"]], "h");
seg("sample", [["In-sample","In-sample"],["Out-of-sample","Out-of-sample"]], "sample");

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

function chart(){
  const box = document.getElementById("chart"), keys = document.getElementById("keys");
  const ok = st.c !== "All countries" && st.v !== "All variables (median)";
  document.getElementById("ctitle").textContent = ok ? `${D.vlabel[st.v]}, ${D.cname[st.c]}` : "Forecasts against outturns";
  if(!ok){ box.innerHTML = `<div class="empty">Choose one country and one variable to see the forecasts against the outturns.</div>`;
    keys.innerHTML=""; document.getElementById("cnote").textContent=""; return; }
  const act = D.A[`${st.c}|${st.v}`] || {}, F = D.F[`${st.c}|${st.v}|${st.h}`] || {};
  const series = [["Outturn","var(--act)",act,true]];
  st.picks.forEach((m,i) => series.push([`${"ABC"[i]}: ${D.models[m]}`, `var(--m${i+1})`, F[m]||{}, false]));
  series.push(["No-change", "var(--rw)", F[RWI]||{}, false]);
  const x0 = D.ins[0], x1 = D.cutoff + 2;
  const pts = series.map(([, , d]) => Object.entries(d).map(([y,v]) => [+y,v]).filter(([y,v]) => y>=x0 && y<=x1 && v!=null).sort((a,b)=>a[0]-b[0]));
  const all = pts.flat().map(p => p[1]);
  if(!all.length){ box.innerHTML = `<div class="empty">No data for this country and variable.</div>`; keys.innerHTML=""; return; }
  // y-range from the outturn, widened by forecasts but capped so one blow-up does not flatten the chart
  const a = pts[0].map(p=>p[1]), amin = Math.min(...a), amax = Math.max(...a), span = (amax-amin)||Math.abs(amax)||1;
  let lo = Math.max(Math.min(...all), amin - span), hi = Math.min(Math.max(...all), amax + span);
  const pad = (hi-lo)*0.06 || 1; lo -= pad; hi += pad;
  const W = 860, H = 300, L = 58, R = 14, T = 12, B = 28;
  const sx = y => L + (y-x0)/(x1-x0)*(W-L-R), sy = v => T + (hi-v)/(hi-lo)*(H-T-B);
  const clip = v => Math.max(lo, Math.min(hi, v));
  const tick = (()=>{const r=(hi-lo)/5, p=Math.pow(10,Math.floor(Math.log10(r))), n=r/p; return (n<1.5?1:n<3.5?2:n<7.5?5:10)*p;})();
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="min-width:560px" role="img" aria-label="Forecasts against outturns">`;
  s += `<rect x="${sx(D.oos[0]+1)}" y="${T}" width="${sx(D.cutoff)-sx(D.oos[0]+1)}" height="${H-T-B}" fill="var(--oos)"/>`;
  s += `<rect x="${sx(D.cutoff)}" y="${T}" width="${sx(x1)-sx(D.cutoff)}" height="${H-T-B}" fill="var(--grid)" opacity=".6"/>`;
  for(let v = Math.ceil(lo/tick)*tick; v <= hi; v += tick){
    s += `<line x1="${L}" x2="${W-R}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/>`;
    s += `<text x="${L-6}" y="${sy(v)+4}" text-anchor="end">${Math.abs(tick)<1 ? v.toFixed(1) : Math.abs(v)>=1e4 ? v.toExponential(1) : v.toFixed(0)}</text>`; }
  for(let y = Math.ceil(x0/5)*5; y <= x1; y += 5) s += `<text x="${sx(y)}" y="${H-8}" text-anchor="middle">${y}</text>`;
  s += `<text x="${(sx(D.oos[0]+1)+sx(D.cutoff))/2}" y="${T+14}" text-anchor="middle">out-of-sample</text>`;
  s += `<text x="${(sx(D.cutoff)+sx(x1))/2}" y="${T+14}" text-anchor="middle">forecast</text>`;
  pts.forEach((p,i) => { if(!p.length) return; const [, col, , isAct] = series[i];
    s += `<polyline fill="none" stroke="${col}" stroke-width="${isAct?2.4:1.6}" ${i===4?'stroke-dasharray="4 3"':''} points="${p.map(([y,v])=>`${sx(y).toFixed(1)},${sy(clip(v)).toFixed(1)}`).join(" ")}"/>`;
    if(isAct){ const [y,v] = p[p.length-1]; s += `<circle cx="${sx(y)}" cy="${sy(clip(v))}" r="3.5" fill="${col}"/>`; } });
  box.innerHTML = s + "</svg>";
  keys.innerHTML = series.map(([n,c],i) => `<span><i class="sw" style="background:${c}"></i>${esc(n)}</span>`).join("");
  document.getElementById("cnote").textContent =
   `Each point is the ${st.h}-year-ahead forecast made ${st.h} year${st.h>1?"s":""} before that year, in ${D.unit[st.v]||"original units"}. Lightly shaded: out-of-sample years ${D.oos[0]+1}–${D.cutoff}. Darker band: forecasts beyond the ${D.cutoff} cut-off, not yet scored. Values far outside the outturn range are clipped to the chart edge.`;
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
  try{localStorage.setItem("ffl2", JSON.stringify(st));}catch(e){}
  document.querySelectorAll("#horizon button").forEach(b => b.setAttribute("aria-pressed", +b.dataset.v===st.h));
  ["A","B","C"].forEach((k,i) => { const inf = D.info[D.models[st.picks[i]]]||{};
    document.getElementById("i"+k).textContent = `${inf.file||""} · ${inf.params||""}`; });
  document.getElementById("mtitle").textContent = `Error matrix · ${D.vlabel[st.v]} · ${D.cname[st.c]}`;
  document.getElementById("mnote").textContent = `${st.h}-year-ahead forecasts. Columns A, B and C are your three models; the no-change forecast is the benchmark that Theil U2 divides by.${st.c==="All countries" ? " All countries pools every country's forecast errors." : ""}`;
  matrix(); chart(); rank();
}
document.getElementById("data").innerHTML = `<thead><tr><th>Variable</th><th style="text-align:left">Source</th><th>FRED-MD tcode</th><th>From</th><th>To</th><th>Countries</th></tr></thead><tbody>` +
  D.data.map(r => `<tr><td>${esc(r.label)}</td><td style="text-align:left">${esc(r.source)}</td><td class="num">${r.tcode}</td><td class="num">${r.first}</td><td class="num">${r.last}</td><td class="num">${r.countries}</td></tr>`).join("") + "</tbody>";
render();
</script>
"""
