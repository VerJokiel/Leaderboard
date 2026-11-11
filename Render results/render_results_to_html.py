"""Render results/leaderboard.html: pick 3 of the 30 models, compare five error measures."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

VAR_LABEL = {
    "All variables (median)": "All variables (median across variables)",
    "infl": "Inflation, CPI % y/y", "CPI": "Consumer price index", "rGDP": "Real GDP",
    "unemp": "Unemployment rate", "strate": "3-month interest rate", "ltrate": "10-year bond yield",
    "USDfx": "Exchange rate, LCU per USD", "REER": "Real effective exchange rate",
    "M2": "Broad money", "CA_GDP": "Current account, % GDP", "gen_govdebt_GDP": "Govt debt, % GDP",
}


def _clean(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def build_html(scores: pd.DataFrame, meta: pd.DataFrame, panel: pd.DataFrame, final: pd.DataFrame,
               path: str, ins, oos, sources: dict, tcodes: dict) -> None:
    rw = "No-change (random walk)"
    sc = scores.copy()
    recs = [{k: _clean(v) for k, v in r.items()} for r in
            sc[["sample", "model", "variable", "horizon", "RMSE", "MAE", "MAPE", "TheilU", "DA", "n"]]
            .round(4).to_dict("records")]
    models = sorted(m for m in sc.model.unique() if m != rw)
    info = {r["model"]: {"file": r["file"], "params": r["params"], "forecasts": int(r["forecasts"]),
                         "failed": int(r["failed_origins"])} for r in meta.to_dict("records")}
    # default picks: the three best on out-of-sample Theil U, h=1
    o = sc[(sc["sample"] == "Out-of-sample") & (sc.variable == "All variables (median)")
           & (sc.horizon == 1) & (sc.model != rw)].sort_values("TheilU")
    default = list(o.model.head(3))
    variables = ["All variables (median)"] + [v for v in VAR_LABEL if v in set(sc.variable)
                                               and v != "All variables (median)"]
    cover = (panel.groupby("variable").agg(first=("year", "min"), last=("year", "max"),
                                           countries=("ISO3", "nunique")).reset_index())
    data_rows = [{"var": v, "label": VAR_LABEL.get(v, v), "source": sources.get(v, ""),
                  "tcode": tcodes.get(v), "first": int(r["first"]), "last": int(r["last"]),
                  "countries": int(r["countries"])}
                 for v in variables[1:] for r in cover[cover.variable == v].to_dict("records")]
    payload = {"scores": recs, "models": models, "rw": rw, "default": default, "info": info,
               "variables": variables, "labels": VAR_LABEL, "data": data_rows,
               "ins": [min(ins), max(ins)], "oos": [min(oos), max(oos)],
               "n_countries": int(panel.ISO3.nunique())}
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


TEMPLATE = r"""<title>Forecast Factory Leaderboard</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&display=swap">
<style>
/* Layout: one working column — controls, then the comparison matrix, then the full ranking and data notes */
:root{
  --bg:#f5f7f8; --panel:#ffffff; --fg:#17212b; --muted:#5b6876; --line:#d9e0e6;
  --accent:#0d6b73; --best:#dff1ec; --bestfg:#0b5c45; --warn:#9a5b00;
  --m1:#0d6b73; --m2:#b4532a; --m3:#5a4fb3;
  --sans:"IBM Plex Sans",system-ui,sans-serif; --cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#11171c; --panel:#182128; --fg:#e4eaef; --muted:#93a1ae; --line:#2b3741;
  --accent:#4fb8bf; --best:#173b33; --bestfg:#7fdcbc; --warn:#e3a54a;
  --m1:#4fb8bf; --m2:#e5875c; --m3:#a39af0; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#11171c; --panel:#182128; --fg:#e4eaef; --muted:#93a1ae; --line:#2b3741;
  --accent:#4fb8bf; --best:#173b33; --bestfg:#7fdcbc; --warn:#e3a54a;
  --m1:#4fb8bf; --m2:#e5875c; --m3:#a39af0; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:14px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-block:28px 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:28px}
h1{font-family:var(--cond);font-size:28px;margin:0;letter-spacing:.01em;text-wrap:balance}
h2{font-family:var(--cond);font-size:18px;margin:0 0 4px}
.lede{color:var(--muted);max-width:72ch;margin:6px 0 0}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.pick{display:grid;gap:4px;min-width:0}
.pick label{font-size:12px;font-weight:600;display:flex;align-items:center;gap:6px}
.dot{width:10px;height:10px;border-radius:50%;flex:none}
select{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg);width:100%;min-width:0}
select:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.meta{font-family:var(--mono);font-size:11px;color:var(--muted);overflow-wrap:anywhere}
.toggles{display:flex;flex-wrap:wrap;gap:16px;align-items:center}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{font:inherit;font-size:13px;border:0;background:var(--panel);color:var(--fg);padding:6px 12px;cursor:pointer}
.seg button+button{border-left:1px solid var(--line)}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--bg);font-weight:600}
.seglabel{font-size:12px;color:var(--muted);margin-right:6px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:170px}
thead th{font-size:12px;font-weight:600;color:var(--muted);vertical-align:bottom}
thead tr.grp th{font-family:var(--cond);font-size:13px;color:var(--fg);text-align:center;border-bottom:2px solid var(--line)}
td.num{font-family:var(--mono);font-size:13px}
td.best{background:var(--best);color:var(--bestfg);font-weight:500}
tr.pooled td{font-weight:600;border-bottom:2px solid var(--line)}
.colA{box-shadow:inset 0 3px 0 var(--m1)} .colB{box-shadow:inset 0 3px 0 var(--m2)} .colC{box-shadow:inset 0 3px 0 var(--m3)}
th.gstart,td.gstart{border-left:1px solid var(--line)}
.na{color:var(--muted)}
.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px 24px;padding:14px 16px;font-size:13px}
.legend b{font-family:var(--mono);font-weight:500}
.rank td:nth-child(2){text-align:left}
.sel{color:var(--accent);font-weight:600}
.bar{display:inline-block;height:6px;border-radius:3px;background:var(--accent);opacity:.55;vertical-align:middle;margin-left:8px}
.note{color:var(--muted);font-size:13px;max-width:80ch}
@media (max-width:640px){.controls{grid-template-columns:1fr}h1{font-size:23px}}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Forecast Factory · model evaluation · cut-off 2025</div>
    <h1>Forecast Factory Leaderboard</h1>
    <p class="lede" id="lede"></p>
  </header>

  <section class="controls" aria-label="Choose three models">
    <div class="pick"><label for="mA"><span class="dot" style="background:var(--m1)"></span>Model A</label><select id="mA"></select><div class="meta" id="iA"></div></div>
    <div class="pick"><label for="mB"><span class="dot" style="background:var(--m2)"></span>Model B</label><select id="mB"></select><div class="meta" id="iB"></div></div>
    <div class="pick"><label for="mC"><span class="dot" style="background:var(--m3)"></span>Model C</label><select id="mC"></select><div class="meta" id="iC"></div></div>
  </section>

  <div class="toggles">
    <div><span class="seglabel">Sample</span><span class="seg" id="sample"></span></div>
    <div><span class="seglabel">Horizon</span><span class="seg" id="horizon"></span></div>
  </div>

  <section>
    <h2>Error matrix</h2>
    <p class="note" id="mnote"></p>
    <div class="card scroll"><table id="matrix"></table></div>
  </section>

  <section class="card legend" aria-label="How to read the measures">
    <div><b>RMSE</b> root mean squared error. Penalises large misses. Level variables in % (100·log ratio), rates in percentage points.</div>
    <div><b>MAE</b> mean absolute error, same units. Less sensitive to single outliers.</div>
    <div><b>MAPE</b> median absolute percentage error, |forecast − actual| / |actual|. Rates only where |actual| ≥ 0.5.</div>
    <div><b>Theil U2</b> RMSE relative to the no-change forecast on the same cases. Below 1 beats the random walk.</div>
    <div><b>DA</b> directional accuracy: share of forecasts that get the sign of the change right. 0.5 is a coin flip.</div>
    <div>Shaded cell = best of the three models on that row. Lower is better for all measures except DA.</div>
  </section>

  <section>
    <h2>All 30 models</h2>
    <p class="note">Ranked by Theil U2 on all variables for the sample and horizon chosen above. Your three picks are highlighted.</p>
    <div class="card scroll"><table id="rank" class="rank"></table></div>
  </section>

  <section>
    <h2>Data</h2>
    <p class="note" id="dnote"></p>
    <div class="card scroll"><table id="data"></table></div>
  </section>
</div>

<script>
const D = /*__DATA__*/null;
const MEAS = [["RMSE","RMSE",-1],["MAE","MAE",-1],["MAPE","MAPE %",-1],["TheilU","Theil U2",-1],["DA","DA",1]];
const SAMPLES = ["In-sample","Out-of-sample"];
let state = {picks:[...D.default], sample:"Out-of-sample", h:1};
try{const s=JSON.parse(localStorage.getItem("ffl")||"null"); if(s&&s.picks&&s.picks.every(m=>D.models.includes(m))) state=s;}catch(e){}
const idx = new Map(D.scores.map(r=>[r.sample+"|"+r.model+"|"+r.variable+"|"+r.horizon, r]));
const get=(m,v)=>idx.get(state.sample+"|"+m+"|"+v+"|"+state.h);
const fmt=(x,k)=>x==null?'<span class="na">–</span>':(k==="DA"||k==="TheilU"?x.toFixed(2):x>=100?x.toFixed(0):x.toFixed(2));
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

document.getElementById("lede").textContent =
 `${D.models.length} model specifications from the Leaderboard repository, estimated on the same panel of ${D.n_countries} countries (World Bank, IMF, FRED; from 1960 where available). Each model is re-estimated at every forecast origin using only data up to that year. In-sample: origins ${D.ins[0]}–${D.ins[1]}. Out-of-sample: origins ${D.oos[0]}–${D.oos[1]}, scored against outturns up to the 2025 cut-off.`;

["A","B","C"].forEach((k,i)=>{
  const s=document.getElementById("m"+k);
  s.innerHTML=D.models.map(m=>`<option>${esc(m)}</option>`).join("");
  s.value=state.picks[i];
  s.addEventListener("change",()=>{state.picks[i]=s.value;render();});
});
function seg(id,opts,key){
  const el=document.getElementById(id);
  el.innerHTML=opts.map(([v,l])=>`<button type="button" data-v="${v}">${l}</button>`).join("");
  el.addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;
    state[key]=key==="h"?+b.dataset.v:b.dataset.v;render();});
}
seg("sample",SAMPLES.map(s=>[s,s]),"sample");
seg("horizon",[[1,"1 year"],[2,"2 years"]],"h");

function render(){
  try{localStorage.setItem("ffl",JSON.stringify(state));}catch(e){}
  document.querySelectorAll("#sample button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===state.sample));
  document.querySelectorAll("#horizon button").forEach(b=>b.setAttribute("aria-pressed",+b.dataset.v===state.h));
  ["A","B","C"].forEach((k,i)=>{const inf=D.info[state.picks[i]]||{};
    document.getElementById("i"+k).textContent=`${inf.file||""} · ${inf.params||""}`;});
  const P=state.picks, cls=["colA","colB","colC"];
  let h=`<thead><tr class="grp"><th></th>${MEAS.map(m=>`<th colspan="3" class="gstart">${m[1]}</th>`).join("")}</tr>`;
  h+=`<tr><th>Variable</th>${MEAS.map(()=>P.map((p,i)=>`<th class="${cls[i]}${i===0?" gstart":""}" title="${esc(p)}">${"ABC"[i]}</th>`).join("")).join("")}</tr></thead><tbody>`;
  for(const v of D.variables){
    const rows=P.map(m=>get(m,v));
    if(rows.every(r=>!r)) continue;
    h+=`<tr class="${v.startsWith("All")?"pooled":""}"><td>${esc(D.labels[v]||v)}</td>`;
    for(const [k,,dir] of MEAS){
      const vals=rows.map(r=>r?r[k]:null), ok=vals.filter(x=>x!=null);
      const best=ok.length>1?(dir<0?Math.min(...ok):Math.max(...ok)):null;
      vals.forEach((x,i)=>{h+=`<td class="num${i===0?" gstart":""}${x!=null&&x===best?" best":""}">${fmt(x,k)}</td>`;});
    }
    h+="</tr>";
  }
  document.getElementById("matrix").innerHTML=h+"</tbody>";
  const nn=P.map(m=>(get(m,"All variables (median)")||{}).n||0);
  document.getElementById("mnote").textContent=
   `${state.sample}, ${state.h}-year-ahead forecasts. Forecasts scored: A ${nn[0]}, B ${nn[1]}, C ${nn[2]} (only cases the random walk also covers). A dash means the model produced no forecasts for that variable.`;

  const rk=D.models.map(m=>({m,r:get(m,"All variables (median)")})).filter(x=>x.r)
     .sort((a,b)=>(a.r.TheilU??9)-(b.r.TheilU??9));
  const mx=Math.max(...rk.map(x=>x.r.TheilU||0),1);
  let t=`<thead><tr><th>#</th><th>Model</th><th>File</th><th>Theil U2</th><th>RMSE</th><th>MAE</th><th>MAPE %</th><th>DA</th><th>n</th></tr></thead><tbody>`;
  rk.forEach((x,i)=>{const s=P.includes(x.m), inf=D.info[x.m]||{};
    t+=`<tr><td class="num">${i+1}</td><td class="${s?"sel":""}" style="text-align:left">${esc(x.m)}</td><td class="meta" style="text-align:left">${esc(inf.file||"")}</td>
    <td class="num">${fmt(x.r.TheilU,"TheilU")}<span class="bar" style="width:${Math.round(60*(x.r.TheilU||0)/mx)}px"></span></td>
    <td class="num">${fmt(x.r.RMSE)}</td><td class="num">${fmt(x.r.MAE)}</td><td class="num">${fmt(x.r.MAPE)}</td><td class="num">${fmt(x.r.DA,"DA")}</td><td class="num">${x.r.n}</td></tr>`;});
  document.getElementById("rank").innerHTML=t+"</tbody>";
}
document.getElementById("dnote").textContent="Transformation codes follow FRED-MD (McCracken & Ng 2016): 1 = level, 2 = first difference, 5 = log difference. Models forecast the transformed series and map forecasts back to original units before scoring.";
document.getElementById("data").innerHTML=`<thead><tr><th>Variable</th><th style="text-align:left">Source</th><th>FRED-MD tcode</th><th>From</th><th>To</th><th>Countries</th></tr></thead><tbody>`+
  D.data.map(r=>`<tr><td>${esc(r.label)}</td><td style="text-align:left">${esc(r.source)}</td><td class="num">${r.tcode}</td><td class="num">${r.first}</td><td class="num">${r.last}</td><td class="num">${r.countries}</td></tr>`).join("")+"</tbody>";
render();
</script>
"""
