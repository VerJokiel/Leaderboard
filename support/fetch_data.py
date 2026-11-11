"""Download an annual country panel from the World Bank, the IMF and FRED.

Output format (forecast-factory): ISO3, variable, year, value.  Cached in data_cache/.

  World Bank WDI (1960-) : CPI, infl, rGDP, M2, USDfx, CA_GDP, nGDP_USD, unemp (1991-, fallback)
  IMF WEO DataMapper     : unemp (1980-, preferred), gen_govdebt_GDP
  FRED (OECD MEI, BIS)   : strate (3-month), ltrate (10-year), REER (BIS broad, annual mean)
"""
from __future__ import annotations
import io
import json
import os
import time
import urllib.request

import pandas as pd

COUNTRIES = {  # ISO3: (ISO2, region)
    "USA": ("US", "AME"), "CAN": ("CA", "AME"), "MEX": ("MX", "AME"), "BRA": ("BR", "AME"),
    "DEU": ("DE", "EUR"), "FRA": ("FR", "EUR"), "ITA": ("IT", "EUR"), "ESP": ("ES", "EUR"),
    "NLD": ("NL", "EUR"), "GBR": ("GB", "EUR"), "SWE": ("SE", "EUR"), "NOR": ("NO", "EUR"),
    "CHE": ("CH", "EUR"), "DNK": ("DK", "EUR"),
    "JPN": ("JP", "APAC"), "KOR": ("KR", "APAC"), "AUS": ("AU", "APAC"), "NZL": ("NZ", "APAC"),
    "IND": ("IN", "APAC"), "ZAF": ("ZA", "APAC"),
}
REGIONS = {k: v[1] for k, v in COUNTRIES.items()}

WB = {"CPI": "FP.CPI.TOTL", "infl": "FP.CPI.TOTL.ZG", "rGDP": "NY.GDP.MKTP.KD",
      "M2": "FM.LBL.BMNY.CN", "USDfx": "PA.NUS.FCRF", "CA_GDP": "BN.CAB.XOKA.GD.ZS",
      "nGDP_USD": "NY.GDP.MKTP.CD", "unemp": "SL.UEM.TOTL.ZS"}
IMF = {"unemp": "LUR", "gen_govdebt_GDP": "GGXWDG_NGDP"}
SOURCE = {**{v: "World Bank WDI" for v in WB}, "unemp": "IMF WEO (World Bank before 1980 n/a)",
          "gen_govdebt_GDP": "IMF WEO", "strate": "FRED / OECD MEI", "ltrate": "FRED / OECD MEI",
          "REER": "FRED / BIS"}

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_cache")


def _get(url: str, tries: int = 3) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def world_bank(last_year: int) -> pd.DataFrame:
    isos = ";".join(COUNTRIES)
    rows = []
    for var, code in WB.items():
        url = (f"https://api.worldbank.org/v2/country/{isos}/indicator/{code}"
               f"?format=json&per_page=20000&date=1960:{last_year}")
        data = json.loads(_get(url))
        for d in data[1] or []:
            if d["value"] is not None:
                rows.append((d["countryiso3code"], var, int(d["date"]), float(d["value"])))
    return pd.DataFrame(rows, columns=["ISO3", "variable", "year", "value"])


def imf_weo(last_year: int) -> pd.DataFrame:
    rows = []
    for var, code in IMF.items():
        data = json.loads(_get(f"https://www.imf.org/external/datamapper/api/v1/{code}/"
                               + "/".join(COUNTRIES)))
        for iso, series in data.get("values", {}).get(code, {}).items():
            for y, v in series.items():
                if v is not None and int(y) <= last_year:      # WEO projections dropped
                    rows.append((iso, var, int(y), float(v)))
    return pd.DataFrame(rows, columns=["ISO3", "variable", "year", "value"])


def fred(last_year: int) -> pd.DataFrame:
    rows = []
    for iso, (c2, _) in COUNTRIES.items():
        for var, sid in (("strate", f"IR3TIB01{c2}A156N"), ("ltrate", f"IRLTLT01{c2}A156N"),
                         ("REER", f"RB{c2}BIS")):
            try:
                df = pd.read_csv(io.BytesIO(_get(
                    f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", tries=2)))
            except Exception:
                continue
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["year"] = pd.to_datetime(df["date"]).dt.year
            ann = df.dropna().groupby("year")["value"].mean()      # annual average
            if var == "REER":                                      # need full years only
                n = df.dropna().groupby("year").size()
                ann = ann[n >= 12]
            rows += [(iso, var, int(y), float(v)) for y, v in ann.items() if y <= last_year]
    return pd.DataFrame(rows, columns=["ISO3", "variable", "year", "value"])


def build_panel(last_year: int = 2026, refresh: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"panel_{last_year}.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    wb, imf, fr = world_bank(last_year), imf_weo(last_year), fred(last_year)
    # unemployment: IMF WEO (from 1980) preferred; World Bank only fills gaps
    wb_u = wb[wb.variable == "unemp"]
    imf_u = imf[imf.variable == "unemp"]
    have = set(zip(imf_u.ISO3, imf_u.year))
    wb_fill = wb_u[[(i, y) not in have for i, y in zip(wb_u.ISO3, wb_u.year)]]
    panel = pd.concat([wb[wb.variable != "unemp"], wb_fill, imf, fr], ignore_index=True)
    panel = panel[panel.ISO3.isin(COUNTRIES)]
    panel = (panel.drop_duplicates(["ISO3", "variable", "year"], keep="last")
                  .sort_values(["ISO3", "variable", "year"]).reset_index(drop=True))
    panel.to_csv(path, index=False)
    return panel


if __name__ == "__main__":
    p = build_panel(refresh=True)
    print(p.groupby("variable").agg(n=("value", "size"), first=("year", "min"),
                                    last=("year", "max"), countries=("ISO3", "nunique")))
