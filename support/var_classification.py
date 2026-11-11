"""Variable classification: transformation code (FRED-MD style) and family per variable.

FRED-MD tcodes (McCracken & Ng 2016):  1 = as_is, 2 = diff, 5 = log_diff.
Annual panel choices follow the FRED-MD codes of the matching US series:
  price index, real GDP, money, exchange rates  -> 5 (log_diff, i.e. growth)
  interest rates, unemployment, debt ratio      -> 2 (diff)
  inflation rate, current account ratio         -> 1 (as_is, already stationary)
"""
TRANSFORM = {
    "CPI": "log_diff", "rGDP": "log_diff", "M2": "log_diff",
    "USDfx": "log_diff", "REER": "log_diff", "nGDP_USD": "log_diff",
    "strate": "diff", "ltrate": "diff", "unemp": "diff", "gen_govdebt_GDP": "diff",
    "infl": "as_is", "CA_GDP": "as_is",
}
TCODE = {"as_is": 1, "diff": 2, "log_diff": 5}
LEVEL_VARS = {"CPI", "rGDP", "M2", "USDfx", "REER", "nGDP_USD"}   # strictly positive levels


def transformation(v: str) -> str:
    return TRANSFORM.get(v, "as_is")


def family(v: str) -> str:
    """'skip' = not a forecast target (used only as a weight)."""
    if v == "nGDP_USD":
        return "skip"
    return "level" if v in LEVEL_VARS else "rate"
