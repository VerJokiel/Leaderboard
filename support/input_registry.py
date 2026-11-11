"""Which auxiliary variables feed each target ('curated' = economically motivated)."""
ALL = ("CPI", "infl", "rGDP", "unemp", "strate", "ltrate", "USDfx", "REER",
       "M2", "CA_GDP", "gen_govdebt_GDP")

CURATED = {
    "infl":   ("rGDP", "unemp", "strate", "M2", "USDfx"),       # Phillips curve + money + FX pass-through
    "CPI":    ("rGDP", "unemp", "strate", "M2", "USDfx"),
    "rGDP":   ("infl", "strate", "ltrate", "unemp", "CA_GDP"),  # IS curve, term spread
    "unemp":  ("rGDP", "infl", "strate"),                       # Okun's law
    "strate": ("infl", "rGDP", "unemp"),                        # Taylor rule
    "ltrate": ("strate", "infl", "gen_govdebt_GDP"),
    "USDfx":  ("infl", "strate", "CA_GDP"),                     # UIP / PPP
    "REER":   ("infl", "CA_GDP", "rGDP"),
    "M2":     ("rGDP", "infl", "strate"),
    "CA_GDP": ("rGDP", "REER", "USDfx"),
    "gen_govdebt_GDP": ("rGDP", "strate", "infl"),
}


def all_macro_inputs():
    return list(ALL)


def resolve_inputs(target, mode="curated"):
    if mode == "target_only":
        return []
    if mode == "all":
        return [v for v in ALL if v != target]
    return list(CURATED.get(target, ()))
