"""Forward / inverse transformations with an anchor context for exact inversion.

forward(values, mode) -> (transformed array, ctx); log_diff and diff drop the first element.
inverse(steps, ctx, mode) -> levels for the periods after the last observed value.
"""
import numpy as np


def forward(values, mode):
    x = np.asarray(values, dtype=float)
    if mode == "log_diff":
        with np.errstate(divide="ignore", invalid="ignore"):
            lx = np.where(x > 0, np.log(x), np.nan)
        return np.diff(lx), {"last": float(lx[-1]) if len(lx) else np.nan}
    if mode == "diff":
        return np.diff(x), {"last": float(x[-1]) if len(x) else np.nan}
    return x.copy(), {"last": float(x[-1]) if len(x) else np.nan}


def inverse(steps, ctx, mode):
    s = np.asarray(steps, dtype=float)
    if mode == "log_diff":
        return list(np.exp(ctx["last"] + np.cumsum(s)))
    if mode == "diff":
        return list(ctx["last"] + np.cumsum(s))
    return list(s)
