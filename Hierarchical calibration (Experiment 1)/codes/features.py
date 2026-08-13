"""PART E: cheap triage features.

The qualifying rule, enforced by construction: a feature may use only the light
curve.  Nothing here touches `analyse`'s likelihood output, because a feature
derived from the GP scan would mean the expensive step had already been paid,
which defeats the purpose of a triage layer.

The one exception is `frac_epochs_cut`, the fraction of epochs the quality cuts
remove.  The cuts are pure data-quality arithmetic with no GP and no scan, so it qualifies.  It is computed here
from the same cut code `analyse` uses, not read back from `analyse`'s output.

Cost is measured separately from the fit cost- the ratio is what decides
whether a triage layer is worth building at all.
"""
import time

import numpy as np
from astropy.timeseries import LombScargle

import lib
import analysis

SF_LAGS = (10.0, 50.0, 200.0, 500.0)
ACF_LAGS = (30.0, 100.0, 300.0)


def _mad_scaled(x):
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def _windows(t, lag, tol):
    """Half-open index window [i0, i1) of points at separation in
    [(1-tol)*lag, (1+tol)*lag] ahead of each epoch.  t must be sorted."""
    i0 = np.searchsorted(t, t + (1.0 - tol) * lag, side="left")
    i1 = np.searchsorted(t, t + (1.0 + tol) * lag, side="right")
    return i0, np.maximum(i1, i0)


def _structure_function(t, y, lag, tol=0.5):
    """SF(lag) = RMS of magnitude differences at separations within +/-tol*lag.

    Vectorised by prefix sums:  sum_j (y_j - y_k)^2 over a contiguous window
    expands to  sum y_j^2 - 2 y_k sum y_j + cnt y_k^2, and each of those is a
    difference of cumulative sums.  Exactly equal to the pairwise form, and it
    matters: this runs once per curve for four lags on 10,000 curves.
    """
    i0, i1 = _windows(t, lag, tol)
    c1 = np.concatenate(([0.0], np.cumsum(y)))
    c2 = np.concatenate(([0.0], np.cumsum(y * y)))
    cnt = i1 - i0
    s1 = c1[i1] - c1[i0]
    s2 = c2[i1] - c2[i0]
    tot = float(np.sum(s2 - 2.0 * y * s1 + cnt * y * y))
    n_pairs = int(cnt.sum())
    if n_pairs < 5:
        return np.nan, n_pairs
    return float(np.sqrt(max(tot, 0.0) / n_pairs)), n_pairs


def _acf_at_lag(t, y, lag, tol=0.25):
    yc = y - y.mean()
    v = float(np.dot(yc, yc) / y.size)
    if v <= 0:
        return np.nan
    i0, i1 = _windows(t, lag, tol)
    c1 = np.concatenate(([0.0], np.cumsum(yc)))
    cnt = i1 - i0
    tot = float(np.sum(yc * (c1[i1] - c1[i0])))
    n_pairs = int(cnt.sum())
    if n_pairs < 5:
        return np.nan
    return float(tot / n_pairs / v)


def compute(t, y, sig, bands, spec, catflags=None):
    """Return (features dict, wall-clock seconds).  Input is the RAW curve, as
    presented to analyse(), before any cut."""
    t0 = time.perf_counter()
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    sig = np.asarray(sig, float)
    b = np.asarray(bands)
    order = np.argsort(t, kind="stable")
    t, y, sig, b = t[order], y[order], sig[order], b[order]

    f = {}
    n = t.size
    f["n_epochs_raw"] = int(n)
    f["baseline_d"] = float(t[-1] - t[0]) if n > 1 else 0.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    f["dt_median_d"] = float(np.median(dt)) if dt.size else np.nan
    f["dt_iqr_d"] = float(np.subtract(*np.percentile(dt, [75, 25]))) \
        if dt.size else np.nan
    f["largest_gap_d"] = float(dt.max()) if dt.size else np.nan
    # duty cycle: fraction of the baseline inside an observing season, taken as
    # the fraction of the span not inside a gap longer than 5x the median dt
    if dt.size:
        big = dt[dt > 5 * f["dt_median_d"]]
        f["duty_cycle_est"] = float(1.0 - big.sum() / f["baseline_d"]) \
            if f["baseline_d"] > 0 else np.nan
    else:
        f["duty_cycle_est"] = np.nan

    w = 1.0 / np.maximum(sig, 1e-6) ** 2
    f["weighted_mean_mag"] = float(np.sum(w * y) / np.sum(w))
    f["robust_scatter_mag"] = _mad_scaled(y)
    f["median_magerr"] = float(np.median(sig))

    var = float(np.var(y, ddof=1)) if n > 1 else np.nan
    mse = float(np.mean(sig ** 2))
    f["excess_var"] = var - mse
    # The textbook normalised excess variance divides by the squared mean
    # magnitude.  These are synthetic zero-mean curves with no absolute
    # magnitude scale, so that form is degenerate (it divides by ~0 and returns
    # a number of order 50 that means nothing).  Normalising by the measurement
    # variance instead keeps it dimensionless and well defined on both
    # synthetic and real curves.
    f["norm_excess_var"] = (var - mse) / mse if mse > 0 else np.nan
    f["excess_var_frac"] = (var - mse) / var if var > 0 else np.nan

    sfv = []
    for lag in SF_LAGS:
        v, _ = _structure_function(t, y, lag)
        f[f"sf_{int(lag)}d"] = v
        sfv.append(v)
    sfv = np.asarray(sfv, dtype=float)
    ok = np.isfinite(sfv) & (sfv > 0)
    if ok.sum() >= 2:
        f["sf_slope"] = float(np.polyfit(np.log10(np.asarray(SF_LAGS)[ok]),
                                         np.log10(sfv[ok]), 1)[0])
    else:
        f["sf_slope"] = np.nan

    try:
        fmin, fmax = 1.0 / spec.P_max_d, 1.0 / spec.P_min_d
        freq = np.linspace(fmin, fmax, 2000)
        ls = LombScargle(t, y, sig)
        pw = ls.power(freq)
        k = int(np.nanargmax(pw))
        f["ls_peak_power"] = float(pw[k])
        f["ls_peak_period_d"] = float(1.0 / freq[k])
        med = float(np.nanmedian(pw))
        f["ls_peak_over_median"] = f["ls_peak_power"] / med if med > 0 else np.nan
        m = np.abs(freq - freq[k]) > 0.05 * freq[k]
        f["ls_peak_over_second"] = (f["ls_peak_power"] / float(np.nanmax(pw[m]))
                                    if m.any() and np.nanmax(pw[m]) > 0 else np.nan)
        f2 = 2.0 * freq[k]
        f["ls_power_at_2f"] = (float(np.interp(f2, freq, pw))
                               if fmin <= f2 <= fmax else np.nan)
        # The two-harmonic periodogram is by far the most expensive feature
        # 370 ms of a 400 ms budget on the 2000-node grid, against 10 ms for the
        # one-term version.  The frequency resolution element is about 1/T, so
        # this range holds roughly 31 independent frequencies and 2000 nodes is
        # 65x oversampled.  500 nodes is still 16x oversampled and is ample for
        # a triage feature that only reports the peak.
        freq2 = np.linspace(fmin, fmax, 500)
        pw2 = LombScargle(t, y, sig, nterms=2).power(freq2)
        f["ls2_peak_power"] = float(np.nanmax(pw2))
        f["ls2_peak_period_d"] = float(1.0 / freq2[int(np.nanargmax(pw2))])
    except Exception:
        for k2 in ("ls_peak_power", "ls_peak_period_d", "ls_peak_over_median",
                   "ls_peak_over_second", "ls_power_at_2f", "ls2_peak_power",
                   "ls2_peak_period_d"):
            f[k2] = np.nan

    for lag in ACF_LAGS:
        f[f"acf_{int(lag)}d"] = _acf_at_lag(t, y, lag)

    cf = np.zeros(n, dtype=int) if catflags is None else np.asarray(catflags)[order]
    keep, counts = analysis.apply_quality_cuts(t, y, sig, b, cf, spec)
    f["frac_epochs_cut"] = float(counts["n_removed_total"] / n) if n else np.nan
    f["n_epochs_after_cuts"] = int(np.count_nonzero(keep))

    labs = sorted(set(b.tolist()))
    if len(labs) == 2:
        m0, m1 = b == labs[0], b == labs[1]
        t0b, y0b = t[m0], y[m0]
        t1b, y1b = t[m1], y[m1]
        if t0b.size and t1b.size:
            j = np.searchsorted(t1b, t0b)
            j = np.clip(j, 1, t1b.size - 1)
            pick = np.where(np.abs(t1b[j] - t0b) < np.abs(t1b[j - 1] - t0b),
                            j, j - 1)
            good = np.abs(t1b[pick] - t0b) < 3.0
            col = y0b[good] - y1b[pick][good]
            f["colour_scatter_mag"] = _mad_scaled(col) if col.size > 5 else np.nan
            f["colour_mean_mag"] = float(np.mean(col)) if col.size else np.nan
            f["n_colour_pairs"] = int(col.size)
        else:
            f["colour_scatter_mag"] = np.nan
            f["colour_mean_mag"] = np.nan
            f["n_colour_pairs"] = 0
    else:
        f["colour_scatter_mag"] = np.nan
        f["colour_mean_mag"] = np.nan
        f["n_colour_pairs"] = 0

    return f, time.perf_counter() - t0


FEATURE_NAMES = [
    "n_epochs_raw", "baseline_d", "duty_cycle_est", "dt_median_d", "dt_iqr_d",
    "largest_gap_d", "weighted_mean_mag", "robust_scatter_mag", "median_magerr",
    "excess_var", "norm_excess_var", "excess_var_frac",
    "sf_10d", "sf_50d", "sf_200d", "sf_500d", "sf_slope",
    "ls_peak_power", "ls_peak_period_d", "ls_peak_over_median",
    "ls_peak_over_second", "ls_power_at_2f", "ls2_peak_power",
    "ls2_peak_period_d", "acf_30d", "acf_100d", "acf_300d",
    "frac_epochs_cut", "n_epochs_after_cuts",
    "colour_scatter_mag", "colour_mean_mag", "n_colour_pairs",
]
