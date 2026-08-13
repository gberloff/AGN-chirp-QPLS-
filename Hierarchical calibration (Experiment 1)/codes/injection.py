"""Injection construction, shared by Parts F and G.

Everything declared in dimensionless form is turned into physical parameters
physical parameters here, in one place, so Parts F and G cannot drift apart.
"""
import numpy as np

import lib
import gen

T_BASELINE = 2000.0
DUTY_FIDUCIAL = lib.FIDUCIAL_DUTY
MIN_EPOCHS = 100

AXES = ["n_cyc", "eta_x", "snr", "tau_over_p", "a2_over_a1", "samples_per_cycle"]
BOUNDS = dict(n_cyc=(3.0, 40.0), eta_x=(0.0, 8.0), snr=(3.0, 50.0),
              tau_over_p=(0.1, 10.0), a2_over_a1=(0.0, 1.0),
              samples_per_cycle=(2.0, 60.0))
LOG_AXES = {"n_cyc", "snr", "tau_over_p", "samples_per_cycle"}

FIDUCIAL_POINT = dict(n_cyc=8.0, eta_x=2.0, snr=15.0, tau_over_p=1.0,
                      a2_over_a1=0.3, samples_per_cycle=20.0)


def unit_to_axes(u):
    """Map a point in the unit cube to the six axes, log where declared."""
    out = {}
    for k, v in zip(AXES, u):
        lo, hi = BOUNDS[k]
        if k in LOG_AXES:
            out[k] = float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo))))
        else:
            out[k] = float(lo + v * (hi - lo))
    return out


def derive(ax, band_structure, baseline_d=T_BASELINE, duty_cycle=DUTY_FIDUCIAL):
    """Physical parameters from the dimensionless axes."""
    P = baseline_d / ax["n_cyc"]
    eta = ax["eta_x"] / ax["n_cyc"]
    tau = ax["tau_over_p"] * P
    dt = P / ax["samples_per_cycle"]
    n_bands = 1 if band_structure == 0 else 2
    return dict(P_true_d=float(P), eta_true=float(eta), tau_d=float(tau),
                dt_d=float(dt), n_bands=int(n_bands),
                baseline_d=float(baseline_d), duty_cycle=float(duty_cycle))


def positive_frequency_ok(eta):
    """f(t) > 0 across the window.  With x in [-T/2, +T/2] and the phase model
    phi = 2 pi f0 (x + eta x^2 / (2 T)), the instantaneous frequency is
    f0 (1 + eta x / T), whose minimum over the window is f0 (1 - |eta| / 2).
    So the condition is exactly |eta| < 2."""
    return abs(eta) < 2.0


def n_epochs_for(d, cadence_seed=20260807):
    """Epoch count the cadence would present to analyse(), before any cut.
    Cheap: builds the cadence only, never the O(n^2) background normalisation."""
    cad = lib.make_cadence(d["baseline_d"], d["dt_d"], d["n_bands"],
                           d["duty_cycle"])
    try:
        t, b, s, _ = gen.build_cadence(cad, cadence_seed)
    except Exception:
        return 0
    return int(t.size)


def feasible(ax, band_structure, baseline_d=T_BASELINE,
             duty_cycle=DUTY_FIDUCIAL):
    """The three rejection rules.  Returns (ok, reason)."""
    d = derive(ax, band_structure, baseline_d, duty_cycle)
    if abs(d["eta_true"]) > 0.55:
        return False, "eta_gt_0.55"
    if not positive_frequency_ok(d["eta_true"]):
        return False, "negative_frequency"
    if n_epochs_for(d) < MIN_EPOCHS:
        return False, "below_min_epochs"
    return True, ""


def build_entry(d, config_index=0, label="inj", sigma_ratio=3.0):
    return lib.make_entry(config_index, label, baseline_d=d["baseline_d"],
                          dt_d=d["dt_d"], n_bands=d["n_bands"],
                          duty_cycle=d["duty_cycle"], tau_d=d["tau_d"],
                          sigma_ratio=sigma_ratio)


def make_curve(g, seed, ax, d, band_structure):
    """Background + chirp, under the declared SNR convention.

    Reproduces Generator.null_lightcurve exactly (latent path from `seed`,
    measurement noise from `seed + 500000`) and then adds the signal, so an
    injection and its own null differ by the signal alone.
    """
    path = g.latent_path(seed)
    rng = np.random.default_rng(seed + 500_000)
    noise = rng.normal(0.0, g.sig)
    s_eff = lib.sigma_eff(g, path)
    n = g.t.size
    amp1 = ax["snr"] * s_eff / np.sqrt(n)
    amp2 = ax["a2_over_a1"] * amp1

    # phases: independent per band when the band structure says free phase
    prng = np.random.default_rng(seed + 900_000)
    labs = sorted(set(g.band.tolist()))
    if band_structure == 1 and len(labs) == 2:
        phases = {lab: (float(prng.uniform(0, 2 * np.pi)),
                        float(prng.uniform(0, 2 * np.pi))) for lab in labs}
    else:
        ph1, ph2 = float(prng.uniform(0, 2 * np.pi)), float(prng.uniform(0, 2 * np.pi))
        phases = {lab: (ph1, ph2) for lab in labs}

    sig_curve = lib.inject_chirp(g, g.t, g.band, d["P_true_d"], d["eta_true"],
                                 amp1, amp2, phases)
    y = path + noise + sig_curve
    return (g.t, y, g.sig, g.band,
            dict(amp1=float(amp1), amp2=float(amp2), sigma_eff=float(s_eff),
                 path_rms=float(np.std(path, ddof=1)),
                 median_magerr=float(np.median(g.sig)),
                 n_epochs_raw=int(n),
                 phases={k: list(v) for k, v in phases.items()}))


def profile_intervals_nan():
    """The same keys, all missing.  Used when analyse returned no surface
    because the minimum-epoch cut fired."""
    out = {}
    for name in ("P", "eta"):
        for lvl in (68, 95):
            out[f"{name}_lo{lvl}"] = np.nan
            out[f"{name}_hi{lvl}"] = np.nan
            out[f"{name}_edge{lvl}"] = 1
    return out


def profile_intervals(surface, Pgrid, etagrid):
    """68% and 95% profile-likelihood intervals on P and eta from the scan.

    The scan surface is a log-likelihood-ratio improvement, so for one
    parameter the interval is the connected region around the maximum with
    Lambda >= Lambda_max - Delta, with Delta = 0.5 for 68.3% and 1.92 for 95%.

    Coverage and posterior concentration are different things, an interval
    touching a grid edge is flagged rather than silently truncated, so the
    two are not conflated.
    """
    L = np.asarray(surface, dtype=float)
    out = {}
    for name, axis, grid in (("P", 1, Pgrid), ("eta", 0, etagrid)):
        prof = np.nanmax(L, axis=axis)
        if not np.any(np.isfinite(prof)):
            for lvl in (68, 95):
                out[f"{name}_lo{lvl}"] = np.nan
                out[f"{name}_hi{lvl}"] = np.nan
                out[f"{name}_edge{lvl}"] = 1
            continue
        k = int(np.nanargmax(prof))
        for lvl, delta in ((68, 0.5), (95, 1.92)):
            ok = prof >= prof[k] - delta
            lo = k
            while lo - 1 >= 0 and ok[lo - 1]:
                lo -= 1
            hi = k
            while hi + 1 < ok.size and ok[hi + 1]:
                hi += 1
            out[f"{name}_lo{lvl}"] = float(grid[lo])
            out[f"{name}_hi{lvl}"] = float(grid[hi])
            out[f"{name}_edge{lvl}"] = int(lo == 0 or hi == ok.size - 1)
    return out


def outcomes(score, threshold, P_hat, eta_hat, P_true, eta_true, prefix=""):
    trig = bool(np.isfinite(score) and score > threshold)
    if not np.isfinite(P_hat) or P_true <= 0:
        correct = alias = chirp = False
    else:
        correct = bool(trig and abs(P_hat - P_true) / P_true < 0.02)
        alias = bool(trig and (abs(P_hat - 0.5 * P_true) / (0.5 * P_true) < 0.02
                               or abs(P_hat - 2.0 * P_true) / (2.0 * P_true) < 0.02))
        chirp = bool(correct and np.isfinite(eta_hat)
                     and np.sign(eta_hat) == np.sign(eta_true)
                     and abs(eta_hat) < 0.58)
    return {prefix + "trigger": int(trig), prefix + "correct": int(correct),
            prefix + "alias": int(alias), prefix + "chirp": int(chirp)}
