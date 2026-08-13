"""Cadence and background generation, band-general.

The parent's `e5_lib.build_cadence` and `background.simulate_path` hard-code a
two-band survey.  The functions here are their band-general form and are
asserted in stage 0 to reproduce them bitwise at the fiducial configuration,
and, through that, to reproduce the parent's frozen cadence.csv exactly.

Nothing here scores anything.  Scoring is `analysis.analyse` only.
"""
import os

import numpy as np
import celerite2
from celerite2 import terms

# Instrumentation: any deterministic signal must go through Generator.chirp,
# which bumps this counter.  A null batch asserts it is still 0.
INJECTION_CALLS = 0


def injection_calls():
    return INJECTION_CALLS


def reset_injection_counter():
    global INJECTION_CALLS
    INJECTION_CALLS = 0


def build_cadence(cad, seed):
    """One cadence realisation.  Band-general form of e5_lib.build_cadence.

    For bands = ['g', 'r'] with r_offset_d and equal steps this reproduces
    e5_lib.build_cadence bitwise, including the RNG draw order
    (jitter -> dropout -> per-epoch errors).
    """
    rng = np.random.default_rng(seed)
    period = cad["season_obs_d"] + cad["season_gap_d"]
    t_end = cad["t_start"] + cad["baseline_d"]

    steps = {"g": cad["cadence_g_d"], "r": cad.get("cadence_r_d", cad["cadence_g_d"])}
    offs = {"g": 0.0, "r": cad.get("r_offset_d", 0.0)}
    plan = [(b, steps[b], offs[b]) for b in cad["bands"]]

    times, bands = [], []
    k = 0
    while cad["t_start"] + k * period < t_end:
        s0 = cad["t_start"] + k * period
        for band, step, off in plan:
            tt = np.arange(s0 + off, s0 + cad["season_obs_d"], step)
            tt = tt[tt < t_end]
            times.append(tt)
            bands.append(np.full(tt.size, band))
        k += 1
    t = np.concatenate(times)
    b = np.concatenate(bands)

    t = t + rng.uniform(-cad["jitter_halfwidth_d"], cad["jitter_halfwidth_d"], t.size)
    keep = rng.random(t.size) >= cad["drop_fraction"]
    t, b = t[keep], b[keep]

    order = np.argsort(t, kind="stable")
    t, b = t[order], b[order]

    n_nudged = 0
    if np.any(np.diff(t) <= 0):
        n_nudged = int(np.sum(np.diff(t) <= 0))
        for i in range(1, t.size):
            if t[i] <= t[i - 1]:
                t[i] = t[i - 1] + 1e-6

    sig = rng.uniform(cad["sigma_err_min_mag"], cad["sigma_err_max_mag"], t.size)
    return t, b, sig, n_nudged


def in_window_factor_drw(t, tau):
    """g such that E[sample variance of the mean-subtracted path] = sigma^2 * g.

    Identical formula and code path to background.in_window_factor for DRW.
    """
    dtm = np.abs(t[:, None] - t[None, :])
    R = np.exp(-dtm / tau)
    n = t.size
    return float((np.trace(R) - R.sum() / n) / (n - 1))


def kernel_sigma2_drw(t, tau, target_variance):
    return target_variance / in_window_factor_drw(t, tau)


def drw_path(t, s2, tau, seed):
    """Latent achromatic DRW path at the cadence times, no measurement noise.

    Mirrors background.simulate_path('drw', ...) exactly: same kernel
    parameterisation, same 1e-12 diagonal, same Cholesky-product sampling.
    """
    gp = celerite2.GaussianProcess(terms.RealTerm(a=s2, c=1.0 / tau), mean=0.0)
    gp.compute(t, diag=np.full(t.size, 1e-12))
    rng = np.random.default_rng(seed)
    return gp.dot_tril(rng.normal(size=t.size))


def chirp_signal(t, P_ref, eta, amp1, amp2, ph1, ph2, t_ref, T_span):
    x = t - t_ref
    f0 = 1.0 / P_ref
    phi = 2.0 * np.pi * f0 * (x + 0.5 * eta * x ** 2 / T_span)
    return amp1 * np.sin(phi + ph1) + amp2 * np.sin(2.0 * phi + ph2)


_REAL_CADENCE_CACHE = {}


def _load_real_cadence(path, obj):
    import json
    key = os.path.abspath(path)
    d = _REAL_CADENCE_CACHE.get(key)
    if d is None:
        with open(key, encoding="utf-8") as f:
            d = json.load(f)
        _REAL_CADENCE_CACHE[key] = d
    e = d[obj]
    t = np.asarray(e["t"], dtype=float)
    b = np.asarray(e["band"], dtype="<U2")
    s = np.asarray(e["sig"], dtype=float)
    order = np.argsort(t, kind="stable")
    return t[order], b[order], s[order]


class Generator:
    """Holds one configuration's frozen cadence and DRW normalisation.

    Two cadence sources.  Synthetic configurations build their cadence from the
    season/dropout model with `cadence_seed`.  Real-cadence configurations
    (SPEC 6.4, config_index 17-22) load the *post-quality-cut* epochs, band
    labels and per-epoch magnitude errors of one ZTF object.  No real
    photometry enters either path, only the sampling and the error structure.
    """

    def __init__(self, entry, cadence_seed):
        self.entry = entry
        self.cadence_seed = int(cadence_seed)
        rc = entry.get("real_cadence")
        self.is_real_cadence = rc is not None
        if self.is_real_cadence:
            here = os.path.dirname(os.path.abspath(__file__))
            base = os.path.dirname(here)
            self.t, self.band, self.sig = _load_real_cadence(
                os.path.join(base, rc["file"]), rc["object"])
            self.n_nudged = 0
        else:
            cad = entry["cadence"]
            self.t, self.band, self.sig, self.n_nudged = build_cadence(cad, cadence_seed)
        self.tau = float(entry["tau_d"])
        self.target_var = float(entry["target_in_window_variance_mag2"])
        self.s2 = kernel_sigma2_drw(self.t, self.tau, self.target_var)
        self.t_ref = 0.5 * (self.t.min() + self.t.max())
        self.T_span = float(self.t.max() - self.t.min())
        self.n_epochs = int(self.t.size)
        self.n_bands = int(len(set(self.band.tolist())))
        self.mean_sigma_phot = float(self.sig.mean())

    def latent_path(self, seed):
        return drw_path(self.t, self.s2, self.tau, seed)

    def null_lightcurve(self, seed):
        """Pure background: one shared path + independent per-epoch noise.

        The measurement-noise stream is seed + 500000, matching the parent.
        """
        path = self.latent_path(seed)
        rng = np.random.default_rng(seed + 500_000)
        y = path + rng.normal(0.0, self.sig)
        return self.t, y, self.sig, self.band

    def chirp(self, times, inj):
        global INJECTION_CALLS
        INJECTION_CALLS += 1
        return chirp_signal(times, inj["P_ref_d"], inj["eta"], inj["amp_h1_mag"],
                            inj["amp_h2_mag"], inj["phase_h1_rad"],
                            inj["phase_h2_rad"], self.t_ref, self.T_span)

    def summary(self):
        return dict(config_index=self.entry["config_index"], label=self.entry["label"],
                    real_cadence=int(getattr(self, "is_real_cadence", False)),
                    n_epochs=self.n_epochs, n_bands=self.n_bands,
                    tau_d=self.tau, s2=self.s2,
                    sigma_kernel_mag=float(np.sqrt(self.s2)),
                    target_in_window_variance_mag2=self.target_var,
                    mean_sigma_phot_mag=self.mean_sigma_phot,
                    T_span_d=self.T_span, t_ref_d=float(self.t_ref),
                    n_nudged_epochs=self.n_nudged)
