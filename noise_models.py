"""
Three stochastic noise generators for the fixed chirp + DRW detector.

The injected chirp, the cadence, the photometric errors and the white noise are
identical across generators; ONLY the correlated-noise component changes.  This
is achieved by reusing simulate.py's per-band RNG stream construction verbatim,
so the "cadence" and "white" streams are byte-identical between runs and only
the "drw" stream is consumed differently.

Generators (total stochastic variance matched to sigma^2 = 0.06^2):

  drw       exact AR(1) recursion at the observed times.  This is the matched
            case - the detector's own model.  For a Markov (OU) process the
            Cholesky factor of the exponential kernel *implements* the AR(1)
            recursion, so given the same standard-normal vector this reproduces
            simulate.py's dense-Cholesky draw to floating-point roundoff
            (~1e-9).  The drw arm is therefore a byte-identical rerun of the
            Figure 1 realisation, which makes it an exact control.

  dho       prior draw from celerite2.terms.SHOTerm with Q = 2 and
            t_damp = 2Q/w0 = 320 d  ->  w0 = 2Q/t_damp = 0.0125 rad/d.
            S0 is rescaled so k(0) = sigma^2 exactly.  Note the SHO oscillation
            period is 2*pi/w0 = 502.7 d.

  powerlaw  Timmer & Koenig (1995) with beta = 3, synthesised on a dense grid
            spanning 10x the baseline, then resampled at the observed times
            from a random contiguous window and rescaled.The reason was Kasliwal et al. (2015)

NOTE ON VARIANCE MATCHING: drw and dho are exact process draws whose kernel
satisfies k(0) = sigma^2, so their variance is matched *in expectation*.  A
beta = 3 power law has no stationary variance (its power diverges at low
frequency), so the powerlaw realisation is instead standardised to a *sample*
standard deviation of sigma over the observed window.  This asymmetry is
unavoidable and is recorded in the verdict.
"""

from __future__ import annotations

import numpy as np
from celerite2 import terms

from chirp_model import BandData, phase
from simulate import BAND_NAMES, CONFIG, _cadence

GENERATORS = ("drw", "dho", "powerlaw")

NOISE_CONFIG = {
    "drw": {"kind": "AR(1) recursion at observed times",
            "tau_days": 320.0, "sigma_mag": 0.06},
    "dho": {"kind": "celerite2 SHOTerm prior draw",
            "Q": 2.0, "t_damp_days": 320.0, "sigma_mag": 0.06},
    "powerlaw": {"kind": "Timmer & Koenig", "beta": 3.0,
                 "grid_span_x_baseline": 10.0, "grid_dt_days": 1.0,
                 "sigma_mag": 0.06},
}


def draw_drw_ar1(rng, t, sigma, tau):
    """Exact DRW (Ornstein-Uhlenbeck) draw by AR(1) recursion at observed times.

        x_0     ~ N(0, sigma^2)
        x_{i+1} = x_i * exp(-dt/tau) + sigma * sqrt(1 - exp(-2 dt/tau)) * z

    Exact for arbitrary (unevenly spaced) t, so k(dt) = sigma^2 exp(-|dt|/tau).
    """
    t = np.asarray(t, dtype=float)
    n = t.size
    x = np.empty(n)
    z = rng.standard_normal(n)
    x[0] = sigma * z[0]
    dt = np.diff(t)
    a = np.exp(-dt / tau)
    s = sigma * np.sqrt(1.0 - a ** 2)
    for i in range(1, n):
        x[i] = a[i - 1] * x[i - 1] + s[i - 1] * z[i]
    return x


def draw_dho(rng, t, sigma, t_damp=320.0, Q=2.0):
    """Prior draw from a celerite2 SHOTerm, variance-matched to sigma^2.

    t_damp = 2Q/w0  ->  w0 = 2Q/t_damp.  S0 is set from the measured k(0) so
    the convention used by celerite2 cannot silently change the variance.
    """
    t = np.asarray(t, dtype=float)
    w0 = 2.0 * Q / t_damp
    k0 = terms.SHOTerm(S0=1.0, w0=w0, Q=Q).get_value(np.array([0.0]))[0]
    term = terms.SHOTerm(S0=sigma ** 2 / k0, w0=w0, Q=Q)

    dt = np.abs(t[:, None] - t[None, :])
    K = term.get_value(dt.ravel()).reshape(dt.shape)
    K = 0.5 * (K + K.T)
    K[np.diag_indices_from(K)] += 1e-10 * sigma ** 2
    L = np.linalg.cholesky(K)
    return L @ rng.standard_normal(t.size)


def draw_powerlaw(rng, t, sigma, beta=3.0, baseline=2000.0,
                  span_mult=10.0, grid_dt=1.0):
    """Timmer & Koenig (1995) red noise, resampled at the observed times.

    A long series (span_mult x baseline) is synthesised on a dense grid so the
    observed window sees genuine low-frequency power rather than the FFT's
    periodic wrap; a random contiguous window is then interpolated onto t and
    standardised to a sample standard deviation of sigma.
    """
    t = np.asarray(t, dtype=float)
    span = span_mult * baseline
    n_grid = int(round(span / grid_dt))
    t_grid = np.arange(n_grid) * grid_dt

    freq = np.fft.rfftfreq(n_grid, d=grid_dt)
    amp = np.zeros_like(freq)
    amp[1:] = freq[1:] ** (-beta / 2.0)          # sqrt of S(f) ~ f^-beta

    re = rng.standard_normal(freq.size) * amp / np.sqrt(2.0)
    im = rng.standard_normal(freq.size) * amp / np.sqrt(2.0)
    spec = re + 1j * im
    spec[0] = 0.0
    if n_grid % 2 == 0:                           # Nyquist must be real
        spec[-1] = spec[-1].real * np.sqrt(2.0)
    x_grid = np.fft.irfft(spec, n=n_grid)

    offset = rng.uniform(0.0, span - baseline - grid_dt)
    x = np.interp(t + offset, t_grid, x_grid)

    x = x - x.mean()
    sd = x.std()
    return x * (sigma / sd) if sd > 0 else x


NOISE_DRAW = {
    "drw": lambda rng, t, cfg: draw_drw_ar1(
        rng, t, NOISE_CONFIG["drw"]["sigma_mag"], NOISE_CONFIG["drw"]["tau_days"]),
    "dho": lambda rng, t, cfg: draw_dho(
        rng, t, NOISE_CONFIG["dho"]["sigma_mag"],
        NOISE_CONFIG["dho"]["t_damp_days"], NOISE_CONFIG["dho"]["Q"]),
    "powerlaw": lambda rng, t, cfg: draw_powerlaw(
        rng, t, NOISE_CONFIG["powerlaw"]["sigma_mag"],
        NOISE_CONFIG["powerlaw"]["beta"], cfg["sampling"]["baseline_days"],
        NOISE_CONFIG["powerlaw"]["grid_span_x_baseline"],
        NOISE_CONFIG["powerlaw"]["grid_dt_days"]),
}


def make_dataset_generator(generator: str, cfg: dict = None,
                           noise_seed: int = None):
    """simulate.make_dataset with a pluggable correlated-noise component.

    The RNG stream construction is copied verbatim from simulate.make_dataset,
    so for a given seed the cadence, photometric errors and white noise are
    identical across generators; only the "drw" stream is consumed differently.

    `noise_seed` overrides the correlated-noise stream only.  Passing a
    different value per generator makes the three arms independent draws (no
    arm reuses another's noise realisation) while keeping the cadence, the
    error bars and the white noise byte-identical across arms.
    """
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}")
    cfg = CONFIG if cfg is None else cfg

    root = np.random.default_rng(cfg["seed"])
    noise_root = (None if noise_seed is None
                  else np.random.default_rng(noise_seed))
    stream_seeds = {}
    streams = {}
    for name in BAND_NAMES:
        s = [int(v) for v in root.integers(0, 2 ** 63 - 1, 3)]
        if noise_root is not None:
            s[1] = int(noise_root.integers(0, 2 ** 63 - 1))
        stream_seeds[name] = s
        streams[name] = {k: np.random.default_rng(v)
                         for k, v in zip(("cadence", "drw", "white"), s)}

    t_band = {name: _cadence(streams[name]["cadence"], cfg,
                             cfg["sampling"]["band_start_offset_days"][name])
              for name in BAND_NAMES}
    t_combined = np.sort(np.concatenate([t_band[n] for n in BAND_NAMES]))
    t_ref = float(np.median(t_combined))
    T = float(t_combined.max() - t_combined.min())

    truth = cfg["truth"]
    f0 = 1.0 / truth["P_days"]
    eta = truth["eta"]

    bands, coeffs, noise_sd = [], {}, {}
    for name in BAND_NAMES:
        bcfg = cfg["bands"][name]
        t = t_band[name]
        phi = phase(t, t_ref, f0, eta, T)

        A1, psi1 = bcfg["harm1_amp_mag"], bcfg["harm1_phase_rad"]
        A2, psi2 = bcfg["harm2_amp_mag"], bcfg["harm2_phase_rad"]
        a1, b1 = A1 * np.cos(psi1), A1 * np.sin(psi1)
        a2, b2 = A2 * np.cos(psi2), A2 * np.sin(psi2)
        mu = bcfg["mu_mag"]

        signal = mu + a1 * np.sin(phi) + b1 * np.cos(phi) \
                    + a2 * np.sin(2 * phi) + b2 * np.cos(2 * phi)

        corr = NOISE_DRAW[generator](streams[name]["drw"], t, cfg)
        noise_sd[name] = float(np.std(corr))

        wr = streams[name]["white"]
        yerr = cfg["photometry"]["yerr_median_mag"] * np.exp(
            cfg["photometry"]["yerr_lognormal_scatter"] * wr.standard_normal(t.size))
        white = yerr * wr.standard_normal(t.size)

        bands.append(BandData(name=name, t=t, y=signal + corr + white, yerr=yerr))
        coeffs[name] = {"mu": mu, "a1": a1, "b1": b1, "a2": a2, "b2": b2,
                        "harm1_amp": A1, "harm2_amp": A2}

    meta = {
        "generator": generator,
        "noise_config": NOISE_CONFIG[generator],
        "stream_seeds": stream_seeds,
        "master_seed": int(cfg["seed"]),
        "noise_seed": None if noise_seed is None else int(noise_seed),
        "t_ref_days": t_ref,
        "T_days": T,
        "n_points": {b.name: int(len(b)) for b in bands},
        "realised_noise_sd": noise_sd,
        "injected_coefficients": coeffs,
    }
    return bands, meta
