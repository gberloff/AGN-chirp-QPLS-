"""
Mock two-band AGN light curves: asymmetric chirp injected into DRW noise.

Cadence is ZTF-like (~3 d with jitter, ~110 d annual seasonal gaps, independent
time stamps per band).  Each band gets its own DRW realisation drawn from the
same (sigma, tau), plus white photometric noise with ~0.02 mag errors and ~30%
scatter in the error bars.  Everything is derived from one master seed, so the
data set is fully reproducible from results/config.json.
"""

from __future__ import annotations

import numpy as np

from chirp_model import BandData, phase


CONFIG = {
    "seed": 20260725,
    "truth": {
        "P_days": 230.0,
        "eta": 0.25,
        "sigma_drw_mag": 0.06,
        "tau_drw_days": 320.0,
    },
    "sampling": {
        "baseline_days": 2000.0,
        "cadence_days": 3.0,
        "cadence_jitter_frac": 0.35,       # dt = cadence * U(1-j, 1+j)
        "season_period_days": 365.25,
        "season_gap_days": 110.0,
        "season_gap_start_phase_days": 250.0,
        "band_start_offset_days": {"ZTF-g": 0.0, "ZTF-r": 1.7},
    },
    "photometry": {
        "yerr_median_mag": 0.02,
        "yerr_lognormal_scatter": 0.30,    # yerr = median * exp(0.30 * N(0,1))
    },
    "bands": {
        "ZTF-g": {
            "mu_mag": 18.30,
            "harm1_amp_mag": 0.15,
            "harm2_amp_mag": 0.075,        # half the fundamental -> asymmetric
            "harm1_phase_rad": 0.70,
            "harm2_phase_rad": 2.10,
        },
        "ZTF-r": {
            "mu_mag": 17.95,
            "harm1_amp_mag": 0.105,
            "harm2_amp_mag": 0.0525,
            "harm1_phase_rad": 1.30,
            "harm2_phase_rad": -0.50,
        },
    },
    "notes": {
        "drw_bands": "independent DRW realisation per band, shared (sigma, tau)",
        "phase_reference": "x = t - median(combined time stamps)",
        "T_definition": "T = span of the combined time stamps (max - min)",
        "linear_params": "per-band offset mu_j + 4 harmonic coefficients, profiled out by GLS",
    },
}

BAND_NAMES = ["ZTF-g", "ZTF-r"]


def _cadence(rng: np.random.Generator, cfg: dict, start_offset: float) -> np.ndarray:
    """ZTF-like time stamps: jittered ~3 d sampling minus annual seasonal gaps."""
    s = cfg["sampling"]
    baseline = s["baseline_days"]
    cad = s["cadence_days"]
    jit = s["cadence_jitter_frac"]

    times = []
    t = start_offset
    while t < baseline:
        times.append(t)
        t += cad * rng.uniform(1.0 - jit, 1.0 + jit)
    t_all = np.array(times)

    ph = np.mod(t_all, s["season_period_days"])
    gap_lo = s["season_gap_start_phase_days"]
    gap_hi = gap_lo + s["season_gap_days"]
    in_gap = (ph >= gap_lo) & (ph < gap_hi)
    if gap_hi > s["season_period_days"]:            # wrap-around
        in_gap |= ph < (gap_hi - s["season_period_days"])
    return np.sort(t_all[~in_gap])


def _drw_draw(rng: np.random.Generator, t: np.ndarray, sigma: float, tau: float) -> np.ndarray:
    """Exact DRW realisation on `t` via Cholesky of the dense kernel matrix."""
    dt = np.abs(t[:, None] - t[None, :])
    K = sigma ** 2 * np.exp(-dt / tau)
    K[np.diag_indices_from(K)] += 1e-10 * sigma ** 2      # numerical floor only
    L = np.linalg.cholesky(K)
    return L @ rng.standard_normal(t.size)


def make_dataset(cfg: dict = None):
    """Build the two-band mock.

    Returns (bands, meta) where meta records t_ref, T and the injected linear
    coefficients (a1, b1, a2, b2, mu) per band in the model's own basis.
    """
    cfg = CONFIG if cfg is None else cfg
    root = np.random.default_rng(cfg["seed"])
    streams = {
        name: {k: np.random.default_rng(s)
               for k, s in zip(("cadence", "drw", "white"), root.integers(0, 2**63 - 1, 3))}
        for name in BAND_NAMES
    }

    #time stamps first: t_ref and T are defined by the combined sampling ---
    t_band = {
        name: _cadence(streams[name]["cadence"], cfg,
                       cfg["sampling"]["band_start_offset_days"][name])
        for name in BAND_NAMES
    }
    t_combined = np.sort(np.concatenate([t_band[n] for n in BAND_NAMES]))
    t_ref = float(np.median(t_combined))
    T = float(t_combined.max() - t_combined.min())

    truth = cfg["truth"]
    f0 = 1.0 / truth["P_days"]
    eta = truth["eta"]

    bands, coeffs = [], {}
    for name in BAND_NAMES:
        bcfg = cfg["bands"][name]
        t = t_band[name]
        phi = phase(t, t_ref, f0, eta, T)

        # A sin(phi + psi) = (A cos psi) sin phi + (A sin psi) cos phi
        A1, psi1 = bcfg["harm1_amp_mag"], bcfg["harm1_phase_rad"]
        A2, psi2 = bcfg["harm2_amp_mag"], bcfg["harm2_phase_rad"]
        a1, b1 = A1 * np.cos(psi1), A1 * np.sin(psi1)
        a2, b2 = A2 * np.cos(psi2), A2 * np.sin(psi2)
        mu = bcfg["mu_mag"]

        signal = mu + a1 * np.sin(phi) + b1 * np.cos(phi) \
                    + a2 * np.sin(2 * phi) + b2 * np.cos(2 * phi)

        drw = _drw_draw(streams[name]["drw"], t,
                        truth["sigma_drw_mag"], truth["tau_drw_days"])

        wr = streams[name]["white"]
        yerr = cfg["photometry"]["yerr_median_mag"] * np.exp(
            cfg["photometry"]["yerr_lognormal_scatter"] * wr.standard_normal(t.size))
        white = yerr * wr.standard_normal(t.size)

        bands.append(BandData(name=name, t=t, y=signal + drw + white, yerr=yerr))
        coeffs[name] = {"mu": mu, "a1": a1, "b1": b1, "a2": a2, "b2": b2,
                        "harm1_amp": A1, "harm2_amp": A2}

    meta = {
        "t_ref_days": t_ref,
        "T_days": T,
        "n_points": {b.name: int(len(b)) for b in bands},
        "fdot_true_per_day2": float(eta * f0 / T),
        "injected_coefficients": coeffs,
    }
    return bands, meta
