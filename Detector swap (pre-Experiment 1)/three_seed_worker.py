"""
Run the *identical* Figure 1 pipeline for one global RNG seed.

Only CONFIG["seed"] is changed.  Because run_figure1 does `from simulate import
CONFIG`, both modules share one dict object, so mutating the seed here also
redirects every downstream stream (cadence, DRW realisation, white noise, MLE
restart perturbations, walker initialisation, conditional beta draws).
Injected truths, phases, priors, sampler settings and the fit are untouched.

Usage:  python three_seed_worker.py <seed>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import simulate
from simulate import CONFIG, make_dataset
import run_figure1 as R
from chirp_model import unpack_theta

assert R.CONFIG is simulate.CONFIG, "CONFIG must be the same object in both modules"

OUTDIR = os.path.join(R.RESULTS, "seeds")


PHASE_CONVENTION = """\
Phase / amplitude convention (identical for injection and recovery)

  model term, harmonic k:   a_k * sin(k*phi) + b_k * cos(k*phi)
  design-matrix columns:    X = [1, sin(phi), cos(phi), sin(2phi), cos(2phi)]
                            -> beta = [mu, a1, b1, a2, b2]

  INJECTION   (simulate.py::make_dataset)
      a_k = A_k * cos(psi_k)
      b_k = A_k * sin(psi_k)
      so the injected term is exactly  A_k * sin(k*phi + psi_k)

  RECOVERY    (run_figure1.py::main, panels 5 and 6)
      A_k_hat   = hypot(a_k_hat, b_k_hat) = sqrt(a_k^2 + b_k^2)
      psi_k_hat = atan2(b_k_hat, a_k_hat)

  Both directions use the same (sin, cos) column order and the same atan2
  argument order, so A_k and psi_k round-trip without a convention change.
  phi(t) = 2*pi*(f0*x + 0.5*fdot*x^2), x = t - t_ref, and the SAME t_ref
  (median of the combined two-band time stamps) is used for injection and fit.
"""


def circular_mean(angles: np.ndarray) -> float:
    """Circular mean of an angle sample, wrapped to (-pi, pi]."""
    a = np.asarray(angles, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.angle(np.mean(np.exp(1j * a))))


def wrap_pi(x: float) -> float:
    """Wrap an angle difference to (-pi, pi]."""
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def run_seed(seed: int) -> dict:
    CONFIG["seed"] = int(seed)          # The only value that differs between seed runs.
    truth = CONFIG["truth"]

    R.banner(f"seed {seed}: simulate")
    bands, meta = make_dataset()
    t_ref, T = meta["t_ref_days"], meta["T_days"]
    print(f"  N = {meta['n_points']}, T = {T:.2f} d, t_ref = {t_ref:.2f} d",
          flush=True)

    R.banner(f"seed {seed}: DRW-only fit")
    sig0, tau0, ll_drw = R.fit_drw_only(bands, t_ref, T)
    print(f"  sigma = {sig0:.4f}, tau = {tau0:.1f} d, logL = {ll_drw:.2f}",
          flush=True)

    R.banner(f"seed {seed}: chirp scan")
    scan = R.chirp_scan(bands, t_ref, T, sig0, tau0)
    print(f"  best node: P = {scan['best_period']:.2f} d, "
          f"eta = {scan['best_eta']:+.2f}, dlogL = {scan['best_dlogl']:.2f}",
          flush=True)

    R.banner(f"seed {seed}: joint MLE")
    theta_mle, ll_mle = R.joint_mle(bands, t_ref, T, scan["best_period"],
                                    scan["best_eta"], sig0, tau0)
    f0_m, eta_m, sig_m, tau_m = unpack_theta(theta_mle)
    print(f"  MLE: P = {1 / f0_m:.3f} d, eta = {eta_m:+.4f}, "
          f"sigma = {sig_m:.4f}, tau = {tau_m:.1f} d, logL = {ll_mle:.3f}",
          flush=True)

    R.banner(f"seed {seed}: MCMC")
    chain, acceptance, tau_acf, thin = R.run_mcmc(bands, t_ref, T, theta_mle)

    R.banner(f"seed {seed}: conditional beta draws")
    betas, n_fail, n_notspd = R.draw_betas(chain, bands, t_ref, T,
                                           CONFIG["seed"] + 51)

    P_s = np.exp(chain[:, 0])
    eta_s = chain[:, 1]
    sig_s = np.exp(chain[:, 2])
    tau_s = np.exp(chain[:, 3])
    bg = betas["ZTF-g"]
    h1_s = np.hypot(bg[:, 1], bg[:, 2])
    h2_s = np.hypot(bg[:, 3], bg[:, 4])
    # recovered phases, same atan2(b, a) convention as the injection
    psi1_s = np.arctan2(bg[:, 2], bg[:, 1])
    psi2_s = np.arctan2(bg[:, 4], bg[:, 3])

    gcfg = CONFIG["bands"]["ZTF-g"]
    stats = {
        "P": R.summarise(P_s, truth["P_days"]),
        "eta": R.summarise(eta_s, truth["eta"]),
        "sigma": R.summarise(sig_s, truth["sigma_drw_mag"]),
        "tau": R.summarise(tau_s, truth["tau_drw_days"]),
        "g_harm1_amp": R.summarise(h1_s, gcfg["harm1_amp_mag"]),
        "g_harm2_amp": R.summarise(h2_s, gcfg["harm2_amp_mag"]),
    }

    psi1_rec, psi2_rec = circular_mean(psi1_s), circular_mean(psi2_s)
    phases = {
        "injected_psi1_rad": gcfg["harm1_phase_rad"],
        "injected_psi2_rad": gcfg["harm2_phase_rad"],
        "recovered_psi1_rad": psi1_rec,
        "recovered_psi2_rad": psi2_rec,
        "delta_psi1_rad": wrap_pi(psi1_rec - gcfg["harm1_phase_rad"]),
        "delta_psi2_rad": wrap_pi(psi2_rec - gcfg["harm2_phase_rad"]),
    }

    n_post = R.N_STEPS - R.N_BURN
    tau_max = float(np.nanmax(tau_acf)) if np.any(np.isfinite(tau_acf)) else np.nan
    diagnostics = {
        "acceptance": float(acceptance),
        "autocorr_times": [float(v) for v in np.atleast_1d(tau_acf)],
        "tau_max": tau_max,
        "thin": int(thin),
        "n_retained": int(chain.shape[0]),
        "n_post_burn_steps": int(n_post),
        "chain_over_tau": float(n_post / tau_max) if np.isfinite(tau_max) else np.nan,
        "autocorr_ok": bool(np.isfinite(tau_max) and n_post >= 50 * tau_max),
        "beta_draw_failures": int(n_fail),
        "beta_nonspd": int(n_notspd),
    }

    print(f"\n  seed {seed}: harmonic-2 bias = "
          f"{stats['g_harm2_amp']['frac_bias_percent']:+.3f} %, "
          f"harmonic-1 bias = "
          f"{stats['g_harm1_amp']['frac_bias_percent']:+.3f} %", flush=True)
    print(f"  seed {seed}: recovered psi2 = {psi2_rec:+.4f} rad vs injected "
          f"{gcfg['harm2_phase_rad']:+.4f} rad "
          f"(delta = {phases['delta_psi2_rad']:+.4f} rad)", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUTDIR, f"seed_{seed}.npz"),
        chain=chain, P=P_s, eta=eta_s, sigma=sig_s, tau=tau_s,
        beta_g=bg, beta_r=betas["ZTF-r"], harm1_g=h1_s, harm2_g=h2_s,
        psi1_g=psi1_s, psi2_g=psi2_s)

    record = {
        "seed": int(seed),
        "stats": stats,
        "phases": phases,
        "diagnostics": diagnostics,
        "fit": {
            "drw_only_sigma": sig0, "drw_only_tau": tau0, "drw_only_logL": ll_drw,
            "scan_best_period": scan["best_period"],
            "scan_best_eta": scan["best_eta"],
            "scan_best_dlogl": scan["best_dlogl"],
            "mle_P": float(1 / f0_m), "mle_eta": float(eta_m),
            "mle_sigma": float(sig_m), "mle_tau": float(tau_m),
            "mle_logL": ll_mle,
        },
        "n_points": meta["n_points"],
    }
    with open(os.path.join(OUTDIR, f"seed_{seed}.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    return record


if __name__ == "__main__":
    seed_arg = int(sys.argv[1])
    print(PHASE_CONVENTION, flush=True)
    print(f"Sampler settings (unchanged from Figure 1): "
          f"{R.N_WALKERS} walkers x {R.N_STEPS} steps, discard {R.N_BURN}, "
          f"thin by autocorrelation.\n", flush=True)
    run_seed(seed_arg)
    print(f"\nseed {seed_arg}: done", flush=True)
