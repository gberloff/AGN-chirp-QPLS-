"""
Run the fixed chirp + DHO detector on one noise generator.

Usage:  python dho_generator_worker.py <drw|dho|powerlaw>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from simulate import CONFIG
import run_figure1 as R
import dho_pipeline as D
from dho_model import SHO_PRIOR_BOUNDS, t_damp, unpack_theta_dho
from noise_models import GENERATORS, NOISE_CONFIG, make_dataset_generator

OUTDIR = os.path.join(R.RESULTS, "generators_dho")

# Independent correlated-noise seed per generator, so no arm reuses another's
# (or Figure 1's) noise realisation.  Cadence, error bars and white noise stay
# byte-identical across arms.
NOISE_SEEDS = {"drw": 31415926, "dho": 27182818, "powerlaw": 16180339}

# Injected truths for the DHO kernel, meaningful only for the matched arm.
DHO_TRUTH_Q = NOISE_CONFIG["dho"]["Q"]
DHO_TRUTH_TDAMP = NOISE_CONFIG["dho"]["t_damp_days"]
DHO_TRUTH_W0 = 2.0 * DHO_TRUTH_Q / DHO_TRUTH_TDAMP


def rail_check(samples, lo, hi, tol=0.02):
    """Fraction of samples within `tol` (in log units) of each prior bound."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    span = np.log(hi) - np.log(lo)
    return {
        "frac_at_low": float(np.mean((np.log(s) - np.log(lo)) < tol * span)),
        "frac_at_high": float(np.mean((np.log(hi) - np.log(s)) < tol * span)),
    }


def run_generator(generator: str) -> dict:
    truth = CONFIG["truth"]
    noise_seed = NOISE_SEEDS[generator]

    R.banner(f"{generator}: simulate (noise seed {noise_seed})")
    bands, meta = make_dataset_generator(generator, noise_seed=noise_seed)
    t_ref, T = meta["t_ref_days"], meta["T_days"]
    print(f"  noise model: {meta['noise_config']}", flush=True)
    print(f"  N = {meta['n_points']}, T = {T:.2f} d, t_ref = {t_ref:.2f} d",
          flush=True)
    print(f"  realised correlated-noise sd = "
          f"{ {k: round(v, 4) for k, v in meta['realised_noise_sd'].items()} }",
          flush=True)

    R.banner(f"{generator}: DHO-only fit")
    sig0, w00, Q0, ll_dho = D.fit_dho_only(bands, t_ref, T)
    print(f"  sigma_kernel = {sig0:.4f}, w0 = {w00:.6f} rad/d, Q = {Q0:.4f}, "
          f"t_damp = {t_damp(Q0, w00):.1f} d, logL = {ll_dho:.2f}", flush=True)

    R.banner(f"{generator}: chirp scan")
    scan = D.chirp_scan_dho(bands, t_ref, T, sig0, w00, Q0)
    print(f"  best node: P = {scan['best_period']:.2f} d, "
          f"eta = {scan['best_eta']:+.2f}, dlogL = {scan['best_dlogl']:.2f}",
          flush=True)

    R.banner(f"{generator}: joint MLE")
    theta_mle, ll_mle = D.joint_mle_dho(bands, t_ref, T, scan["best_period"],
                                        scan["best_eta"], sig0, w00, Q0)
    f0_m, eta_m, sig_m, w0_m, Q_m = unpack_theta_dho(theta_mle)
    print(f"  MLE: P = {1 / f0_m:.3f} d, eta = {eta_m:+.4f}, "
          f"sigma_kernel = {sig_m:.4f}, w0 = {w0_m:.6f}, Q = {Q_m:.4f}, "
          f"t_damp = {t_damp(Q_m, w0_m):.1f} d, logL = {ll_mle:.3f}", flush=True)

    R.banner(f"{generator}: MCMC")
    chain, acceptance, tau_acf, thin = D.run_mcmc_dho(bands, t_ref, T, theta_mle)

    R.banner(f"{generator}: conditional beta draws")
    betas, n_fail, n_notspd = D.draw_betas_dho(chain, bands, t_ref, T,
                                               CONFIG["seed"] + 51)

    P_s = np.exp(chain[:, 0])
    eta_s = chain[:, 1]
    amp_s = np.exp(chain[:, 2])          # kernel marginal RMS
    w0_s = np.exp(chain[:, 3])
    Q_s = np.exp(chain[:, 4])
    tdamp_s = t_damp(Q_s, w0_s)
    bg = betas["ZTF-g"]
    h1_s = np.hypot(bg[:, 1], bg[:, 2])
    h2_s = np.hypot(bg[:, 3], bg[:, 4])

    gcfg = CONFIG["bands"]["ZTF-g"]
    stats = {
        "P": R.summarise(P_s, truth["P_days"]),
        "eta": R.summarise(eta_s, truth["eta"]),
        "amp": R.summarise(amp_s, truth["sigma_drw_mag"]),
        "t_damp": R.summarise(tdamp_s, DHO_TRUTH_TDAMP),
        "g_harm1_amp": R.summarise(h1_s, gcfg["harm1_amp_mag"]),
        "g_harm2_amp": R.summarise(h2_s, gcfg["harm2_amp_mag"]),
        "Q": R.summarise(Q_s, DHO_TRUTH_Q),
        "w0": R.summarise(w0_s, DHO_TRUTH_W0),
    }
    for key in ("t_damp", "Q", "w0"):
        stats[key]["truth_applies"] = (generator == "dho")

    alias = {
        "P_median": stats["P"]["median"],
        "near_460": bool(abs(stats["P"]["median"] - 460.0) / 460.0 < 0.05),
        "frac_samples_above_400d": float(np.mean(P_s > 400.0)),
        "frac_samples_within_5pct_of_460": float(
            np.mean(np.abs(P_s - 460.0) / 460.0 < 0.05)),
    }

    railing = {
        "Q": rail_check(Q_s, *SHO_PRIOR_BOUNDS["Q"]),
        "w0": rail_check(w0_s, *SHO_PRIOR_BOUNDS["w0"]),
        "amp": rail_check(amp_s, *SHO_PRIOR_BOUNDS["sigma_kernel"]),
        "Q_median": float(np.median(Q_s)),
        "overdamped_fraction": float(np.mean(Q_s < 0.5)),
    }

    n_post = D.N_STEPS - D.N_BURN
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

    print(f"\n  {generator}: P = {stats['P']['median']:.2f} d "
          f"({stats['P']['frac_bias_percent']:+.2f}%), "
          f"eta = {stats['eta']['median']:.4f} "
          f"({stats['eta']['frac_bias_percent']:+.2f}%)", flush=True)
    print(f"  {generator}: t_damp = {stats['t_damp']['median']:.1f} d "
          f"[{stats['t_damp']['p16']:.1f}-{stats['t_damp']['p84']:.1f}], "
          f"Q = {stats['Q']['median']:.3f}, "
          f"overdamped fraction = {railing['overdamped_fraction']:.3f}",
          flush=True)
    if railing["Q"]["frac_at_low"] > 0.1:
        print(f"  {generator}: *** Q railing LOW ("
              f"{railing['Q']['frac_at_low']:.2f} of samples) - kernel is "
              f"trying to become a DRW ***", flush=True)
    if railing["Q"]["frac_at_high"] > 0.1:
        print(f"  {generator}: *** Q railing HIGH ("
              f"{railing['Q']['frac_at_high']:.2f}) - forcing sharp "
              f"quasi-periodicity ***", flush=True)
    if alias["near_460"]:
        print(f"  {generator}: *** P median within 5% of 460 d - aliasing ***",
              flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUTDIR, f"gen_{generator}.npz"),
        chain=chain, P=P_s, eta=eta_s, amp=amp_s, w0=w0_s, Q=Q_s,
        t_damp=tdamp_s, beta_g=bg, beta_r=betas["ZTF-r"],
        harm1_g=h1_s, harm2_g=h2_s)

    record = {
        "generator": generator,
        "detector": "chirp + SHOTerm (DHO)",
        "noise_config": NOISE_CONFIG[generator],
        "master_seed": meta["master_seed"],
        "noise_seed": meta["noise_seed"],
        "stream_seeds": meta["stream_seeds"],
        "realised_noise_sd": meta["realised_noise_sd"],
        "stats": stats,
        "aliasing": alias,
        "railing": railing,
        "diagnostics": diagnostics,
        "fit": {
            "dho_only_sigma_kernel": sig0, "dho_only_w0": w00,
            "dho_only_Q": Q0, "dho_only_t_damp": float(t_damp(Q0, w00)),
            "dho_only_logL": ll_dho,
            "scan_best_period": scan["best_period"],
            "scan_best_eta": scan["best_eta"],
            "scan_best_dlogl": scan["best_dlogl"],
            "mle_P": float(1 / f0_m), "mle_eta": float(eta_m),
            "mle_amp": float(sig_m), "mle_w0": float(w0_m),
            "mle_Q": float(Q_m), "mle_t_damp": float(t_damp(Q_m, w0_m)),
            "mle_logL": ll_mle,
        },
        "n_points": meta["n_points"],
    }
    with open(os.path.join(OUTDIR, f"gen_{generator}.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    return record


if __name__ == "__main__":
    gen = sys.argv[1]
    if gen not in GENERATORS:
        raise SystemExit(f"generator must be one of {GENERATORS}")
    print(f"Detector fixed: chirp + DHO (celerite2 SHOTerm).  "
          f"Noise generator under test: {gen}\n"
          f"Q prior [{SHO_PRIOR_BOUNDS['Q'][0]:g}, {SHO_PRIOR_BOUNDS['Q'][1]:g}] "
          f"- overdamped regime reachable, Q not floored at 0.5.\n"
          f"Sampler: {D.N_WALKERS} walkers x {D.N_STEPS} steps, discard "
          f"{D.N_BURN}, thin by autocorrelation.\n", flush=True)
    run_generator(gen)
    print(f"\n{gen}: done", flush=True)
