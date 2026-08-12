"""
Run the fixed chirp + DRW detector on one noise generator.

Only the correlated-noise generator changes.  The detector (DRW + chirp), the
injected chirp, the cadence, the photometric errors, the white noise, the
priors and the MCMC settings are identical to the Figure 1 run.

Usage:  python three_generator_worker.py <drw|dho|powerlaw>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from simulate import CONFIG
import run_figure1 as R
from chirp_model import unpack_theta
from noise_models import GENERATORS, NOISE_CONFIG, make_dataset_generator

OUTDIR = os.path.join(R.RESULTS, "generators")


def run_generator(generator: str) -> dict:
    truth = CONFIG["truth"]

    R.banner(f"{generator}: simulate")
    bands, meta = make_dataset_generator(generator)
    t_ref, T = meta["t_ref_days"], meta["T_days"]
    print(f"  noise model: {meta['noise_config']}", flush=True)
    print(f"  N = {meta['n_points']}, T = {T:.2f} d, t_ref = {t_ref:.2f} d",
          flush=True)
    print(f"  realised correlated-noise sd = "
          f"{ {k: round(v, 4) for k, v in meta['realised_noise_sd'].items()} }",
          flush=True)

    R.banner(f"{generator}: DRW-only fit")
    sig0, tau0, ll_drw = R.fit_drw_only(bands, t_ref, T)
    print(f"  sigma = {sig0:.4f}, tau = {tau0:.1f} d, logL = {ll_drw:.2f}",
          flush=True)

    R.banner(f"{generator}: chirp scan")
    scan = R.chirp_scan(bands, t_ref, T, sig0, tau0)
    print(f"  best node: P = {scan['best_period']:.2f} d, "
          f"eta = {scan['best_eta']:+.2f}, dlogL = {scan['best_dlogl']:.2f}",
          flush=True)

    R.banner(f"{generator}: joint MLE")
    theta_mle, ll_mle = R.joint_mle(bands, t_ref, T, scan["best_period"],
                                    scan["best_eta"], sig0, tau0)
    f0_m, eta_m, sig_m, tau_m = unpack_theta(theta_mle)
    print(f"  MLE: P = {1 / f0_m:.3f} d, eta = {eta_m:+.4f}, "
          f"sigma = {sig_m:.4f}, tau = {tau_m:.1f} d, logL = {ll_mle:.3f}",
          flush=True)

    R.banner(f"{generator}: MCMC")
    chain, acceptance, tau_acf, thin = R.run_mcmc(bands, t_ref, T, theta_mle)

    R.banner(f"{generator}: conditional beta draws")
    betas, n_fail, n_notspd = R.draw_betas(bands=bands, chain=chain, t_ref=t_ref,
                                           T=T, seed=CONFIG["seed"] + 51)

    P_s = np.exp(chain[:, 0])
    eta_s = chain[:, 1]
    sig_s = np.exp(chain[:, 2])
    tau_s = np.exp(chain[:, 3])
    bg = betas["ZTF-g"]
    h1_s = np.hypot(bg[:, 1], bg[:, 2])
    h2_s = np.hypot(bg[:, 3], bg[:, 4])

    gcfg = CONFIG["bands"]["ZTF-g"]
    # Panel-4 truth is generator dependent: only the drw arm has a true
    # tau_DRW.  Bias vs 320 d is still reported for reference, flagged below.
    stats = {
        "P": R.summarise(P_s, truth["P_days"]),
        "eta": R.summarise(eta_s, truth["eta"]),
        "sigma": R.summarise(sig_s, truth["sigma_drw_mag"]),
        "tau": R.summarise(tau_s, truth["tau_drw_days"]),
        "g_harm1_amp": R.summarise(h1_s, gcfg["harm1_amp_mag"]),
        "g_harm2_amp": R.summarise(h2_s, gcfg["harm2_amp_mag"]),
    }
    stats["tau"]["truth_applies"] = (generator == "drw")

    # aliasing diagnostic: does this realisation sit near 2 x 230 = 460 d?
    alias = {
        "P_median": stats["P"]["median"],
        "near_460": bool(abs(stats["P"]["median"] - 460.0) / 460.0 < 0.05),
        "frac_samples_above_400d": float(np.mean(P_s > 400.0)),
        "frac_samples_within_5pct_of_460": float(
            np.mean(np.abs(P_s - 460.0) / 460.0 < 0.05)),
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

    print(f"\n  {generator}: P = {stats['P']['median']:.2f} d "
          f"({stats['P']['frac_bias_percent']:+.2f}%), "
          f"eta = {stats['eta']['median']:.4f} "
          f"({stats['eta']['frac_bias_percent']:+.2f}%), "
          f"fitted tau = {stats['tau']['median']:.1f} d", flush=True)
    if alias["near_460"]:
        print(f"  {generator}: *** P median is within 5% of 460 d - "
              f"aliasing flag ***", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUTDIR, f"gen_{generator}.npz"),
        chain=chain, P=P_s, eta=eta_s, sigma=sig_s, tau=tau_s,
        beta_g=bg, beta_r=betas["ZTF-r"], harm1_g=h1_s, harm2_g=h2_s)

    record = {
        "generator": generator,
        "noise_config": NOISE_CONFIG[generator],
        "master_seed": meta["master_seed"],
        "stream_seeds": meta["stream_seeds"],
        "realised_noise_sd": meta["realised_noise_sd"],
        "stats": stats,
        "aliasing": alias,
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
    with open(os.path.join(OUTDIR, f"gen_{generator}.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    return record


if __name__ == "__main__":
    gen = sys.argv[1]
    if gen not in GENERATORS:
        raise SystemExit(f"generator must be one of {GENERATORS}")
    print(f"Detector fixed: chirp + DRW.  Noise generator under test: {gen}\n"
          f"Sampler settings (unchanged from Figure 1): {R.N_WALKERS} walkers x "
          f"{R.N_STEPS} steps, discard {R.N_BURN}, thin by autocorrelation.\n",
          flush=True)
    run_generator(gen)
    print(f"\n{gen}: done", flush=True)
