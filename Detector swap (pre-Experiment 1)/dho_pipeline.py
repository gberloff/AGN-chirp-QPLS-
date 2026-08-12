"""
Pipeline stages for the chirp + DHO detector.

Mirrors run_figure1.py stage for stage - same scan grid, same optimizer restart
policy, same MCMC settings - with the DRW likelihood swapped for the SHOTerm
one.  Shared settings are imported from run_figure1 rather than re-declared so
the two detectors cannot drift apart.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

import emcee

import run_figure1 as R
from dho_model import (
    log_prob_dho, profile_loglike_sho, profile_solution_sho, t_damp,
    theta_bounds_dho, unpack_theta_dho,
)
from chirp_model import draw_beta, is_spd
from simulate import CONFIG

N_WALKERS = R.N_WALKERS
N_STEPS = R.N_STEPS
N_BURN = R.N_BURN
N_RESTARTS = R.N_RESTARTS
SCAN_ETA = R.SCAN_ETA
SCAN_N_PERIOD = R.SCAN_N_PERIOD


def fit_dho_only(bands, t_ref, T):
    """MLE of (sigma_kernel, w0, Q) with the chirp switched off."""
    bounds = theta_bounds_dho()
    kb = [bounds[2], bounds[3], bounds[4]]

    def nll(p):
        sig, w0, Q = np.exp(p)
        ll = profile_loglike_sho(bands, t_ref, 1.0 / 230.0, 0.0, T, sig, w0, Q,
                                 with_chirp=False)
        return 1e10 if not np.isfinite(ll) else -ll

    rng = np.random.default_rng(CONFIG["seed"] + 11)
    starts = [np.array([np.log(0.05), np.log(2 * np.pi / 500.0), np.log(1.0)])]
    for _ in range(N_RESTARTS - 1):
        starts.append(np.array([rng.uniform(*kb[0]), rng.uniform(*kb[1]),
                                rng.uniform(*kb[2])]))
    best = None
    for p0 in starts:
        r = minimize(nll, np.clip(p0, [b[0] for b in kb], [b[1] for b in kb]),
                     method="L-BFGS-B", bounds=kb)
        if best is None or r.fun < best.fun:
            best = r
    sig, w0, Q = np.exp(best.x)
    return float(sig), float(w0), float(Q), float(-best.fun)


def chirp_scan_dho(bands, t_ref, T, sigma_k, w0, Q):
    """Delta log-likelihood vs no chirp over the same (period, eta) grid."""
    periods = np.logspace(np.log10(50.0), np.log10(800.0), SCAN_N_PERIOD)
    ll0 = profile_loglike_sho(bands, t_ref, 1.0 / 230.0, 0.0, T, sigma_k, w0, Q,
                              with_chirp=False)

    dll = np.empty((SCAN_ETA.size, periods.size))
    for i, eta in enumerate(SCAN_ETA):
        for j, P in enumerate(periods):
            dll[i, j] = profile_loglike_sho(bands, t_ref, 1.0 / P, eta, T,
                                            sigma_k, w0, Q) - ll0
        print(f"  eta = {eta:+.2f}: max dlogL = {np.nanmax(dll[i]):9.2f} "
              f"at P = {periods[np.nanargmax(dll[i])]:7.2f} d", flush=True)

    i, j = np.unravel_index(np.nanargmax(dll), dll.shape)
    return {"periods": periods, "eta_grid": SCAN_ETA, "dlogl": dll,
            "ll_nochirp": ll0, "best_period": float(periods[j]),
            "best_eta": float(SCAN_ETA[i]), "best_dlogl": float(dll[i, j])}


def joint_mle_dho(bands, t_ref, T, P0, eta0, sigma_k0, w00, Q0):
    bounds = theta_bounds_dho()

    def nll(theta):
        lp = log_prob_dho(theta, bands, t_ref, T)
        return 1e10 if not np.isfinite(lp) else -lp

    rng = np.random.default_rng(CONFIG["seed"] + 23)
    p0 = np.array([np.log(P0), eta0, np.log(sigma_k0), np.log(w00), np.log(Q0)])
    starts = [p0]
    for _ in range(N_RESTARTS - 1):
        jit = np.array([rng.normal(0, 0.02), rng.normal(0, 0.05),
                        rng.normal(0, 0.15), rng.normal(0, 0.30),
                        rng.normal(0, 0.40)])
        starts.append(np.clip(p0 + jit, [b[0] for b in bounds],
                              [b[1] for b in bounds]))

    best = None
    for k, s in enumerate(starts):
        r = minimize(nll, s, method="L-BFGS-B", bounds=bounds)
        print(f"  restart {k + 1}/{len(starts)}: logL = {-r.fun:.3f}", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, float(-best.fun)


def run_mcmc_dho(bands, t_ref, T, theta_mle):
    bounds = np.array(theta_bounds_dho())
    rng = np.random.default_rng(CONFIG["seed"] + 37)
    scale = np.array([1e-3, 3e-3, 1e-2, 2e-2, 2e-2])

    p0 = theta_mle + scale * rng.standard_normal((N_WALKERS, len(theta_mle)))
    p0 = np.clip(p0, bounds[:, 0] + 1e-9, bounds[:, 1] - 1e-9)

    sampler = emcee.EnsembleSampler(N_WALKERS, len(theta_mle), log_prob_dho,
                                    args=(bands, t_ref, T))
    t0 = time.time()
    sampler.run_mcmc(p0, N_STEPS, progress=False)
    print(f"  {N_STEPS} steps x {N_WALKERS} walkers in {time.time() - t0:.1f} s",
          flush=True)

    acc = float(np.mean(sampler.acceptance_fraction))
    try:
        tau_acf = sampler.get_autocorr_time(discard=N_BURN, quiet=True)
    except Exception:
        tau_acf = np.full(len(theta_mle), np.nan)
    tau_acf = np.asarray(tau_acf, dtype=float)

    finite = tau_acf[np.isfinite(tau_acf)]
    thin = max(1, int(0.5 * np.min(finite))) if finite.size else 1
    chain = sampler.get_chain(discard=N_BURN, thin=thin, flat=True)
    print(f"  acceptance = {acc:.3f}; autocorr times = {np.round(tau_acf, 1)}; "
          f"thin = {thin}; retained = {chain.shape[0]} samples", flush=True)
    return chain, acc, tau_acf, thin


def draw_betas_dho(chain, bands, t_ref, T, seed):
    """beta | theta, y ~ N(beta_hat, (X^T C^-1 X)^-1) at each retained sample."""
    rng = np.random.default_rng(seed)
    names = [b.name for b in bands]
    out = {n: [] for n in names}
    n_fail = n_notspd = 0

    for theta in chain:
        f0, eta, sigma_k, w0, Q = unpack_theta_dho(theta)
        sol = profile_solution_sho(bands, t_ref, f0, eta, T, sigma_k, w0, Q)
        if sol is None:
            n_fail += 1
            for n in names:
                out[n].append(np.full(5, np.nan))
            continue
        for name, (_, beta_hat, A, _) in zip(names, sol):
            if not is_spd(A):
                n_notspd += 1
                out[name].append(np.full(5, np.nan))
                continue
            d = draw_beta(beta_hat, A, rng)
            out[name].append(np.full(5, np.nan) if d is None else d)

    for n in names:
        out[n] = np.asarray(out[n])
    print(f"  beta draws: {chain.shape[0]} samples, {n_fail} GLS failures, "
          f"{n_notspd} non-SPD conditional covariances", flush=True)
    return out, n_fail, n_notspd
