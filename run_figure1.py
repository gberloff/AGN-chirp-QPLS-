"""
An asymmetric chirp injected into DRW noise in two ZTF bands, recovered by a
joint chirp + DRW model. Reproduction of the figure at the end of the AGN periodicity note. 

Pipeline
    1. simulate the two-band mock
    2. DRW-only fit (no chirp) for baseline hyperparameters
    3. chirp scan over trial period x coarse eta, profiling beta at each node
    4. joint MLE from the best scan node (L-BFGS-B, multiple restarts)
    5. emcee over (ln P, eta, ln sigma, ln tau), with an exact Gaussian
       conditional draw of beta at every retained sample
    6. the annotated 6-panel posterior figure, plus CSV and summary

Run:  python run_figure1.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import skew
import emcee

from chirp_model import (
    PRIOR_BOUNDS, draw_beta, is_spd, log_prob, profile_loglike,
    profile_solution, theta_bounds, unpack_theta,
)
from simulate import CONFIG, make_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Sampler settings, fixed up front and not tuned.
N_WALKERS = 32
N_STEPS = 10000
N_BURN = 2500
N_RESTARTS = 6
SCAN_N_PERIOD = 600
SCAN_ETA = np.round(np.linspace(-0.6, 0.6, 13), 3)   # includes 0 exactly


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def fit_drw_only(bands, t_ref, T):
    """MLE of (sigma, tau) with the chirp switched off (offsets profiled out)."""
    (_, _), (_, _), s_b, t_b = theta_bounds()

    def nll(p):
        sigma, tau = np.exp(p)
        ll = profile_loglike(bands, t_ref, 1.0 / 230.0, 0.0, T, sigma, tau,
                             with_chirp=False)
        return 1e10 if not np.isfinite(ll) else -ll

    best = None
    rng = np.random.default_rng(CONFIG["seed"] + 11)
    starts = [np.array([np.log(0.05), np.log(200.0)])]
    for _ in range(N_RESTARTS - 1):
        starts.append(np.array([rng.uniform(*s_b), rng.uniform(*t_b)]))
    for p0 in starts:
        r = minimize(nll, p0, method="L-BFGS-B", bounds=[s_b, t_b])
        if best is None or r.fun < best.fun:
            best = r
    sigma, tau = np.exp(best.x)
    return float(sigma), float(tau), float(-best.fun)


def chirp_scan(bands, t_ref, T, sigma, tau):
    """Delta log-likelihood vs the no-chirp model over (trial period, eta)."""
    periods = np.logspace(np.log10(PRIOR_BOUNDS["P"][0]),
                          np.log10(PRIOR_BOUNDS["P"][1]), SCAN_N_PERIOD)
    ll0 = profile_loglike(bands, t_ref, 1.0 / 230.0, 0.0, T, sigma, tau,
                          with_chirp=False)

    dll = np.empty((SCAN_ETA.size, periods.size))
    for i, eta in enumerate(SCAN_ETA):
        for j, P in enumerate(periods):
            ll = profile_loglike(bands, t_ref, 1.0 / P, eta, T, sigma, tau)
            dll[i, j] = ll - ll0
        print(f"  eta = {eta:+.2f}: max dlogL = {np.nanmax(dll[i]):9.2f} "
              f"at P = {periods[np.nanargmax(dll[i])]:7.2f} d", flush=True)

    i, j = np.unravel_index(np.nanargmax(dll), dll.shape)
    return {"periods": periods, "eta_grid": SCAN_ETA, "dlogl": dll,
            "ll_nochirp": ll0, "best_period": float(periods[j]),
            "best_eta": float(SCAN_ETA[i]), "best_dlogl": float(dll[i, j])}


def plot_scan(scan, truth_P, path):
    periods, etas, dll = scan["periods"], scan["eta_grid"], scan["dlogl"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8),
                                   gridspec_kw={"height_ratios": [1.15, 1]})

    for i, eta in enumerate(etas):
        ax1.plot(periods, dll[i], lw=0.8, alpha=0.55,
                 color=plt.cm.viridis(i / max(1, len(etas) - 1)))
    ax1.plot(periods, dll.max(axis=0), lw=1.8, color="k",
             label="max over $\\eta$ grid")
    ax1.axvline(truth_P, color="red", lw=1.6, ls="--",
                label=f"injected P = {truth_P:g} d")
    ax1.axvline(scan["best_period"], color="blue", lw=1.4, ls=":",
                label=f"best node = {scan['best_period']:.2f} d, "
                      f"$\\eta$ = {scan['best_eta']:+.2f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("trial period [d]")
    ax1.set_ylabel(r"$\Delta \log \mathcal{L}$ vs no chirp")
    ax1.set_title("Chirp scan: profiled $\\Delta\\log\\mathcal{L}$ "
                  "(DRW hyperparameters fixed at the DRW-only fit)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    extent = [np.log10(periods[0]), np.log10(periods[-1]),
              etas[0] - 0.05, etas[-1] + 0.05]
    im = ax2.imshow(dll, aspect="auto", origin="lower", extent=extent,
                    cmap="magma", vmin=max(-50.0, np.nanmin(dll)))
    ax2.axvline(np.log10(truth_P), color="red", lw=1.4, ls="--")
    ax2.axhline(CONFIG["truth"]["eta"], color="red", lw=1.4, ls="--")
    ax2.set_xlabel(r"$\log_{10}$ trial period [d]")
    ax2.set_ylabel(r"$\eta$")
    fig.colorbar(im, ax=ax2, label=r"$\Delta \log \mathcal{L}$")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def joint_mle(bands, t_ref, T, start_P, start_eta, start_sigma, start_tau):
    bounds = theta_bounds()

    def nll(theta):
        lp = log_prob(theta, bands, t_ref, T)
        return 1e10 if not np.isfinite(lp) else -lp

    rng = np.random.default_rng(CONFIG["seed"] + 23)
    p0 = np.array([np.log(start_P), start_eta,
                   np.log(start_sigma), np.log(start_tau)])
    starts = [p0]
    for _ in range(N_RESTARTS - 1):
        jitter = np.array([rng.normal(0, 0.02), rng.normal(0, 0.05),
                           rng.normal(0, 0.15), rng.normal(0, 0.30)])
        starts.append(np.clip(p0 + jitter,
                              [b[0] for b in bounds], [b[1] for b in bounds]))

    best = None
    for k, s in enumerate(starts):
        r = minimize(nll, s, method="L-BFGS-B", bounds=bounds)
        print(f"  restart {k + 1}/{len(starts)}: logL = {-r.fun:.3f}", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, float(-best.fun)


def run_mcmc(bands, t_ref, T, theta_mle):
    bounds = np.array(theta_bounds())
    rng = np.random.default_rng(CONFIG["seed"] + 37)
    scale = np.array([1e-3, 3e-3, 1e-2, 2e-2])

    p0 = theta_mle + scale * rng.standard_normal((N_WALKERS, len(theta_mle)))
    p0 = np.clip(p0, bounds[:, 0] + 1e-9, bounds[:, 1] - 1e-9)

    sampler = emcee.EnsembleSampler(
        N_WALKERS, len(theta_mle), log_prob, args=(bands, t_ref, T))
    t0 = time.time()
    sampler.run_mcmc(p0, N_STEPS, progress=False)
    print(f"  {N_STEPS} steps x {N_WALKERS} walkers in "
          f"{time.time() - t0:.1f} s", flush=True)

    acc = float(np.mean(sampler.acceptance_fraction))
    try:
        tau_acf = sampler.get_autocorr_time(discard=N_BURN, quiet=True)
    except Exception:
        tau_acf = np.full(len(theta_mle), np.nan)
    tau_acf = np.asarray(tau_acf, dtype=float)

    finite = tau_acf[np.isfinite(tau_acf)]
    thin = max(1, int(0.5 * np.min(finite))) if finite.size else 1
    chain = sampler.get_chain(discard=N_BURN, thin=thin, flat=True)
    print(f"  acceptance = {acc:.3f}; autocorr times = "
          f"{np.round(tau_acf, 1)}; thin = {thin}; "
          f"retained = {chain.shape[0]} samples", flush=True)
    return chain, acc, tau_acf, thin


def draw_betas(chain, bands, t_ref, T, seed):
    """Exact conditional draw beta | theta, y ~ N(beta_hat, (X^T C^-1 X)^-1)."""
    rng = np.random.default_rng(seed)
    names = [b.name for b in bands]
    out = {n: [] for n in names}
    n_fail = n_notspd = 0

    for theta in chain:
        f0, eta, sigma, tau = unpack_theta(theta)
        sol = profile_solution(bands, t_ref, f0, eta, T, sigma, tau)
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
            draw = draw_beta(beta_hat, A, rng)
            out[name].append(np.full(5, np.nan) if draw is None else draw)

    for n in names:
        out[n] = np.asarray(out[n])
    print(f"  beta draws: {chain.shape[0]} samples, {n_fail} GLS failures, "
          f"{n_notspd} non-SPD conditional covariances", flush=True)
    return out, n_fail, n_notspd


def summarise(samples, truth):
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    p16, med, p84 = np.percentile(s, [16.0, 50.0, 84.0])
    return {"truth": float(truth), "median": float(med),
            "p16": float(p16), "p84": float(p84),
            "hi": float(p84 - med), "lo": float(med - p16),
            "frac_bias_percent": float(100.0 * (med - truth) / truth)}


def make_figure(panels, acceptance, paths):
    """2x3 grid of posterior histograms with truth / median / 16-84% band."""
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4))
    axes = axes.ravel()

    for k, (ax, panel) in enumerate(zip(axes, panels)):
        s = np.asarray(panel["samples"], dtype=float)
        s = s[np.isfinite(s)]
        st = panel["stats"]
        fmt = panel["fmt"]

        # display window: clip a heavy tail at p99.5 so the bulk stays legible,
        # but never clip the truth line, the median or the 16-84% band
        lo = min(float(s.min()), st["truth"], st["p16"])
        hi = max(float(np.percentile(s, 99.5)), st["truth"], st["p84"])
        pad = 0.03 * (hi - lo)

        counts, edges, _ = ax.hist(
            s, bins=60, range=(lo, hi), color="0.65", edgecolor="0.35",
            linewidth=0.4, label="posterior samples")
        ax.set_xlim(lo - pad, hi + pad)

        ax.axvspan(st["p16"], st["p84"], color="tab:blue", alpha=0.18,
                   label="16-84% interval")
        ax.axvline(st["truth"], color="red", lw=2.0, label="injected truth")
        ax.axvline(st["median"], color="blue", lw=2.0, ls="--",
                   label="posterior median")

        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel("samples")

        # headroom above the tallest bar so the text box never sits on the data
        # (panel 1 needs more: it also carries the legend)
        fill = 0.52 if k == 0 else 0.62
        ax.set_ylim(0, counts.max() / fill)

        # put the box in whichever top corner has the shorter bars under it
        centers = 0.5 * (edges[:-1] + edges[1:])
        span = edges[-1] - edges[0]
        left_max = counts[centers < edges[0] + 0.45 * span].max(initial=0.0)
        right_max = counts[centers > edges[-1] - 0.45 * span].max(initial=0.0)
        box_right = right_max <= left_max
        if k == 0:
            legend_loc = "upper left" if box_right else "upper right"
            ax.legend(fontsize=7.5, loc=legend_loc, framealpha=0.85)

        text = (f"truth  = {st['truth']:{fmt}}\n"
                f"median = {st['median']:{fmt}} "
                f"(+{st['hi']:{fmt}}/-{st['lo']:{fmt}})\n"
                f"bias   = {st['frac_bias_percent']:+.2f} %")
        ax.text(0.975 if box_right else 0.025, 0.975, text,
                transform=ax.transAxes, ha="right" if box_right else "left",
                va="top", fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.78, edgecolor="0.6", linewidth=0.6))

    fig.suptitle("Joint DRW + asymmetric chirp posterior; "
                 f"acceptance={acceptance:.3f}", fontsize=15, y=0.985)

    by_name = {p["key"]: p["stats"] for p in panels}
    line = (f"Recovery: P = {by_name['P']['median']:.2f} d "
            f"(+{by_name['P']['hi']:.2f}/-{by_name['P']['lo']:.2f}, "
            f"{by_name['P']['frac_bias_percent']:+.2f}% vs "
            f"{by_name['P']['truth']:.0f} d)   |   "
            f"eta = {by_name['eta']['median']:.3f} "
            f"(+{by_name['eta']['hi']:.3f}/-{by_name['eta']['lo']:.3f}, "
            f"{by_name['eta']['frac_bias_percent']:+.2f}% vs "
            f"{by_name['eta']['truth']:.2f})   |   "
            f"tau = {by_name['tau']['median']:.0f} d "
            f"(+{by_name['tau']['hi']:.0f}/-{by_name['tau']['lo']:.0f}, "
            f"{by_name['tau']['frac_bias_percent']:+.2f}% vs "
            f"{by_name['tau']['truth']:.0f} d)")
    fig.text(0.5, 0.945, line, ha="center", va="top", fontsize=9.5)

    fig.tight_layout(rect=[0, 0, 1, 0.932])
    for p in paths:
        fig.savefig(p, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    truth = CONFIG["truth"]

    banner("Step 1: simulate the two-band mock")
    bands, meta = make_dataset()
    t_ref, T = meta["t_ref_days"], meta["T_days"]
    for b in bands:
        print(f"  {b.name}: N = {len(b)}, span = {b.t.max() - b.t.min():.1f} d, "
              f"median err = {np.median(b.yerr):.4f} mag", flush=True)
    print(f"  t_ref = {t_ref:.2f} d, T = {T:.2f} d", flush=True)

    config_out = {
        "config": CONFIG,
        "derived": meta,
        "priors": {
            "P_days": {"type": "log-uniform", "bounds": PRIOR_BOUNDS["P"]},
            "eta": {"type": "uniform", "bounds": PRIOR_BOUNDS["eta"]},
            "sigma_mag": {"type": "log-uniform", "bounds": PRIOR_BOUNDS["sigma"]},
            "tau_days": {"type": "log-uniform", "bounds": PRIOR_BOUNDS["tau"]},
            "beta": "improper flat (profiled out by GLS; sampled from its exact Gaussian conditional)",
        },
        "sampler": {"n_walkers": N_WALKERS, "n_steps": N_STEPS,
                    "n_burn": N_BURN, "n_restarts": N_RESTARTS,
                    "scan_n_period": SCAN_N_PERIOD,
                    "scan_eta_grid": SCAN_ETA.tolist()},
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "emcee": emcee.__version__},
    }
    with open(os.path.join(RESULTS, "config.json"), "w") as fh:
        json.dump(config_out, fh, indent=2)

    banner("Step 2: DRW-only fit (no chirp)")
    sig0, tau0, ll_drw = fit_drw_only(bands, t_ref, T)
    print(f"  sigma = {sig0:.4f} mag, tau = {tau0:.1f} d, logL = {ll_drw:.2f}",
          flush=True)

    banner("Step 3: chirp scan")
    scan = chirp_scan(bands, t_ref, T, sig0, tau0)
    print(f"  best node: P = {scan['best_period']:.2f} d, "
          f"eta = {scan['best_eta']:+.2f}, "
          f"dlogL = {scan['best_dlogl']:.2f}", flush=True)
    np.savez(os.path.join(RESULTS, "scan.npz"), **{
        k: v for k, v in scan.items() if isinstance(v, np.ndarray)})
    plot_scan(scan, truth["P_days"], os.path.join(RESULTS, "scan.png"))

    banner("Step 4: joint MLE")
    theta_mle, ll_mle = joint_mle(bands, t_ref, T, scan["best_period"],
                                  scan["best_eta"], sig0, tau0)
    f0_m, eta_m, sig_m, tau_m = unpack_theta(theta_mle)
    print(f"  MLE: P = {1 / f0_m:.3f} d, eta = {eta_m:+.4f}, "
          f"sigma = {sig_m:.4f}, tau = {tau_m:.1f} d, logL = {ll_mle:.3f}",
          flush=True)

    banner("Step 5: MCMC")
    chain, acceptance, tau_acf, thin = run_mcmc(bands, t_ref, T, theta_mle)

    banner("Step 6: conditional beta draws")
    betas, n_fail, n_notspd = draw_betas(chain, bands, t_ref, T,
                                         CONFIG["seed"] + 51)

    P_s = np.exp(chain[:, 0])
    eta_s = chain[:, 1]
    sig_s = np.exp(chain[:, 2])
    tau_s = np.exp(chain[:, 3])
    bg = betas["ZTF-g"]
    h1_s = np.hypot(bg[:, 1], bg[:, 2])
    h2_s = np.hypot(bg[:, 3], bg[:, 4])

    gcoef = meta["injected_coefficients"]["ZTF-g"]
    panels = [
        dict(key="P", xlabel="reference period [d]", samples=P_s,
             truth=truth["P_days"], fmt=".2f"),
        dict(key="eta", xlabel="eta = fdot T / f0", samples=eta_s,
             truth=truth["eta"], fmt=".3f"),
        dict(key="sigma", xlabel="DRW amplitude", samples=sig_s,
             truth=truth["sigma_drw_mag"], fmt=".3f"),
        dict(key="tau", xlabel="DRW tau [d]", samples=tau_s,
             truth=truth["tau_drw_days"], fmt=".0f"),
        dict(key="g_harm1_amp", xlabel="ZTF-g harmonic 1 amp", samples=h1_s,
             truth=gcoef["harm1_amp"], fmt=".4f"),
        dict(key="g_harm2_amp", xlabel="ZTF-g harmonic 2 amp", samples=h2_s,
             truth=gcoef["harm2_amp"], fmt=".4f"),
    ]
    for p in panels:
        p["stats"] = summarise(p["samples"], p["truth"])

    banner("Step 7: outputs")
    make_figure(panels, acceptance,
                [os.path.join(RESULTS, "figure1_note_annotated.png"),
                 os.path.join(RESULTS, "figure1_note_annotated.pdf")])

    header = ["parameter", "truth", "median", "p16", "p84", "frac_bias_percent"]
    rows = [[p["key"], p["stats"]["truth"], p["stats"]["median"],
             p["stats"]["p16"], p["stats"]["p84"],
             p["stats"]["frac_bias_percent"]] for p in panels]
    with open(os.path.join(RESULTS, "recovered_values.csv"), "w",
              newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(f"{r[0]}," + ",".join(f"{v:.6g}" for v in r[1:]) + "\n")

    widths = [14, 12, 12, 12, 12, 18]
    print("  " + "".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  " + "-" * sum(widths))
    for r in rows:
        cells = [r[0].ljust(widths[0])] + \
                [f"{v:.6g}".ljust(w) for v, w in zip(r[1:], widths[1:])]
        print("  " + "".join(cells))

    sP, se, st = panels[0]["stats"], panels[1]["stats"], panels[3]["stats"]
    tau_skew = float(skew(tau_s))
    tau_width = (st["p84"] - st["p16"]) / st["median"]

    checks = [
        ("P median within ~1% of 230 d",
         abs(sP["frac_bias_percent"]) <= 1.0,
         f"|bias| = {abs(sP['frac_bias_percent']):.3f}% "
         f"(median {sP['median']:.2f} d)"),
        ("eta median within ~15% of 0.25",
         abs(se["frac_bias_percent"]) <= 15.0,
         f"|bias| = {abs(se['frac_bias_percent']):.2f}% "
         f"(median {se['median']:.4f})"),
        ("tau posterior broad and right-skewed, median below 320 d",
         (st["median"] < truth["tau_drw_days"]) and (tau_skew > 0)
         and (tau_width > 0.3),
         f"median = {st['median']:.1f} d (< {truth['tau_drw_days']:.0f}), "
         f"skewness = {tau_skew:+.3f}, "
         f"(p84-p16)/median = {tau_width:.3f}, "
         f"upper arm {st['hi']:.1f} d vs lower arm {st['lo']:.1f} d"),
        ("annotated 6-panel figure exists (truth lines, median lines, bands)",
         all(os.path.exists(os.path.join(RESULTS, f)) for f in
             ("figure1_note_annotated.png", "figure1_note_annotated.pdf"))
         and len(panels) == 6,
         "6 panels, each with 1 red truth line, 1 blue median line, "
         "1 shaded 16-84% band; .png and .pdf both written"),
    ]
    overall = "PASS" if all(c[1] for c in checks) else "FAIL"

    lines = [
        "# Figure 1: joint DRW + asymmetric chirp: validation summary",
        "",
        f"**Overall: {overall}**",
        "",
        "| check | result | evidence |",
        "|---|---|---|",
    ]
    for name, ok, ev in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {ev} |")
    lines += [
        "",
        "## Recovered values",
        "",
        "| parameter | truth | median | p16 | p84 | frac bias % |",
        "|---|---|---|---|---|---|",
    ]
    for p in panels:
        s = p["stats"]
        lines.append(f"| {p['key']} | {s['truth']:.6g} | {s['median']:.6g} | "
                     f"{s['p16']:.6g} | {s['p84']:.6g} | "
                     f"{s['frac_bias_percent']:+.3f} |")
    lines += [
        "",
        "## Run details",
        "",
        f"- Injected: P = {truth['P_days']:g} d, eta = {truth['eta']:g}, "
        f"sigma = {truth['sigma_drw_mag']:g} mag, tau = {truth['tau_drw_days']:g} d",
        f"- Data: {meta['n_points']} points, T = {T:.1f} d, t_ref = {t_ref:.1f} d",
        f"- DRW-only baseline: sigma = {sig0:.4f} mag, tau = {tau0:.1f} d, "
        f"logL = {ll_drw:.2f}",
        f"- Best scan node: P = {scan['best_period']:.2f} d, "
        f"eta = {scan['best_eta']:+.2f}, dlogL = {scan['best_dlogl']:.2f} "
        f"vs no chirp",
        f"- Joint MLE: P = {1 / f0_m:.3f} d, eta = {eta_m:+.4f}, "
        f"sigma = {sig_m:.4f} mag, tau = {tau_m:.1f} d, logL = {ll_mle:.3f}",
        f"- MCMC: {N_WALKERS} walkers x {N_STEPS} steps, discard {N_BURN}, "
        f"thin {thin}, {chain.shape[0]} retained samples",
        f"- Mean acceptance fraction: {acceptance:.4f}",
        f"- Integrated autocorrelation times (lnP, eta, ln sigma, ln tau): "
        f"{np.round(tau_acf, 1).tolist()}",
        f"- Conditional beta draws: {n_fail} GLS failures, "
        f"{n_notspd} non-SPD conditional covariances "
        f"(SPD verified before every draw)",
        "",
        "## Priors",
        "",
        f"- P: log-uniform on [{PRIOR_BOUNDS['P'][0]:g}, "
        f"{PRIOR_BOUNDS['P'][1]:g}] d",
        f"- eta: uniform on [{PRIOR_BOUNDS['eta'][0]:g}, "
        f"{PRIOR_BOUNDS['eta'][1]:g}]",
        f"- sigma: log-uniform on [{PRIOR_BOUNDS['sigma'][0]:g}, "
        f"{PRIOR_BOUNDS['sigma'][1]:g}] mag",
        f"- tau: log-uniform on [{PRIOR_BOUNDS['tau'][0]:g}, "
        f"{PRIOR_BOUNDS['tau'][1]:g}] d",
        "- beta (per-band offsets + harmonic coefficients): improper flat, "
        "profiled out by GLS and redrawn from N(beta_hat, (X^T C^-1 X)^-1)",
        "",
    ]
    with open(os.path.join(RESULTS, "figure1_summary.md"), "w") as fh:
        fh.write("\n".join(lines))

    np.savez_compressed(os.path.join(RESULTS, "posterior_samples.npz"),
                        chain=chain, P=P_s, eta=eta_s, sigma=sig_s, tau=tau_s,
                        beta_g=bg, beta_r=betas["ZTF-r"],
                        harm1_g=h1_s, harm2_g=h2_s)

    banner("Validation gate")
    for name, ok, ev in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {ev}")
    print(f"\n  OVERALL: {overall}\n")


if __name__ == "__main__":
    main()
