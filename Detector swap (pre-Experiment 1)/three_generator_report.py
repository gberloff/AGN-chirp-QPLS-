"""
Overlay the three noise-generator recoveries under the fixed chirp + DRW
detector, and compute the P/eta robustness verdict.

Usage:  python three_generator_report.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_figure1 as R
from noise_models import GENERATORS, NOISE_CONFIG
from three_generator_worker import OUTDIR

RESULTS = R.RESULTS
COLORS = {"drw": "tab:blue", "dho": "tab:orange", "powerlaw": "tab:green"}
LABEL = {"drw": "drw (matched)", "dho": "dho (SHO, Q=2)",
         "powerlaw": "powerlaw (beta=3)"}
SHORT = {"drw": "DRW ", "dho": "DHO ", "powerlaw": "PL  "}

PANELS = [
    ("P", "reference period [d]", ".2f"),
    ("eta", "eta = fdot T / f0", ".3f"),
    ("sigma", "DRW amplitude", ".3f"),
    ("tau", "DRW tau [d]", ".0f"),
    ("g_harm1_amp", "ZTF-g harmonic 1 amp", ".4f"),
    ("g_harm2_amp", "ZTF-g harmonic 2 amp", ".4f"),
]
SAMPLE_KEY = {"P": "P", "eta": "eta", "sigma": "sigma", "tau": "tau",
              "g_harm1_amp": "harm1_g", "g_harm2_amp": "harm2_g"}

P_TOL, ETA_TOL = 1.0, 5.0          # robustness thresholds, per cent
ACC_LO, ACC_HI = 0.40, 0.75


def build_headline(records):
    lines = ["Chirp recovery under a fixed DRW detector:"]
    for r in records:
        sP, se = r["stats"]["P"], r["stats"]["eta"]
        lines.append(
            f" {SHORT[r['generator']]} input: "
            f"P {sP['median']:.2f} d ({sP['frac_bias_percent']:+.2f}%), "
            f"eta {se['median']:.3f} ({se['frac_bias_percent']:+.2f}%)")

    pb = {r["generator"]: abs(r["stats"]["P"]["frac_bias_percent"])
          for r in records}
    eb = {r["generator"]: abs(r["stats"]["eta"]["frac_bias_percent"])
          for r in records}
    aliased = [r["generator"] for r in records if r["aliasing"]["near_460"]]

    robust = all(v <= P_TOL for v in pb.values()) and \
             all(v <= ETA_TOL for v in eb.values())
    if robust:
        verdict = ("Chirp position robust to noise misspecification; "
                   "DRW-timescale recovery is not (see panel 4).")
    else:
        broke = []
        for g in [r["generator"] for r in records]:
            bits = []
            if pb[g] > P_TOL:
                bits.append(f"P off by {pb[g]:.2f}% (tol {P_TOL:g}%)")
            if eb[g] > ETA_TOL:
                bits.append(f"eta off by {eb[g]:.2f}% (tol {ETA_TOL:g}%)")
            if bits:
                broke.append(f"{g}: " + " and ".join(bits))
        verdict = "Chirp position NOT robust - " + "; ".join(broke) + "."
    if aliased:
        verdict += (" ALIASING FLAG: " + ", ".join(aliased) +
                    " has a P posterior within 5% of 2x230 = 460 d.")
    return "\n".join(lines), verdict


def use_log_tau(records, samples):
    """Log x-axis for panel 4 if the three tau posteriors span over a decade."""
    lo = min(r["stats"]["tau"]["p16"] for r in records)
    hi = max(r["stats"]["tau"]["p84"] for r in records)
    return (hi / lo) > 10.0, lo, hi


def make_figure(records, samples, headline, verdict, log_tau, paths):
    n_ref = min(s["P"].size for s in samples)
    truth_tau = records[0]["stats"]["tau"]["truth"]

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.6))
    axes = axes.ravel()

    for k, (key, xlabel, fmt) in enumerate(PANELS):
        ax = axes[k]
        truth = records[0]["stats"][key]["truth"]
        cols = [np.asarray(s[SAMPLE_KEY[key]], dtype=float) for s in samples]
        cols = [c[np.isfinite(c)] for c in cols]
        pooled = np.concatenate(cols)

        logx = (key == "tau") and log_tau
        if logx:
            lo = max(float(pooled.min()), 1e-3)
            hi = float(pooled.max())
            edges = np.logspace(np.log10(lo), np.log10(hi), 61)
        else:
            lo = min(float(pooled.min()), truth)
            hi = max(float(np.percentile(pooled, 99.5)), truth,
                     *[r["stats"][key]["p84"] for r in records])
            edges = np.linspace(lo, hi, 61)

        top = 0.0
        for c, rec in zip(cols, records):
            g = rec["generator"]
            w = np.full(c.size, n_ref / c.size)
            counts, _, _ = ax.hist(c, bins=edges, weights=w,
                                   histtype="stepfilled", alpha=0.45,
                                   color=COLORS[g], edgecolor=COLORS[g],
                                   linewidth=1.4, label=LABEL[g])
            top = max(top, counts.max())
            ax.axvline(rec["stats"][key]["median"], color=COLORS[g], lw=1.8,
                       ls="--")

        tau_legend = None
        if key == "tau":
            # solid red applies to the drw arm only; the dho 320 d is a damping
            # time, not a tau_DRW; the power law has no characteristic timescale
            h_solid = ax.axvline(truth_tau, color="red", lw=2.2)
            h_dash = ax.axvline(truth_tau, color="darkred", lw=2.0,
                                ls=(0, (5, 4)))
            tau_legend = ([h_solid, h_dash],
                          [f"drw true tau = {truth_tau:.0f} d",
                           f"dho t_damp = {truth_tau:.0f} d (not a tau)"])
        else:
            ax.axvline(truth, color="red", lw=2.2,
                       label="injected truth" if k == 0 else None)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("samples")
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(edges[0], edges[-1])
        else:
            pad = 0.03 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(0, top / (0.58 if k in (0, 3) else 0.70))

        centers = 0.5 * (edges[:-1] + edges[1:])
        if logx:
            left_cut = 10 ** (np.log10(edges[0]) +
                              0.45 * (np.log10(edges[-1]) - np.log10(edges[0])))
            right_cut = 10 ** (np.log10(edges[-1]) -
                               0.45 * (np.log10(edges[-1]) - np.log10(edges[0])))
        else:
            span = edges[-1] - edges[0]
            left_cut = edges[0] + 0.45 * span
            right_cut = edges[-1] - 0.45 * span
        lm = max(np.histogram(c, bins=edges)[0][centers < left_cut].max()
                 for c in cols)
        rm = max(np.histogram(c, bins=edges)[0][centers > right_cut].max()
                 for c in cols)
        box_right = rm <= lm

        if k == 0:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(plt.Line2D([], [], color="0.35", ls="--", lw=1.8))
            labels.append("posterior median")
            ax.legend(handles, labels, fontsize=7.5,
                      loc="upper left" if box_right else "upper right",
                      framealpha=0.85)

        if key == "tau":
            ax.legend(tau_legend[0], tau_legend[1], fontsize=7,
                      loc="upper left" if box_right else "upper right",
                      framealpha=0.85)
            true_note = {"drw": "true 320", "dho": "no true tau",
                         "powerlaw": "no timescale"}
            txt = "fitted tau (DRW detector)\n" + "\n".join(
                f"{r['generator']:<8} = {r['stats']['tau']['median']:5.0f} d  "
                f"{true_note[r['generator']]}" for r in records)
        else:
            txt = f"truth  = {truth:{fmt}}\n" + "\n".join(
                f"{r['generator']:<8} = {r['stats'][key]['median']:{fmt}} "
                f"({r['stats'][key]['frac_bias_percent']:+.2f}%)"
                for r in records)
        ax.text(0.975 if box_right else 0.025, 0.975, txt,
                transform=ax.transAxes, ha="right" if box_right else "left",
                va="top", fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.80, edgecolor="0.6", linewidth=0.6))

    fig.suptitle("Identical injected chirp on three noise processes, "
                 "analysed by the same DRW-based detector",
                 fontsize=14, y=0.995)

    fig.text(0.5, 0.965, headline, ha="center", va="top", fontsize=9.5,
             family="monospace")
    fig.text(0.5, 0.893, verdict, ha="center", va="top", fontsize=10.5,
             color="darkred" if "NOT robust" in verdict or "ALIASING" in verdict
             else "black", wrap=True)

    caption = (
        "Truth lines: panels 1, 2, 5, 6 - injected chirp, identical for all "
        "three generators. Panel 3 - 0.06, the matched total stochastic "
        "variance. Panel 4 - the solid red line at 320 d is a true tau_DRW for "
        "the drw generator ONLY; for dho the same 320 d is a damping time "
        "t_damp = 2Q/w0 (dashed, not a tau), and the power law has no "
        "characteristic timescale at all."
        + ("  Panel 4 uses a log x-axis: the three tau posteriors span more "
           "than a decade." if log_tau else ""))
    fig.text(0.5, 0.018, caption, ha="center", va="bottom", fontsize=8.5,
             wrap=True)

    fig.tight_layout(rect=[0, 0.055, 1, 0.878])
    for p in paths:
        fig.savefig(p, dpi=200)
    plt.close(fig)


def main():
    records, samples = [], []
    for g in GENERATORS:
        with open(os.path.join(OUTDIR, f"gen_{g}.json")) as fh:
            records.append(json.load(fh))
        samples.append(dict(np.load(os.path.join(OUTDIR, f"gen_{g}.npz"))))

    R.banner("MCMC mixing check")
    bad = []
    for r in records:
        d = r["diagnostics"]
        ok_acc = ACC_LO <= d["acceptance"] <= ACC_HI
        if not ok_acc or not d["autocorr_ok"]:
            bad.append(r["generator"])
        print(f"  {r['generator']:<9}: acceptance = {d['acceptance']:.3f} "
              f"[{'ok' if ok_acc else 'OUT OF BAND'}], tau_max = "
              f"{d['tau_max']:.1f}, post-burn/tau_max = "
              f"{d['chain_over_tau']:.1f} "
              f"[{'ok' if d['autocorr_ok'] else 'FAILS 50x'}], retained = "
              f"{d['n_retained']}", flush=True)

    log_tau, tau_lo, tau_hi = use_log_tau(records, samples)
    print(f"\n  panel 4: tau posteriors span p16 {tau_lo:.1f} d to p84 "
          f"{tau_hi:.1f} d (ratio {tau_hi / tau_lo:.1f}) -> "
          f"{'log' if log_tau else 'linear'} x-axis", flush=True)

    headline, verdict = build_headline(records)
    R.banner("Recovery")
    print(headline, flush=True)
    R.banner("VERDICT")
    print("  " + verdict + "\n", flush=True)

    make_figure(records, samples, headline, verdict, log_tau,
                [os.path.join(RESULTS, "three_generator_recovery.png"),
                 os.path.join(RESULTS, "three_generator_recovery.pdf")])

    with open(os.path.join(RESULTS, "three_generator_values.csv"), "w",
              newline="") as fh:
        fh.write("generator,parameter,truth,truth_applies,median,p16,p84,"
                 "frac_bias_percent\n")
        for r in records:
            for key, _, _ in PANELS:
                s = r["stats"][key]
                applies = s.get("truth_applies", True)
                fh.write(f"{r['generator']},{key},{s['truth']:.6g},{applies},"
                         f"{s['median']:.6g},{s['p16']:.6g},{s['p84']:.6g},"
                         f"{s['frac_bias_percent']:.6g}\n")

    write_verdict_md(records, headline, verdict, log_tau, bad)
    print("  wrote three_generator_recovery.png/.pdf, "
          "three_generator_values.csv, three_generator_verdict.md", flush=True)


def write_verdict_md(records, headline, verdict, log_tau, bad):
    L = [
        "# Chirp recovery under a fixed DRW detector, three input noise processes",
        "",
        "The same injected chirp (P = 230 d, eta = 0.25, ZTF-g harmonic-1 amp "
        "0.15, harmonic-2 amp 0.075, Figure 1 phases) is placed on three "
        "different noise processes and analysed by the same chirp + DRW "
        "detector. This mirrors reality: the detector is chosen, the true "
        "noise of a source is not known.",
        "",
        "## Verdict",
        "",
        "```",
        headline,
        "```",
        "",
        f"**{verdict}**",
        "",
        "## Recovery per generator",
        "",
        "| generator | parameter | truth | median | p16 | p84 | frac bias % |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        for key, _, _ in PANELS:
            s = r["stats"][key]
            t = f"{s['truth']:.6g}" + ("" if s.get("truth_applies", True)
                                       else " (n/a)")
            L.append(f"| {r['generator']} | {key} | {t} | {s['median']:.6g} | "
                     f"{s['p16']:.6g} | {s['p84']:.6g} | "
                     f"{s['frac_bias_percent']:+.3f} |")

    L += [
        "",
        "## Panel 4: the fitted DRW timescale",
        "",
        "The detector always reports a tau, but only one of the three inputs "
        "actually has one.",
        "",
        "| generator | true timescale | fitted tau median | fitted tau 16-84% |",
        "|---|---|---|---|",
    ]
    true_ts = {"drw": "tau_DRW = 320 d (a genuine tau)",
               "dho": "t_damp = 2Q/w0 = 320 d (a damping time, NOT a tau_DRW)",
               "powerlaw": "none - a beta = 3 power law has no characteristic "
                           "timescale"}
    for r in records:
        s = r["stats"]["tau"]
        L.append(f"| {r['generator']} | {true_ts[r['generator']]} | "
                 f"{s['median']:.1f} d | {s['p16']:.1f} - {s['p84']:.1f} d |")
    L += [
        "",
        f"Panel 4 x-axis: **{'logarithmic' if log_tau else 'linear'}** "
        + ("(the three posteriors span more than a decade)." if log_tau
           else "(the three posteriors span less than a decade)."),
        "",
        "For the misspecified generators the fitted tau is not an estimate of "
        "anything real, it is the DRW kernel's best attempt at absorbing a "
        "correlation structure it cannot represent. A large excursion here is "
        "the expected result, not an error.",
        "",
        "## Aliasing check (P near 2 x 230 = 460 d)",
        "",
        "| generator | P median | within 5% of 460 d | frac samples > 400 d | "
        "frac samples within 5% of 460 d |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        a = r["aliasing"]
        L.append(f"| {r['generator']} | {a['P_median']:.2f} d | "
                 f"{'**YES**' if a['near_460'] else 'no'} | "
                 f"{a['frac_samples_above_400d']:.4f} | "
                 f"{a['frac_samples_within_5pct_of_460']:.4f} |")
    L += [
        "",
        "**A single realisation per generator cannot establish an aliasing "
        "*rate*.** It shows only whether this particular realisation aliased. "
        "The previously reported ~25% aliasing rate for the DRW fitter on DHO "
        "noise would need many realisations per generator to confirm or refute; "
        "that is deliberately out of scope here.",
        "",
        "## Noise generators",
        "",
        "| generator | construction | variance matching |",
        "|---|---|---|",
        "| drw | " + NOISE_CONFIG["drw"]["kind"] +
        f", tau = {NOISE_CONFIG['drw']['tau_days']:g} d | kernel k(0) = "
        "sigma^2 exactly (matched in expectation) |",
        "| dho | " + NOISE_CONFIG["dho"]["kind"] +
        f", Q = {NOISE_CONFIG['dho']['Q']:g}, t_damp = "
        f"{NOISE_CONFIG['dho']['t_damp_days']:g} d "
        f"(w0 = 2Q/t_damp = {2 * NOISE_CONFIG['dho']['Q'] / NOISE_CONFIG['dho']['t_damp_days']:.4f} "
        f"rad/d; oscillation period 2pi/w0 = "
        f"{2 * np.pi / (2 * NOISE_CONFIG['dho']['Q'] / NOISE_CONFIG['dho']['t_damp_days']):.1f} d) "
        "| S0 rescaled so k(0) = sigma^2 (matched in expectation) |",
        "| powerlaw | " + NOISE_CONFIG["powerlaw"]["kind"] +
        f", beta = {NOISE_CONFIG['powerlaw']['beta']:g}, dense grid "
        f"{NOISE_CONFIG['powerlaw']['grid_span_x_baseline']:g}x baseline at "
        f"{NOISE_CONFIG['powerlaw']['grid_dt_days']:g} d, random contiguous "
        "window resampled at the observed times | realisation standardised to "
        "sample sd = sigma |",
        "",
        "A beta = 3 power law has no stationary variance, so it cannot be "
        "matched in expectation the way drw and dho are; its realisation is "
        "standardised instead. The realised correlated-noise standard "
        "deviations were:",
        "",
        "| generator | ZTF-g | ZTF-r |",
        "|---|---|---|",
    ]
    for r in records:
        sd = r["realised_noise_sd"]
        L.append(f"| {r['generator']} | {sd['ZTF-g']:.4f} | {sd['ZTF-r']:.4f} |")

    L += [
        "",
        "The dho oscillation period of 502.7 d is worth noting alongside the "
        "aliasing check: it is the nearest thing the DHO input has to a "
        "preferred timescale, and it sits close to 2 x 230 = 460 d.",
        "",
        "## Controlled comparison",
        "",
        f"- Master seed {records[0]['master_seed']} for all three runs; "
        "per-band stream seeds are identical, so the cadence, the photometric "
        "error bars and the white noise are **byte-identical** across "
        "generators. Only the correlated-noise draw differs.",
        "- Per-band stream seeds (cadence, correlated noise, white): "
        + "; ".join(f"{b} = {v}" for b, v in
                    records[0]["stream_seeds"].items()),
        "- Detector, injected chirp, priors and MCMC settings identical to "
        f"Figure 1: {R.N_WALKERS} walkers x {R.N_STEPS} steps, discard "
        f"{R.N_BURN}, thin by autocorrelation.",
        "- Single realisation per generator by design: this is one comparative "
        "figure, not a statistical study.",
        "- The `drw` arm reproduces the Figure 1 realisation exactly. For a "
        "Markov (OU) process the Cholesky factor of the exponential kernel "
        "implements the AR(1) recursion, so given the same standard-normal "
        "vector the two draws agree to ~1e-9. It is an exact control, not "
        "merely a statistically equivalent one.",
        "",
        "## MCMC mixing",
        "",
        "| generator | acceptance | autocorr times | tau_max | "
        "post-burn/tau_max | retained | beta failures | non-SPD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        d = r["diagnostics"]
        L.append(f"| {r['generator']} | {d['acceptance']:.4f} | "
                 f"{np.round(d['autocorr_times'], 1).tolist()} | "
                 f"{d['tau_max']:.1f} | {d['chain_over_tau']:.1f} | "
                 f"{d['n_retained']} | {d['beta_draw_failures']} | "
                 f"{d['beta_nonspd']} |")
    L += ["",
          ("All three chains mix comparably to the original Figure 1 run "
           "(acceptance ~0.59) and pass the 50x autocorrelation check."
           if not bad else
           "**Mixing concerns: " + ", ".join(bad) + ".** A long "
           "autocorrelation time is expected where tau is unidentified (the "
           "power-law case in particular); treat the affected posterior widths "
           "as indicative."),
          ""]

    with open(os.path.join(RESULTS, "three_generator_verdict.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
