"""
Overlay the three noise-generator recoveries under the fixed chirp + DHO
detector.  Layout matches three_generator_report.py so the two detector figures
can be compared side by side.

Usage:  python dho_generator_report.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_figure1 as R
import dho_pipeline as D
from dho_model import SHO_PRIOR_BOUNDS
from noise_models import GENERATORS, NOISE_CONFIG
from dho_generator_worker import (
    DHO_TRUTH_Q, DHO_TRUTH_TDAMP, DHO_TRUTH_W0, NOISE_SEEDS, OUTDIR,
)

RESULTS = R.RESULTS
COLORS = {"drw": "tab:blue", "dho": "tab:orange", "powerlaw": "tab:green"}
LABEL = {"drw": "drw (beta=1 / OU)", "dho": "dho (SHO, Q=2) [matched]",
         "powerlaw": "powerlaw (beta=3)"}
SHORT = {"drw": "DRW ", "dho": "DHO ", "powerlaw": "PL  "}

PANELS = [
    ("P", "reference period [d]", ".2f"),
    ("eta", "eta = fdot T / f0", ".3f"),
    ("amp", "DHO stochastic amplitude", ".3f"),
    ("t_damp", "DHO t_damp [d]", ".0f"),
    ("g_harm1_amp", "ZTF-g harmonic 1 amp", ".4f"),
    ("g_harm2_amp", "ZTF-g harmonic 2 amp", ".4f"),
]
SAMPLE_KEY = {"P": "P", "eta": "eta", "amp": "amp", "t_damp": "t_damp",
              "g_harm1_amp": "harm1_g", "g_harm2_amp": "harm2_g"}
CSV_PARAMS = [p[0] for p in PANELS] + ["Q", "w0"]

P_TOL, ETA_TOL = 1.0, 5.0
ACC_LO, ACC_HI = 0.40, 0.75

DRW_DIR = os.path.join(RESULTS, "generators")


def load_drw_detector():
    """Earlier DRW-detector results, if present, for comparison."""
    out = {}
    for g in GENERATORS:
        p = os.path.join(DRW_DIR, f"gen_{g}.json")
        if os.path.exists(p):
            with open(p) as fh:
                out[g] = json.load(fh)
    return out


def build_headline(records):
    lines = ["Chirp recovery under a fixed DHO detector:"]
    for r in records:
        sP, se = r["stats"]["P"], r["stats"]["eta"]
        tag = "   [matched]" if r["generator"] == "dho" else ""
        lines.append(
            f" {SHORT[r['generator']]} input: "
            f"P {sP['median']:.2f} d ({sP['frac_bias_percent']:+.2f}%), "
            f"eta {se['median']:.3f} ({se['frac_bias_percent']:+.2f}%){tag}")

    pb = {r["generator"]: abs(r["stats"]["P"]["frac_bias_percent"])
          for r in records}
    eb = {r["generator"]: abs(r["stats"]["eta"]["frac_bias_percent"])
          for r in records}

    robust = all(v <= P_TOL for v in pb.values()) and \
             all(v <= ETA_TOL for v in eb.values())
    if robust:
        verdict = ("Chirp position robust under the DHO detector; "
                   "stochastic-timescale recovery varies with input "
                   "(see panel 4).")
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
    return "\n".join(lines), verdict


def aliasing_note(records, drw_det):
    """Aliasing flag, compared with the DRW detector where available."""
    aliased = [r["generator"] for r in records if r["aliasing"]["near_460"]]
    bits = []
    if aliased:
        bits.append("ALIASING FLAG: " + ", ".join(aliased) +
                    " has a P posterior within 5% of 2x230 = 460 d.")
    else:
        bits.append("No aliasing: every P posterior sits at the injected "
                    "period, none within 5% of 460 d.")
    if "dho" in drw_det:
        a_dho_here = next(r for r in records
                          if r["generator"] == "dho")["aliasing"]
        a_dho_drw = drw_det["dho"]["aliasing"]
        bits.append(
            f"On DHO input the DHO detector puts "
            f"{a_dho_here['frac_samples_within_5pct_of_460']:.4f} of samples "
            f"within 5% of 460 d (P median {a_dho_here['P_median']:.2f} d); "
            f"the earlier DRW detector on DHO input put "
            f"{a_dho_drw['frac_samples_within_5pct_of_460']:.4f} "
            f"(P median {a_dho_drw['P_median']:.2f} d).")
    return " ".join(bits)


def tdamp_note(records):
    """Whether t_damp is recovered near 320 d on the matched arm, or only weakly constrained."""
    r = next(r for r in records if r["generator"] == "dho")
    s = r["stats"]["t_damp"]
    width = (s["p84"] - s["p16"]) / s["median"]
    near = abs(s["median"] - DHO_TRUTH_TDAMP) / DHO_TRUTH_TDAMP <= 0.25
    well = near and width < 1.0
    txt = (f"Matched (dho) arm: t_damp median {s['median']:.0f} d "
           f"[{s['p16']:.0f}-{s['p84']:.0f}] vs true {DHO_TRUTH_TDAMP:.0f} d, "
           f"fractional 16-84% width {width:.2f}; Q median "
           f"{r['stats']['Q']['median']:.2f} (true {DHO_TRUTH_Q:g}). ")
    if well:
        txt += "t_damp is recovered near the truth and reasonably constrained."
    elif near:
        txt += ("t_damp brackets the truth but is weakly identified. The "
                "posterior is broad.")
    else:
        txt += ("t_damp is NOT recovered near the truth. The DHO timescale is "
                "weakly identified even on its own data.")
    return txt


def overdamped_timescales(Q, w0):
    """The two real decay times of an overdamped SHO (Q < 1/2).

    Roots of s^2 + (w0/Q) s + w0^2 give decay rates
        r_pm = (w0 / 2Q) * (1 +/- sqrt(1 - 4 Q^2)),
    i.e. a fast and a slow exponential.  For Q < 1/2 the nominal
    t_damp = 2Q/w0 is NOT either of them, so quoting it
    alone misrepresents how the kernel is actually behaving.
    """
    if Q >= 0.5:
        return None
    disc = np.sqrt(1.0 - 4.0 * Q ** 2)
    r_fast = (w0 / (2.0 * Q)) * (1.0 + disc)
    r_slow = (w0 / (2.0 * Q)) * (1.0 - disc)
    return 1.0 / r_fast, 1.0 / r_slow


def use_log_tdamp(records):
    lo = min(r["stats"]["t_damp"]["p16"] for r in records)
    hi = max(r["stats"]["t_damp"]["p84"] for r in records)
    return (hi / lo) > 10.0, lo, hi


def make_figure(records, samples, headline, verdict, log_td, paths):
    n_ref = min(s["P"].size for s in samples)

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.6))
    axes = axes.ravel()

    for k, (key, xlabel, fmt) in enumerate(PANELS):
        ax = axes[k]
        truth = records[0]["stats"][key]["truth"]
        cols = [np.asarray(s[SAMPLE_KEY[key]], dtype=float) for s in samples]
        cols = [c[np.isfinite(c)] for c in cols]
        pooled = np.concatenate(cols)

        logx = (key == "t_damp") and log_td
        if logx:
            lo, hi = max(float(pooled.min()), 1e-3), float(pooled.max())
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

        td_legend = None
        if key == "t_damp":
            # solid red = the matched dho arm's genuine t_damp; dashed = the
            # DRW tau, only an approximate counterpart; powerlaw has none
            h_solid = ax.axvline(DHO_TRUTH_TDAMP, color="red", lw=2.2)
            h_dash = ax.axvline(DHO_TRUTH_TDAMP, color="darkred", lw=2.0,
                                ls=(0, (5, 4)))
            td_legend = ([h_solid, h_dash],
                         [f"dho true t_damp = {DHO_TRUTH_TDAMP:.0f} d",
                          f"drw tau = {DHO_TRUTH_TDAMP:.0f} d (approx match)"])
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
        ax.set_ylim(0, top / (0.55 if k in (0, 3) else 0.70))

        centers = 0.5 * (edges[:-1] + edges[1:])
        if logx:
            lg = np.log10(edges)
            left_cut = 10 ** (lg[0] + 0.45 * (lg[-1] - lg[0]))
            right_cut = 10 ** (lg[-1] - 0.45 * (lg[-1] - lg[0]))
        else:
            span = edges[-1] - edges[0]
            left_cut, right_cut = edges[0] + 0.45 * span, edges[-1] - 0.45 * span
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

        if key == "t_damp":
            ax.legend(td_legend[0], td_legend[1], fontsize=7,
                      loc="upper left" if box_right else "upper right",
                      framealpha=0.85)
            note = {"drw": "no true t_damp", "dho": "true 320",
                    "powerlaw": "no timescale"}
            txt = "fitted t_damp (DHO detector)\n" + "\n".join(
                f"{r['generator']:<8} = {r['stats']['t_damp']['median']:6.0f} d"
                f"  {note[r['generator']]}" for r in records)
            txt += "\n" + "\n".join(
                f"{r['generator']:<8} Q = {r['stats']['Q']['median']:.2f}"
                for r in records)
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
                 "analysed by the same DHO (SHOTerm) detector",
                 fontsize=14, y=0.995)
    fig.text(0.5, 0.965, headline, ha="center", va="top", fontsize=9.5,
             family="monospace")
    fig.text(0.5, 0.885, verdict, ha="center", va="top", fontsize=10.5,
             color="darkred" if "NOT robust" in verdict else "black")

    caption = (
        "Truth lines: panels 1, 2, 5, 6 - injected chirp, identical for all "
        "three generators. Panel 3 - 0.06, the matched total stochastic "
        "variance, compared against the SHOTerm marginal RMS sqrt(S0*w0*Q). "
        "Panel 4 - the solid red line at 320 d is a genuine t_damp for the "
        "matched dho generator ONLY; for drw the same 320 d is a DRW tau, an "
        "approximate counterpart rather than a t_damp (dashed), and the power "
        "law has no characteristic timescale at all. Q is quoted per generator "
        "because t_damp alone hides the damping regime (Q < 0.5 overdamped / "
        "DRW-like, Q > 0.5 underdamped)."
        + ("  Panel 4 uses a log x-axis: the three posteriors span more than a "
           "decade." if log_td else ""))
    fig.text(0.5, 0.018, caption, ha="center", va="bottom", fontsize=8.5,
             wrap=True)

    fig.tight_layout(rect=[0, 0.058, 1, 0.872])
    for p in paths:
        fig.savefig(p, dpi=200)
    plt.close(fig)


def main():
    records, samples = [], []
    for g in GENERATORS:
        with open(os.path.join(OUTDIR, f"gen_{g}.json")) as fh:
            records.append(json.load(fh))
        samples.append(dict(np.load(os.path.join(OUTDIR, f"gen_{g}.npz"))))
    drw_det = load_drw_detector()

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

    R.banner("Q railing check")
    for r in records:
        rl = r["railing"]
        print(f"  {r['generator']:<9}: Q median = {rl['Q_median']:.3f}, "
              f"frac at low bound = {rl['Q']['frac_at_low']:.3f}, "
              f"frac at high bound = {rl['Q']['frac_at_high']:.3f}, "
              f"overdamped (Q<0.5) fraction = "
              f"{rl['overdamped_fraction']:.3f}", flush=True)

    log_td, lo, hi = use_log_tdamp(records)
    print(f"\n  panel 4: t_damp spans p16 {lo:.1f} d to p84 {hi:.1f} d "
          f"(ratio {hi / lo:.1f}) -> {'log' if log_td else 'linear'} x-axis",
          flush=True)

    headline, verdict = build_headline(records)
    alias_txt = aliasing_note(records, drw_det)
    td_txt = tdamp_note(records)

    R.banner("Recovery")
    print(headline, flush=True)
    R.banner("VERDICT")
    print("  " + verdict, flush=True)
    print("\n  (i)  " + alias_txt, flush=True)
    print("\n  (ii) " + td_txt + "\n", flush=True)

    make_figure(records, samples, headline, verdict, log_td,
                [os.path.join(RESULTS, "three_generator_recovery_dho.png"),
                 os.path.join(RESULTS, "three_generator_recovery_dho.pdf")])

    with open(os.path.join(RESULTS, "three_generator_values_dho.csv"), "w",
              newline="") as fh:
        fh.write("generator,parameter,truth,truth_applies,median,p16,p84,"
                 "frac_bias_percent\n")
        for r in records:
            for key in CSV_PARAMS:
                s = r["stats"][key]
                applies = s.get("truth_applies", True)
                bias = f"{s['frac_bias_percent']:.6g}" if applies else ""
                truth = f"{s['truth']:.6g}" if applies else ""
                fh.write(f"{r['generator']},{key},{truth},{applies},"
                         f"{s['median']:.6g},{s['p16']:.6g},{s['p84']:.6g},"
                         f"{bias}\n")

    write_verdict_md(records, headline, verdict, alias_txt, td_txt, log_td,
                     bad, drw_det)
    print("  wrote three_generator_recovery_dho.png/.pdf, "
          "three_generator_values_dho.csv, three_generator_verdict_dho.md",
          flush=True)


def write_verdict_md(records, headline, verdict, alias_txt, td_txt, log_td,
                     bad, drw_det):
    L = [
        "# Chirp recovery under a fixed DHO detector, three input noise processes",
        "",
        "The same injected chirp (P = 230 d, eta = 0.25, ZTF-g harmonic-1 amp "
        "0.15, harmonic-2 amp 0.075, Figure 1 phases) is placed on three "
        "different noise processes and analysed by the same chirp + DHO "
        "(celerite2 SHOTerm) detector. Companion to the DRW-detector figure; "
        "the layouts match so the two can be read side by side.",
        "",
        "## Verdict",
        "",
        "```",
        headline,
        "```",
        "",
        f"**{verdict}**",
        "",
        f"**(i) Aliasing.** {alias_txt}",
        "",
        f"**(ii) Matched-case timescale.** {td_txt}",
        "",
        "## Recovery per generator",
        "",
        "| generator | parameter | truth | median | p16 | p84 | frac bias % |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        for key in CSV_PARAMS:
            s = r["stats"][key]
            applies = s.get("truth_applies", True)
            t = f"{s['truth']:.6g}" if applies else "n/a"
            b = f"{s['frac_bias_percent']:+.3f}" if applies else "-"
            L.append(f"| {r['generator']} | {key} | {t} | {s['median']:.6g} | "
                     f"{s['p16']:.6g} | {s['p84']:.6g} | {b} |")

    L += [
        "",
        "## Panel 4: the fitted stochastic timescale",
        "",
        "The detector always reports a t_damp, but only the matched arm has a "
        "true one. Q is quoted alongside because t_damp = 2Q/w0 alone hides "
        "whether the kernel went under- or over-damped.",
        "",
        "| generator | true timescale | fitted t_damp | 16-84% | Q median | "
        "regime | overdamped frac |",
        "|---|---|---|---|---|---|---|",
    ]
    true_ts = {
        "drw": "tau_DRW = 320 d (a DRW tau, not a t_damp)",
        "dho": "t_damp = 320 d (**genuine**, matched case)",
        "powerlaw": "none - a beta = 3 power law has no characteristic timescale",
    }
    for r in records:
        s, rl = r["stats"]["t_damp"], r["railing"]
        q = r["stats"]["Q"]["median"]
        regime = "overdamped (DRW-like)" if q < 0.5 else "underdamped"
        L.append(f"| {r['generator']} | {true_ts[r['generator']]} | "
                 f"{s['median']:.1f} d | {s['p16']:.1f} - {s['p84']:.1f} d | "
                 f"{q:.3f} | {regime} | {rl['overdamped_fraction']:.3f} |")

    L += [
        "",
        f"Panel 4 x-axis: **{'logarithmic' if log_td else 'linear'}** "
        + ("(the three posteriors span more than a decade)." if log_td
           else "(the three posteriors span less than a decade)."),
        "",
    ]

    od = [(r, overdamped_timescales(r["stats"]["Q"]["median"],
                                    r["stats"]["w0"]["median"]))
          for r in records]
    od = [(r, ts) for r, ts in od if ts is not None]
    if od:
        L += [
            "### Overdamped arms: t_damp is not the decay time",
            "",
            "Where Q < 1/2 the kernel is overdamped and has two *real* decay "
            "times, the roots of s^2 + (w0/Q)s + w0^2 giving rates "
            "(w0/2Q)(1 +/- sqrt(1 - 4Q^2)). The nominal t_damp = 2Q/w0 is "
            "neither of them - it lies between the fast and slow times - so "
            "for these arms panel 4's value understates the correlation length "
            "the kernel actually imposes. Computed from the posterior medians "
            "of Q and w0:",
            "",
            "| generator | Q median | t_damp = 2Q/w0 | fast decay | slow decay |",
            "|---|---|---|---|---|",
        ]
        for r, (t_fast, t_slow) in od:
            L.append(f"| {r['generator']} | {r['stats']['Q']['median']:.3f} | "
                     f"{r['stats']['t_damp']['median']:.1f} d | "
                     f"{t_fast:.1f} d | **{t_slow:.1f} d** |")
        L.append("")

    L += [
        "### Q railing",
        "",
        "A Q pressed against a prior bound is diagnostic: Q -> 0.05 means the "
        "kernel is trying to become a DRW (expected on drw input); Q railing "
        "high means it is forcing sharp quasi-periodicity.",
        "",
        "| generator | Q median | frac at low bound (0.05) | frac at high "
        "bound (50) | verdict |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        rl = r["railing"]
        if rl["Q"]["frac_at_low"] > 0.1:
            v = "**railing LOW - kernel becoming a DRW**"
        elif rl["Q"]["frac_at_high"] > 0.1:
            v = "**railing HIGH - forcing sharp quasi-periodicity**"
        else:
            v = "not railing"
        L.append(f"| {r['generator']} | {rl['Q_median']:.3f} | "
                 f"{rl['Q']['frac_at_low']:.3f} | {rl['Q']['frac_at_high']:.3f} "
                 f"| {v} |")

    L += [
        "",
        "## Aliasing check (P near 2 x 230 = 460 d)",
        "",
        "| generator | P median | within 5% of 460 d | frac samples > 400 d | "
        "frac within 5% of 460 d |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        a = r["aliasing"]
        L.append(f"| {r['generator']} | {a['P_median']:.2f} d | "
                 f"{'**YES**' if a['near_460'] else 'no'} | "
                 f"{a['frac_samples_above_400d']:.4f} | "
                 f"{a['frac_samples_within_5pct_of_460']:.4f} |")
    if drw_det:
        L += ["", "Same check under the earlier DRW detector, for comparison:",
              "", "| generator | P median (DRW det.) | frac within 5% of 460 d |",
              "|---|---|---|"]
        for g in GENERATORS:
            if g in drw_det:
                a = drw_det[g]["aliasing"]
                L.append(f"| {g} | {a['P_median']:.2f} d | "
                         f"{a['frac_samples_within_5pct_of_460']:.4f} |")
    L += [
        "",
        "**A single realisation per generator cannot establish a *rate* of "
        "aliasing, or of timescale scatter.** It shows only how this particular "
        "draw behaved. Comparing aliasing propensity between the two detectors "
        "would need many realisations each; that is out of scope here.",
        "",
        "## Detector",
        "",
        "- Chirp + `celerite2.terms.SHOTerm(S0, w0, Q)` + `diag(sigma_i^2)`, "
        "the same deterministic chirp model, GLS-profiled linear coefficients "
        "and conditional beta draws as the DRW detector.",
        "- Sampled in `(ln P, eta, ln sigma_kernel, ln w0, ln Q)` where "
        "`sigma_kernel = sqrt(S0*w0*Q)` is the kernel's marginal RMS, so "
        "`S0 = sigma_kernel^2/(w0*Q)`. This is an **exact reparameterisation** "
        "of (S0, w0, Q): k(0) = S0*w0*Q holds for celerite2's SHOTerm at every "
        "Q (verified numerically over Q in [0.05, 50], including the Q = 1/2 "
        "critical point). It was chosen so panel 3 is directly comparable to "
        "the injected 0.06 and to the DRW detector's sigma, and because the "
        "marginal RMS is far better conditioned than S0.",
        "- Reported timescale: `t_damp = 2Q/w0` [d], with Q and w0 recorded "
        "separately.",
        "",
        "### Priors",
        "",
        "| parameter | prior | bounds |",
        "|---|---|---|",
        f"| P | log-uniform | {SHO_PRIOR_BOUNDS['P'][0]:g} - "
        f"{SHO_PRIOR_BOUNDS['P'][1]:g} d |",
        f"| eta | uniform | {SHO_PRIOR_BOUNDS['eta'][0]:g} - "
        f"{SHO_PRIOR_BOUNDS['eta'][1]:g} |",
        f"| sigma_kernel | log-uniform | {SHO_PRIOR_BOUNDS['sigma_kernel'][0]:g}"
        f" - {SHO_PRIOR_BOUNDS['sigma_kernel'][1]:g} mag |",
        f"| w0 | log-uniform | {SHO_PRIOR_BOUNDS['w0'][0]:.3e} - "
        f"{SHO_PRIOR_BOUNDS['w0'][1]:.3e} rad/d (oscillation period 20 - "
        f"20000 d) |",
        f"| Q | log-uniform | {SHO_PRIOR_BOUNDS['Q'][0]:g} - "
        f"{SHO_PRIOR_BOUNDS['Q'][1]:g} |",
        "",
        "**Q is not floored at 0.5**: the overdamped regime is reachable, so "
        "the DHO can represent DRW-like noise.",
        "",
        "## Controlled comparison",
        "",
        "- **Independent correlated-noise seed per generator** "
        + ", ".join(f"{g} = {NOISE_SEEDS[g]}" for g in GENERATORS) +
        ". Unlike the DRW-detector figure (where all three arms shared the "
        "Figure 1 master seed and the drw arm reproduced Figure 1 exactly), no "
        "arm here reuses another's noise realisation.",
        f"- Cadence, photometric error bars and white noise remain "
        f"byte-identical across the three arms (master seed "
        f"{records[0]['master_seed']}); only the correlated-noise draw differs.",
        "- Per-band stream seeds (cadence, correlated noise, white): "
        + "; ".join(f"{r['generator']}: "
                    + ", ".join(f"{b} = {v}" for b, v in
                                r["stream_seeds"].items())
                    for r in records),
        f"- Detector, injected chirp, priors and MCMC settings identical "
        f"across the three: {D.N_WALKERS} walkers x {D.N_STEPS} steps, discard "
        f"{D.N_BURN}, thin by autocorrelation.",
        "- Single realisation per generator by design.",
        "",
        "### Realised correlated-noise sd",
        "",
        "| generator | ZTF-g | ZTF-r |",
        "|---|---|---|",
    ]
    for r in records:
        sd = r["realised_noise_sd"]
        L.append(f"| {r['generator']} | {sd['ZTF-g']:.4f} | {sd['ZTF-r']:.4f} |")

    L += [
        "",
        "drw and dho are exact process draws with k(0) = sigma^2 so their "
        "variance is matched in expectation; a beta = 3 power law has no "
        "stationary variance, so that realisation is standardised to a sample "
        "sd of sigma instead.",
        "",
        "**Reading panel 3 correctly.** The red line is the nominal 0.06, but "
        "only the powerlaw arm realises it exactly (it is standardised). A DRW "
        "with tau = 320 d over a 2000 d baseline contains few independent "
        "correlation times, so its realised sd scatters widely about the "
        "nominal value - here it came out low. The per-arm amplitude bias in "
        "panel 3 should therefore be judged against the realised sd in the "
        "table above, not against 0.06:",
        "",
        "| generator | realised sd (g) | fitted amplitude median | vs nominal "
        "0.06 | vs realised (g) |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        sd_g = r["realised_noise_sd"]["ZTF-g"]
        m = r["stats"]["amp"]["median"]
        L.append(f"| {r['generator']} | {sd_g:.4f} | {m:.4f} | "
                 f"{r['stats']['amp']['frac_bias_percent']:+.2f}% | "
                 f"{100.0 * (m - sd_g) / sd_g:+.2f}% |")
    L += [
        "",
        "So the drw arm's large negative amplitude bias is mostly the noise "
        "draw, not the detector. The powerlaw arm's positive bias is genuine "
        "misspecification: a beta = 3 spectrum piles power at low frequency, "
        "and the SHOTerm inflates its marginal RMS to mimic the long-timescale "
        "wander.",
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
          ("All three chains pass the acceptance band and the 50x "
           "autocorrelation check."
           if not bad else
           "**Mixing concerns: " + ", ".join(bad) + ".** Slow mixing is "
           "expected where the stochastic timescale is unidentified (the "
           "power-law case in particular); treat the affected posterior widths "
           "as indicative rather than converged."),
          "",
          "Autocorrelation times are for the 5 sampled coordinates in order: "
          "(ln P, eta, ln sigma_kernel, ln w0, ln Q).",
          ""]

    with open(os.path.join(RESULTS, "three_generator_verdict_dho.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
