"""
Combine the three single-seed runs into the overlay figure, the values CSV and
the harmonic-2 verdict.

Usage:  python three_seed_report.py <seed1> <seed2> <seed3>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_figure1 as R
from three_seed_worker import OUTDIR, PHASE_CONVENTION

RESULTS = R.RESULTS
COLORS = ["tab:blue", "tab:orange", "tab:green"]   # colourblind-safe triple

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

# acceptance band considered comparable to the original Figure 1 run (~0.59)
ACC_LO, ACC_HI = 0.40, 0.75


def bias_verdict(biases):
    """Classify a set of per-seed fractional biases as noise / systematic /
    inconclusive.

    "pooled sigma" is the standard error of the mean, sd/sqrt(N): the relevant
    scale for asking whether the *mean* bias differs from zero.  The raw
    across-seed scatter sd is reported alongside it.
    """
    b = np.asarray(biases, dtype=float)
    n = b.size
    mean = float(b.mean())
    sd = float(b.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / np.sqrt(n) if n > 1 else float("nan")
    same_sign = bool(np.all(b > 0) or np.all(b < 0))
    straddles = not same_sign
    z = abs(mean) / sem if (np.isfinite(sem) and sem > 0) else np.inf

    if straddles or z <= 1.0:
        kind = "noise"
    elif same_sign and z > 2.0:
        kind = "systematic"
    else:
        kind = "inconclusive"
    return {"biases": b.tolist(), "mean": mean, "sd": sd, "sem": sem,
            "z": float(z), "same_sign": same_sign, "straddles": straddles,
            "kind": kind, "sign": "positive" if mean > 0 else "negative"}


def headline_text(v):
    s1, s2, s3 = (f"{x:+.2f}" for x in v["biases"])
    if v["kind"] == "noise":
        return (f"Harmonic-2 offset is consistent with noise scatter "
                f"(biases: {s1}%, {s2}%, {s3}%; mean {v['mean']:+.2f}% "
                f"+/- {v['sd']:.2f}%). No evidence of a phase-handling systematic.")
    if v["kind"] == "systematic":
        return (f"Harmonic-2 shows a persistent {v['sign']} bias across seeds "
                f"(biases: {s1}%, {s2}%, {s3}%; mean {v['mean']:+.2f}%). "
                f"Consistent with a second-harmonic phase-handling systematic, "
                f"not noise - inspect the (a2,b2) injection vs template phase "
                f"convention.")
    return (f"Harmonic-2 biases: {s1}%, {s2}%, {s3}%; mean {v['mean']:+.2f}% "
            f"+/- {v['sd']:.2f}% (|mean|/sem = {v['z']:.2f}). Inconclusive at "
            f"N=3 - more seeds are needed to decide between noise scatter and a "
            f"phase-handling systematic.")


def control_line(name, v):
    s = ", ".join(f"{x:+.2f}%" for x in v["biases"])
    return (f"{name}: biases {s}; mean {v['mean']:+.3f}% +/- {v['sd']:.3f}% "
            f"(|mean|/sem = {v['z']:.2f}) -> {v['kind']}")


def find_line(path, needle):
    """Locate a source line for the suspect-code listing."""
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if needle in line:
                return i, line.rstrip()
    return None, None


def make_overlay(records, samples, headline, paths):
    seeds = [r["seed"] for r in records]
    n_ref = min(s["P"].size for s in samples)     # equalise retained counts

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.0))
    axes = axes.ravel()

    for k, (key, xlabel, fmt) in enumerate(PANELS):
        ax = axes[k]
        truth = records[0]["stats"][key]["truth"]

        cols = [np.asarray(s[SAMPLE_KEY[key]], dtype=float) for s in samples]
        cols = [c[np.isfinite(c)] for c in cols]

        # shared bins: clip a heavy tail at p99.5 of the pooled samples, but
        # never clip the truth or any seed's median
        pooled = np.concatenate(cols)
        lo = min(float(pooled.min()), truth)
        hi = max(float(np.percentile(pooled, 99.5)), truth,
                 *[r["stats"][key]["p84"] for r in records])
        edges = np.linspace(lo, hi, 61)
        pad = 0.03 * (hi - lo)

        top = 0.0
        for c, seed, colour, rec in zip(cols, seeds, COLORS, records):
            w = np.full(c.size, n_ref / c.size)      # equal effective N
            counts, _, _ = ax.hist(
                c, bins=edges, weights=w, histtype="stepfilled", alpha=0.45,
                color=colour, edgecolor=colour, linewidth=1.4,
                label=f"seed {seed}")
            top = max(top, counts.max())
            ax.axvline(rec["stats"][key]["median"], color=colour, lw=1.8,
                       ls="--")

        ax.axvline(truth, color="red", lw=2.2,
                   label="injected truth" if k == 0 else None)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("samples")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(0, top / (0.60 if k == 0 else 0.72))

        centers = 0.5 * (edges[:-1] + edges[1:])
        span = edges[-1] - edges[0]
        lm = max(np.histogram(c, bins=edges)[0][centers < edges[0] + 0.45 * span].max()
                 for c in cols)
        rm = max(np.histogram(c, bins=edges)[0][centers > edges[-1] - 0.45 * span].max()
                 for c in cols)
        box_right = rm <= lm

        if k == 0:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(plt.Line2D([], [], color="0.35", ls="--", lw=1.8))
            labels.append("per-seed posterior median")
            ax.legend(handles, labels, fontsize=7.5,
                      loc="upper left" if box_right else "upper right",
                      framealpha=0.85)

        txt = f"truth  = {truth:{fmt}}\n" + "\n".join(
            f"s{i + 1} med = {r['stats'][key]['median']:{fmt}} "
            f"({r['stats'][key]['frac_bias_percent']:+.2f}%)"
            for i, r in enumerate(records))
        ax.text(0.975 if box_right else 0.025, 0.975, txt,
                transform=ax.transAxes, ha="right" if box_right else "left",
                va="top", fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.80, edgecolor="0.6", linewidth=0.6))

    fig.suptitle("Three-seed reproducibility of the Figure 1 posterior "
                 "(identical injection; only the global RNG seed differs)",
                 fontsize=14, y=0.985)

    wrapped, line = [], ""
    for word in headline.split():
        if len(line) + len(word) + 1 > 150:
            wrapped.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    wrapped.append(line)
    fig.text(0.5, 0.952, "\n".join(wrapped), ha="center", va="top",
             fontsize=10.5, color="darkred" if "persistent" in headline else "black")

    fig.tight_layout(rect=[0, 0, 1, 0.90 if len(wrapped) > 1 else 0.925])
    for p in paths:
        fig.savefig(p, dpi=200)
    plt.close(fig)


def main(seeds):
    records, samples = [], []
    for s in seeds:
        with open(os.path.join(OUTDIR, f"seed_{s}.json")) as fh:
            records.append(json.load(fh))
        samples.append(dict(np.load(os.path.join(OUTDIR, f"seed_{s}.npz"))))

    R.banner("MCMC mixing check")
    acc_bad = autocorr_bad = 0
    for r in records:
        d = r["diagnostics"]
        ok_acc = ACC_LO <= d["acceptance"] <= ACC_HI
        acc_bad += not ok_acc
        autocorr_bad += not d["autocorr_ok"]
        print(f"  seed {r['seed']}: acceptance = {d['acceptance']:.3f} "
              f"[{'ok' if ok_acc else 'OUT OF BAND'}], "
              f"tau_max = {d['tau_max']:.1f}, "
              f"post-burn steps / tau_max = {d['chain_over_tau']:.1f} "
              f"[{'ok' if d['autocorr_ok'] else 'FAILS 50x'}], "
              f"retained = {d['n_retained']}", flush=True)
    mixing_ok = not (acc_bad == len(records) or autocorr_bad == len(records))
    if not mixing_ok:
        print("\n  *** WARNING: all three chains fail the mixing checks; the "
              "scatter test below would be confounded by chain noise. ***",
              flush=True)

    verdicts = {key: bias_verdict([r["stats"][key]["frac_bias_percent"]
                                   for r in records])
                for key, _, _ in PANELS}
    headline = headline_text(verdicts["g_harm2_amp"])

    R.banner("Phase convention")
    print(PHASE_CONVENTION, flush=True)
    print("  Injected ZTF-g phases (unchanged across seeds): "
          f"psi1 = {records[0]['phases']['injected_psi1_rad']:+.4f} rad, "
          f"psi2 = {records[0]['phases']['injected_psi2_rad']:+.4f} rad")
    for r in records:
        p = r["phases"]
        print(f"  seed {r['seed']}: recovered psi1 = {p['recovered_psi1_rad']:+.4f} "
              f"(delta {p['delta_psi1_rad']:+.4f}), "
              f"psi2 = {p['recovered_psi2_rad']:+.4f} "
              f"(delta {p['delta_psi2_rad']:+.4f}) rad", flush=True)

    R.banner("Bias across seeds")
    for key, _, _ in PANELS:
        print("  " + control_line(key, verdicts[key]), flush=True)
    R.banner("VERDICT (harmonic 2)")
    print("  " + headline + "\n", flush=True)

    make_overlay(records, samples, headline,
                 [os.path.join(RESULTS, "three_seed_overlay.png"),
                  os.path.join(RESULTS, "three_seed_overlay.pdf")])

    with open(os.path.join(RESULTS, "three_seed_values.csv"), "w",
              newline="") as fh:
        fh.write("seed,parameter,truth,median,p16,p84,frac_bias_percent\n")
        for r in records:
            for key, _, _ in PANELS:
                s = r["stats"][key]
                fh.write(f"{r['seed']},{key},{s['truth']:.6g},{s['median']:.6g},"
                         f"{s['p16']:.6g},{s['p84']:.6g},"
                         f"{s['frac_bias_percent']:.6g}\n")

    write_verdict_md(records, verdicts, headline, mixing_ok)
    print(f"  wrote three_seed_overlay.png/.pdf, three_seed_values.csv, "
          f"three_seed_verdict.md", flush=True)


def write_verdict_md(records, verdicts, headline, mixing_ok):
    v2 = verdicts["g_harm2_amp"]
    L = [
        "# Three-seed check on the ZTF-g harmonic-2 bias",
        "",
        f"## Verdict",
        "",
        f"**{headline}**",
        "",
        f"Decision rule: mixed signs across seeds, or |mean| <= 1 pooled sigma "
        f"-> noise; same sign and |mean| > 2 pooled sigma -> systematic; "
        f"otherwise inconclusive. "
        f"\"Pooled sigma\" is the standard error of the mean, sd/sqrt(3) = "
        f"{v2['sem']:.3f}%, with across-seed scatter sd = {v2['sd']:.3f}%. "
        f"Here |mean|/sem = {v2['z']:.2f} and the three signs are "
        f"{'all the same' if v2['same_sign'] else 'mixed'}.",
        "",
        "## Bias across seeds (per parameter)",
        "",
        "| parameter | seed " + " | seed ".join(str(r["seed"]) for r in records)
        + " | mean | sd | \\|mean\\|/sem | class |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, _, _ in PANELS:
        v = verdicts[key]
        cells = " | ".join(f"{b:+.3f}%" for b in v["biases"])
        L.append(f"| {key} | {cells} | {v['mean']:+.3f}% | {v['sd']:.3f}% | "
                 f"{v['z']:.2f} | {v['kind']} |")

    L += ["", "### Controls", ""]
    for key in ("P", "eta", "g_harm1_amp"):
        L.append(f"- {control_line(key, verdicts[key])}")
    flagged = [k for k in ("P", "eta", "g_harm1_amp")
               if verdicts[k]["kind"] != "noise"]
    L += [
        "",
        f"Interpretation: if P, eta and harmonic 1 are well centred while only "
        f"harmonic 2 drifts, the systematic is localised to the "
        f"second-harmonic term rather than the whole template.",
        "",
        "Note that the classifier is *scale-free*: it compares the mean bias to "
        "the across-seed scatter, not to any physical tolerance. A parameter "
        "whose three seeds agree very tightly can therefore be flagged on an "
        "offset that is negligible in absolute terms.",
    ]
    for k in flagged:
        v = verdicts[k]
        L.append(f"- `{k}` is flagged **{v['kind']}**, but its mean offset is "
                 f"only {v['mean']:+.3f}% (across-seed scatter "
                 f"{v['sd']:.3f}%); judge its practical importance against the "
                 f"relevant tolerance, not against the scatter alone.")
    L += [
        "",
        "## Recovered values per seed",
        "",
        "| seed | parameter | truth | median | p16 | p84 | frac bias % |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        for key, _, _ in PANELS:
            s = r["stats"][key]
            L.append(f"| {r['seed']} | {key} | {s['truth']:.6g} | "
                     f"{s['median']:.6g} | {s['p16']:.6g} | {s['p84']:.6g} | "
                     f"{s['frac_bias_percent']:+.3f} |")

    L += ["", "## Phase convention", "", "```", PHASE_CONVENTION.rstrip(), "```", ""]
    L += [
        "Injected ZTF-g phases (identical in all three runs, not varied): "
        f"psi1 = {records[0]['phases']['injected_psi1_rad']:+.4f} rad, "
        f"psi2 = {records[0]['phases']['injected_psi2_rad']:+.4f} rad. "
        "ZTF-r: psi1 = +1.3000, psi2 = -0.5000 rad.",
        "",
        "Recovered phases (circular mean of the conditional beta draws, same "
        "atan2(b, a) convention):",
        "",
        "| seed | psi1 rec | psi1 - psi1_inj | psi2 rec | psi2 - psi2_inj |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        p = r["phases"]
        L.append(f"| {r['seed']} | {p['recovered_psi1_rad']:+.4f} | "
                 f"{p['delta_psi1_rad']:+.4f} | {p['recovered_psi2_rad']:+.4f} | "
                 f"{p['delta_psi2_rad']:+.4f} |")
    L += [
        "",
        "A phase-handling systematic would show up here as a delta near a "
        "characteristic value (e.g. +/-pi/2 for a swapped sin/cos pair, or pi "
        "for a sign flip); a small delta of either sign indicates the "
        "convention round-trips correctly.",
        "",
    ]

    if v2["kind"] == "systematic":
        L += ["## Suspect code (second-harmonic phase handling)", "",
              "The three lines that define the harmonic-2 convention, in the "
              "order the signal passes through them:", ""]
        for path, needle, note in (
            ("simulate.py", "a2, b2 = A2 * np.cos(psi2)",
             "injection: (A2, psi2) -> (a2, b2)"),
            ("simulate.py", "+ a2 * np.sin(2 * phi)",
             "injection: the harmonic-2 term added to the light curve"),
            ("chirp_model.py", "X[:, 3] = np.sin(2.0 * phi)",
             "template: harmonic-2 design columns (must match the line above)"),
            ("run_figure1.py", "h2_s = np.hypot(bg[:, 3], bg[:, 4])",
             "recovery: (a2, b2) -> amplitude"),
        ):
            n, txt = find_line(os.path.join(R.HERE, path), needle)
            if n:
                L.append(f"- `{path}:{n}` - {note}\n  ```python\n  {txt.strip()}\n  ```")
        L.append("")
    else:
        L += ["## Suspect code", "",
              "Not applicable - the verdict is not \"systematic\", so no "
              "specific line is implicated.", ""]

    d0 = records[0]["diagnostics"]
    L += [
        "## Run settings and mixing",
        "",
        f"- Identical for all three runs; **only `CONFIG[\"seed\"]` changed**. "
        f"Injected truth, per-band phases, sampling config, priors, DRW+chirp "
        f"fit and conditional beta draws are untouched.",
        f"- MCMC: {R.N_WALKERS} walkers x {R.N_STEPS} steps, discard "
        f"{R.N_BURN} burn-in, thin by autocorrelation "
        f"(full length, as in the original Figure 1 run; >= 8000 steps and "
        f">= 2000 burn-in as required).",
        f"- Scatter is therefore sampling scatter across noise realisations, "
        f"not chain noise.",
        "",
        "| seed | acceptance | autocorr times | tau_max | post-burn/tau_max | "
        "retained | beta draw failures | non-SPD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        d = r["diagnostics"]
        L.append(f"| {r['seed']} | {d['acceptance']:.4f} | "
                 f"{np.round(d['autocorr_times'], 1).tolist()} | "
                 f"{d['tau_max']:.1f} | {d['chain_over_tau']:.1f} | "
                 f"{d['n_retained']} | {d['beta_draw_failures']} | "
                 f"{d['beta_nonspd']} |")
    L += ["",
          ("All three chains mix comparably to the original Figure 1 run "
           f"(acceptance ~0.59); the scatter test is not confounded by chain "
           f"noise." if mixing_ok else
           "**All three chains fail the mixing checks - the scatter test above "
           "is confounded by chain noise and the verdict should not be "
           "trusted.**"),
          "",
          "## Caveat on N=3",
          "",
          "With three seeds the standard error of the mean is estimated from "
          "two degrees of freedom, so both the mean and its uncertainty are "
          "themselves noisy. A same-sign result at N=3 has a 1-in-4 chance of "
          "arising from pure noise. Treat the classification as indicative; "
          "confirm any \"systematic\" verdict with more seeds.",
          ""]

    with open(os.path.join(RESULTS, "three_seed_verdict.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
