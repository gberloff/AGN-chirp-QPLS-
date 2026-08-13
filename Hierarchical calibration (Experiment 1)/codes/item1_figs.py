"""PART G figures.

Colour: a validated default palette, used unchanged in its fixed slot order.
Keeping the reference palette's own slots in their documented order keeps
the figures inside a configuration already checked for contrast and
colour-blind safety, rather than using ad hoc colours that have not been
checked.  The palette validator itself is not run here.

Every categorical series also carries a second encoding (line style plus a
direct label), so identity never rests on hue alone.

Every title states background, null, cut mode and operating point.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

import lib
import surface as S
import injection as inj

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
       "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SURFACE = "#fcfcfb"

NICE = {
    "n_cyc": r"$N_{\rm cyc} = T/P$",
    "eta_x": r"$\eta_x = \eta N_{\rm cyc}$",
    "snr": "SNR",
    "tau_over_p": r"$\tau/P$",
    "a2_over_a1": r"$A_2/A_1$",
    "samples_per_cycle": "samples per cycle",
    "band_structure": "band structure (0 = 1 band, 1 = 2 bands)",
    "duty_cycle": "duty cycle",
}

OUTCOME_STYLE = {
    "trigger": (C1, "-", "trigger"),
    "correct": (C2, "--", "correct period"),
    "alias":   (C3, "-.", "alias"),
    "chirp":   (C4, ":", "chirp"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 9.5,
    "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0,
})


def _banner(th, pd_, fap=1e-4):
    return (f"DRW background and DRW null · cut mode {th['adopted_cut_mode']} · "
            f"{pd_['grid_choice']} · FAP {fap:g}")


def load():
    f = pd.read_csv(os.path.join(lib.RESULTS, "part_f_per_fit.csv"),
                    float_precision="round_trip")
    f["block"] = "screen"
    g = pd.read_csv(os.path.join(lib.RESULTS, "part_g_per_fit.csv"),
                    float_precision="round_trip")
    both = pd.concat([f, g[g.block == "refine"]], ignore_index=True)
    th = json.load(open(os.path.join(lib.RESULTS, "thresholds.json"),
                        encoding="utf-8"))
    pg = json.load(open(os.path.join(lib.RESULTS, "part_g.json"),
                        encoding="utf-8"))
    pd_ = json.load(open(os.path.join(lib.RESULTS, "part_d.json"),
                         encoding="utf-8"))
    import pickle
    surf = pickle.load(open(os.path.join(lib.RESULTS,
                                         "selection_function.pkl"), "rb"))
    return f, g, both, th, pg, pd_, surf


def fig_surface(both, th, pg, pd_, surf):
    a1, a2 = pg["dominant_axes"][:2]
    lo1, hi1 = inj.BOUNDS[a1]
    lo2, hi2 = inj.BOUNDS[a2]
    n = 120
    g1 = (np.exp(np.linspace(np.log(lo1), np.log(hi1), n)) if a1 in inj.LOG_AXES
          else np.linspace(lo1, hi1, n))
    g2 = (np.exp(np.linspace(np.log(lo2), np.log(hi2), n)) if a2 in inj.LOG_AXES
          else np.linspace(lo2, hi2, n))
    G1, G2 = np.meshgrid(g1, g2)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, bs, name in zip(axes, (0, 1), ("1 band", "2 bands, free phase")):
        d = pd.DataFrame({a: np.full(G1.size, inj.FIDUCIAL_POINT[a])
                          for a in S.AXES})
        d[a1] = G1.ravel()
        d[a2] = G2.ravel()
        d["band_structure"] = bs
        P = surf.predict_df(d).reshape(G1.shape)
        im = ax.contourf(G1, G2, P, levels=np.linspace(0, 1, 11),
                         colors=SEQ[:10], extend="neither")
        cs = ax.contour(G1, G2, P, levels=[0.5, 0.9], colors=[INK, INK],
                        linewidths=[2.0, 1.2], linestyles=["-", "--"])
        ax.clabel(cs, fmt={0.5: "50%", 0.9: "90%"}, fontsize=8)
        if a1 in inj.LOG_AXES:
            ax.set_xscale("log")
            ticks = [t for t in (3, 5, 10, 20, 50, 100) if lo1 <= t <= hi1]
            ax.set_xticks(ticks)
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if a2 in inj.LOG_AXES:
            ax.set_yscale("log")
        ax.set_xlabel(NICE.get(a1, a1))
        ax.set_title(name)
        ax.grid(alpha=0.35)
    axes[0].set_ylabel(NICE.get(a2, a2))
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("P(chirp recovery)")
    fig.suptitle(f"Chirp-recovery completeness, {a1} against {a2}\n"
                 + _banner(th, pd_) + ", other axes at the fiducial point",
                 fontsize=9.5, y=1.06)
    fig.savefig(os.path.join(lib.FIGS, "G_selection_surface.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


def fig_eta_x(both, th, pd_):
    pf = json.load(open(os.path.join(lib.RESULTS, "part_f.json"),
                        encoding="utf-8"))
    prof = pf["eta_x_profile"]
    x = [r["mean_eta_x"] for r in prof]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, col, ls, lab in (("chirp", C4, ":", "chirp recovery"),
                              ("chirp_given_correct", C1, "-",
                               "chirp recovery given the period was recovered")):
        y = [r[key] for r in prof]
        ax.plot(x, y, color=col, ls=ls, marker="o", ms=5, label=lab)
    y = [r["chirp"] for r in prof]
    lo = [r["chirp_lo"] for r in prof]
    hi = [r["chirp_hi"] for r in prof]
    ax.fill_between(x, lo, hi, color=C4, alpha=0.18, lw=0)
    ax.axvline(1.0, color=INK2, lw=1.2, ls="--")
    ax.text(1.05, 0.94, r"$\eta_x = 1$: drift spans one" "\n"
                        "frequency resolution element",
            color=INK2, fontsize=8, va="top")
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xlabel(r"$\eta_x = \eta \times N_{\rm cyc}$")
    ax.set_ylabel("recovery fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.set_title("The chirp-detectability transition in $\\eta_x$\n"
                 + _banner(th, pd_), fontsize=9.5)
    fig.savefig(os.path.join(lib.FIGS, "G_eta_x_transition.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


def fig_1d(pg, th, pd_):
    curves = pg["one_d_curves"]
    keys = [k for k in curves if curves[k]]
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.0, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, keys):
        rows = curves[k]
        for name, (col, ls, lab) in OUTCOME_STYLE.items():
            r = sorted([q for q in rows if q["outcome"] == name],
                       key=lambda q: q["level"])
            if not r:
                continue
            x = [q["level"] for q in r]
            y = [q["p"] for q in r]
            ax.plot(x, y, color=col, ls=ls, marker="o", ms=4, label=lab)
            ax.fill_between(x, [q["lo"] for q in r], [q["hi"] for q in r],
                            color=col, alpha=0.13, lw=0)
        if k in inj.LOG_AXES:
            ax.set_xscale("log")
        ax.set_title(NICE.get(k, k), fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.35)
    for ax in axes[len(keys):]:
        ax.axis("off")
    axes[0].set_ylabel("recovery fraction")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("One-dimensional efficiency curves about the fiducial point, "
                 "with binomial 95% intervals\n" + _banner(th, pd_),
                 fontsize=9.5, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(os.path.join(lib.FIGS, "G_1d_curves.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


def fig_breakdown(both, th, pd_):
    edges = np.array([3, 5, 7, 10, 14, 20, 30, 50])
    idx = np.digitize(both["snr"].to_numpy(), edges) - 1
    xs, series = [], {k: ([], [], []) for k in OUTCOME_STYLE}
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n < 10:
            continue
        sub = both[m]
        xs.append(float(sub["snr"].mean()))
        for k in OUTCOME_STYLE:
            p = float(sub[k].mean())
            se = np.sqrt(max(p * (1 - p), 1e-12) / n)
            series[k][0].append(p)
            series[k][1].append(max(0, p - 1.96 * se))
            series[k][2].append(min(1, p + 1.96 * se))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for k, (col, ls, lab) in OUTCOME_STYLE.items():
        y, lo, hi = series[k]
        ax.plot(xs, y, color=col, ls=ls, marker="o", ms=5, label=lab)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
    # direct labels, staggered so that series ending at the same height (chirp
    # and correct-period nearly coincide) do not overprint each other
    ends = sorted(((series[k][0][-1], k) for k in OUTCOME_STYLE), reverse=True)
    placed = []
    for yv, k in ends:
        yy = yv
        while any(abs(yy - p) < 0.05 for p in placed):
            yy -= 0.05
        placed.append(yy)
        ax.annotate(OUTCOME_STYLE[k][2], xy=(xs[-1], yy), xytext=(8, 0),
                    textcoords="offset points", color=OUTCOME_STYLE[k][0],
                    fontsize=8.5, va="center", annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("SNR  (integrated: $A_1\\sqrt{N}/\\sigma_{\\rm eff}$)")
    ax.set_ylabel("fraction of injections")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(right=max(xs) * 1.6)
    ax.legend(loc="upper left")
    ax.set_title("How much of 'detection' is aliased: all four recovery "
                 "definitions against SNR\n" + _banner(th, pd_), fontsize=9.5)
    fig.savefig(os.path.join(lib.FIGS, "G_recovery_breakdown.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


def fig_calibration(pg, th, pd_):
    cal = pg["holdout"]["calibration"]
    rows = [b for b in cal["bins"] if b["n"] > 0]
    x = [b["mean_predicted"] for b in rows]
    y = [b["observed"] for b in rows]
    lo = [b["observed"] - b["observed_lo"] for b in rows]
    hi = [b["observed_hi"] - b["observed"] for b in rows]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5.6, 5.8), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1.2, ls="--", label="perfect")
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=C1, ms=6, lw=1.5,
                capsize=3, label="held-out observed")
    ax.set_ylabel("observed chirp-recovery fraction")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper left")
    ax.set_title(f"Held-out calibration, {pg['holdout']['n_holdout']} curves "
                 f"split by seed\nBrier {cal['brier']:.4f} · "
                 + _banner(th, pd_), fontsize=9.5)
    ax2.bar(x, [b["n"] for b in rows], width=0.07, color=SEQ[6])
    ax2.set_ylabel("n")
    ax2.set_xlabel("predicted probability")
    ax2.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(lib.FIGS, "G_calibration.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


def main():
    f, g, both, th, pg, pd_, surf = load()
    fig_surface(both, th, pg, pd_, surf)
    fig_eta_x(both, th, pd_)
    fig_1d(pg, th, pd_)
    fig_breakdown(both, th, pd_)
    fig_calibration(pg, th, pd_)
    print("written 5 figures to figs/")


if __name__ == "__main__":
    main()
