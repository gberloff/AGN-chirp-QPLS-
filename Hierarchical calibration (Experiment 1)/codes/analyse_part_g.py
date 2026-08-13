"""PART G analysis: the selection function, its validation, and the figures.

Refits the response surface on Parts F and G combined, with 15% held out by
seed before fitting.  Writes results/selection_function.pkl, its grid CSV
and its README.  Reports parameter recovery (being bias, RMSE, 68% and 95%
coverage )and builds the one-dimensional efficiency curves with binomial
intervals.  The five figures are in item1_figs.py.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

import lib
import surface as S
import injection as inj

FAP = 1e-4
HOLDOUT = 0.15


def load_combined():
    f = pd.read_csv(os.path.join(lib.RESULTS, "part_f_per_fit.csv"),
                    float_precision="round_trip")
    f["block"] = "screen"
    g = pd.read_csv(os.path.join(lib.RESULTS, "part_g_per_fit.csv"),
                    float_precision="round_trip")
    both = pd.concat([f, g[g.block == "refine"]], ignore_index=True)
    return f, g, both


def split_by_seed(df, frac=HOLDOUT, seed=20260811):
    """Hold out by SEED, never by row, so no curve can straddle the split."""
    seeds = np.unique(df["seed"].to_numpy())
    rng = np.random.default_rng(seed)
    rng.shuffle(seeds)
    n_hold = int(round(frac * seeds.size))
    hold = set(seeds[:n_hold].tolist())
    m = df["seed"].isin(hold).to_numpy()
    return df[~m].copy(), df[m].copy()


def recovery_stats(df, col_true, col_hat, lo68, hi68, lo95, hi95, by, edges,
                   log=False):
    rows = []
    v = df[by].to_numpy()
    idx = np.digitize(v, edges) - 1
    for b in range(len(edges) - 1):
        m = idx == b
        sub = df[m]
        if len(sub) < 5:
            continue
        t = sub[col_true].to_numpy()
        h = sub[col_hat].to_numpy()
        ok = np.isfinite(t) & np.isfinite(h)
        t, h = t[ok], h[ok]
        if t.size < 5:
            continue
        if log:
            err = np.log(h / t)
            bias, rmse = float(np.mean(err)), float(np.sqrt(np.mean(err ** 2)))
        else:
            err = h - t
            bias, rmse = float(np.mean(err)), float(np.sqrt(np.mean(err ** 2)))
        s = sub[ok]
        c68 = float(np.mean((s[lo68].to_numpy() <= s[col_true].to_numpy())
                            & (s[col_true].to_numpy() <= s[hi68].to_numpy())))
        c95 = float(np.mean((s[lo95].to_numpy() <= s[col_true].to_numpy())
                            & (s[col_true].to_numpy() <= s[hi95].to_numpy())))
        width68 = float(np.median(s[hi68].to_numpy() - s[lo68].to_numpy()))
        rows.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]),
                         n=int(t.size), bias=bias, rmse=rmse,
                         coverage68=c68, coverage95=c95,
                         median_width68=width68,
                         frac_interval_one_node=float(np.mean(
                             s[hi68].to_numpy() == s[lo68].to_numpy()))))
    return rows


def one_d_curves(g):
    cur = g[g.block == "curves"]
    out = {}
    for axis in sorted(cur["varied_axis"].unique()):
        sub = cur[cur.varied_axis == axis]
        key = ("duty_cycle" if axis == "duty_cycle"
               else "band_structure" if axis == "band_structure" else axis)
        rows = []
        for v, s in sub.groupby(key):
            n = len(s)
            for name in ("trigger", "correct", "alias", "chirp"):
                k = int(s[name].sum())
                p = k / n
                se = np.sqrt(max(p * (1 - p), 1e-12) / n)
                rows.append(dict(level=float(v), outcome=name, n=n, k=k,
                                 p=p, lo=max(0.0, p - 1.96 * se),
                                 hi=min(1.0, p + 1.96 * se)))
        out[axis] = rows
    return out


def main():
    f, g, both = load_combined()
    th = json.load(open(os.path.join(lib.RESULTS, "thresholds.json"),
                        encoding="utf-8"))
    pd_ = json.load(open(os.path.join(lib.RESULTS, "part_d.json"),
                         encoding="utf-8"))
    print(f"PART G: {len(f)} screening + "
          f"{int((g.block=='refine').sum())} refinement = {len(both)} for the "
          f"surface, {int((g.block=='curves').sum())} one-dimensional fits\n")

    out = dict(n_screen=int(len(f)), n_refine=int((g.block == "refine").sum()),
               n_curves=int((g.block == "curves").sum()), fap=FAP,
               adopted_cut_mode=th["adopted_cut_mode"],
               grid_choice=pd_["grid_choice"])

    train, hold = split_by_seed(both)
    surf = S.Surface(fap=FAP, outcome="chirp",
                     cut_mode=th["adopted_cut_mode"],
                     grid_choice=pd_["grid_choice"],
                     duty_cycle=inj.DUTY_FIDUCIAL,
                     snr_convention="A1*sqrt(N)/sigma_eff, sigma_eff = "
                                    "quadrature sum of median photometric error "
                                    "and the background's in-window RMS")
    surf.fit(train, train["chirp"].to_numpy())
    p_hold = surf.predict_df(hold)
    cal = S.calibration(hold["chirp"].to_numpy(), p_hold, n_bins=10)
    out["holdout"] = dict(n_train=int(len(train)), n_holdout=int(len(hold)),
                          split="by seed, 15% held out before fitting",
                          calibration=cal)
    print(f"Held-out calibration ({len(hold)} curves, split by seed):")
    print(f"  {'bin':<12} {'n':>5} {'predicted':>10} {'observed':>10} "
          f"{'95% CI':>18}")
    for b in cal["bins"]:
        if b["n"] == 0:
            continue
        print(f"  [{b['lo']:.1f},{b['hi']:.1f})   {b['n']:>5} "
              f"{b['mean_predicted']:>10.3f} {b['observed']:>10.3f} "
              f"[{b['observed_lo']:.3f},{b['observed_hi']:.3f}]")
    print(f"  Brier score {cal['brier']:.4f}, largest deviation "
          f"{cal['max_abs_deviation']:.3f}, "
          f"{cal['n_bins_predicted_outside_ci']} of "
          f"{cal['n_bins_occupied']} occupied bins have the prediction outside "
          f"the observed interval")

    final = S.Surface(fap=FAP, outcome="chirp", cut_mode=th["adopted_cut_mode"],
                      grid_choice=pd_["grid_choice"],
                      duty_cycle=inj.DUTY_FIDUCIAL,
                      snr_convention=surf.snr_convention)
    final.fit(both, both["chirp"].to_numpy())
    rank = final.deviance_ranking(both, both["chirp"].to_numpy())
    out["final_surface"] = dict(coefficients=final.coefficients(), ranking=rank,
                                n_train=int(len(both)))
    print(f"\nFinal surface on {len(both)} injections: pseudo-R^2 "
          f"{rank['pseudo_r2']:.4f}")
    for r in rank["ranking"]:
        print(f"   {r['axis']:<20} {r['deviance_increase']:9.1f}  "
              f"{100*r['frac_of_explained']:5.1f}%")

    with open(os.path.join(lib.RESULTS, "selection_function.pkl"), "wb") as fh:
        pickle.dump(final, fh)

    top2 = [r["axis"] for r in rank["ranking"] if r["axis"] in S.AXES][:2]
    out["dominant_axes"] = top2
    grid_rows = []
    axg = {}
    for a in S.AXES:
        lo, hi = inj.BOUNDS[a]
        n = 12 if a in top2 else 3
        axg[a] = (np.exp(np.linspace(np.log(lo), np.log(hi), n))
                  if a in inj.LOG_AXES else np.linspace(lo, hi, n))
    import itertools
    combos = list(itertools.product(*[axg[a] for a in S.AXES], (0, 1)))
    dfg = pd.DataFrame(combos, columns=S.AXES + ["band_structure"])
    dfg["probability"] = final.predict_df(dfg)
    p, (plo, phi) = final.predict(**{a: dfg[a].to_numpy() for a in S.AXES},
                                  band_structure=dfg["band_structure"].to_numpy(),
                                  fap=FAP)
    dfg["prob_lo95"], dfg["prob_hi95"] = plo, phi
    dfg["fap"] = FAP
    dfg.to_csv(os.path.join(lib.RESULTS, "selection_function_grid.csv"),
               index=False)
    print(f"\nselection_function_grid.csv: {len(dfg)} rows")

    trig = both[both.trigger == 1].copy()
    snr_edges = np.array([3, 6, 10, 16, 25, 50])
    etax_edges = np.array([0, 0.5, 1, 2, 3, 5, 8])
    # A triggered injection whose period is an alias, or simply wrong,
    # contributes a large error unrelated to how well a genuine recovery
    # is measured, so the same statistics restricted to correct-period
    # recoveries are reported beside the all-triggered ones.  Both are
    # given, neither replaces the other.
    corr_only = both[both.correct == 1].copy()
    rec = dict(
        n_triggered=int(len(trig)),
        n_correct=int(len(corr_only)),
        P_by_snr_correct_only=recovery_stats(
            corr_only, "P_true_d", "P_hat", "P_lo68", "P_hi68", "P_lo95",
            "P_hi95", "snr", snr_edges, log=True),
        eta_by_eta_x_correct_only=recovery_stats(
            corr_only, "eta_true", "eta_hat", "eta_lo68", "eta_hi68",
            "eta_lo95", "eta_hi95", "eta_x", etax_edges),
        P_by_snr=recovery_stats(trig, "P_true_d", "P_hat", "P_lo68", "P_hi68",
                                "P_lo95", "P_hi95", "snr", snr_edges, log=True),
        P_by_eta_x=recovery_stats(trig, "P_true_d", "P_hat", "P_lo68", "P_hi68",
                                  "P_lo95", "P_hi95", "eta_x", etax_edges,
                                  log=True),
        eta_by_snr=recovery_stats(trig, "eta_true", "eta_hat", "eta_lo68",
                                  "eta_hi68", "eta_lo95", "eta_hi95", "snr",
                                  snr_edges),
        eta_by_eta_x=recovery_stats(trig, "eta_true", "eta_hat", "eta_lo68",
                                    "eta_hi68", "eta_lo95", "eta_hi95", "eta_x",
                                    etax_edges),
        note="P bias and RMSE are in log P (fractional), eta in absolute units. "
             "Intervals are profile-likelihood regions from the same scan the "
             "score comes from.",
    )
    out["parameter_recovery"] = rec
    print(f"\nParameter recovery, {len(trig)} triggered injections at FAP 1e-4")
    print(f"  P_hat by SNR:   {'bin':<12} {'n':>5} {'bias(logP)':>11} "
          f"{'RMSE':>8} {'cov68':>7} {'cov95':>7} {'1-node':>7}")
    for r in rec["P_by_snr"]:
        print(f"                  [{r['lo']:>4.0f},{r['hi']:>4.0f})  {r['n']:>5} "
              f"{r['bias']:>11.4f} {r['rmse']:>8.4f} {r['coverage68']:>7.3f} "
              f"{r['coverage95']:>7.3f} {r['frac_interval_one_node']:>7.3f}")
    print(f"  P_hat by SNR, correct-period recoveries only:")
    for r in rec["P_by_snr_correct_only"]:
        print(f"                  [{r['lo']:>4.0f},{r['hi']:>4.0f})  {r['n']:>5} "
              f"{r['bias']:>11.4f} {r['rmse']:>8.4f} {r['coverage68']:>7.3f} "
              f"{r['coverage95']:>7.3f} {r['frac_interval_one_node']:>7.3f}")
    print(f"  eta_hat by eta_x:")
    for r in rec["eta_by_eta_x"]:
        print(f"                  [{r['lo']:>4.1f},{r['hi']:>4.1f})  {r['n']:>5} "
              f"{r['bias']:>11.4f} {r['rmse']:>8.4f} {r['coverage68']:>7.3f} "
              f"{r['coverage95']:>7.3f} {r['frac_interval_one_node']:>7.3f}")

    out["one_d_curves"] = one_d_curves(g)
    print("\nOne-dimensional curves at FAP 1e-4 (chirp recovery):")
    for axis, rows in out["one_d_curves"].items():
        ch = [r for r in rows if r["outcome"] == "chirp"]
        s = "  ".join(f"{r['level']:g}:{r['p']:.2f}" for r in ch)
        print(f"  {axis:<20} {s}")

    with open(os.path.join(lib.RESULTS, "part_g.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)

    write_readme(out, th, pd_, final)
    print("\nwritten results/part_g.json, selection_function.pkl, "
          "selection_function_grid.csv, selection_function_README.md")
    return out


def write_readme(out, th, pd_, final):
    cal = out["holdout"]["calibration"]
    txt = f"""# selection_function.pkl: what it is and what it assumes

Anyone using this file must read this block of text.  The numbers in it are conditional
on every assumption below, and a different choice moves the contours without
any physics changing.

## Interface

    import pickle
    surf = pickle.load(open("selection_function.pkl", "rb"))
    p, (lo, hi) = surf.predict(n_cyc=8, eta_x=2, snr=15, tau_over_p=1.0,
                               a2_over_a1=0.3, samples_per_cycle=20,
                               band_structure=1, fap=1e-4)

`predict` returns the probability of **chirp recovery** and a 95% interval from
the delta method on the logit.  Passing a `fap` other than {FAP:g} raises: this
surface is calibrated at one operating point and will not pretend otherwise.
`selection_function_grid.csv` is the same surface on a regular grid.

Axes are the dimensionless quantities of Section 8.1: `n_cyc` = T/P,
`eta_x` = eta*n_cyc, `snr`, `tau_over_p`, `a2_over_a1`, `samples_per_cycle`,
and `band_structure` (0 = one band, 1 = two bands with free phase).

## Operating point

False-alarm probability **{FAP:g}**.  Threshold DeltaLambda_chirp >
**{th['fiducial_corrected']['DeltaLambda_chirp']['fap_1e_4']['threshold']:.4f}**.
No efficiency from this file means anything without that operating point
attached.

## Definition of the outcome

Chirp recovery = trigger, and |P_hat - P_true|/P_true < 0.02, and
sign(eta_hat) = sign(eta_true), and |eta_hat| < 0.58 so the estimate is not
censored at the grid edge.  Trigger, correct-period and alias recoveries are
recorded separately in the campaign tables and are **not** folded into this
number.

## SNR convention

SNR = A_1 * sqrt(N_epochs) / sigma_eff, with sigma_eff the quadrature sum of
the median per-epoch photometric error and the background's realised in-window
RMS.  Integrated signal-to-noise, not per-epoch depth.  N_epochs counts epochs
presented to the pipeline, before the quality cuts.

## Background family

**DRW throughout.**  Efficiency under quasi-periodic or red-noise backgrounds
is item 2 of the note and is not measured here.

## Fixed quantities

- Baseline T = 2000 d.
- Duty cycle fixed at **{inj.DUTY_FIDUCIAL:.4f}**, varied only in the
  one-dimensional curves of Section 9.4.  Interactions involving duty cycle
  are untested.
- Cut mode **{th['adopted_cut_mode']}** (Part C).
- Search grid: **{pd_['grid_choice']}** (Part D).

## Validation

Fitted on {out['holdout']['n_train']} injections with
{out['holdout']['n_holdout']} held out **by seed** before fitting, so no curve
appears on both sides.

- Brier score **{cal['brier']:.4f}**
- largest deviation between observed and predicted in ten probability bins:
  **{cal['max_abs_deviation']:.3f}**
- **{cal['n_bins_predicted_outside_ci']} of {cal['n_bins_occupied']}** occupied
  bins have the prediction outside the observed binomial interval

The delivered surface is refitted on all {out['final_surface']['n_train']}
injections, the validation numbers above come from the held-out fit.

## Limits

The injected waveform is the same two-harmonic family the detector fits, so
this is a **matched-template** selection function and an upper bound on real
performance.  The surface is fitted, not measured pointwise.  Efficiency at
FAP {FAP:g} rests on a tail extrapolation that was validated but not proven.
Everything is synthetic with known truth.
"""
    with open(os.path.join(lib.RESULTS, "selection_function_README.md"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
