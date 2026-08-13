"""PART C analysis: did the corrected cuts fix the bias, and by how much.

Reports, per configuration and arm: the fraction of curves on which the cut
fired, the mean fraction of epochs removed, and the 95th and 99th percentile
thresholds with bootstrap intervals.  Then the corrected fiducial threshold and
its fractional difference from the biased value at FAP 1e-4 and 5e-2, and the
adopted cut mode.
"""
import json
import os

import numpy as np
import pandas as pd

import lib
from part_c_cuts import CONFIGS, ARMS

ARM_NAME = [f"{m}/{int(w)}d" for m, w in ARMS]


def gpd_threshold(x, fap, u_pct=90.0, n_boot=400, seed=7):
    """Generalised Pareto tail extrapolation, the same construction stage 1 used:
    MLE on exceedances above the u_pct percentile, location fixed at u."""
    from scipy.stats import genpareto
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    u = float(np.percentile(x, u_pct))
    exc = x[x > u] - u
    if exc.size < 20:
        return dict(threshold=np.nan, ci95=[np.nan, np.nan], u=u,
                    n_exceedances=int(exc.size))
    zeta = exc.size / x.size

    def fit(e):
        xi, _, beta = genpareto.fit(e, floc=0.0)
        return xi, beta

    xi, beta = fit(exc)

    def thr(xi, beta):
        p = fap / zeta
        if abs(xi) < 1e-8:
            return u - beta * np.log(p)
        return u + beta / xi * (p ** (-xi) - 1.0)

    t0 = thr(xi, beta)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        e = rng.choice(exc, size=exc.size, replace=True)
        try:
            xb, bb = fit(e)
            bs.append(thr(xb, bb))
        except Exception:
            pass
    bs = np.asarray([b for b in bs if np.isfinite(b)])
    return dict(threshold=float(t0),
                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                if bs.size else [np.nan, np.nan],
                u=u, n_exceedances=int(exc.size), zeta=float(zeta),
                shape_xi=float(xi), scale_beta=float(beta))


def main():
    df = pd.read_csv(os.path.join(lib.RESULTS, "part_c_per_fit.csv"),
                     float_precision="round_trip")
    print(f"PART C: {len(df)} rows\n")
    out = dict(n_rows=int(len(df)), arms=ARM_NAME,
               configs=[c["label"] for c in CONFIGS], cells=[])

    print(f"{'configuration':<16} {'arm':<18} {'n':>4} {'fired':>7} "
          f"{'mean %epochs':>12} {'p95':>7} {'p95 CI':>16} "
          f"{'p99':>7} {'p99 CI':>16}")
    print("-" * 112)
    for ci, c in enumerate(CONFIGS):
        for ai, arm in enumerate(ARM_NAME):
            s = df[(df.config_index == ci) & (df.arm_index == ai)]
            if s.empty:
                continue
            x = s["DeltaLambda_chirp"].to_numpy()
            p95, l95, h95 = lib.boot_pct(x, 95)
            p99, l99, h99 = lib.boot_pct(x, 99)
            xo = s["Lambda_osc"].to_numpy()
            o95, ol95, oh95 = lib.boot_pct(xo, 95)
            o99, ol99, oh99 = lib.boot_pct(xo, 99)
            cell = dict(
                config_index=ci, config=c["label"], arm_index=ai, arm=arm,
                cut_mode=ARMS[ai][0], window_d=ARMS[ai][1], n=int(len(s)),
                frac_cut_fired=float(s["cut_fired"].mean()),
                mean_frac_epochs_removed=float(s["frac_removed"].mean()),
                max_frac_epochs_removed=float(s["frac_removed"].max()),
                mean_epochs_removed=float(s["n_removed_total"].mean()),
                n_insufficient=int(s["insufficient_data"].sum()),
                dchirp_p95=p95, dchirp_p95_ci=[l95, h95],
                dchirp_p99=p99, dchirp_p99_ci=[l99, h99],
                losc_p95=o95, losc_p95_ci=[ol95, oh95],
                losc_p99=o99, losc_p99_ci=[ol99, oh99],
                dchirp_median=float(np.median(x)),
            )
            out["cells"].append(cell)
            print(f"{c['label']:<16} {arm:<18} {len(s):>4} "
                  f"{100*cell['frac_cut_fired']:>6.1f}% "
                  f"{100*cell['mean_frac_epochs_removed']:>11.4f}% "
                  f"{p95:>7.3f} [{l95:>6.3f},{h95:>6.3f}] "
                  f"{p99:>7.3f} [{l99:>6.3f},{h99:>6.3f}]")

    corrected = [c for c in out["cells"]
                 if c["cut_mode"] in ("mad_scaled", "sigma_clip")
                 and c["window_d"] == 50.0]
    ms = [c for c in corrected if c["cut_mode"] == "mad_scaled"]
    fires_ok = all(c["frac_cut_fired"] < 0.05 for c in ms)
    removes_ok = all(c["mean_frac_epochs_removed"] < 0.0005 for c in ms)

    fid_raw = [c for c in out["cells"]
               if c["config_index"] == 0 and c["arm_index"] == 0][0]
    fid_ms = [c for c in out["cells"]
              if c["config_index"] == 0 and c["arm_index"] == 1][0]
    fid_sc = [c for c in out["cells"]
              if c["config_index"] == 0 and c["arm_index"] == 2][0]
    fid_w150 = [c for c in out["cells"]
                if c["config_index"] == 0 and c["arm_index"] == 3][0]

    shifts = []
    for ci in range(len(CONFIGS)):
        r = [c for c in out["cells"] if c["config_index"] == ci
             and c["arm_index"] == 0][0]
        s = [c for c in out["cells"] if c["config_index"] == ci
             and c["arm_index"] == 1][0]
        shifts.append(dict(config=r["config"],
                           p95_shift_frac=(s["dchirp_p95"] - r["dchirp_p95"])
                           / r["dchirp_p95"],
                           p99_shift_frac=(s["dchirp_p99"] - r["dchirp_p99"])
                           / r["dchirp_p99"]))
    out["shift_mad_scaled_vs_mad_raw"] = shifts
    big_shift = max(abs(s["p95_shift_frac"]) for s in shifts) > 0.10

    if fires_ok and removes_ok:
        reading = "a"
        adopted = "mad_scaled"
        text = ("Reading (a): the corrected cuts fire on under 5% of clean "
                "synthetic curves and remove under 0.05% of epochs. "
                "mad_scaled is adopted downstream.")
    elif not fires_ok:
        reading = "b"
        adopted = "sigma_clip"
        text = ("Reading (b): the corrected cuts still fire broadly, so the "
                "running median rather than the scaling is the problem. "
                "sigma_clip is adopted downstream.")
    else:
        reading = "a"
        adopted = "mad_scaled"
        text = ("Reading (a) on the firing criterion, see the table for the "
                "epoch-removal figure.")
    out["reading"] = reading
    out["reading_text"] = text
    out["adopted_cut_mode"] = adopted
    out["adopted_window_d"] = 50.0
    out["reading_c_thresholds_moved_more_than_10pct"] = bool(big_shift)

    ai_ad = 1 if adopted == "mad_scaled" else 2
    fid_ad = fid_ms if adopted == "mad_scaled" else fid_sc
    # The correction factor is computed in exactly ONE place (thresholds.py)
    # so this report and the thresholds Parts F and G apply cannot disagree.
    # It is paired: both arms scored the same 300 curves, and the bootstrap
    # resamples curves, so realisation noise largely cancels.
    import thresholds as TH
    raw_s = df[(df.config_index == 0) & (df.arm_index == 0)].sort_values(
        "realisation")
    ad_s = df[(df.config_index == 0) & (df.arm_index == ai_ad)].sort_values(
        "realisation")
    assert (raw_s["realisation"].to_numpy()
            == ad_s["realisation"].to_numpy()).all(), "arms are not paired"
    xr = raw_s["DeltaLambda_chirp"].to_numpy()
    xa = ad_s["DeltaLambda_chirp"].to_numpy()

    stage1_1e4 = 11.658837680317514
    stage1_5e2 = 5.7766658
    c_1e4 = TH.paired_correction(xr, xa, 1e-4)
    c_5e2 = TH.paired_correction(xr, xa, 5e-2)

    out["corrected_thresholds"] = dict(
        adopted_mode=adopted,
        fap_1e_4=dict(
            correction=c_1e4, stage1_biased_value=stage1_1e4,
            corrected_stage1_value=float(stage1_1e4 * c_1e4["factor"]),
            corrected_ci95=[float(stage1_1e4 * v) for v in c_1e4["ci95"]],
            fractional_difference=float(c_1e4["factor"] - 1.0),
            note="Part C has 300 nulls per cell, so its own 1e-4 point is a "
                 "tail extrapolation. The number carried downstream is stage "
                 "1's 5000-null 1e-4 threshold rescaled by the paired factor "
                 "measured here."),
        fap_5e_2=dict(
            correction=c_5e2, stage1_biased_value=stage1_5e2,
            corrected_stage1_value=float(stage1_5e2 * c_5e2["factor"]),
            corrected_ci95=[float(stage1_5e2 * v) for v in c_5e2["ci95"]],
            fractional_difference=float(c_5e2["factor"] - 1.0)),
    )

    out["window_effect"] = dict(
        description="mad_scaled at a 150 d running-median window against the "
                    "same mode at 50 d, reported separately",
        cells=[dict(config=CONFIGS[ci]["label"],
                    p95_50d=[c for c in out["cells"] if c["config_index"] == ci
                             and c["arm_index"] == 1][0]["dchirp_p95"],
                    p95_150d=[c for c in out["cells"] if c["config_index"] == ci
                              and c["arm_index"] == 3][0]["dchirp_p95"],
                    fired_50d=[c for c in out["cells"] if c["config_index"] == ci
                               and c["arm_index"] == 1][0]["frac_cut_fired"],
                    fired_150d=[c for c in out["cells"] if c["config_index"] == ci
                                and c["arm_index"] == 3][0]["frac_cut_fired"])
               for ci in range(len(CONFIGS))])
    for w in out["window_effect"]["cells"]:
        w["p95_shift_frac"] = (w["p95_150d"] - w["p95_50d"]) / w["p95_50d"]

    with open(os.path.join(lib.RESULTS, "part_c.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)

    print("\nPre-registered reading:", text)
    print(f"\nThreshold shift, mad_scaled vs mad_raw (DeltaLambda_chirp p95):")
    for s in shifts:
        print(f"  {s['config']:<16} p95 {100*s['p95_shift_frac']:+7.2f}%   "
              f"p99 {100*s['p99_shift_frac']:+7.2f}%")
    print(f"  reading (c) trigger, |shift| > 10%: {big_shift}")
    print(f"\nCorrected fiducial thresholds ({adopted}), paired against mad_raw:")
    for k, lab in (("fap_5e_2", "FAP 5e-2"), ("fap_1e_4", "FAP 1e-4")):
        e = out["corrected_thresholds"][k]
        c = e["correction"]
        print(f"  {lab}: correction factor {c['factor']:.4f} "
              f"[{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] "
              f"({100*e['fractional_difference']:+.2f}%), consistent with no "
              f"change: {c['consistent_with_unity']}")
        print(f"            stage-1 threshold {e['stage1_biased_value']:.4f} -> "
              f"**{e['corrected_stage1_value']:.4f}** "
              f"[{e['corrected_ci95'][0]:.4f}, {e['corrected_ci95'][1]:.4f}]")
    print("\nWindow effect (mad_scaled, 150 d vs 50 d):")
    for w in out["window_effect"]["cells"]:
        print(f"  {w['config']:<16} p95 {w['p95_50d']:.3f} -> {w['p95_150d']:.3f} "
              f"({100*w['p95_shift_frac']:+.2f}%)   fired "
              f"{100*w['fired_50d']:.1f}% -> {100*w['fired_150d']:.1f}%")
    print("\nwritten results/part_c.json")
    return out


if __name__ == "__main__":
    main()
