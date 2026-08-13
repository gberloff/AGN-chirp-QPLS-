"""PART G (batch): boundary refinement and the one-dimensional curves.

Two blocks, both written to results/part_g_per_fit.csv through Part F's worker
so the code path is identical:

  refine   3000 injections concentrated where the Part F surface predicts
           chirp-recovery probability between 0.2 and 0.8, varying the axes
           Part F ranked significant and holding the rest at fiducial values.
  curves   one-dimensional efficiency curves at the fiducial point, one axis at
           a time, including duty cycle (0.4, 0.657, 0.9). The axis held fixed
           in Parts F and G's main design.  1475 fits before feasibility cuts.

The analysis, the selection function and the figures are in analyse_part_g.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import lib
import injection as inj
import part_f_inject as pf
import surface as S

PART = "G"
PI = lib.PART_INDEX["G"]
N_REFINE = 3000
OUT = os.path.join(lib.RESULTS, "part_g_per_fit.csv")
PLAN = os.path.join(lib.RESULTS, "part_g_plan.json")

FIELDS = pf.FIELDS + ["block", "varied_axis"]

# config_index slots: refine 20-22 (3000 points), curves 40+
CI_REFINE = 20
CI_CURVES = 40

CURVE_LEVELS = 9
CURVE_REPS = 25


def load_part_f():
    return pd.read_csv(os.path.join(lib.RESULTS, "part_f_per_fit.csv"),
                       float_precision="round_trip")


def fit_part_f_surface(df, fap):
    s = S.Surface(fap=fap, outcome="chirp")
    s.fit(df, df["chirp"].to_numpy())
    rank = s.deviance_ranking(df, df["chirp"].to_numpy())
    return s, rank


def significant_axes(rank, k=3):
    """The axes Part F ranked significant, by unique deviance explained."""
    cont = [r for r in rank["ranking"] if r["axis"] in S.AXES]
    return [r["axis"] for r in cont[:k]]


def build_refinement(surf, rank, band_levels=(0, 1), seed=4242):
    """Rejection-sample the box until the surface predicts 0.2 < p < 0.8.

    Only the axes Part F ranked significant are varied, the rest are held at the
    fiducial point of Section 8.1 and that choice is recorded.
    """
    varied = significant_axes(rank, k=3)
    held = [a for a in S.AXES if a not in varied]
    rng = np.random.default_rng(seed)
    pts, n_examined, n_infeasible, n_outside = [], 0, 0, 0
    per_level = N_REFINE // len(band_levels)
    counts = {bs: 0 for bs in band_levels}

    while any(counts[bs] < per_level for bs in band_levels) and n_examined < 4_000_000:
        m = 4000
        U = rng.random((m, 6))
        ax_list = []
        for u in U:
            ax = dict(inj.FIDUCIAL_POINT)
            for a in varied:
                lo, hi = inj.BOUNDS[a]
                v = u[S.AXES.index(a)]
                ax[a] = float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo)))
                              if a in inj.LOG_AXES else lo + v * (hi - lo))
            ax_list.append(ax)
        for bs in band_levels:
            if counts[bs] >= per_level:
                continue
            ok_ax = []
            for ax in ax_list:
                n_examined += 1
                ok, _ = inj.feasible(ax, bs)
                if ok:
                    ok_ax.append(ax)
                else:
                    n_infeasible += 1
            if not ok_ax:
                continue
            d = pd.DataFrame(ok_ax)
            d["band_structure"] = bs
            p = surf.predict_df(d)
            sel = (p > 0.2) & (p < 0.8)
            n_outside += int((~sel).sum())
            for ax, keep in zip(ok_ax, sel):
                if keep and counts[bs] < per_level:
                    pts.append((ax, bs))
                    counts[bs] += 1
    return pts, dict(varied_axes=varied, held_fixed=held,
                     held_values={a: inj.FIDUCIAL_POINT[a] for a in held},
                     n_examined=n_examined, n_infeasible=n_infeasible,
                     n_outside_band=n_outside, n_accepted=len(pts),
                     per_band_level=counts,
                     band_target="predicted chirp-recovery probability in (0.2, 0.8)")


def build_curves():
    """One axis at a time about the fiducial point, plus duty cycle."""
    tasks = []
    idx = 0
    plans = []
    for a in S.AXES:
        lo, hi = inj.BOUNDS[a]
        if a in inj.LOG_AXES:
            levels = np.exp(np.linspace(np.log(lo), np.log(hi), CURVE_LEVELS))
        else:
            levels = np.linspace(lo, hi, CURVE_LEVELS)
        for v in levels:
            ax = dict(inj.FIDUCIAL_POINT)
            ax[a] = float(v)
            ok, why = inj.feasible(ax, 1)
            plans.append(dict(axis=a, level=float(v), feasible=bool(ok),
                              reason=why))
            if not ok:
                continue
            for r in range(CURVE_REPS):
                tasks.append(dict(ax=ax, band_structure=1, duty_cycle=inj.DUTY_FIDUCIAL,
                                  varied_axis=a, i=idx))
                idx += 1
    # duty cycle, the axis held fixed everywhere else
    for duty in (0.4, inj.DUTY_FIDUCIAL, 0.9):
        ax = dict(inj.FIDUCIAL_POINT)
        ok, why = inj.feasible(ax, 1, duty_cycle=duty)
        plans.append(dict(axis="duty_cycle", level=float(duty),
                          feasible=bool(ok), reason=why))
        if not ok:
            continue
        for r in range(CURVE_REPS):
            tasks.append(dict(ax=ax, band_structure=1, duty_cycle=float(duty),
                              varied_axis="duty_cycle", i=idx))
            idx += 1
    for bs in (0, 1):
        ax = dict(inj.FIDUCIAL_POINT)
        ok, why = inj.feasible(ax, bs)
        plans.append(dict(axis="band_structure", level=float(bs),
                          feasible=bool(ok), reason=why))
        if not ok:
            continue
        for r in range(CURVE_REPS):
            tasks.append(dict(ax=ax, band_structure=bs,
                              duty_cycle=inj.DUTY_FIDUCIAL,
                              varied_axis="band_structure", i=idx))
            idx += 1
    return tasks, plans


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    th = pf.thresholds()
    fu = pf.grid_setting()
    cm = pf.cut_mode()
    thr = th["fiducial_corrected"]["DeltaLambda_chirp"]["fap_1e_4"]["threshold"]
    thr_o = th["fiducial_corrected"]["Lambda_osc"]["fap_1e_4"]["threshold"]
    src = ("threshold model over baseline and duty cycle, corrected by Part C, "
           "FAP 1e-4")

    dff = load_part_f()
    surf, rank = fit_part_f_surface(dff, fap=1e-4)
    print("PART G: Part F axis ranking by unique deviance explained")
    for r in rank["ranking"]:
        print(f"   {r['axis']:<20} {r['deviance_increase']:9.1f} "
              f"({100*r['frac_of_explained']:5.1f}% of explained)")

    pts, meta = build_refinement(surf, rank)
    print(f"PART G: {len(pts)} refinement points, varying "
          f"{meta['varied_axes']}, holding {meta['held_fixed']} at fiducial")

    tasks = []
    for k, (ax, bs) in enumerate(pts):
        tasks.append(dict(ax=ax, band_structure=bs, i=k,
                          config_index=CI_REFINE + k // 1000,
                          part=PART, part_index=PI,
                          duty_cycle=inj.DUTY_FIDUCIAL,
                          threshold=thr, threshold_losc=thr_o,
                          threshold_source=src, cut_mode=cm, freq_uniform=fu,
                          extra=dict(block="refine", varied_axis="")))
    n_ref = len(tasks)

    # The duty-cycle curve is the only place where baseline or duty leaves
    # the fiducial point.  The threshold model over baseline and duty cycle
    # FAILED its leave-one-configuration-out validation, and failed on its
    # own axis, predicting both duty configurations about 12% high, so the
    # fallback applies: per-configuration calibration, taking the duty
    # dependence from stage 1's directly measured p99 at each level.
    import thresholds as TH
    validated = bool(th["threshold_model"]["validated"])
    ctasks, plans = build_curves()
    for t in ctasks:
        duty = t.get("duty_cycle", inj.DUTY_FIDUCIAL)
        if abs(duty - inj.DUTY_FIDUCIAL) < 1e-9:
            th_c, th_o2, s = thr, thr_o, src
        else:
            d = TH.duty_ratio_per_configuration(duty)
            th_c, th_o2 = thr * d["ratio_dchirp"], thr_o * d["ratio_losc"]
            s = (f"per-configuration calibration (the Section 8.2 model failed "
                 f"leave-one-out validation): corrected fiducial threshold x "
                 f"stage-1 measured p99 ratio for {d['label']} "
                 f"(x{d['ratio_dchirp']:.4f}, n={d['n_nulls']} nulls), FAP 1e-4")
        t.update(config_index=CI_CURVES + t["i"] // 1000, part=PART,
                 part_index=PI, threshold=th_c, threshold_losc=th_o2,
                 threshold_source=s, cut_mode=cm, freq_uniform=fu,
                 extra=dict(block="curves", varied_axis=t["varied_axis"]))
    tasks += ctasks
    print(f"PART G: {n_ref} refinement + {len(ctasks)} one-dimensional "
          f"= {len(tasks)} fits")

    with open(PLAN, "w", encoding="utf-8") as f:
        json.dump(dict(refinement=meta, part_f_ranking=rank,
                       curve_plan=plans, curve_levels=CURVE_LEVELS,
                       curve_reps=CURVE_REPS,
                       fiducial_point=inj.FIDUCIAL_POINT,
                       n_refine=n_ref, n_curves=len(ctasks)), f, indent=1,
                  default=float)

    fw = lib.RowWriter(pf.FEAT_OUT, pf.FEAT_FIELDS)
    try:
        pf.run_pairs(tasks, pf.worker, OUT, FIELDS, fw, n_jobs=n_jobs,
                     part=PART, script="part_g_refine.py")
    finally:
        fw.close()


if __name__ == "__main__":
    main()
