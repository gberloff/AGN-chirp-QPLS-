
import numpy as np

import qpls.statistic_v3 as v3
from qpls.detector_comparison import CFG
from qpls.injection_recovery import (null_threshold, recovery_grid,
                                     gls_peak_power)
from qpls.source_scenarios import SOURCE_LIBRARY

SEEDS = [42, 1729, 8191, 314159, 20260806]

ANCHOR_PRIORITY = ["AGB170", "AGB215", "RGBiso", "AGBiso", "AGB260"]

BLIND = ["global_max", "blind_harmonic_sum"]


def seed_thresholds(t, sigma, freq, det_hs, seed, n_null):
    thr = {}
    thr["global_max"], _ = null_threshold(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], freq,
        gls_peak_power, CFG["fap"], n_null, np.random.default_rng(seed))
    thr["blind_harmonic_sum"], _ = null_threshold(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], freq,
        det_hs, CFG["fap"], n_null, np.random.default_rng(seed))
    return thr


def anchor_curves(t, sigma, freq, det_hs, thr, anchor, seed):
    R_star = SOURCE_LIBRARY[anchor]["R"]
    fwhm_days = v3.t_mag_hours(R_star) / 24.0
    P_grid = np.array([v3.P_V3])
    out = {}
    rng = np.random.default_rng(seed)
    out["global_max"] = recovery_grid(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid,
        v3.A_SWEEP, freq, thr["global_max"], gls_peak_power, CFG["n_inj"],
        fwhm_days, rng, template=v3.TEMPLATE)[:, 0]
    rng = np.random.default_rng(seed)
    out["blind_harmonic_sum"] = recovery_grid(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid,
        v3.A_SWEEP, freq, thr["blind_harmonic_sum"], det_hs, CFG["n_inj"],
        fwhm_days, rng, template=v3.TEMPLATE)[:, 0]
    return out, fwhm_days


def floors_from_curves(curves):
    return {det: v3.floor_at_half(v3.A_SWEEP, curves[det]) for det in BLIND}


def r50_from_floors(anchors_in_radius_order, floors_by_anchor, det):
    radii = [SOURCE_LIBRARY[a]["R"] for a in anchors_in_radius_order]
    fl = [floors_by_anchor[a][det] for a in anchors_in_radius_order]
    finite = [(R, f) for R, f in zip(radii, fl) if f is not None]
    if not finite:
        return None
    i = radii.index(finite[0][0])
    if i == 0:
        return radii[0]
    return 0.5 * (radii[i - 1] + radii[i])


def main(argv=None):
    import argparse, json, time, pathlib
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--anchors", nargs="+", default=ANCHOR_PRIORITY)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    t, sigma, freq, det_hs = v3.setup(verbose=True)
    print(f"\nseeds={a.seeds}  n_null={a.n_null}  n_inj={CFG['n_inj']}  "
          f"anchors(priority order)={a.anchors}", flush=True)

    rec = dict(n_null=a.n_null, n_inj=CFG["n_inj"], seeds=a.seeds,
               anchors=a.anchors, A_SWEEP=list(map(float, v3.A_SWEEP)),
               P_days=float(v3.P_V3), template=v3.TEMPLATE, per_seed={})
    t_start = time.perf_counter()
    for si, seed in enumerate(a.seeds):
        print(f"\n===== SEED {seed} =====", flush=True)
        ts = time.perf_counter()
        thr = seed_thresholds(t, sigma, freq, det_hs, seed, a.n_null)
        print(f"  thresholds: global_max={thr['global_max']:.6f}  "
              f"blind_harmonic_sum={thr['blind_harmonic_sum']:.6f}", flush=True)
        block = dict(thresholds={k: float(v) for k, v in thr.items()},
                     anchors={})
        for ai, anc in enumerate(a.anchors):
            c, fwhm = anchor_curves(t, sigma, freq, det_hs, thr, anc, seed)
            fl = floors_from_curves(c)
            block["anchors"][anc] = dict(
                R_Rsun=float(SOURCE_LIBRARY[anc]["R"]), fwhm_days=float(fwhm),
                curves={d: list(map(float, c[d])) for d in BLIND},
                floors={d: (None if fl[d] is None else float(fl[d]))
                        for d in BLIND})
            print(f"  [{anc:7s}] gm_max={max(c['global_max']):.4f} "
                  f"hs_max={max(c['blind_harmonic_sum']):.4f} "
                  f"floors gm={fl['global_max']} hs={fl['blind_harmonic_sum']}",
                  flush=True)
            if si == 0 and ai == 0:
                per = time.perf_counter() - ts
                est = (per * len(a.anchors) + 30) * len(a.seeds) / 3600.0
                print(f"  >>> WALL-TIME ESTIMATE after first anchor: "
                      f"{est:.2f} h for the full run <<<", flush=True)
        order = sorted(a.anchors, key=lambda k: SOURCE_LIBRARY[k]["R"])
        block["R_50"] = {d: r50_from_floors(
            order, {k: v["floors"] for k, v in block["anchors"].items()}, d)
            for d in BLIND}
        print(f"  R_50: {block['R_50']}   ({time.perf_counter()-ts:.0f} s)",
              flush=True)
        rec["per_seed"][str(seed)] = block

    rec["elapsed_s"] = time.perf_counter() - t_start
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {a.out}   total {rec['elapsed_s']/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
