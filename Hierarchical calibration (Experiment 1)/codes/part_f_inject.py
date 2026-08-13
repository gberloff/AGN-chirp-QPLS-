"""PART F: screening injections.

A Latin hypercube of 2500 points over the six continuous axes, one realisation
each, repeated at both band-structure levels: 5000 injections.  Background is
DRW throughout, matching the note's scope for item 1.

More points with one realisation each beats fewer points with many, for fitting
a response surface to binary outcomes.

Every curve also gets its Part E feature vector, computed here so the campaign
does not have to be regenerated later.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

import lib
import analysis
import features as feat
import injection as inj

lib.install_cut_modes()

PART = "F"
PI = lib.PART_INDEX["F"]
N_POINTS = 2500
OVERSAMPLE = 6
BAND_LEVELS = (0, 1)          # 0 = 1 band, 1 = 2 bands free phase
CONFIG_OF_BAND = {0: 0, 1: 10}   # 1000-wide seed slots, 3 each
OUT = os.path.join(lib.RESULTS, "part_f_per_fit.csv")
FEAT_OUT = os.path.join(lib.RESULTS, "features.csv")
DESIGN = os.path.join(lib.RESULTS, "part_f_design.json")

FIELDS = ([
    "part", "config_index", "i", "band_structure", "seed",
    "n_cyc", "eta_x", "snr", "tau_over_p", "a2_over_a1", "samples_per_cycle",
    "P_true_d", "eta_true", "A1_true", "A2_true", "tau_d", "dt_d",
    "T_baseline_d", "duty_cycle", "n_bands", "sigma_eff", "path_rms",
    "n_epochs_raw", "n_epochs", "n_epochs_in",
    "n_removed_catflags", "n_removed_magerr", "n_removed_mad",
    "n_removed_total", "insufficient_data",
    "threshold_used", "threshold_source", "threshold_losc",
    "Lambda_osc", "Lambda_per0", "DeltaLambda_chirp",
    "P_hat", "eta_hat", "sigma_hat", "tau_hat", "boundary_flag", "converged",
    "n_grid_nodes", "runtime_s", "feature_runtime_s", "cut_impl",
    "trigger", "correct", "alias", "chirp",
    "losc_trigger", "losc_correct", "losc_alias", "losc_chirp",
    "P_lo68", "P_hi68", "P_edge68", "P_lo95", "P_hi95", "P_edge95",
    "eta_lo68", "eta_hi68", "eta_edge68", "eta_lo95", "eta_hi95", "eta_edge95",
])

FEAT_FIELDS = ["part", "config_index", "i"] + feat.FEATURE_NAMES

_TH = {}


def thresholds():
    if not _TH:
        with open(os.path.join(lib.RESULTS, "thresholds.json"),
                  encoding="utf-8") as f:
            _TH.update(json.load(f))
    return _TH


def grid_setting():
    with open(os.path.join(lib.RESULTS, "part_d.json"), encoding="utf-8") as f:
        return bool(json.load(f)["grid_choice_freq_uniform"])


def cut_mode():
    return thresholds()["adopted_cut_mode"]


def build_design(seed=20260811):
    """Latin hypercube over the box, filtered to the feasible region.

    Feasibility is band-dependent (a one-band cadence presents half the epochs),
    so each level is filtered against its own constraint and the rejection
    fraction is reported for each.
    """
    eng = qmc.LatinHypercube(d=6, seed=seed)
    U = eng.random(N_POINTS * OVERSAMPLE)
    design, stats = {}, {}
    for bs in BAND_LEVELS:
        pts, reasons = [], {}
        for u in U:
            if len(pts) >= N_POINTS:
                break
            ax = inj.unit_to_axes(u)
            ok, why = inj.feasible(ax, bs)
            if ok:
                pts.append(ax)
            else:
                reasons[why] = reasons.get(why, 0) + 1
        n_examined = len(pts) + sum(reasons.values())
        design[bs] = pts
        stats[bs] = dict(n_accepted=len(pts), n_examined=n_examined,
                         rejection_fraction=1.0 - len(pts) / n_examined,
                         reasons=reasons,
                         reason_fractions={k: v / n_examined
                                           for k, v in reasons.items()})
    return design, stats


def _corr(design):
    """Induced correlations among the axes in the feasible region."""
    out = {}
    for bs, pts in design.items():
        M = np.array([[np.log(p["n_cyc"]), p["eta_x"], np.log(p["snr"]),
                       np.log(p["tau_over_p"]), p["a2_over_a1"],
                       np.log(p["samples_per_cycle"])] for p in pts])
        C = np.corrcoef(M.T)
        names = ["log_n_cyc", "eta_x", "log_snr", "log_tau_over_p",
                 "a2_over_a1", "log_spc"]
        out[bs] = {f"{names[i]}~{names[j]}": float(C[i, j])
                   for i in range(6) for j in range(i + 1, 6)}
    return out


def worker(task):
    """One injection.  Parts F and G share this code path exactly, Part G
    passes its own part label, seed index, duty cycle and bookkeeping fields.
    """
    lib.install_cut_modes()          # joblib workers do not inherit this
    ax = task["ax"]
    bs = task["band_structure"]
    ci = task["config_index"]
    part = task.get("part", PART)
    pi = task.get("part_index", PI)
    duty = task.get("duty_cycle", inj.DUTY_FIDUCIAL)
    d = inj.derive(ax, bs, duty_cycle=duty)
    entry = inj.build_entry(d, config_index=ci, label=f"{part}_bs{bs}")
    lib.set_grid_spacing(task["freq_uniform"])
    spec = lib.build_spec(entry, cut_mode=task["cut_mode"], window_d=50.0)
    lib.assert_cut_mode_active(spec)
    g = __import__("gen").Generator(entry, 20260807)
    seed = lib.seed_for(pi, ci, task["i"])

    t, y, sig, band, meta = inj.make_curve(g, seed, ax, d, bs)
    # analyse returns early, and WITHOUT a surface, when the minimum-epoch cut
    # fires, so both return shapes must be handled.
    out = analysis.analyse(t, y, sig, band, spec, return_surface=True)
    if isinstance(out, tuple):
        ares, surface = out
        Pgrid, etagrid = analysis.build_grid(spec)
        iv = inj.profile_intervals(surface, Pgrid, etagrid)
    else:
        ares, iv = out, inj.profile_intervals_nan()
    res = ares.as_dict()
    fv, ft = feat.compute(t, y, sig, band, spec)

    thr = task["threshold"]
    thr_o = task["threshold_losc"]
    o1 = inj.outcomes(res["DeltaLambda_chirp"], thr, res["P_hat"], res["eta_hat"],
                      d["P_true_d"], d["eta_true"])
    o2 = inj.outcomes(res["Lambda_osc"], thr_o, res["P_hat"], res["eta_hat"],
                      d["P_true_d"], d["eta_true"], prefix="losc_")

    row = dict(part=part, config_index=ci, i=task["i"], band_structure=bs,
               seed=seed, T_baseline_d=d["baseline_d"],
               duty_cycle=d["duty_cycle"], n_bands=d["n_bands"],
               P_true_d=d["P_true_d"], eta_true=d["eta_true"], tau_d=d["tau_d"],
               dt_d=d["dt_d"], A1_true=meta["amp1"], A2_true=meta["amp2"],
               sigma_eff=meta["sigma_eff"], path_rms=meta["path_rms"],
               n_epochs_raw=meta["n_epochs_raw"],
               threshold_used=thr, threshold_source=task["threshold_source"],
               threshold_losc=thr_o, feature_runtime_s=ft,
               cut_impl=lib.cut_impl_name(), **o1, **o2, **iv)
    for k in ("n_cyc", "eta_x", "snr", "tau_over_p", "a2_over_a1",
              "samples_per_cycle"):
        row[k] = ax[k]
    for k in ("n_epochs", "n_epochs_in", "n_removed_catflags", "n_removed_magerr",
              "n_removed_mad", "n_removed_total", "insufficient_data",
              "Lambda_osc", "Lambda_per0", "DeltaLambda_chirp", "P_hat",
              "eta_hat", "sigma_hat", "tau_hat", "boundary_flag", "converged",
              "n_grid_nodes", "runtime_s"):
        row[k] = res[k]
    row.update(task.get("extra", {}))
    frow = dict(part=part, config_index=ci, i=task["i"])
    frow.update({k: fv.get(k, np.nan) for k in feat.FEATURE_NAMES})
    return row, frow


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n_points = int(sys.argv[2]) if len(sys.argv) > 2 else N_POINTS
    bands = BAND_LEVELS
    if len(sys.argv) > 3:
        bands = tuple(int(x) for x in sys.argv[3].split(","))

    th = thresholds()
    fu = grid_setting()
    cm = cut_mode()
    thr = th["fiducial_corrected"]["DeltaLambda_chirp"]["fap_1e_4"]["threshold"]
    thr_o = th["fiducial_corrected"]["Lambda_osc"]["fap_1e_4"]["threshold"]
    src = ("threshold model over baseline and duty cycle, evaluated at "
           "T=2000 d, duty=0.657, corrected by Part C, FAP 1e-4")
    print(f"PART F: cut_mode={cm}  freq_uniform_grid={fu}")
    print(f"        threshold DeltaLambda_chirp = {thr:.4f} at FAP 1e-4")
    print(f"        threshold Lambda_osc        = {thr_o:.4f} at FAP 1e-4")

    design, stats = build_design()
    with open(DESIGN, "w", encoding="utf-8") as f:
        json.dump(dict(n_points=n_points, oversample=OVERSAMPLE,
                       rejection=stats, induced_correlations=_corr(design),
                       bounds=inj.BOUNDS, log_axes=sorted(inj.LOG_AXES),
                       baseline_d=inj.T_BASELINE, duty_cycle=inj.DUTY_FIDUCIAL),
                  f, indent=1, default=float)
    for bs in BAND_LEVELS:
        s = stats[bs]
        print(f"        band level {bs}: {s['n_accepted']} accepted of "
              f"{s['n_examined']} examined, rejection "
              f"{100*s['rejection_fraction']:.1f}%  {s['reasons']}")

    tasks = []
    for bs in bands:
        for k, ax in enumerate(design[bs][:n_points]):
            tasks.append(dict(ax=ax, band_structure=bs, i=k,
                              config_index=CONFIG_OF_BAND[bs],
                              threshold=thr, threshold_losc=thr_o,
                              threshold_source=src, cut_mode=cm,
                              freq_uniform=fu))
    print(f"PART F: {len(tasks)} injections")

    fw = lib.RowWriter(FEAT_OUT, FEAT_FIELDS)
    try:
        run_pairs(tasks, worker, OUT, FIELDS, fw, n_jobs=n_jobs)
    finally:
        fw.close()


def run_pairs(tasks, worker_fn, out_path, fields, feat_writer, n_jobs=2,
              part=PART, script="part_f_inject.py"):
    """Like lib.run_batch, but each worker returns (result row, feature row)."""
    from joblib import Parallel, delayed
    n_jobs = max(1, min(int(n_jobs), 4))
    batch_size = 30
    done = lib.existing_keys(out_path)
    todo = [t for t in tasks
            if (t.get("part", part), int(t["config_index"]), int(t["i"]))
            not in done]
    print(f"[{part}] {len(tasks)} tasks, {len(done)} on disk, {len(todo)} to run, "
          f"n_jobs={n_jobs}")
    print(f"RESUME COMMAND:\n"
          f"    python {script}\n")
    if not todo:
        return
    wr = lib.RowWriter(out_path, fields)
    hb = lib.Heartbeat(part, every=10)
    since, t0 = 0, time.time()
    try:
        with Parallel(n_jobs=n_jobs, backend="loky", batch_size=1,
                      max_nbytes=None) as pool:
            for k in range(0, len(todo), batch_size):
                chunk = todo[k:k + batch_size]
                out = pool(delayed(worker_fn)(t) for t in chunk)
                for t, (row, frow) in zip(chunk, out):
                    wr.write(row)
                    feat_writer.write(frow)
                    hb.beat(int(t["config_index"]), int(t["i"]), wr.n)
                since += len(chunk)
                el = time.time() - t0
                print(f"  [{part}] {wr.n}/{len(todo)}  {el/60:.1f} min  "
                      f"{wr.n/el*60:.1f} rows/min", flush=True)
                if since >= 300:
                    print(f"  [{part}] thermal pause 90 s", flush=True)
                    time.sleep(90.0)
                    since = 0
                elif k + batch_size < len(todo):
                    time.sleep(5.0)
    finally:
        hb.beat(-1, -1, wr.n, force=True)
        wr.close()
    print(f"[{part}] done: {wr.n} rows in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
