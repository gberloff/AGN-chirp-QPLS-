"""PART C: repair the outlier cut.

Three configurations x three cut modes x 300 nulls = 2700 fits, plus a fourth
arm that widens the running-median window from 50 d to 150 d under mad_scaled
(3 x 300 = 900 fits), reported separately as the window effect.  3600 fits.

Nothing is injected here.  Pure background throughout.

The four arms SHARE the light curve for a given (configuration, realisation):
the seed depends on the realisation only, never on the arm.  That makes the
arms paired, so a threshold difference between them is measured without the
realisation noise that would otherwise dominate it.
"""
import os
import sys

import numpy as np

import lib
import analysis

lib.install_cut_modes()

PART = "C"
PI = lib.PART_INDEX["C"]
N_NULL = 300
OUT = os.path.join(lib.RESULTS, "part_c_per_fit.csv")

CONFIGS = [
    dict(label="fiducial", baseline_d=2000.0, tau_d=320.0),
    dict(label="tau_over_T=1.6", baseline_d=2000.0, tau_d=3200.0),
    dict(label="T=4000", baseline_d=4000.0, tau_d=640.0),
]
ARMS = [("mad_raw", 50.0), ("mad_scaled", 50.0), ("sigma_clip", 50.0),
        ("mad_scaled", 150.0)]

FIELDS = ["part", "config_index", "i", "realisation", "arm_index", "cut_mode",
          "window_d", "label", "seed", "baseline_d", "tau_d", "tau_over_T",
          "n_bands", "n_epochs_in", "n_epochs", "n_removed_catflags",
          "n_removed_magerr", "n_removed_mad", "n_removed_total",
          "insufficient_data", "cut_impl", "cut_fired", "frac_removed",
          "Lambda_osc", "Lambda_per0", "DeltaLambda_chirp", "P_hat", "eta_hat",
          "sigma_hat", "tau_hat", "boundary_flag", "converged",
          "n_grid_nodes", "runtime_s"]

_GC = lib.GenCache()


def entry_for(ci):
    c = CONFIGS[ci]
    return lib.make_entry(ci, c["label"], baseline_d=c["baseline_d"],
                          tau_d=c["tau_d"])


def worker(task):
    lib.install_cut_modes()          # joblib workers do not inherit this
    ci, ai, real = task["config_index"], task["arm_index"], task["realisation"]
    c = CONFIGS[ci]
    entry = entry_for(ci)
    g = _GC.get(entry)
    mode, win = ARMS[ai]
    spec = lib.build_spec(entry, cut_mode=mode, window_d=win)
    lib.assert_cut_mode_active(spec)
    seed = lib.seed_for(PI, ci, real)          # arm-independent: paired arms
    res = analysis.run_on_null(g, seed, spec).as_dict()
    n_in = int(res["n_epochs_in"])
    row = dict(
        part=PART, config_index=ci, i=task["i"], realisation=real,
        arm_index=ai, cut_mode=mode, window_d=win, label=c["label"], seed=seed,
        baseline_d=c["baseline_d"], tau_d=c["tau_d"],
        tau_over_T=c["tau_d"] / c["baseline_d"], n_bands=int(res["n_bands"]),
        cut_impl=lib.cut_impl_name(),
        cut_fired=int(res["n_removed_total"] > 0),
        frac_removed=(res["n_removed_total"] / n_in) if n_in else np.nan,
    )
    for k in ("n_epochs_in", "n_epochs", "n_removed_catflags", "n_removed_magerr",
              "n_removed_mad", "n_removed_total", "insufficient_data",
              "Lambda_osc", "Lambda_per0", "DeltaLambda_chirp", "P_hat",
              "eta_hat", "sigma_hat", "tau_hat", "boundary_flag", "converged",
              "n_grid_nodes", "runtime_s"):
        row[k] = res[k]
    return row


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tasks = []
    for ai in range(len(ARMS)):
        for ci in range(len(CONFIGS)):
            for k in range(N_NULL):
                # i encodes the arm so that (part, config_index, i) is unique
                tasks.append(dict(config_index=ci, arm_index=ai, realisation=k,
                                  i=ai * 1000 + k))
    print(f"PART C: {len(tasks)} fits "
          f"({len(CONFIGS)} configs x {len(ARMS)} arms x {N_NULL} nulls)")
    lib.run_batch(tasks, worker, OUT, FIELDS, PART, n_jobs=n_jobs,
                  batch_size=30, pause_between_batches_s=5.0,
                  pause_every=300, pause_long_s=90.0, heartbeat_every=10)


if __name__ == "__main__":
    main()
