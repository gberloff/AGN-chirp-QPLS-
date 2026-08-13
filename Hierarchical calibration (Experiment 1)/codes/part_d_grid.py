"""PART D: settle the search grid.

Fiducial configuration, T = 2000 d, adopted cut mode, 300 nulls on each of
seven grid variants.  2100 fits.  G2 is the previous grid and acts as control.

Grid resolution sets the look-elsewhere volume, hence every threshold and every
efficiency, and it has never been varied.

As in Part C the variants SHARE the light curve for a given realisation: the
seed depends on the realisation only, so the comparison between variants is
paired.
"""
import json
import os
import sys

import numpy as np

import lib
import analysis

lib.install_cut_modes()

PART = "D"
PI = lib.PART_INDEX["D"]
N_NULL = 300
T = 2000.0
OUT = os.path.join(lib.RESULTS, "part_d_per_fit.csv")

VARIANTS = [
    dict(name="G1", P_n=80,  P_min_d=60.0,  P_max_d=T / 2.5),
    dict(name="G2", P_n=160, P_min_d=60.0,  P_max_d=T / 2.5),   # control
    dict(name="G3", P_n=320, P_min_d=60.0,  P_max_d=T / 2.5),
    dict(name="G4", P_n=160, P_min_d=30.0,  P_max_d=T / 2.5),
    dict(name="G5", P_n=160, P_min_d=120.0, P_max_d=T / 2.5),
    dict(name="G6", P_n=160, P_min_d=60.0,  P_max_d=T / 5.0),
# G7 is not one of the six variants originally planned.  It exists because
# adopting frequency-uniform spacing CHANGES the look-elsewhere volume, and
# every threshold this run corrected was measured on a log-spaced grid.
# Without it, Parts F and G would apply a G2 threshold to a grid that is
# not G2.  Same 300 realisations as every variant, so the ratio is paired.
    dict(name="G7", P_n=160, P_min_d=60.0,  P_max_d=T / 2.5, freq_uniform=True),
]

FIELDS = ["part", "config_index", "i", "realisation", "variant", "P_n",
          "P_min_d", "P_max_d", "seed", "n_epochs_in", "n_epochs",
          "n_removed_total", "insufficient_data",
          "Lambda_osc", "Lambda_per0", "DeltaLambda_chirp", "P_hat", "eta_hat",
          "sigma_hat", "tau_hat", "boundary_flag", "converged",
          "n_grid_nodes", "runtime_s", "cut_impl",
          "P_short20_cut_d", "in_short20", "below_100d"]

_GC = lib.GenCache()
_CUT_MODE = None


def cut_mode():
    global _CUT_MODE
    if _CUT_MODE is None:
        p = os.path.join(lib.RESULTS, "part_c.json")
        with open(p, encoding="utf-8") as f:
            _CUT_MODE = json.load(f)["adopted_cut_mode"]
    return _CUT_MODE


def entry_for(vi):
    v = VARIANTS[vi]
    e = lib.make_entry(vi, v["name"], baseline_d=T)
    e["grid_P"] = dict(n=v["P_n"], min_d=v["P_min_d"], max_d=v["P_max_d"],
                       spacing="log")
    return e


def worker(task):
    lib.install_cut_modes()          # joblib workers do not inherit this
    vi, real = task["config_index"], task["realisation"]
    v = VARIANTS[vi]
    entry = entry_for(vi)
    g = _GC.get(entry)
    lib.set_grid_spacing(bool(v.get("freq_uniform", False)))
    spec = lib.build_spec(entry, cut_mode=cut_mode(), window_d=50.0)
    lib.assert_cut_mode_active(spec)
    seed = lib.seed_for(PI, 0, real)      # variant-independent: paired variants
    res = analysis.run_on_null(g, seed, spec).as_dict()
    # the shortest fifth of the searched LOG-P range
    cut20 = v["P_min_d"] * (v["P_max_d"] / v["P_min_d"]) ** 0.2
    row = dict(part=PART, config_index=vi, i=task["i"], realisation=real,
               variant=v["name"], P_n=v["P_n"], P_min_d=v["P_min_d"],
               P_max_d=v["P_max_d"], seed=seed, cut_impl=lib.cut_impl_name(),
               P_short20_cut_d=cut20,
               in_short20=int(res["P_hat"] < cut20),
               below_100d=int(res["P_hat"] < 100.0))
    for k in ("n_epochs_in", "n_epochs", "n_removed_total", "insufficient_data",
              "Lambda_osc", "Lambda_per0", "DeltaLambda_chirp", "P_hat",
              "eta_hat", "sigma_hat", "tau_hat", "boundary_flag", "converged",
              "n_grid_nodes", "runtime_s"):
        row[k] = res[k]
    return row


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"PART D: adopted cut mode = {cut_mode()}")
    tasks = [dict(config_index=vi, realisation=k, i=k)
             for vi in range(len(VARIANTS)) for k in range(N_NULL)]
    print(f"PART D: {len(tasks)} fits ({len(VARIANTS)} variants x {N_NULL} nulls)")
    lib.run_batch(tasks, worker, OUT, FIELDS, PART, n_jobs=n_jobs,
                  batch_size=30, pause_between_batches_s=5.0,
                  pause_every=300, pause_long_s=90.0, heartbeat_every=10)


if __name__ == "__main__":
    main()
