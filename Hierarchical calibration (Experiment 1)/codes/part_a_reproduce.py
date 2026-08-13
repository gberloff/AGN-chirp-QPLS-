"""PART A addendum: functional verification of analyse() by reproduction.

PROVENANCE.md records the sha256 of `code/analysis.py` as it stood on
2026-08-08 12:07, i.e. BEFORE the documented v2 change that moved the quality
cuts inside `analyse`.
PROVENANCE.md was never regenerated afterwards, so the recorded hash is stale
and the file on disk is the v2 file that actually produced
`results/stage1_per_fit.csv`.

A stale hash cannot distinguish "the code changed as documented" from "the code
is corrupt".  This script settles it the only way that matters: it re-runs the
current `analyse` on stored rows, regenerating each light curve from its stored
seed, and compares every scored field bitwise against what is on disk.

Run read-only against hcal/.  Writes only item1/results/part_a_reproduce.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ITEM1 = os.path.dirname(HERE)
HCAL = os.path.dirname(ITEM1)
sys.path.insert(0, os.path.join(HCAL, "code"))

import analysis  # noqa: E402
import gen       # noqa: E402

# fields that must agree bitwise (runtime excluded)
FIELDS = ["Lambda_osc", "Lambda_per0", "DeltaLambda_chirp", "P_hat", "eta_hat",
          "sigma_hat", "tau_hat", "l0", "boundary_flag", "converged",
          "n_epochs", "n_epochs_in", "n_removed_mad", "n_removed_total",
          "n_grid_nodes", "n_linear_params", "argmax_ip", "argmax_je"]


def main(n_per_config=1, configs=(0, 1, 5, 12, 17, 20, 99)):
    cfg = json.load(open(os.path.join(HCAL, "config.json"), encoding="utf-8"))
    entries = {e["config_index"]: e for e in cfg["stage1a"]["configs"]}
    # config_index 99 re-uses the fiducial entry (stage 1b tail block)
    entries[99] = dict(entries[0], config_index=99)
    cadence_seed = int(cfg["cadence_seed"])

    # float_precision="round_trip" is required: pandas' default C float parser
    # is not exactly round-trip and by itself introduces diffs of ~1e-13 on
    # fields of order 1e3, which would masquerade as a code difference.
    per_fit = pd.read_csv(os.path.join(HCAL, "results", "stage1_per_fit.csv"),
                          float_precision="round_trip")

    rows, t0 = [], time.time()
    for ci in configs:
        sub = per_fit[per_fit["config_index"] == ci]
        if sub.empty:
            continue
        entry = entries[ci]
        spec = analysis.Spec.from_config(cfg, entry)
        g = gen.Generator(entry, cadence_seed)
        for _, r in sub.head(n_per_config).iterrows():
            res = analysis.run_on_null(g, int(r["seed"]), spec)
            d = res.as_dict()
            diffs = {}
            for k in FIELDS:
                a, b = float(d[k]), float(r[k])
                diffs[k] = abs(a - b)
            worst = max(diffs.values())
            n_exact = sum(1 for v in diffs.values() if v == 0.0)
            rows.append(dict(config_index=int(ci), i=int(r["i"]),
                             seed=int(r["seed"]), worst_abs_diff=worst,
                             n_exact=n_exact, n_fields=len(FIELDS),
                             stored_Lambda_osc=float(r["Lambda_osc"]),
                             recomputed_Lambda_osc=float(d["Lambda_osc"])))
            print(f"  config {ci:>2}  i={int(r['i']):<4} worst |diff| = {worst:.3e}"
                  f"   exact {n_exact}/{len(FIELDS)}")

    worst_overall = max(r["worst_abs_diff"] for r in rows)
    n_all_exact = sum(1 for r in rows if r["n_exact"] == r["n_fields"])
    out = dict(
        purpose="functional verification of analyse() against stored stage-1 rows",
        n_rows_checked=len(rows), n_bitwise_identical=n_all_exact,
        worst_abs_diff=worst_overall,
        verdict="PASS" if worst_overall == 0.0 else "FAIL",
        fields=FIELDS, rows=rows, elapsed_s=time.time() - t0,
        note=("PROVENANCE.md hash for code/analysis.py is stale (recorded "
              "2026-08-08 12:07, before the documented v2 quality-cut change). "
              "This reproduction test replaces it."),
    )
    with open(os.path.join(ITEM1, "results", "part_a_reproduce.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\n{len(rows)} rows checked, {n_all_exact} bitwise identical on all "
          f"{len(FIELDS)} fields, worst |diff| = {worst_overall:.3e}")
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
