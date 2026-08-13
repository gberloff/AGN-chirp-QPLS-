"""End-to-end dry run of Parts F, G and H on a tiny sample, in a scratch
directory, so bugs in the analysis chain surface before the real campaign is
spent rather than after.

Writes nothing to item1/results or item1/figs.  Uses real code paths throughout
the only differences are the sample sizes and the output directory.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

import lib

SCRATCH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "_dryrun")


def setup():
    if os.path.isdir(SCRATCH):
        shutil.rmtree(SCRATCH)
    os.makedirs(os.path.join(SCRATCH, "results"))
    os.makedirs(os.path.join(SCRATCH, "figs"))
    lib.RESULTS = os.path.join(SCRATCH, "results")
    lib.FIGS = os.path.join(SCRATCH, "figs")

    # placeholder Part C / Part D products, structurally identical to the real
    # ones, so the downstream code is exercised exactly as it will be
    json.dump(dict(adopted_cut_mode="mad_scaled", adopted_window_d=50.0,
                   reading="a"),
              open(os.path.join(lib.RESULTS, "part_c.json"), "w"), indent=1)
    json.dump(dict(grid_choice="G2 retained: 160 log nodes, 60 d to T/2.5",
                   grid_choice_freq_uniform=False, reading="3"),
              open(os.path.join(lib.RESULTS, "part_d.json"), "w"), indent=1)
    json.dump(dict(
        adopted_cut_mode="mad_scaled",
        fiducial_corrected=dict(
            DeltaLambda_chirp=dict(
                fap_1e_4=dict(threshold=11.66, ci95=[10.4, 13.1],
                              correction_factor=1.0, biased_value=11.66),
                fap_5e_2=dict(threshold=5.78, ci95=[5.65, 5.90],
                              correction_factor=1.0, biased_value=5.78)),
            Lambda_osc=dict(
                fap_1e_4=dict(threshold=24.0, ci95=[22, 26],
                              correction_factor=1.0, biased_value=24.0),
                fap_5e_2=dict(threshold=16.45, ci95=[15.9, 16.8],
                              correction_factor=1.0, biased_value=16.45))),
        threshold_model=dict(coefficients=dict(intercept=-1.0,
                                               log_baseline=0.35,
                                               duty_cycle=0.5)),
        threshold_model_losc=dict(coefficients=dict(intercept=0.5,
                                                    log_baseline=0.35,
                                                    duty_cycle=0.4)),
        operating_points=dict(primary=1e-4, secondary=5e-2)),
        open(os.path.join(lib.RESULTS, "thresholds.json"), "w"), indent=1)


def main():
    setup()
    import part_f_inject as pf
    pf.OUT = os.path.join(lib.RESULTS, "part_f_per_fit.csv")
    pf.FEAT_OUT = os.path.join(lib.RESULTS, "features.csv")
    pf.DESIGN = os.path.join(lib.RESULTS, "part_f_design.json")
    pf.N_POINTS = 90
    pf.OVERSAMPLE = 60

    print("=== dry run: Part F (180 injections) ===")
    sys.argv = ["dry", "2", "90"]
    pf.main()

    print("\n=== dry run: analyse Part F ===")
    import analyse_part_f
    analyse_part_f.main()

    print("\n=== dry run: Part G ===")
    import part_g_refine as pg
    pg.OUT = os.path.join(lib.RESULTS, "part_g_per_fit.csv")
    pg.PLAN = os.path.join(lib.RESULTS, "part_g_plan.json")
    pg.N_REFINE = 60
    pg.CURVE_LEVELS = 3
    pg.CURVE_REPS = 4
    sys.argv = ["dry", "2"]
    pg.main()

    print("\n=== dry run: analyse Part G ===")
    import analyse_part_g
    analyse_part_g.main()

    print("\n=== dry run: figures ===")
    import item1_figs
    item1_figs.main()
    print(os.listdir(lib.FIGS))

    print("\n=== dry run: Part H ===")
    import part_h_triage
    part_h_triage.main()

    print("\n=== DRY RUN COMPLETE ===")
    print("scratch:", SCRATCH)


if __name__ == "__main__":
    main()
