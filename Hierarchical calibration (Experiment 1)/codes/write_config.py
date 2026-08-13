"""Write item1/config.json.  Every constant this specification uses.

Two fields are written null here and filled in by the part that decides them:
`cut_mode_adopted` (Part C) and `grid_choice` (Part D).  They are never
guessed ahead of the measurement.
"""
import json
import os
import platform
import sys
from datetime import datetime, timezone

import lib


def main():
    import numpy, scipy, pandas, joblib, celerite2, matplotlib, sklearn, astropy

    cfg = dict(
        experiment="item 1 of the periodicity note: full hierarchical calibration "
                   "(injection-recovery and the selection function)",
        spec_version=lib.SPEC_VERSION,
        written=datetime.now(timezone.utc).isoformat(),
        working_folder="~/AGN_chirp_experiment/hcal/item1/",
        scope="item 1 only, item 2 (flexible stochastic nulls) is out of scope, "
              "except optional Part H which runs last",

        seed_rule="seed = 20260811 + 1000000*part_index + 1000*config_index + i",
        seed_base=lib.SEED_BASE,
        part_index=lib.PART_INDEX,
        seed_collision_note=(
            "Generator.null_lightcurve draws the latent path from `seed` and the "
            "measurement noise from `seed + 500000`.  config_index is kept below "
            "100 and i below 1000 per slot, so 1000*config_index + i < 100000 and "
            "no noise stream can ever coincide with another fit's path stream. "
            "Where a part needs more than 1000 realisations under one label it is "
            "given several 1000-wide config_index slots, as stage 1b did."),

        inherited=dict(
            cadence_csv="../../cadence.csv (765 epochs, 1996.448 d), shared by "
                        "every run, never regenerated",
            cadence_seed=20260807,
            analyse="hcal/code/analysis.py analyse(), verified bitwise against "
                    "stored stage-1 rows in Part A (7/7, worst |diff| 0.000e+00) "
                    "and again through the item1 cut-mode extension in the "
                    "pre-flight gate (5/5, 0.000e+00)",
            biased_thresholds=dict(
                source="hcal/results/tail_fit.json, mad_raw cut",
                fap_1e_4=11.658837680317514,
                fap_1e_4_ci95=[10.41457666774615, 13.144725056818949],
                fap_1e_3=9.650587382203211,
                fap_5e_2=5.7766658,
                note="all carry the mad_raw bias, Part C measures the correction"),
            sampling_floor_p95_halfrange=0.075,
            sampling_floor_p95_sd=0.2506,
        ),

        detector=dict(
            null_family="drw", n_harmonics=2,
            sigma_bounds_mag=[0.001, 0.5], tau_bounds_d=[5.0, 20000.0],
            restarts=3, starts=[[0.06, 320.0], [0.02, 60.0], [0.15, 3000.0]],
            boundary_tol_frac=1e-3, require_positive_frequency=True,
            eta_grid=lib.ETA_GRID,
            scoring="Lambda_osc = max over the (P, eta) grid, Lambda_per0 = max "
                    "over the eta = 0 row, DeltaLambda_chirp = the difference"),

        quality_cuts=dict(
            applied_where="inside analysis.analyse: identically on real data, "
                          "nulls and injections, without exception",
            implementation="lib.install_cut_modes() replaces the module-level "
                           "function analyse calls internally.  analysis.py is "
                           "not edited: it is outside item1/.  With cut_mode = "
                           "'mad_raw' the dispatcher delegates to the original "
                           "function, and the gate proves the path is bitwise "
                           "identical.",
            catflags=True, magerr_max_mag=0.20, min_epochs=100, n_sigma=5.0,
            modes=dict(
                mad_raw="|resid from a running median| > 5 * MAD.  As built. "
                        "Unscaled MAD is 0.6745 sigma for a Gaussian, so this is "
                        "a 3.37 sigma clip.  The known bias.",
                mad_scaled="|resid from a running median| > 5 * 1.4826 * MAD. "
                           "A true 5 sigma clip.",
                sigma_clip="|y - median(y_band)| > 5 * 1.4826 * MAD(y_band). "
                           "No running median at all, which is what separates "
                           "'the scaling was wrong' from 'the running median was "
                           "wrong'."),
            running_median_window_d=[50.0, 150.0],
            cut_mode_adopted=None,
            running_median_window_adopted_d=None,
        ),

        grid=dict(
            control="G2: 160 log-spaced period nodes, 60 d to T/2.5, 25 eta nodes "
                    "= 4000 grid nodes.  The grid every previous run used.",
            variants=dict(
                G1=dict(P_n=80, P_min_d=60.0, P_max="T/2.5"),
                G2=dict(P_n=160, P_min_d=60.0, P_max="T/2.5"),
                G3=dict(P_n=320, P_min_d=60.0, P_max="T/2.5"),
                G4=dict(P_n=160, P_min_d=30.0, P_max="T/2.5"),
                G5=dict(P_n=160, P_min_d=120.0, P_max="T/2.5"),
                G6=dict(P_n=160, P_min_d=60.0, P_max="T/5")),
            grid_choice=None,
            grid_choice_justification=None,
            never_reduce="the search grid is never shrunk for compute reasons: "
                         "it would change the look-elsewhere volume and break "
                         "comparability with every previous run"),

        snr_convention=dict(
            formula="SNR = A_1 * sqrt(N_epochs) / sigma_eff",
            sigma_eff="quadrature sum of the median per-epoch photometric error "
                      "and the background's realised in-window RMS: "
                      "sqrt(median(sigma_phot)^2 + var(latent DRW path in window))",
            realised="the RMS is that of the actual drawn path, not the "
                     "configuration's nominal target, so the requested SNR is the "
                     "achieved SNR for that realisation",
            N_epochs="epochs presented to analyse(), before the quality cuts",
            meaning="integrated signal-to-noise, not per-epoch depth",
            A_1="amplitude of the first harmonic, in magnitudes, identical in "
                "every band",
            note="the selection function is conditional on this convention, a "
                 "different convention moves the contours without changing the "
                 "physics"),

        operating_points=dict(
            primary_fap=1e-4,
            secondary_fap=5e-2,
            rule="no efficiency appears in any output without its false-alarm "
                 "probability attached"),

        injection_space=dict(
            baseline_T_d=2000.0,
            duty_cycle_fixed=lib.FIDUCIAL_DUTY,
            duty_cycle_note="held at the fiducial 0.657 in Parts F and G, varied "
                            "only in Part G's one-dimensional curves.  A design "
                            "decision: interactions involving duty cycle are "
                            "untested.",
            axes=dict(
                N_cyc=dict(range=[3, 40], sampling="log", derived="P = T / N_cyc"),
                eta_x=dict(range=[0, 8], sampling="linear",
                           derived="eta = eta_x / N_cyc"),
                SNR=dict(range=[3, 50], sampling="log"),
                tau_over_P=dict(range=[0.1, 10], sampling="log",
                                derived="tau = tau_over_P * P"),
                a2_over_a1=dict(range=[0, 1], sampling="linear"),
                samples_per_cycle=dict(range=[2, 60], sampling="log",
                                       derived="dt = P / samples_per_cycle"),
                band_structure=dict(levels=["1 band", "2 bands free phase"],
                                    sampling="discrete")),
            rejection_rules=[
                "|eta| > 0.55 (the eta grid runs to +/-0.60, an injection at the "
                "edge cannot be recovered without censoring)",
                "f(t) <= 0 anywhere in the window",
                "the cadence would present fewer than min_epochs = 100 epochs to "
                "analyse(), where the pipeline returns insufficient_data and no "
                "score exists by construction, see DEVIATIONS.md D3"],
        ),

        recovery=dict(
            trigger="DeltaLambda_chirp > threshold",
            correct="trigger and |P_hat - P_true| / P_true < 0.02",
            alias="trigger and P_hat within 2% of P_true/2 or 2*P_true, counted "
                  "separately, never folded into success or failure",
            chirp="correct recovery, sign(eta_hat) == sign(eta_true), and "
                  "|eta_hat| < 0.58 so the estimate is not censored at the grid "
                  "edge",
            also_recorded="the same four using Lambda_osc against its own "
                          "threshold, so 'found a periodicity' and 'found a "
                          "chirp' stay separable"),

        thermal=dict(
            n_jobs=2, n_jobs_max=4,
            blas_threads=1,
            pause_between_batches_s=5.0,
            batch_size=30,
            batch_size_note="the 5 s inter-batch pause is per joblib batch, a "
                            "batch of 2*n_jobs would pause every 4 fits and cost "
                            "about 40% of throughput, so batches are 30 fits",
            pause_long_s=90.0, pause_every_fits=300,
            heartbeat_every_fits=10,
            flush="one row per fit, fsync'd, so a kill leaves whole rows",
            history="four thermal shutdowns, the v1 batch died at n_jobs = 4"),

        environment=dict(
            python=sys.version, platform=platform.platform(),
            numpy=numpy.__version__, scipy=scipy.__version__,
            pandas=pandas.__version__, joblib=joblib.__version__,
            celerite2=celerite2.__version__, matplotlib=matplotlib.__version__,
            sklearn=sklearn.__version__, astropy=astropy.__version__),
    )

    path = os.path.join(lib.ITEM1, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    print("written", path)


if __name__ == "__main__":
    main()
