"""The thresholds Parts F and G apply.

Built from stage 1's null ensembles, rescaled by Part C's correction factor,
then modelled over baseline and duty cycle and validated
leave-one-configuration-out.

Every threshold here is quoted with its false-alarm probability.  Nothing in
this module ever sees an injection.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import genpareto

import lib

FAP_PRIMARY = 1e-4
FAP_SECONDARY = 5e-2


def gpd_extrapolate(x, fap, u_pct=90.0, n_boot=400, seed=11):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    u = float(np.percentile(x, u_pct))
    exc = x[x > u] - u
    zeta = exc.size / x.size
    xi, _, beta = genpareto.fit(exc, floc=0.0)

    def thr(xi, beta):
        p = fap / zeta
        if abs(xi) < 1e-8:
            return u - beta * np.log(p)
        return u + beta / xi * (p ** (-xi) - 1.0)

    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        e = rng.choice(exc, size=exc.size, replace=True)
        try:
            xb, _, bb = genpareto.fit(e, floc=0.0)
            bs.append(thr(xb, bb))
        except Exception:
            pass
    bs = np.asarray([b for b in bs if np.isfinite(b)])
    return dict(threshold=float(thr(xi, beta)),
                ci95=[float(np.percentile(bs, 2.5)),
                      float(np.percentile(bs, 97.5))] if bs.size else
                [np.nan, np.nan],
                u=u, n_exceedances=int(exc.size), zeta=float(zeta),
                shape_xi=float(xi), scale_beta=float(beta),
                n=int(x.size), fap=fap,
                supported_directly=bool(x.size >= 1.0 / fap))


def paired_correction(x_raw, x_ad, fap, n_boot=400, seed=23):
    """Correction factor for a threshold at `fap`, from paired Part C arms.

    The two arms scored the SAME light curves, so the bootstrap resamples
    curves and recomputes both arms on the same resample. The realisation
    noise that dominates a 300-null percentile then cancels, which an
    unpaired comparison would leave in.

    Above fap = 0.01 the empirical percentile is used directly, below it the
    threshold is beyond direct support at n = 300 and a GPD tail fit supplies
    the ratio.  This is the ONLY place a correction factor is computed, so the
    Part C report and the thresholds Parts F and G apply cannot disagree.
    """
    x_raw = np.asarray(x_raw, float)
    x_ad = np.asarray(x_ad, float)
    assert x_raw.size == x_ad.size

    def thr_of(x):
        if fap >= 0.01:
            return float(np.percentile(x, 100 * (1 - fap)))
        return gpd_extrapolate(x, fap, n_boot=0)["threshold"]

    f0 = thr_of(x_ad) / thr_of(x_raw)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, x_raw.size, x_raw.size)
        try:
            v = thr_of(x_ad[idx]) / thr_of(x_raw[idx])
            if np.isfinite(v):
                bs.append(v)
        except Exception:
            pass
    bs = np.asarray(bs)
    ci = ([float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
          if bs.size else [np.nan, np.nan])
    return dict(factor=float(f0), ci95=ci, fap=fap, n_pairs=int(x_raw.size),
                method=("empirical percentile ratio, paired bootstrap"
                        if fap >= 0.01 else
                        "GPD tail-fit ratio, paired bootstrap"),
                consistent_with_unity=bool(ci[0] <= 1.0 <= ci[1])
                if np.isfinite(ci[0]) else None)


def grid_correction():
    """Paired threshold ratio for the Part D grid change, G2 -> G7.

    Returns unity factors, flagged, if Part D retained G2, then no grid
    change happened and no correction is due.
    """
    pd_path = os.path.join(lib.RESULTS, "part_d.json")
    if not os.path.exists(pd_path):
        raise FileNotFoundError("Part D must run before thresholds: the grid "
                                "choice sets the look-elsewhere volume")
    with open(pd_path, encoding="utf-8") as f:
        pdj = json.load(f)
    unity = {n: {k: dict(factor=1.0, ci95=[1.0, 1.0], applied=False,
                         reason="Part D retained G2, no grid change")
                 for k in ("fap_1e_4", "fap_5e_2")}
             for n in ("DeltaLambda_chirp", "Lambda_osc")}
    if not pdj.get("grid_choice_freq_uniform"):
        return unity
    d = pd.read_csv(os.path.join(lib.RESULTS, "part_d_per_fit.csv"),
                    float_precision="round_trip")
    g2 = d[d.variant == "G2"].sort_values("realisation")
    g7 = d[d.variant == "G7"].sort_values("realisation")
    if g7.empty:
        raise RuntimeError("Part D adopted frequency-uniform spacing but the "
                           "G7 ensemble is missing, the corrected thresholds "
                           "would describe the wrong grid")
    assert (g2["realisation"].to_numpy() == g7["realisation"].to_numpy()).all()
    out = {}
    for name in ("DeltaLambda_chirp", "Lambda_osc"):
        x2, x7 = g2[name].to_numpy(), g7[name].to_numpy()
        out[name] = {}
        for key, fap in (("fap_1e_4", FAP_PRIMARY), ("fap_5e_2", FAP_SECONDARY)):
            r = paired_correction(x2, x7, fap)
            r["applied"] = True
            r["reason"] = ("Part D adopted frequency-uniform spacing, ratio of "
                           "the frequency-uniform null to the log-spaced null "
                           "on the same 300 realisations")
            out[name][key] = r
    return out


def build(verbose=True):
    """Return the threshold table, writing results/thresholds.json."""
    per_fit = pd.read_csv(os.path.join(lib.HCAL, "results", "stage1_per_fit.csv"),
                          float_precision="round_trip")
    summ = pd.read_csv(os.path.join(lib.HCAL, "results", "config_summary.csv"),
                       float_precision="round_trip")
    partc = json.load(open(os.path.join(lib.RESULTS, "part_c.json"),
                           encoding="utf-8"))
    adopted = partc["adopted_cut_mode"]

    tail = per_fit[per_fit.config_index == 99]
    raw = {}
    for name, col in (("DeltaLambda_chirp", "DeltaLambda_chirp"),
                      ("Lambda_osc", "Lambda_osc")):
        x = tail[col].to_numpy()
        raw[name] = dict(
            fap_1e_4=gpd_extrapolate(x, FAP_PRIMARY),
            fap_5e_2=dict(threshold=float(np.percentile(x, 95)),
                          ci95=list(lib.boot_pct(x, 95)[1:]), fap=FAP_SECONDARY,
                          n=int(x.size), supported_directly=True,
                          method="empirical 95th percentile"))

    pc = pd.read_csv(os.path.join(lib.RESULTS, "part_c_per_fit.csv"),
                     float_precision="round_trip")
    ai_ad = {"mad_scaled": 1, "sigma_clip": 2}[adopted]
    fid_raw = pc[(pc.config_index == 0) & (pc.arm_index == 0)].sort_values(
        "realisation")
    fid_ad = pc[(pc.config_index == 0) & (pc.arm_index == ai_ad)].sort_values(
        "realisation")
    assert (fid_raw["realisation"].to_numpy()
            == fid_ad["realisation"].to_numpy()).all(), "arms are not paired"
    corr = {}
    for name in ("DeltaLambda_chirp", "Lambda_osc"):
        xr, xa = fid_raw[name].to_numpy(), fid_ad[name].to_numpy()
        corr[name] = dict(
            fap_1e_4=paired_correction(xr, xa, FAP_PRIMARY),
            fap_5e_2=paired_correction(xr, xa, FAP_SECONDARY),
            n=int(len(fid_raw)),
            note="paired: the same 300 light curves scored under both cut modes")
    # Part D adopted frequency-uniform node spacing.  That changes the
    # look-elsewhere volume, and every threshold above was measured on a
    # LOG-spaced grid, so it does not apply to the new grid unmodified.  Part
    # D's G7 variant is frequency-uniform run on the SAME 300 realisations as
    # G2, which makes the ratio paired.  Without this the campaign would apply
    # a G2 threshold to a grid that is not G2.
    grid_corr = grid_correction()

    corrected = {}
    for name in ("DeltaLambda_chirp", "Lambda_osc"):
        gc = grid_corr[name]
        c99 = corr[name]["fap_1e_4"]["factor"] * gc["fap_1e_4"]["factor"]
        c95 = corr[name]["fap_5e_2"]["factor"] * gc["fap_5e_2"]["factor"]
        corrected[name] = {
            "fap_1e_4": dict(
                threshold=float(raw[name]["fap_1e_4"]["threshold"] * c99),
                ci95=[float(v * c99) for v in raw[name]["fap_1e_4"]["ci95"]],
                biased_value=raw[name]["fap_1e_4"]["threshold"],
                correction_factor=c99,
                cut_mode_factor=corr[name]["fap_1e_4"]["factor"],
                grid_factor=gc["fap_1e_4"]["factor"],
                correction_ci95=corr[name]["fap_1e_4"]["ci95"],
                correction_consistent_with_unity=corr[name]["fap_1e_4"]
                ["consistent_with_unity"],
                fap=FAP_PRIMARY, supported_directly=False,
                method="stage-1 GPD extrapolation on 5000 nulls, rescaled by "
                       "Part C's paired correction factor at the same "
                       "operating point"),
            "fap_5e_2": dict(
                threshold=float(raw[name]["fap_5e_2"]["threshold"] * c95),
                ci95=[float(v * c95) for v in raw[name]["fap_5e_2"]["ci95"]],
                biased_value=raw[name]["fap_5e_2"]["threshold"],
                correction_factor=c95,
                cut_mode_factor=corr[name]["fap_5e_2"]["factor"],
                grid_factor=gc["fap_5e_2"]["factor"],
                correction_ci95=corr[name]["fap_5e_2"]["ci95"],
                correction_consistent_with_unity=corr[name]["fap_5e_2"]
                ["consistent_with_unity"],
                fap=FAP_SECONDARY, supported_directly=True,
                method="stage-1 empirical p95 on 5000 nulls, rescaled by "
                       "Part C's paired correction factor at the same "
                       "operating point"),
        }

    syn = summ[summ.is_real_cadence == 0].copy()
    model = fit_threshold_model(syn, verbose=verbose)
    model_losc = fit_threshold_model(syn, col="Lambda_osc_p99", verbose=False)

    out = dict(
        adopted_cut_mode=adopted,
        fiducial_raw=raw, correction_factors=corr,
        grid_correction_factors=grid_corr, fiducial_corrected=corrected,
        threshold_model=model,
        threshold_model_losc=model_losc,
        operating_points=dict(primary=FAP_PRIMARY, secondary=FAP_SECONDARY),
        note="Parts F and G hold baseline at 2000 d and duty cycle at the "
             "fiducial 0.657, so the threshold model is evaluated at a single "
             "point there and reduces to the corrected fiducial threshold. It "
             "is fitted and validated because Part G's one-dimensional curves "
             "vary duty cycle.",
    )
    with open(os.path.join(lib.RESULTS, "thresholds.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    if verbose:
        print(f"\nadopted cut mode: {adopted}")
        for name in ("DeltaLambda_chirp", "Lambda_osc"):
            for k in ("fap_1e_4", "fap_5e_2"):
                c = corrected[name][k]
                print(f"  {name:<18} {k:<9} {c['threshold']:8.4f}  "
                      f"(biased {c['biased_value']:8.4f}, cut-mode "
                      f"x{c['cut_mode_factor']:.4f}, grid "
                      f"x{c['grid_factor']:.4f}, total "
                      f"x{c['correction_factor']:.4f})")
    return out


def fit_threshold_model(syn, col="DeltaLambda_chirp_p99", verbose=True):
    """log p99 ~ a + b*log(baseline) + c*duty, validated leave-one-out.

    Also reports whether epoch count adds anything, because Parts F and G span
    a far wider epoch range than stage 1 did and the specification's two axes
    were chosen on stage-1 evidence alone.
    """
    y = np.log(syn[col].to_numpy())
    X = np.column_stack([np.ones(len(syn)),
                         np.log(syn["baseline_d"].to_numpy()),
                         syn["duty_cycle"].to_numpy()])
    Xn = np.column_stack([X, np.log(syn["n_epochs"].to_numpy())])
    labels = syn["label"].tolist()

    def loo(X, y):
        pred = np.empty(len(y))
        for k in range(len(y)):
            m = np.ones(len(y), bool)
            m[k] = False
            beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            pred[k] = X[k] @ beta
        return pred

    p_bd = loo(X, y)
    p_bdn = loo(Xn, y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    rows = []
    for k in range(len(y)):
        rows.append(dict(label=labels[k], config_index=int(syn["config_index"].iloc[k]),
                         observed_p99=float(np.exp(y[k])),
                         predicted_p99=float(np.exp(p_bd[k])),
                         frac_error=float(np.exp(p_bd[k] - y[k]) - 1.0),
                         observed_p99_lo=float(syn[col + "_lo"].iloc[k]),
                         observed_p99_hi=float(syn[col + "_hi"].iloc[k]),
                         inside_ci=bool(syn[col + "_lo"].iloc[k]
                                        <= np.exp(p_bd[k]) <=
                                        syn[col + "_hi"].iloc[k])))
    n_inside = sum(r["inside_ci"] for r in rows)
    rmse_bd = float(np.sqrt(np.mean((p_bd - y) ** 2)))
    rmse_bdn = float(np.sqrt(np.mean((p_bdn - y) ** 2)))
    med_abs = float(np.median([abs(r["frac_error"]) for r in rows]))
    validated = bool(n_inside >= 0.8 * len(rows))

    if verbose:
        print(f"\nthreshold model, leave-one-configuration-out over "
              f"{len(rows)} synthetic configurations:")
        for r in rows:
            print(f"  {r['label']:<22} observed p99 {r['observed_p99']:6.3f} "
                  f"[{r['observed_p99_lo']:5.3f},{r['observed_p99_hi']:6.3f}]  "
                  f"predicted {r['predicted_p99']:6.3f}  "
                  f"{100*r['frac_error']:+6.1f}%  "
                  f"{'inside' if r['inside_ci'] else 'OUTSIDE'}")
        print(f"  {n_inside}/{len(rows)} predictions inside the observed "
              f"bootstrap interval, median |fractional error| "
              f"{100*med_abs:.1f}%")
        print(f"  LOO RMSE in log p99: baseline+duty {rmse_bd:.4f}, "
              f"+log n_epochs {rmse_bdn:.4f}")

    return dict(
        form=f"log({col}) ~ 1 + log(baseline_d) + duty_cycle",
        column=col,
        coefficients=dict(intercept=float(beta[0]), log_baseline=float(beta[1]),
                          duty_cycle=float(beta[2])),
        loo=rows, n_inside_ci=int(n_inside), n_configs=int(len(rows)),
        median_abs_frac_error=med_abs,
        loo_rmse_log=rmse_bd, loo_rmse_log_with_n_epochs=rmse_bdn,
        epoch_count_helps=bool(rmse_bdn < 0.95 * rmse_bd),
        validated=validated,
        fallback_if_failed="per-configuration calibration, Parts F and G would "
                           "be reduced to the configurations that have their own "
                           "null ensemble",
    )


def duty_ratio_per_configuration(duty, summ=None):
    """The Section 8.2 FALLBACK: per-configuration calibration.

    The threshold model failed leave-one-out validation, and it failed on its
    own axis: both duty-cycle configurations are predicted about 12% high.
    So the duty dependence is taken from stage 1's directly measured p99 at
    each duty level instead of from the fitted surface.

    This costs Parts F and G almost nothing, because they hold duty at the
    fiducial, where a 5000-null measurement exists, the only excursions are
    Part G's three duty levels, and each has its own 400-null stage-1
    ensemble.  The price is that those two thresholds carry 400-null sampling
    error rather than a smooth model, which is the honest position.
    """
    if summ is None:
        summ = pd.read_csv(os.path.join(lib.HCAL, "results",
                                        "config_summary.csv"),
                           float_precision="round_trip")
    syn = summ[summ.is_real_cadence == 0]
    fid = syn[syn.label == "fiducial"].iloc[0]
    # Match on baseline as well as duty.  Stage 1 has a T=4000/duty=0.4 CORNER
    # configuration that would otherwise be picked for duty=0.4 and would
    # confound the baseline effect: the strongest threshold axis there is
    # ith the duty effect being measured.
    same_T = syn[np.isclose(syn["baseline_d"].to_numpy(dtype=float),
                            float(fid["baseline_d"]))]
    cand = same_T.iloc[(same_T["duty_cycle"] - duty).abs().argsort()[:1]].iloc[0]
    if abs(float(cand["duty_cycle"]) - duty) > 1e-3:
        raise ValueError(
            f"no stage-1 configuration at duty {duty:.4f}, per-configuration "
            f"calibration cannot be done and the model is not validated")
    return dict(
        duty=float(cand["duty_cycle"]), label=str(cand["label"]),
        ratio_dchirp=float(cand["DeltaLambda_chirp_p99"]
                           / fid["DeltaLambda_chirp_p99"]),
        ratio_losc=float(cand["Lambda_osc_p99"] / fid["Lambda_osc_p99"]),
        n_nulls=int(cand["n_fits"]),
        source="stage-1 per-configuration p99, mad_raw, used as a RATIO so the "
               "cut-mode correction cancels")


def apply_correction_to_model(model, corr_factor):
    """The model is fitted on biased (mad_raw) p99 values, the corrected model
    is the same shape with the intercept shifted by log(correction factor)."""
    m = dict(model)
    m["coefficients"] = dict(model["coefficients"])
    m["coefficients"]["intercept"] += float(np.log(corr_factor))
    m["corrected_by"] = corr_factor
    return m


def predict_threshold(model, baseline_d, duty_cycle):
    c = model["coefficients"]
    return float(np.exp(c["intercept"] + c["log_baseline"] * np.log(baseline_d)
                        + c["duty_cycle"] * duty_cycle))


if __name__ == "__main__":
    build()
