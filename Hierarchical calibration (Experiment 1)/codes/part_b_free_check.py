"""PART B: the free check.  No compute.

Threshold sensitivity to background amplitude sigma_DRW/sigma_phot, read
straight out of hcal/results/config_summary.csv and compared against the
+/-7.5% identical-configuration sampling floor measured in stage 1 from twelve
disjoint 400-null blocks.

Decides how carefully the SNR axis must be sampled in Part F.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ITEM1 = os.path.dirname(HERE)
HCAL = os.path.dirname(ITEM1)

SAMPLING_FLOOR_HALFRANGE = 0.075   # stage 1, 12 disjoint 400-null blocks
SAMPLING_FLOOR_SD_P95 = 0.2506     # absolute sd of p95, same source


def main():
    summ = pd.read_csv(os.path.join(HCAL, "results", "config_summary.csv"),
                       float_precision="round_trip")

    fid = summ[summ["config_index"] == 0].iloc[0]
    axis = summ[(summ["axis"] == "sigma_ratio") | (summ["config_index"] == 0)]
    axis = axis.sort_values("sigma_ratio")

    # keep only rows that are fiducial in every OTHER respect
    fid_keys = ["baseline_d", "n_bands", "duty_cycle", "sampling_dt_d", "tau_over_T"]
    mask = np.ones(len(axis), dtype=bool)
    for k in fid_keys:
        mask &= np.isclose(axis[k].to_numpy(dtype=float), float(fid[k]))
    axis = axis[mask]

    out = {"sampling_floor_halfrange": SAMPLING_FLOOR_HALFRANGE,
           "sampling_floor_sd_p95": SAMPLING_FLOOR_SD_P95,
           "levels": [], "source": "hcal/results/config_summary.csv (biased mad_raw cut)"}

    rows = []
    for _, r in axis.iterrows():
        rows.append(dict(
            config_index=int(r["config_index"]), label=str(r["label"]),
            sigma_ratio=float(r["sigma_ratio"]), n_fits=int(r["n_fits"]),
            dchirp_p95=float(r["DeltaLambda_chirp_p95"]),
            dchirp_p95_lo=float(r["DeltaLambda_chirp_p95_lo"]),
            dchirp_p95_hi=float(r["DeltaLambda_chirp_p95_hi"]),
            dchirp_p99=float(r["DeltaLambda_chirp_p99"]),
            dchirp_p99_lo=float(r["DeltaLambda_chirp_p99_lo"]),
            dchirp_p99_hi=float(r["DeltaLambda_chirp_p99_hi"]),
            losc_p95=float(r["Lambda_osc_p95"]),
            losc_p99=float(r["Lambda_osc_p99"]),
            n_epochs=float(r["n_epochs"]),
        ))
    out["levels"] = rows

    def spread(vals):
        vals = np.asarray(vals, dtype=float)
        med = float(np.median(vals))
        return dict(min=float(vals.min()), max=float(vals.max()), median=med,
                    halfrange_frac=float(0.5 * (vals.max() - vals.min()) / med),
                    fullrange_frac=float((vals.max() - vals.min()) / med),
                    cv=float(vals.std(ddof=1) / med),
                    range_in_noise_sd=float((vals.max() - vals.min())
                                            / SAMPLING_FLOOR_SD_P95))

    out["spread_dchirp_p95"] = spread([r["dchirp_p95"] for r in rows])
    out["spread_dchirp_p99"] = spread([r["dchirp_p99"] for r in rows])
    out["spread_losc_p95"] = spread([r["losc_p95"] for r in rows])
    out["spread_losc_p99"] = spread([r["losc_p99"] for r in rows])

    los = [r["dchirp_p95_lo"] for r in rows]
    his = [r["dchirp_p95_hi"] for r in rows]
    out["p95_intervals_all_overlap"] = bool(max(los) <= min(his))
    los9 = [r["dchirp_p99_lo"] for r in rows]
    his9 = [r["dchirp_p99_hi"] for r in rows]
    out["p99_intervals_all_overlap"] = bool(max(los9) <= min(his9))

    h95 = out["spread_dchirp_p95"]["halfrange_frac"]
    if h95 > 2.0 * SAMPLING_FLOOR_HALFRANGE and not out["p95_intervals_all_overlap"]:
        concl = "high"
    elif h95 <= SAMPLING_FLOOR_HALFRANGE and out["p95_intervals_all_overlap"]:
        concl = "low"
    else:
        concl = "inconclusive"
    out["conclusion"] = concl
    out["conclusion_rule"] = (
        "high: half-range > 2x the +/-7.5% sampling floor AND the three bootstrap "
        "intervals do not all overlap, low: half-range <= 7.5% AND all intervals "
        "overlap, otherwise inconclusive")

    with open(os.path.join(ITEM1, "results", "part_b.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print("PART B: threshold sensitivity to background amplitude "
          "sigma_DRW/sigma_phot\n")
    print(f"{'sigma_ratio':>12} {'cfg':>4} {'n':>5} "
          f"{'dChirp p95':>11} {'95% CI':>18} {'dChirp p99':>11} {'95% CI':>18}")
    for r in rows:
        print(f"{r['sigma_ratio']:>12.1f} {r['config_index']:>4d} {r['n_fits']:>5d} "
              f"{r['dchirp_p95']:>11.4f} "
              f"[{r['dchirp_p95_lo']:>7.4f},{r['dchirp_p95_hi']:>7.4f}] "
              f"{r['dchirp_p99']:>11.4f} "
              f"[{r['dchirp_p99_lo']:>7.4f},{r['dchirp_p99_hi']:>7.4f}]")
    s = out["spread_dchirp_p95"]
    print(f"\nDeltaLambda_chirp p95: half-range {100*s['halfrange_frac']:.2f}% "
          f"of median, CV {100*s['cv']:.2f}%, "
          f"full range = {s['range_in_noise_sd']:.2f} sampling sd")
    s9 = out["spread_dchirp_p99"]
    print(f"DeltaLambda_chirp p99: half-range {100*s9['halfrange_frac']:.2f}% "
          f"of median, CV {100*s9['cv']:.2f}%, "
          f"full range = {s9['range_in_noise_sd']:.2f} sampling sd")
    print(f"sampling floor (identical configuration, n=400): "
          f"+/-{100*SAMPLING_FLOOR_HALFRANGE:.1f}%")
    print(f"all three p95 bootstrap intervals overlap: "
          f"{out['p95_intervals_all_overlap']}")
    print(f"all three p99 bootstrap intervals overlap: "
          f"{out['p99_intervals_all_overlap']}")
    print(f"\nCONCLUSION: threshold sensitivity to background amplitude = "
          f"{concl.upper()}")


if __name__ == "__main__":
    main()
