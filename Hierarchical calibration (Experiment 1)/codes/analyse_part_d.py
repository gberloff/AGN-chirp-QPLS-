"""PART D analysis: which reading holds, and the grid choice.

Reports per variant the fraction of P_hat in the shortest fifth of the searched
range, the fraction below 100 d in absolute terms, and the 95th percentile
threshold with bootstrap intervals.

It also reports one diagnostic the specification does not ask for, because the
It also reports one diagnostic beyond the planned set, because the
is a fifth in LOG P, and a fifth in log P is not a fifth in frequency.  A
periodogram's independent trials are spaced uniformly in frequency, at about
1/T, so the natural null for where the maximum lands is uniform in FREQUENCY,
not uniform in log P.  Each variant therefore has its own predicted pile-up
fraction, computable in advance, and the six predictions differ from each other.
That turns a vague anomaly into a falsifiable test.
"""
import json
import os

import numpy as np
import pandas as pd

import lib
from part_d_grid import VARIANTS, T


def freq_fraction_of_short20(P_min, P_max):
    """Fraction of the frequency range occupied by the shortest fifth in log P."""
    cut = P_min * (P_max / P_min) ** 0.2
    f_lo, f_hi, f_cut = 1.0 / P_max, 1.0 / P_min, 1.0 / cut
    return float((f_hi - f_cut) / (f_hi - f_lo))


def freq_fraction_below(P_min, P_max, P):
    f_lo, f_hi = 1.0 / P_max, 1.0 / P_min
    if P <= P_min:
        return 0.0
    if P >= P_max:
        return 1.0
    return float((f_hi - 1.0 / P) / (f_hi - f_lo))


def main():
    df = pd.read_csv(os.path.join(lib.RESULTS, "part_d_per_fit.csv"),
                     float_precision="round_trip")
    out = dict(n_rows=int(len(df)), variants=[])

    print(f"PART D: {len(df)} rows\n")
    print(f"{'variant':<8} {'nodes':>6} {'Pmin':>6} {'Pmax':>7} {'n':>4} "
          f"{'short20':>9} {'pred(f)':>8} {'<100d':>8} {'pred(f)':>8} "
          f"{'p95':>7} {'p95 CI':>16}")
    print("-" * 104)
    for vi, v in enumerate(VARIANTS):
        s = df[df.config_index == vi]
        if s.empty:
            continue
        x = s["DeltaLambda_chirp"].to_numpy()
        p95, l95, h95 = lib.boot_pct(x, 95)
        p99, l99, h99 = lib.boot_pct(x, 99)
        f_short = float(s["in_short20"].mean())
        f_100 = float(s["below_100d"].mean())
        pred_short = freq_fraction_of_short20(v["P_min_d"], v["P_max_d"])
        pred_100 = freq_fraction_below(v["P_min_d"], v["P_max_d"], 100.0)
        n = len(s)
        se_short = float(np.sqrt(f_short * (1 - f_short) / n))
        rec = dict(
            variant=v["name"], P_n=v["P_n"], P_min_d=v["P_min_d"],
            P_max_d=v["P_max_d"], n=int(n),
            n_grid_nodes=int(s["n_grid_nodes"].iloc[0]),
            frac_short20=f_short, frac_short20_se=se_short,
            frac_short20_ci=[f_short - 1.96 * se_short, f_short + 1.96 * se_short],
            predicted_short20_uniform_in_frequency=pred_short,
            predicted_short20_uniform_in_logP=0.20,
            frac_below_100d=f_100,
            predicted_below_100d_uniform_in_frequency=pred_100,
            dchirp_p95=p95, dchirp_p95_ci=[l95, h95],
            dchirp_p99=p99, dchirp_p99_ci=[l99, h99],
            losc_p95=lib.boot_pct(s["Lambda_osc"].to_numpy(), 95)[0],
            median_runtime_s=float(s["runtime_s"].median()),
        )
        out["variants"].append(rec)
        print(f"{v['name']:<8} {v['P_n']:>6} {v['P_min_d']:>6.0f} "
              f"{v['P_max_d']:>7.0f} {n:>4} "
              f"{f_short:>9.3f} {pred_short:>8.3f} {f_100:>8.3f} "
              f"{pred_100:>8.3f} {p95:>7.3f} [{l95:>6.3f},{h95:>6.3f}]")

    R = {r["variant"]: r for r in out["variants"]}

    node_arm = [R[k] for k in ("G1", "G2", "G3") if k in R]
    lower_arm = [R[k] for k in ("G2", "G4", "G5") if k in R]

    def spread_vs_se(arm):
        f = np.array([a["frac_short20"] for a in arm])
        se = np.array([a["frac_short20_se"] for a in arm])
        return float((f.max() - f.min()) / np.sqrt((se ** 2).sum() / len(se)))

    node_z = spread_vs_se(node_arm) if len(node_arm) == 3 else np.nan
    lower_z = spread_vs_se(lower_arm) if len(lower_arm) == 3 else np.nan
    node_monotone = bool(len(node_arm) == 3 and (
        np.all(np.diff([a["frac_short20"] for a in node_arm]) > 0)
        or np.all(np.diff([a["frac_short20"] for a in node_arm]) < 0)))
    lower_monotone = bool(len(lower_arm) == 3 and (
        np.all(np.diff([a["frac_short20"] for a in lower_arm]) > 0)
        or np.all(np.diff([a["frac_short20"] for a in lower_arm]) < 0)))

    tracks_nodes = bool(node_z > 3.0 and node_monotone)
    tracks_lower = bool(lower_z > 3.0 and lower_monotone)

    f_obs = np.array([r["frac_short20"] for r in out["variants"]])
    f_pred = np.array([r["predicted_short20_uniform_in_frequency"]
                       for r in out["variants"]])
    se = np.array([r["frac_short20_se"] for r in out["variants"]])
    resid_sd = (f_obs - f_pred) / se
    chi2_freq = float(np.sum(resid_sd ** 2))
    chi2_logp = float(np.sum(((f_obs - 0.20) / se) ** 2))

    out["readings"] = dict(
        reading1_tracks_node_count=dict(
            arm=["G1", "G2", "G3"],
            fractions=[a["frac_short20"] for a in node_arm],
            range_in_se=node_z, monotone=node_monotone, holds=tracks_nodes),
        reading2_tracks_lower_limit=dict(
            arm=["G2", "G4", "G5"],
            fractions=[a["frac_short20"] for a in lower_arm],
            range_in_se=lower_z, monotone=lower_monotone, holds=tracks_lower),
        frequency_uniform_diagnostic=dict(
            observed=f_obs.tolist(), predicted=f_pred.tolist(),
            residual_in_se=resid_sd.tolist(),
            chi2_vs_frequency_uniform=chi2_freq,
            chi2_vs_logP_uniform=chi2_logp,
            dof=int(len(f_obs)),
            note="uniform in log P is the null the specification's 20% figure "
                 "assumes, uniform in frequency is the null a periodogram "
                 "actually obeys, since its independent trials are spaced at "
                 "about 1/T in frequency"),
    )

    if tracks_nodes and not tracks_lower:
        reading = "1"
        grid_choice = "frequency-uniform spacing, 160 nodes, 60 d to T/2.5"
        freq_uniform = True
        just = ("Reading 1 holds: the short-fifth fraction tracks node count. "
                "The specification's rule then adopts frequency-uniform node "
                "spacing at the same total node count.")
    elif tracks_lower and not tracks_nodes:
        reading = "2"
        grid_choice = "G2 retained: 160 log nodes, 60 d to T/2.5"
        freq_uniform = False
        just = ("Reading 2 holds: the fraction tracks the lower limit in "
                "absolute days. The specification's rule retains G2.")
    elif tracks_nodes and tracks_lower:
        reading = "1+2"
        grid_choice = "G2 retained: 160 log nodes, 60 d to T/2.5"
        freq_uniform = False
        just = ("Both arms move.  The specification's rule adopts "
                "frequency-uniform spacing only when reading 1 holds alone, "
                "with the readings not separable, G2 is retained, which is the "
                "choice that preserves comparability.")
    else:
        reading = "3"
        grid_choice = "G2 retained: 160 log nodes, 60 d to T/2.5"
        freq_uniform = False
        just = ("Reading 3: the fraction tracks neither node count nor lower "
                "limit in the pre-registered sense.  The specification's rule "
                "retains G2.  Carried as an open assumption.")

    out["reading"] = reading
    out["grid_choice"] = grid_choice
    out["grid_choice_freq_uniform"] = freq_uniform
    out["grid_choice_justification"] = just

    with open(os.path.join(lib.RESULTS, "part_d.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)

    print(f"\nreading 1 (node count, G1/G2/G3): range = {node_z:.2f} se, "
          f"monotone = {node_monotone}  -> holds = {tracks_nodes}")
    print(f"reading 2 (lower limit, G2/G4/G5): range = {lower_z:.2f} se, "
          f"monotone = {lower_monotone}  -> holds = {tracks_lower}")
    print(f"\nfrequency-uniform null: chi2 = {chi2_freq:.1f} on "
          f"{len(f_obs)} variants")
    print(f"log-P-uniform null (the 20% figure): chi2 = {chi2_logp:.1f} on "
          f"{len(f_obs)} variants")
    print(f"\nPRE-REGISTERED READING: {reading}")
    print(f"GRID FOR PARTS E-G: {grid_choice}")
    print(just)
    print("\nwritten results/part_d.json")
    return out


if __name__ == "__main__":
    main()
