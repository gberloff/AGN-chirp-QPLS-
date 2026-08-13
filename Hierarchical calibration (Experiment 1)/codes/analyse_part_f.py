"""PART F analysis: the response surface, the axis ranking, and eta_x.

Every efficiency here carries its false-alarm probability.  The primary
operating point is FAP 1e-4, the secondary 5e-2 is reported alongside for the
headline rates.
"""
import json
import os

import numpy as np
import pandas as pd

import lib
import surface as S

FAP = 1e-4


def outcome_summary(df, label):
    n = len(df)
    return dict(
        label=label, n=int(n), fap=FAP,
        trigger=float(df["trigger"].mean()),
        correct=float(df["correct"].mean()),
        alias=float(df["alias"].mean()),
        chirp=float(df["chirp"].mean()),
        losc_trigger=float(df["losc_trigger"].mean()),
        losc_correct=float(df["losc_correct"].mean()),
        losc_alias=float(df["losc_alias"].mean()),
        losc_chirp=float(df["losc_chirp"].mean()),
        insufficient_data=float(df["insufficient_data"].mean()),
        alias_share_of_triggers=float(df["alias"].sum()
                                      / max(df["trigger"].sum(), 1)),
    )


def eta_x_profile(df, edges=None):
    """Chirp recovery against eta_x, the axis that should dominate.

    Below eta_x ~ 1 the frequency drift across the campaign is less than one
    frequency resolution element, so no method can detect it in principle.
    """
    if edges is None:
        edges = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    rows = []
    idx = np.digitize(df["eta_x"].to_numpy(), edges) - 1
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        sub = df[m]
        # restrict to curves that were detected at all, so the eta_x effect is
        # not confounded with simply being too faint to trigger
        det = sub[sub["correct"] == 1]
        k = int(sub["chirp"].sum())
        p = k / n
        se = np.sqrt(max(p * (1 - p), 1e-12) / n)
        rows.append(dict(
            lo=float(edges[b]), hi=float(edges[b + 1]), n=n,
            mean_eta_x=float(sub["eta_x"].mean()),
            chirp=p, chirp_lo=max(0.0, p - 1.96 * se),
            chirp_hi=min(1.0, p + 1.96 * se),
            trigger=float(sub["trigger"].mean()),
            correct=float(sub["correct"].mean()),
            chirp_given_correct=(float(det["chirp"].mean())
                                 if len(det) else np.nan),
            n_correct=int(len(det)),
        ))
    return rows


def main():
    df = pd.read_csv(os.path.join(lib.RESULTS, "part_f_per_fit.csv"),
                     float_precision="round_trip")
    th = json.load(open(os.path.join(lib.RESULTS, "thresholds.json"),
                        encoding="utf-8"))
    print(f"PART F: {len(df)} injections, FAP {FAP:g}\n")

    out = dict(n=int(len(df)), fap=FAP,
               threshold_dchirp=float(df["threshold_used"].iloc[0]),
               threshold_losc=float(df["threshold_losc"].iloc[0]),
               adopted_cut_mode=th["adopted_cut_mode"])

    out["overall"] = outcome_summary(df, "all")
    out["by_band_structure"] = [
        outcome_summary(df[df.band_structure == bs], f"band_structure={bs}")
        for bs in sorted(df.band_structure.unique())]
    print("Recovery fractions at FAP 1e-4 (all four definitions, always):")
    for r in [out["overall"]] + out["by_band_structure"]:
        print(f"  {r['label']:<20} n={r['n']:<5} trigger {r['trigger']:.3f}  "
              f"correct {r['correct']:.3f}  alias {r['alias']:.3f}  "
              f"chirp {r['chirp']:.3f}   (alias is "
              f"{100*r['alias_share_of_triggers']:.1f}% of triggers)")
    print("  same, scoring on Lambda_osc against its own threshold:")
    for r in [out["overall"]] + out["by_band_structure"]:
        print(f"  {r['label']:<20} n={r['n']:<5} "
              f"trigger {r['losc_trigger']:.3f}  correct {r['losc_correct']:.3f}"
              f"  alias {r['losc_alias']:.3f}  chirp {r['losc_chirp']:.3f}")

    y = df["chirp"].to_numpy()
    surf = S.Surface(fap=FAP, outcome="chirp",
                     cut_mode=th["adopted_cut_mode"])
    surf.fit(df, y)
    coefs = surf.coefficients()
    rank = surf.deviance_ranking(df, y)
    out["logistic_chirp"] = dict(coefficients=coefs, ranking=rank)

    print(f"\nLogistic response surface for chirp recovery at FAP 1e-4:")
    print(f"  pseudo-R^2 {rank['pseudo_r2']:.4f}, deviance explained "
          f"{rank['deviance_explained']:.1f} of {rank['deviance_null']:.1f}")
    print("  axis ranking by unique deviance explained "
          "(main effect and every interaction it enters):")
    for r in rank["ranking"]:
        print(f"    {r['axis']:<20} {r['deviance_increase']:9.1f}  "
              f"{100*r['frac_of_explained']:5.1f}%")
    print("  main effects (z-scored scale):")
    for c in coefs:
        if ":" not in c["term"] and c["term"] != "(intercept)":
            print(f"    {c['term']:<20} {c['coef']:+8.3f} +/- {c['se']:.3f} "
                  f"(z = {c['z']:+.1f})")
    strong = [c for c in coefs if ":" in c["term"] and abs(c["z"]) > 3]
    strong.sort(key=lambda c: -abs(c["z"]))
    print(f"  interactions with |z| > 3: {len(strong)} of "
          f"{sum(1 for c in coefs if ':' in c['term'])}")
    for c in strong[:8]:
        print(f"    {c['term']:<28} {c['coef']:+8.3f} +/- {c['se']:.3f} "
              f"(z = {c['z']:+.1f})")

    # also fit the trigger outcome, so "found a periodicity" stays separable
    surf_t = S.Surface(fap=FAP, outcome="trigger")
    surf_t.fit(df, df["trigger"].to_numpy())
    out["logistic_trigger"] = dict(
        ranking=surf_t.deviance_ranking(df, df["trigger"].to_numpy()))

    prof = eta_x_profile(df)
    out["eta_x_profile"] = prof
    print("\neta_x: chirp recovery at FAP 1e-4")
    print(f"  {'eta_x bin':<14} {'n':>5} {'trigger':>8} {'correct':>8} "
          f"{'chirp':>8} {'95% CI':>16} {'chirp|correct':>14}")
    for r in prof:
        print(f"  [{r['lo']:>4.2f},{r['hi']:>4.2f}) {r['n']:>5} "
              f"{r['trigger']:>8.3f} {r['correct']:>8.3f} {r['chirp']:>8.3f} "
              f"[{r['chirp_lo']:.3f},{r['chirp_hi']:.3f}] "
              f"{r['chirp_given_correct']:>14.3f}")

    lo = df[df.eta_x < 1.0]
    hi = df[(df.eta_x >= 1.0) & (df.eta_x < 3.0)]
    lo_c = lo[lo.correct == 1]
    hi_c = hi[hi.correct == 1]
    out["eta_x_transition"] = dict(
        below_1=dict(n=int(len(lo)), chirp=float(lo["chirp"].mean()),
                     n_correct=int(len(lo_c)),
                     chirp_given_correct=float(lo_c["chirp"].mean())
                     if len(lo_c) else np.nan),
        between_1_and_3=dict(n=int(len(hi)), chirp=float(hi["chirp"].mean()),
                             n_correct=int(len(hi_c)),
                             chirp_given_correct=float(hi_c["chirp"].mean())
                             if len(hi_c) else np.nan),
        expectation="below about eta_x = 1 the drift spans less than one "
                    "frequency resolution element and no method can detect it "
                    "in principle, a sharp transition near 1 is expected",
    )
    b = out["eta_x_transition"]
    print(f"\n  eta_x < 1      : chirp {b['below_1']['chirp']:.3f} "
          f"(n={b['below_1']['n']}), given correct-P "
          f"{b['below_1']['chirp_given_correct']:.3f}")
    print(f"  1 <= eta_x < 3 : chirp {b['between_1_and_3']['chirp']:.3f} "
          f"(n={b['between_1_and_3']['n']}), given correct-P "
          f"{b['between_1_and_3']['chirp_given_correct']:.3f}")

    out["cost"] = dict(
        median_fit_s=float(df["runtime_s"].median()),
        median_feature_s=float(df["feature_runtime_s"].median()),
        mean_fit_s=float(df["runtime_s"].mean()),
        mean_feature_s=float(df["feature_runtime_s"].mean()),
        feature_to_fit_ratio=float(df["feature_runtime_s"].sum()
                                   / df["runtime_s"].sum()),
        note="Part E: the ratio is what decides whether a triage layer is worth "
             "building at all")
    print(f"\nPart E cost: features {out['cost']['median_feature_s']*1000:.0f} ms "
          f"median against fit {out['cost']['median_fit_s']:.2f} s median, "
          f"feature/fit = {out['cost']['feature_to_fit_ratio']:.4f}")

    with open(os.path.join(lib.RESULTS, "part_f.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwritten results/part_f.json")
    return out


if __name__ == "__main__":
    main()
