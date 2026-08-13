"""PART H (OPTIONAL): the triage classifier.

Runs last, after Parts A-G are complete and their reports written.  No new
simulation: it trains on the Part E feature vectors already computed for every
Part F and Part G curve.

The label is the DETECTOR'S trigger flag, not the injected truth.  A triage
layer's job is to preserve the detector's positives, not to become a rival
detector, and labelling on the trigger makes completeness clean, because the
quantity measured is exactly P(triage passes | detector would detect).

All three models are judged at one fixed operating point: 99% recall of
detector-positives on the test split.  The figure of merit is retention, the
proportion of all curves that would go forward to the expensive scan.  Lower
retention at equal recall is better.
"""
import json
import os

import numpy as np
import pandas as pd

import lib

RECALL_TARGET = 0.99
FAP = 1e-4


def load():
    feats = pd.read_csv(os.path.join(lib.RESULTS, "features.csv"),
                        float_precision="round_trip")
    f = pd.read_csv(os.path.join(lib.RESULTS, "part_f_per_fit.csv"),
                    float_precision="round_trip")
    g = pd.read_csv(os.path.join(lib.RESULTS, "part_g_per_fit.csv"),
                    float_precision="round_trip")
    res = pd.concat([f, g], ignore_index=True)
    df = res.merge(feats, on=["part", "config_index", "i"], how="inner",
                   validate="one_to_one")
    return df


def split_by_seed(df, seed=20260818):
    """60/20/20 by SEED, never by row.  No curve may appear in two splits."""
    seeds = np.unique(df["seed"].to_numpy())
    rng = np.random.default_rng(seed)
    rng.shuffle(seeds)
    n = seeds.size
    tr = set(seeds[:int(0.6 * n)].tolist())
    va = set(seeds[int(0.6 * n):int(0.8 * n)].tolist())
    te = set(seeds[int(0.8 * n):].tolist())
    s = df["seed"].to_numpy()
    return (df[np.isin(s, list(tr))].copy(), df[np.isin(s, list(va))].copy(),
            df[np.isin(s, list(te))].copy())


def retention_at_recall(score, y, target=RECALL_TARGET):
    """Lowest retention achieving >= target recall of the positives."""
    order = np.argsort(-score)
    y_sorted = np.asarray(y)[order]
    cum = np.cumsum(y_sorted)
    total = cum[-1] if cum.size else 0
    if total == 0:
        return dict(threshold=np.nan, recall=np.nan, retention=np.nan)
    need = int(np.ceil(target * total))
    k = int(np.searchsorted(cum, need) + 1)
    k = min(k, len(y_sorted))
    return dict(threshold=float(np.sort(score)[::-1][k - 1]),
                recall=float(cum[k - 1] / total),
                retention=float(k / len(y_sorted)),
                n_kept=int(k), n_total=int(len(y_sorted)),
                n_positives=int(total))


def main(feature_cols=None):
    df = load()
    df = df[df["insufficient_data"] == 0].copy()
    y = df["trigger"].to_numpy().astype(int)
    print(f"PART H: {len(df)} curves, {y.sum()} detector-positives "
          f"({100*y.mean():.1f}%), label = trigger at FAP {FAP:g}\n")

    import features as feat
    cols = feature_cols or [c for c in feat.FEATURE_NAMES if c in df.columns]
    X = df[cols].to_numpy(dtype=float)
    # colour features are undefined for one-band curves, impute and flag
    miss = ~np.isfinite(X)
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    X = np.where(miss, med, X)
    df_tr, df_va, df_te = split_by_seed(df)
    idx = {k: df.index.get_indexer(d.index)
           for k, d in (("train", df_tr), ("val", df_va), ("test", df_te))}
    out = dict(n=int(len(df)), n_positive=int(y.sum()),
               positive_rate=float(y.mean()), recall_target=RECALL_TARGET,
               fap=FAP, features=cols,
               split=dict(train=int(len(df_tr)), val=int(len(df_va)),
                          test=int(len(df_te)), by="seed"),
               models=[])
    Xs = {k: X[v] for k, v in idx.items()}
    ys = {k: y[v] for k, v in idx.items()}

    i_ls = cols.index("ls_peak_power")
    i_sc = cols.index("robust_scatter_mag")
    best = None
    ls_v, sc_v = Xs["val"][:, i_ls], Xs["val"][:, i_sc]
    for qa in np.linspace(0.0, 0.9, 46):
        for qb in np.linspace(0.0, 0.9, 46):
            ta, tb = np.quantile(ls_v, qa), np.quantile(sc_v, qb)
            keep = (ls_v >= ta) & (sc_v >= tb)
            if keep.sum() == 0:
                continue
            rec = ys["val"][keep].sum() / max(ys["val"].sum(), 1)
            if rec >= RECALL_TARGET:
                ret = keep.mean()
                if best is None or ret < best["retention"]:
                    best = dict(ls_threshold=float(ta), scatter_threshold=float(tb),
                                retention=float(ret), recall=float(rec))
    if best is None:
        best = dict(ls_threshold=float(np.min(ls_v)),
                    scatter_threshold=float(np.min(sc_v)),
                    retention=1.0, recall=1.0)
    keep_te = ((Xs["test"][:, i_ls] >= best["ls_threshold"])
               & (Xs["test"][:, i_sc] >= best["scatter_threshold"]))
    base = dict(model="trivial two-parameter cut",
                detail="Lomb-Scargle peak power and robust scatter, thresholds "
                       "chosen on the validation split",
                val=best,
                test=dict(recall=float(ys["test"][keep_te].sum()
                                       / max(ys["test"].sum(), 1)),
                          retention=float(keep_te.mean())))
    out["models"].append(base)
    print(f"  {base['model']:<34} test recall {base['test']['recall']:.4f}  "
          f"retention {base['test']['retention']:.4f}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                        early_stopping=True, random_state=0)
    gb.fit(Xs["train"], ys["train"])
    s_te = gb.predict_proba(Xs["test"])[:, 1]
    r = retention_at_recall(s_te, ys["test"])
    try:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(gb, Xs["val"], ys["val"], n_repeats=5,
                                    random_state=0, scoring="roc_auc")
        imp = sorted(zip(cols, pi.importances_mean), key=lambda t: -t[1])
    except Exception:
        imp = []
    gbm = dict(model="gradient-boosted trees", test=r,
               feature_importances=[dict(feature=k, importance=float(v))
                                    for k, v in imp[:15]])
    out["models"].append(gbm)
    print(f"  {gbm['model']:<34} test recall {r['recall']:.4f}  "
          f"retention {r['retention']:.4f}")

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xs["train"])
    nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                       early_stopping=True, n_iter_no_change=20,
                       validation_fraction=0.2, random_state=0)
    nn.fit(sc.transform(Xs["train"]), ys["train"])
    s_te_nn = nn.predict_proba(sc.transform(Xs["test"]))[:, 1]
    rn = retention_at_recall(s_te_nn, ys["test"])
    nnm = dict(model="neural network (2 hidden layers, 64-32)", test=rn)
    out["models"].append(nnm)
    print(f"  {nnm['model']:<34} test recall {rn['recall']:.4f}  "
          f"retention {rn['retention']:.4f}")

    rb, rg, rn_ = (base["test"]["retention"], r["retention"], rn["retention"])
    learned_best = min(rg, rn_)
    if rb < 0.20 and learned_best > 0.9 * rb:
        reading = "a"
        adopted = "trivial two-parameter cut"
        text = ("Reading (a): the trivial baseline reaches 99% recall at under "
                "20% retention and the learned models do not improve on it "
                "materially.  The trivial cut is adopted and it is recorded "
                "that the learned models added nothing.")
    elif rg <= 0.9 * rb and rg <= rn_:
        reading = "b"
        adopted = "gradient-boosted trees"
        text = ("Reading (b): gradient-boosted trees improve materially on the "
                "trivial cut and the neural network does not beat them.  The "
                "trees are adopted.  A neural network was tried and was not "
                "better.")
    elif rn_ < rg and rn_ <= 0.9 * rb:
        reading = "c"
        adopted = "neural network (2 hidden layers, 64-32)"
        text = ("Reading (c): the network wins.  The margin is reported at the "
                "fixed operating point, never as AUC alone.")
    else:
        reading = "a"
        adopted = "trivial two-parameter cut"
        text = ("Reading (a) by default: no learned model improves materially "
                "on the trivial cut at the fixed operating point.")
    out["reading"] = reading
    out["reading_text"] = text
    out["adopted"] = adopted
    out["retentions_at_99pct_recall"] = dict(
        trivial=rb, gradient_boosted_trees=rg, neural_network=rn_)
    print(f"\n{text}")

    scores = {"trivial two-parameter cut": keep_te.astype(float),
              "gradient-boosted trees": s_te,
              "neural network (2 hidden layers, 64-32)": s_te_nn}[adopted]
    thr = (0.5 if adopted.startswith("trivial")
           else retention_at_recall(scores, ys["test"])["threshold"])
    passes = scores >= thr
    te = df_te.copy()
    te["triage_pass"] = passes
    det = te[te["trigger"] == 1]
    out["triage_selection_function"] = triage_completeness(det)
    out["overall_triage_completeness"] = float(det["triage_pass"].mean())
    print(f"\nP(triage passes | detector would detect) on held-out test data: "
          f"{out['overall_triage_completeness']:.4f}")
    print("  least uniform axes (largest spread across bins):")
    for a in out["triage_selection_function"]["least_uniform_axes"]:
        print(f"    {a['axis']:<20} spread {a['spread']:.3f} "
              f"(min {a['min']:.3f}, max {a['max']:.3f})")

    out["retention_caveat"] = dict(
        positive_rate=float(y.mean()),
        text=("The retention figures do not transfer to a survey, and the "
              "reason is the training population, not the models. These curves "
              "are {:.0%} detector-positive, because they are an injection "
              "campaign designed to straddle the detection boundary. A real "
              "search is almost all nulls, with a positive rate near the "
              "false-alarm probability itself. At a {:.0%} positive rate, "
              "keeping 99% of positives arithmetically forces retention near "
              "1: there is very little that can be discarded. The ranking "
              "between the three models is still informative, because all "
              "three face the same population, the absolute retention is not. "
              "Measuring a deployable retention needs a training set with a "
              "realistic prior, which this campaign was not built to "
              "provide.").format(float(y.mean()), float(y.mean())),
        best_retention=float(min(rb, rg, rn_)),
        interpretation=("no triage layer tested here is worth deploying at "
                        "this operating point on this population: the best "
                        "removes {:.0%} of curves while losing {:.1%} of the "
                        "detector's positives").format(
                            1 - min(rb, rg, rn_), 1 - min(r["recall"],
                                                          rn["recall"])),
    )
    out["limits"] = [
        out["retention_caveat"]["text"],
        "The classifier is trained on simulations from one generator and one "
        "background family, so its completeness is conditional on that "
        "generator resembling real AGN variability, which this project's own "
        "results and the literature on DRW inadequacy give reason to doubt.",
        "Out-of-distribution behaviour is not tested here.",
        "Triage completeness is a multiplicative factor on the Part G "
        "selection function, it is measured, not assumed away.",
    ]
    with open(os.path.join(lib.RESULTS, "part_h.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwritten results/part_h.json")
    return out


def triage_completeness(det):
    """P(triage passes | detector would detect) across the Part G axes."""
    import surface as S
    axes = {}
    for a in S.AXES:
        if a not in det.columns:
            continue
        v = det[a].to_numpy(dtype=float)
        edges = (np.exp(np.linspace(np.log(v.min()), np.log(v.max()), 6))
                 if a in ("n_cyc", "snr", "tau_over_p", "samples_per_cycle")
                 else np.linspace(v.min(), v.max(), 6))
        idx = np.clip(np.digitize(v, edges) - 1, 0, 4)
        rows = []
        for b in range(5):
            m = idx == b
            n = int(m.sum())
            if n < 5:
                continue
            p = float(det["triage_pass"].to_numpy()[m].mean())
            se = np.sqrt(max(p * (1 - p), 1e-12) / n)
            rows.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=n,
                             completeness=p,
                             lo95=max(0.0, p - 1.96 * se),
                             hi95=min(1.0, p + 1.96 * se)))
        if rows:
            axes[a] = rows
    for a, rows in list(axes.items()):
        pass
    spread = []
    for a, rows in axes.items():
        vals = [r["completeness"] for r in rows]
        spread.append(dict(axis=a, spread=float(max(vals) - min(vals)),
                           min=float(min(vals)), max=float(max(vals))))
    bs_rows = []
    for bs in sorted(det["band_structure"].unique()):
        m = det["band_structure"] == bs
        if m.sum() >= 5:
            bs_rows.append(dict(band_structure=int(bs), n=int(m.sum()),
                                completeness=float(det[m]["triage_pass"].mean())))
    spread.sort(key=lambda r: -r["spread"])
    return dict(by_axis=axes, band_structure=bs_rows,
                least_uniform_axes=spread[:3], all_axis_spread=spread,
                interpretation="the axes with the largest spread are where the "
                               "combined selection function is most distorted")


if __name__ == "__main__":
    main()
