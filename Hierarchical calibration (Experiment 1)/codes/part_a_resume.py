"""PART A: recover state.  Read-only inventory.  No fits are run here.

Writes item1/results/part_a.json and prints the check table.  Nothing
outside item1/ is written.
"""
import hashlib
import json
import os
import platform
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ITEM1 = os.path.dirname(HERE)
HCAL = os.path.dirname(ITEM1)
PARENT = os.path.dirname(HCAL)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CHECKS = []


def record(name, expected, observed, status, detail=""):
    CHECKS.append(dict(check=name, expected=expected, observed=observed,
                       status=status, detail=detail))
    return status


def main():
    t0 = time.time()
    out = {}

    f_per_fit = os.path.join(HCAL, "results", "stage1_per_fit.csv")
    per_fit = None
    if not os.path.exists(f_per_fit):
        record("results/stage1_per_fit.csv", "13,800 rows", "MISSING", "FAIL")
    else:
        try:
            per_fit = pd.read_csv(f_per_fit)
            record("results/stage1_per_fit.csv parses", "parses",
                   f"{len(per_fit)} rows x {per_fit.shape[1]} cols",
                   "PASS" if len(per_fit) == 13800 else "PARTIAL",
                   "expected 13,800")
        except Exception as e:  # noqa
            record("results/stage1_per_fit.csv parses", "parses", f"ERROR {e}", "FAIL")

    if per_fit is not None:
        vc = per_fit["config_index"].value_counts().sort_index()
        n_at_400 = int((vc[vc.index != 99] == 400).sum())
        n_cfg = int((vc.index != 99).sum())
        n99 = int(vc.get(99, 0))
        record("22 configurations at 400 rows",
               "22 configs x 400", f"{n_cfg} configs, {n_at_400} of them at exactly 400",
               "PASS" if (n_cfg == 22 and n_at_400 == 22) else "FAIL")
        record("config_index 99 tail block", "5000 rows", f"{n99} rows",
               "PASS" if n99 == 5000 else "FAIL")
        out["rows_per_config"] = {int(k): int(v) for k, v in vc.items()}

        n = len(per_fit)
        finite_cols = ["Lambda_osc", "Lambda_per0", "DeltaLambda_chirp"]
        finite_ok = np.isfinite(per_fit[finite_cols].to_numpy()).all(axis=1)
        nonneg = per_fit["DeltaLambda_chirp"].to_numpy() >= -1e-9
        stage_num = np.where(per_fit["stage"].astype(str).str.startswith("1"), 1, 0)
        seed_exp = (20260808 + 100000 * stage_num
                    + 1000 * per_fit["config_index"].to_numpy()
                    + per_fit["i"].to_numpy())
        seed_ok = per_fit["seed"].to_numpy() == seed_exp
        dup = per_fit.duplicated(subset=["stage", "config_index", "i"], keep=False)
        valid = finite_ok & nonneg & seed_ok & ~dup.to_numpy()
        out["row_validation"] = dict(
            n_rows=int(n), n_valid=int(valid.sum()),
            n_dropped=int((~valid).sum()),
            n_nonfinite=int((~finite_ok).sum()),
            n_negative_dlambda=int((~nonneg).sum()),
            n_seed_mismatch=int((~seed_ok).sum()),
            n_duplicated=int(dup.sum()),
            n_insufficient_data=int(per_fit["insufficient_data"].sum()),
        )
        record("row validation: finite scores", "0 non-finite",
               f"{int((~finite_ok).sum())}", "PASS" if finite_ok.all() else "FAIL")
        record("row validation: DeltaLambda_chirp >= -1e-9", "0 violations",
               f"{int((~nonneg).sum())}", "PASS" if nonneg.all() else "FAIL")
        record("row validation: seed rule", "0 mismatches",
               f"{int((~seed_ok).sum())}", "PASS" if seed_ok.all() else "FAIL")
        record("row validation: duplicate (stage, config_index, i)", "0",
               f"{int(dup.sum())}", "PASS" if not dup.any() else "FAIL")

    f_summary = os.path.join(HCAL, "results", "config_summary.csv")
    summ = None
    if os.path.exists(f_summary):
        try:
            summ = pd.read_csv(f_summary)
            record("results/config_summary.csv", "22 rows",
                   f"{len(summ)} rows x {summ.shape[1]} cols",
                   "PASS" if len(summ) == 22 else "FAIL")
        except Exception as e:  # noqa
            record("results/config_summary.csv", "22 rows", f"ERROR {e}", "FAIL")
    else:
        record("results/config_summary.csv", "22 rows", "MISSING", "FAIL")

    f_tail = os.path.join(HCAL, "results", "tail_fit.json")
    tail = None
    if os.path.exists(f_tail):
        try:
            tail = json.load(open(f_tail, encoding="utf-8"))
            has_holdout = "holdout" in tail and bool(tail["holdout"])
            record("results/tail_fit.json", "parses, has hold-out",
                   f"parses, holdout={'present' if has_holdout else 'ABSENT'}",
                   "PASS" if has_holdout else "FAIL")
            out["tail_operating_points"] = tail.get("operating_points")
            out["tail_holdout"] = tail.get("holdout")
        except Exception as e:  # noqa
            record("results/tail_fit.json", "parses", f"ERROR {e}", "FAIL")
    else:
        record("results/tail_fit.json", "parses", "MISSING", "FAIL")

    f_s0 = os.path.join(HCAL, "stage0_verdict.md")
    if os.path.exists(f_s0):
        txt = open(f_s0, encoding="utf-8").read()
        s0j = json.load(open(os.path.join(HCAL, "results", "stage0.json"),
                             encoding="utf-8"))
        ok = (abs(float(s0j.get("check4_worst_abs_diff", 9e9))) == 0.0
              and int(s0j.get("check4_n_exact", -1)) == 20)
        record("stage0_verdict.md regression", "0.000e+00 on 20/20",
               f"worst_abs_diff={s0j.get('check4_worst_abs_diff')}, "
               f"n_exact={s0j.get('check4_n_exact')}/20",
               "PASS" if ok else "FAIL",
               "0.000e+00 also quoted verbatim in stage0_verdict.md"
               if "0.000e+00" in txt else "string not found in md")
        out["stage0"] = s0j
    else:
        record("stage0_verdict.md regression", "0.000e+00 on 20/20", "MISSING", "FAIL")

    prov = os.path.join(HCAL, "PROVENANCE.md")
    code_dir = os.path.join(HCAL, "code")
    prov_hashes = {}
    if os.path.exists(prov):
        for line in open(prov, encoding="utf-8"):
            m = re.search(r"`(code/[A-Za-z0-9_./]+\.py)`.*?`([0-9a-f]{64})`", line)
            if m:
                prov_hashes[m.group(1)] = m.group(2)
    live = {}
    for fn in sorted(os.listdir(code_dir)):
        if fn.endswith(".py"):
            live["code/" + fn] = sha256(os.path.join(code_dir, fn))
    match, mismatch, absent = [], [], []
    for k, v in prov_hashes.items():
        if k not in live:
            absent.append(k)
        elif live[k] == v:
            match.append(k)
        else:
            mismatch.append(k)
    record("code/ populated", "populated", f"{len(live)} .py files",
           "PASS" if len(live) > 0 else "FAIL")
    record("sha256 vs PROVENANCE.md",
           f"{len(prov_hashes)} recorded files match",
           f"{len(match)} match, {len(mismatch)} mismatch, {len(absent)} absent",
           "PASS" if not mismatch and not absent else "MISMATCH",
           "mismatched: " + ", ".join(mismatch) if mismatch else "")
    out["provenance"] = dict(recorded=prov_hashes, live=live,
                             match=match, mismatch=mismatch, absent=absent)

    analyse_present = os.path.exists(os.path.join(code_dir, "analysis.py"))
    record("analyse() present", "code/analysis.py with analyse()",
           "present" if analyse_present else "MISSING",
           "PASS" if analyse_present else "FAIL")

    f_targets = os.path.join(HCAL, "real_data", "targets.csv")
    n_targets = -1
    if os.path.exists(f_targets):
        n_targets = len(pd.read_csv(f_targets))
    raw_dir = os.path.join(HCAL, "real_data", "raw")
    n_raw = len([f for f in os.listdir(raw_dir) if f.endswith(".csv")]) \
        if os.path.isdir(raw_dir) else -1
    f_rdr = os.path.join(HCAL, "results", "real_data_results.csv")
    n_rdr = len(pd.read_csv(f_rdr)) if os.path.exists(f_rdr) else -1
    ok = (n_targets == 6 and n_raw == 6 and n_rdr == 6)
    record("real_data targets / raw / results", "6 objects each",
           f"targets={n_targets}, raw={n_raw}, results={n_rdr}",
           "PASS" if ok else "FAIL")

    f_cad = os.path.join(PARENT, "cadence.csv")
    if os.path.exists(f_cad):
        cad = pd.read_csv(f_cad)
        span = float(cad["t_days"].max() - cad["t_days"].min())
        ok = (len(cad) == 765 and abs(span - 1996.4) < 0.15)
        record("../cadence.csv", "765 epochs, 1996.4 d",
               f"{len(cad)} epochs, span {span:.3f} d",
               "PASS" if ok else "FAIL", "never regenerated, shared by every run")
        out["cadence"] = dict(n=len(cad), span_d=span, sha256=sha256(f_cad))
    else:
        record("../cadence.csv", "765 epochs, 1996.4 d", "MISSING", "FAIL")

    n_fail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    n_tot = len(CHECKS)
    out["checks"] = CHECKS
    out["n_checks"] = n_tot
    out["n_fail"] = n_fail
    out["frac_fail"] = n_fail / n_tot
    out["environment"] = dict(
        python=sys.version, platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__,
    )
    try:
        import scipy, joblib, celerite2, matplotlib, sklearn
        out["environment"].update(scipy=scipy.__version__, joblib=joblib.__version__,
                                  celerite2=celerite2.__version__,
                                  matplotlib=matplotlib.__version__,
                                  sklearn=sklearn.__version__)
    except Exception as e:  # noqa
        out["environment"]["import_note"] = str(e)
    out["elapsed_s"] = time.time() - t0

    with open(os.path.join(ITEM1, "results", "part_a.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)

    w = max(len(c["check"]) for c in CHECKS) + 2
    print(f"{'CHECK':<{w}} {'STATUS':<9} OBSERVED")
    print("-" * (w + 60))
    for c in CHECKS:
        print(f"{c['check']:<{w}} {c['status']:<9} {c['observed']}")
    print("-" * (w + 60))
    print(f"{n_fail}/{n_tot} FAIL   ({100*n_fail/n_tot:.1f}%)")
    if n_fail / n_tot > 0.25:
        print("STOP CONDITION: more than a quarter of checks failed -> "
              "recommend disk health check")
    return out


if __name__ == "__main__":
    main()
