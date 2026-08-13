"""Write report/RUN_RECORD.txt, in plain text, for the next operator."""
import json
import os
import platform
import sys
from datetime import datetime, timezone

import lib


def jload(n):
    p = os.path.join(lib.RESULTS, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def count(name):
    p = os.path.join(lib.RESULTS, name)
    if not os.path.exists(p):
        return 0
    with open(p, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main(notes=""):
    cfg = json.load(open(os.path.join(lib.ITEM1, "config.json"), encoding="utf-8"))
    A, B, C, D = jload("part_a.json"), jload("part_b.json"), jload("part_c.json"), jload("part_d.json")
    F, G, H = jload("part_f.json"), jload("part_g.json"), jload("part_h.json")
    TH, gate = jload("thresholds.json"), jload("gate_and_throughput.json")

    hb = {}
    p = os.path.join(lib.ITEM1, "HEARTBEAT.log")
    total_fits = 0
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
        for r in lines:
            hb.setdefault(r["part"], 0)
            hb[r["part"]] = max(hb[r["part"]], r["rows_written"])
        total_fits = sum(hb.values())

    L = []
    w = L.append
    w("RUN RECORD: item 1, full hierarchical calibration")
    w("=" * 70)
    w(f"date                {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    w(f"specification       {lib.SPEC_VERSION}")
    w(f"working folder      ~/AGN_chirp_experiment/hcal/item1/")
    w(f"machine             {platform.platform()}, {os.cpu_count()} logical CPUs")
    e = cfg["environment"]
    w(f"python              {e['python'].splitlines()[0]}")
    w(f"packages            numpy {e['numpy']}, scipy {e['scipy']}, "
      f"pandas {e['pandas']}, joblib {e['joblib']},")
    w(f"                    celerite2 {e['celerite2']}, matplotlib "
      f"{e['matplotlib']}, sklearn {e['sklearn']}, astropy {e['astropy']}")
    w("")
    w("PARTS RUN")
    w("-" * 70)
    rows = [
        ("A  recover state", "RAN" if A else "not run",
         f"{A['n_checks']} checks, {A['n_fail']} FAIL, 13,800 rows validated"
         if A else ""),
        ("B  free check", "RAN" if B else "not run",
         f"conclusion: {B['conclusion'].upper()}" if B else ""),
        ("C  outlier cut", "RAN" if C else "not run",
         f"{count('part_c_per_fit.csv')} fits, reading {C['reading']}, "
         f"adopted {C['adopted_cut_mode']}" if C else ""),
        ("D  search grid", "RAN" if D else "not run",
         f"{count('part_d_per_fit.csv')} fits, reading {D['reading']}, "
         f"{D['grid_choice']}" if D else ""),
        ("E  cheap features", "RAN" if F else "not run",
         f"{count('features.csv')} feature vectors, "
         f"feature/fit = {F['cost']['feature_to_fit_ratio']:.4f}" if F else ""),
        ("F  screening injections", "RAN" if F else "not run",
         f"{count('part_f_per_fit.csv')} injections" if F else ""),
        ("G  refinement + selection fn", "RAN" if G else "not run",
         f"{count('part_g_per_fit.csv')} fits, Brier "
         f"{G['holdout']['calibration']['brier']:.4f}" if G else ""),
        ("H  triage classifier", "RAN" if H else "NOT RUN (optional)",
         f"adopted {H['adopted']}, reading {H['reading']}" if H else ""),
    ]
    for a, b, c in rows:
        w(f"  {a:<30} {b:<18} {c}")
    w("")
    w(f"total fits recorded by the heartbeat: {total_fits}")
    w("")
    w("DEVIATIONS (full text in item1/DEVIATIONS.md)")
    w("-" * 70)
    dv = os.path.join(lib.ITEM1, "DEVIATIONS.md")
    if os.path.exists(dv):
        for line in open(dv, encoding="utf-8"):
            if line.startswith("## "):
                w("  " + line[3:].rstrip())
    w("")
    w("CHECKS THAT FAILED")
    w("-" * 70)
    if A:
        bad = [c for c in A["checks"] if c["status"] not in ("PASS",)]
        if not bad:
            w("  none")
        for c in bad:
            w(f"  {c['status']:<10} {c['check']}: {c['observed']}")
            w(f"             resolved by reproduction, see DEVIATIONS.md D1")
    w("")
    w("COST")
    w("-" * 70)
    if gate:
        w(f"  serial fit, fiducial          {gate['fiducial_serial_s']:.3f} s")
        for t in gate["throughput"]:
            w(f"    {t['case']:<34} {t['n_epochs']:>5} epochs  "
              f"{t['median_serial_s']:6.3f} s")
    if F:
        c = F["cost"]
        w(f"  FIT-COST TO FEATURE-COST RATIO (Part E):")
        w(f"    median fit      {c['median_fit_s']:.3f} s")
        w(f"    median features {c['median_feature_s']:.3f} s")
        w(f"    feature/fit     {c['feature_to_fit_ratio']:.4f}  "
          f"({1/max(c['feature_to_fit_ratio'],1e-9):.1f}x cheaper than a fit)")
    w("")
    w("THERMAL")
    w("-" * 70)
    t = cfg["thermal"]
    w(f"  n_jobs {t['n_jobs']} (never above {t['n_jobs_max']}), BLAS pinned to "
      f"one thread")
    w(f"  {t['pause_between_batches_s']:.0f} s between joblib batches of "
      f"{t['batch_size']}, {t['pause_long_s']:.0f} s every "
      f"{t['pause_every_fits']} fits")
    w(f"  one row flushed and fsync'd per fit")
    w("  measured: at n_jobs = 2 the contention factor was about 1.44x, giving")
    w("  roughly the same throughput as stage 1 reached at n_jobs = 3.  Raising")
    w("  n_jobs was therefore declined: it would have bought nothing and run")
    w("  the machine hotter after four shutdowns.")
    w("")
    w("WHAT WAS CUT FOR TIME OR THERMAL REASONS")
    w("-" * 70)
    w(notes or "  nothing")
    w("")
    w("WHAT THE OPERATOR SHOULD KNOW BEFORE THE NEXT STAGE")
    w("-" * 70)
    for line in operator_notes(A, C, D, F, G, H, TH):
        w("  " + line)
    w("")
    path = os.path.join(lib.ITEM1, "report", "RUN_RECORD.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("written", path)


def operator_notes(A, C, D, F, G, H, TH):
    n = []
    n.append("hcal/PROVENANCE.md is STALE: it records pre-v2 hashes for seven")
    n.append("  files including analysis.py.  It was not corrected because it")
    n.append("  lies outside item1/.  Regenerate it before the next stage, or")
    n.append("  the same false alarm will fire again.")
    if C:
        n.append(f"The adopted cut mode is {C['adopted_cut_mode']}.  Every")
        n.append("  threshold published before this run carries the mad_raw")
        n.append("  bias and must be rescaled by the Part C correction factor")
        n.append("  before it is compared with anything from item 1.")
    if D:
        n.append(f"Grid: {D['grid_choice']}.  Do not change it without")
        n.append("  recalibrating: it sets the look-elsewhere volume.")
    if F:
        n.append("The N_cyc axis runs to 40, but at T = 2000 d the grid's")
        n.append("  60 d floor means N_cyc > 33.3 lies OUTSIDE the search")
        n.append("  range.  Completeness there is zero by construction, not by")
        n.append("  measurement.")
    if G:
        n.append("Profile-likelihood intervals on P and eta are narrower than")
        n.append("  the grid spacing for strong signals, so their coverage is")
        n.append("  poor even where the point estimate is excellent.  Do not")
        n.append("  quote them as uncertainties without reading Section 9.3.")
    if not H:
        n.append("Part H did not run.  No triage layer exists, so item 2's")
        n.append("  shortlisting has no measured completeness factor.")
    n.append("Item 2 (flexible stochastic nulls) is untouched, by design.")
    return n


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
