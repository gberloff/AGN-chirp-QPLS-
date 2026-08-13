"""Pre-flight gate for item1, run before Part C.

Gate 1  The cut-mode extension must be a strict superset of the existing
        behaviour: with cut_mode = "mad_raw" and window 50 d, item1's path must
        reproduce stored stage-1 rows bitwise.  If it does not, the "identical
        code on data, nulls and injections" requirement is already broken.

Gate 2  The frequency-uniform grid option must return the declared node count,
        span the same range, and keep eta = 0 on the grid.

Throughput  Serial cost of one fit at the three Part C / Part D configurations
        and at the extremes of the Part F space, so the campaign can be costed
        before Part F starts.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import lib
import analysis

lib.install_cut_modes()


def gate1():
    cfg = lib.load_hcal_config()
    entries = {e["config_index"]: e for e in cfg["stage1a"]["configs"]}
    entries[99] = dict(entries[0], config_index=99)
    per_fit = pd.read_csv(os.path.join(lib.HCAL, "results", "stage1_per_fit.csv"),
                          float_precision="round_trip")
    FIELDS = ["Lambda_osc", "Lambda_per0", "DeltaLambda_chirp", "P_hat",
              "eta_hat", "sigma_hat", "tau_hat", "l0", "n_epochs",
              "n_removed_mad", "n_removed_total", "n_grid_nodes"]
    gc = lib.GenCache(int(cfg["cadence_seed"]))
    rows = []
    for ci in (0, 3, 12, 17, 99):
        sub = per_fit[per_fit["config_index"] == ci]
        if sub.empty:
            continue
        entry = entries[ci]
        spec = lib.build_spec(entry, cut_mode="mad_raw", window_d=50.0)
        assert isinstance(spec, lib.SpecExt) and spec.cut_mode == "mad_raw"
        g = gc.get(entry) if not entry.get("real_cadence") else \
            __import__("gen").Generator(entry, int(cfg["cadence_seed"]))
        r = sub.iloc[0]
        res = analysis.run_on_null(g, int(r["seed"]), spec).as_dict()
        worst = max(abs(float(res[k]) - float(r[k])) for k in FIELDS)
        rows.append(dict(config_index=int(ci), worst_abs_diff=worst))
        print(f"  gate1 config {ci:>2}: worst |diff| = {worst:.3e}")
    worst = max(r["worst_abs_diff"] for r in rows)
    ok = worst == 0.0
    print(f"  GATE 1: {'PASS' if ok else 'FAIL'} "
          f"(worst |diff| {worst:.3e} over {len(rows)} rows)")
    return dict(pass_=bool(ok), worst_abs_diff=worst, rows=rows)


def gate2():
    entry = lib.make_entry(0, "fiducial")
    spec = lib.build_spec(entry, cut_mode="mad_scaled")
    lib.set_grid_spacing(True)
    P, eta = analysis.build_grid(spec)
    lib.set_grid_spacing(False)
    Plog, etalog = analysis.build_grid(spec)
    ok = (P.size == spec.P_n == Plog.size
          and abs(P.min() - spec.P_min_d) < 1e-9
          and abs(P.max() - spec.P_max_d) < 1e-6
          and np.isclose(eta, 0.0).sum() == 1
          and np.all(np.diff(P) > 0))
    print(f"  GATE 2: {'PASS' if ok else 'FAIL'}  freq-uniform nodes={P.size}, "
          f"span [{P.min():.3f}, {P.max():.3f}] d, "
          f"log nodes below 100 d: {(Plog<100).sum()}, "
          f"freq-uniform nodes below 100 d: {(P<100).sum()}")
    return dict(pass_=bool(ok), n_nodes=int(P.size),
                n_below_100d_log=int((Plog < 100).sum()),
                n_below_100d_frequniform=int((P < 100).sum()))


def timeit(entry, n=3, cut_mode="mad_scaled"):
    gc = lib.GenCache()
    g = gc.get(entry)
    spec = lib.build_spec(entry, cut_mode=cut_mode)
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        analysis.run_on_null(g, 999000 + i, spec)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), int(g.n_epochs)


def throughput():
    cases = [
        ("C/D fiducial  T=2000 dt=3 2band", lib.make_entry(0, "fid")),
        ("C tau/T=1.6   tau=3200",          lib.make_entry(1, "tau", tau_d=3200.0)),
        ("C T=4000",                        lib.make_entry(2, "T4000",
                                                           baseline_d=4000.0)),
        ("F cheap   N_cyc=3  spc=35 2band", lib.make_entry(3, "cheap",
                                                           dt_d=666.7 / 35)),
        ("F costly  N_cyc=33 spc=60 2band", lib.make_entry(4, "costly",
                                                           dt_d=60.0 / 60)),
        ("F 1-band  fiducial dt",           lib.make_entry(5, "1band", n_bands=1)),
        ("G duty=0.9",                      lib.make_entry(6, "duty9",
                                                           duty_cycle=0.9)),
    ]
    out = []
    for name, e in cases:
        t, n_ep = timeit(e)
        out.append(dict(case=name, median_serial_s=t, n_epochs=n_ep))
        print(f"  {name:<34} {n_ep:>5} epochs   {t:6.3f} s serial")
    return out


def main():
    print("PRE-FLIGHT GATE\n")
    g1 = gate1()
    g2 = gate2()
    print("\nTHROUGHPUT (serial, one core)\n")
    tp = throughput()

    fid = [r for r in tp if r["case"].startswith("C/D fiducial")][0]
    out = dict(gate1_mad_raw_reproduces_stage1=g1, gate2_freq_uniform_grid=g2,
               throughput=tp, fiducial_serial_s=fid["median_serial_s"])
    with open(os.path.join(lib.RESULTS, "gate_and_throughput.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwritten results/gate_and_throughput.json")


if __name__ == "__main__":
    main()
