
import argparse
import csv
from functools import partial

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qpls.injection_recovery as _ir
from qpls.injection_recovery import (
    RESULTS_DIR, ztf_cadence, gls_peak_power, gls_harmonic_sum,
    null_threshold, recovery_grid, flare_train_sawtooth, make_lc,
)
from qpls.detector_comparison import CFG, oracle_recovery
from qpls.source_scenarios import scenario_cell

FIG_DIR = RESULTS_DIR.parent / "figures"

STATISTIC = "v2-physical"
SEED = 42
P_PHYS = 180.0
FWHM_PHYS = 16.0 / 24.0
RISE_FRAC = 0.1
TEMPLATE = "sawtooth"

A_SWEEP = np.geomspace(0.02, 3.0, 12)
DETECTORS = ["global_max", "oracle_single_freq", "blind_harmonic_sum"]
COMMITTED_THR_V1 = {"global_max": 0.4277, "blind_harmonic_sum": 0.8047}

SPOT_CELLS = [("RGBiso", 0.1, 35.00), ("RGBiso", 0.5, 38.49),
              ("BSG", 0.1, 35.00), ("BSG", 0.5, 38.49)]


def setup(verbose=True):
    _ir.RNG.bit_generator.state = np.random.default_rng(0).bit_generator.state
    t, sigma = ztf_cadence(CFG["baseline_yr"] * 365.25, CFG["cadence_days"],
                           CFG["season_frac"], err_med=CFG["err_med"])
    base = CFG["baseline_yr"] * 365.25
    p_min, p_max = 2 * CFG["cadence_days"], 0.5 * base
    freq = np.linspace(1.0 / p_max, 1.0 / p_min, 1000)
    det_hs = partial(gls_harmonic_sum, n_harmonics=CFG["n_harmonics"])

    if verbose:
        print("=== P2.1 committed light-curve baseline ===")
        print(f"  baseline            = {base:.2f} d ({CFG['baseline_yr']} yr)")
        print(f"  nominal cadence     = {CFG['cadence_days']} d")
        print(f"  median cadence      = {np.median(np.diff(np.sort(t))):.3f} d "
              f"(realised, seasonal gaps included)")
        print(f"  n observations      = {t.size}")
        print(f"  n_flares = baseline / 180 d = {base/P_PHYS:.2f}")
        print()
        print("=== P2.2 frequency grid ===")
        print(f"  v1 grid: linspace(1/{p_max:.2f} d, 1/{p_min:.2f} d, 1000) "
              f"-> periods {p_min:.2f}-{p_max:.2f} d")
        print(f"  v2 grid: IDENTICAL, 250 d is already inside the v1 grid "
              f"(p_max = {p_max:.2f} d), so no extension is warranted.")
        print("  Consequence: trials factor unchanged, so thresholds should be")
        print("  unchanged. Recalibrated below and checked, not assumed.")
    return t, sigma, freq, det_hs


def thresholds(t, sigma, freq, det_hs, verbose=True):
    thr = {}
    thr["global_max"], _ = null_threshold(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], freq,
        gls_peak_power, CFG["fap"], CFG["n_null"], np.random.default_rng(SEED))
    thr["blind_harmonic_sum"], _ = null_threshold(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], freq,
        det_hs, CFG["fap"], CFG["n_null"], np.random.default_rng(SEED))
    if verbose:
        print("\n  detector            v1 threshold   v2 threshold   delta")
        for k, v1 in COMMITTED_THR_V1.items():
            print(f"  {k:19s} {v1:12.4f} {thr[k]:14.4f} {thr[k]-v1:+9.6f}")
        print("  (nulls are flare-free: template, FWHM and recurrence cannot "
              "affect them)")
    return thr


def wiring_check(t, sigma, path, A=1.0, dpi=300):
    rng = np.random.default_rng(SEED)
    t0 = rng.uniform(0, P_PHYS)
    train = flare_train_sawtooth(t, A, P_PHYS, t0, 2.0 * FWHM_PHYS, RISE_FRAC)
    n_hit = int((train > 0).sum())
    peak = float(train.max())

    dense = np.linspace(t.min(), t.max(), 400000)
    dense_train = flare_train_sawtooth(dense, A, P_PHYS, t0, 2.0 * FWHM_PHYS,
                                       RISE_FRAC)
    n_flares = int(np.ceil((t.max() - t0) / P_PHYS))

    rng2 = np.random.default_rng(SEED)
    mag = make_lc(t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"],
                  P_obs=P_PHYS, A_mag=A, fwhm_days=FWHM_PHYS, rng=rng2,
                  template=TEMPLATE)

    print("\n=== P2.3 WIRING CHECK ===")
    print(f"  flares in baseline        : {n_flares}")
    print(f"  observations inside flare : {n_hit}")
    print(f"  peak sampled amplitude    : {peak:.4f} mag (injected A = {A})")
    ok = n_hit >= 1
    print(f"  >= 1 flare sampled        : {'PASS' if ok else 'FAIL: STOP'}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(dense, -dense_train, lw=0.8, color="0.7",
            label="injected train (continuous)")
    ax.plot(t, mag, ".", ms=3, color="C0", label="sampled light curve")
    hit = train > 0
    ax.plot(t[hit], mag[hit], "o", ms=7, mfc="none", mec="crimson", mew=1.4,
            label=f"observations inside a flare ({n_hit})")
    ax.invert_yaxis()
    ax.set_xlabel("t (d)")
    ax.set_ylabel("mag (brighter up)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"v2 physical regime: P = {P_PHYS:g} d, FWHM = 16 hr, "
                 f"sawtooth, A = {A:g} mag", fontsize=10)

    ax2 = axes[1]
    if n_hit:
        c = t[hit][0]
    else:
        c = t0
    m = (dense > c - 4) & (dense < c + 4)
    ax2.plot(dense[m], dense_train[m], lw=1.2, color="0.4",
             label="flare profile")
    sel = (t > c - 4) & (t < c + 4)
    ax2.plot(t[sel], train[sel], "o", ms=6, color="crimson",
             label="sampled points")
    ax2.set_xlabel("t (d)")
    ax2.set_ylabel("injected amplitude (mag)")
    ax2.legend(fontsize=8)
    ax2.set_title("zoom on one flare, 1.33 d total width against 3 d cadence",
                  fontsize=10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")
    return ok, n_hit, peak


def measure_RA(t, sigma, freq, det_hs, thr, A_grid, label="R(A)"):
    P_grid = np.array([P_PHYS])
    res = {}
    rng = np.random.default_rng(SEED)
    res["global_max"] = recovery_grid(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid, A_grid,
        freq, thr["global_max"], gls_peak_power, CFG["n_inj"], FWHM_PHYS, rng,
        template=TEMPLATE)[:, 0]
    rng = np.random.default_rng(SEED)
    res["blind_harmonic_sum"] = recovery_grid(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid, A_grid,
        freq, thr["blind_harmonic_sum"], det_hs, CFG["n_inj"], FWHM_PHYS, rng,
        template=TEMPLATE)[:, 0]
    _, R_or, thr_or = oracle_recovery(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid, A_grid,
        freq, CFG["fap"], CFG["n_null"], CFG["n_inj"], FWHM_PHYS, seed=SEED,
        template=TEMPLATE)
    res["oracle_single_freq"] = R_or[:, 0]
    print(f"[{label}] done; oracle threshold at 1/{P_PHYS:g} d = "
          f"{list(thr_or.values())[0]:.4f}")
    return res, float(list(thr_or.values())[0])


def floor_at_half(A_grid, R):
    for i in range(1, len(R)):
        if R[i - 1] < 0.5 <= R[i]:
            x0, x1 = np.log10(A_grid[i - 1]), np.log10(A_grid[i])
            y0, y1 = R[i - 1], R[i]
            return 10.0 ** (x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    if R[0] >= 0.5:
        return float(A_grid[0])
    return None


def load_v1_RA():
    out = {d: {} for d in DETECTORS}
    try:
        with open(RESULTS_DIR / "qpls_template_bridge.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["comparison"] != "amplitude_sweep" or r["template"] != "gaussian":
                    continue
                out[r["detector"]][float(r["A_mag"])] = float(r["recovery_fraction"])
    except FileNotFoundError:
        return None
    return {d: (np.array(sorted(v)), np.array([v[k] for k in sorted(v)]))
            for d, v in out.items()}


def fig_RA(A_grid, res_v2, v1, path, dpi=300):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    titles = {"global_max": "global-max GLS",
              "oracle_single_freq": "oracle single-freq (optimistic control)",
              "blind_harmonic_sum": "blind harmonic-sum"}
    n = CFG["n_inj"]
    for ax, det in zip(axes, DETECTORS):
        R2 = res_v2[det]
        se2 = np.sqrt(R2 * (1 - R2) / n)
        ax.plot(A_grid, R2, "-o", ms=4, color="crimson",
                label="v2 physical (180 d, 16 hr, sawtooth)")
        ax.fill_between(A_grid, R2 - se2, R2 + se2, color="crimson", alpha=0.18)
        if v1 and len(v1[det][0]):
            A1, R1 = v1[det]
            ax.plot(A1, R1, "--s", ms=4, color="C0",
                    label="v1 frozen (18 d, 4 d, Gaussian)")
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("injected amplitude $A$ (mag)")
        ax.set_ylabel("recovery fraction")
        ax.set_title(titles[det], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Recovery vs amplitude: physical regime (v2) against the frozen "
                 "statistic (v1)\n"
                 f"quiescent host, FAP={CFG['fap']}, n_inj={CFG['n_inj']}, "
                 f"seed={SEED}; identical frequency grid and thresholds",
                 fontsize=10)
    fig.text(0.5, 0.005,
             "v2 duty cycle is ~6x lower (0.67 d flare every 180 d vs 4 d every "
             "18 d) and 180 d sits near the semi-annual alias.",
             ha="center", va="bottom", fontsize=8)
    fig.get_layout_engine().set(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Physical-regime statistic v2.")
    ap.parse_args(argv)

    t, sigma, freq, det_hs = setup()
    thr = thresholds(t, sigma, freq, det_hs)
    for k, v1 in COMMITTED_THR_V1.items():
        if abs(thr[k] - v1) >= 5e-4:
            print(f"\nSTOP: {k} threshold moved ({thr[k]:.4f} vs {v1}) even "
                  f"though the grid is unchanged, wiring fault.")
            return 1

    ok, n_hit, peak = wiring_check(t, sigma, FIG_DIR / "fig9_wiring_check.png")
    if not ok:
        print("\nSTOP: no injected flare intersects any observation.")
        return 1

    print("\n=== P2.4 R(A) in the physical regime ===")
    res, thr_oracle = measure_RA(t, sigma, freq, det_hs, thr, A_SWEEP)
    for det in DETECTORS:
        R = res[det]
        if not np.all((R >= 0) & (R <= 1)):
            print(f"\nSTOP: {det} produced R outside [0,1].")
            return 1

    v1 = load_v1_RA()
    fig_RA(A_SWEEP, res, v1, FIG_DIR / "fig9_RA_v2.png")

    print("\n  per-detector v2 floors (A at R = 0.5):")
    floors = {}
    for det in DETECTORS:
        f2 = floor_at_half(A_SWEEP, res[det])
        floors[det] = f2
        f1 = floor_at_half(*v1[det]) if v1 else None
        s2 = f"{f2:.4f} mag" if f2 else "none (never reaches 0.5)"
        s1 = f"{f1:.4f} mag" if f1 else "none"
        ratio = (f"{f2/f1:.1f}x" if (f2 and f1) else "n/a")
        print(f"    {det:19s} v1 {s1:28s} v2 {s2:28s} {ratio}")

    print("\n=== P2.5 SPOT-CHECK: 4 cells run in full under v2 ===")
    cells = [scenario_cell(s, 2.0e10, 0.0, 19.0, dl, dm)
             for s, dl, dm in SPOT_CELLS]
    A_spot = np.array([c["delta_m_peak"] for c in cells])
    res_spot, _ = measure_RA(t, sigma, freq, det_hs, thr, A_spot, "spot")

    bad = 0
    rows = []
    for det in DETECTORS:
        pred = np.interp(np.log10(A_spot), np.log10(A_SWEEP), res[det])
        for c, a, p, m in zip(cells, A_spot, pred, res_spot[det]):
            se = np.sqrt(max(m * (1 - m), 1e-6) / CFG["n_inj"])
            okc = abs(m - p) <= 2 * max(se, 1e-3)
            bad += 0 if okc else 1
            print(f"  {det:19s} {c['source']:7s} D_L={c['D_L_Gpc']:4.2f} "
                  f"A={a:.4f}  direct={m:.3f}  mapped={p:.3f}  "
                  f"|d|={abs(m-p):.3f}  2se={2*se:.3f}  "
                  f"{'ok' if okc else 'MISMATCH'}")
            rows.append(dict(statistic=STATISTIC, kind="spot_check",
                             detector=det, source=c["source"],
                             D_L_Gpc=c["D_L_Gpc"], A_V=0.0, m_nuc=19.0,
                             M_total_Msun=2.0e10, A_mag=f"{a:.6f}",
                             recovery_fraction=f"{m:.4f}",
                             mapped_prediction=f"{p:.4f}",
                             binom_se=f"{se:.4f}",
                             P_days=P_PHYS, fwhm_days=FWHM_PHYS,
                             template=TEMPLATE, threshold=f"{thr.get(det, thr_oracle):.6f}",
                             n_inj=CFG["n_inj"], seed=SEED))
    if bad:
        print(f"\nSTOP: {bad} spot-check cell(s) outside 2x binomial error.")
        return 1
    print("  all 4 cells x 3 detectors within 2x binomial error: PASS")

    for det in DETECTORS:
        for a, R in zip(A_SWEEP, res[det]):
            rows.append(dict(statistic=STATISTIC, kind="amplitude_sweep",
                             detector=det, source="", D_L_Gpc="", A_V="",
                             m_nuc="", M_total_Msun="", A_mag=f"{a:.6f}",
                             recovery_fraction=f"{R:.4f}", mapped_prediction="",
                             binom_se=f"{np.sqrt(R*(1-R)/CFG['n_inj']):.4f}",
                             P_days=P_PHYS, fwhm_days=FWHM_PHYS,
                             template=TEMPLATE,
                             threshold=f"{thr.get(det, thr_oracle):.6f}",
                             n_inj=CFG["n_inj"], seed=SEED))

    out = RESULTS_DIR / "qpls_statistic_v2.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {out}  ({len(rows)} rows, all statistic={STATISTIC})")

    with open(RESULTS_DIR / "qpls_statistic_v2_floors.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["statistic", "detector", "floor_A_at_R0.5_mag"])
        for det in DETECTORS:
            w.writerow([STATISTIC, det,
                        f"{floors[det]:.6f}" if floors[det] else "none"])
    print(f"saved -> {RESULTS_DIR/'qpls_statistic_v2_floors.csv'}")
    print(f"\nRegeneration (seed={SEED}):\n  python -m qpls.statistic_v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
