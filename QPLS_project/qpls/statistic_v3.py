
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
from qpls.source_scenarios import SOURCE_LIBRARY

POSTMS_DIR = RESULTS_DIR / "Post-MS_giant_results"
FIG_DIR = POSTMS_DIR

STATISTIC = "v3-physical"
SEED = 42
TEMPLATE = "sawtooth"
RISE_FRAC = 0.1

M10 = 2.0
T_YR = 1.0
ECC = 0.0
D_KPC = 1.0
T_MAG_REF_HR = 16.0

P_V3 = 365.25 * T_YR / 4.0
P_FOLD_NOTE = 365.25 * T_YR / 8.0

SOURCES_V3 = ["RGBiso", "AGBiso", "BSG"]
A_SWEEP = np.geomspace(0.02, 3.0, 12)
DETECTORS = ["global_max", "oracle_single_freq", "blind_harmonic_sum"]
COMMITTED_THR = {"global_max": 0.4277, "blind_harmonic_sum": 0.8047}

STOP_IF_UNSAMPLED = {"RGBiso", "AGBiso"}

CHECKPOINTS_HR = {"RGBiso": 196.0, "BSG": 28.5}
TOL = 0.05


def t_mag_hours(R_Rsun, M10_=M10, T_yr=T_YR, e=ECC, D_kpc=D_KPC):
    return (T_MAG_REF_HR * (1.0 - e ** 2) ** -0.5 * M10_ ** (-1.0 / 6.0)
            * T_yr ** (-1.0 / 3.0) * (R_Rsun / 10.0) * D_kpc ** 0.5)


def setup(verbose=True):
    _ir.RNG.bit_generator.state = np.random.default_rng(0).bit_generator.state
    t, sigma = ztf_cadence(CFG["baseline_yr"] * 365.25, CFG["cadence_days"],
                           CFG["season_frac"], err_med=CFG["err_med"])
    base = CFG["baseline_yr"] * 365.25
    freq = np.linspace(1.0 / (0.5 * base), 1.0 / (2 * CFG["cadence_days"]), 1000)
    det_hs = partial(gls_harmonic_sum, n_harmonics=CFG["n_harmonics"])
    if verbose:
        print(f"baseline {base:.2f} d, {t.size} observations, "
              f"recurrence P = {P_V3:.4f} d (T/4, cusp regime)")
        print(f"  n_flares over baseline = {base/P_V3:.2f}")
        print(f"  (fold variant T/8 = {P_FOLD_NOTE:.2f} d is NOT run today)")
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
        for k, v in COMMITTED_THR.items():
            ok = abs(thr[k] - v) < 5e-4
            print(f"  [threshold] {k:19s} {thr[k]:.4f} vs committed {v} -> "
                  f"{'PASS' if ok else 'FAIL'}")
    return thr


def step1_table():
    print("\n=== STEP 1: Eq. 6 widths (M10=2, T=1 yr, e=0, D_LS=1 kpc) ===")
    print(f"  prefactor 16 * 2^(-1/6) = {T_MAG_REF_HR * M10**(-1/6):.4f} hr "
          f"per 10 Rsun")
    print(f"  {'source':8s} {'L (Lsun)':>10s} {'R (Rsun)':>9s} "
          f"{'t_mag (hr)':>11s} {'t_mag (d)':>10s} {'duty %':>8s}")
    widths, ok_all = {}, True
    for s in SOURCES_V3:
        R = SOURCE_LIBRARY[s]["R"]
        hr = t_mag_hours(R)
        d = hr / 24.0
        widths[s] = d
        duty = 2.0 * d / P_V3 * 100.0
        print(f"  {s:8s} {SOURCE_LIBRARY[s]['L']:10.6g} {R:9.4g} "
              f"{hr:11.2f} {d:10.4f} {duty:8.2f}")
    print("\n  checkpoints (+/-5%):")
    for s, exp in CHECKPOINTS_HR.items():
        got = t_mag_hours(SOURCE_LIBRARY[s]["R"])
        dev = abs(got - exp) / exp
        ok_all &= dev <= TOL
        print(f"    {s:8s} expected {exp:7.1f} hr   got {got:7.2f} hr   "
              f"dev {dev*100:4.2f}%   {'PASS' if dev <= TOL else 'FAIL'}")
    R_agb = SOURCE_LIBRARY["AGBiso"]["R"]
    formula = 16.0 * 0.891 * (R_agb / 10.0)
    got = t_mag_hours(R_agb)
    dev = abs(got - formula) / formula
    ok_all &= dev <= TOL
    print(f"    {'AGBiso':8s} formula 16*0.891*(R/10) = {formula:7.2f} hr   "
          f"got {got:7.2f} hr   dev {dev*100:4.2f}%   "
          f"{'PASS' if dev <= TOL else 'FAIL'}")
    return widths, ok_all


def wiring_figure(t, sigma, widths, path, A=1.0, dpi=300):
    fig, axes = plt.subplots(len(SOURCES_V3), 1, figsize=(11, 3.0 * len(SOURCES_V3)),
                             constrained_layout=True, sharex=True)
    hits = {}
    for ax, s in zip(axes, SOURCES_V3):
        fw = widths[s]
        rng = np.random.default_rng(SEED)
        t0 = rng.uniform(0, P_V3)
        train = flare_train_sawtooth(t, A, P_V3, t0, 2.0 * fw, RISE_FRAC)
        n_hit = int((train > 0).sum())
        hits[s] = n_hit
        rng2 = np.random.default_rng(SEED)
        mag = make_lc(t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"],
                      P_obs=P_V3, A_mag=A, fwhm_days=fw, rng=rng2,
                      template=TEMPLATE)
        dense = np.linspace(t.min(), t.max(), 200000)
        dtr = flare_train_sawtooth(dense, A, P_V3, t0, 2.0 * fw, RISE_FRAC)
        ax.plot(dense, -dtr, lw=0.7, color="0.75", label="injected train")
        ax.plot(t, mag, ".", ms=3, color="C0", label="sampled")
        m = train > 0
        ax.plot(t[m], mag[m], "o", ms=6, mfc="none", mec="crimson", mew=1.2,
                label=f"in-flare samples ({n_hit})")
        ax.invert_yaxis()
        ax.set_ylabel("mag")
        ax.legend(fontsize=7, loc="lower right", ncol=3)
        ax.set_title(f"{s}: R = {SOURCE_LIBRARY[s]['R']:g} Rsun, "
                     f"t_mag = {fw*24:.1f} hr = {fw:.2f} d, "
                     f"duty = {2*fw/P_V3*100:.1f}%   ({n_hit} of {t.size} "
                     f"observations inside a flare)", fontsize=9)
    axes[-1].set_xlabel("t (d)")
    fig.suptitle(f"v3 wiring check: source-specific Eq. 6 widths, "
                 f"P = {P_V3:.2f} d (T/4 cusp), A = {A:g} mag", fontsize=10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")
    return hits


def measure(t, sigma, freq, det_hs, thr, fwhm_days, A_grid, label):
    P_grid = np.array([P_V3])
    res = {}
    rng = np.random.default_rng(SEED)
    res["global_max"] = recovery_grid(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid, A_grid,
        freq, thr["global_max"], gls_peak_power, CFG["n_inj"], fwhm_days, rng,
        template=TEMPLATE)[:, 0]
    rng = np.random.default_rng(SEED)
    res["blind_harmonic_sum"] = recovery_grid(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid, A_grid,
        freq, thr["blind_harmonic_sum"], det_hs, CFG["n_inj"], fwhm_days, rng,
        template=TEMPLATE)[:, 0]
    _, R_or, thr_or = oracle_recovery(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid, A_grid,
        freq, CFG["fap"], CFG["n_null"], CFG["n_inj"], fwhm_days, seed=SEED,
        template=TEMPLATE)
    res["oracle_single_freq"] = R_or[:, 0]
    print(f"  [{label}] done", flush=True)
    return res, float(list(thr_or.values())[0])


def floor_at_half(A_grid, R):
    for i in range(1, len(R)):
        if R[i - 1] < 0.5 <= R[i]:
            x0, x1 = np.log10(A_grid[i - 1]), np.log10(A_grid[i])
            return 10.0 ** (x0 + (0.5 - R[i - 1]) * (x1 - x0) / (R[i] - R[i - 1]))
    return float(A_grid[0]) if R[0] >= 0.5 else None


def _load_sweep(path, filt):
    out = {d: {} for d in DETECTORS}
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if not filt(r):
                    continue
                out[r["detector"]][float(r["A_mag"])] = float(r["recovery_fraction"])
    except FileNotFoundError:
        return None
    return {d: (np.array(sorted(v)), np.array([v[k] for k in sorted(v)]))
            for d, v in out.items() if v}


def fig_RA(res_by_src, v1, v2, path, dpi=300):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    titles = {"global_max": "global-max GLS",
              "oracle_single_freq": "oracle single-freq (optimistic control)",
              "blind_harmonic_sum": "blind harmonic-sum"}
    colours = {"RGBiso": "darkgreen", "AGBiso": "darkorange", "BSG": "purple"}
    for ax, det in zip(axes, DETECTORS):
        if v1 and det in v1:
            ax.plot(*v1[det], "--", color="0.55", lw=1.2,
                    label="v1 frozen (18 d, 4 d Gaussian)")
        if v2 and det in v2:
            ax.plot(*v2[det], ":", color="crimson", lw=1.4,
                    label="v2 (180 d, flat 16 hr)")
        for s in SOURCES_V3:
            ax.plot(A_SWEEP, res_by_src[s][det], "-o", ms=4, color=colours[s],
                    label=f"v3 {s} ({t_mag_hours(SOURCE_LIBRARY[s]['R'])/24:.2f} d)")
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("injected amplitude $A$ (mag)")
        ax.set_ylabel("recovery fraction")
        ax.set_title(titles[det], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Recovery vs amplitude with source-specific Eq. 6 crossing "
                 f"widths (v3), P = {P_V3:.2f} d (T/4 cusp)\n"
                 f"quiescent host, FAP={CFG['fap']}, n_inj={CFG['n_inj']}, "
                 f"seed={SEED}; thresholds unchanged from v1/v2", fontsize=10)
    fig.text(0.5, 0.005,
             "ROLE INVERSION: under physical widths the compact BSG control is "
             "the HARD case (width ~ R), the opposite of its v1 role. "
             "Radius enters twice: amplitude ~ R^-0.64, width ~ R.",
             ha="center", va="bottom", fontsize=8)
    fig.get_layout_engine().set(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Statistic v3: Eq. 6 widths.")
    ap.parse_args(argv)

    POSTMS_DIR.mkdir(parents=True, exist_ok=True)
    t, sigma, freq, det_hs = setup()
    thr = thresholds(t, sigma, freq, det_hs)
    for k, v in COMMITTED_THR.items():
        if abs(thr[k] - v) >= 5e-4:
            print(f"\nSTOP: {k} threshold moved, wiring fault.")
            return 1

    widths, ok = step1_table()
    if not ok:
        print("\nSTOP: Eq. 6 checkpoint outside 5%. Nothing hand-tuned.")
        return 1

    hits = wiring_figure(t, sigma, widths, FIG_DIR / "fig11_wiring_v3.png")
    print("\n=== STEP 2 wiring verdicts ===")
    for s in SOURCES_V3:
        n = hits[s]
        if n == 0 and s in STOP_IF_UNSAMPLED:
            print(f"  {s:8s} {n:4d} in-flare samples -> STOP (giant source must "
                  f"intersect the cadence)")
            return 1
        note = ("expected physics for a compact source, recorded not fatal"
                if (n == 0 or s not in STOP_IF_UNSAMPLED) else "ok")
        print(f"  {s:8s} {n:4d} in-flare samples   {note}")

    print("\n=== STEP 2 R(A) per source ===")
    res_by_src, rows = {}, []
    for s in SOURCES_V3:
        res, thr_or = measure(t, sigma, freq, det_hs, thr, widths[s], A_SWEEP, s)
        for det in DETECTORS:
            R = res[det]
            if not np.all((R >= 0) & (R <= 1)):
                print(f"\nSTOP: {s}/{det} produced R outside [0,1].")
                return 1
        res_by_src[s] = res
        for det in DETECTORS:
            for a, R in zip(A_SWEEP, res[det]):
                rows.append(dict(
                    statistic=STATISTIC, source=s,
                    R_Rsun=SOURCE_LIBRARY[s]["R"], L_Lsun=SOURCE_LIBRARY[s]["L"],
                    t_mag_hr=f"{widths[s]*24:.6g}", t_mag_days=f"{widths[s]:.6g}",
                    P_days=f"{P_V3:.6g}", detector=det, A_mag=f"{a:.6f}",
                    recovery_fraction=f"{R:.4f}",
                    binom_se=f"{np.sqrt(R*(1-R)/CFG['n_inj']):.4f}",
                    threshold=f"{thr.get(det, thr_or):.6f}",
                    template=TEMPLATE, n_inj=CFG["n_inj"], seed=SEED))

    v1 = _load_sweep(RESULTS_DIR / "qpls_template_bridge.csv",
                     lambda r: r["comparison"] == "amplitude_sweep"
                     and r["template"] == "gaussian")
    v2 = _load_sweep(RESULTS_DIR / "qpls_statistic_v2.csv",
                     lambda r: r["kind"] == "amplitude_sweep")
    fig_RA(res_by_src, v1, v2, FIG_DIR / "fig11_RA_v3.png")

    print("\n=== STEP 2 floors (A at R = 0.5) ===")
    v1f = {d: floor_at_half(*v1[d]) for d in DETECTORS} if v1 else {}
    floors = {}
    print(f"  {'source':8s} {'detector':19s} {'v1 floor':>12s} {'v3 floor':>12s}  ratio")
    for s in SOURCES_V3:
        for det in DETECTORS:
            f3 = floor_at_half(A_SWEEP, res_by_src[s][det])
            floors[(s, det)] = f3
            f1 = v1f.get(det)
            r = f"{f3/f1:.2f}x" if (f3 and f1) else "n/a"
            print(f"  {s:8s} {det:19s} "
                  f"{(f'{f1:.4f}' if f1 else 'none'):>12s} "
                  f"{(f'{f3:.4f}' if f3 else 'none'):>12s}  {r}")

    out = POSTMS_DIR / "qpls_statistic_v3.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {out}  ({len(rows)} rows, all statistic={STATISTIC})")

    fp = POSTMS_DIR / "qpls_statistic_v3_floors.csv"
    with open(fp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["statistic", "source", "R_Rsun", "t_mag_days", "detector",
                    "floor_A_at_R0.5_mag"])
        for s in SOURCES_V3:
            for det in DETECTORS:
                f3 = floors[(s, det)]
                w.writerow([STATISTIC, s, SOURCE_LIBRARY[s]["R"],
                            f"{widths[s]:.6g}", det,
                            f"{f3:.6f}" if f3 else "none"])
    print(f"saved -> {fp}")
    print(f"\nRegeneration (seed={SEED}):\n  python -m qpls.statistic_v3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
