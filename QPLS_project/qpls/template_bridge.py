
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
    null_threshold, recovery_grid,
)
from qpls.detector_comparison import CFG, oracle_recovery
from qpls.source_scenarios import scenario_cell

FIG_DIR = RESULTS_DIR.parent / "figures"

SEED = 42
P_INJECT = 18.0071
TEMPLATES = ["gaussian", "sawtooth"]
A_SWEEP = np.geomspace(0.02, 3.0, 10)

DETECTORS = ["global_max", "oracle_single_freq", "blind_harmonic_sum"]
COMMITTED_THR = {"global_max": 0.4277, "blind_harmonic_sum": 0.8047}

ROW_D_L = [(0.1, 35.00), (0.2, 36.51), (0.35, 37.72), (0.5, 38.49),
           (0.7, 39.23), (0.85, 39.65), (1.0, 40.00), (2.0, 41.51),
           (2.9, 42.31)]

STOP_DIFF = 0.15

EXPECTATION = (
    "Differences within a few percent, harmonic-sum >= Gaussian, global-max <= "
    "Gaussian; a null result is the expected outcome and would demonstrate "
    "template shape is second-order for power-spectrum detectors (time-reversal "
    "invariance), directing future effort to the duty-cycle/period axis instead.")


def setup():
    _ir.RNG.bit_generator.state = np.random.default_rng(0).bit_generator.state
    t, sigma = ztf_cadence(CFG["baseline_yr"] * 365.25, CFG["cadence_days"],
                           CFG["season_frac"], err_med=CFG["err_med"])
    freq = np.linspace(1.0 / (0.5 * CFG["baseline_yr"] * 365.25),
                       1.0 / (2 * CFG["cadence_days"]), 1000)
    det_hs = partial(gls_harmonic_sum, n_harmonics=CFG["n_harmonics"])

    thr = {}
    thr["global_max"], _ = null_threshold(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], freq,
        gls_peak_power, CFG["fap"], CFG["n_null"], np.random.default_rng(SEED))
    thr["blind_harmonic_sum"], _ = null_threshold(
        t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], freq,
        det_hs, CFG["fap"], CFG["n_null"], np.random.default_rng(SEED))
    for k, exp in COMMITTED_THR.items():
        status = "PASS" if abs(thr[k] - exp) < 5e-4 else "FAIL"
        print(f"[threshold] {k:19s} {thr[k]:.4f} vs committed {exp} -> {status}")
        if status == "FAIL":
            raise SystemExit("STOP: threshold non-reproduction (true error).")
    return t, sigma, freq, det_hs, thr


def sweep(t, sigma, freq, det_hs, thr, A_grid, label):
    out = {}
    P_grid = np.array([P_INJECT])
    for template in TEMPLATES:
        res = {}
        rng = np.random.default_rng(SEED)
        res["global_max"] = recovery_grid(
            t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid,
            A_grid, freq, thr["global_max"], gls_peak_power, CFG["n_inj"],
            CFG["fwhm_days"], rng, template=template)[:, 0]
        rng = np.random.default_rng(SEED)
        res["blind_harmonic_sum"] = recovery_grid(
            t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid,
            A_grid, freq, thr["blind_harmonic_sum"], det_hs, CFG["n_inj"],
            CFG["fwhm_days"], rng, template=template)[:, 0]
        _, R_or, thr_or = oracle_recovery(
            t, sigma, CFG["tau_drw_days"], CFG["sf_inf_quiescent"], P_grid,
            A_grid, freq, CFG["fap"], CFG["n_null"], CFG["n_inj"],
            CFG["fwhm_days"], seed=SEED, template=template)
        res["oracle_single_freq"] = R_or[:, 0]
        out[template] = res
        print(f"[{label}] {template} done", flush=True)
    return out


def check_stop(out, A_grid, label):
    worst, bad = 0.0, []
    for det in DETECTORS:
        d = out["sawtooth"][det] - out["gaussian"][det]
        for a, dv in zip(A_grid, d):
            if abs(dv) > worst:
                worst = abs(dv)
            if abs(dv) > STOP_DIFF:
                bad.append((det, float(a), float(dv)))
    print(f"[{label}] max |paired difference| = {worst:.3f} "
          f"(stop threshold {STOP_DIFF})")
    if bad:
        print(f"STOP: {len(bad)} paired difference(s) exceed {STOP_DIFF}, "
              f"wiring suspect, not a discovery. No commit.")
        for det, a, dv in bad:
            print(f"   {det:19s} A={a:.4f}  diff={dv:+.3f}")
    return bad, worst


def write_csv(path, rows):
    cols = ["comparison", "template", "detector", "x_label", "x_value",
            "A_mag", "recovery_fraction", "binom_se", "n_inj", "seed",
            "threshold", "P_inject_days", "fwhm_days"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {path}  ({len(rows)} rows)")


def plot_bridge(A_grid, out, path, dpi=300):
    n = CFG["n_inj"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), constrained_layout=True,
                             sharex=True)
    titles = {"global_max": "global-max GLS",
              "oracle_single_freq": "oracle single-freq (optimistic control)",
              "blind_harmonic_sum": "blind harmonic-sum"}
    for j, det in enumerate(DETECTORS):
        ax = axes[0, j]
        for template, style in [("gaussian", "-o"), ("sawtooth", "--s")]:
            R = out[template][det]
            se = np.sqrt(R * (1 - R) / n)
            ax.plot(A_grid, R, style, ms=4, label=template)
            ax.fill_between(A_grid, R - se, R + se, alpha=0.18)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(titles[det], fontsize=10)
        ax.set_ylabel("recovery fraction")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        axd = axes[1, j]
        d = out["sawtooth"][det] - out["gaussian"][det]
        se_d = np.sqrt(out["gaussian"][det] * (1 - out["gaussian"][det]) / n
                       + out["sawtooth"][det] * (1 - out["sawtooth"][det]) / n)
        axd.axhline(0.0, color="k", lw=0.8)
        axd.fill_between(A_grid, -se_d, se_d, alpha=0.20,
                         label="$\\pm 1\\sigma$ binomial")
        axd.plot(A_grid, d, "-o", ms=4, color="crimson",
                 label="sawtooth $-$ gaussian")
        axd.set_xscale("log")
        axd.set_ylim(-0.25, 0.25)
        axd.axhline(STOP_DIFF, color="grey", ls=":", lw=0.8)
        axd.axhline(-STOP_DIFF, color="grey", ls=":", lw=0.8)
        axd.set_xlabel("injected amplitude $A$ (mag)")
        axd.set_ylabel("paired $\\Delta R$")
        axd.grid(alpha=0.3)
        axd.legend(fontsize=8)

    fig.suptitle("Template bridge: Gaussian vs sawtooth, paired seeds\n"
                 f"quiescent host, $P_{{inj}}={P_INJECT}$ d, FWHM=4 d matched, "
                 f"FAP={CFG['fap']}, n_inj={CFG['n_inj']}, seed={SEED}; "
                 "thresholds NOT recalibrated (nulls are template-independent)",
                 fontsize=10)
    fig.text(0.5, 0.005,
             "Dotted grey lines mark the +-0.15 paired-difference stop "
             "threshold. Shaded bands are binomial 1-sigma at n=200.",
             ha="center", va="bottom", fontsize=8)
    fig.get_layout_engine().set(rect=(0.0, 0.04, 1.0, 0.96))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gaussian vs sawtooth bridge.")
    ap.parse_args(argv)

    print("PRE-REGISTERED EXPECTATION (verbatim):")
    print(f"  {EXPECTATION}\n")

    t, sigma, freq, det_hs, thr = setup()
    rows = []

    print("\n=== (a) amplitude sweep ===")
    out_a = sweep(t, sigma, freq, det_hs, thr, A_SWEEP, "sweep")
    bad_a, worst_a = check_stop(out_a, A_SWEEP, "sweep")

    for template in TEMPLATES:
        for det in DETECTORS:
            for a, R in zip(A_SWEEP, out_a[template][det]):
                rows.append(dict(
                    comparison="amplitude_sweep", template=template,
                    detector=det, x_label="A_mag", x_value=f"{a:.6f}",
                    A_mag=f"{a:.6f}", recovery_fraction=f"{R:.4f}",
                    binom_se=f"{np.sqrt(R*(1-R)/CFG['n_inj']):.4f}",
                    n_inj=CFG["n_inj"], seed=SEED,
                    threshold=f"{thr.get(det, float('nan')):.6f}"
                    if det in thr else "", P_inject_days=P_INJECT,
                    fwhm_days=CFG["fwhm_days"]))

    print("\n=== (b) reach-map row: RGB, A_V=0, M=2e10, nine D_L ===")
    cells = [scenario_cell("RGB", 2.0e10, 0.0, 19.0, dl, dm) for dl, dm in ROW_D_L]
    A_row = np.array([c["delta_m_peak"] for c in cells])
    for c in cells:
        print(f"   D_L={c['D_L_Gpc']:4.2f} Gpc  dm_peak={c['delta_m_peak']:.5f}")
    out_b = sweep(t, sigma, freq, det_hs, thr, A_row, "row")
    bad_b, worst_b = check_stop(out_b, A_row, "row")

    for template in TEMPLATES:
        for det in DETECTORS:
            for c, R in zip(cells, out_b[template][det]):
                rows.append(dict(
                    comparison="reach_row_RGB_Av0", template=template,
                    detector=det, x_label="D_L_Gpc",
                    x_value=f"{c['D_L_Gpc']:g}",
                    A_mag=f"{c['delta_m_peak']:.6f}",
                    recovery_fraction=f"{R:.4f}",
                    binom_se=f"{np.sqrt(R*(1-R)/CFG['n_inj']):.4f}",
                    n_inj=CFG["n_inj"], seed=SEED,
                    threshold=f"{thr.get(det, float('nan')):.6f}"
                    if det in thr else "", P_inject_days=P_INJECT,
                    fwhm_days=CFG["fwhm_days"]))

    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    write_csv(RESULTS_DIR / "qpls_template_bridge.csv", rows)

    if bad_a or bad_b:
        print("\nSTOP CONDITION TRIGGERED, so figure not written, do not commit.")
        return 1

    plot_bridge(A_SWEEP, out_a, FIG_DIR / "fig8_template_bridge.png")

    print("\n" + "=" * 78)
    print("TEMPLATE BRIDGE REPORT")
    print("=" * 78)
    print(f"max |paired dR|: sweep {worst_a:.3f}, reach row {worst_b:.3f} "
          f"(stop threshold {STOP_DIFF})")
    print("\nSigned mean paired difference (sawtooth - gaussian):")
    for det in DETECTORS:
        d_a = out_a["sawtooth"][det] - out_a["gaussian"][det]
        d_b = out_b["sawtooth"][det] - out_b["gaussian"][det]
        print(f"  {det:19s} sweep {d_a.mean():+.4f}  row {d_b.mean():+.4f}  "
              f"(sweep max |d| {np.abs(d_a).max():.3f})")
    print("\nDirectional pre-registration check:")
    d_hs = (out_a['sawtooth']['blind_harmonic_sum']
            - out_a['gaussian']['blind_harmonic_sum']).mean()
    d_gm = (out_a['sawtooth']['global_max']
            - out_a['gaussian']['global_max']).mean()
    print(f"  harmonic-sum >= Gaussian predicted: mean dR = {d_hs:+.4f} -> "
          f"{'consistent' if d_hs >= 0 else 'NOT consistent'}")
    print(f"  global-max   <= Gaussian predicted: mean dR = {d_gm:+.4f} -> "
          f"{'consistent' if d_gm <= 0 else 'NOT consistent'}")
    print(f"\nRegeneration (seed={SEED}):\n  python -m qpls.template_bridge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
