
import argparse
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qpls.injection_recovery import RESULTS_DIR
from qpls.source_scenarios import SOURCE_LIBRARY
from qpls.statistic_v3 import (
    POSTMS_DIR, SEED, A_SWEEP, DETECTORS, COMMITTED_THR, P_V3, P_FOLD_NOTE,
    TEMPLATE, RISE_FRAC, t_mag_hours, setup, thresholds, measure,
    floor_at_half, wiring_figure,
)

STATISTIC = "v3-physical"
BLIND = ["global_max", "blind_harmonic_sum"]

NEW_SOURCES = ["AGB170", "AGB215", "AGB260"]
ANCHORS = ["RGBiso", "AGB170", "AGB215", "AGB260", "AGBiso"]

CHECKPOINTS_D = {"AGB170": 10.1, "AGB215": 12.8, "AGB260": 15.4}
TOL = 0.05


def step2_table():
    print("\n=== STEP 2: Eq. 6 widths for the new anchors ===")
    print(f"  {'source':8s} {'L (Lsun)':>10s} {'R (Rsun)':>9s} "
          f"{'t_mag (hr)':>11s} {'t_mag (d)':>10s} {'duty %':>8s}")
    widths, ok = {}, True
    for s in NEW_SOURCES:
        R = SOURCE_LIBRARY[s]["R"]
        d = t_mag_hours(R) / 24.0
        widths[s] = d
        print(f"  {s:8s} {SOURCE_LIBRARY[s]['L']:10.6g} {R:9.4g} "
              f"{d*24:11.2f} {d:10.4f} {2*d/P_V3*100:8.2f}")
    print("\n  checkpoints (+/-5%):")
    for s, exp in CHECKPOINTS_D.items():
        got = widths[s]
        dev = abs(got - exp) / exp
        ok &= dev <= TOL
        print(f"    {s:8s} expected {exp:6.2f} d   got {got:6.3f} d   "
              f"dev {dev*100:4.2f}%   {'PASS' if dev <= TOL else 'FAIL'}")
    return widths, ok


def load_v3_curves(path=None):
    path = path or POSTMS_DIR / "qpls_statistic_v3.csv"
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["source"], {}).setdefault(r["detector"], {})[
                float(r["A_mag"])] = float(r["recovery_fraction"])
    return {s: {d: np.array([v[k] for k in sorted(v)]) for d, v in dv.items()}
            for s, dv in out.items()}


def r50_from_anchors(radii, floors):
    finite = [(R, f) for R, f in zip(radii, floors) if f is not None]
    if not finite:
        return None
    R_first = finite[0][0]
    i = radii.index(R_first)
    if i == 0:
        return (radii[0], None, radii[0])
    return (0.5 * (radii[i - 1] + radii[i]), radii[i - 1], radii[i])


def fig_radius_scan(curves, widths_all, floors, path, dpi=300):
    radii = [SOURCE_LIBRARY[s]["R"] for s in ANCHORS]
    cmap = plt.get_cmap("viridis")
    cols = {s: cmap(i / (len(ANCHORS) - 1)) for i, s in enumerate(ANCHORS)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, det in zip(axes[:2], BLIND):
        for s in ANCHORS:
            ax.plot(A_SWEEP, curves[s][det], "-o", ms=4, color=cols[s],
                    label=f"{s} ({SOURCE_LIBRARY[s]['R']:.0f} R$_\\odot$, "
                          f"{widths_all[s]:.1f} d)")
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("injected amplitude $A$ (mag)")
        ax.set_ylabel("recovery fraction")
        ax.set_title(det.replace("blind_", "").replace("_", "-"), fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    for det, mk in zip(BLIND, ["-o", "--s"]):
        ax.plot(radii, [max(curves[s][det]) for s in ANCHORS], mk,
                color="C0" if det == "global_max" else "C1",
                label=f"max R, {det.replace('blind_','')}")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xlabel("source radius $R$ (R$_\\odot$)")
    ax.set_ylabel("max blind recovery fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    for det, mk in zip(BLIND, ["-^", "--v"]):
        y = [floors[(s, det)] for s in ANCHORS]
        xs = [r for r, v in zip(radii, y) if v is not None]
        ys = [v for v in y if v is not None]
        if xs:
            ax2.plot(xs, ys, mk, color="crimson" if det == "global_max"
                     else "darkred", ms=5,
                     label=f"floor, {det.replace('blind_','')}")
    ax2.set_ylabel("floor: $A$ at $R=0.5$ (mag)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax.set_title("switch-on with radius", fontsize=10)

    fig.suptitle("Blind recovery against source radius, statistic v3 "
                 f"(Eq. 6 widths, P = {P_V3:.2f} d)\n"
                 "five isochrone anchors: RGB tip 137.5 R$_\\odot$ then four AGB "
                 "points to the AGB tip 300.7 R$_\\odot$", fontsize=10)
    fig.text(0.5, 0.005,
             "Radius enters twice: amplitude ~ R^-0.64 (penalty), crossing width "
             "~ R (samplability bonus). Floors plotted only where R = 0.5 is "
             "actually reached.",
             ha="center", va="bottom", fontsize=8)
    fig.get_layout_engine().set(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="AGB radius scan + T/8 discriminator.")
    ap.parse_args(argv)
    POSTMS_DIR.mkdir(parents=True, exist_ok=True)

    t, sigma, freq, det_hs = setup()
    thr = thresholds(t, sigma, freq, det_hs)
    for k, v in COMMITTED_THR.items():
        if abs(thr[k] - v) >= 5e-4:
            print(f"\nSTOP: {k} threshold moved, wiring fault.")
            return 1

    widths_new, ok = step2_table()
    if not ok:
        print("\nSTOP: Eq. 6 checkpoint outside 5%. Nothing hand-tuned.")
        return 1

    import qpls.statistic_v3 as v3
    saved = v3.SOURCES_V3
    v3.SOURCES_V3 = NEW_SOURCES
    hits = wiring_figure(t, sigma, widths_new,
                         POSTMS_DIR / "fig12_wiring_radius_scan.png")
    v3.SOURCES_V3 = saved
    print("\n  wiring:", {s: hits[s] for s in NEW_SOURCES})
    for s in NEW_SOURCES:
        if hits[s] == 0:
            print(f"\nSTOP: {s} flare train never intersects an observation.")
            return 1

    print("\n=== STEP 3a: R(A) for the new anchors ===")
    rows, curves = [], {}
    for s in NEW_SOURCES:
        res, thr_or = measure(t, sigma, freq, det_hs, thr, widths_new[s],
                              A_SWEEP, s)
        for det in DETECTORS:
            R = res[det]
            if not np.all((R >= 0) & (R <= 1)):
                print(f"\nSTOP: {s}/{det} produced R outside [0,1].")
                return 1
        curves[s] = res
        for det in DETECTORS:
            for a, R in zip(A_SWEEP, res[det]):
                rows.append(dict(statistic=STATISTIC, kind="radius_scan",
                                 source=s, R_Rsun=SOURCE_LIBRARY[s]["R"],
                                 L_Lsun=SOURCE_LIBRARY[s]["L"],
                                 t_mag_days=f"{widths_new[s]:.6g}",
                                 P_days=f"{P_V3:.6g}", detector=det,
                                 A_mag=f"{a:.6f}", recovery_fraction=f"{R:.4f}",
                                 threshold=f"{thr.get(det, thr_or):.6f}",
                                 n_inj=200, seed=SEED))

    print(f"\n=== STEP 3b: T/8 fold discriminator, RGBiso at P = "
          f"{P_FOLD_NOTE:.2f} d ===")
    w_rgb = t_mag_hours(SOURCE_LIBRARY["RGBiso"]["R"]) / 24.0
    v3.P_V3 = P_FOLD_NOTE
    res_fold, thr_or = measure(t, sigma, freq, det_hs, thr, w_rgb, A_SWEEP,
                               "RGBiso T/8")
    v3.P_V3 = P_V3
    for det in DETECTORS:
        for a, R in zip(A_SWEEP, res_fold[det]):
            rows.append(dict(statistic=STATISTIC, kind="fold_discriminator",
                             source="RGBiso", R_Rsun=SOURCE_LIBRARY["RGBiso"]["R"],
                             L_Lsun=SOURCE_LIBRARY["RGBiso"]["L"],
                             t_mag_days=f"{w_rgb:.6g}",
                             P_days=f"{P_FOLD_NOTE:.6g}", detector=det,
                             A_mag=f"{a:.6f}", recovery_fraction=f"{R:.4f}",
                             threshold=f"{thr.get(det, thr_or):.6f}",
                             n_inj=200, seed=SEED))

    committed = load_v3_curves()
    print(f"  {'detector':19s} {'T/4 max R':>10s} {'T/8 max R':>10s}  ratio  verdict")
    for det in BLIND:
        m4 = float(max(committed["RGBiso"][det]))
        m8 = float(max(res_fold[det]))
        ratio = m8 / m4 if m4 > 0 else float("nan")
        verdict = ("cycle-count-limited (longer baselines help)"
                   if ratio >= 1.7 else
                   "red-noise-limited (needs period priors or better detectors)"
                   if ratio <= 1.3 else "ambiguous")
        print(f"  {det:19s} {m4:10.3f} {m8:10.3f}  {ratio:5.2f}  {verdict}")

    for s in ["RGBiso", "AGBiso"]:
        curves[s] = committed[s]
    widths_all = {s: t_mag_hours(SOURCE_LIBRARY[s]["R"]) / 24.0 for s in ANCHORS}
    floors = {(s, det): floor_at_half(A_SWEEP, curves[s][det])
              for s in ANCHORS for det in DETECTORS}

    print("\n=== STEP 3c: five-anchor ladder (blind detectors) ===")
    print(f"  {'source':8s} {'R':>7s} {'t_mag':>7s} "
          f"{'gm max':>7s} {'gm floor':>9s} {'hs max':>7s} {'hs floor':>9s}")
    for s in ANCHORS:
        gm, hs = curves[s]["global_max"], curves[s]["blind_harmonic_sum"]
        fg, fh = floors[(s, "global_max")], floors[(s, "blind_harmonic_sum")]
        print(f"  {s:8s} {SOURCE_LIBRARY[s]['R']:7.1f} {widths_all[s]:7.2f} "
              f"{max(gm):7.3f} {(f'{fg:.4f}' if fg else 'none'):>9s} "
              f"{max(hs):7.3f} {(f'{fh:.4f}' if fh else 'none'):>9s}")

    fig_radius_scan(curves, widths_all, floors,
                    POSTMS_DIR / "fig12_radius_scan.png")

    from qpls.populations import load_isochrone, cumulative_lf, radius_at
    iso = load_isochrone(verbose=False)
    radii = [SOURCE_LIBRARY[s]["R"] for s in ANCHORS]
    print("\n=== STEP 4a: switch-on radius R_50 ===")
    r50 = {}
    for det in BLIND:
        got = r50_from_anchors(radii, [floors[(s, det)] for s in ANCHORS])
        r50[det] = got
        if got is None:
            print(f"  {det:19s} never reaches R = 0.5 at any anchor")
        else:
            R50, lo, hi = got
            br = f"bracketed by {lo:.1f}-{hi:.1f} Rsun" if lo else \
                 f"already on at the smallest anchor ({hi:.1f} Rsun)"
            print(f"  {det:19s} R_50 = {R50:.1f} Rsun   ({br})")

    print("\n=== STEP 4b: blind-detectable giants per 1e6 Msun ===")
    Lg = np.geomspace(1.0, 1.0e4, 6000)
    Rg = radius_at(iso, Lg)
    for det in BLIND:
        if r50[det] is None:
            print(f"  {det:19s} n/a (no switch-on)")
            continue
        R50 = r50[det][0]
        cand = Lg[Rg >= R50]
        if cand.size == 0:
            print(f"  {det:19s} R_50 = {R50:.1f} Rsun exceeds every L on the "
                  f"track -> N = 0")
            continue
        L50 = float(cand.min())
        N = float(cumulative_lf(iso, np.array([L50]))[0])
        print(f"  {det:19s} R_50 = {R50:.1f} Rsun -> L(R_50) = {L50:.1f} Lsun "
              f"-> N(>L) = {N:.4g} per 1e6 Msun")

    out = POSTMS_DIR / "qpls_radius_scan.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {out}  ({len(rows)} rows)")

    fp = POSTMS_DIR / "qpls_radius_scan_floors.csv"
    with open(fp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["statistic", "source", "R_Rsun", "t_mag_days", "detector",
                    "floor_A_at_R0.5_mag", "max_recovery"])
        for s in ANCHORS:
            for det in DETECTORS:
                f = floors[(s, det)]
                w.writerow([STATISTIC, s, SOURCE_LIBRARY[s]["R"],
                            f"{widths_all[s]:.6g}", det,
                            f"{f:.6f}" if f else "none",
                            f"{max(curves[s][det]):.4f}"])
    print(f"saved -> {fp}")
    print(f"\nRegeneration (seed={SEED}):\n  python -m qpls.radius_scan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
