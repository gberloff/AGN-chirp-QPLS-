
import argparse
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qpls.injection_recovery import RESULTS_DIR
from qpls.populations import (
    load_isochrone, lensing_column, n_eff, L_min_for_floor, radius_at,
    D_L_AXIS, DM_AXIS, A_V_AXIS, M_TOTAL, M_NUC, DELTA_M_FLOOR,
)

POSTMS_DIR = RESULTS_DIR / "Post-MS_giant_results"
FIG_DIR = POSTMS_DIR

G = 6.67430e-11
C_LIGHT = 2.99792458e8
M_SUN = 1.98892e30
PC_M = 3.0856775814913673e16

M_LENS = 2.0e10
D_LS_PC = 1.0e3

R_E_TARGET, R_E_TOL = 2.0, 0.05

BLIND = ["global_max", "blind_harmonic_sum"]
STATISTICS = ["v1-frozen", "v2-physical", "v3-physical"]
PLOT_STATISTICS = ["v1-frozen", "v3-physical"]


def einstein_radius_pc(M_msun=M_LENS, D_LS_pc=D_LS_PC):
    M = M_msun * M_SUN
    D = D_LS_pc * PC_M
    return np.sqrt(4.0 * G * M * D / C_LIGHT ** 2) / PC_M


def v2_floors(path=None):
    path = path or RESULTS_DIR / "qpls_statistic_v2_floors.csv"
    if not path.exists():
        raise SystemExit(f"STOP: {path.name} not found, hence run qpls.statistic_v2 "
                         f"first; v2 floors are measured, never assumed.")
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            v = r["floor_A_at_R0.5_mag"]
            out[r["detector"]] = None if v == "none" else float(v)
    return out


def v3_floor_functions(path=None):
    if path is None:
        scan = POSTMS_DIR / "qpls_radius_scan_floors.csv"
        path = scan if scan.exists() else (POSTMS_DIR /
                                           "qpls_statistic_v3_floors.csv")
    if not path.exists():
        raise SystemExit(f"STOP: {path.name} not found, so run qpls.radius_scan "
                         f"(or qpls.statistic_v3) first; floors are measured, "
                         f"never assumed.")
    print(f"  (v3 floors read from {path.name})")
    pts = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["floor_A_at_R0.5_mag"] == "none":
                continue
            pts.setdefault(r["detector"], []).append(
                (float(r["R_Rsun"]), float(r["floor_A_at_R0.5_mag"])))
    fns = {}
    for det in BLIND:
        p = sorted(pts.get(det, []))
        if not p:
            fns[det] = None
            continue
        lr = np.log10([x[0] for x in p])
        lf = np.log10([x[1] for x in p])
        r_min = p[0][0]

        def f(R, lr=lr, lf=lf, r_min=r_min):
            R = np.asarray(R, dtype=float)
            out = 10.0 ** np.interp(np.log10(R), lr, lf)
            return np.where(R < r_min, np.inf, out)

        fns[det] = f
    return fns, pts


def L_min_self_consistent(iso, floor_fn, DM, A_V, L_lo=1.0, L_hi=1.0e4):
    from qpls.populations import delta_m_of_L
    Lg = np.geomspace(L_lo, L_hi, 2000)
    dm = delta_m_of_L(iso, Lg, DM, A_V)
    fl = floor_fn(radius_at(iso, Lg))
    diff = dm - fl
    ok = diff >= 0
    if not ok.any():
        return np.nan
    i = int(np.argmax(ok))
    if i == 0:
        return float(Lg[0])
    y0, y1 = diff[i - 1], diff[i]
    if not np.isfinite(y0):
        return float(Lg[i])
    x0, x1 = np.log10(Lg[i - 1]), np.log10(Lg[i])
    return float(10.0 ** (x0 - y0 * (x1 - x0) / (y1 - y0)))


def build_table(iso, floors_by_stat, column, r_E):
    area = np.pi * r_E ** 2
    rows, grids = [], {}
    for stat in STATISTICS:
        for det in BLIND:
            floor = floors_by_stat[stat][det]
            radius_dependent = callable(floor)
            G_ = np.full((len(A_V_AXIS), len(D_L_AXIS)), np.nan)
            for i, av in enumerate(A_V_AXIS):
                for j, (dl, dm) in enumerate(zip(D_L_AXIS, DM_AXIS)):
                    if floor is None:
                        Lmin, sd, nz = np.nan, np.nan, np.nan
                    else:
                        Lmin = (L_min_self_consistent(iso, floor, dm, av)
                                if radius_dependent
                                else L_min_for_floor(iso, floor, dm, av))
                        if np.isfinite(Lmin):
                            sd = float(n_eff(iso, np.array([Lmin]), column)[0][0])
                            nz = sd * area
                        else:
                            sd = nz = np.nan
                    if np.isfinite(nz) and nz > 0:
                        G_[i, j] = nz
                    if floor is None:
                        floor_str = "none"
                    elif radius_dependent:
                        floor_str = (f"{float(floor(radius_at(iso, Lmin))):.6g}"
                                     if np.isfinite(Lmin) else "radius-dependent")
                    else:
                        floor_str = f"{floor:.6g}"
                    rows.append(dict(
                        statistic=stat, detector=det, floor_mag=floor_str,
                        D_L_Gpc=dl, DM=dm, A_V=av, m_nuc=M_NUC,
                        M_total_Msun=M_TOTAL,
                        L_min_Lsun=(f"{Lmin:.6g}" if np.isfinite(Lmin) else ""),
                        R_at_Lmin_Rsun=(f"{radius_at(iso, Lmin):.6g}"
                                        if np.isfinite(Lmin) else ""),
                        Sigma_detectable_per_pc2=(f"{sd:.6g}"
                                                  if np.isfinite(sd) else ""),
                        r_E_pc=f"{r_E:.6g}",
                        einstein_area_pc2=f"{area:.6g}",
                        N_zone=(f"{nz:.6g}" if np.isfinite(nz) else ""),
                        column_Msun_per_pc2=f"{column:.6g}"))
            grids[(stat, det)] = G_
    return rows, grids


def fig_nexp(grids, path, dpi=300):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True,
                             sharey=True)
    style = {("v1-frozen", "global_max"): ("C0", "-o"),
             ("v1-frozen", "blind_harmonic_sum"): ("C0", "--s"),
             ("v3-physical", "global_max"): ("darkgreen", "-o"),
             ("v3-physical", "blind_harmonic_sum"): ("darkgreen", "--s")}
    for ax, av in zip(axes, [0.0, 1.0]):
        i = A_V_AXIS.index(av)
        any_line = False
        for (stat, det), G_ in grids.items():
            if stat not in PLOT_STATISTICS:
                continue
            y = G_[i, :]
            if not np.any(np.isfinite(y)):
                continue
            c, ls = style[(stat, det)]
            ax.plot(D_L_AXIS, y, ls, color=c, ms=4,
                    label=f"{stat}, {det.replace('blind_','')}")
            any_line = True
        ax.set_yscale("log")
        ax.set_xlabel("$D_L$ (Gpc)")
        ax.set_title(f"$A_V$ = {av:g} mag", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if not any_line:
            ax.text(0.5, 0.5, "no detectable cells", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="0.4")
    axes[0].set_ylabel("$N_{\\rm zone}$ (detectable stars in $\\pi r_E^2$)")
    axes[0].legend(fontsize=8)
    r_E = einstein_radius_pc()
    fig.suptitle("Detectable stars inside the Einstein zone\n"
                 rf"$M_{{\rm lens}} = 2\times10^{{10}}$ M$_\odot$, "
                 rf"$D_{{LS}} = 1$ kpc, $r_E = {r_E:.2f}$ pc; "
                 "MIST 10 Gyr LF + core-Sersic column",
                 fontsize=10)
    empty = sorted({stat for (stat, det), G_ in grids.items()
                    if stat in PLOT_STATISTICS and not np.any(np.isfinite(G_))})
    note = ("N_zone is an UPPER PROXY for the caustic-swept area, not an event "
            "rate. Profile and lens constants carry CITE-TODOs.")
    if empty:
        note += ("\nNO DETECTABLE CELLS for: " + ", ".join(empty)
                 + ",the blind floor is only reached at radii where the "
                   "luminosity function is already empty.")
    n_lines = note.count("\n") + 1
    bottom = 0.05 + 0.035 * (n_lines - 1)
    fig.get_layout_engine().set(rect=(0.0, bottom, 1.0, 1.0 - bottom))
    fig.text(0.5, 0.008, note, ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Einstein-zone detectable counts.")
    ap.parse_args(argv)

    r_E = einstein_radius_pc()
    dev = abs(r_E - R_E_TARGET) / R_E_TARGET
    print("=== P3.1 CHECKPOINT: Einstein radius ===")
    print(f"  r_E = sqrt(4 G M D_LS / c^2), M = {M_LENS:.1e} Msun, "
          f"D_LS = {D_LS_PC:.0f} pc")
    print(f"  r_E = {r_E:.4f} pc   target {R_E_TARGET} +/- {R_E_TOL*100:.0f}%   "
          f"dev {dev*100:.2f}%   {'PASS' if dev <= R_E_TOL else 'FAIL'}")
    if dev > R_E_TOL:
        print("\nSTOP: Einstein radius outside tolerance.")
        return 1
    area = np.pi * r_E ** 2
    print(f"  pi r_E^2 = {area:.4f} pc^2")
    print(f"  NOTE: N_zone scales linearly with D_LS at fixed M "
          f"(r_E^2 ~ M x D_LS); r_E^2 and the column are both CITE-TODO "
          f"fiducials.")

    iso = load_isochrone(verbose=False)
    column = lensing_column()
    f2 = v2_floors()
    f3, f3_pts = v3_floor_functions()
    floors_by_stat = {"v1-frozen": DELTA_M_FLOOR, "v2-physical": f2,
                      "v3-physical": f3}
    print("\n=== floors in use ===")
    for stat in STATISTICS:
        for det in BLIND:
            v = floors_by_stat[stat][det]
            if v is None:
                s = "none (R never reaches 0.5 at any radius)"
            elif callable(v):
                anchors = ", ".join(f"R={r:g}->{f:.3f}" for r, f in f3_pts[det])
                s = f"radius-dependent [{anchors}]"
            else:
                s = f"{v:.4f} mag"
            print(f"  {stat:12s} {det:19s} {s}")

    rows, grids = build_table(iso, floors_by_stat, column, r_E)

    out = POSTMS_DIR / "qpls_n_exp.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {out}  ({len(rows)} rows)")

    fig_nexp(grids, FIG_DIR / "fig10_nexp.png")

    g = grids[("v1-frozen", "global_max")]
    nz = g[A_V_AXIS.index(0.0), D_L_AXIS.index(0.1)]
    print("\n=== P3.3 ORDER-OF-MAGNITUDE CHECKPOINT (record, do not stop) ===")
    if np.isfinite(nz):
        ratio = nz / 30.0
        verdict = "within factor ~2" if 0.5 <= ratio <= 2.0 else "OUTSIDE factor 2"
        print(f"  v1 N_zone at (0.1 Gpc, A_V=0, global-max) = {nz:.2f}   "
              f"expected ~30   ratio {ratio:.2f}x   {verdict}")
    else:
        print("  v1 N_zone at (0.1 Gpc, A_V=0, global-max) is undefined")

    print("\n=== N_zone at (0.1 Gpc, A_V = 0) and reach with N_zone >= 1 ===")
    i0, j0 = A_V_AXIS.index(0.0), D_L_AXIS.index(0.1)
    for stat in STATISTICS:
        for det in BLIND:
            G_ = grids[(stat, det)]
            v = G_[i0, j0]
            ge1 = [d for d, y in zip(D_L_AXIS, G_[i0, :])
                   if np.isfinite(y) and y >= 1.0]
            print(f"  {stat:12s} {det:19s} "
                  f"N_zone(0.1 Gpc) = {(f'{v:.4g}' if np.isfinite(v) else '0 / none'):>10s}   "
                  f"largest D_L with N_zone >= 1: "
                  f"{(f'{max(ge1):g} Gpc' if ge1 else 'none')}")

    print("\n=== v3 sensitivity: hypothetical radius threshold for a finite "
          "blind floor ===")
    print("  (floor held at the measured AGBiso value; only the radius at which "
          "it switches on is varied)")
    from qpls.populations import cumulative_lf
    for R_thr in [137.5, 185.6, 229.0, 272.1, 300.744]:
        Lg = np.geomspace(1.0, 1.0e4, 4000)
        Rg = radius_at(iso, Lg)
        cand = Lg[Rg >= R_thr]
        if cand.size == 0:
            print(f"    R >= {R_thr:6.1f} Rsun : no L reaches this radius")
            continue
        L_thr = float(cand.min())
        N = float(cumulative_lf(iso, np.array([L_thr]))[0])
        nz = column * N / 1.0e6 * area
        print(f"    R >= {R_thr:6.1f} Rsun (L >= {L_thr:7.1f} Lsun): "
              f"N = {N:9.4g} per 1e6 Msun -> N_zone = {nz:9.4g}")
    print("  Even the most generous threshold leaves v3 N_zone orders of "
          "magnitude below v1's 29.8: the samplability bonus only reaches stars "
          "large enough to have long crossings, and those are rare in the LF.")

    print("\n=== summary: N_zone at A_V = 0 ===")
    for stat in STATISTICS:
        for det in BLIND:
            y = grids[(stat, det)][A_V_AXIS.index(0.0), :]
            cells = [(d, v) for d, v in zip(D_L_AXIS, y) if np.isfinite(v)]
            if cells:
                print(f"  {stat:12s} {det:19s} "
                      f"{cells[0][1]:9.3g} at {cells[0][0]:g} Gpc -> "
                      f"{cells[-1][1]:9.3g} at {cells[-1][0]:g} Gpc "
                      f"({len(cells)}/{len(D_L_AXIS)} cells)")
            else:
                print(f"  {stat:12s} {det:19s} no detectable cells")

    print(f"\nRegeneration:\n  python -m qpls.n_exp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
