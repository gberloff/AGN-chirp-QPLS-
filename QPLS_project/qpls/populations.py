
import argparse
import csv
import glob
import pathlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qpls.injection_recovery import RESULTS_DIR
from qpls.source_scenarios import M_BOL_SUN, mu_cusp_peak

FIG_DIR = RESULTS_DIR.parent / "figures"
ISO_DIR = pathlib.Path("data/isochrones")

REQUIRE = dict(version="1.2", feh=-0.25, vvcrit=0.0, log_age=10.0)

COL = dict(EEP=0, log_age=1, initial_mass=2, star_mass=3, log_L=7,
           log_Teff=10, log_R=11, phase=24)

PHASE = {-1: "PMS", 0: "MS", 2: "RGB", 3: "CHeB", 4: "EAGB", 5: "TPAGB",
         6: "postAGB", 9: "WR"}
GIANT_PHASES = (2, 3, 4, 5)


def find_isochrone():
    hits = sorted(glob.glob(str(ISO_DIR / "*.iso")))
    if not hits:
        raise SystemExit(
            f"STOP: no *.iso in {ISO_DIR}. See {ISO_DIR/'README.md'}, the "
            f"packaged MIST v1.2 grid must be fetched first.")
    return pathlib.Path(hits[0])


def load_isochrone(path=None, verbose=True, require=None):
    REQUIRE_ = REQUIRE if require is None else require
    path = path or find_isochrone()
    text = path.read_text()
    header = [l for l in text.splitlines() if l.startswith("#")][:8]

    version = feh = vvcrit = None
    for i, l in enumerate(header):
        if "MIST version number" in l:
            version = l.split("=")[1].strip()
        if "Yinit" in l and "[Fe/H]" in l:
            vals = header[i + 1].lstrip("#").split()
            feh, vvcrit = float(vals[2]), float(vals[4])

    if verbose:
        print(f"isochrone: {path.name}")
        print(f"  MIST version = {version}   [Fe/H] = {feh}   v/vcrit = {vvcrit}")

    problems = []
    if version != REQUIRE_["version"]:
        problems.append(f"version {version!r} != {REQUIRE_['version']!r}")
    if feh is None or abs(feh - REQUIRE_["feh"]) > 1e-9:
        problems.append(f"[Fe/H] {feh} != {REQUIRE_['feh']}")
    if vvcrit is None or abs(vvcrit - REQUIRE_["vvcrit"]) > 1e-9:
        problems.append(f"v/vcrit {vvcrit} != {REQUIRE_['vvcrit']}")
    if problems:
        raise SystemExit("STOP: isochrone header does not match the gate: "
                         + "; ".join(problems))

    rows = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 25:
            continue
        if abs(float(p[COL["log_age"]]) - REQUIRE_["log_age"]) < 1e-9:
            rows.append(p)
    if not rows:
        raise SystemExit(f"STOP: no rows at log10(age) = {REQUIRE_['log_age']}")
    a = np.array(rows, dtype=float)
    if verbose:
        print(f"  age 10 Gyr: {len(a)} EEP rows, "
              f"initial mass {a[:, COL['initial_mass']].min():.3f}-"
              f"{a[:, COL['initial_mass']].max():.3f} Msun")
    return a


KROUPA_BREAKS = [0.08, 0.5, 150.0]
KROUPA_ALPHA = [1.3, 2.3]
M_MIN, M_MAX = 0.08, 150.0


def _kroupa_unnorm(m):
    m = np.atleast_1d(np.asarray(m, dtype=float))
    out = np.zeros_like(m)
    lo = (m >= KROUPA_BREAKS[0]) & (m < KROUPA_BREAKS[1])
    hi = (m >= KROUPA_BREAKS[1]) & (m <= KROUPA_BREAKS[2])
    out[lo] = m[lo] ** -KROUPA_ALPHA[0]
    k = KROUPA_BREAKS[1] ** (-KROUPA_ALPHA[0]) / KROUPA_BREAKS[1] ** (-KROUPA_ALPHA[1])
    out[hi] = k * m[hi] ** -KROUPA_ALPHA[1]
    return out


def kroupa_normalisation(total_mass=1.0e6):
    m = np.geomspace(M_MIN, M_MAX, 200001)
    integrand = m * _kroupa_unnorm(m)
    mass_per_unit = np.trapezoid(integrand, m)
    return total_mass / mass_per_unit


def cumulative_lf(iso, L_grid, total_mass=1.0e6):
    A = kroupa_normalisation(total_mass)
    mi = iso[:, COL["initial_mass"]]
    L = 10.0 ** iso[:, COL["log_L"]]

    order = np.argsort(mi, kind="stable")
    mi_s, L_s = mi[order], L[order]
    dm = np.diff(mi_s)
    m_mid = 0.5 * (mi_s[1:] + mi_s[:-1])
    L_mid = 0.5 * (L_s[1:] + L_s[:-1])
    weight = A * _kroupa_unnorm(m_mid) * dm

    return np.array([weight[L_mid >= Lmin].sum() for Lmin in L_grid])


def giant_branch_RL(iso):
    ph = iso[:, COL["phase"]].astype(int)
    m = np.isin(ph, GIANT_PHASES)
    logL, logR = iso[m, COL["log_L"]], iso[m, COL["log_R"]]
    o = np.argsort(logL)
    lu, idx = np.unique(np.round(logL[o], 6), return_index=True)
    return lu, logR[o][idx]


def radius_at(iso, L):
    lu, lr = giant_branch_RL(iso)
    return 10.0 ** np.interp(np.log10(L), lu, lr)


CORE_SERSIC = dict(
    Sigma_b=3.0e3,
    R_b=100.0,
    gamma=0.10,
    alpha=5.0,
    n=4.0,
    R_e=1.0e4,
)


def sigma_star(R_pc, p=CORE_SERSIC):
    R = np.atleast_1d(np.asarray(R_pc, dtype=float))
    b_n = 2.0 * p["n"] - 1.0 / 3.0 + 0.009876 / p["n"]
    inner = (1.0 + (p["R_b"] / R) ** p["alpha"]) ** (p["gamma"] / p["alpha"])
    outer = np.exp(-b_n * ((R ** p["alpha"] + p["R_b"] ** p["alpha"])
                           / p["R_e"] ** p["alpha"]) ** (1.0 / (p["alpha"] * p["n"])))
    norm = ((1.0 + 1.0) ** (p["gamma"] / p["alpha"])
            * np.exp(-b_n * ((2.0 * p["R_b"] ** p["alpha"])
                             / p["R_e"] ** p["alpha"]) ** (1.0 / (p["alpha"] * p["n"]))))
    return p["Sigma_b"] * inner * outer / norm


D_LS_MIN_PC, D_LS_MAX_PC = 500.0, 2000.0


def lensing_column(d_min=D_LS_MIN_PC, d_max=D_LS_MAX_PC, p=CORE_SERSIC):
    r = np.geomspace(d_min, d_max, 4001)
    rho = sigma_star(r, p) / (np.pi * r)
    return float(np.trapezoid(rho, r))


def n_eff(iso, L_min_grid, column=None, total_mass=1.0e6):
    column = lensing_column() if column is None else column
    N = cumulative_lf(iso, L_min_grid, total_mass)
    return column * N / total_mass, N, column


DELTA_M_FLOOR = {"global_max": 0.07, "blind_harmonic_sum": 0.125}

D_L_AXIS = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0, 2.0, 2.9]
DM_AXIS = [35.00, 36.51, 37.72, 38.49, 39.23, 39.65, 40.00, 41.51, 42.31]
A_V_AXIS = [0.0, 0.5, 1.0, 2.0]
M_TOTAL = 2.0e10
M_NUC = 19.0


def delta_m_of_L(iso, L, DM, A_V, m_nuc=M_NUC, M_total=M_TOTAL):
    R = radius_at(iso, L)
    mu = mu_cusp_peak(M_total, R)
    m_unlensed = M_BOL_SUN - 2.5 * np.log10(L) + DM
    m_lensed = m_unlensed - 2.5 * np.log10(mu) + A_V
    f = 10.0 ** (-0.4 * (m_lensed - m_nuc))
    return 2.5 * np.log10(1.0 + f)


def L_min_for_floor(iso, floor, DM, A_V, L_lo=1.0, L_hi=1.0e4):
    Lg = np.geomspace(L_lo, L_hi, 2000)
    dm = delta_m_of_L(iso, Lg, DM, A_V)
    ok = dm >= floor
    if not ok.any():
        return np.nan
    i = np.argmax(ok)
    if i == 0:
        return Lg[0]
    x0, x1 = np.log10(Lg[i - 1]), np.log10(Lg[i])
    y0, y1 = dm[i - 1], dm[i]
    return 10.0 ** (x0 + (floor - y0) * (x1 - x0) / (y1 - y0))


def fig_lf(L_grid, N, path, dpi=300):
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.loglog(L_grid, np.maximum(N, 1e-12), lw=2)
    ax.set_xlabel("$L$ (L$_\\odot$)")
    ax.set_ylabel("$N(>L)$ per $10^6$ M$_\\odot$ initial mass")
    ax.set_title("Old-population luminosity function\n"
                 "MIST v1.2, 10 Gyr, [Fe/H] = $-0.25$, Kroupa IMF", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(1e-2, max(1e2, N.max() * 2))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def fig_neff(L_grid, ne, column, path, dpi=300):
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.loglog(L_grid, np.maximum(ne, 1e-30), lw=2, color="darkgreen")
    ax.set_xlabel("$L_{\\min}$ (L$_\\odot$)")
    ax.set_ylabel("$n_{\\rm eff}$ (detectable sources pc$^{-2}$)")
    ax.set_title("Detectable source surface density behind the binary\n"
                 f"column = {column:.3g} M$_\\odot$ pc$^{{-2}}$, "
                 f"$D_{{LS}}$ = 0.5-2 kpc (isothermal deprojection)",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def fig_visibility(grids, path, dpi=300):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for ax, (det, G) in zip(axes, grids.items()):
        finite = G[np.isfinite(G)]
        vmin = np.floor(finite.min()) if finite.size else -6
        vmax = np.ceil(finite.max()) if finite.size else 0
        im = ax.imshow(G, origin="lower", aspect="auto", cmap="magma",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(D_L_AXIS)), [f"{d:g}" for d in D_L_AXIS])
        ax.set_yticks(range(len(A_V_AXIS)), [f"{a:g}" for a in A_V_AXIS])
        ax.set_xlabel("$D_L$ (Gpc)")
        ax.set_ylabel("$A_V$ (mag)")
        ax.set_title(f"{det} (floor {DELTA_M_FLOOR[det]:.3g} mag)", fontsize=10)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                if not np.isfinite(G[i, j]):
                    ax.text(j, i, "--", ha="center", va="center", fontsize=8,
                            color="0.35")
                else:
                    ax.text(j, i, f"{G[i, j]:.1f}", ha="center", va="center",
                            fontsize=8,
                            color="w" if G[i, j] < (vmin + vmax) / 2 else "k")
        fig.colorbar(im, ax=ax, label="$\\log_{10}\\,\\Sigma_{\\rm detectable}$ (pc$^{-2}$)")
    fig.suptitle("Step-2 visibility map, old-population slice\n"
                 "MIST v1.2 10 Gyr [Fe/H]=$-0.25$ LF + core-Sersic column; "
                 f"$M_{{tot}}=2\\times10^{{10}}$ M$_\\odot$, $m_{{nuc}}=19$",
                 fontsize=10)
    fig.text(0.5, 0.005, "'--' = floor unreachable at any L <= 1e4 Lsun. "
             "Profile constants are fiducials (CITE-TODO); bolometric, grey.",
             ha="center", va="bottom", fontsize=8)
    fig.get_layout_engine().set(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Old-population LF, R(L), n_eff.")
    ap.add_argument(",check-only", action="store_true")
    args = ap.parse_args(argv)

    iso = load_isochrone()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    L_grid = np.geomspace(1.0, 1.0e4, 200)
    N = cumulative_lf(iso, L_grid)

    R_2e3 = radius_at(iso, 2.0e3)
    ok_c = 70.0 <= R_2e3 <= 150.0
    print("\n=== A1(c) CHECKPOINT: R at L = 2e3 Lsun ===")
    print(f"  R = {R_2e3:.1f} Rsun   window 70-150   "
          f"{'PASS' if ok_c else 'FAIL'}")
    print(f"  (scenario grid assumed 100 Rsun for its RGB source; "
          f"mu ~ R^-0.64 so the isochrone value gives "
          f"{(R_2e3/100.0)**-0.64:.3f}x the magnification)")

    N_2e3 = float(cumulative_lf(iso, np.array([2.0e3]))[0])
    ok_d = 1.0 <= N_2e3 <= 100.0
    print("\n=== A1(d) CHECKPOINT: N(L >= 2e3 Lsun) per 1e6 Msun ===")
    print(f"  N = {N_2e3:.2f}   expected ~5-50, STOP window 1-100   "
          f"{'PASS' if ok_d else 'FAIL'}")

    if not (ok_c and ok_d):
        print("\nSTOP: checkpoint outside its window. Constants NOT adjusted.")
        return 1

    print("\n=== A1(e) empirical anchor ===")
    print(f"  Predicted bright giants (L >= 2e3 Lsun) in a 1e6 Msun old system: "
          f"{N_2e3:.1f}")
    print("  \\verify TODO: compare with literature bright-giant counts for a")
    print("  47 Tuc-like cluster (~1e6 Msun, old, metal-rich). NO literature")
    print("  value is asserted here. Note the metallicity mismatch: this")
    print("  isochrone is [Fe/H] = -0.25, whereas 47 Tuc is nearer -0.7, so the")
    print("  comparison is indicative only until a matched-metallicity track is")
    print("  run.")

    if args.check_only:
        return 0

    ne, N_of_L, column = n_eff(iso, L_grid)
    print("\n=== A2 surface density ===")
    print(f"  Sigma_star(0 pc)      = {sigma_star(1e-3).item():.4g} Msun/pc^2 "
          f"(target order 1e3-1e4)")
    print(f"  Sigma_star(R_b=100pc) = {sigma_star(100.0).item():.4g} Msun/pc^2")
    print(f"  lensing column        = {column:.4g} Msun/pc^2 "
          f"(D_LS 0.5-2 kpc, isothermal deprojection, ASSUMED)")

    with open(RESULTS_DIR / "qpls_population_lf.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["L_Lsun", "N_gt_L_per_1e6Msun", "R_of_L_Rsun",
                    "n_eff_per_pc2", "column_Msun_per_pc2"])
        for L, n, e in zip(L_grid, N_of_L, ne):
            w.writerow([f"{L:.6g}", f"{n:.6g}", f"{radius_at(iso, L):.6g}",
                        f"{e:.6g}", f"{column:.6g}"])
    print(f"saved -> {RESULTS_DIR/'qpls_population_lf.csv'}")

    fig_lf(L_grid, N_of_L, FIG_DIR / "fig7_LF.png")
    fig_neff(L_grid, ne, column, FIG_DIR / "fig7_neff.png")

    lu, lr = giant_branch_RL(iso)
    grids, rows = {}, []
    for det, floor in DELTA_M_FLOOR.items():
        G = np.full((len(A_V_AXIS), len(D_L_AXIS)), np.nan)
        for i, av in enumerate(A_V_AXIS):
            for j, (dl, dm) in enumerate(zip(D_L_AXIS, DM_AXIS)):
                Lmin = L_min_for_floor(iso, floor, dm, av)
                if np.isfinite(Lmin):
                    sd = float(n_eff(iso, np.array([Lmin]), column)[0][0])
                    G[i, j] = np.log10(sd) if sd > 0 else np.nan
                else:
                    sd = np.nan
                rows.append(dict(detector=det, floor_mag=floor, D_L_Gpc=dl, DM=dm,
                                 A_V=av, m_nuc=M_NUC, M_total_Msun=M_TOTAL,
                                 L_min_Lsun=f"{Lmin:.6g}",
                                 R_at_Lmin_Rsun=(f"{radius_at(iso, Lmin):.6g}"
                                                 if np.isfinite(Lmin) else ""),
                                 Sigma_detectable_per_pc2=(f"{sd:.6g}"
                                                           if np.isfinite(Lmin) else ""),
                                 column_Msun_per_pc2=f"{column:.6g}"))
        grids[det] = G

    Lmin_check = L_min_for_floor(iso, DELTA_M_FLOOR["global_max"], 38.49, 0.0)
    ok_e = np.isfinite(Lmin_check) and Lmin_check <= 2.0e3
    print("\n=== A2(d) CONSISTENCY CHECKPOINT ===")
    print(f"  (0.5 Gpc, A_V=0, M=2e10), global-max floor "
          f"{DELTA_M_FLOOR['global_max']} mag:")
    print(f"  L_min = {Lmin_check:.1f} Lsun   must be <= 2e3   "
          f"{'PASS' if ok_e else 'FAIL'}")
    if not ok_e:
        print("\nSTOP: the committed grid recovered that cell with a 2e3 Lsun "
              "source, so L_min must not exceed it. Constants NOT adjusted.")
        return 1

    with open(RESULTS_DIR / "qpls_visibility_map.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {RESULTS_DIR/'qpls_visibility_map.csv'}  ({len(rows)} rows)")

    fig_visibility(grids, FIG_DIR / "fig7_visibility.png")

    print("\nRegeneration:\n  python -m qpls.populations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
