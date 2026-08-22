
import argparse
import csv
import pathlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial

import qpls.injection_recovery as _ir
from qpls.injection_recovery import (
    RESULTS_DIR, ztf_cadence, gls_peak_power, gls_harmonic_sum,
    null_threshold, recovery_grid,
)
from qpls.detector_comparison import CFG, oracle_recovery
from qpls.source_amplitude import M_BOL_SUN, MU_CUSP_PEAK_NORM

FIG_DIR = RESULTS_DIR.parent / "figures"

SEED = 42
HOST = "quiescent"
P_INJECT = 18.0071
R_MAX_NORM = 3500.0

SOURCE_LIBRARY = {
    "RGB": dict(L=2.0e3, R=100.0, label="tip-RGB star"),
    "RGBiso": dict(L=2199.0, R=137.5,
                   label="tip-RGB, MIST v1.2 isochrone-anchored"),
    "AGBiso": dict(L=6027.09, R=300.744,
                   label="AGB tip, MIST v1.2 isochrone-anchored"),
    "AGB170": dict(L=2688.763996139983, R=170.1963493780872,
                   label="AGB (isochrone, ~170 Rsun)"),
    "AGB215": dict(L=3683.890372263349, R=214.97406799964722,
                   label="AGB (isochrone, ~215 Rsun)"),
    "AGB260": dict(L=4709.267690147197, R=260.51159521815595,
                   label="AGB (isochrone, ~260 Rsun)"),
    "VY": dict(L=2.7e5, R=1420.0, label="VY CMa-like red hypergiant"),
    "BSG": dict(L=1.0e5, R=20.0,
                label="control: population-implausible in quiescent elliptical host"),
}
DEFAULT_SOURCE_KEYS = ["RGB", "VY"]
SOURCES = {k: SOURCE_LIBRARY[k] for k in DEFAULT_SOURCE_KEYS}
M_TOTAL_GRID = [1.0e8, 1.0e9, 1.0e10, 2.0e10]
A_V_GRID = [0.0, 0.5, 1.0, 2.0]
M_NUC_GRID = [19.0]
D_L_GRID = [(0.5, 38.49), (1.0, 40.00), (2.0, 41.51), (2.9, 42.31)]

APERTURE_D_L_MAX = 0.2
APERTURE_NOTE = ("aperture-favourable: fixed m_nuc=19 understates nuclear light "
                 "at this distance")


def mu_cusp_peak(M_total, R_Rsun):
    return MU_CUSP_PEAK_NORM * (M_total / 2.0e10) ** 0.44 * (R_Rsun / 10.0) ** -0.64


def r_max_rsun(M_total):
    return R_MAX_NORM * (M_total / 2.0e10) ** (1.0 / 6.0)


def scenario_cell(source, M_total, A_V, m_nuc, D_L_Gpc, DM):
    L, R = SOURCE_LIBRARY[source]["L"], SOURCE_LIBRARY[source]["R"]
    Rmax = r_max_rsun(M_total)

    m_unlensed = M_BOL_SUN - 2.5 * np.log10(L) + DM
    mu = mu_cusp_peak(M_total, R)
    m_lensed = m_unlensed - 2.5 * np.log10(mu) + A_V
    f = 10.0 ** (-0.4 * (m_lensed - m_nuc))
    delta_m_peak = 2.5 * np.log10(1.0 + f)

    if R > Rmax:
        validity = "INVALID"
    elif R > 0.95 * Rmax:
        validity = "VALID-marginal"
    else:
        validity = "VALID"

    return dict(
        source=source, L_Lsun=L, R_Rsun=R, M_total_Msun=M_total, A_V=A_V,
        m_nuc=m_nuc, D_L_Gpc=D_L_Gpc, DM=DM, R_max_Rsun=Rmax, validity=validity,
        m_unlensed=m_unlensed, mu_cusp=mu, m_lensed=m_lensed, flux_ratio=f,
        delta_m_peak=delta_m_peak,
        aperture_note=APERTURE_NOTE if D_L_Gpc <= APERTURE_D_L_MAX else "",
        source_label=SOURCE_LIBRARY[source]["label"],
    )


def build_grid():
    cells = []
    for source in SOURCES:
        for M_total in M_TOTAL_GRID:
            for A_V in A_V_GRID:
                for m_nuc in M_NUC_GRID:
                    for D_L_Gpc, DM in D_L_GRID:
                        cells.append(scenario_cell(source, M_total, A_V, m_nuc,
                                                   D_L_Gpc, DM))
    return cells


CHECKPOINTS = [
    ("RGB", 0.5, 0.17), ("RGB", 1.0, 0.044), ("RGB", 2.0, 0.011), ("RGB", 2.9, 0.0054),
    ("VY", 0.5, 1.78), ("VY", 1.0, 0.77), ("VY", 2.0, 0.25), ("VY", 2.9, 0.126),
]
TOL = 0.20


def print_table(cells):
    hdr = (f"{'source':6s} {'M_total':>9s} {'A_V':>4s} {'m_nuc':>5s} {'D_L':>5s} "
           f"{'DM':>6s} {'R_max':>7s} {'m_unlens':>9s} {'mu_cusp':>10s} "
           f"{'m_lensed':>9s} {'f':>10s} {'dm_peak':>9s}  validity")
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        print(f"{c['source']:6s} {c['M_total_Msun']:9.2e} {c['A_V']:4.1f} "
              f"{c['m_nuc']:5.1f} {c['D_L_Gpc']:5.2f} {c['DM']:6.2f} "
              f"{c['R_max_Rsun']:7.1f} {c['m_unlensed']:9.3f} {c['mu_cusp']:10.3e} "
              f"{c['m_lensed']:9.3f} {c['flux_ratio']:10.3e} "
              f"{c['delta_m_peak']:9.5f}  {c['validity']}")


def check_checkpoints(cells, checkpoints=None):
    checkpoints = CHECKPOINTS if checkpoints is None else checkpoints
    print()
    print("Pre-registered checkpoints (A_V=0, m_nuc=19, M=2e10), tolerance +-20%:")
    ok = True
    for source, D_L, expected in checkpoints:
        match = [c for c in cells if c["source"] == source and c["A_V"] == 0.0
                 and c["m_nuc"] == 19.0 and c["M_total_Msun"] == 2.0e10
                 and c["D_L_Gpc"] == D_L]
        got = match[0]["delta_m_peak"]
        dev = abs(got - expected) / expected
        flag = "PASS" if dev <= TOL else "FAIL"
        ok &= dev <= TOL
        print(f"  {source:4s} D_L={D_L:4.2f} Gpc : expected {expected:8.4f}  "
              f"got {got:8.5f}  dev {dev * 100:5.1f}%  {flag}")
    print(f"  -> {'ALL CHECKPOINTS PASS' if ok else 'CHECKPOINT FAILURE'}")
    return ok


def run_recovery(cells, cfg=CFG):
    sf = cfg["sf_inf_quiescent"]
    tau = cfg["tau_drw_days"]

    _ir.RNG.bit_generator.state = np.random.default_rng(0).bit_generator.state
    t, sigma = ztf_cadence(cfg["baseline_yr"] * 365.25, cfg["cadence_days"],
                           cfg["season_frac"], err_med=cfg["err_med"])

    p_min = 2 * cfg["cadence_days"]
    p_max = 0.5 * cfg["baseline_yr"] * 365.25
    freq = np.linspace(1.0 / p_max, 1.0 / p_min, 1000)

    valid = [c for c in cells if c["validity"] != "INVALID"]
    A_grid = np.array([c["delta_m_peak"] for c in valid])
    P_grid = np.array([P_INJECT])
    print(f"\n[recovery] host={HOST}  cells={len(valid)} valid / {len(cells)} total")
    print(f"[recovery] P_inject={P_INJECT} d  n_inj={cfg['n_inj']}  "
          f"n_null={cfg['n_null']}  fap={cfg['fap']}  seed={SEED}")

    results = {}

    rng = np.random.default_rng(SEED)
    thr_gm, _ = null_threshold(t, sigma, tau, sf, freq, gls_peak_power,
                               cfg["fap"], cfg["n_null"], rng)
    print(f"[global_max]   threshold = {thr_gm:.4f}  (committed: 0.4277)", flush=True)
    R_gm = recovery_grid(t, sigma, tau, sf, P_grid, A_grid, freq, thr_gm,
                         gls_peak_power, cfg["n_inj"], cfg["fwhm_days"], rng)
    results["global_max"] = (float(thr_gm), R_gm[:, 0])

    det_hs = partial(gls_harmonic_sum, n_harmonics=cfg["n_harmonics"])
    rng = np.random.default_rng(SEED)
    thr_hs, _ = null_threshold(t, sigma, tau, sf, freq, det_hs,
                               cfg["fap"], cfg["n_null"], rng)
    print(f"[harmonic_sum] threshold = {thr_hs:.4f}  (committed: 0.8047)", flush=True)
    R_hs = recovery_grid(t, sigma, tau, sf, P_grid, A_grid, freq, thr_hs,
                         det_hs, cfg["n_inj"], cfg["fwhm_days"], rng)
    results["blind_harmonic_sum"] = (float(thr_hs), R_hs[:, 0])

    thr_or, R_or, thr_by_period = oracle_recovery(
        t, sigma, tau, sf, P_grid, A_grid, freq,
        cfg["fap"], cfg["n_null"], cfg["n_inj"], cfg["fwhm_days"], seed=SEED)
    print(f"[oracle]       threshold = {list(thr_by_period.values())[0]:.4f}", flush=True)
    results["oracle_single_freq"] = (float(list(thr_by_period.values())[0]), R_or[:, 0])

    for det, thr, exp in [("global_max", thr_gm, 0.4277),
                          ("harmonic_sum", thr_hs, 0.8047)]:
        status = "PASS" if abs(thr - exp) < 5e-4 else "FAIL"
        print(f"[self-check] {det}: {thr:.4f} vs committed {exp} -> {status}")

    return valid, results


CSV_COLS = ["source", "L_Lsun", "R_Rsun", "M_total_Msun", "A_V", "m_nuc",
            "D_L_Gpc", "DM", "R_max_Rsun", "validity", "m_unlensed", "mu_cusp",
            "m_lensed", "flux_ratio", "delta_m_peak", "P_inject_days", "detector",
            "threshold", "recovery_fraction", "binom_se", "n_inj", "seed",
            "aperture_note", "source_label"]


def write_csv(cells, valid, results, path, n_inj):
    idx = {id(c): i for i, c in enumerate(valid)}
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for c in cells:
            for det, (thr, R) in results.items():
                row = {k: c.get(k, "") for k in CSV_COLS}
                row["P_inject_days"] = P_INJECT
                row["detector"] = det
                row["threshold"] = f"{thr:.6f}"
                if c["validity"] == "INVALID":
                    row["recovery_fraction"] = ""
                    row["binom_se"] = ""
                else:
                    r = float(R[idx[id(c)]])
                    row["recovery_fraction"] = f"{r:.4f}"
                    row["binom_se"] = f"{np.sqrt(r * (1.0 - r) / n_inj):.4f}"
                row["n_inj"] = n_inj
                row["seed"] = SEED
                w.writerow(row)
    print(f"saved -> {path}")


def draw_heatmap_panels(panel_specs, x_vals, y_vals, suptitle, path,
                        dpi=300, annotation=None, xlabel="$D_L$ (Gpc)",
                        ylabel="$A_V$ (mag)"):
    n = len(panel_specs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (label, grid) in zip(axes, panel_specs):
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(x_vals)), [f"{v:g}" for v in x_vals])
        ax.set_yticks(range(len(y_vals)), [f"{v:g}" for v in y_vals])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(label, fontsize=10)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            color="w" if grid[i, j] < 0.6 else "k", fontsize=8)
        fig.colorbar(im, ax=ax, label="recovery fraction")

    fig.suptitle(suptitle, fontsize=10)
    if annotation:
        n_lines = annotation.count("\n") + 1
        bottom = 0.06 + 0.03 * (n_lines - 1)
        fig.get_layout_engine().set(rect=(0.0, bottom, 1.0, 1.0 - bottom))
        fig.text(0.5, 0.015, annotation, ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def plot_source(source, cells, valid, results, path, dpi=300):
    idx = {id(c): i for i, c in enumerate(valid)}
    dls = [d for d, _ in D_L_GRID]
    panels = [("blind_harmonic_sum", "blind harmonic-sum"),
              ("oracle_single_freq", "oracle single-freq (optimistic control)")]

    specs = []
    for key, label in panels:
        _, R = results[key]
        grid = np.full((len(A_V_GRID), len(dls)), np.nan)
        for c in cells:
            if (c["source"] != source or c["M_total_Msun"] != 2.0e10
                    or c["m_nuc"] != 19.0 or c["validity"] == "INVALID"):
                continue
            grid[A_V_GRID.index(c["A_V"]), dls.index(c["D_L_Gpc"])] = R[idx[id(c)]]
        specs.append((label, grid))

    draw_heatmap_panels(
        specs, dls, A_V_GRID,
        f"{source} source, quiescent host, $M_{{tot}}=2\\times10^{{10}}$ "
        f"$M_\\odot$, $m_{{nuc}}=19$\n"
        f"$P_{{inj}}={P_INJECT}$ d, FAP={CFG['fap']}, "
        f"n_inj={CFG['n_inj']}, seed={SEED}",
        path, dpi=dpi)


def report(cells, valid, results, cmd):
    idx = {id(c): i for i, c in enumerate(valid)}
    blind = ["global_max", "blind_harmonic_sum"]

    print()
    print("=" * 78)
    print("STEP 5 REPORT")
    print("=" * 78)

    for source in SOURCES:
        print(f"\nTop-3 recovery VALID cells: {source}:")
        for det, (_, R) in results.items():
            rows = [(R[idx[id(c)]], c) for c in cells
                    if c["source"] == source and c["validity"] != "INVALID"]
            rows.sort(key=lambda x: -x[0])
            for r, c in rows[:3]:
                print(f"  [{det:19s}] R={r:.3f}  D_L={c['D_L_Gpc']:3.1f} "
                      f"A_V={c['A_V']:3.1f} M={c['M_total_Msun']:.1e} "
                      f"dm_peak={c['delta_m_peak']:.4f}  {c['validity']}")

    print("\nRGB blind-detector recovery > 0.05 in ANY cell?")
    for det in blind:
        _, R = results[det]
        hits = [(R[idx[id(c)]], c) for c in cells
                if c["source"] == "RGB" and c["validity"] != "INVALID"
                and R[idx[id(c)]] > 0.05]
        if hits:
            best = max(hits, key=lambda x: x[0])
            print(f"  {det:19s}: YES: {len(hits)} cell(s); best R={best[0]:.3f} "
                  f"at D_L={best[1]['D_L_Gpc']} A_V={best[1]['A_V']} "
                  f"M={best[1]['M_total_Msun']:.1e}")
        else:
            print(f"  {det:19s}: NO: no VALID cell exceeds 0.05")

    print("\nSmallest D_L at which RGB blind recovery exceeds 0.5:")
    for det in blind:
        _, R = results[det]
        for A_V in (0.0, 1.0):
            cand = [c["D_L_Gpc"] for c in cells
                    if c["source"] == "RGB" and c["A_V"] == A_V
                    and c["validity"] != "INVALID" and R[idx[id(c)]] > 0.5]
            val = f"{min(cand):g} Gpc" if cand else "none"
            print(f"  {det:19s}  A_V={A_V:3.1f}: {val}")

    print(f"\nRegeneration command (seed={SEED}):\n  {cmd}")


REPORT_SRC_NAME = {
    "RGB": "RGB",
    "RGBiso": "RGB (isochrone-anchored)",
    "VY": "RSG hypergiant",
    "BSG": "BSG (control)",
}


def mass_tag(m):
    exp = int(np.floor(np.log10(m)))
    mant = m / 10.0 ** exp
    return f"1e{exp}" if abs(mant - 1.0) < 1e-9 else f"{mant:g}e{exp}"


def lens_term(m):
    return "SMBHB" if m >= 1.0e6 else "IMBH binary"


def export_report_style(suffix, outdir=FIG_DIR, dpi=300):
    path = RESULTS_DIR / f"qpls_source_scenarios{suffix}.csv"
    if not path.exists():
        raise SystemExit(f"STOP: {path} not found; run the grid first.")

    cells = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["recovery_fraction"] == "":
                continue
            key = (r["source"], float(r["M_total_Msun"]), float(r["m_nuc"]))
            cells.setdefault(key, {})[(float(r["A_V"]), float(r["D_L_Gpc"]),
                                       r["detector"])] = float(r["recovery_fraction"])

    panels = [("global_max", "global-max GLS"),
              ("oracle_single_freq", "oracle single-freq (optimistic control)"),
              ("blind_harmonic_sum", "blind harmonic-sum")]
    entries = []
    for (src, mass, m_nuc), vals in sorted(cells.items()):
        avs = sorted({k[0] for k in vals})
        dls = sorted({k[1] for k in vals})
        specs = []
        for det, label in panels:
            g = np.full((len(avs), len(dls)), np.nan)
            for i, av in enumerate(avs):
                for j, dl in enumerate(dls):
                    if (av, dl, det) in vals:
                        g[i, j] = vals[(av, dl, det)]
            specs.append((label, g))
        name = (f"{REPORT_SRC_NAME.get(src, src)} lensed by "
                f"{mass_tag(mass)} {lens_term(mass)}.png")
        draw_heatmap_panels(
            specs, dls, avs, SOURCE_LIBRARY[src]["label"], outdir / name,
            dpi=dpi,
            annotation="Favourable-case peak amplitudes; frozen 18 d template; "
                       "recovery fractions are upper bounds.")
        entries.append((name, src, mass, m_nuc))

    cap = outdir / f"captions_scenarios{suffix}.txt"
    with open(cap, "w", encoding="utf-8") as fh:
        fh.write("% QPLS scenario figure captions (auto-generated by "
                 f"qpls.source_scenarios, report-style, suffix {suffix})\n")
        fh.write("% Source-identifying line stays ON-FIGURE; the parameter line "
                 "below belongs in the LaTeX caption.\n\n")
        for name, src, mass, m_nuc in entries:
            exp = int(np.floor(np.log10(mass)))
            mant = mass / 10.0 ** exp
            mtex = (rf"10^{{{exp}}}" if abs(mant - 1) < 1e-9
                    else rf"{mant:g}\times10^{{{exp}}}")
            fh.write(f"% ---- {name} ----\n")
            fh.write("\\caption{%s Quiescent host, $M_{\\rm tot} = %s\\,M_\\odot$, "
                     "$m_{\\rm nuc} = %g$, $P_{\\rm inj} = %s$\\,d, FAP $= %s$, "
                     "$n_{\\rm inj} = %d$, seed %d. Favourable-case peak "
                     "amplitudes; frozen 18\\,d template; recovery fractions are "
                     "upper bounds. Regenerate: \\texttt{python -m "
                     "qpls.source\\_scenarios -{}-report-style -{}-suffix %s}}\n\n"
                     % (SOURCE_LIBRARY[src]["label"].replace("--", "---") + ".",
                        mtex, m_nuc, P_INJECT, CFG["fap"], CFG["n_inj"], SEED,
                        suffix.replace("_", r"\_")))
    print(f"saved -> {cap}  ({len(entries)} figures)")
    return entries


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _dl_pairs(s):
    out = []
    for item in s.split(","):
        if not item.strip():
            continue
        gpc, dm = item.split(":")
        out.append((float(gpc), float(dm)))
    return out


def _checkpoints(s):
    out = []
    for item in s.split(","):
        if not item.strip():
            continue
        src, dl, exp = item.split(":")
        out.append((src, float(dl), float(exp)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check-only", action="store_true",
                    help="Step 3 only: print the table and checkpoints, no recovery")
    ap.add_argument("--sources", default=None,
                    help="comma-separated source keys (default: all)")
    ap.add_argument("--masses", type=_floats, default=None,
                    help="comma-separated M_total in Msun")
    ap.add_argument("--av", type=_floats, default=None,
                    help="comma-separated A_V in mag")
    ap.add_argument("--mnuc", type=_floats, default=None,
                    help="comma-separated nucleus magnitudes")
    ap.add_argument("--dl", type=_dl_pairs, default=None,
                    help="comma-separated D_L_Gpc:DM pairs, e.g. 0.1:35.00,0.2:36.51")
    ap.add_argument("--checkpoints", type=_checkpoints, default=None,
                    help="comma-separated SOURCE:D_L:expected checkpoint triples")
    ap.add_argument("--suffix", default="",
                    help="suffix for output CSV/figure names (avoids overwriting)")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip figure generation")
    ap.add_argument("--report-style", action="store_true",
                    help="emit report-grade figures + captions from the "
                         "committed CSV for --suffix; runs no recovery")
    args = ap.parse_args(argv)

    if args.report_style:
        FIG_DIR.mkdir(exist_ok=True)
        print(f"Report-style export from "
              f"qpls_source_scenarios{args.suffix}.csv:")
        export_report_style(args.suffix)
        return 0

    global SOURCES, M_TOTAL_GRID, A_V_GRID, M_NUC_GRID, D_L_GRID
    if args.sources is not None:
        keep = [s.strip() for s in args.sources.split(",")]
        unknown = [k for k in keep if k not in SOURCE_LIBRARY]
        if unknown:
            ap.error(f"unknown source(s) {unknown}; "
                     f"available: {list(SOURCE_LIBRARY)}")
        SOURCES = {k: SOURCE_LIBRARY[k] for k in keep}
    if args.masses is not None:
        M_TOTAL_GRID = args.masses
    if args.av is not None:
        A_V_GRID = args.av
    if args.mnuc is not None:
        M_NUC_GRID = args.mnuc
    if args.dl is not None:
        D_L_GRID = args.dl

    cells = build_grid()
    print(f"Scenario grid: {len(SOURCES)} sources x {len(M_TOTAL_GRID)} masses x "
          f"{len(A_V_GRID)} A_V x {len(M_NUC_GRID)} m_nuc x {len(D_L_GRID)} D_L "
          f"= {len(cells)} cells")
    print("SIMPLIFICATION (stated): A_V applied as a pure screen directly in the")
    print("   observing band, source only -- no extinction law, no wavelength term.")
    print("SIMPLIFICATION (stated): mu_cusp is a FAVOURABLE-CASE PEAK (ecc., D_LS at")
    print("   paper fiducial); R_max (Eq. 5) at T=1yr, D_LS=1kpc, e=0, D_L-independent.")
    print()
    n_ap = sum(bool(c["aperture_note"]) for c in cells)
    if n_ap:
        print(f"APERTURE CAVEAT: {n_ap} cell(s) at D_L <= {APERTURE_D_L_MAX} Gpc "
              f"flagged -- {APERTURE_NOTE}.")
        print()
    print_table(cells)
    ok = check_checkpoints(cells, args.checkpoints)
    if not ok:
        print("\nSTOP: pre-registered checkpoint outside tolerance. "
              "Constants NOT adjusted. No recovery run.")
        return 1
    if args.check_only:
        return 0

    n_valid = sum(c["validity"] != "INVALID" for c in cells)
    n_marg = sum(c["validity"] == "VALID-marginal" for c in cells)
    print(f"\nValidity: {n_valid} valid ({n_marg} marginal), "
          f"{len(cells) - n_valid} invalid/excluded")

    valid, results = run_recovery(cells)

    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"qpls_source_scenarios{args.suffix}.csv"
    write_csv(cells, valid, results, csv_path, CFG["n_inj"])
    if not args.no_figures:
        for source in SOURCES:
            plot_source(source, cells, valid, results,
                        FIG_DIR / f"fig3_scenarios_{source}{args.suffix}.png")

    import sys
    cmd = ("python -m qpls.source_scenarios "
           + " ".join(argv if argv is not None else sys.argv[1:])).strip()
    report(cells, valid, results, cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
