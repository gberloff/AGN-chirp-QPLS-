
import argparse
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qpls.injection_recovery import RESULTS_DIR
from qpls.populations import (
    ISO_DIR, COL, PHASE, load_isochrone, cumulative_lf, radius_at,
    giant_branch_RL, lensing_column, delta_m_of_L, D_L_AXIS, DM_AXIS,
    DELTA_M_FLOOR,
)
from qpls.n_exp import einstein_radius_pc, v3_floor_functions, BLIND

POSTMS_DIR = RESULTS_DIR / "Post-MS_giant_results"

R_50 = 192.6
N_METALPOOR_REF = 0.348

F_BURST = 0.1

HOSTS = [
    dict(key="old_metalpoor", label="old, [Fe/H]=-0.25 (10 Gyr)",
         file="MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_basic.iso",
         feh=-0.25, log_age=10.0, colour="C0"),
    dict(key="old_metalrich", label="old, [Fe/H]=+0.25 (10 Gyr)",
         file="MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_basic.iso",
         feh=0.25, log_age=10.0, colour="darkgreen"),
    dict(key="intermediate", label="intermediate, [Fe/H]=+0.25 (1.585 Gyr)",
         file="MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_basic.iso",
         feh=0.25, log_age=9.20, colour="crimson"),
]

WIN_OLD = dict(rgb_tip_R=(120.0, 300.0), rgb_tip_L=(1500.0, 4000.0))
WIN_INT = dict(turnoff_mass=(1.6, 2.2), agb_L=(4000.0, 20000.0),
               agb_R=(250.0, 600.0))

AGB_PHASES = (4, 5)


def describe(iso, key):
    logL, logR, ph = iso[:, COL["log_L"]], iso[:, COL["log_R"]], \
        iso[:, COL["phase"]].astype(int)
    mi = iso[:, COL["initial_mass"]]
    L, R = 10 ** logL, 10 ** logR
    out = {}
    print(f"\n--- {key} ---")
    print(f"  {'phase':9s} {'n':>5s} {'L_tip':>10s} {'R@tip':>8s} {'R_max':>8s} "
          f"{'mass range':>16s}")
    for p in sorted(set(ph)):
        m = ph == p
        i = int(np.argmax(logL[m]))
        print(f"  {PHASE.get(p, p):9s}{m.sum():5d} {L[m].max():10.5g} "
              f"{R[m][i]:8.1f} {R[m].max():8.1f} "
              f"{mi[m].min():7.3f}-{mi[m].max():7.3f}")
    if 2 in set(ph):
        m = ph == 2
        i = int(np.argmax(logL[m]))
        out["rgb_tip_L"], out["rgb_tip_R"] = float(L[m].max()), float(R[m][i])
    else:
        out["rgb_tip_L"] = out["rgb_tip_R"] = None
    agb = np.isin(ph, AGB_PHASES)
    if agb.any():
        i = int(np.argmax(logL[agb]))
        out["agb_L"], out["agb_R"] = float(L[agb].max()), float(R[agb][i])
        out["agb_Rmax"] = float(R[agb].max())
    else:
        out["agb_L"] = out["agb_R"] = out["agb_Rmax"] = None
    out["turnoff_mass"] = float(mi.max())
    print(f"  RGB tip: L = {out['rgb_tip_L']}, R = {out['rgb_tip_R']}")
    print(f"  AGB max-L: L = {out['agb_L']}, R = {out['agb_R']} "
          f"(AGB max R = {out['agb_Rmax']})")
    print(f"  turnoff (most massive living) mass = {out['turnoff_mass']:.3f} Msun")
    return out


def census(iso, R_thresh=R_50):
    Lg = np.geomspace(1.0, 1.0e5, 8000)
    Rg = radius_at(iso, Lg)
    cand = Lg[Rg >= R_thresh]
    if cand.size == 0:
        return None, 0.0
    L_min = float(cand.min())
    return L_min, float(cumulative_lf(iso, np.array([L_min]))[0])


def fig_lf_and_RL(data, path, dpi=300):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    Lg = np.geomspace(1.0, 1.0e5, 400)
    for h in HOSTS:
        d = data[h["key"]]
        axes[0].loglog(Lg, np.maximum(cumulative_lf(d["iso"], Lg), 1e-6),
                       lw=2, color=h["colour"], label=h["label"])
        lu, lr = giant_branch_RL(d["iso"])
        axes[1].loglog(10 ** lu, 10 ** lr, lw=2, color=h["colour"],
                       label=h["label"])
    axes[0].set_xlabel("$L$ (L$_\\odot$)")
    axes[0].set_ylabel("$N(>L)$ per $10^6$ M$_\\odot$")
    axes[0].set_title("cumulative luminosity function", fontsize=10)
    axes[0].set_ylim(1e-3, 1e3)
    axes[1].axhline(R_50, color="k", ls="--", lw=1.2,
                    label=f"$R_{{50}}$ = {R_50:g} R$_\\odot$")
    axes[1].set_xlabel("$L$ (L$_\\odot$)")
    axes[1].set_ylabel("$R$ (R$_\\odot$)")
    axes[1].set_title("giant-branch $R(L)$ vs the blind switch-on", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Host-type axis: luminosity function and radius relation, "
                 "MIST v1.2", fontsize=10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def fig_verdict(curves, path, dpi=300):
    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    for (host, stat, det), (x, y) in curves.items():
        if det != "global_max" or not len(x):
            continue
        h = next(z for z in HOSTS if z["key"] == host)
        ls = "-o" if stat == "v1-frozen" else "--s"
        ax.plot(x, y, ls, ms=4, color=h["colour"],
                label=f"{h['label']} | {stat}")
    ax.axhline(1.0, color="k", lw=1.2, ls=":", label="$N_{\\rm zone}=1$")
    ax.set_yscale("log")
    ax.set_xlabel("$D_L$ (Gpc)")
    ax.set_ylabel("$N_{\\rm zone}$ (detectable stars in $\\pi r_E^2$)")
    ax.set_title("Host-type verdict: detectable stars in the Einstein zone\n"
                 "global-max detector, $A_V$ = 0, "
                 f"$M_{{\\rm lens}}=2\\times10^{{10}}$ M$_\\odot$, "
                 f"$r_E$ = {einstein_radius_pc():.2f} pc", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.text(0.5, 0.008,
             "Intermediate-age curves are per unit stellar mass; multiply by "
             f"f_burst = {F_BURST:g} (CITE-TODO) for a post-starburst nucleus.\n"
             "N_zone is an upper proxy for the caustic-swept area, not an event "
             "rate. R_50 transferred across populations (assumption, see NOTES).",
             ha="center", va="bottom", fontsize=7.5)
    fig.get_layout_engine().set(rect=(0.0, 0.09, 1.0, 0.91))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Host-type axis.")
    ap.parse_args(argv)
    POSTMS_DIR.mkdir(parents=True, exist_ok=True)

    data, stop = {}, False
    for h in HOSTS:
        req = dict(version="1.2", feh=h["feh"], vvcrit=0.0, log_age=h["log_age"])
        iso = load_isochrone(ISO_DIR / h["file"], verbose=True, require=req)
        print(f"  gate PASS: [Fe/H]={h['feh']:+.2f}, log10(age)={h['log_age']}, "
              f"{len(iso)} rows")
        data[h["key"]] = dict(iso=iso, **describe(iso, h["key"]))

    print("\n=== SANITY WINDOWS ===")
    for key in ("old_metalpoor", "old_metalrich"):
        d = data[key]
        for fld, (lo, hi) in WIN_OLD.items():
            v = d[fld]
            ok = v is not None and lo <= v <= hi
            stop |= not ok
            print(f"  {key:14s} {fld:12s} = {v:9.4g}  window [{lo:g},{hi:g}]  "
                  f"{'PASS' if ok else 'FAIL'}")
    d = data["intermediate"]
    for fld, (lo, hi) in [("turnoff_mass", WIN_INT["turnoff_mass"]),
                          ("agb_L", WIN_INT["agb_L"]),
                          ("agb_Rmax", WIN_INT["agb_R"])]:
        v = d[fld]
        ok = v is not None and lo <= v <= hi
        stop |= not ok
        print(f"  {'intermediate':14s} {fld:12s} = {v:9.4g}  "
              f"window [{lo:g},{hi:g}]  {'PASS' if ok else 'FAIL'}")
    if stop:
        print("\nSTOP: a sanity window failed. Nothing tuned.")
        return 1

    print("\n=== DECISIVE MEASUREMENT: metal-rich 10 Gyr RGB tip vs R_50 ===")
    r_mr = data["old_metalrich"]["rgb_tip_R"]
    r_mp = data["old_metalpoor"]["rgb_tip_R"]
    print(f"  metal-poor RGB tip R = {r_mp:.1f} Rsun   (R_50 = {R_50:g})")
    print(f"  metal-rich RGB tip R = {r_mr:.1f} Rsun   "
          f"-> {'ABOVE R_50: RGB population crosses the blind switch-on'
               if r_mr >= R_50 else
               'BELOW R_50: old metal-rich hosts fail by census too'}")

    print("\n=== STEP 2: census at R_50 ===")
    print(f"  {'host':14s} {'L(R_50)':>10s} {'N(>L) per 1e6 Msun':>20s} "
          f"{'vs metal-poor':>14s}")
    for h in HOSTS:
        L_min, N = census(data[h["key"]]["iso"])
        data[h["key"]]["L_min"], data[h["key"]]["N"] = L_min, N
        ratio = N / N_METALPOOR_REF if N_METALPOOR_REF else float("nan")
        print(f"  {h['key']:14s} "
              f"{(f'{L_min:.1f}' if L_min else 'unreached'):>10s} "
              f"{N:20.4g} {ratio:13.1f}x")
    print(f"  (committed old metal-poor reference: {N_METALPOOR_REF})")

    r_E = einstein_radius_pc()
    area = np.pi * r_E ** 2
    column = lensing_column()
    f3, _ = v3_floor_functions()
    rows, curves = [], {}
    print(f"\n=== STEP 3: N_zone (r_E = {r_E:.4f} pc, column = {column:.4g} "
          f"Msun/pc^2) ===")
    for h in HOSTS:
        iso = data[h["key"]]["iso"]
        for stat in ("v1-frozen", "v3-physical"):
            for det in BLIND:
                xs, ys = [], []
                for dl, dm in zip(D_L_AXIS, DM_AXIS):
                    Lg = np.geomspace(1.0, 1.0e5, 4000)
                    amp = delta_m_of_L(iso, Lg, dm, 0.0)
                    if stat == "v1-frozen":
                        ok = amp >= DELTA_M_FLOOR[det]
                    else:
                        fl = f3[det](radius_at(iso, Lg)) if f3[det] else np.inf
                        ok = amp >= fl
                    if not np.any(ok):
                        nz = np.nan
                        L_min = np.nan
                    else:
                        L_min = float(Lg[int(np.argmax(ok))])
                        N = float(cumulative_lf(iso, np.array([L_min]))[0])
                        nz = column * N / 1e6 * area
                    rows.append(dict(
                        host=h["key"], host_label=h["label"], statistic=stat,
                        detector=det, D_L_Gpc=dl, DM=dm, A_V=0.0,
                        L_min_Lsun=(f"{L_min:.6g}" if np.isfinite(L_min) else ""),
                        N_zone=(f"{nz:.6g}" if np.isfinite(nz) else ""),
                        N_zone_f_burst=(f"{nz*F_BURST:.6g}"
                                        if (np.isfinite(nz) and
                                            h["key"] == "intermediate") else ""),
                        f_burst=(F_BURST if h["key"] == "intermediate" else ""),
                        r_E_pc=f"{r_E:.6g}", column_Msun_per_pc2=f"{column:.6g}"))
                    if np.isfinite(nz) and nz > 0:
                        xs.append(dl); ys.append(nz)
                curves[(h["key"], stat, det)] = (xs, ys)
                if det == "global_max":
                    lab = (f"{h['key']:14s} {stat:12s}")
                    if xs:
                        ge1 = [x for x, y in zip(xs, ys) if y >= 1.0]
                        print(f"  {lab} N_zone(0.1 Gpc) = {ys[0]:10.4g}   "
                              f"largest D_L with N_zone >= 1: "
                              f"{(f'{max(ge1):g} Gpc' if ge1 else 'none')}")
                    else:
                        print(f"  {lab} no detectable cells")

    ni = [v for (hk, st, dt), (xs, v) in curves.items()
          if hk == "intermediate" and st == "v3-physical" and dt == "global_max"]
    if ni and ni[0]:
        print(f"\n  intermediate-age, v3, global-max, 0.1 Gpc: "
              f"{ni[0][0]:.4g} per unit stellar mass -> "
              f"{ni[0][0]*F_BURST:.4g} at f_burst = {F_BURST:g} (CITE-TODO)")

    out = POSTMS_DIR / "qpls_host_types.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {out}  ({len(rows)} rows)")

    fig_lf_and_RL(data, POSTMS_DIR / "fig13_host_populations.png")
    fig_verdict(curves, POSTMS_DIR / "fig13_host_verdict.png")
    print(f"\nRegeneration:\n  python -m qpls.host_types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
