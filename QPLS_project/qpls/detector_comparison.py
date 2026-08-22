
import argparse
import json
import pathlib
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.timeseries import LombScargle

from qpls.injection_recovery import (
    RESULTS_DIR, ztf_cadence, make_lc, gls_peak_power, gls_harmonic_sum,
    null_threshold, recovery_grid, _p_in_search_range,
)

CFG = dict(
    baseline_yr=5.0, cadence_days=3.0, season_frac=0.66,
    err_med=0.02, fwhm_days=4.0,
    fap=0.01, n_null=200, n_inj=200,
    tau_drw_days=200.0,
    sf_inf_AGN=0.20,
    sf_inf_quiescent=0.02,
    n_harmonics=3,
)


def _power_at(t, mag, sigma, f0):
    return LombScargle(t, mag, sigma).power(np.asarray([f0]), method="cython")[0]


def oracle_recovery(t, sigma, tau, sf_inf, P_grid, A_grid, freq,
                    fap=0.01, n_null=200, n_inj=200, fwhm_days=4.0, seed=42,
                    template="gaussian"):
    rng = np.random.default_rng(seed)
    in_range = [(j, P) for j, P in enumerate(P_grid)
                if _p_in_search_range(P, freq)]

    null_pow = {j: np.empty(n_null) for j, _ in in_range}
    for k in range(n_null):
        mag = make_lc(t, sigma, tau, sf_inf, P_obs=None, A_mag=0.0, rng=rng)
        ls = LombScargle(t, mag, sigma)
        for j, P in in_range:
            null_pow[j][k] = ls.power(np.asarray([1.0 / P]), method="cython")[0]
    thr = {j: float(np.quantile(null_pow[j], 1 - fap)) for j, _ in in_range}

    R = np.full((A_grid.size, P_grid.size), np.nan)
    for j, P in in_range:
        f0 = 1.0 / P
        for i, A in enumerate(A_grid):
            hits = 0
            for _ in range(n_inj):
                mag = make_lc(t, sigma, tau, sf_inf, P_obs=P, A_mag=A,
                              fwhm_days=fwhm_days, rng=rng, template=template)
                if _power_at(t, mag, sigma, f0) > thr[j]:
                    hits += 1
            R[i, j] = hits / n_inj
        print(f'   [oracle] P={P:6.2f} d done', flush=True)

    thr_by_period = {float(P): thr[j] for j, P in in_range}
    return thr, R, thr_by_period


def build_panels(host, cfg=CFG):
    sf = cfg["sf_inf_AGN"] if host == "AGN" else cfg["sf_inf_quiescent"]
    tau = cfg["tau_drw_days"]

    t, sigma = ztf_cadence(cfg["baseline_yr"] * 365.25, cfg["cadence_days"],
                           cfg["season_frac"], err_med=cfg["err_med"])

    p_min = 2 * cfg["cadence_days"]
    p_max = 0.5 * cfg["baseline_yr"] * 365.25
    P_grid = np.geomspace(4.0, 60.0, 10)
    A_grid = np.geomspace(0.02, 3.0, 10)
    freq = np.linspace(1.0 / p_max, 1.0 / p_min, 1000)

    panels = {}

    rng = np.random.default_rng(42)
    thr_gm, _ = null_threshold(t, sigma, tau, sf, freq, gls_peak_power,
                               cfg["fap"], cfg["n_null"], rng)
    print(f'[{host} global_max] null done thr={thr_gm:.4f}', flush=True)
    R_gm = recovery_grid(t, sigma, tau, sf, P_grid, A_grid, freq, thr_gm,
                         gls_peak_power, cfg["n_inj"], cfg["fwhm_days"], rng)
    panels["global_max"] = dict(threshold=float(thr_gm), R=R_gm)

    thr_or, R_or, thr_by_period = oracle_recovery(
        t, sigma, tau, sf, P_grid, A_grid, freq,
        cfg["fap"], cfg["n_null"], cfg["n_inj"], cfg["fwhm_days"], seed=42)
    print(f'[{host} oracle] thresholds '
          f'{min(thr_or.values()):.4f}-{max(thr_or.values()):.4f}', flush=True)
    panels["oracle_single_freq"] = dict(threshold=thr_by_period, R=R_or)

    det_hs = partial(gls_harmonic_sum, n_harmonics=cfg["n_harmonics"])
    rng = np.random.default_rng(42)
    thr_hs, _ = null_threshold(t, sigma, tau, sf, freq, det_hs,
                               cfg["fap"], cfg["n_null"], rng)
    print(f'[{host} harmonic_sum] null done thr={thr_hs:.4f}', flush=True)
    R_hs = recovery_grid(t, sigma, tau, sf, P_grid, A_grid, freq, thr_hs,
                         det_hs, cfg["n_inj"], cfg["fwhm_days"], rng)
    panels["blind_harmonic_sum"] = dict(threshold=float(thr_hs), R=R_hs)

    known = {("AGN", "global_max"): 0.7191, ("AGN", "harmonic_sum"): 1.2565,
             ("quiescent", "global_max"): 0.4277,
             ("quiescent", "harmonic_sum"): 0.8047}
    for det, thr in [("global_max", thr_gm), ("harmonic_sum", thr_hs)]:
        exp = known[(host, det)]
        assert abs(thr - exp) < 5e-4, f"{host} {det}: {thr:.4f} != {exp}"
    print(f'[{host}] threshold self-check passed', flush=True)

    return dict(sf=sf, P_grid=P_grid, A_grid=A_grid, panels=panels)


PANEL_ORDER = [
    ("global_max", "Global-max"),
    ("oracle_single_freq", "Oracle single-freq\n(knows injected P)"),
    ("blind_harmonic_sum", "Blind harmonic-sum"),
]


def _thr_str(threshold):
    if isinstance(threshold, dict):
        vals = list(threshold.values())
        return f"thr = {min(vals):.3f}–{max(vals):.3f}"
    return f"thr = {threshold:.3f}"


def plot(host, built, fig_path, dpi=180):
    cfg = CFG
    P, A = built["P_grid"], built["A_grid"]
    sf = built["sf"]

    norm = Normalize(vmin=0, vmax=1)
    contour_levels = [0.5, 0.9]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    im = None
    for ax, (key, title) in zip(axes, PANEL_ORDER):
        R = np.asarray(built["panels"][key]["R"], dtype=float)
        ax.set_facecolor("0.85")
        R_plot = np.ma.masked_invalid(R)
        im = ax.pcolormesh(P, A, R_plot, norm=norm, cmap="inferno",
                           shading="nearest")
        if np.any(np.isfinite(R)):
            R_filled = np.where(np.isfinite(R), R, 0.0)
            cs = ax.contour(P, A, R_filled, levels=contour_levels,
                            colors=["cyan", "lime"], linewidths=1.4)
            ax.clabel(cs, fmt="%g", fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Recurrence period $P_{\\rm obs}$ (days)", fontsize=14)
        ax.set_ylabel("Flare amplitude $A_{\\rm mag}$ (mag)", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.tick_params(axis="both", which="minor", labelsize=10)
        thr_str = _thr_str(built["panels"][key]["threshold"])
        ax.set_title(f"{title}\n{thr_str}   FAP={cfg['fap']}", fontsize=14)

    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.01)
    cb.set_label("Recovery fraction", fontsize=13)
    cb.ax.tick_params(labelsize=11)

    host_label = "AGN" if host == "AGN" else "quiescent"
    fig.suptitle(
        f"{host_label} host (SF$_\\infty$={sf:.2f} mag, "
        f"$\\tau$={int(cfg['tau_drw_days'])} d, FWHM={cfg['fwhm_days']:.0f} d)"
        f"  —  detector comparison",
        fontsize=16)

    fig.savefig(fig_path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {pathlib.Path(fig_path).name}  (dpi={dpi})")


def dump_json(host, built, json_path):
    out = {
        "description": f"{host} detector comparison: recovery vs detector type",
        "fiducials": {"SF_inf_mag": built["sf"], "tau_d": int(CFG["tau_drw_days"]),
                      "FWHM_d": CFG["fwhm_days"], "FAP": CFG["fap"],
                      "n_inj": CFG["n_inj"], "n_null": CFG["n_null"]},
        "P_grid_d": built["P_grid"].tolist(),
        "A_grid_mag": built["A_grid"].tolist(),
    }
    for key, _ in PANEL_ORDER:
        out[key] = {"threshold": built["panels"][key]["threshold"],
                    "R": np.asarray(built["panels"][key]["R"]).tolist()}
    with open(json_path, "w") as fh:
        json.dump(out, fh)
    print(f"saved -> {pathlib.Path(json_path).name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("host", choices=["AGN", "quiescent", "both"],
                    default="quiescent")
    args = ap.parse_args()

    hosts = ["AGN", "quiescent"] if args.host == "both" else [args.host]
    import qpls.injection_recovery as _ir

    FIG_DIR = RESULTS_DIR.parent / "figures"
    FIG_DIR.mkdir(exist_ok=True)
    PUB_NAME = {"AGN": "fig1_AGN_regenerated.png",
                "quiescent": "fig2_quiescent_regenerated.png"}

    for host in hosts:
        _ir.RNG.bit_generator.state = np.random.default_rng(0).bit_generator.state
        suffix = "" if host == "AGN" else "_quiescent"
        fig_path = RESULTS_DIR / f"qpls_detector_comparison{suffix}.png"
        json_path = RESULTS_DIR / f"qpls_detector_comparison{suffix}.json"

        built = build_panels(host)
        plot(host, built, fig_path)
        plot(host, built, FIG_DIR / PUB_NAME[host], dpi=300)
        dump_json(host, built, json_path)
