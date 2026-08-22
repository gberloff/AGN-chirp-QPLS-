
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.timeseries import LombScargle

RNG = np.random.default_rng(0)

def ztf_cadence(baseline_days=5 * 365.25, dt=3.0, season_frac=0.66,
                jitter=0.4, err_med=0.02, err_scatter=0.007, rng=RNG):
    t = np.arange(0.0, baseline_days, dt)
    t = t + rng.normal(0, jitter, size=t.size)
    phase = (t % 365.25) / 365.25
    t = t[phase < season_frac]
    t.sort()
    sigma = np.abs(rng.normal(err_med, err_scatter, size=t.size)) + 1e-3
    return t, sigma


def simulate_drw(t, tau, sf_inf, rng=RNG):
    n = t.size
    var = sf_inf ** 2 / 2.0
    x = np.empty(n)
    x[0] = rng.normal(0, np.sqrt(var))
    dt = np.diff(t)
    for i in range(1, n):
        rho = np.exp(-dt[i - 1] / tau)
        x[i] = rho * x[i - 1] + rng.normal(0, np.sqrt(var * (1 - rho ** 2)))
    return x


def flare_train(t, P_obs, A_mag, fwhm_days=4.0, rng=RNG):
    if A_mag <= 0:
        return np.zeros_like(t)
    sig = fwhm_days / 2.3548
    t0 = rng.uniform(0, P_obs)
    peaks = np.arange(t0, t.max() + P_obs, P_obs)
    f = np.zeros_like(t)
    for p in peaks:
        f += np.exp(-0.5 * ((t - p) / sig) ** 2)
    return A_mag * f


def flare_train_sawtooth(t, A_mag, P, t0, width_days, rise_frac=0.1):
    if A_mag <= 0:
        return np.zeros_like(t)
    peak_at = rise_frac * width_days
    f = np.zeros_like(t)
    for p in np.arange(t0, t.max() + P, P):
        dt = t - p
        inside = (dt >= 0.0) & (dt <= width_days)
        rise = inside & (dt < peak_at)
        decay = inside & (dt >= peak_at)
        f[rise] += dt[rise] / peak_at
        f[decay] += (width_days - dt[decay]) / (width_days - peak_at)
    return A_mag * f


def make_lc(t, sigma, tau, sf_inf, P_obs=None, A_mag=0.0, fwhm_days=4.0, rng=RNG,
            template="gaussian"):
    mag = simulate_drw(t, tau, sf_inf, rng)
    mag = mag + rng.normal(0, sigma)
    if P_obs is not None and A_mag > 0:
        if template == "gaussian":
            mag = mag - flare_train(t, P_obs, A_mag, fwhm_days, rng)
        elif template == "sawtooth":
            t0 = rng.uniform(0, P_obs)
            mag = mag - flare_train_sawtooth(t, A_mag, P_obs, t0,
                                             2.0 * fwhm_days)
        else:
            raise ValueError(f"unknown template {template!r}")
    return mag


def gls_peak_power(t, mag, sigma, freq):
    power = LombScargle(t, mag, sigma).power(freq, method="fast")
    k = np.argmax(power)
    return power[k], 1.0 / freq[k]


def gls_harmonic_sum(t, mag, sigma, freq, n_harmonics=3):
    power = LombScargle(t, mag, sigma).power(freq, method="fast")
    df = freq[1] - freq[0]
    score = power.copy()
    for h in range(2, n_harmonics + 1):
        f_h = h * freq
        idx = np.searchsorted(freq, f_h).clip(0, len(freq) - 1)
        in_band = np.abs(freq[idx] - f_h) <= df
        score[in_band] += power[idx[in_band]]
    k = np.argmax(score)
    return score[k], 1.0 / freq[k]


def gls_harmonic_sum_weighted(t, mag, sigma, freq, n_harmonics=3,
                              median_window=51):
    from scipy.ndimage import median_filter
    power = LombScargle(t, mag, sigma).power(freq, method="fast")
    df = freq[1] - freq[0]
    noise = np.maximum(median_filter(power, size=median_window, mode="reflect"),
                       1e-10)
    whitened = power / noise
    score = whitened.copy()
    for h in range(2, n_harmonics + 1):
        f_h = h * freq
        idx = np.searchsorted(freq, f_h).clip(0, len(freq) - 1)
        in_band = np.abs(freq[idx] - f_h) <= df
        score[in_band] += whitened[idx[in_band]]
    k = np.argmax(score)
    return score[k], 1.0 / freq[k]


def _p_in_search_range(P, freq):
    return (1.0 / P) >= freq[0] and (1.0 / P) <= freq[-1]


def null_threshold(t, sigma, tau, sf_inf, freq, detector_fn,
                   fap=0.01, n_null=1000, rng=RNG):
    peaks = np.empty(n_null)
    for i in range(n_null):
        mag = make_lc(t, sigma, tau, sf_inf, P_obs=None, A_mag=0.0, rng=rng)
        peaks[i], _ = detector_fn(t, mag, sigma, freq)
    return np.quantile(peaks, 1 - fap), peaks


def recovery_grid(t, sigma, tau, sf_inf, P_grid, A_grid,
                  freq, threshold, detector_fn,
                  n_inj=200, fwhm_days=4.0, rng=RNG, template="gaussian"):
    R = np.full((A_grid.size, P_grid.size), np.nan)
    for j, P in enumerate(P_grid):
        if not _p_in_search_range(P, freq):
            print(f'   P={P:6.2f} d  OUT OF SEARCH RANGE', flush=True)
            continue
        for i, A in enumerate(A_grid):
            hits = 0
            for _ in range(n_inj):
                mag = make_lc(t, sigma, tau, sf_inf, P_obs=P, A_mag=A,
                              fwhm_days=fwhm_days, rng=rng, template=template)
                pk, _ = detector_fn(t, mag, sigma, freq)
                if pk > threshold:
                    hits += 1
            R[i, j] = hits / n_inj
        print(f'   P={P:6.2f} d done', flush=True)
    return R


RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def plot_recovery_map(results_path=RESULTS_DIR / "qpls_results.json",
                      fig_path=RESULTS_DIR / "qpls_recovery_map.png"):
    with open(results_path) as fh:
        res = json.load(fh)

    cfg = res["cfg"]
    P = np.asarray(res["P_grid"])
    A = np.asarray(res["A_grid"])

    panel_order = [("AGN_global_max", "AGN global-max"),
                   ("AGN_harmonic_sum", "AGN harmonic-sum"),
                   ("quiescent_global_max", "quiescent global-max"),
                   ("quiescent_harmonic_sum", "quiescent harmonic-sum")]
    panels = [(k, t) for k, t in panel_order if k in res]

    n_maps = len(panels)
    fig = plt.figure(figsize=(4.5 * (n_maps + 1), 5), constrained_layout=True)
    gs = fig.add_gridspec(1, n_maps + 1,
                          width_ratios=[1] * n_maps + [1.15])

    norm = Normalize(vmin=0, vmax=1)
    contour_levels = [0.5, 0.9]
    map_axes = []

    for idx, (key, title_prefix) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        map_axes.append(ax)
        R = np.asarray(res[key]["R"], dtype=float)

        thr_str = f"thr={res[key]['threshold']:.3f}"
        sf = cfg["sf_inf_AGN"] if "AGN" in key else cfg["sf_inf_quiescent"]

        R_plot = np.ma.masked_invalid(R)
        ax.set_facecolor("0.85")

        im = ax.pcolormesh(P, A, R_plot, norm=norm, cmap="inferno",
                           shading="nearest")
        if np.any(np.isfinite(R)):
            R_filled = np.where(np.isfinite(R), R, 0.0)
            cs = ax.contour(P, A, R_filled, levels=contour_levels,
                            colors=["cyan", "lime"], linewidths=1.4)
            ax.clabel(cs, fmt="%g", fontsize=8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Recurrence period $P_{\\rm obs}$ (days)")
        ax.set_ylabel("Flare amplitude $A_{\\rm mag}$ (mag)")
        ax.set_title(f"{title_prefix}\nSF$_\\infty$={sf}  {thr_str}  "
                     f"FAP={cfg['fap']}")

    cb = fig.colorbar(im, ax=map_axes, fraction=0.03, pad=0.01)
    cb.set_label("Recovery fraction")

    ax_lc = fig.add_subplot(gs[0, n_maps])
    rng_lc = np.random.default_rng(99)
    t, sigma = ztf_cadence(cfg["baseline_yr"] * 365.25, cfg["cadence_days"],
                           cfg["season_frac"], err_med=cfg["err_med"], rng=rng_lc)
    P_ex = P[len(P) // 2]
    A_ex = A[len(A) // 2]
    mag_bg = make_lc(t, sigma, cfg["tau_drw_days"], cfg["sf_inf_quiescent"],
                     rng=rng_lc)
    mag_sig = make_lc(t, sigma, cfg["tau_drw_days"], cfg["sf_inf_quiescent"],
                      P_obs=P_ex, A_mag=A_ex, fwhm_days=cfg["fwhm_days"],
                      rng=rng_lc)

    ax_lc.scatter(t, mag_bg, s=2, alpha=0.4, color="0.55", label="DRW only")
    ax_lc.scatter(t, mag_sig, s=2, alpha=0.6, color="C1", label="DRW + QPLS")
    ax_lc.invert_yaxis()
    ax_lc.set_xlabel("Time (days)")
    ax_lc.set_ylabel("Relative mag")
    ax_lc.set_title(f"Example LC  P={P_ex:.1f} d, A={A_ex:.2f} mag\n"
                    f"quiescent host, FWHM={cfg['fwhm_days']} d")
    ax_lc.legend(fontsize=7, markerscale=3)

    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"saved -> {pathlib.Path(fig_path).name}")
    return fig_path


if __name__ == "__main__":

    cfg = dict(
        baseline_yr=5.0, cadence_days=3.0, season_frac=0.66,
        err_med=0.02, fwhm_days=4.0,
        fap=0.01, n_null=200, n_inj=200,
        tau_drw_days=200.0,
        sf_inf_AGN=0.20,
        sf_inf_quiescent=0.02,
        n_harmonics=3,
    )

    t, sigma = ztf_cadence(cfg["baseline_yr"] * 365.25, cfg["cadence_days"],
                           cfg["season_frac"], err_med=cfg["err_med"])

    p_min = 2 * cfg["cadence_days"]
    p_max = 0.5 * cfg["baseline_yr"] * 365.25

    P_grid = np.geomspace(4.0, 60.0, 10)
    A_grid = np.geomspace(0.02, 3.0, 10)
    freq = np.linspace(1.0 / p_max, 1.0 / p_min, 1000)

    from functools import partial

    results = {"cfg": cfg, "p_min": p_min, "p_max": p_max,
               "P_grid": P_grid.tolist(), "A_grid": A_grid.tolist(),
               "n_epochs": int(t.size)}

    tau = cfg["tau_drw_days"]
    n_h = cfg["n_harmonics"]

    detectors = [
        ("global_max", gls_peak_power),
        ("harmonic_sum", partial(gls_harmonic_sum, n_harmonics=n_h)),
    ]

    for bg_label, sf in [("AGN", cfg["sf_inf_AGN"]),
                         ("quiescent", cfg["sf_inf_quiescent"])]:
        for det_label, det_fn in detectors:
            key = f"{bg_label}_{det_label}"
            rng = np.random.default_rng(42)
            thr, nullpk = null_threshold(t, sigma, tau, sf, freq, det_fn,
                                         cfg["fap"], cfg["n_null"], rng)
            print(f'[{key}] null done thr={thr:.4f}', flush=True)
            R = recovery_grid(t, sigma, tau, sf, P_grid, A_grid, freq, thr,
                              det_fn, cfg["n_inj"], cfg["fwhm_days"], rng)
            results[key] = {
                "threshold": float(thr), "R": R.tolist(),
                "null_med": float(np.median(nullpk)),
                "detector": det_label, "n_harmonics": n_h}
            valid = R[np.isfinite(R)]
            peak = valid.max() if valid.size else 0.0
            print(f"[{key}] peak_recovery={peak:.2f}")

    with open(RESULTS_DIR / "qpls_results.json", "w") as fh:
        json.dump(results, fh)
    print("saved -> qpls_results.json")

    plot_recovery_map()
