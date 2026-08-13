"""THE analysis path.

`analyse` is the only route by which any light curve is ever scored.  Three
thin callers wrap it:

    run_on_data(...)       real light curves
    run_on_null(...)       generate background, then analyse
    run_on_injection(...)  generate background, add signal, then analyse

Every step that inspects the data before producing the final score lives
inside `analyse`: time ordering, the band set, t_ref and T_span, the search
grid, the null fit, the positive-frequency check and the boundary flags.
Nothing may reach the algebra except through `analyse`.

Relationship to the parent run.  The copied modules `e5_lib.py`,
`detectors.py` and `background.py` are kept verbatim as the audit reference.
The algebra below is the band-general form of detector A: for two bands it
builds a bitwise identical design matrix and returns bitwise identical
results, which `stage0_verify.py` asserts.  Generalising was necessary because
the 1-band configuration in stage 1 makes the parent's fixed 10-column design
rank-deficient.
"""
from dataclasses import dataclass, asdict, field
import time

import numpy as np
import celerite2
from celerite2 import terms
from scipy.optimize import minimize


@dataclass(frozen=True)
class Spec:
    """Everything `analyse` is allowed to know.  No hidden state anywhere."""
    P_min_d: float
    P_max_d: float
    P_n: int
    eta_min: float
    eta_max: float
    eta_n: int
    null_family: str = "drw"
    sigma_bounds_mag: tuple = (0.001, 0.5)
    tau_bounds_d: tuple = (5.0, 20000.0)
    restarts: int = 3
    starts: tuple = ((0.06, 320.0), (0.02, 60.0), (0.15, 3000.0))
    boundary_tol_frac: float = 1e-3
    require_positive_frequency: bool = True
    n_harmonics: int = 2
    sort_times: bool = True
    label: str = ""
    # Quality-cut parameters are declared on the Spec and applied inside
    # analyse, so real data, nulls and injections run identical cuts.
    apply_quality_cuts: bool = True
    cut_catflags: bool = True
    cut_magerr_max: float = 0.20
    cut_mad_n: float = 5.0
    cut_running_median_window_d: float = 50.0
    min_epochs: int = 100

    @staticmethod
    def from_config(cfg, entry):
        d = cfg["detector"]
        q = cfg.get("quality_cuts", {})
        return Spec(
            apply_quality_cuts=bool(q.get("enabled", True)),
            cut_catflags=bool(q.get("catflags", True)),
            cut_magerr_max=float(q.get("magerr_max_mag", 0.20)),
            cut_mad_n=float(q.get("mad_n", 5.0)),
            cut_running_median_window_d=float(q.get("running_median_window_d", 50.0)),
            min_epochs=int(q.get("min_epochs", 100)),
            P_min_d=entry["grid_P"]["min_d"],
            P_max_d=entry["grid_P"]["max_d"],
            P_n=entry["grid_P"]["n"],
            eta_min=entry["grid_eta"]["min"],
            eta_max=entry["grid_eta"]["max"],
            eta_n=entry["grid_eta"]["n"],
            null_family="drw",
            sigma_bounds_mag=tuple(d["sigma_bounds_mag"]),
            tau_bounds_d=tuple(d["tau_bounds_d"]),
            restarts=int(d["restarts"]),
            starts=tuple(tuple(s) for s in d["starts"]),
            boundary_tol_frac=float(d["boundary_tol_frac"]),
            label=entry.get("label", ""),
        )


@dataclass
class AnalysisResult:
    Lambda_osc: float = np.nan
    Lambda_per0: float = np.nan
    DeltaLambda_chirp: float = np.nan
    P_hat: float = np.nan
    eta_hat: float = np.nan
    sigma_hat: float = np.nan
    tau_hat: float = np.nan
    l0: float = np.nan
    boundary_flag: int = 0
    converged: int = 0
    n_restarts_used: int = 0
    n_restarts_ok: int = 0
    best_restart: int = -1
    n_epochs: int = 0
    n_epochs_in: int = 0
    n_removed_catflags: int = 0
    n_removed_magerr: int = 0
    n_removed_mad: int = 0
    n_removed_total: int = 0
    insufficient_data: int = 0
    n_bands: int = 0
    n_linear_params: int = 0
    n_grid_nodes: int = 0
    n_node_fail: int = 0
    n_node_negfreq: int = 0
    argmax_ip: int = -1
    argmax_je: int = -1
    runtime_s: float = np.nan
    fields_compared: tuple = field(default=(), repr=False)

    # fields compared for the bitwise identity/determinism tests (runtime excluded)
    COMPARE = ("Lambda_osc", "Lambda_per0", "DeltaLambda_chirp", "P_hat", "eta_hat",
               "sigma_hat", "tau_hat", "l0", "boundary_flag", "converged",
               "n_restarts_used", "n_restarts_ok", "best_restart", "n_epochs",
               "n_bands", "n_linear_params", "n_grid_nodes", "n_node_fail",
               "n_node_negfreq", "argmax_ip", "argmax_je",
               "n_epochs_in", "n_removed_catflags", "n_removed_magerr",
               "n_removed_mad", "n_removed_total", "insufficient_data")

    def key(self):
        return tuple(getattr(self, k) for k in self.COMPARE)

    def as_dict(self):
        d = asdict(self)
        d.pop("fields_compared", None)
        return d


def build_grid(spec):
    P = np.logspace(np.log10(spec.P_min_d), np.log10(spec.P_max_d), spec.P_n)
    step = (spec.eta_max - spec.eta_min) / (spec.eta_n - 1)
    eta = np.round(np.arange(spec.eta_min, spec.eta_max + 0.5 * step, step), 10)
    assert eta.size == spec.eta_n, (eta.size, spec.eta_n)
    assert np.isclose(eta, 0.0).sum() == 1, "eta = 0 must be on the grid"
    return P, eta


def design_block(t, band_idx, n_b, P, eta, t_ref, T_span, n_harm=2):
    """(N, n_eta * (4*n_b + n_b)) design matrix for one P over all eta.

    Column order within an eta block, for bands in sorted label order:
        band0: sin(phi), cos(phi), sin(2phi), cos(2phi)
        band1: sin(phi), cos(phi), sin(2phi), cos(2phi)
        ...
        mu_band0, mu_band1, ...
    For n_b = 2 this is exactly e5_lib.design_block's ordering.
    """
    x = t - t_ref
    f0 = 1.0 / P
    phi = 2.0 * np.pi * f0 * (x[None, :] + 0.5 * eta[:, None] * x[None, :] ** 2 / T_span)
    ne, N = phi.shape
    n_col = (2 * n_harm) * n_b + n_b
    A = np.zeros((N, ne, n_col))
    masks = [(band_idx == j).astype(float) for j in range(n_b)]
    for h in range(n_harm):
        k = h + 1
        S = np.sin(k * phi).T                      # (N, ne)
        C = np.cos(k * phi).T
        for j in range(n_b):
            A[:, :, (2 * n_harm) * j + 2 * h] = S * masks[j][:, None]
            A[:, :, (2 * n_harm) * j + 2 * h + 1] = C * masks[j][:, None]
    for j in range(n_b):
        A[:, :, (2 * n_harm) * n_b + j] = masks[j][:, None]
    return A.reshape(N, ne * n_col)


def _offsets(band_idx, n_b):
    return np.column_stack([(band_idx == j).astype(float) for j in range(n_b)])


def _term(spec, p):
    if spec.null_family == "drw":
        sigma, tau = p
        return terms.RealTerm(a=sigma ** 2, c=1.0 / tau)
    raise ValueError(spec.null_family)


# The quality cuts live INSIDE analyse and therefore run on real data, on
# every null and on every injection, without exception.
def _running_median_per_band(t, y, b, window_d):
    """Median of all same-band points within +/- window/2 of each epoch.

    t must be sorted ascending.  Returns an array the same shape as y.
    """
    half = 0.5 * window_d
    out = np.empty(y.size, dtype=float)
    for lab in np.unique(b):
        m = np.flatnonzero(b == lab)
        tb, yb = t[m], y[m]
        lo = np.searchsorted(tb, tb - half, side="left")
        hi = np.searchsorted(tb, tb + half, side="right")
        for k in range(tb.size):
            out[m[k]] = np.median(yb[lo[k]:hi[k]])
    return out


def apply_quality_cuts(t, y, sig, b, catflags, spec):
    """Return (keep_mask, counts).  Cuts are applied in the declared order.

    1. catflags != 0                      (no-op on synthetic: catflags all 0)
    2. magnitude error > cut_magerr_max
    3. |residual from a 50-day per-band running median| > cut_mad_n * MAD
    """
    n = t.size
    keep = np.ones(n, dtype=bool)
    counts = dict(n_removed_catflags=0, n_removed_magerr=0, n_removed_mad=0)

    if spec.cut_catflags:
        bad = np.asarray(catflags) != 0
        counts["n_removed_catflags"] = int(np.count_nonzero(bad & keep))
        keep &= ~bad

    bad = sig > spec.cut_magerr_max
    counts["n_removed_magerr"] = int(np.count_nonzero(bad & keep))
    keep &= ~bad

    # cut 3 is evaluated on the survivors of cuts 1-2
    idx = np.flatnonzero(keep)
    if idx.size >= 3:
        tk, yk, bk = t[idx], y[idx], b[idx]
        rm = _running_median_per_band(tk, yk, bk, spec.cut_running_median_window_d)
        resid = yk - rm
        drop_local = np.zeros(idx.size, dtype=bool)
        for lab in np.unique(bk):
            m = bk == lab
            r = resid[m]
            mad = float(np.median(np.abs(r - np.median(r))))
            if mad > 0.0:
                drop_local[m] = np.abs(r) > spec.cut_mad_n * mad
        counts["n_removed_mad"] = int(np.count_nonzero(drop_local))
        keep[idx[drop_local]] = False

    counts["n_removed_total"] = int(n - np.count_nonzero(keep))
    return keep, counts


def _fit_null(t, y, sig, band_idx, n_b, spec):
    bnds = [tuple(spec.sigma_bounds_mag), tuple(spec.tau_bounds_d)]
    A0 = _offsets(band_idx, n_b)

    def nll(theta):
        p = np.exp(theta)
        try:
            gp = celerite2.GaussianProcess(_term(spec, p), mean=0.0)
            gp.compute(t, yerr=sig)
            Ciy = gp.apply_inverse(y)
            M0 = A0.T @ gp.apply_inverse(A0)
            b0 = np.linalg.solve(M0, A0.T @ Ciy)
            v = -gp.log_likelihood(y - A0 @ b0)
        except Exception:
            return 1e12
        return v if np.isfinite(v) else 1e12

    log_bounds = [(np.log(lo), np.log(hi)) for lo, hi in bnds]
    starts = list(spec.starts)[:spec.restarts]

    best, best_idx, n_ok = None, -1, 0
    for k, s0 in enumerate(starts):
        try:
            r = minimize(nll, np.log(np.asarray(s0, dtype=float)),
                         method="L-BFGS-B", bounds=log_bounds)
        except Exception:
            continue
        if not np.isfinite(r.fun) or r.fun >= 1e11:
            continue
        n_ok += 1
        if best is None or r.fun < best.fun:
            best, best_idx = r, k

    if best is None:
        p_hat = np.asarray(starts[0], dtype=float)
        return dict(params=p_hat, boundary_flag=1, n_restarts_used=len(starts),
                    n_restarts_ok=0, best_restart=-1, converged=0)

    p_hat = np.exp(best.x)
    tol = spec.boundary_tol_frac
    bflag = 0
    for val, (lo, hi) in zip(p_hat, bnds):
        if val <= lo * (1.0 + tol) or val >= hi * (1.0 - tol):
            bflag = 1
    return dict(params=p_hat, boundary_flag=bflag, n_restarts_used=len(starts),
                n_restarts_ok=n_ok, best_restart=best_idx + 1,
                converged=int(bool(best.success)))


def _scan(term, t, y, sig, band_idx, n_b, Pgrid, eta, t_ref, T_span, spec,
          return_surface=False):
    gp = celerite2.GaussianProcess(term, mean=0.0)
    gp.compute(t, yerr=sig)
    Ciy = gp.apply_inverse(y)

    A0 = _offsets(band_idx, n_b)
    M0 = A0.T @ gp.apply_inverse(A0)
    v0 = A0.T @ Ciy
    b0 = np.linalg.solve(M0, v0)
    l0 = float(gp.log_likelihood(y - A0 @ b0))
    q0 = float(v0 @ np.linalg.solve(M0, v0))

    ne = eta.size
    n_col = (2 * spec.n_harmonics) * n_b + n_b
    L = np.full((Pgrid.size, ne), -np.inf)
    x = t - t_ref
    if spec.require_positive_frequency:
        ok = np.all(1.0 + eta[:, None] * x[None, :] / T_span > 0.0, axis=1)
    else:
        ok = np.ones(ne, dtype=bool)
    n_negfreq = int((~ok).sum()) * Pgrid.size
    n_fail = 0

    for ip, P in enumerate(Pgrid):
        Ab = design_block(t, band_idx, n_b, P, eta, t_ref, T_span, spec.n_harmonics)
        Z = gp.apply_inverse(Ab)
        for j in range(ne):
            if not ok[j]:
                continue
            sl = slice(j * n_col, (j + 1) * n_col)
            Aj = Ab[:, sl]
            M = Aj.T @ Z[:, sl]
            v = Aj.T @ Ciy
            try:
                q = float(v @ np.linalg.solve(M, v))
            except np.linalg.LinAlgError:
                n_fail += 1
                continue
            if not np.isfinite(q):
                n_fail += 1
                continue
            L[ip, j] = 0.5 * (q - q0)

    j0 = int(np.argmin(np.abs(eta)))
    Lam_osc = float(np.nanmax(L))
    Lam_p0 = float(np.nanmax(L[:, j0]))
    ip, je = np.unravel_index(np.nanargmax(L), L.shape)
    out = dict(l0=l0, Lambda_osc=Lam_osc, Lambda_per0=Lam_p0,
               DeltaLambda_chirp=Lam_osc - Lam_p0,
               P_hat=float(Pgrid[ip]), eta_hat=float(eta[je]),
               argmax_ip=int(ip), argmax_je=int(je),
               n_node_fail=n_fail, n_node_negfreq=n_negfreq, n_linear_params=n_col)
    if return_surface:
        out["surface"] = L
    return out


def analyse(times, fluxes, errors, bands, spec, catflags=None, return_surface=False):
    """The single analysis path.  Nothing else may touch the algebra."""
    t0 = time.perf_counter()

    t = np.asarray(times, dtype=float)
    y = np.asarray(fluxes, dtype=float)
    sig = np.asarray(errors, dtype=float)
    b = np.asarray(bands)
    if not (t.size == y.size == sig.size == b.size):
        raise ValueError("times, fluxes, errors, bands must have equal length")
    # synthetic data carries catflags = 0 for every epoch, which is the point:
    # cut 1 is then a no-op there and a real cut on ZTF photometry.
    cf = np.zeros(t.size, dtype=int) if catflags is None else np.asarray(catflags)
    n_in = int(t.size)

    #Post-selection step: time ordering (celerite2 needs increasing t)
    if spec.sort_times:
        order = np.argsort(t, kind="stable")
        t, y, sig, b, cf = t[order], y[order], sig[order], b[order], cf[order]

    if spec.apply_quality_cuts:
        keep, qc = apply_quality_cuts(t, y, sig, b, cf, spec)
        t, y, sig, b, cf = t[keep], y[keep], sig[keep], b[keep], cf[keep]
    else:
        qc = dict(n_removed_catflags=0, n_removed_magerr=0, n_removed_mad=0,
                  n_removed_total=0)

    if spec.apply_quality_cuts and t.size < spec.min_epochs:
        return AnalysisResult(
            n_epochs=int(t.size), n_epochs_in=n_in,
            n_removed_catflags=qc["n_removed_catflags"],
            n_removed_magerr=qc["n_removed_magerr"],
            n_removed_mad=qc["n_removed_mad"],
            n_removed_total=qc["n_removed_total"],
            insufficient_data=1,
            n_bands=int(len(set(str(x) for x in b.tolist()))),
            runtime_s=time.perf_counter() - t0,
        )

    #Post-selection step: the band set, in sorted label order
    labels = sorted(set(str(x) for x in b.tolist()))
    n_b = len(labels)
    lut = {lab: j for j, lab in enumerate(labels)}
    band_idx = np.array([lut[str(x)] for x in b.tolist()], dtype=int)

    t_ref = 0.5 * (t.min() + t.max())
    T_span = float(t.max() - t.min())

    Pgrid, eta = build_grid(spec)

    fit = _fit_null(t, y, sig, band_idx, n_b, spec)
    term = _term(spec, fit["params"])
    res = _scan(term, t, y, sig, band_idx, n_b, Pgrid, eta, t_ref, T_span, spec,
                return_surface=return_surface)

    out = AnalysisResult(
        Lambda_osc=res["Lambda_osc"], Lambda_per0=res["Lambda_per0"],
        DeltaLambda_chirp=res["DeltaLambda_chirp"],
        P_hat=res["P_hat"], eta_hat=res["eta_hat"],
        sigma_hat=float(fit["params"][0]), tau_hat=float(fit["params"][1]),
        l0=res["l0"],
        boundary_flag=fit["boundary_flag"], converged=fit["converged"],
        n_restarts_used=fit["n_restarts_used"], n_restarts_ok=fit["n_restarts_ok"],
        best_restart=fit["best_restart"],
        n_epochs=int(t.size), n_epochs_in=n_in,
        n_removed_catflags=qc["n_removed_catflags"],
        n_removed_magerr=qc["n_removed_magerr"],
        n_removed_mad=qc["n_removed_mad"],
        n_removed_total=qc["n_removed_total"],
        insufficient_data=0,
        n_bands=int(n_b),
        n_linear_params=int(res["n_linear_params"]),
        n_grid_nodes=int(Pgrid.size * eta.size),
        n_node_fail=res["n_node_fail"], n_node_negfreq=res["n_node_negfreq"],
        argmax_ip=res["argmax_ip"], argmax_je=res["argmax_je"],
        runtime_s=time.perf_counter() - t0,
    )
    if return_surface:
        return out, res["surface"]
    return out


def run_on_data(times, fluxes, errors, bands, spec, catflags=None, **kw):
    """Real light curves.  No generation, no modification."""
    return analyse(times, fluxes, errors, bands, spec, catflags=catflags, **kw)


def run_on_null(gen, seed, spec, times=None, fluxes=None, errors=None, bands=None,
                catflags=None, **kw):
    """Generate pure background at the configuration's cadence, then analyse.

    `gen` is a Generator (see gen.py).  If an explicit light curve is supplied
    it is used verbatim: that path exists only for the stage-0 identity test.
    """
    if fluxes is None:
        times, fluxes, errors, bands = gen.null_lightcurve(seed)
    return analyse(times, fluxes, errors, bands, spec, catflags=catflags, **kw)


def run_on_injection(gen, seed, spec, injection, times=None, fluxes=None,
                     errors=None, bands=None, catflags=None, **kw):
    """Generate background, add a deterministic signal, then analyse.

    `injection` is None for the identity test (nothing is added), otherwise a
    dict of chirp parameters.
    """
    if fluxes is None:
        times, fluxes, errors, bands = gen.null_lightcurve(seed)
        if injection is not None:
            fluxes = fluxes + gen.chirp(times, injection)
    return analyse(times, fluxes, errors, bands, spec, catflags=catflags, **kw)
