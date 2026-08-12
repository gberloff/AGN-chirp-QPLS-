"""
Joint DRW + asymmetric-chirp model for AGN light curves.

Model (per band j):
    s_j(t) = a1_j sin(phi) + b1_j cos(phi) + a2_j sin(2 phi) + b2_j cos(2 phi) + mu_j
    phi(t) = 2 pi [ f0 * x + 0.5 * fdot * x^2 ],   x = t - t_ref
    eta    = fdot * T / f0                          (dimensionless chirp strength)

f0 and eta are shared across bands; mu_j and the four harmonic coefficients are
per band.  x is centred on t_ref = median of the *combined* time stamps so that
both bands share one phase convention.

Noise: DRW, k(dt) = sigma^2 exp(-|dt|/tau), realised with
celerite2.terms.RealTerm(a=sigma^2, c=1/tau).  Total covariance C = K + diag(sigma_i^2).
Each band carries an independent DRW realisation with shared hyperparameters, so
C is block diagonal by band.

Likelihood: at fixed nonlinear parameters (f0, eta, sigma, tau) the linear
coefficients beta are profiled out by generalised least squares,

    beta_hat = (X^T C^-1 X)^-1 X^T C^-1 y,

and the Gaussian log-likelihood of the residual y - X beta_hat is evaluated
under C.  C is never inverted: celerite2 performs O(N) Cholesky solves, and the
small (n_lin x n_lin) normal matrix is handled with a Cholesky solve plus a
jitter fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import celerite2
from celerite2 import terms
from scipy.linalg import cho_factor, cho_solve, LinAlgError

LOG2PI = np.log(2.0 * np.pi)

# Columns per band when the chirp is included: offset + 2 harmonics x (sin, cos)
N_LIN_CHIRP = 5
N_LIN_FLAT = 1


@dataclass
class BandData:
    """One photometric band: time stamps, magnitudes, per-point uncertainties."""

    name: str
    t: np.ndarray
    y: np.ndarray
    yerr: np.ndarray

    def __len__(self) -> int:
        return self.t.size


def eta_to_fdot(eta: float, f0: float, T: float) -> float:
    """eta = fdot * T / f0  ->  fdot."""
    return eta * f0 / T


def fdot_to_eta(fdot: float, f0: float, T: float) -> float:
    return fdot * T / f0


def phase(t: np.ndarray, t_ref: float, f0: float, eta: float, T: float) -> np.ndarray:
    """phi(t) = 2 pi [f0 x + 0.5 fdot x^2] with x = t - t_ref."""
    x = t - t_ref
    fdot = eta_to_fdot(eta, f0, T)
    return 2.0 * np.pi * (f0 * x + 0.5 * fdot * x * x)


def design_matrix(
    t: np.ndarray,
    t_ref: float,
    f0: float,
    eta: float,
    T: float,
    with_chirp: bool = True,
) -> np.ndarray:
    """Per-band design matrix.

    Column order (with_chirp=True): [1, sin phi, cos phi, sin 2phi, cos 2phi].
    With with_chirp=False only the constant offset column is returned, which is
    the "no chirp" null model.
    """
    n = t.size
    if not with_chirp:
        return np.ones((n, N_LIN_FLAT))
    phi = phase(t, t_ref, f0, eta, T)
    X = np.empty((n, N_LIN_CHIRP))
    X[:, 0] = 1.0
    X[:, 1] = np.sin(phi)
    X[:, 2] = np.cos(phi)
    X[:, 3] = np.sin(2.0 * phi)
    X[:, 4] = np.cos(2.0 * phi)
    return X


def _make_gp(t: np.ndarray, yerr: np.ndarray, sigma: float, tau: float,
             jitter: float = 0.0) -> celerite2.GaussianProcess:
    """celerite2 GP for C = sigma^2 exp(-|dt|/tau) + diag(yerr^2) (+ jitter)."""
    term = terms.RealTerm(a=sigma ** 2, c=1.0 / tau)
    gp = celerite2.GaussianProcess(term, mean=0.0)
    gp.compute(t, diag=yerr ** 2 + jitter)
    return gp


def _gp_with_jitter(t, yerr, sigma, tau):
    """Build the GP, escalating a small jitter if the factorisation fails."""
    base = float(np.median(yerr) ** 2)
    for jitter in (0.0, 1e-12, 1e-10 * base, 1e-8 * base, 1e-6 * base):
        try:
            return _make_gp(t, yerr, sigma, tau, jitter)
        except Exception:
            continue
    return None


def _logdet(gp: celerite2.GaussianProcess, n: int) -> float:
    """log|C| from celerite2: ll(0) = -0.5 (log|C| + n log 2pi)."""
    ll0 = gp.log_likelihood(np.zeros(n))
    return -2.0 * ll0 - n * LOG2PI


def _chol_spd(A: np.ndarray):
    """Cholesky factor of a symmetrised A, with a jitter ladder. None on failure."""
    A = 0.5 * (A + A.T)
    scale = float(np.mean(np.diag(A)))
    if not np.isfinite(scale) or scale <= 0:
        return None
    for jit in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        try:
            return cho_factor(A + jit * scale * np.eye(A.shape[0]), lower=True)
        except LinAlgError:
            continue
    return None


def band_gls(band: BandData, t_ref: float, f0: float, eta: float, T: float,
             sigma: float, tau: float, with_chirp: bool = True):
    """GLS solve for one band.

    Returns (loglike, beta_hat, A, chol_A) where A = X^T C^-1 X is the inverse
    of the conditional covariance of beta.  Returns None if the linear algebra
    fails for these parameters.
    """
    gp = _gp_with_jitter(band.t, band.yerr, sigma, tau)
    if gp is None:
        return None

    X = design_matrix(band.t, t_ref, f0, eta, T, with_chirp=with_chirp)
    n = band.t.size

    try:
        Ci_X = gp.apply_inverse(X)
        Ci_y = gp.apply_inverse(band.y[:, None])[:, 0]
        logdet = _logdet(gp, n)
    except Exception:
        return None

    A = X.T @ Ci_X
    A = 0.5 * (A + A.T)
    b = X.T @ Ci_y

    chol = _chol_spd(A)
    if chol is None:
        return None
    beta = cho_solve(chol, b)

    # (y - X beta)^T C^-1 (y - X beta) = y^T C^-1 y - b^T beta   (since A beta = b)
    chi2 = float(band.y @ Ci_y - b @ beta)
    if not np.isfinite(chi2) or not np.isfinite(logdet):
        return None

    loglike = -0.5 * (chi2 + logdet + n * LOG2PI)
    return loglike, beta, A, chol


def profile_loglike(bands, t_ref: float, f0: float, eta: float, T: float,
                    sigma: float, tau: float, with_chirp: bool = True) -> float:
    """Total profiled log-likelihood summed over bands (-inf on failure)."""
    total = 0.0
    for band in bands:
        out = band_gls(band, t_ref, f0, eta, T, sigma, tau, with_chirp=with_chirp)
        if out is None:
            return -np.inf
        total += out[0]
    return float(total) if np.isfinite(total) else -np.inf


def profile_solution(bands, t_ref, f0, eta, T, sigma, tau, with_chirp=True):
    """Per-band (loglike, beta_hat, A, chol_A); None if any band fails."""
    out = []
    for band in bands:
        res = band_gls(band, t_ref, f0, eta, T, sigma, tau, with_chirp=with_chirp)
        if res is None:
            return None
        out.append(res)
    return out


def is_spd(A: np.ndarray, sym_tol: float = 1e-8) -> bool:
    """Symmetric (to a relative tolerance) and positive definite."""
    if not np.all(np.isfinite(A)):
        return False
    scale = np.max(np.abs(A))
    if scale <= 0:
        return False
    if np.max(np.abs(A - A.T)) > sym_tol * scale:
        return False
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return False
    return np.all(np.linalg.eigvalsh(0.5 * (A + A.T)) > 0)


def draw_beta(beta_hat: np.ndarray, A: np.ndarray, rng: np.random.Generator):
    """Draw beta | theta, y ~ N(beta_hat, A^-1) with A = X^T C^-1 X.

    Uses A = L L^T and beta = beta_hat + L^-T z, whose covariance is
    L^-T L^-1 = (L L^T)^-1 = A^-1.  Returns None if A is not verified SPD.
    """
    A = 0.5 * (A + A.T)
    if not is_spd(A):
        return None
    L = np.linalg.cholesky(A)
    z = rng.standard_normal(A.shape[0])
    u = np.linalg.solve(L.T, z)
    return beta_hat + u


# Sampled vector: theta = (ln P, eta, ln sigma, ln tau).
# Uniform priors on ln P, ln sigma, ln tau  ==  log-uniform on P, sigma, tau.

PRIOR_BOUNDS = {
    "P": (50.0, 800.0),        # d, log-uniform
    "eta": (-0.6, 0.6),        # uniform
    "sigma": (0.005, 1.0),     # mag, log-uniform
    "tau": (10.0, 5000.0),     # d, log-uniform
}


def theta_bounds():
    """L-BFGS-B / emcee bounds in the sampled (log) coordinates."""
    lo_P, hi_P = PRIOR_BOUNDS["P"]
    lo_e, hi_e = PRIOR_BOUNDS["eta"]
    lo_s, hi_s = PRIOR_BOUNDS["sigma"]
    lo_t, hi_t = PRIOR_BOUNDS["tau"]
    return [
        (np.log(lo_P), np.log(hi_P)),
        (lo_e, hi_e),
        (np.log(lo_s), np.log(hi_s)),
        (np.log(lo_t), np.log(hi_t)),
    ]


def unpack_theta(theta):
    """(ln P, eta, ln sigma, ln tau) -> (f0, eta, sigma, tau)."""
    lnP, eta, lnsig, lntau = theta
    P = np.exp(lnP)
    return 1.0 / P, float(eta), np.exp(lnsig), np.exp(lntau)


def log_prior(theta) -> float:
    for value, (lo, hi) in zip(theta, theta_bounds()):
        if not (lo <= value <= hi):
            return -np.inf
    return 0.0


def log_prob(theta, bands, t_ref, T) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    f0, eta, sigma, tau = unpack_theta(theta)
    ll = profile_loglike(bands, t_ref, f0, eta, T, sigma, tau, with_chirp=True)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll
