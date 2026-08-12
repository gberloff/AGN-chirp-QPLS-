"""
Joint DHO (celerite2 SHOTerm) + asymmetric-chirp detector.
Gleb, DON'T OVERCOMPLICATE THIS. 

Same deterministic chirp model and same GLS-profiled likelihood as
chirp_model.py; only the stochastic kernel changes, from

    RealTerm(a = sigma^2, c = 1/tau)          (DRW)
to
    SHOTerm(S0, w0, Q)                        (DHO)

with total covariance C = K + diag(sigma_i^2).  C is never inverted: celerite2
supplies O(N) Cholesky solves and the log-determinant.

PARAMETERISATION
The kernel is sampled as (sigma_kernel, w0, Q) rather than (S0, w0, Q), where

    sigma_kernel = sqrt(k(0)) = sqrt(S0 * w0 * Q)        (marginal RMS)
    =>  S0 = sigma_kernel^2 / (w0 * Q)

k(0) = S0*w0*Q holds exactly for celerite2's SHOTerm at every Q, including the
Q = 1/2 critical point (verified numerically over Q in [0.05, 50]).  This is an
exact reparameterisation of (S0, w0, Q) - no information is added or removed -
chosen for two reasons: the amplitude prior is then directly comparable to the
DRW detector's sigma prior, and the marginal RMS is far better conditioned than
S0, which is strongly degenerate with w0 and Q.

The stochastic timescale is reported as the damping time

    t_damp = 2Q / w0   [days]

Q is retained separately because t_damp alone hides the damping regime:
Q < 0.5 is overdamped (DRW-like), Q > 0.5 underdamped (quasi-periodic).
Q is NOT floored at 0.5 - the overdamped regime must be reachable so the DHO
can represent DRW-like noise.
"""

from __future__ import annotations

import numpy as np
import celerite2
from celerite2 import terms
from scipy.linalg import cho_solve

from chirp_model import LOG2PI, _chol_spd, design_matrix


# w0 is bounded through the oscillation period 2*pi/w0, which is the
# interpretable quantity: 20 d (well below the cadence-resolved scale) to
# 20000 d (10x the baseline).
OSC_PERIOD_BOUNDS = (20.0, 20000.0)

SHO_PRIOR_BOUNDS = {
    "P": (50.0, 800.0),                 # d, log-uniform
    "eta": (-0.6, 0.6),                 # uniform
    "sigma_kernel": (0.005, 1.0),       # mag, log-uniform (as the DRW sigma)
    "w0": (2.0 * np.pi / OSC_PERIOD_BOUNDS[1],
           2.0 * np.pi / OSC_PERIOD_BOUNDS[0]),   # rad/d, log-uniform
    "Q": (0.05, 50.0),                  # log-uniform, overdamped reachable
}


def theta_bounds_dho():
    b = SHO_PRIOR_BOUNDS
    return [
        (np.log(b["P"][0]), np.log(b["P"][1])),
        (b["eta"][0], b["eta"][1]),
        (np.log(b["sigma_kernel"][0]), np.log(b["sigma_kernel"][1])),
        (np.log(b["w0"][0]), np.log(b["w0"][1])),
        (np.log(b["Q"][0]), np.log(b["Q"][1])),
    ]


def unpack_theta_dho(theta):
    """(ln P, eta, ln sigma_k, ln w0, ln Q) -> (f0, eta, sigma_k, w0, Q)."""
    lnP, eta, lnsig, lnw0, lnQ = theta
    return (1.0 / np.exp(lnP), float(eta), np.exp(lnsig),
            np.exp(lnw0), np.exp(lnQ))


def t_damp(Q, w0):
    """DHO damping time, t_damp = 2Q/w0 [days]."""
    return 2.0 * np.asarray(Q) / np.asarray(w0)


def S0_from_amp(sigma_kernel, w0, Q):
    """S0 such that the SHOTerm marginal variance is sigma_kernel^2."""
    return sigma_kernel ** 2 / (w0 * Q)


def make_sho_term(sigma_kernel, w0, Q):
    return terms.SHOTerm(S0=S0_from_amp(sigma_kernel, w0, Q), w0=w0, Q=Q)


def _gp_sho(t, yerr, sigma_kernel, w0, Q):
    """SHOTerm GP with a jitter ladder; None if it cannot be factorised."""
    base = float(np.median(yerr) ** 2)
    for jitter in (0.0, 1e-12, 1e-10 * base, 1e-8 * base, 1e-6 * base):
        try:
            gp = celerite2.GaussianProcess(
                make_sho_term(sigma_kernel, w0, Q), mean=0.0)
            gp.compute(t, diag=yerr ** 2 + jitter)
            return gp
        except Exception:
            continue
    return None


def band_gls_sho(band, t_ref, f0, eta, T, sigma_kernel, w0, Q,
                 with_chirp: bool = True):
    """GLS solve for one band under the SHOTerm kernel.

    Returns (loglike, beta_hat, A, chol_A) with A = X^T C^-1 X, or None if the
    linear algebra fails at these parameters.
    """
    gp = _gp_sho(band.t, band.yerr, sigma_kernel, w0, Q)
    if gp is None:
        return None

    X = design_matrix(band.t, t_ref, f0, eta, T, with_chirp=with_chirp)
    n = band.t.size

    try:
        Ci_X = gp.apply_inverse(X)
        Ci_y = gp.apply_inverse(band.y[:, None])[:, 0]
        logdet = -2.0 * gp.log_likelihood(np.zeros(n)) - n * LOG2PI
    except Exception:
        return None

    A = X.T @ Ci_X
    A = 0.5 * (A + A.T)
    b = X.T @ Ci_y

    chol = _chol_spd(A)
    if chol is None:
        return None
    beta = cho_solve(chol, b)

    chi2 = float(band.y @ Ci_y - b @ beta)
    if not np.isfinite(chi2) or not np.isfinite(logdet):
        return None
    return -0.5 * (chi2 + logdet + n * LOG2PI), beta, A, chol


def profile_loglike_sho(bands, t_ref, f0, eta, T, sigma_kernel, w0, Q,
                        with_chirp: bool = True) -> float:
    total = 0.0
    for band in bands:
        out = band_gls_sho(band, t_ref, f0, eta, T, sigma_kernel, w0, Q,
                           with_chirp=with_chirp)
        if out is None:
            return -np.inf
        total += out[0]
    return float(total) if np.isfinite(total) else -np.inf


def profile_solution_sho(bands, t_ref, f0, eta, T, sigma_kernel, w0, Q,
                         with_chirp: bool = True):
    out = []
    for band in bands:
        res = band_gls_sho(band, t_ref, f0, eta, T, sigma_kernel, w0, Q,
                           with_chirp=with_chirp)
        if res is None:
            return None
        out.append(res)
    return out


def log_prior_dho(theta) -> float:
    for value, (lo, hi) in zip(theta, theta_bounds_dho()):
        if not (lo <= value <= hi):
            return -np.inf
    return 0.0


def log_prob_dho(theta, bands, t_ref, T) -> float:
    lp = log_prior_dho(theta)
    if not np.isfinite(lp):
        return -np.inf
    f0, eta, sigma_k, w0, Q = unpack_theta_dho(theta)
    ll = profile_loglike_sho(bands, t_ref, f0, eta, T, sigma_k, w0, Q,
                             with_chirp=True)
    return -np.inf if not np.isfinite(ll) else lp + ll
