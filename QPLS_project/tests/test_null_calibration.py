import numpy as np
from qpls.injection_recovery import ztf_cadence, simulate_drw, null_threshold, gls_peak_power


def test_drw_variance():
    t, _ = ztf_cadence(baseline_days=4000, dt=3.0)
    sf = 0.2
    x = simulate_drw(t, tau=200.0, sf_inf=sf, rng=np.random.default_rng(1))
    assert abs(np.var(x) - sf**2 / 2) < 0.5 * (sf**2 / 2)


def test_null_fap_selfconsistent():
    rng = np.random.default_rng(0)
    t, sigma = ztf_cadence()
    freq = np.linspace(1/900, 1/6, 800)
    thr, _ = null_threshold(t, sigma, 200.0, 0.2, freq, gls_peak_power,
                            fap=0.10, n_null=300, rng=rng)
    hits = 0
    N = 300
    for _ in range(N):
        mag = simulate_drw(t, 200.0, 0.2, rng) + rng.normal(0, sigma)
        pk, _ = gls_peak_power(t, mag, sigma, freq)
        hits += pk > thr
    assert 0.05 < hits / N < 0.16
