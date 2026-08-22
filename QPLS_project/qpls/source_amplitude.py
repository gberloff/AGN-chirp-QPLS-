
import argparse

import numpy as np

M_BOL_SUN = 4.74
PC = 1.0


MU_CUSP_PEAK_NORM = 1.8e6

FIDUCIAL = dict(
    M10=1.0,
    D_kpc=1.0,
    T_yr=1.0,
    one_mp_e=1.0,
)

D_L_PC = 2.9e9


def mu_cusp(R10, f=FIDUCIAL):
    return (1e6 * f["one_mp_e"] ** -0.7 * f["M10"] ** 0.44
            * f["T_yr"] ** -0.47 * R10 ** -0.64 * f["D_kpc"] ** 0.67)


def mu_fold(R10, f=FIDUCIAL):
    return (4e5 * f["one_mp_e"] ** -1.0 * f["M10"] ** (5 / 12)
            * f["T_yr"] ** (-2 / 3) * R10 ** -0.5 * f["D_kpc"] ** 0.75)


def einstein_radius_pc(f=FIDUCIAL):
    return 1.38 * f["M10"] ** 0.5 * f["D_kpc"] ** 0.5


def caustic_diamond_pc(f=FIDUCIAL):
    d = 0.0075 * f["T_yr"] ** (2 / 3) * f["M10"] ** (-1 / 6) * f["D_kpc"] ** -0.5
    return einstein_radius_pc(f) * d ** 2, d


def source_apparent_mag(L_star_Lsun, distance_pc, A_V):
    M_bol = M_BOL_SUN - 2.5 * np.log10(L_star_Lsun)
    dist_mod = 5.0 * np.log10(distance_pc / 10.0)
    return M_bol + dist_mod + A_V, M_bol, dist_mod


def observed_amplitude(L_star_Lsun, R_star_Rsun, distance_pc, m_n, A_V,
                       f=FIDUCIAL):
    R10 = R_star_Rsun / 10.0
    m_s_unlensed, M_bol, dist_mod = source_apparent_mag(L_star_Lsun, distance_pc, A_V)

    mc = mu_cusp(R10, f)
    mf = mu_fold(R10, f)
    mu_peak = max(mc, mf)
    which = "cusp" if mc >= mf else "fold"

    m_s_lensed = m_s_unlensed - 2.5 * np.log10(mu_peak)
    flux_ratio = 10.0 ** (-0.4 * (m_s_lensed - m_n))
    delta_m = 2.5 * np.log10(1.0 + flux_ratio)

    caustic_pc, d_ratio = caustic_diamond_pc(f)

    return dict(
        L=L_star_Lsun, R=R_star_Rsun, A_V=A_V, m_n=m_n, R10=R10,
        M_bol=M_bol, dist_mod=dist_mod, m_s_unlensed=m_s_unlensed,
        mu_cusp=mc, mu_fold=mf, mu_peak=mu_peak, mu_which=which,
        m_s_lensed=m_s_lensed, flux_ratio=flux_ratio, delta_m=delta_m,
        xi0_pc=einstein_radius_pc(f), caustic_diamond_pc=caustic_pc, d_ratio=d_ratio,
        crossing_duration="UNVERIFIED",
    )


def mu_cusp_peak(mtotal_Msun, R_Rsun):
    return MU_CUSP_PEAK_NORM * (mtotal_Msun / 2.0e10) ** 0.44 * (R_Rsun / 10.0) ** -0.64


def luminosity_distance_pc_from_z(z):
    from astropy.cosmology import Planck18
    return Planck18.luminosity_distance(z).to("pc").value


def high_z_source(distance_pc, mtotal_Msun, L_Lsun, R_Rsun, period_yr, z):
    dist_mod = 5.0 * np.log10(distance_pc / 10.0)
    mu = mu_cusp_peak(mtotal_Msun, R_Rsun)
    m_unlensed = M_BOL_SUN - 2.5 * np.log10(L_Lsun) + dist_mod
    m_lensed = m_unlensed - 2.5 * np.log10(mu)
    P_observed = (1.0 + z) * period_yr
    return dict(dist_mod=dist_mod, mu_cusp_peak=mu, m_unlensed=m_unlensed,
                m_lensed=m_lensed, P_observed=P_observed)


CASES = [(1.0, 1.0), (2.0e3, 100.0), (1.0e5, 20.0)]


def build_parser():
    p = argparse.ArgumentParser(
        description="QPLS source -> observed amplitude calculator.")
    p.add_argument("--z", type=float, default=None,
                   help="source/host redshift; if given, D_L is computed from "
                        "astropy Planck18 and OVERRIDES --dl")
    p.add_argument("--dl", type=float, default=2.90,
                   help="luminosity distance in Gpc (default: 2.90)")
    p.add_argument("--mtotal", type=float, default=2.0e10,
                   help="total binary mass in Msun (default: 2e10, paper fiducial)")
    p.add_argument("--lum", type=float, default=1.0e5,
                   help="source luminosity in Lsun (default: 1e5)")
    p.add_argument("--radius", type=float, default=20.0,
                   help="source radius in Rsun (default: 20)")
    p.add_argument("--period", type=float, default=1.0,
                   help="intrinsic binary orbital period in years (default: 1.0)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    D_L_pc = args.dl * 1.0e9
    z = 0.0 if args.z is None else args.z
    if args.z is not None:
        D_L_pc = luminosity_distance_pc_from_z(args.z)
        print(f"NOTE: --z {args.z} given -> D_L computed from astropy Planck18 "
              f"({D_L_pc:.4e} pc = {D_L_pc / 1e9:.3f} Gpc); "
              f"OVERRIDES --dl {args.dl} Gpc.")
        print()

    print("=" * 90)
    print("QPLS source -> observed amplitude   (Task 12)")
    print("=" * 90)
    print("Fiducial lens/orbit (system A; docs/physics_reference.md Eqs 1,3-4):")
    for k, v in FIDUCIAL.items():
        print(f"    {k:10s} = {v}")
    print(f"    D_L (luminosity distance) = {D_L_pc:.3e} pc   [ASSUMED / stated input]")
    print(f"    m_n (nucleus apparent mag) = 19.0")
    print(f"    xi0 (Einstein radius, Eq.1) = {einstein_radius_pc():.4f} pc")
    cpc, d = caustic_diamond_pc()
    print(f"    caustic diamond diameter ~ xi0*d^2 = {cpc:.4e} pc   (d = a/xi0 = {d:.4f})")
    print("    caustic-crossing DURATION: UNVERIFIED (no velocity/duration relation in repo)")
    print()

    m_n = 19.0
    hdr = ("L[Lsun]  R[Rsun] A_V   M_bol   distmod  m_s(unlens)  "
           "mu_cusp    mu_fold    mu_peak(which) m_s(lensed)  Fs/Fn      delta_m   t_cross")
    print(hdr)
    print("-" * len(hdr))
    for (L, R) in CASES:
        for A_V in (0.0, 2.0):
            r = observed_amplitude(L, R, D_L_pc, m_n, A_V)
            print(f"{L:8g} {R:6g} {A_V:3g}  {r['M_bol']:6.2f}  {r['dist_mod']:6.2f}  "
                  f"{r['m_s_unlensed']:9.2f}   {r['mu_cusp']:.3e} {r['mu_fold']:.3e} "
                  f"{r['mu_peak']:.3e}({r['mu_which']:4s}) {r['m_s_lensed']:9.2f}  "
                  f"{r['flux_ratio']:.3e}  {r['delta_m']:7.4f}  {r['crossing_duration']}")
    print()
    print("delta_m = 2.5*log10(1 + 10**(-0.4*(m_s_lensed - m_n))), m_n=19.")
    print("Tags: L,R,A_V,m_n = inputs; M_bol,m_s,mu,delta_m = DERIVED; lens/orbit fiducials")
    print("as tagged in FIDUCIAL (M10,D_kpc READ; T_yr,(1-/+e) ASSUMED).")

    hz = high_z_source(D_L_pc, args.mtotal, args.lum, args.radius, args.period, z)
    print()
    print("=" * 90)
    print("Single-source scaled calculation (mass / luminosity / radius / redshift)")
    print("=" * 90)
    print("SIMPLIFICATION (stated): bolometric magnitudes, NO K-correction, no bandpass")
    print("   or bolometric correction applied.")
    print("SIMPLIFICATION (stated): mu_cusp below is a FAVOURABLE-CASE PEAK, eccentricity")
    print("   and D_LS are held at the paper fiducial; it is not a typical magnification.")
    print(f"NOTE: unified constant, this block and the table above both use")
    print(f"   M_bol,sun = {M_BOL_SUN} (IAU 2015 bolometric).")
    print()
    if args.z is not None:
        print(f"    z (redshift)                = {args.z:.4g}")
        print(f"    D_L (from Planck18)         = {D_L_pc:.4e} pc  ({D_L_pc / 1e9:.3f} Gpc)")
    else:
        print(f"    z (redshift)                = not given (treated as 0)")
        print(f"    D_L (stated input)          = {D_L_pc:.4e} pc  ({D_L_pc / 1e9:.3f} Gpc)")
    print(f"    M_total (binary)            = {args.mtotal:.4g} Msun")
    print(f"    L_source                    = {args.lum:.4g} Lsun")
    print(f"    R_source                    = {args.radius:.4g} Rsun")
    print(f"    P_intrinsic                 = {args.period:.4g} yr")
    print(f"    distance modulus            = {hz['dist_mod']:.4f} mag")
    print(f"    mu_cusp (favourable peak)   = {hz['mu_cusp_peak']:.4e}")
    print(f"    m_unlensed                  = {hz['m_unlensed']:.4f} mag")
    print(f"    m_lensed                    = {hz['m_lensed']:.4f} mag")
    print(f"    P_observed                  = {hz['P_observed']:.4f} yr")


if __name__ == "__main__":
    main()
