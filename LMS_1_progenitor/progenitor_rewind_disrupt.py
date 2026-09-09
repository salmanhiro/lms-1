#!/usr/bin/env python
"""
LMS-1 progenitor rewind + tidal disruption with two embedded globular clusters.

Scientific question

Yuan et al. (2020) and Malhan et al. (2021) associate NGC 5024 (M53) and
NGC 5053 with the LMS-1 / Wukong accretion event, but their orbital models
contain *no dynamical friction*.  Here we ask: if both clusters were born
inside the LMS-1 dwarf 4 Gyr ago, does dynamical friction sink them into the
LMS-1 core and keep them there while the MW tidally disrupts the dwarf into a
stream?  Or do they end up scattered/stripped from the disrupting core?

Design (modifies 7_dynfric_stream)

  LMS-1 progenitor orbit  ->  centroid of the NGC 5024 + NGC 5053 phase space
                              today, rewound 4 Gyr in the MW potential.  The
                              rewind is done WITH and WITHOUT Chandrasekhar
                              friction of the 10^9 Msun halo against the MW
                              (exactly as folder 7).  This trajectory is the
                              moving centre of the LMS-1 NFW halo.

  Two GCs (NGC 5024, NGC 5053 masses)  ->  planted INSIDE the LMS-1 halo at
                              t = -4 Gyr on bound internal orbits, then carried
                              forward to today.  They are live point masses
                              (softened Plummer) that
                                * feel MW + moving LMS-1 NFW + each other,
                                * in the "DF" run also feel Chandrasekhar
                                  friction against the LMS-1 dark halo, which
                                  is what Yuan+2020 / Malhan+2021 omit,
                                * perturb the LMS-1 stellar stream particles.

  LMS-1 stellar body  ->  N_STREAM massless Plummer tracers sampled inside the
                              LMS-1 NFW, disrupted by MW tides into a stream
                              (as in folder 7) and additionally stirred by the
                              two embedded GCs.

The two scenarios share everything except dynamical friction:
  * "With DF"  : haloMW friction on the orbit  +  GChalo friction.
  * "No DF"    : Yuan/Malhan-like, no friction anywhere.

Outputs

All files of a run are written to the pair of scenario folders

    mass_<M_200>_df/      e.g. mass_1e10_df/
    mass_<M_200>_nodf/    e.g. mass_1e10_nodf/

so that runs at different LMS-1 subhalo masses never overwrite each other.
Snapshot arrays go to the folder of their own scenario; the combined
(DF | no-DF) figures, animations and summary are written to both.

  lms1gc_{xy,xz,yz}.png   2 rows (DF / no-DF) x 5 epochs: stream + GC tracks
  lms1gc_{xy,xz,yz}.mp4   side-by-side animations (DF left . no-DF right)
  lms1gc_core_retention.png   headline: GC distance from LMS-1 core vs time,
                              core (bound half-mass) radius band, bound fraction
  lms1gc_gc_separation.png    GCGC separation + GC galactocentric radius
  lms1gc_radec.png        RA/Dec sky map at t = 0
  lms1gc_energy_L.png     E vs L of the stream + GCs at t = 0
  snap_*.npy              particle / GC snapshot arrays
  results_summary.txt     core-retention verdict
"""

import os
import shutil
import numpy as np
import scipy.special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import agama
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u

agama.setUnits(length=1, velocity=1, mass=1)

T_BACKWARD = 5.0          # Gyr

# Cosmology form Dutton & Maccio 2014 c-M
LITTLE_H   = 0.70
OMEGA_M    = 0.30
OMEGA_L    = 0.70
Z_HALO     = 1.0          # ~7-8 Gyr ago
RHO_CRIT_0 = 277.5 * LITTLE_H**2     # M_sun/kpc^3 at z=0


def E2(z, Om=OMEGA_M, Ol=OMEGA_L):
    return Om * (1.0 + z)**3 + Ol


def rho_crit(z):
    return RHO_CRIT_0 * E2(z)


def dutton_maccio_c200(M_200, z=Z_HALO, h=LITTLE_H):
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    b = -0.101 + 0.026 * z
    return 10.0**(a + b * np.log10(M_200 * h / 1e12))


def nfw_params_from_M200(M_200, z=Z_HALO):
    c      = dutton_maccio_c200(M_200, z=z)
    r200   = (3.0 * M_200 / (4.0 * np.pi * 200.0 * rho_crit(z)))**(1.0/3.0)
    rs     = r200 / c
    f_c    = np.log(1.0 + c) - c / (1.0 + c)
    M_char = M_200 / f_c
    return c, r200, rs, M_char


# LMS-1 NFW halo
# 1e10 (vs folder 7's 1e9): a deeper, denser well -> faster GC dynamical-friction
# inspiral AND stronger binding against MW tides, so BOTH clusters can be kept in
# the core (co-retention test) rather than only the massive one.
M_HALO = 1.0e10
C_NFW, R200, RS_NFW, M_NFW_CHAR = nfw_params_from_M200(M_HALO, z=Z_HALO)


# Output folders: one per (subhalo mass, DF setting), e.g. mass_1e10_df/
def _mass_tag(m):
    """1e10 -> '1e10', 3.2e9 -> '3p2e9' (filesystem-safe mantissa)."""
    e = int(np.floor(np.log10(m) + 1e-9))
    c = m / 10.0**e
    c_s = (f"{int(round(c))}" if abs(c - round(c)) < 1e-6
           else f"{c:.2f}".rstrip("0").rstrip(".").replace(".", "p"))
    return f"{c_s}e{e}"


OUT_ROOT = os.path.dirname(os.path.abspath(__file__))
MASS_TAG = _mass_tag(M_HALO)
OUT_DIRS = {tag: os.path.join(OUT_ROOT, f"mass_{MASS_TAG}_{tag}")
            for tag in ("df", "nodf")}
for _d in OUT_DIRS.values():
    os.makedirs(_d, exist_ok=True)


def out_path(fname, tag):
    """Path of an output file in the 'df' or 'nodf' folder of this mass."""
    return os.path.join(OUT_DIRS[tag], fname)


def save_fig_both(fig, fname, **kw):
    """Combined DF/no-DF figure -> written into BOTH scenario folders."""
    p = out_path(fname, "df")
    fig.savefig(p, **kw)
    shutil.copy2(p, out_path(fname, "nodf"))
    return p


def _both_msg(fname):
    return f"{fname}  ->  mass_{MASS_TAG}_{{df,nodf}}/"

# LMS-1 stellar body (stream particles)
N_STREAM     = 10_000     # test particles (massless in integration)
M_STARS_LMS1 = 1.0e5      # Msun   for velocity sampling only
B_LMS1       = 2.0        # kpc    Plummer scale radius of LMS-1 progenitor

# Globular clusters planted inside LMS-1 at t = -T_BACKWARD
# Masses: Harris (2010) / Baumgardt stellar masses, as used in folder 2.
# r0 : initial galactocentric radius INSIDE the LMS-1 halo (kpc)
# Plummer softening b_gc keeps them as compact, near-point masses.
B_GC = 0.010              # kpc (10 pc) Plummer scale of each GC
# Co-retention test: both planted NEAR the core on coplanar (xy) prograde orbits
# (small r0), so DF only has to resist tidal stripping rather than drive a long
# inspiral  the "deposited centrally as a pair" picture for NGC 5024 / 5053.
GCS = [
    dict(name="NGC 5024", mass=5.02e5, b=B_GC, r0=0.25, color="#1f4e8c",
         ls_df="-", ls_nodf="", marker="o",
         e_r=np.array([1.0, 0.0, 0.0]), e_v=np.array([0.0,  1.0, 0.0])),
    dict(name="NGC 5053", mass=6.28e4, b=B_GC, r0=0.40, color="#d1751f",
         ls_df="-.", ls_nodf=":", marker="s",
         e_r=np.array([-1.0, 0.0, 0.0]), e_v=np.array([0.0, -1.0, 0.0])),
]
N_GC = len(GCS)

# Chandrasekhar DF parameters-
# haloMW friction (folder 7 values)
LN_LAMBDA_HALO = 4.6
# GChalo friction:  ln(Lambda) ~ ln(M_halo / M_gc), floored.
V_FLOOR_DF   = 1.0         # km/s
SIG_FLOOR_DF = 5.0         # km/s

# Leapfrog timestep
TAU = 2.0**-11             # Gyr (~0.488 Myr)

# Snapshot epochs (tied to T_BACKWARD so the first panel = planting time) 
T_SNAPS = np.arange(-round(T_BACKWARD), 0.5, 1.0)   # Gyr, e.g. [-5..0]

# Galactic frame consistent with MWPotential2014 (Bovy 2015)
_gc_frame_mw14 = Galactocentric(
    galcen_distance=8.0*u.kpc,
    galcen_v_sun=[11.1, 220.0+12.24, 7.25]*u.km/u.s,
    z_sun=25.0*u.pc)


def observed_ic(ra, dec, dist, pmra, pmdec, vlos):
    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.kpc,
                  pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr,
                  radial_velocity=vlos*u.km/u.s, frame='icrs')
    g = sc.transform_to(_gc_frame_mw14)
    return np.array([g.x.to(u.kpc).value, g.y.to(u.kpc).value, g.z.to(u.kpc).value,
                     g.v_x.to(u.km/u.s).value, g.v_y.to(u.km/u.s).value,
                     g.v_z.to(u.km/u.s).value])


# NGC 5024 (Baumgardt&Hilker 2018 dist; Vasiliev&Baumgardt 2021 PM; Harris vlos)
NGC5024_IC = observed_ic(198.230, 18.168, 18.13, -0.133, -0.095, -62.9)
# NGC 5053 (Baumgardt&Hilker 2018; Vasiliev&Baumgardt 2021; Harris vlos)
NGC5053_IC = observed_ic(199.113, 17.698, 17.54, -0.330, -0.158, 44.0)

# present-day GC phase space -> rewound backward to set the planting IC, so the
# forward integration returns each GC to its OBSERVED position today.
GC_OBS_IC = np.array([NGC5024_IC, NGC5053_IC])

# LMS-1 progenitor orbit = MASS-WEIGHTED mean phase space of its members.
# A plain (unweighted) average cancels the clusters' opposite line-of-sight and
# vertical velocities (NGC 5024: v_los=-63, v_z=-55; NGC 5053: v_los=+44,
# v_z=+51), leaving a nearly circular orbit that never falls in from large
# radius.  Mass-weighting tracks the dominant cluster NGC 5024 (~8x heavier),
# keeping the dwarf on its physical, eccentric infall orbit (folder-7 like).
_w24, _w53 = GCS[0]["mass"], GCS[1]["mass"]
LMS1_IC = (_w24 * NGC5024_IC + _w53 * NGC5053_IC) / (_w24 + _w53)

# observed cluster sky positions today (for the RA/Dec panel)
NGC5024_RA_DEG, NGC5024_DEC_DEG = 198.230, 18.168
NGC5053_RA_DEG, NGC5053_DEC_DEG = 199.113, 17.698

print("=" * 70)
print("LMS-1 progenitor rewind + disruption with two embedded GCs")
print("  Aim: add the dynamical friction that Yuan+2020 / Malhan+2021 omit")
print("-" * 70)
print(f"  LMS-1 NFW : M_200 = {M_HALO:.2e} Msun  c = {C_NFW:.2f}"
      f"  r_200 = {R200:.2f} kpc  r_s = {RS_NFW*1e3:.0f} pc")
print(f"  Stream    : N = {N_STREAM}  M_prog = {M_STARS_LMS1:.2e} Msun"
      f"  b = {B_LMS1:.1f} kpc")
for g in GCS:
    print(f"  GC        : {g['name']:9s}  M = {g['mass']:.2e} Msun"
          f"  r0 = {g['r0']:.2f} kpc inside halo")
print(f"  Timestep  : tau = {TAU*1e3:.4f} Myr  T = {T_BACKWARD:.1f} Gyr")
print(f"  Output    : mass_{MASS_TAG}_df/  and  mass_{MASS_TAG}_nodf/")
print("=" * 70)

# Potentials
pot_host  = agama.Potential("MWPotential2014.ini")
pot_nfw   = agama.Potential(type="NFW", mass=M_NFW_CHAR, scaleRadius=RS_NFW)
pot_stars = agama.Potential(type="Plummer", mass=M_STARS_LMS1, scaleRadius=B_LMS1)
pot_lms1  = agama.Potential(pot_stars, pot_nfw)         # internal LMS-1 potential

# GC point-mass (softened) potentials  translated each step
pot_gc = [agama.Potential(type="Plummer", mass=g["mass"], scaleRadius=g["b"])
          for g in GCS]


# LMS-1 internal circular velocity / velocity dispersion-
# sigma(r) ~ v_circ(r)/sqrt(2) used both for GC initial conditions and for the
# Chandrasekhar formula inside the halo.
_r_grid = np.geomspace(1e-3, 3.0 * R200, 400)
_acc    = pot_lms1.force(np.column_stack([_r_grid, 0*_r_grid, 0*_r_grid]))
_vcirc  = np.sqrt(np.abs(_acc[:, 0]) * _r_grid)
_vcirc_of_r = interp1d(_r_grid, _vcirc, bounds_error=False,
                       fill_value=(_vcirc[0], _vcirc[-1]))


def v_circ_lms1(r):
    return float(_vcirc_of_r(np.clip(r, _r_grid[0], _r_grid[-1])))


def sigma_lms1(r):
    return max(v_circ_lms1(r) / np.sqrt(2.0), SIG_FLOOR_DF)


# LMS-1 subhalo tidal (Jacobi) radius in the MW field-
def mw_mass_enclosed(R):
    """Spherical enclosed MW mass from the circular speed:  M(<R)=v_c^2 R/G."""
    a = np.asarray(pot_host.force([[R, 0.0, 0.0]])[0], float)
    return abs(a[0]) * R * R / agama.G


def jacobi_radius(R):
    """Tidal radius of the LMS-1 NFW (enclosed-mass form, solved iteratively):
        r_t = R (M_NFW(<r_t) / 3 M_MW(<R))^(1/3)."""
    R = max(float(R), 1e-3)
    Mmw = mw_mass_enclosed(R)
    rt = 1.0
    for _ in range(60):
        Mn = M_NFW_CHAR * (np.log(1.0 + rt/RS_NFW) - (rt/RS_NFW)/(1.0 + rt/RS_NFW))
        rt = R * (Mn / (3.0 * Mmw))**(1.0/3.0)
    return rt



# Chandrasekhar dynamical friction


def chandrasekhar_acc(vel_rel, M_sat, ln_Lambda, rho, sig):
    """Drag acceleration given relative velocity, drag mass, local density,
    and local 1-D velocity dispersion."""
    vel_rel = np.asarray(vel_rel, float).reshape(3)
    v = float(np.sqrt(np.dot(vel_rel, vel_rel)))
    if v < V_FLOOR_DF or rho <= 0.0:
        return np.zeros(3)
    X   = v / (np.sqrt(2.0) * sig)
    fac = scipy.special.erf(X) - 2.0 * X / np.sqrt(np.pi) * np.exp(-X * X)
    if fac <= 0.0:
        return np.zeros(3)
    prefac = -4.0 * np.pi * agama.G**2 * M_sat * ln_Lambda * rho * fac / v**3
    return prefac * vel_rel


def halo_df_acc(pos, vel, M_sat, ln_Lambda):
    """Friction of the LMS-1 halo (M_sat) against the MW field (folder 7)."""
    p2d = np.asarray(pos, float).reshape(1, 3)
    rho = float(pot_host.density(p2d)[0])
    Phi = float(pot_host.potential(p2d)[0])
    sig = np.sqrt(max(-Phi / 2.0, 10.0**2))
    return chandrasekhar_acc(vel, M_sat, ln_Lambda, rho, sig)



# Rewind the LMS-1 progenitor orbit  ->  moving NFW centre


def rewind_orbit(ic_today, t_backward, apply_df=False):
    """KDK leapfrog backward in MW [+ haloMW DF].
    Returns (times, traj) sorted ascending in time (past -> today)."""
    nsteps = int(round(t_backward / TAU))
    tau    = -t_backward / nsteps      # negative = backward
    state  = np.asarray(ic_today, float).copy()
    times  = [0.0]
    traj   = [state.copy()]

    def accel(pos, vel):
        a = np.asarray(pot_host.force(pos.reshape(1, 3))[0], float)
        if apply_df:
            a = a + halo_df_acc(pos, vel, M_HALO, LN_LAMBDA_HALO)
        return a

    acc = accel(state[:3], state[3:])
    for k in range(nsteps):
        state[3:] += acc * (tau / 2)
        state[:3] += state[3:] * tau
        acc        = accel(state[:3], state[3:])
        state[3:] += acc * (tau / 2)
        times.append((k + 1) * tau)
        traj.append(state.copy())

    times = np.asarray(times, float)
    traj  = np.stack(traj, axis=0)
    order = np.argsort(times)
    return times[order], traj[order]


def make_center_funcs(ta, xva):
    """Interpolators t -> (centre position, centre velocity) of the halo."""
    itp_p = interp1d(ta, xva[:, :3], axis=0,
                     bounds_error=False, fill_value="extrapolate")
    itp_v = interp1d(ta, xva[:, 3:], axis=0,
                     bounds_error=False, fill_value="extrapolate")
    return (lambda t: np.asarray(itp_p(t), float),
            lambda t: np.asarray(itp_v(t), float))


print("\n Rewind LMS-1 progenitor (DF on / off)")
ta_DF,   xva_DF   = rewind_orbit(LMS1_IC, T_BACKWARD, apply_df=True)
ta_noDF, xva_noDF = rewind_orbit(LMS1_IC, T_BACKWARD, apply_df=False)
print(f"  DF   past COM : {xva_DF[0, :3].round(3)} kpc  v = {xva_DF[0, 3:].round(2)} km/s")
print(f"  noDF past COM : {xva_noDF[0, :3].round(3)} kpc  v = {xva_noDF[0, 3:].round(2)} km/s")
_d = np.linalg.norm(xva_DF[0, :3] - xva_noDF[0, :3])
print(f"  past-IC shift (DF vs no-DF): |delta| = {_d:.3f} kpc")

cen_p_DF,   cen_v_DF   = make_center_funcs(ta_DF,   xva_DF)
cen_p_noDF, cen_v_noDF = make_center_funcs(ta_noDF, xva_noDF)



# Sample LMS-1 stream particles + plant GCs at t = -4 Gyr


def sample_stream_at(ic_past):
    """Sample N_STREAM particles from the LMS-1 Plummer DF inside the NFW,
    translate the CoM to ic_past (position + velocity)."""
    df = agama.DistributionFunction(type="quasispherical",
                                    density=pot_stars, potential=pot_lms1)
    gm = agama.GalaxyModel(pot_lms1, df)
    xv, _ = gm.sample(N_STREAM)
    xv = np.asarray(xv, float).copy()
    xv += ic_past          # translate CoM (6-vector) to past phase-space point
    return xv


def plant_gcs_at(ic_past):
    """Place the GCs inside the LMS-1 halo on bound internal orbits at -4 Gyr.
    GC velocity = halo bulk velocity + circular velocity at r0."""
    P0, V0 = ic_past[:3], ic_past[3:]
    xv = np.zeros((N_GC, 6))
    for i, g in enumerate(GCS):
        e_r = g["e_r"] / np.linalg.norm(g["e_r"])
        e_v = g["e_v"] / np.linalg.norm(g["e_v"])
        vc  = v_circ_lms1(g["r0"])
        xv[i, :3] = P0 + g["r0"] * e_r
        xv[i, 3:] = V0 + vc * e_v
    return xv


print(f"\n Sampling stream (N = {N_STREAM})")
agama.setRandomSeed(42)
xv_stream_DF = sample_stream_at(xva_DF[0])
agama.setRandomSeed(42)
xv_stream_noDF = sample_stream_at(xva_noDF[0])
# GC planting ICs are set below (rewound from the observed positions) once
# accel_gc and the halo centre interpolators are available.



# Snapshot bookkeeping

_nsteps = int(round(T_BACKWARD / TAU))
_SNAP_GYR = [max(0, min(int(round((ts + T_BACKWARD) / TAU)), _nsteps))
             for ts in T_SNAPS]
N_ANIM = 100
_SNAP_ANIM = np.round(np.linspace(0, _nsteps, N_ANIM, endpoint=True)).astype(int)
_ALL_STEPS = sorted(set(_SNAP_GYR) | set(_SNAP_ANIM.tolist()))



# Forward integration:  stream tracers + live GCs (with optional GChalo DF)


def accel_stream(xs, ctr, gc_pos):
    """MW + moving NFW + both GCs, on the (massless) stream particles."""
    a = np.asarray(pot_host.force(xs), float)
    a += np.asarray(pot_nfw.force(xs - ctr), float)
    for i in range(N_GC):
        a += np.asarray(pot_gc[i].force(xs - gc_pos[i]), float)
    return a


def accel_gc(xv_gc, ctr, vctr, apply_df):
    """MW + moving NFW + mutual GC gravity [+ Chandrasekhar friction in halo]."""
    pos = xv_gc[:, :3]
    a = np.asarray(pot_host.force(pos), float)
    a += np.asarray(pot_nfw.force(pos - ctr), float)
    # mutual GCGC gravity
    for i in range(N_GC):
        for j in range(N_GC):
            if i == j:
                continue
            a[i] += np.asarray(pot_gc[j].force(pos[i:i+1] - pos[j])[0], float)
    if apply_df:
        for i in range(N_GC):
            r_rel = pos[i] - ctr
            r     = float(np.linalg.norm(r_rel))
            rho   = float(pot_nfw.density(r_rel.reshape(1, 3))[0])
            sig   = sigma_lms1(r)
            lnL   = max(2.0, np.log(M_HALO / GCS[i]["mass"]))
            a[i] += chandrasekhar_acc(xv_gc[i, 3:] - vctr,
                                      GCS[i]["mass"], lnL, rho, sig)
    return a


def rewind_gcs(cen_p, cen_v, apply_df):
    """KDK leapfrog of the GCs BACKWARD from their observed present-day phase
    space to -T_BACKWARD, using the same forces as the forward step (MW + moving
    NFW + mutual GC gravity [+ GChalo DF]).  Forward-integrating the returned
    state then lands each GC back on its observed position today."""
    n   = int(round(T_BACKWARD / TAU))
    tau = -T_BACKWARD / n
    xv  = GC_OBS_IC.copy()
    t   = 0.0
    a   = accel_gc(xv, cen_p(t), cen_v(t), apply_df)
    for k in range(n):
        xv[:, 3:] += a * (tau / 2)
        xv[:, :3] += xv[:, 3:] * tau
        t          = (k + 1) * tau
        a          = accel_gc(xv, cen_p(t), cen_v(t), apply_df)
        xv[:, 3:] += a * (tau / 2)
    return xv


print(f"\n Planting {N_GC} GCs (rewound from observed positions)")
xv_gc_DF   = rewind_gcs(cen_p_DF,   cen_v_DF,   apply_df=True)
xv_gc_noDF = rewind_gcs(cen_p_noDF, cen_v_noDF, apply_df=False)
for i, g in enumerate(GCS):
    d0 = np.linalg.norm(xv_gc_DF[i, :3] - xva_DF[0, :3])
    print(f"  {g['name']}: planted {d0:.2f} kpc from LMS-1 core at t=-{T_BACKWARD:.0f} Gyr")


def run_forward(xv_stream, xv_gc, cen_p, cen_v, apply_df, label):
    """KDK leapfrog forward from -T_BACKWARD to 0.  Returns snapshots and the
    full GC trajectory (every step) for the core-retention diagnostics."""
    print(f"\n== {label} ==")
    tau     = TAU
    t_start = -T_BACKWARD
    snap_set = set(_ALL_STEPS)

    xv_stream = xv_stream.copy()
    xv_gc     = xv_gc.copy()

    t_now = t_start
    ctr   = cen_p(t_now)
    vctr  = cen_v(t_now)
    acc_s = accel_stream(xv_stream[:, :3], ctr, xv_gc[:, :3])
    acc_g = accel_gc(xv_gc, ctr, vctr, apply_df)

    snaps = {0: (xv_stream[:, :3].copy(), xv_gc[:, :3].copy(), t_now)}
    gc_traj  = [xv_gc.copy()]
    gc_times = [t_now]

    for k in range(_nsteps):
        # kick
        xv_stream[:, 3:] += acc_s * (tau / 2)
        xv_gc[:, 3:]     += acc_g * (tau / 2)
        # drift
        xv_stream[:, :3] += xv_stream[:, 3:] * tau
        xv_gc[:, :3]     += xv_gc[:, 3:] * tau
        # update fields
        t_now = t_start + (k + 1) * tau
        ctr   = cen_p(t_now)
        vctr  = cen_v(t_now)
        acc_s = accel_stream(xv_stream[:, :3], ctr, xv_gc[:, :3])
        acc_g = accel_gc(xv_gc, ctr, vctr, apply_df)
        # kick
        xv_stream[:, 3:] += acc_s * (tau / 2)
        xv_gc[:, 3:]     += acc_g * (tau / 2)

        gc_traj.append(xv_gc.copy())
        gc_times.append(t_now)

        step = k + 1
        if step in snap_set:
            snaps[step] = (xv_stream[:, :3].copy(), xv_gc[:, :3].copy(), t_now)
        if step % max(1, _nsteps // 10) == 0:
            sep = np.linalg.norm(xv_gc[0, :3] - ctr)
            print(f"  step {step:5d}/{_nsteps}  t = {t_now:+.3f} Gyr"
                  f"  |r_GC1 - core| = {sep:.3f} kpc")

    gyr_snaps  = [snaps[s] for s in _SNAP_GYR]
    anim_snaps = [snaps[s] for s in _SNAP_ANIM]
    return dict(label=label, gyr_snaps=gyr_snaps, anim_snaps=anim_snaps,
                xv_stream_final=xv_stream.copy(), xv_gc_final=xv_gc.copy(),
                gc_traj=np.stack(gc_traj), gc_times=np.asarray(gc_times),
                cen_p=cen_p, cen_v=cen_v)


res_DF   = run_forward(xv_stream_DF,   xv_gc_DF,   cen_p_DF,   cen_v_DF,
                       apply_df=True,  label="LMS-1 + GCs (with DF)")
res_noDF = run_forward(xv_stream_noDF, xv_gc_noDF, cen_p_noDF, cen_v_noDF,
                       apply_df=False, label="LMS-1 + GCs (no DF)")



# Core-retention diagnostics

# bound stellar core: particles with E_internal < 0 in the moving halo frame.
# core radius = half-number radius of the bound particles.

def bound_core(xyz, vel, ctr, vctr):
    rel  = xyz - ctr
    relv = vel - vctr
    ke   = 0.5 * np.sum(relv**2, axis=1)
    pe   = np.asarray(pot_nfw.potential(rel), float) \
         + np.asarray(pot_stars.potential(rel), float)
    bound = (ke + pe) < 0
    if bound.sum() < 10:
        return bound, np.nan
    r_bound = np.linalg.norm(rel[bound], axis=1)
    return bound, float(np.median(r_bound))    # half-number radius


def gc_core_track(res):
    """At every saved snapshot epoch, distance of each GC from the LMS-1
    centre and the bound-core half-mass radius + stream bound fraction."""
    cen_p, cen_v = res["cen_p"], res["cen_v"]
    traj, times  = res["gc_traj"], res["gc_times"]
    # subsample the full trajectory for the time series (every ~5 Myr)
    stride = max(1, len(times) // 400)
    idx    = np.arange(0, len(times), stride)
    d_gc   = np.zeros((len(idx), N_GC))
    for n, ii in enumerate(idx):
        ctr = cen_p(times[ii])
        for i in range(N_GC):
            d_gc[n, i] = np.linalg.norm(traj[ii, i, :3] - ctr)
    return times[idx], d_gc


# distance of each GC from core (time series)
t_ser_DF,   dgc_DF   = gc_core_track(res_DF)
t_ser_noDF, dgc_noDF = gc_core_track(res_noDF)

# bound core radius + stream bound fraction at the Gyr snapshots
core_r = {"DF": [], "noDF": []}
bfrac  = {"DF": [], "noDF": []}
for tag, res in [("DF", res_DF), ("noDF", res_noDF)]:
    cen_p, cen_v = res["cen_p"], res["cen_v"]
    for (xyz_s, xyz_g, t_s) in res["gyr_snaps"]:
        # need stream velocities -> only have final velocities; use final-state
        # bound test at t=0 and geometric half-mass radius for earlier epochs.
        ctr = cen_p(t_s)
        r   = np.linalg.norm(xyz_s - ctr, axis=1)
        core_r[tag].append(float(np.median(r[r < np.percentile(r, 90)])))
    # bound fraction at t=0 using full final phase space
    xv = res["xv_stream_final"]
    ctr, vctr = cen_p(0.0), cen_v(0.0)
    bound, _ = bound_core(xv[:, :3], xv[:, 3:], ctr, vctr)
    bfrac[tag] = float(bound.mean())

# final-state verdict at t = 0
print("\n" + "=" * 70)
print("CORE-RETENTION VERDICT at t = 0")
print("-" * 70)
verdict_lines = []
for tag, res in [("With DF", res_DF), ("No DF", res_noDF)]:
    cen_p, cen_v = res["cen_p"], res["cen_v"]
    ctr, vctr = cen_p(0.0), cen_v(0.0)
    xv = res["xv_stream_final"]
    bound, r_core = bound_core(xv[:, :3], xv[:, 3:], ctr, vctr)
    line = f"[{tag}]  bound stellar fraction = {bound.mean():.3f}" \
           f"   core half-mass radius = {r_core:.3f} kpc"
    print(line); verdict_lines.append(line)
    for i, g in enumerate(GCS):
        d = np.linalg.norm(res["xv_gc_final"][i, :3] - ctr)
        inside = "INSIDE core" if d < r_core else "outside core"
        line = (f"    {g['name']:9s}: dist from LMS-1 centre = {d:.3f} kpc"
                f"  ->  {inside}")
        print(line); verdict_lines.append(line)
print("=" * 70)



# Axis limits + style

_all_xyz = np.concatenate(
    [xyz for xyz, _, _ in res_DF["gyr_snaps"]]
    + [xyz for xyz, _, _ in res_noDF["gyr_snaps"]]
    + [xva_DF[:, :3], xva_noDF[:, :3]], axis=0)
# robust limits: clip the handful of fast escapers so the stream stays visible
PAD = 5.0
AX_LIMS = [(np.percentile(_all_xyz[:, k], 1.0) - PAD,
            np.percentile(_all_xyz[:, k], 99.0) + PAD)
           for k in range(3)]

PROJECTIONS = [
    dict(i=0, j=1, name="XY", xl="x [kpc]", yl="y [kpc]"),
    dict(i=0, j=2, name="XZ", xl="x [kpc]", yl="z [kpc]"),
    dict(i=1, j=2, name="YZ", xl="y [kpc]", yl="z [kpc]"),
]

BG            = "white"
C_DF          = "#1B7F2D"
C_DF_STREAM   = "#A8D5A2"
C_noDF        = "#C0392B"
C_noDF_STREAM = "#F4B8A0"


def style(ax, title, xl, yl):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel(xl, fontsize=8, color="0.35")
    ax.set_ylabel(yl, fontsize=8, color="0.35")
    ax.tick_params(colors="0.40", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("0.65")
    ax.grid(True, alpha=0.18, color="0.75", lw=0.5)



# Static snapshot figures  2 rows x 5 columns

print("\nRendering snapshot figures...")
N_EPOCHS = len(T_SNAPS)

for proj in PROJECTIONS:
    i, j   = proj["i"], proj["j"]
    name   = proj["name"]
    fname  = f"lms1gc_{name.lower()}.png"
    xl, yl = proj["xl"], proj["yl"]
    xlim, ylim = AX_LIMS[i], AX_LIMS[j]

    fig, axs = plt.subplots(2, N_EPOCHS, figsize=(3.5 * N_EPOCHS, 7.4), dpi=150,
                            sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)

    row_cfg = [
        (res_DF,   C_DF,   C_DF_STREAM,   xva_DF,   "With DF"),
        (res_noDF, C_noDF, C_noDF_STREAM, xva_noDF, "No DF"),
    ]
    for row, (res, c_orb, c_st, xva, row_label) in enumerate(row_cfg):
        for col, (xyz_s, xyz_g, t_s) in enumerate(res["gyr_snaps"]):
            ax = axs[row, col]
            title = (f"{row_label}\nt = {t_s:+.0f} Gyr"
                     if col == 0 else f"t = {t_s:+.0f} Gyr")
            style(ax, title, xl, yl)
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
            ax.plot(xva[:, i], xva[:, j], color=c_orb, lw=0.6, alpha=0.20, zorder=1)
            ax.scatter(xyz_s[:, i], xyz_s[:, j], s=3.0, c=c_st, alpha=0.6,
                       linewidths=0, rasterized=True, zorder=2,
                       label="LMS-1 stream" if col == 0 else None)
            # halo centre
            ctr = res["cen_p"](t_s)
            ax.plot(ctr[i], ctr[j], marker="o", ms=5, color=c_orb,
                    mec="black", mew=0.4, zorder=4,
                    label="LMS-1 core" if col == 0 else None)
            # LMS-1 subhalo tidal (Jacobi) radius at this epoch
            r_t = jacobi_radius(np.linalg.norm(ctr[:3]))
            _th = np.linspace(0, 2*np.pi, 120)
            ax.plot(ctr[i] + r_t*np.cos(_th), ctr[j] + r_t*np.sin(_th),
                    color=c_orb, lw=1.0, ls="", alpha=0.7, zorder=3,
                    label=("LMS-1 tidal radius" if col == 0 else None))
            # the two GCs
            for gi, g in enumerate(GCS):
                ax.plot(xyz_g[gi, i], xyz_g[gi, j], marker="*", ms=11,
                        color=g["color"], mec="black", mew=0.5, zorder=6,
                        label=g["name"] if col == 0 else None)
    # single legend OUTSIDE the plot grid (centred below)
    _h, _l = axs[0, 0].get_legend_handles_labels()
    fig.legend(_h, _l, fontsize=9, loc="upper center",
               bbox_to_anchor=(0.5, 0.045), ncol=len(_l),
               markerscale=1.6, framealpha=0.9)

    fig.suptitle(
        f"LMS-1 disruption with embedded GCs  {name} projection\n"
        f"(M_halo = {M_HALO:.1e} Msun,  b_prog = {B_LMS1} kpc)\n"
        f"Top: WITH dynamical friction  .  Bottom: NO friction (Yuan+/Malhan+ like)",
        fontsize=11, y=1.02)
    plt.tight_layout()
    save_fig_both(fig, fname, bbox_inches="tight", dpi=150,
                  facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {_both_msg(fname)}")



# Headline: GC distance from LMS-1 core vs time  one panel per scenario

print("\nRendering core-retention diagnostics...")
fig, axs = plt.subplots(1, 2, figsize=(15, 5.4), dpi=140, sharey=True)
fig.patch.set_facecolor(BG)

scenarios = [
    ("with DF", t_ser_DF,   dgc_DF,   core_r["DF"]),
    ("no DF",   t_ser_noDF, dgc_noDF, core_r["noDF"]),
]
for ax, (tag, t_ser, dgc, cr) in zip(axs, scenarios):
    style(ax, f"GC distance from the LMS-1 core  ({tag})",
          "t [Gyr]  (today = 0)", "|r_GC - r_core| [kpc]")
    for i, g in enumerate(GCS):
        ax.plot(t_ser, dgc[:, i], color=g["color"], lw=2.0,
                ls=g["ls_df"], label=g["name"])
    # core half-mass radius band at the Gyr snapshots
    ax.plot(T_SNAPS, cr, color="0.4", lw=1.2, marker="o", ms=4,
            label="LMS-1 core radius")
    ax.fill_between(T_SNAPS, 0, cr, color="0.7", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle(
    "Do the GCs stay in the LMS-1 core?\n"
    f"stream bound fraction at t=0:  DF = {bfrac['DF']:.2f}   no-DF = {bfrac['noDF']:.2f}",
    fontsize=12, y=1.02)
plt.tight_layout()
save_fig_both(fig, "lms1gc_core_distance.png", bbox_inches="tight", dpi=140,
              facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  saved: {_both_msg('lms1gc_core_distance.png')}")


# GC galactocentric radius (absolute orbit)

fig, ax = plt.subplots(figsize=(8, 5.4), dpi=140)
fig.patch.set_facecolor(BG)
style(ax, "GC galactocentric radius (absolute orbit)",
      "t [Gyr]  (today = 0)", "R_gc [kpc]")
ax.plot(ta_DF,   np.linalg.norm(xva_DF[:, :3],   axis=1),
        color="0.3", lw=1.4, label="LMS-1 core (DF)")
ax.plot(ta_noDF, np.linalg.norm(xva_noDF[:, :3], axis=1),
        color="0.3", lw=1.4, ls="", label="LMS-1 core (no DF)")
for i, g in enumerate(GCS):
    R_DF   = np.linalg.norm(res_DF["gc_traj"][:, i, :3],   axis=1)
    R_noDF = np.linalg.norm(res_noDF["gc_traj"][:, i, :3], axis=1)
    ax.plot(res_DF["gc_times"],   R_DF,   color=g["color"], lw=1.8,
            ls=g["ls_df"], label=f"{g['name']} (DF)")
    ax.plot(res_noDF["gc_times"], R_noDF, color=g["color"], lw=1.4,
            ls=g["ls_nodf"], alpha=0.8, label=f"{g['name']} (no DF)")
ax.legend(fontsize=8, loc="best", ncol=2)

plt.tight_layout()
save_fig_both(fig, "lms1gc_core_retention.png", bbox_inches="tight", dpi=140,
              facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  saved: {_both_msg('lms1gc_core_retention.png')}")



# GCGC separation

fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
fig.patch.set_facecolor(BG)
style(ax, "NGC 5024  NGC 5053 separation",
      "t [Gyr]  (today = 0)", "separation [kpc]")
sep_DF   = np.linalg.norm(res_DF["gc_traj"][:, 0, :3]
                          - res_DF["gc_traj"][:, 1, :3], axis=1)
sep_noDF = np.linalg.norm(res_noDF["gc_traj"][:, 0, :3]
                          - res_noDF["gc_traj"][:, 1, :3], axis=1)
ax.plot(res_DF["gc_times"],   sep_DF,   color=C_DF,   lw=2.0, label="with DF")
ax.plot(res_noDF["gc_times"], sep_noDF, color=C_noDF, lw=2.0, label="no DF")
_obs_sep = np.linalg.norm(NGC5024_IC[:3] - NGC5053_IC[:3])
ax.axhline(_obs_sep, color="0.4", ls=":", lw=1.2,
           label=f"observed today ({_obs_sep:.2f} kpc)")
ax.legend(fontsize=10)
plt.tight_layout()
save_fig_both(fig, "lms1gc_gc_separation.png", bbox_inches="tight", dpi=140,
              facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  saved: {_both_msg('lms1gc_gc_separation.png')}")



# RA/Dec sky map at t = 0

from astropy.coordinates import ICRS, CartesianRepresentation
# use the SAME Galactic frame as the cluster ICs (mw14) so model and observation
# are transformed consistently.
_gc_frame = _gc_frame_mw14


def gc_to_radec(xyz_kpc):
    c = _gc_frame.realize_frame(CartesianRepresentation(
        x=xyz_kpc[:, 0]*u.kpc, y=xyz_kpc[:, 1]*u.kpc, z=xyz_kpc[:, 2]*u.kpc))
    eq = c.transform_to(ICRS())
    return eq.ra.deg, eq.dec.deg


# observed LMS-1 stream (STREAMFINDER / Ibata+2024)
import pandas as pd
_obs_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cleaned_streamfinder_ibata24.csv")
_obs = pd.read_csv(_obs_csv)
_obs = _obs[_obs["Name"] == "LMS-1"]
OBS_RA, OBS_DEC = _obs["RAdeg"].values, _obs["DEdeg"].values
print(f"  observed LMS-1 members (Ibata+2024): {len(_obs)}")


T_ORB = T_BACKWARD   # show the orbit over the FULL disruption time of the run


def _mask_ra_wrap(ra, dec):
    """Insert NaN where RA jumps >180 deg so the polyline doesn't cross the sky."""
    ra = np.asarray(ra, float).copy(); dec = np.asarray(dec, float).copy()
    j = np.where(np.abs(np.diff(ra)) > 180.0)[0]
    return np.insert(ra, j + 1, np.nan), np.insert(dec, j + 1, np.nan)


def orbit_track_radec(xva_rewind, ta):
    """RA/Dec of the rewound PAST trajectory over the last T_ORB Gyr, ending at
    the present (today is the endpoint).  Uses this run's DF/no-DF rewind, so the
    two panels differ in proportion to the haloMW friction."""
    recent = xva_rewind[:, :3][ta > -T_ORB]
    ra, dec = gc_to_radec(recent)
    return _mask_ra_wrap(ra, dec)


def tidal_circle_radec(ctr_xyz, r_t, n=160):
    """Angular circle of radius r_t (kpc) around a Galactocentric point,
    returned as RA/Dec arrays for plotting on the sky."""
    rc, dc = gc_to_radec(ctr_xyz.reshape(1, 3))
    rc, dc = float(rc[0]), float(dc[0])
    d_hel = float(np.linalg.norm(ctr_xyz - np.array([-8.0, 0.0, 0.025])))
    ang = np.degrees(r_t / d_hel)
    th = np.linspace(0, 2*np.pi, n)
    return rc + ang*np.cos(th)/np.cos(np.radians(dc)), dc + ang*np.sin(th)


print("\nRendering RA/Dec sky map (t = 0) ...")
fig, axs = plt.subplots(1, 2, figsize=(16, 5.5), dpi=140)
fig.patch.set_facecolor(BG)
sky_cfg = [(axs[0], res_DF, C_DF_STREAM, xva_DF, ta_DF, "With DF"),
           (axs[1], res_noDF, C_noDF_STREAM, xva_noDF, ta_noDF, "No DF")]
for ax, res, c_st, xva, ta, lbl in sky_cfg:
    ax.set_facecolor(BG); ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_title(f"{lbl}  (t = 0)", fontsize=11, pad=8)
    ra, dec = gc_to_radec(res["xv_stream_final"][:, :3])
    ax.scatter(ra, dec, s=3.0, c=c_st, alpha=0.6, linewidths=0,
               rasterized=True, label="LMS-1 stream (model)", zorder=2)
    # observed LMS-1 stream
    ax.scatter(OBS_RA, OBS_DEC, s=8, c="#111111", marker=".", alpha=0.8,
               linewidths=0, zorder=4, label="LMS-1 (STREAMFINDER)")
    # progenitor orbit line (rewound past, ends at today; this panel's DF setting)
    orb_ra, orb_dec = orbit_track_radec(xva, ta)
    ax.plot(orb_ra, orb_dec, color="black", lw=1.8, alpha=0.8, zorder=5,
            solid_capstyle="round",
            label=f"LMS-1 orbit ({T_ORB:.0f} Gyr, {lbl})")
    # present-day position = endpoint of the orbit
    _nra, _ndec = gc_to_radec(LMS1_IC[:3].reshape(1, 3))
    ax.plot(_nra[0], _ndec[0], marker="X", ms=11, color="black", mec="white",
            mew=0.8, zorder=7, label="LMS-1 today")
    # LMS-1 subhalo tidal radius (angular) around the core
    ctr0 = res["cen_p"](0.0)
    r_t0 = jacobi_radius(np.linalg.norm(ctr0[:3]))
    tc_ra, tc_dec = tidal_circle_radec(ctr0[:3], r_t0)
    ax.plot(tc_ra, tc_dec, color=c_st, lw=1.2, ls="", alpha=0.9, zorder=3,
            label=f"LMS-1 tidal radius ({r_t0:.1f} kpc)")
    # simulated GC end points
    ra_g, dec_g = gc_to_radec(res["xv_gc_final"][:, :3])
    for gi, g in enumerate(GCS):
        ax.plot(ra_g[gi], dec_g[gi], marker="*", ms=15, color=g["color"],
                mec="k", mew=0.6, zorder=6, label=f"{g['name']} (sim)")
    # observed positions
    ax.plot(NGC5024_RA_DEG, NGC5024_DEC_DEG, marker="o", ms=9, mfc="none",
            mec="k", mew=1.3, zorder=5, label="NGC 5024 (obs)")
    ax.plot(NGC5053_RA_DEG, NGC5053_DEC_DEG, marker="s", ms=9, mfc="none",
            mec="k", mew=1.3, zorder=5, label="NGC 5053 (obs)")
    ax.set_xlabel("RA [deg]", fontsize=9); ax.set_ylabel("Dec [deg]", fontsize=9)
    ax.set_xlim(360, 0); ax.set_ylim(-90, 90)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              markerscale=1.6, framealpha=0.85, ncol=4, borderaxespad=0)
fig.suptitle("LMS-1 stream + embedded GCs vs STREAMFINDER — RA/Dec (ICRS)  |  t = 0",
             fontsize=12, y=1.02)
plt.tight_layout()
save_fig_both(fig, "lms1gc_radec.png", bbox_inches="tight", dpi=140,
              facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  saved: {_both_msg('lms1gc_radec.png')}")



# Energy & angular momentum at t = 0

print("\nComputing E & L at t = 0 ...")


def compute_E_L(xv):
    pos, vel = xv[:, :3], xv[:, 3:]
    KE   = 0.5 * np.sum(vel**2, axis=1)
    Phi  = np.asarray(pot_host.potential(pos), float)
    E    = KE + Phi
    Lz   = pos[:, 0]*vel[:, 1] - pos[:, 1]*vel[:, 0]
    L    = np.linalg.norm(np.cross(pos, vel), axis=1)
    return E, Lz, L


E_DF,   Lz_DF,   L_DF   = compute_E_L(res_DF["xv_stream_final"])
E_noDF, Lz_noDF, L_noDF = compute_E_L(res_noDF["xv_stream_final"])
Eg_DF,  Lzg_DF,  Lg_DF  = compute_E_L(res_DF["xv_gc_final"])
Eg_noDF, Lzg_noDF, Lg_noDF = compute_E_L(res_noDF["xv_gc_final"])

fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), dpi=140)
fig.patch.set_facecolor(BG)
ax = axs[0]
style(ax, r"$E$ vs $L_z$ (integrals of motion)",
      r"$L_z$ [kpc km/s]", r"$E$ [km$^2$/s$^2$]")
ax.scatter(Lz_noDF, E_noDF, s=1.5, c=C_noDF_STREAM, alpha=0.4, linewidths=0,
           rasterized=True, label="stream (no DF)")
ax.scatter(Lz_DF, E_DF, s=1.5, c=C_DF_STREAM, alpha=0.4, linewidths=0,
           rasterized=True, label="stream (DF)")
for gi, g in enumerate(GCS):
    ax.plot(Lzg_DF[gi], Eg_DF[gi], marker="*", ms=14, color=g["color"],
            mec="k", mew=0.6, label=f"{g['name']} (DF)")
ax.legend(fontsize=8, markerscale=3)

ax = axs[1]
style(ax, r"$E$ vs $|L|$",
      r"$|L|$ [kpc km/s]", r"$E$ [km$^2$/s$^2$]")
ax.scatter(L_noDF, E_noDF, s=1.5, c=C_noDF_STREAM, alpha=0.4, linewidths=0,
           rasterized=True, label="stream (no DF)")
ax.scatter(L_DF, E_DF, s=1.5, c=C_DF_STREAM, alpha=0.4, linewidths=0,
           rasterized=True, label="stream (DF)")
for gi, g in enumerate(GCS):
    ax.plot(Lg_DF[gi], Eg_DF[gi], marker="*", ms=14, color=g["color"],
            mec="k", mew=0.6, label=f"{g['name']} (DF)")
ax.legend(fontsize=8, markerscale=3)
fig.suptitle("LMS-1 stream + GCs — integrals of motion at t = 0", fontsize=12, y=1.01)
plt.tight_layout()
save_fig_both(fig, "lms1gc_energy_L.png", bbox_inches="tight", dpi=140,
              facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  saved: {_both_msg('lms1gc_energy_L.png')}")



# Animations  side by side (DF left . no-DF right)

print("\nBuilding animations...")
FPS = 30

# Locate ffmpeg (HPC module installs it outside the default PATH); fall back to
# an animated GIF (Pillow, no ffmpeg needed) if no ffmpeg binary is found.
_FFMPEG = shutil.which("ffmpeg")
if _FFMPEG is None:
    for _cand in ("/cluster/software/ffmpeg/6.1/bin/ffmpeg",):
        if os.path.exists(_cand):
            _FFMPEG = _cand
            break
if _FFMPEG is not None:
    matplotlib.rcParams["animation.ffmpeg_path"] = _FFMPEG
    print(f"  using ffmpeg: {_FFMPEG}")
else:
    print("  ffmpeg not found -> writing .gif animations instead")


def save_anim(fig, upd, filename, n_frames):
    anim = animation.FuncAnimation(fig, upd, frames=n_frames,
                                   interval=1000/FPS, blit=False)
    if _FFMPEG is not None:
        writer = animation.FFMpegWriter(fps=FPS, bitrate=2500, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p"])
        out = filename
    else:
        writer = animation.PillowWriter(fps=FPS)
        out = filename.rsplit(".", 1)[0] + ".gif"
    p = out_path(out, "df")
    anim.save(p, writer=writer, dpi=140,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    shutil.copy2(p, out_path(out, "nodf"))
    plt.close(fig)
    print(f"  saved: {_both_msg(out)}")


for proj in PROJECTIONS:
    i, j   = proj["i"], proj["j"]
    name   = proj["name"]
    fname  = f"lms1gc_{name.lower()}.mp4"
    xl, yl = proj["xl"], proj["yl"]
    xlim, ylim = AX_LIMS[i], AX_LIMS[j]

    fig, axs = plt.subplots(1, 2, figsize=(14.4, 7.2), dpi=140)
    fig.patch.set_facecolor(BG)
    panels = [
        dict(ax=axs[0], res=res_DF,   xva=xva_DF,   c_orb=C_DF,
             c_st=C_DF_STREAM,   label="With DF"),
        dict(ax=axs[1], res=res_noDF, xva=xva_noDF, c_orb=C_noDF,
             c_st=C_noDF_STREAM, label="No DF"),
    ]
    for p in panels:
        ax = p["ax"]
        style(ax, f"LMS-1 + GCs  {p['label']}  ({name})", xl, yl)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
        ax.plot(p["xva"][:, i], p["xva"][:, j], color=p["c_orb"], lw=0.7, alpha=0.2)
        p["sc"]  = ax.scatter([], [], s=3.0, c=p["c_st"], alpha=0.7,
                              linewidths=0, label="LMS-1 stream", rasterized=True)
        p["dot"] = ax.plot([], [], "o", ms=6, color=p["c_orb"], mec="black",
                           mew=0.4, label="LMS-1 core", zorder=7)[0]
        p["gcs"] = [ax.plot([], [], "*", ms=12, color=g["color"], mec="k",
                            mew=0.5, label=g["name"], zorder=8)[0] for g in GCS]
        p["tid"] = ax.plot([], [], color=p["c_orb"], lw=1.1, ls="", alpha=0.7,
                           label="LMS-1 tidal radius", zorder=6)[0]
        p["ttl"] = ax.text(0.02, 0.96, "", transform=ax.transAxes, color="black",
                           fontsize=10, va="top")
        ax.legend(fontsize=8, loc="lower right", markerscale=1.6)
    fig.suptitle(f"LMS-1 disruption + embedded GCs  {name}  (DF left . no-DF right)",
                 fontsize=13, y=0.995)

    def _make_upd(panels_=panels, i_=i, j_=j):
        def upd(fi):
            for p in panels_:
                xyz_s, xyz_g, t = p["res"]["anim_snaps"][fi]
                ctr = p["res"]["cen_p"](t)
                p["sc"].set_offsets(np.column_stack([xyz_s[:, i_], xyz_s[:, j_]]))
                p["dot"].set_data([ctr[i_]], [ctr[j_]])
                for gi in range(N_GC):
                    p["gcs"][gi].set_data([xyz_g[gi, i_]], [xyz_g[gi, j_]])
                r_t = jacobi_radius(np.linalg.norm(ctr[:3]))
                _th = np.linspace(0, 2*np.pi, 120)
                p["tid"].set_data(ctr[i_] + r_t*np.cos(_th),
                                  ctr[j_] + r_t*np.sin(_th))
                p["ttl"].set_text(f"t = {t:+.3f} Gyr")
            return ()
        return upd

    save_anim(fig, _make_upd(), fname, n_frames=N_ANIM)



# Save snapshot arrays + summary

print("\nSaving snapshot arrays ...")
for tag, res in [("df", res_DF), ("nodf", res_noDF)]:
    for ep_i, t_ep in enumerate(T_SNAPS):
        xyz_s, xyz_g, _ = res["gyr_snaps"][ep_i]
        np.save(out_path(f"snap_stream_{tag}_t{int(round(t_ep)):+d}gyr.npy", tag),
                xyz_s)
        np.save(out_path(f"snap_gc_{tag}_t{int(round(t_ep)):+d}gyr.npy", tag),
                xyz_g)
    np.save(out_path(f"snap_stream_{tag}_final.npy", tag), res["xv_stream_final"])
    np.save(out_path(f"snap_gc_{tag}_final.npy", tag), res["xv_gc_final"])
    print(f"  saved snap_stream_{tag}_* and snap_gc_{tag}_* in "
          f"mass_{MASS_TAG}_{tag}/")

_summary = ["LMS-1 progenitor rewind + disruption with two embedded GCs",
            "Dynamical friction added (absent in Yuan+2020 / Malhan+2021)",
            f"LMS-1 subhalo mass M_200 = {M_HALO:.2e} Msun  (tag {MASS_TAG})",
            "=" * 60, ""] + verdict_lines
for _tag in ("df", "nodf"):
    with open(out_path("results_summary.txt", _tag), "w") as f:
        f.write("\n".join(_summary) + "\n")
print(f"  saved: {_both_msg('results_summary.txt')}")

print("\nAll done.")
