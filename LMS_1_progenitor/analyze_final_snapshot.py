#!/usr/bin/env python
"""
Capitalise the FINAL N-body snapshots (t = 0) of the LMS-1 progenitor
rewind+disrupt run and confront the debris with the *observed* LMS-1 stream.

What this does
--------------
The forward integration in ``progenitor_rewind_disrupt.py`` saves the full
6-D phase space of every stream particle at t = 0:

    snap_stream_df_final.npy    (N, 6)   x,y,z [kpc] , vx,vy,vz [km/s]
    snap_stream_nodf_final.npy  (N, 6)
    snap_gc_df_final.npy        (2, 6)
    snap_gc_nodf_final.npy      (2, 6)

all in the Galactocentric frame consistent with MWPotential2014 (Bovy 2015,
the same frame used to build the cluster initial conditions).  Here we

  1. transform that 6-D debris back to *observable* coordinates
     (RA, Dec, distance, pm_ra_cosdec, pm_dec, v_los),
  2. load the cleaned STREAMFINDER catalogue (Ibata+2024) and pull the stars
     tagged ``LMS-1``  -> the real stream on the sky today,
  3. overlay the observation on the model in RA/Dec (updates lms1gc_radec.png),
  4. select the model debris that falls *inside the observed RA/Dec footprint*
     and compare its proper motion (pmRA, pmDE) and line-of-sight velocity
     against the catalogue -> does the N-body stream reproduce the kinematics?

Outputs
-------
  lms1gc_radec.png            RA/Dec model (DF | no-DF) + observed LMS-1 stars
  lms1gc_pm_compare.png       pmRA, pmDE vs RA : model-in-footprint vs observed
  lms1gc_vlos_compare.png     v_los vs RA      : model-in-footprint vs observed
  obs_comparison_summary.txt  median/scatter table + verdict
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import ConvexHull

import astropy.units as u
from astropy.coordinates import (SkyCoord, Galactocentric, ICRS,
                                 CartesianRepresentation,
                                 CartesianDifferential)
import agama

agama.setUnits(length=1, velocity=1, mass=1)

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "..", "data", "cleaned_streamfinder_ibata24.csv")
STREAM_NAME = "LMS-1"

# --- MW potential + LMS-1 NFW (same parameters as progenitor_rewind_disrupt) --
pot_host = agama.Potential(os.path.join(HERE, "..", "MWPotential2014.ini"))
# read M_HALO from the sibling run script so mass-variation folders are correct
import re as _re
_mainsrc = open(os.path.join(HERE, "progenitor_rewind_disrupt.py")).read()
M_HALO = float(_re.search(r"^M_HALO = ([0-9.eE+\-]+)", _mainsrc, _re.M).group(1))
T_BACKWARD = float(_re.search(r"^T_BACKWARD = ([0-9.]+)", _mainsrc, _re.M).group(1))
Z_HALO, LITTLE_H = 1.0, 0.70
print(f"  analyze using M_HALO = {M_HALO:.3e} Msun, T_BACKWARD = {T_BACKWARD} Gyr"
      "  (from run script)")


# --- per-(mass, DF) result folders, matching progenitor_rewind_disrupt.py -----
def _mass_tag(m):
    """1e10 -> '1e10', 3.2e9 -> '3p2e9' (filesystem-safe mantissa)."""
    e = int(np.floor(np.log10(m) + 1e-9))
    c = m / 10.0**e
    c_s = (f"{int(round(c))}" if abs(c - round(c)) < 1e-6
           else f"{c:.2f}".rstrip("0").rstrip(".").replace(".", "p"))
    return f"{c_s}e{e}"


MASS_TAG = _mass_tag(M_HALO)
OUT_DIRS = {tag: os.path.join(HERE, f"mass_{MASS_TAG}_{tag}")
            for tag in ("df", "nodf")}
for _d in OUT_DIRS.values():
    os.makedirs(_d, exist_ok=True)
print(f"  result folders: mass_{MASS_TAG}_df/  and  mass_{MASS_TAG}_nodf/")


def out_path(fname, tag):
    """Path of an output file in the 'df' or 'nodf' folder of this mass."""
    return os.path.join(OUT_DIRS[tag], fname)


def in_path(fname, tag):
    """Snapshot input path; falls back to HERE for pre-folder runs."""
    p = out_path(fname, tag)
    return p if os.path.exists(p) else os.path.join(HERE, fname)


def save_fig_both(fig, fname, **kw):
    """Combined DF/no-DF figure -> written into BOTH scenario folders."""
    p = out_path(fname, "df")
    fig.savefig(p, **kw)
    shutil.copy2(p, out_path(fname, "nodf"))
    return f"{fname}  ->  mass_{MASS_TAG}_{{df,nodf}}/"
_E2 = 0.30*(1+Z_HALO)**3 + 0.70
_rhoc = 277.5*LITTLE_H**2 * _E2
_a = 0.520 + (0.905-0.520)*np.exp(-0.617*Z_HALO**1.21)
_b = -0.101 + 0.026*Z_HALO
_c = 10.0**(_a + _b*np.log10(M_HALO*LITTLE_H/1e12))
_r200 = (3*M_HALO/(4*np.pi*200*_rhoc))**(1/3)
RS_NFW = _r200/_c
M_NFW_CHAR = M_HALO/(np.log(1+_c) - _c/(1+_c))


def mw_mass_enclosed(R):
    a = np.asarray(pot_host.force([[R, 0.0, 0.0]])[0], float)
    return abs(a[0]) * R * R / agama.G


def jacobi_radius(R):
    R = max(float(R), 1e-3); Mmw = mw_mass_enclosed(R); rt = 1.0
    for _ in range(60):
        Mn = M_NFW_CHAR*(np.log(1+rt/RS_NFW) - (rt/RS_NFW)/(1+rt/RS_NFW))
        rt = R*(Mn/(3*Mmw))**(1/3)
    return rt


# --- progenitor rewind with / without halo--MW dynamical friction -------------
# identical scheme to progenitor_rewind_disrupt.py so the orbit line matches the
# actual moving-centre trajectory used in the simulation.
import scipy.special
LN_LAMBDA_HALO = 4.6
TAU = 2.0**-11          # Gyr


def _halo_df_acc(pos, vel):
    p = pos.reshape(1, 3)
    rho = float(pot_host.density(p)[0])
    Phi = float(pot_host.potential(p)[0])
    sig = np.sqrt(max(-Phi/2.0, 100.0))
    v = float(np.linalg.norm(vel))
    if v < 1.0 or rho <= 0.0:
        return np.zeros(3)
    X = v/(np.sqrt(2.0)*sig)
    fac = scipy.special.erf(X) - 2.0*X/np.sqrt(np.pi)*np.exp(-X*X)
    if fac <= 0.0:
        return np.zeros(3)
    return -4.0*np.pi*agama.G**2*M_HALO*LN_LAMBDA_HALO*rho*fac/v**3 * vel


def rewind(ic_today, t_back, apply_df):
    """KDK leapfrog backward; returns trajectory (oldest -> today), today last."""
    n = int(round(t_back/TAU)); tau = -t_back/n
    s = np.asarray(ic_today, float).copy()
    def acc(p, v):
        a = np.asarray(pot_host.force(p.reshape(1, 3))[0], float)
        return a + (_halo_df_acc(p, v) if apply_df else 0.0)
    a = acc(s[:3], s[3:]); tr = [s.copy()]
    for _ in range(n):
        s[3:] += a*tau/2; s[:3] += s[3:]*tau
        a = acc(s[:3], s[3:]); s[3:] += a*tau/2
        tr.append(s.copy())
    return np.asarray(tr)[::-1]          # oldest first, today (= ic) last


def mask_ra_wrap(ra, dec):
    """Insert NaN where RA jumps >180 deg so the polyline doesn't cross the sky."""
    ra = np.asarray(ra, float).copy(); dec = np.asarray(dec, float).copy()
    jump = np.where(np.abs(np.diff(ra)) > 180.0)[0]
    ra = np.insert(ra, jump + 1, np.nan); dec = np.insert(dec, jump + 1, np.nan)
    return ra, dec

# --- Galactic frame: IDENTICAL to the one used to build the ICs in the sim ----
# (progenitor_rewind_disrupt.py: _gc_frame_mw14). Converting the final snapshot
# back with the same frame keeps the model<->observation comparison consistent.
GC_FRAME = Galactocentric(
    galcen_distance=8.0 * u.kpc,
    galcen_v_sun=[11.1, 220.0 + 12.24, 7.25] * u.km / u.s,
    z_sun=25.0 * u.pc)

# observed GC reference points (Baumgardt&Hilker dist; Vasiliev&Baumgardt PM)
GC_OBS = [
    dict(name="NGC 5024", color="#1f4e8c", ra=198.230, dec=18.168,
         dist=18.13, pmra=-0.133, pmdec=-0.095, vlos=-62.9, mass=5.02e5),
    dict(name="NGC 5053", color="#8c1f6b", ra=199.113, dec=17.698,
         dist=17.54, pmra=-0.330, pmdec=-0.158, vlos=44.0, mass=6.28e4),
]


# ==============================================================================
# 6-D Galactocentric  ->  ICRS observables
# ==============================================================================
def galcen6d_to_obs(xv):
    """xv : (N,6) [x,y,z (kpc), vx,vy,vz (km/s)] in GC_FRAME.
    Returns dict of ra, dec [deg], dist [kpc], pmra (=pm_ra_cosdec),
    pmdec [mas/yr], vlos [km/s]."""
    xv = np.atleast_2d(np.asarray(xv, float))
    rep = CartesianRepresentation(
        xv[:, 0] * u.kpc, xv[:, 1] * u.kpc, xv[:, 2] * u.kpc,
        differentials=CartesianDifferential(
            xv[:, 3] * u.km / u.s, xv[:, 4] * u.km / u.s, xv[:, 5] * u.km / u.s))
    gc = GC_FRAME.realize_frame(rep)
    icrs = gc.transform_to(ICRS())
    return dict(
        ra=icrs.ra.deg, dec=icrs.dec.deg,
        dist=icrs.distance.to(u.kpc).value,
        pmra=icrs.pm_ra_cosdec.to(u.mas / u.yr).value,
        pmdec=icrs.pm_dec.to(u.mas / u.yr).value,
        vlos=icrs.radial_velocity.to(u.km / u.s).value)


def obs_to_galcen6d(ra, dec, dist, pmra, pmdec, vlos):
    """Inverse: observed ICRS -> 6-D Galactocentric [x,y,z,vx,vy,vz]."""
    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.kpc,
                  pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr,
                  radial_velocity=vlos*u.km/u.s, frame="icrs")
    g = sc.transform_to(GC_FRAME)
    return np.array([g.x.to(u.kpc).value, g.y.to(u.kpc).value,
                     g.z.to(u.kpc).value, g.v_x.to(u.km/u.s).value,
                     g.v_y.to(u.km/u.s).value, g.v_z.to(u.km/u.s).value])


def xyz_to_radec(xyz):
    """Galactocentric positions (N,3) -> RA/Dec [deg] (positions only)."""
    xyz = np.atleast_2d(np.asarray(xyz, float))
    rep = CartesianRepresentation(xyz[:, 0]*u.kpc, xyz[:, 1]*u.kpc,
                                  xyz[:, 2]*u.kpc)
    eq = GC_FRAME.realize_frame(rep).transform_to(ICRS())
    return eq.ra.deg, eq.dec.deg


# ==============================================================================
# Load model final snapshots + observation
# ==============================================================================
print("=" * 70)
print("LMS-1  N-body final snapshot  vs  STREAMFINDER (Ibata+2024) observation")
print("-" * 70)

models = {}
for tag in ("df", "nodf"):
    xv_s = np.load(in_path(f"snap_stream_{tag}_final.npy", tag))
    xv_g = np.load(in_path(f"snap_gc_{tag}_final.npy", tag))
    models[tag] = dict(stream=galcen6d_to_obs(xv_s), gc=galcen6d_to_obs(xv_g))
    print(f"  loaded snap_stream_{tag}_final.npy : {xv_s.shape[0]} particles")

cat = pd.read_csv(DATA_CSV)
obs = cat[cat["Name"] == STREAM_NAME].copy()
print(f"  STREAMFINDER '{STREAM_NAME}' members : {len(obs)}")
# valid line-of-sight velocities only (e_VHel = 300 is the no-measurement flag)
obs_v = obs[obs["e_VHel"] < 300.0].copy()
print(f"  ... with measured v_los (e_VHel<300) : {len(obs_v)}")

obs_ra, obs_dec = obs["RAdeg"].values, obs["DEdeg"].values
obs_pmra, obs_pmdec = obs["pmRA"].values, obs["pmDE"].values


# ==============================================================================
# Observed footprint  ->  select model debris that lands inside it
# ==============================================================================
# convex hull of the observed (RA,Dec) points, slightly dilated, used as a
# spatial mask to pick the model particles that overlap the real stream.
hull_pts = np.column_stack([obs_ra, obs_dec])
hull = ConvexHull(hull_pts)
poly = hull_pts[hull.vertices]
centroid = poly.mean(axis=0)
poly_dil = centroid + 1.15 * (poly - centroid)        # 15 % dilation
footprint = Path(poly_dil)


def in_footprint(ra, dec):
    return footprint.contains_points(np.column_stack([ra, dec]))


for tag in ("df", "nodf"):
    s = models[tag]["stream"]
    s["mask"] = in_footprint(s["ra"], s["dec"])
    print(f"  model[{tag}] debris inside observed footprint : "
          f"{s['mask'].sum()} / {s['ra'].size}")


# --- LMS-1 progenitor core today + orbit line + tidal radius ------------------
# core today = mass-weighted mean phase space of the two clusters (= the moving
# NFW centre at t=0, identical to LMS1_IC in progenitor_rewind_disrupt.py)
_ics = np.array([obs_to_galcen6d(g["ra"], g["dec"], g["dist"],
                                 g["pmra"], g["pmdec"], g["vlos"]) for g in GC_OBS])
_w = np.array([g["mass"] for g in GC_OBS])
LMS1_IC = (_w[:, None] * _ics).sum(0) / _w.sum()
R0 = float(np.linalg.norm(LMS1_IC[:3]))
R_T = jacobi_radius(R0)
print(f"  LMS-1 core today : R = {R0:.2f} kpc   tidal radius r_t = {R_T:.2f} kpc")

# orbit line = rewound PAST trajectory ending at the present (today = endpoint),
# computed separately WITH and WITHOUT halo--MW dynamical friction (they diverge
# back in time; identical at t=0).  Span the FULL disruption time of this run.
T_ORB = T_BACKWARD
ORB = {}
for tag, df in [("df", True), ("nodf", False)]:
    tr = rewind(LMS1_IC, T_ORB, apply_df=df)
    ra, dec = xyz_to_radec(tr[:, :3])
    ORB[tag] = mask_ra_wrap(ra, dec)
# present-day position (shared endpoint of both)
NOW_RA, NOW_DEC = xyz_to_radec(LMS1_IC[:3].reshape(1, 3))
NOW_RA, NOW_DEC = float(NOW_RA[0]), float(NOW_DEC[0])
print(f"  LMS-1 today on sky : RA = {NOW_RA:.1f}, Dec = {NOW_DEC:.1f} deg")


def tidal_circle_radec(ctr_xyz, r_t, n=160):
    rc, dc = xyz_to_radec(np.asarray(ctr_xyz).reshape(1, 3))
    rc, dc = float(rc[0]), float(dc[0])
    d_hel = float(np.linalg.norm(np.asarray(ctr_xyz) - np.array([-8.0, 0.0, 0.025])))
    ang = np.degrees(r_t / d_hel)
    th = np.linspace(0, 2*np.pi, n)
    return rc + ang*np.cos(th)/np.cos(np.radians(dc)), dc + ang*np.sin(th)


TC_RA, TC_DEC = tidal_circle_radec(LMS1_IC[:3], R_T)


# ==============================================================================
# Figure 1 : RA/Dec  (model DF | no-DF)  +  observed LMS-1 stream  (t = 0)
# ==============================================================================
BG = "white"
C_DF_STREAM, C_noDF_STREAM = "#A8D5A2", "#F4B8A0"
C_OBS = "#111111"

print("\nRendering RA/Dec sky map with observation overlay ...")
fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), dpi=140)
fig.patch.set_facecolor(BG)
cfg = [(axs[0], "df", C_DF_STREAM, "With DF"),
       (axs[1], "nodf", C_noDF_STREAM, "No DF")]
for ax, tag, c_st, lbl in cfg:
    ax.set_facecolor(BG); ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_title(f"{lbl}   (t = 0)", fontsize=11, pad=8)
    s, g = models[tag]["stream"], models[tag]["gc"]
    ax.scatter(s["ra"], s["dec"], s=3.0, c=c_st, alpha=0.55, linewidths=0,
               rasterized=True, label="LMS-1 stream (model)", zorder=2)
    # observed footprint outline
    ax.plot(np.append(poly_dil[:, 0], poly_dil[0, 0]),
            np.append(poly_dil[:, 1], poly_dil[0, 1]),
            color="0.45", lw=1.0, ls="--", zorder=3,
            label="observed footprint")
    # observed LMS-1 stream stars
    ax.scatter(obs_ra, obs_dec, s=8, c=C_OBS, marker=".", alpha=0.8,
               linewidths=0, zorder=4, label="LMS-1 (STREAMFINDER)")
    # LMS-1 progenitor orbit (rewound past, this panel's DF setting; ends at now)
    orb_ra, orb_dec = ORB[tag]
    ax.plot(orb_ra, orb_dec, color="black", lw=1.8, alpha=0.8, zorder=5,
            solid_capstyle="round",
            label=f"LMS-1 orbit ({T_ORB:.0f} Gyr, {'DF' if tag=='df' else 'no DF'})")
    # present-day progenitor position = endpoint of the orbit
    ax.plot(NOW_RA, NOW_DEC, marker="X", ms=11, color="black", mec="white",
            mew=0.8, zorder=7, label="LMS-1 today")
    # LMS-1 subhalo tidal radius (angular circle around the core)
    ax.plot(TC_RA, TC_DEC, color=c_st, lw=1.3, ls="--", alpha=0.95, zorder=3,
            label=f"LMS-1 tidal radius ({R_T:.1f} kpc)")
    # model + observed GCs
    for gi, go in enumerate(GC_OBS):
        ax.plot(g["ra"][gi], g["dec"][gi], marker="*", ms=15, color=go["color"],
                mec="k", mew=0.6, zorder=6, label=f"{go['name']} (model)")
        ax.plot(go["ra"], go["dec"], marker="o", ms=10, mfc="none", mec="k",
                mew=1.3, zorder=5, label=f"{go['name']} (obs)")
    ax.set_xlabel("RA [deg]", fontsize=9); ax.set_ylabel("Dec [deg]", fontsize=9)
    ax.set_xlim(360, 0); ax.set_ylim(-90, 90)
    # per-panel legend (each panel's orbit is its OWN DF / no-DF rewind)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=4, markerscale=1.3, framealpha=0.9, borderaxespad=0)
fig.suptitle("LMS-1 stream model vs STREAMFINDER (Ibata+2024) — RA/Dec (ICRS) | t = 0\n"
             "(footprint = kinematic-comparison region)",
             fontsize=12, y=1.04)
plt.tight_layout()
# this is the definitive RA/Dec figure (overwrites the main script's)
out = save_fig_both(fig, "lms1gc_radec.png", bbox_inches="tight", dpi=140,
                    facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  saved: {out}")


# ==============================================================================
# Figure 2 : proper motion vs RA  (model in footprint vs observed)
# ==============================================================================
print("Rendering proper-motion comparison ...")
fig, axs = plt.subplots(2, 1, figsize=(10, 8.5), dpi=140, sharex=True)
fig.patch.set_facecolor(BG)
pm_keys = [("pmra", r"$\mu_{\alpha}\cos\delta$ [mas/yr]", obs_pmra),
           ("pmdec", r"$\mu_{\delta}$ [mas/yr]", obs_pmdec)]
for ax, (key, ylab, obs_arr) in zip(axs, pm_keys):
    ax.set_facecolor(BG); ax.grid(True, alpha=0.2, lw=0.5)
    for tag, col, lbl in [("df", "#1B7F2D", "model (DF)"),
                          ("nodf", "#C0392B", "model (no DF)")]:
        s = models[tag]["stream"]; m = s["mask"]
        ax.scatter(s["ra"][m], s[key][m], s=6, c=col, alpha=0.45,
                   linewidths=0, rasterized=True, label=lbl)
    ax.scatter(obs_ra, obs_arr, s=14, c=C_OBS, marker=".",
               label="LMS-1 (STREAMFINDER)", zorder=5)
    ax.set_ylabel(ylab, fontsize=10)
axs[-1].set_xlabel("RA [deg]", fontsize=10)
axs[0].set_xlim(min(obs_ra.min(), 195) - 5, max(obs_ra.max(), 275) + 5)
# shared legend outside, to the right
h, lab = axs[0].get_legend_handles_labels()
fig.legend(h, lab, fontsize=9, loc="center left", bbox_to_anchor=(0.83, 0.5),
           markerscale=2, framealpha=0.9)
fig.suptitle("LMS-1 proper motion: N-body debris in observed footprint vs STREAMFINDER",
             fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 0.82, 1])
out = save_fig_both(fig, "lms1gc_pm_compare.png", bbox_inches="tight", dpi=140,
                    facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  saved: {out}")


# ==============================================================================
# Figure 3 : v_los vs RA  (model in footprint vs observed-with-velocity)
# ==============================================================================
print("Rendering line-of-sight velocity comparison ...")
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=140)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG); ax.grid(True, alpha=0.2, lw=0.5)
for tag, col, lbl in [("df", "#1B7F2D", "model (DF)"),
                      ("nodf", "#C0392B", "model (no DF)")]:
    s = models[tag]["stream"]; m = s["mask"]
    ax.scatter(s["ra"][m], s["vlos"][m], s=6, c=col, alpha=0.45,
               linewidths=0, rasterized=True, label=lbl)
ax.scatter(obs_v["RAdeg"], obs_v["VHel"], s=22, c=C_OBS, marker="o",
           edgecolors="k", linewidths=0.4,
           label=f"LMS-1 v_los (STREAMFINDER, N={len(obs_v)})", zorder=5)
ax.set_xlabel("RA [deg]", fontsize=10)
ax.set_ylabel(r"$v_{\rm los}$  (heliocentric) [km/s]", fontsize=10)
ax.set_xlim(min(obs_ra.min(), 195) - 5, max(obs_ra.max(), 275) + 5)
ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5),
          markerscale=2, framealpha=0.9)
fig.suptitle("LMS-1 line-of-sight velocity: N-body debris in footprint vs STREAMFINDER",
             fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 0.78, 1])
out = save_fig_both(fig, "lms1gc_vlos_compare.png", bbox_inches="tight", dpi=140,
                    facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  saved: {out}")


# ==============================================================================
# Quantitative comparison + verdict
# ==============================================================================
def stat(a):
    a = np.asarray(a, float)
    return np.nanmedian(a), np.nanpercentile(a, 16), np.nanpercentile(a, 84)


lines = []
lines.append("LMS-1 N-body final snapshot vs STREAMFINDER (Ibata+2024)")
lines.append("=" * 64)
lines.append(f"observed members : {len(obs)}   (with v_los: {len(obs_v)})")
lines.append("footprint = convex hull of observed RA/Dec, dilated 15%")
lines.append("model statistics use only debris INSIDE that footprint.")
lines.append("")

# observed reference
o_pmra = stat(obs_pmra); o_pmdec = stat(obs_pmdec); o_vlos = stat(obs_v["VHel"])
lines.append("OBSERVED  (median [16-84 pct]):")
lines.append(f"  pmRA = {o_pmra[0]:+.3f} [{o_pmra[1]:+.3f}, {o_pmra[2]:+.3f}] mas/yr")
lines.append(f"  pmDE = {o_pmdec[0]:+.3f} [{o_pmdec[1]:+.3f}, {o_pmdec[2]:+.3f}] mas/yr")
lines.append(f"  vlos = {o_vlos[0]:+.1f} [{o_vlos[1]:+.1f}, {o_vlos[2]:+.1f}] km/s")
lines.append("")

for tag in ("df", "nodf"):
    s = models[tag]["stream"]; m = s["mask"]
    if m.sum() == 0:
        lines.append(f"MODEL [{tag}] : no debris inside footprint."); continue
    m_pmra = stat(s["pmra"][m]); m_pmdec = stat(s["pmdec"][m])
    m_vlos = stat(s["vlos"][m]); m_dist = stat(s["dist"][m])
    lines.append(f"MODEL [{tag.upper()}]  (N_in={m.sum()}, median [16-84 pct]):")
    lines.append(f"  pmRA = {m_pmra[0]:+.3f} [{m_pmra[1]:+.3f}, {m_pmra[2]:+.3f}] mas/yr"
                 f"   (Delta_med = {m_pmra[0]-o_pmra[0]:+.3f})")
    lines.append(f"  pmDE = {m_pmdec[0]:+.3f} [{m_pmdec[1]:+.3f}, {m_pmdec[2]:+.3f}] mas/yr"
                 f"   (Delta_med = {m_pmdec[0]-o_pmdec[0]:+.3f})")
    lines.append(f"  vlos = {m_vlos[0]:+.1f} [{m_vlos[1]:+.1f}, {m_vlos[2]:+.1f}] km/s"
                 f"   (Delta_med = {m_vlos[0]-o_vlos[0]:+.1f})")
    lines.append(f"  dist = {m_dist[0]:.1f} [{m_dist[1]:.1f}, {m_dist[2]:.1f}] kpc")
    lines.append("")

txt = "\n".join(lines)
print("\n" + txt)
for _tag in ("df", "nodf"):
    with open(out_path("obs_comparison_summary.txt", _tag), "w") as f:
        f.write(txt + "\n")
print(f"\n  saved: obs_comparison_summary.txt  ->  mass_{MASS_TAG}_{{df,nodf}}/")
print("\nAll done.")
