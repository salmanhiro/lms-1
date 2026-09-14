"""Helpers for the NGC 5024 (M53) / LMS-1 stream-selection notebook.

Two halves:

* **Cluster-agnostic building blocks** -- coordinates and the great-circle
  (phi1, phi2) frame, photometry / isochrones / the stellar locus, the
  Baumgardt + Harris catalogue readers, the DESI cross-match, stream tracks and
  the kinematic matched filter, CMD / Hess diagnostics, proper-motion
  Gaussian-mixture membership, and sky maps. These came from the ``streamcutter``
  package (``DESI/Streamcutter/src/streamcutter/``) and are unchanged, so fixes
  can still be carried across in either direction.
* **The NGC 5024 pipeline steps** at the bottom, one per selection stage. They
  take the cluster parameters and cut values as arguments -- the notebook's
  CONFIG cell owns those -- so nothing here reads a notebook global.

Used as ``from utils import *`` by
``select_ngc5024_stream_candidates.ipynb``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from astropy.table import Table, vstack, Column
from astropy.coordinates import SkyCoord
from astropy import units as u


def legend_outside(ax, where="right", pad_pt=8, handles=None, labels=None,
                   **kw):
    """Draw ``ax``'s legend outside the axes.

    ``where="right"`` hangs it to the right of the axes and of anything sitting
    beside it on the same row (e.g. its colorbar); ``where="below"`` hangs it
    under the x tick labels -- use that for a panel with another panel to its
    right. The offsets are measured on the drawn figure, so call this once the
    layout is final (after ``tight_layout`` / ``set_position``). Inline figures
    are rendered with ``bbox_inches="tight"``, so the canvas grows to fit it.
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    tight = ax.get_tightbbox(renderer)
    pad = pad_pt * fig.dpi / 72.0
    to_axes = ax.transAxes.inverted()
    if where == "right":
        x1 = tight.x1
        for other in fig.axes:
            if other is ax:
                continue
            ob = other.get_window_extent(renderer)
            if ob.x0 >= box.x1 - 1 and ob.y1 > box.y0 and ob.y0 < box.y1:
                x1 = max(x1, other.get_tightbbox(renderer).x1)
        x = to_axes.transform((x1 + pad, box.y1))[0]
        return ax.legend(handles, labels, loc="upper left",
                         bbox_to_anchor=(x, 1.0), borderaxespad=0.0, **kw)
    y = to_axes.transform((box.x0, tight.y0 - pad))[1]
    return ax.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5, y), borderaxespad=0.0, **kw)


# ===========================================================================
# coordinates: great-circle frame + (phi1, phi2) transforms
# ===========================================================================
def sph_to_cart(ra_deg, dec_deg):
    """Unit position vectors (N, 3) from RA/Dec in degrees."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T


def tangent_basis_icrs(ra_deg, dec_deg):
    """Unit vectors e_alpha (increasing RA) and e_delta (increasing Dec) in ICRS."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    e_a = np.vstack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)]).T
    e_d = np.vstack([-np.cos(ra) * np.sin(dec),
                     -np.sin(ra) * np.sin(dec),
                     np.cos(dec)]).T
    return e_a, e_d


def _rotate_vecs(v, ex, ey, ez):
    """Components of vectors ``v`` along the stream basis (ex, ey, ez)."""
    return np.vstack([v @ ex, v @ ey, v @ ez]).T


def icrs_to_phi12(ra_deg, dec_deg, ex, ey, ez):
    """Return (phi1, phi2) in degrees for ICRS positions in the stream frame."""
    r = sph_to_cart(np.asarray(ra_deg, float), np.asarray(dec_deg, float))
    x = r @ ex
    y = r @ ey
    z = r @ ez
    phi1 = np.degrees(np.arctan2(y, x))
    phi2 = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return phi1, phi2


def icrs_to_phi12_and_pm(ra_deg, dec_deg, pm_ra_cosdec, pm_dec, ex, ey, ez):
    """Transform positions and proper motions into the stream frame.

    ``pm_ra_cosdec`` is mu_alpha* = mu_alpha cos(dec). Returns
    ``phi1, phi2, mu_phi1_cosphi2, mu_phi2`` (pm units unchanged).
    """
    ra_deg = np.asarray(ra_deg, float)
    dec_deg = np.asarray(dec_deg, float)
    pm_ra_cosdec = np.asarray(pm_ra_cosdec, float)
    pm_dec = np.asarray(pm_dec, float)

    r = sph_to_cart(ra_deg, dec_deg)
    e_a, e_d = tangent_basis_icrs(ra_deg, dec_deg)

    # angular velocity vector on the unit sphere in the orthonormal ICRS basis
    rdot = pm_ra_cosdec[:, None] * e_a + pm_dec[:, None] * e_d

    r_p = _rotate_vecs(r, ex, ey, ez)
    rdot_p = _rotate_vecs(rdot, ex, ey, ez)

    xp, yp, zp = r_p[:, 0], r_p[:, 1], r_p[:, 2]
    phi1 = np.degrees(np.arctan2(yp, xp))
    phi2 = np.degrees(np.arcsin(np.clip(zp, -1.0, 1.0)))

    phi1r = np.deg2rad(phi1)
    phi2r = np.deg2rad(phi2)
    e_phi1 = np.vstack([-np.sin(phi1r), np.cos(phi1r), np.zeros_like(phi1r)]).T
    e_phi2 = np.vstack([-np.cos(phi1r) * np.sin(phi2r),
                        -np.sin(phi1r) * np.sin(phi2r),
                        np.cos(phi2r)]).T

    mu_phi1_cosphi2 = np.sum(rdot_p * e_phi1, axis=1)
    mu_phi2 = np.sum(rdot_p * e_phi2, axis=1)
    return phi1, phi2, mu_phi1_cosphi2, mu_phi2


def great_circle_frame(stream_ra, stream_dec, origin_ra=None, origin_dec=None):
    """Build a stream ``(ex, ey, ez)`` frame from a set of stream points.

    The great-circle pole ``ez`` is the smallest principal axis of the points'
    unit vectors (the normal to the best-fit plane through them). The origin
    ``ex`` (phi1 = 0) is placed at ``(origin_ra, origin_dec)`` if given, else at
    the mean stream position, projected into the plane. ``ez`` is oriented so
    that +phi2 points roughly toward increasing Dec at the origin.
    """
    pts = sph_to_cart(np.asarray(stream_ra, float), np.asarray(stream_dec, float))

    # pole = eigenvector of the smallest eigenvalue of the scatter matrix
    _, vecs = np.linalg.eigh(pts.T @ pts)
    ez = vecs[:, 0]
    ez = ez / np.linalg.norm(ez)

    if origin_ra is not None and origin_dec is not None:
        origin = sph_to_cart(np.array([origin_ra]), np.array([origin_dec]))[0]
    else:
        origin = pts.mean(axis=0)

    # orient +phi2 toward local North at the origin (nicety, sign is otherwise free)
    _, e_d = tangent_basis_icrs(np.atleast_1d(
        np.degrees(np.arctan2(origin[1], origin[0]))),
        np.atleast_1d(np.degrees(np.arcsin(np.clip(origin[2], -1, 1)))))
    if ez @ e_d[0] < 0:
        ez = -ez

    # ex = origin projected into the plane perpendicular to ez
    ex = origin - (origin @ ez) * ez
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey = ey / np.linalg.norm(ey)
    ex = np.cross(ey, ez)  # re-orthogonalise
    ex = ex / np.linalg.norm(ex)
    return ex, ey, ez


# ===========================================================================
# photometry: Tractor fluxes -> dereddened mags; isochrones; stellar locus
# ===========================================================================
def mag(t, band: str) -> np.ndarray:
    """Dereddened AB magnitude from Tractor nanomaggie fluxes.

    ``mag = 22.5 - 2.5*log10(flux / mw_transmission)`` for ``flux_{band}`` and
    ``mw_transmission_{band}`` columns.
    """
    flux = np.asarray(t[f"flux_{band}"], float)
    trans = np.asarray(t[f"mw_transmission_{band}"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 22.5 - 2.5 * np.log10(flux / trans)


def mist_isochrone(age_gyr, feh, filters=("DECam_g", "DECam_r"),
                   n_mass=10000, mass_range=(0.1, 100)):
    """Interpolate one MIST isochrone with ``minimint``.

    Returns ``(iso, iso_mask, g_abs, r_abs)``: the raw ``minimint`` table, the
    finite-photometry mask, and the two absolute-magnitude arrays already
    masked -- what :func:`select_around_isochrone` and the CMD figures want.
    """
    import minimint

    interp = minimint.Interpolator(list(filters))
    massgrid = 10 ** np.linspace(np.log10(mass_range[0]),
                                 np.log10(mass_range[1]), n_mass)
    iso = interp(massgrid, np.log10(age_gyr * 1e9), feh)
    iso_mask = np.isfinite(iso[filters[0]]) & np.isfinite(iso[filters[1]])
    return (iso, iso_mask,
            np.asarray(iso[filters[0]])[iso_mask],
            np.asarray(iso[filters[1]])[iso_mask])

# --------------------------------------------------------------------------- #
# Stellar locus: the (g-z) vs (g-r) point-source colour-colour sequence
# --------------------------------------------------------------------------- #
# Reference locus for DECaLS-like grz photometry, ``(g-z) = 1.7*(g-r) - 0.17``
# (the linear regime of the point-source stellar locus). Real stars scatter
# tightly about it; unresolved galaxies, blends and objects with a bad band sit
# off it, so a cut on the perpendicular-ish residual cleans the catalogue before
# any CMD work. Per-cluster notebooks under ``notebooks/`` prefer to *fit* the
# slope/intercept on cluster members (see :func:`fit_stellar_locus`), which
# absorbs small photometric-zeropoint offsets; these are the fallback values.
LOCUS_SLOPE_REF = 1.7
LOCUS_INTERCEPT_REF = -0.17


def locus_delta(gr, gz, slope=LOCUS_SLOPE_REF, intercept=LOCUS_INTERCEPT_REF):
    """Colour residual from the stellar locus: ``(g-z) - [slope*(g-r) + b]``."""
    with np.errstate(invalid="ignore"):
        return np.asarray(gz, float) - (slope * np.asarray(gr, float) + intercept)


def fit_stellar_locus(gr, gz, gmag=None, gr_range=(0.1, 1.2), gmax=21.0):
    """Least-squares fit of ``(g-z) = slope*(g-r) + intercept``.

    Fit it on a clean calibration sample -- e.g. stars inside a small aperture on
    the cluster, which are nearly all real point sources at one distance. Only
    stars with ``gr_range[0] < g-r < gr_range[1]`` (the linear regime of the
    locus) and ``g < gmax`` (where the photometry is reliable) are used.

    Returns
    -------
    slope, intercept, n_fit
    """
    from scipy.stats import linregress

    gr = np.asarray(gr, float)
    gz = np.asarray(gz, float)
    m = np.isfinite(gr) & np.isfinite(gz)
    lo, hi = gr_range
    if lo is not None:
        m &= gr > lo
    if hi is not None:
        m &= gr < hi
    if gmag is not None and gmax is not None:
        m &= np.asarray(gmag, float) < gmax
    if m.sum() < 10:
        raise ValueError(f"only {int(m.sum())} stars in the locus fit sample")
    slope, intercept = linregress(gr[m], gz[m])[:2]
    return float(slope), float(intercept), int(m.sum())


def select_stellar_locus(gr, gz, gmag, slope=LOCUS_SLOPE_REF,
                         intercept=LOCUS_INTERCEPT_REF, tol=0.1,
                         g_lim=(16.0, 22.0), gr_min=0.1, gr_max=1.2,
                         return_delta=False):
    """Keep point sources within ``+/- tol`` mag of the stellar locus.

    On top of the locus band this applies the g-magnitude window ``g_lim``
    (saturation .. survey depth) and the colour window ``gr_min .. gr_max``,
    outside which the locus stops being linear. Either colour limit may be
    ``None`` to leave that side open.
    """
    gmag = np.asarray(gmag, float)
    delta = locus_delta(gr, gz, slope, intercept)
    keep = np.isfinite(delta) & np.isfinite(gmag) & (np.abs(delta) <= tol)
    gr_arr = np.asarray(gr, float)
    if gr_min is not None:
        keep &= gr_arr >= gr_min
    if gr_max is not None:
        keep &= gr_arr <= gr_max
    if g_lim is not None:
        lo, hi = g_lim
        if lo is not None:
            keep &= gmag >= lo
        if hi is not None:
            keep &= gmag < hi
    if return_delta:
        return keep, delta
    return keep


def figure_stellar_locus(gr, gz, delta, slope, intercept, tol, label,
                         gr_min=0.1, gr_max=1.2, locus_source="reference",
                         nbins=(200, 200)):
    """Diagnostic for the locus cut: colour-colour density + residual histogram.

    Left: every star in ``(g-r, g-z)`` with the fitted locus and its ``+/- tol``
    band. Right: the residual distribution with the same band marked -- real
    point sources make the narrow spike, the tail is what the cut removes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    gr = np.asarray(gr, float)
    gz = np.asarray(gz, float)
    delta = np.asarray(delta, float)
    ok = np.isfinite(gr) & np.isfinite(gz)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    H, xe, ye = np.histogram2d(gr[ok], gz[ok], bins=nbins,
                               range=[[-0.5, 2.5], [-1.0, 4.5]])
    pos = H[H > 0]
    im = ax.imshow(H.T, origin="lower", aspect="auto", cmap="Greys",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   norm=LogNorm(vmin=1, vmax=max(np.percentile(pos, 99.5), 2)
                                if pos.size else 2))
    fig.colorbar(im, ax=ax, shrink=0.85, label="stars / bin")
    xs = np.linspace(xe[0] if gr_min is None else gr_min,
                     xe[-1] if gr_max is None else gr_max, 100)
    ax.plot(xs, slope * xs + intercept, c="crimson", lw=2,
            label=f"locus: $g-z$ = {slope:.3f}$(g-r)$ {intercept:+.3f}")
    ax.plot(xs, slope * xs + intercept - tol, c="crimson", lw=1.2, ls="--",
            label=f"$\\pm$ {tol:.2f} mag")
    ax.plot(xs, slope * xs + intercept + tol, c="crimson", lw=1.2, ls="--")
    if gr_min is not None:
        ax.axvline(gr_min, c="navy", lw=1.2, ls=":",
                   label=f"{gr_min} $\\leq (g-r)_0$")
    if gr_max is not None:
        ax.axvline(gr_max, c="navy", lw=1.2, ls=":",
                   label=f"$(g-r)_0 \\leq$ {gr_max}")
    ax.set_xlabel("$(g - r)_0$")
    ax.set_ylabel("$(g - z)_0$")
    ax.set_title(f"{label}: stellar locus ({locus_source})")

    ax = axes[1]
    ax.hist(delta[np.isfinite(delta)], bins=200, range=(-0.5, 0.5),
            color="0.6")
    ax.axvline(-tol, ls="--", c="crimson")
    ax.axvline(+tol, ls="--", c="crimson")
    ax.set_xlabel(r"$\Delta_{locus} = (g-z)_0 - [$slope$\,(g-r)_0 + b]$")
    ax.set_ylabel("stars")
    ax.set_title(f"residual about the locus ($\\pm$ {tol:.2f} mag kept)")

    fig.tight_layout()
    legend_outside(axes[0], where="below", fontsize=9, ncol=2)
    return fig


# ===========================================================================
# catalogues: cluster parameters from Baumgardt / Harris
# ===========================================================================
def _norm(name: str) -> str:
    """Normalise a cluster name for matching: lower-case, spaces->underscores."""
    return str(name).strip().lower().replace(" ", "_")


def baumgardt_row(name: str, ecsv_file) -> dict:
    """Return the Baumgardt catalog row for ``name`` as a plain dict."""
    t = Table.read(ecsv_file)
    key = _norm(name)
    # match against either the raw or normalised cluster name column
    cols = [c for c in ("Cluster", "Cluster_norm") if c in t.colnames]
    for col in cols:
        hit = np.where([_norm(c) == key for c in t[col]])[0]
        if len(hit):
            return {c: t[c][hit[0]] for c in t.colnames}
    raise KeyError(f"{name!r} not found in Baumgardt catalog {ecsv_file}")


def harris_feh(name: str, dat_file) -> float:
    """Return [Fe/H] for ``name`` from the Harris ``mwgc.dat`` Part II block."""
    text = Path(dat_file).read_text().splitlines()
    # isolate the Part II (metallicity/photometry) section
    start = next(i for i, ln in enumerate(text) if "Part II" in ln)
    end = next(i for i, ln in enumerate(text) if "Part III" in ln)

    target = _norm(name)
    for ln in text[start:end]:
        s = ln.strip()
        if not s:
            continue
        # data rows start with the cluster ID; match the longest leading ID
        # that normalises to the requested name (handles "NGC 5904" etc.)
        toks = s.split()
        for ntok in (2, 1):  # try two-word then one-word IDs
            if len(toks) > ntok and _norm(" ".join(toks[:ntok])) == target:
                try:
                    return float(toks[ntok])
                except ValueError:
                    return np.nan
    raise KeyError(f"{name!r} not found in Harris catalog {dat_file}")


def load_gc_params(name, baumgardt_file, harris_file, harris_name=None) -> dict:
    """Gather the physical M5-style cluster parameters from both catalogs.

    Parameters
    ----------
    name : str
        Cluster name as it appears in the Baumgardt catalog (e.g. ``"NGC_5904"``).
    baumgardt_file, harris_file : path-like
        Paths to ``mw_gc_parameters_*.ecsv`` and ``mwgc.dat``.
    harris_name : str, optional
        Name to look up in Harris if it differs from ``name`` (defaults to
        ``name`` with underscores turned into spaces).

    Returns
    -------
    dict with keys RA, DEC, DIST, VLOS, PMRA, PMDEC (the ``GC`` mapping the
    plotting/matched-filter routines expect) plus RT_PC and FEH.
    """
    b = baumgardt_row(name, baumgardt_file)
    feh = harris_feh(harris_name or str(name).replace("_", " "), harris_file)
    return {
        "RA": float(b["RA"]),
        "DEC": float(b["DEC"]),
        "DIST": float(b["Rsun"]),
        "VLOS": float(b["<RV>"]),
        "PMRA": float(b["mualpha"]),
        "PMDEC": float(b["mu_delta"]),
        "RT_PC": float(b["rt"]),
        "FEH": float(feh),
    }


# ===========================================================================
# crossmatch: DESI MWS RV / [Fe/H]
# ===========================================================================
def crossmatch_desi(candidates, desi_mws,
                    id_col="ref_id", desi_id_col="REF_ID",
                    copy_cols=("VRAD_CORRECTED", "VRAD_ERR",
                               "FEH_CORRECTED", "FEH_ERR"),
                    add_aliases=True):
    """Match candidates to the DESI MWS catalogue on Gaia ref_id and copy spectra.

    Returns a copy of ``candidates`` with the requested ``copy_cols`` (NaN where
    unmatched), convenience aliases (RV/RV_ERR/FEH/FEH_ERR), and a boolean
    ``has_desi`` column.
    """
    cand = Table(candidates).copy()
    dcols = desi_mws.colnames

    ref = np.asarray(cand[id_col]).astype(np.int64)
    tid = np.asarray(desi_mws[desi_id_col]).astype(np.int64)

    order = np.argsort(tid)
    tid_sorted = tid[order]
    pos = np.clip(np.searchsorted(tid_sorted, ref), 0, len(tid_sorted) - 1)
    matched = tid_sorted[pos] == ref
    didx = order[pos]

    n = len(cand)
    for col in copy_cols:
        if col not in dcols:
            print(f"  [warn] '{col}' not in DESI file")
            continue
        vals = np.asarray(desi_mws[col], float)[didx]
        out = np.full(n, np.nan)
        out[matched] = vals[matched]
        cand[col] = out

    if add_aliases:
        alias = {"RV": "VRAD_CORRECTED", "RV_ERR": "VRAD_ERR",
                 "FEH": "FEH_CORRECTED", "FEH_ERR": "FEH_ERR"}
        for new, src in alias.items():
            if src in cand.colnames:
                cand[new] = cand[src]

    cand["has_desi"] = matched
    print(f"[crossmatch] {matched.sum()} / {n} matched; "
          f"copied {[c for c in copy_cols if c in dcols]}")
    return cand


# ===========================================================================
# stream: tracks, PM matched filter, density profiles
# ===========================================================================
def binned_track(x, y, nbins=60, min_count=5):
    """Median ``y`` in equal-width ``x`` bins; only bins with enough points."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    xc, yc = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= min_count:
            xc.append(x[m].mean())
            yc.append(np.median(y[m]))
    return np.array(xc), np.array(yc)


def binned_band(x, y, nbins=60, qlo=16, qhi=84, min_count=5):
    """Per-bin median plus a percentile envelope where the data is dense."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    xc, mid, lo, hi = [], [], [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= min_count:
            xc.append(x[m].mean())
            mid.append(np.median(y[m]))
            lo.append(np.percentile(y[m], qlo))
            hi.append(np.percentile(y[m], qhi))
    return np.array(xc), np.array(mid), np.array(lo), np.array(hi)


def extend_track(xr, mid, ra_lim, poly_deg=3):
    """Fit a polynomial to the reliable track and evaluate over the full RA range."""
    ra_full = np.linspace(min(ra_lim), max(ra_lim), 500)
    if len(xr) >= poly_deg + 1:
        mid_full = np.polyval(np.polyfit(xr, mid, poly_deg), ra_full)
    else:
        mid_full = np.interp(ra_full, xr, mid, left=mid[0], right=mid[-1])
    return ra_full, mid_full


def pm_profiles(stream_tab, pmra_has_cosdec=True):
    """Return callables mu_a*(RA), mu_d(RA) from the simulated stream."""
    ra = np.asarray(stream_tab["RA"])
    dec = np.asarray(stream_tab["DEC"])
    pmra = np.asarray(stream_tab["PMRA"])
    pmde = np.asarray(stream_tab["PMDEC"])
    pmra_star = pmra if pmra_has_cosdec else pmra * np.cos(np.radians(dec))
    xa, ya = binned_track(ra, pmra_star)
    xd, yd = binned_track(ra, pmde)
    f_a = lambda rr: np.interp(rr, xa, ya, left=ya[0], right=ya[-1])
    f_d = lambda rr: np.interp(rr, xd, yd, left=yd[0], right=yd[-1])
    return f_a, f_d


def plot_stream(tab, gc, pmra_has_cosdec=True, ra_lim=None, ra_pad=5.0,
                style="line", lw=1.0, point_size=1.0):
    """Diagnostic multi-panel plot of the simulated stream vs RA.

    ``gc`` is a mapping with keys RA, DEC, DIST, VLOS, PMRA, PMDEC.
    """
    ra = np.asarray(tab["RA"])
    dec = np.asarray(tab["DEC"])
    dist = np.asarray(tab["DIST"])
    vlos = np.asarray(tab["VLOS"])
    pmra = np.asarray(tab["PMRA"])
    pmde = np.asarray(tab["PMDEC"])
    pmra_star = pmra if pmra_has_cosdec else pmra * np.cos(np.radians(dec))

    if ra_lim is None:
        ra_lim = (ra.min() - ra_pad, ra.max() + ra_pad)

    panels = [
        (dec, r"dec ($\degree$)", gc["DEC"]),
        (dist, r"d (kpc)", gc["DIST"]),
        (vlos, r"$v_{\rm hel}$", gc["VLOS"]),
        (pmra_star, r"$\mu_\alpha\cos(\delta)$ (mas/yr)", gc["PMRA"]),
        (pmde, r"$\mu_\delta$ (mas/yr)", gc["PMDEC"]),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 9), sharex=True)
    fig.subplots_adjust(hspace=0.0, left=0.12, right=0.97, top=0.98, bottom=0.07)

    order = np.argsort(ra)
    for ax, (y, ylab, gcy) in zip(axes, panels):
        if style == "line":
            ax.plot(ra[order], y[order], "-", c="k", lw=lw)
        elif style == "track":
            xt, yt = binned_track(ra, y, min_count=3)
            ax.plot(xt, yt, "-", c="k", lw=lw)
        else:
            ax.scatter(ra, y, s=point_size, c="k", lw=0, rasterized=True)
        ax.scatter([gc["RA"]], [gcy], s=140, facecolors="none",
                   edgecolors="k", lw=1.2, zorder=5)
        ax.set_ylabel(ylab)
        ax.set_xlim(*ra_lim)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.invert_xaxis()

    axes[-1].set_xlabel(r"R.A. ($\degree$ J2000)")
    return fig


def pm_weight(obs_tab, stream_tab,
              obs_cols=("ra", "dec", "pmra", "pmdec"),
              obs_pmra_has_cosdec=True, stream_pmra_has_cosdec=True,
              pmra_err=None, pmdec_err=None, pmra_ivar=None, pmdec_ivar=None):
    """Grillmair (2009, eq. 1) kinematic weight for each star.

    A Gaussian in the proper-motion offset between the star and the predicted
    stream PM at its RA, divided by the per-star PM uncertainty. Returns an
    array in [0, 1]; ~1 means PM consistent with the stream.
    """
    rc, dc, pac, pdc = obs_cols
    ra = np.asarray(obs_tab[rc], float)
    dec = np.asarray(obs_tab[dc], float)
    pma = np.asarray(obs_tab[pac], float)
    pmd = np.asarray(obs_tab[pdc], float)
    if not obs_pmra_has_cosdec:
        pma = pma * np.cos(np.radians(dec))

    f_a, f_d = pm_profiles(stream_tab, stream_pmra_has_cosdec)
    da = pma - f_a(ra)
    dd = pmd - f_d(ra)

    if pmra_ivar is not None:
        sa = 1.0 / np.sqrt(np.asarray(obs_tab[pmra_ivar], float))
    elif pmra_err is not None:
        sa = np.asarray(obs_tab[pmra_err], float)
    else:
        raise ValueError("provide pmra_ivar or pmra_err")
    if pmdec_ivar is not None:
        sd = 1.0 / np.sqrt(np.asarray(obs_tab[pmdec_ivar], float))
    elif pmdec_err is not None:
        sd = np.asarray(obs_tab[pmdec_err], float)
    else:
        raise ValueError("provide pmdec_ivar or pmdec_err")

    return np.exp(-0.5 * ((da / sa) ** 2 + (dd / sd) ** 2))


def matched_filter_map(obs_tab, stream_tab, gc, gc_label="GC",
                       obs_cols=("ra", "dec", "pmra", "pmdec"),
                       obs_pmra_has_cosdec=True, stream_pmra_has_cosdec=True,
                       pmra_err=None, pmdec_err=None,
                       pmra_ivar=None, pmdec_ivar=None,
                       pix=0.2, smooth_deg=0.2, ra_lim=None, dec_lim=None,
                       vmin_pct=50, show_track=True, show_band=True,
                       band_pct=(16, 84), band_width_deg=None,
                       track_poly_deg=3, show_contour=False, return_mask=False,
                       overlay_tab=None, overlay_cols=("ra", "dec"),
                       overlay_color=None, overlay_cmap="coolwarm",
                       overlay_clim=None, overlay_size=45, overlay_label="DESI",
                       cmap="gray"):
    """Weight each star by a Gaussian in its PM offset from the predicted stream
    PM at its RA, then make a smoothed weighted density map (Grillmair eq. 1).

    ``gc`` is a mapping with keys RA, DEC; ``gc_label`` annotates the marker.
    """
    rc, dc, pac, pdc = obs_cols
    ra = np.asarray(obs_tab[rc], float)
    dec = np.asarray(obs_tab[dc], float)

    # kinematic weight (Grillmair eq. 1)
    w = pm_weight(obs_tab, stream_tab, obs_cols=obs_cols,
                  obs_pmra_has_cosdec=obs_pmra_has_cosdec,
                  stream_pmra_has_cosdec=stream_pmra_has_cosdec,
                  pmra_err=pmra_err, pmdec_err=pmdec_err,
                  pmra_ivar=pmra_ivar, pmdec_ivar=pmdec_ivar)

    n_eff = w.sum() ** 2 / np.sum(w ** 2)
    print(f"[matched filter] N_stars={w.size}  N_eff={n_eff:.0f}  "
          f"(N_eff/N={n_eff / w.size:.3f})")
    if n_eff / w.size > 0.5:
        print("  -> weights ~uniform: sigma too large, PM filter barely "
              "cutting field stars.")

    # stream track + band from the simulation
    sra = np.asarray(stream_tab["RA"], float)
    sde = np.asarray(stream_tab["DEC"], float)
    xr, mid, lo_p, hi_p = binned_band(sra, sde, qlo=band_pct[0],
                                      qhi=band_pct[1], min_count=5)

    if ra_lim is None:
        ra_lim = (ra.min(), ra.max())
    if dec_lim is None:
        dec_lim = (dec.min(), dec.max())

    # The simulated stream spans only part of the footprint. Restrict the drawn
    # track + band (and the in-band selection) to that RA extent rather than
    # extrapolating a polynomial across the whole map -- otherwise the band runs
    # the full width, diverges from the (clamped) selection corridor, and PM
    # members appear to "stray" outside the drawn band.
    ra_s_lo, ra_s_hi = float(np.nanmin(xr)), float(np.nanmax(xr))
    ra_draw = np.linspace(ra_s_lo, ra_s_hi, 400)
    mid_draw = np.interp(ra_draw, xr, mid)

    # in-band corridor for the membership mask
    if band_width_deg is not None:
        mid_at = np.interp(ra, xr, mid, left=mid[0], right=mid[-1])
        lo_at = mid_at - band_width_deg
        hi_at = mid_at + band_width_deg
    else:
        lo_at = np.interp(ra, xr, lo_p, left=lo_p[0], right=lo_p[-1])
        hi_at = np.interp(ra, xr, hi_p, left=hi_p[0], right=hi_p[-1])
    in_band = ((dec >= lo_at) & (dec <= hi_at) &
               (ra >= ra_s_lo) & (ra <= ra_s_hi))
    print(f"[matched filter] stars inside band = {in_band.sum()}/{in_band.size}"
          f"  (sum w inside = {w[in_band].sum():.0f} of {w.sum():.0f})")

    # 2-D weighted density map
    nx = int(round(abs(ra_lim[1] - ra_lim[0]) / pix))
    ny = int(round(abs(dec_lim[1] - dec_lim[0]) / pix))
    H, xe, ye = np.histogram2d(ra, dec, bins=[nx, ny],
                               range=[sorted(ra_lim), sorted(dec_lim)],
                               weights=w)
    H = gaussian_filter(H, sigma=smooth_deg / pix)

    fig, ax = plt.subplots(figsize=(11, 5))
    pos = H[H > 0]
    vmin = np.percentile(pos, vmin_pct) if pos.size else 1e-3
    im = ax.imshow(H.T, origin="lower", aspect="auto", cmap=cmap,
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   norm=LogNorm(vmin=max(vmin, 1e-3), vmax=H.max()))

    if show_band:
        if band_width_deg is not None:
            ax.fill_between(ra_draw, mid_draw - band_width_deg,
                            mid_draw + band_width_deg, color="cyan",
                            alpha=0.22, lw=0,
                            label=f"predicted stream +/-{band_width_deg:.2f} deg")
        else:
            sel = (xr >= ra_s_lo) & (xr <= ra_s_hi)
            ax.fill_between(xr[sel], lo_p[sel], hi_p[sel], color="cyan",
                            alpha=0.22, lw=0,
                            label=f"stream {band_pct[0]}-{band_pct[1]}%")

    if show_track:
        ax.plot(ra_draw, mid_draw, "--", c="cyan", lw=1.5, alpha=0.9,
                label="predicted track")

    # mark the RA extent actually covered by the simulated stream
    for _j, _rs in enumerate((ra_s_lo, ra_s_hi)):
        ax.axvline(_rs, color="cyan", ls=":", lw=1.1, alpha=0.85,
                   label="stream extent" if _j == 0 else None)


    if show_contour:
        Hs, xse, yse = np.histogram2d(sra, sde, bins=[nx, ny],
                                      range=[sorted(ra_lim), sorted(dec_lim)])
        Hs = gaussian_filter(Hs, sigma=smooth_deg / pix)
        xs = 0.5 * (xse[:-1] + xse[1:])
        ys = 0.5 * (yse[:-1] + yse[1:])
        levels = np.percentile(Hs[Hs > 0], [70, 90, 98])
        ax.contour(xs, ys, Hs.T, levels=levels, colors="cyan",
                   linewidths=0.8, alpha=0.8)

    if overlay_tab is not None:
        ora = np.asarray(overlay_tab[overlay_cols[0]], float)
        ode = np.asarray(overlay_tab[overlay_cols[1]], float)
        if overlay_color is not None:
            cval = np.asarray(overlay_tab[overlay_color], float)
            good = np.isfinite(cval)
            vmn, vmx = overlay_clim if overlay_clim else (None, None)
            sc = ax.scatter(ora[good], ode[good], c=cval[good], s=overlay_size,
                            cmap=overlay_cmap, vmin=vmn, vmax=vmx,
                            edgecolors="k", linewidths=0.6, zorder=6,
                            label=f"{overlay_label} ({overlay_color})")
            fig.colorbar(sc, ax=ax, pad=0.02, label=overlay_color)
        else:
            ax.scatter(ora, ode, s=overlay_size, facecolor="lime",
                       edgecolors="k", linewidths=0.6, zorder=6,
                       label=overlay_label)

    ax.scatter([gc["RA"]], [gc["DEC"]], marker="*", s=260, facecolor="red",
               edgecolor="k", lw=1.0, zorder=7)
    ax.annotate(gc_label, (gc["RA"], gc["DEC"]), xytext=(6, -14),
                textcoords="offset points", fontsize=12)
    ax.set_xlabel("R.A. (J2000)")
    ax.set_ylabel("dec (J2000)")
    ax.set_xlim(ra_lim[1], ra_lim[0])  # RA increases to the left
    ax.set_ylim(dec_lim[0], dec_lim[1])
    fig.colorbar(im, ax=ax, label="matched-filter weighted density (log)")
    fig.tight_layout()
    legend_outside(ax, fontsize=9)
    plt.show()
    return (fig, in_band) if return_mask else fig


def density_profile_figure(obs_tab, stream_tab, gc, gc_label="GC",
                           obs_cols=("ra", "dec"), frame=None, weights=None,
                           gc_mask_deg=None,
                           on_width=0.5, bg_offset=0.5, bg_width=0.5,
                           normalize_bg=True, bg_smooth_deg=None,
                           track_nbins=30, track_smooth=2.0,
                           phi1_lim=None, phi2_lim=None, nbins=60,
                           pix=0.15, smooth_deg=0.4,
                           vmin_pct=40, vmax_pct=99.0, vmax=None, cmap="magma",
                           show_particles=True, particle_size=4, image_aspect=None,
                           figsize=(11, 5.5), show_map=False,
                           overlay_tab=None, overlay_cols=("ra", "dec"),
                           overlay_label="confirmed candidates",
                           overlay_layers=None, phi1_bin_edges=None,
                           phi1_bin_labels=None, show_title=True):
    """Matched-filter weighted density map + background-subtracted profile in
    stream coordinates (Price-Whelan & Bonaca 2018, GD-1 Fig. 2 style).

    Stars are described by an along-stream coordinate ``phi1`` and the cross-track
    residual ``phi2 - track(phi1)``, where ``track`` is the simulated stream's
    median ``phi2(phi1)`` (a curved centreline). The Stream / Background bands are
    defined on that residual, so they follow the curved track (the same
    track +/- band corridor as :func:`matched_filter_map`) rather than a straight
    cut. Two stacked panels share the ``phi1`` axis:

      1. a single ``weights``-weighted density image showing the (curved) stream,
         with the band boundaries drawn as curves hugging the track:
         Stream (``|resid| < on_width``) and Background
         (``bg_offset <= |resid| < bg_offset + bg_width``).
      2. weighted counts/deg vs phi1: on-stream band, area-scaled background
         band, and the (stream - background) difference.

    Parameters
    ----------
    obs_tab : table
        Stars passing the loose CMD + PM selection.
    stream_tab : table
        Simulated stream (defines the phi2 track).
    gc : mapping
        Cluster parameters with at least RA, DEC.
    frame : (ex, ey, ez) or None
        Stream-frame basis (e.g. from :func:`streamcutter.great_circle_frame`).
        If given, ``(phi1, phi2)`` are true stream coordinates via
        :func:`streamcutter.icrs_to_phi12`. If None, RA/Dec are used (flat sky).
    weights : array or None
        Per-star matched-filter weight (e.g. from :func:`streamcutter.pm_weight`).
        Used to weight both the 2-D map and the 1-D profile. Defaults to ones.
    gc_mask_deg : float or None
        If set, drop stars within this angular radius [deg] of the cluster.
    on_width, bg_offset, bg_width : float
        Cross-track band geometry [deg]. The background is scaled by
        ``on_width / bg_width`` so it matches the on-stream extent.
    normalize_bg : bool
        If True, divide the *image* by a smooth 2-D background surface fit to the
        off-track background pixels (and interpolated under the stream). This
        captures background variation in both phi1 and phi2, so the map becomes a
        contrast (background ~1, stream an overdensity) shown on a linear scale.
        The bottom profile is unaffected.
    bg_smooth_deg : float or None
        Smoothing scale [deg] of the background surface. Must be wide enough to
        bridge the excluded stream band; defaults to ``1.5*(on_width+bg_width)``.
    track_nbins, track_smooth : int, float
        The stream centreline is the median phi2 in ``track_nbins`` phi1 bins,
        Gaussian-smoothed by ``track_smooth`` bins. Fewer bins / more smoothing
        give a cleaner curve for long, curved streams (set track_smooth=0 to
        disable smoothing).
    phi2_lim : (lo, hi) or None
        Explicit cross-track display range [deg] for the image panel. If None,
        it auto-fits to the curved track +/- the band extent.
    overlay_tab : table or None
        If given (and ``show_map`` is set), scatter these stars on the standalone
        density map as star markers -- e.g. the RV + [Fe/H] confirmed stream
        candidates over the matched-filter density. Coordinates are read from
        ``overlay_cols`` and transformed with the same ``frame`` as the map.
    overlay_cols : (ra_col, dec_col)
        Column names for the overlay coordinates (default ``("ra", "dec")``).
    overlay_label : str
        Legend label for the overlaid points.
    overlay_layers : list of dict or None
        Multiple overlay catalogues, each a dict with keys ``tab`` (table),
        ``cols`` (ra, dec column names), ``label``, and optional matplotlib
        scatter styling (``color``, ``marker``, ``s``, ``alpha``, ``edgecolor``,
        ``lw``). ``overlay_tab`` is prepended as the first layer if given.
    pix, smooth_deg, vmin_pct, cmap : map rendering controls (see matched_filter_map).
    image_aspect : float, str or None
        Aspect of the 2-D image panel. ``None`` (default) picks automatically:
        ``"auto"`` (stretch to fill) for a stream frame, and a sky-correct
        ``1/cos(Dec)`` for flat-sky RA/Dec (``frame=None``) so the on-sky map is
        not distorted. Pass ``"auto"``/``"equal"``/a number to override.
    """
    rc, dc = obs_cols
    ra = np.asarray(obs_tab[rc], float)
    dec = np.asarray(obs_tab[dc], float)
    sra = np.asarray(stream_tab["RA"], float)
    sde = np.asarray(stream_tab["DEC"], float)
    w = np.ones(ra.size) if weights is None else np.asarray(weights, float)

    # mask the cluster itself (otherwise it spikes the on-stream profile)
    if gc_mask_deg is not None:
        rstar = sph_to_cart(ra, dec)
        rgc = sph_to_cart(np.array([gc["RA"]]), np.array([gc["DEC"]]))[0]
        sep = np.degrees(np.arccos(np.clip(rstar @ rgc, -1.0, 1.0)))
        keep = sep > gc_mask_deg
        ra, dec, w = ra[keep], dec[keep], w[keep]
        print(f"[density profile] masked {(~keep).sum()} stars within "
              f"{gc_mask_deg:.2f} deg of {gc_label}")

    # map ICRS -> stream coordinates (true frame, or flat-sky RA/Dec proxy)
    if frame is not None:
        ex, ey, ez = frame
        phi1, phi2 = icrs_to_phi12(ra, dec, ex, ey, ez)
        sphi1, sphi2 = icrs_to_phi12(sra, sde, ex, ey, ez)
        g1_arr, g2_raw = icrs_to_phi12(np.array([gc["RA"]]),
                                       np.array([gc["DEC"]]), ex, ey, ez)
        g1, g2_raw = float(g1_arr[0]), float(g2_raw[0])
        invert_x = False
    else:
        phi1, phi2 = ra, dec
        sphi1, sphi2 = sra, sde
        g1, g2_raw = float(gc["RA"]), float(gc["DEC"])
        invert_x = True  # RA increases to the left

    # axis labels: true stream-frame coords vs flat-sky RA/Dec
    if frame is not None:
        xlabel, ylabel = r"$\phi_1$ (degree)", r"$\phi_2$ (degree)"
    else:
        xlabel, ylabel = "R.A. (degree)", "Dec. (degree)"

    # stream phi2 track (median phi2 along phi1 from the simulation) -- the curved
    # stream centreline. The bands follow this curve, not a straight line. A long
    # stream is not a great circle, so phi2(phi1) curves; smooth the binned medians
    # so the track is a clean curve rather than a jagged per-bin line.
    tx, ty = binned_track(sphi1, sphi2, nbins=track_nbins, min_count=3)
    order = np.argsort(tx)
    tx, ty = tx[order], ty[order]
    ty_raw = ty.copy()
    if track_smooth and len(ty) >= 3:
        ty = gaussian_filter(ty, sigma=track_smooth)
    track = lambda x: np.interp(x, tx, ty, left=ty[0], right=ty[-1])

    # The frame flattens the stream, so the residual phi2(phi1) curvature in these
    # coordinates is small *by construction* (the great-circle frame absorbed the
    # sky curvature). Report it so a near-flat midline is read as "the simulation
    # is straight in this frame", not as a hard-coded straight cut.
    if len(ty):
        track_ptp = float(ty.max() - ty.min())
        track_ptp_raw = float(ty_raw.max() - ty_raw.min())
        print(f"[density profile] frame phi2 track curvature: "
              f"{track_ptp:.3f} deg peak-to-peak (smoothed; {track_ptp_raw:.3f} "
              f"raw) over {tx.max() - tx.min():.1f} deg of phi1 -- "
              f"the simulation is nearly a great circle in this frame")
    else:
        track_ptp = 0.0

    # membership is the cross-track residual from the curved track (= the
    # track +/- band corridor used by matched_filter_map), so the on-stream and
    # background bands hug the arc instead of cutting straight across it.
    resid = phi2 - track(phi1)
    on = np.abs(resid) < on_width
    bg = (np.abs(resid) >= bg_offset) & (np.abs(resid) < bg_offset + bg_width)
    scale = on_width / bg_width                      # match phi2 extent

    if phi1_lim is None:
        phi1_lim = (phi1.min(), phi1.max())

    print(f"[density profile] weighted on-stream={w[on].sum():.0f}  "
          f"background={w[bg].sum():.0f}  (bg scale={scale:.2f})")

    # ---- single weighted density image; bands follow the curved track --------
    grid = np.linspace(min(phi1_lim), max(phi1_lim), 400)
    tc = track(grid)
    pad = bg_offset + bg_width
    if phi2_lim is not None:
        ylo, yhi = min(phi2_lim), max(phi2_lim)
    else:
        ylo, yhi = tc.min() - 1.2 * pad, tc.max() + 1.2 * pad
    nx = max(int(round(abs(phi1_lim[1] - phi1_lim[0]) / pix)), 1)
    ny = max(int(round((yhi - ylo) / pix)), 1)
    H, xe, ye = np.histogram2d(phi1, phi2, bins=[nx, ny],
                               range=[sorted(phi1_lim), [ylo, yhi]], weights=w)
    H = gaussian_filter(H, sigma=smooth_deg / pix)

    # Background model: a smooth 2-D surface fit to the off-track background
    # pixels only (the stream band is excluded), then interpolated *under* the
    # stream via normalised (NaN-aware) convolution. This captures background
    # variation in BOTH phi1 and phi2 -- not a single per-column scalar -- so a
    # narrow stream divided by it keeps its overdensity signal.
    if normalize_bg:
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        resid_grid = yc[None, :] - track(xc)[:, None]            # (nx, ny)
        bgmask = ((np.abs(resid_grid) >= bg_offset) &
                  (np.abs(resid_grid) < bg_offset + bg_width)).astype(float)
        # smoothing scale: wide enough to bridge the excluded stream band
        sig = (bg_smooth_deg if bg_smooth_deg is not None
               else 1.5 * (on_width + bg_width)) / pix
        num = gaussian_filter(H * bgmask, sigma=sig)
        den = gaussian_filter(bgmask, sigma=sig)
        bg_surface = num / np.maximum(den, 1e-12)
        floor = max(np.median(bg_surface[bg_surface > 0]) * 1e-2, 1e-9) \
            if np.any(bg_surface > 0) else 1e-9
        Hdisp = H / np.maximum(bg_surface, floor)
        cbar_label = "density / background"
    else:
        Hdisp = H
        cbar_label = "weighted density (log)"

    pos = Hdisp[Hdisp > 0]
    if pos.size:
        vmin = max(np.percentile(pos, vmin_pct), 1e-3)
        if vmax is None:
            vmax = max(np.percentile(pos, vmax_pct), vmin * 1.5)
    else:
        vmin = 1e-3
        if vmax is None:
            vmax = 1.0
    # contrast map -> linear scale (no log / 10^0 ticks); raw density -> log
    if normalize_bg:
        img_norm = Normalize(vmin=0.0, vmax=vmax)
    else:
        img_norm = LogNorm(vmin=vmin, vmax=vmax)

    # Overlay catalogue(s) (e.g. RV+[Fe/H] confirmed / final candidates, or an
    # external stream catalogue). A single table via overlay_tab is treated as the
    # first layer; coordinates map through the same frame as the density map.
    def _draw_overlays(ax):
        layers = list(overlay_layers) if overlay_layers else []
        if overlay_tab is not None and len(overlay_tab):
            layers.insert(0, dict(tab=overlay_tab, cols=overlay_cols,
                                  label=overlay_label, color="lime",
                                  marker="*", s=70))
        drew = False
        for ly in layers:
            tab = ly.get("tab")
            if tab is None or not len(tab):
                continue
            orc, odc = ly.get("cols", ("ra", "dec"))
            ora = np.asarray(tab[orc], float)
            ode = np.asarray(tab[odc], float)
            if frame is not None:
                op1, op2 = icrs_to_phi12(ora, ode, ex, ey, ez)
            else:
                op1, op2 = ora, ode
            if ly.get("style") == "line":
                # draw a smooth centreline: bin along-track and connect the
                # per-bin medians (raw particles are too scattered for a clean
                # line). Useful for overlaying simulated-stream tracks.
                bx, by = binned_track(op1, op2, nbins=ly.get("nbins", track_nbins),
                                      min_count=3)
                order = np.argsort(bx)
                ax.plot(bx[order], by[order], "-", color=ly.get("color", "lime"),
                        lw=ly.get("lw", 2.0), alpha=ly.get("alpha", 1.0),
                        zorder=6, label=ly.get("label"))
            else:
                ax.scatter(op1, op2, s=ly.get("s", 70),
                           marker=ly.get("marker", "*"),
                           facecolor=ly.get("color", "lime"),
                           edgecolor=ly.get("edgecolor", "k"),
                           lw=ly.get("lw", 0.7),
                           alpha=ly.get("alpha", 1.0), zorder=6,
                           label=ly.get("label"))
            drew = True
        return drew

    # map-panel legend style (overlay catalogues + simulated particles)
    map_legend_kw = dict(fontsize=10, markerscale=0.8, labelspacing=0.8,
                         handletextpad=0.7)

    fig, axes = plt.subplots(2, 1, figsize=figsize,
                             gridspec_kw=dict(height_ratios=[1.1, 1]), sharex=True)
    # both panels share the same left/right so their widths match; the colorbar
    # lives in its own axes and does not shrink the image panel.
    fig.subplots_adjust(hspace=0.08, left=0.08, right=0.88, top=0.93, bottom=0.11)

    # Image aspect: the great-circle stream frame is a flattened thin strip, so
    # "auto" (stretch to fill the panel) reads best. Flat-sky RA/Dec, however, is
    # a real on-sky map -- show it with the correct projected proportions
    # (1 deg Dec = 1 deg RA / cos(Dec)) so the stream is not distorted, like the
    # other RA/Dec sky figures. For a wide RA strip this letterboxes vertically
    # and keeps the x-axis aligned with the profile panel below.
    if image_aspect is not None:
        aspect = image_aspect
    elif frame is not None:
        aspect = "auto"
    else:
        aspect = 1.0 / np.cos(np.radians(0.5 * (ylo + yhi)))
    ax = axes[0]
    im = ax.imshow(Hdisp.T, origin="lower", aspect=aspect, cmap=cmap,
                   interpolation="bilinear",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]], norm=img_norm)
    # a fixed/sky-correct aspect shrinks the image box; anchor it to the left so
    # we can snap the profile panel + colorbar to its true drawn width below.
    if aspect != "auto":
        ax.set_anchor("W")
    # curved barriers following the track: stream edges (+/-on_width) and the
    # outer background edges (+/-(bg_offset+bg_width)).
    ax.plot(grid, tc, ls="-", c="w", lw=0.7, alpha=0.4)
    for s in (+1, -1):
        ax.plot(grid, tc + s * on_width, ls="--", c="w", lw=0.9, alpha=0.7)
        ax.plot(grid, tc + s * (bg_offset + bg_width), ls=":", c="w",
                lw=0.9, alpha=0.5)
    # overlay the simulated stream particles themselves -- the direct proof that
    # the (near-flat) midline traces the simulation, not a straight cut. Drawn as
    # small translucent dots so the underlying density still reads through.
    if show_particles:
        vis = ((sphi1 >= min(phi1_lim)) & (sphi1 <= max(phi1_lim)) &
               (sphi2 >= ylo) & (sphi2 <= yhi))
        ax.scatter(sphi1[vis], sphi2[vis], s=particle_size, marker=".",
                   facecolor="cyan", edgecolor="none", alpha=0.35, zorder=4,
                   rasterized=True, label="simulated particles")

    # region labels at the displayed-left edge, following the curve
    x_left = max(phi1_lim) if invert_x else min(phi1_lim)
    t_left = float(track(np.array([x_left]))[0])
    ytrans = ax.get_yaxis_transform()   # x in axes fraction, y in data units
    ax.text(0.012, t_left, "Stream", transform=ytrans, fontsize=8,
            color="w", alpha=0.9, va="center")
    for s in (+1, -1):
        ax.text(0.012, t_left + s * (bg_offset + 0.5 * bg_width), "Background",
                transform=ytrans, fontsize=8, color="w", alpha=0.9, va="center")
    ax.scatter([g1], [g2_raw], marker="*", s=180, facecolor="cyan",
               edgecolor="k", lw=0.8, zorder=5)
    _draw_overlays(ax)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylo, yhi)
    _kind = "background-normalised" if normalize_bg else "weighted"
    if show_title:
        ax.set_title(f"{gc_label}: {_kind} density after selected CMD and PM",
                     loc="left", fontsize=11)
    # note that the (near-flat) midline is the simulation in stream-frame coords,
    # whose residual curvature is genuinely small -- not a straight-line cut.
    ax.set_title(rf"sim track curvature {track_ptp:.2f}$^\circ$ p2p in frame",
                 loc="right", fontsize=8, color="0.4")
    ax.tick_params(which="both", direction="in", top=True, right=True, color="w")
    p = ax.get_position()
    cax = fig.add_axes([p.x1 + 0.012, p.y0, 0.015, p.height])
    fig.colorbar(im, cax=cax, label=cbar_label)

    # ---- bottom: weighted, background-subtracted profile along phi1 ----------
    edges = np.linspace(min(phi1_lim), max(phi1_lim), nbins + 1)
    binw = (max(phi1_lim) - min(phi1_lim)) / nbins
    cen = 0.5 * (edges[:-1] + edges[1:])
    dens_on = np.histogram(phi1[on], bins=edges, weights=w[on])[0] / binw
    dens_bg = np.histogram(phi1[bg], bins=edges, weights=w[bg])[0] * scale / binw

    ax = axes[1]
    ax.step(cen, dens_on, where="mid", c="0.6", lw=1.0, label="Stream")
    ax.step(cen, dens_bg, where="mid", c="tab:orange", lw=1.0, label="Background")
    ax.step(cen, dens_on - dens_bg, where="mid", c="k", lw=1.5,
            label="Stream - Background")
    ax.axhline(0, ls=":", c="k", lw=0.8)
    ax.set_ylabel(r"weighted N deg$^{-1}$")
    ax.set_xlabel(xlabel)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    # mark the phi1 extent actually covered by the simulated stream (outside this
    # the track/profile is extrapolation, not signal). tx = binned stream track.
    s_lo, s_hi = float(np.nanmin(tx)), float(np.nanmax(tx))
    for _ax in axes:
        _ax.axvline(s_lo, color="cyan", ls=":", lw=1.2, alpha=0.85)
        _ax.axvline(s_hi, color="cyan", ls=":", lw=1.2, alpha=0.85)
    # optional: draw the phi1 distance-bin boundaries used by the CMD selection
    # as vertical dividers, with each segment numbered on the map panel.
    if phi1_bin_edges is not None and len(phi1_bin_edges) >= 2:
        be = np.asarray(phi1_bin_edges, float)
        for e in be:
            axes[0].axvline(e, color="w", ls="-", lw=0.8, alpha=0.55, zorder=4)
            axes[1].axvline(e, color="0.5", ls="-", lw=0.8, alpha=0.55)
        bc = 0.5 * (be[:-1] + be[1:])
        labels = (phi1_bin_labels if phi1_bin_labels is not None
                  else range(len(bc)))
        xtr = axes[0].get_xaxis_transform()   # x in data, y in axes fraction
        for c, lab in zip(bc, labels):
            axes[0].text(c, 0.96, str(lab), transform=xtr, ha="center", va="top",
                         color="w", fontsize=9, fontweight="bold", zorder=6)

    _h, _l = axes[1].get_legend_handles_labels()
    _h.append(Line2D([0], [0], color="cyan", ls=":", lw=1.2))
    _l.append("stream extent")

    for ax in axes:
        if invert_x:
            ax.set_xlim(max(phi1_lim), min(phi1_lim))   # RA increases to the left
        else:
            ax.set_xlim(min(phi1_lim), max(phi1_lim))

    # With a fixed/sky-correct aspect the image box shrinks to its true drawn
    # size, so snap the profile panel and colorbar to that box: match the bottom
    # panel's width/left to the image, and re-hang the colorbar on the image's
    # right edge at its height. (For aspect="auto" the box is unchanged, so this
    # is skipped and the original full-width layout is kept.)
    if aspect != "auto":
        fig.canvas.draw()
        bb0 = axes[0].get_position()
        bb1 = axes[1].get_position()
        axes[1].set_position([bb0.x0, bb1.y0, bb0.width, bb1.height])
        cax.set_position([bb0.x1 + 0.012, bb0.y0, 0.015, bb0.height])

    legend_outside(axes[0], **map_legend_kw)
    legend_outside(axes[1], handles=_h, labels=_l, fontsize=9)
    plt.show()

    # Optional standalone figure: just the density/background image, no overlays
    # (no track, band edges, particles, cluster marker, or region labels).
    if show_map:
        figm, axm = plt.subplots(figsize=figsize)
        imm = axm.imshow(Hdisp.T, origin="lower", aspect=aspect, cmap=cmap,
                         interpolation="bilinear",
                         extent=[xe[0], xe[-1], ye[0], ye[-1]], norm=img_norm)
        # the stream centreline (curved track) as a dotted line -- the only overlay
        axm.plot(grid, track(grid), ls=":", c="w", lw=1.2, alpha=0.8)
        _draw_overlays(axm)
        if invert_x:
            axm.set_xlim(max(phi1_lim), min(phi1_lim))
        else:
            axm.set_xlim(min(phi1_lim), max(phi1_lim))
        axm.set_ylim(ylo, yhi)
        axm.set_xlabel(xlabel)
        axm.set_ylabel(ylabel)
        if show_title:
            axm.set_title(f"{gc_label}: {_kind} density after selected CMD and PM",
                          loc="left", fontsize=11)
        axm.tick_params(which="both", direction="in", top=True, right=True,
                        color="w")
        figm.colorbar(imm, ax=axm, label=cbar_label)
        legend_outside(axm, **map_legend_kw)
        plt.show()

    return fig


def rv_selection_plot(spec_tab, stream_tab, gc, gc_label="GC",
                      ra_col="ra", rv_col="RV", rverr_col="RV_ERR",
                      color_col=None):
    """Radial-velocity selection diagnostic (cf. notebooks/test-wide-m5).

    The simulated stream's v_hel(RA) track with the cross-matched DESI
    candidates (RV +/- error) overplotted. If ``color_col`` is given (e.g.
    ``"P_rv"``), the points are coloured by it -- showing the RV membership
    likelihood -- otherwise they are a single colour.
    """
    ra = np.asarray(spec_tab[ra_col], float)
    rv = np.asarray(spec_tab[rv_col], float)
    rve = np.asarray(spec_tab[rverr_col], float)
    good = np.isfinite(rv)

    xr, yr = binned_track(np.asarray(stream_tab["RA"], float),
                          np.asarray(stream_tab["VLOS"], float), min_count=3)
    order = np.argsort(xr)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xr[order], yr[order], "-", c="k", lw=1.5, label="simulated stream")
    ax.errorbar(ra[good], rv[good], yerr=rve[good], fmt="none",
                ecolor="0.6", elinewidth=1, capsize=2, zorder=3)
    if color_col is not None and color_col in spec_tab.colnames:
        cval = np.asarray(spec_tab[color_col], float)[good]
        sc = ax.scatter(ra[good], rv[good], c=cval, s=45, cmap="viridis",
                        vmin=0, vmax=1, edgecolor="k", lw=0.4, zorder=4,
                        label=f"DESI candidates ({good.sum()})")
        fig.colorbar(sc, ax=ax, label=color_col)
    else:
        ax.scatter(ra[good], rv[good], s=45, c="crimson", edgecolor="k",
                   lw=0.4, zorder=4, label=f"DESI candidates ({good.sum()})")
    ax.scatter([gc["RA"]], [gc["VLOS"]], marker="*", s=240, c="gold",
               edgecolor="k", zorder=5, label=gc_label)
    ax.set_xlabel("R.A. (deg, J2000)")
    ax.set_ylabel(r"$v_{\rm hel}$ (km/s)")
    ax.invert_xaxis()
    # robust y-limit so a single bad RV does not compress the plot
    if good.sum():
        lo = min(np.percentile(rv[good], 2), yr.min(), gc["VLOS"])
        hi = max(np.percentile(rv[good], 98), yr.max(), gc["VLOS"])
        pad = 0.15 * (hi - lo) + 30
        ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    legend_outside(ax, fontsize=9)
    plt.show()
    return fig


# ===========================================================================
# cmd_hess: distance bins, CMD / Hess diagnostics
# ===========================================================================
# Hess-diagram binning + brick area (deg^2), matching region_cmd_gmm.
BRICK_AREA = 0.25 * 0.25
XBINS = np.linspace(0.0, 1.0, 50)
YBINS = np.linspace(16.0, 22.0, 50)


def _strip(x):
    return np.asarray(getattr(x, "value", x), dtype=float)


def _great_circle_sep(ra, dec, ra0, dec0):
    r1 = sph_to_cart(np.asarray(ra, float), np.asarray(dec, float))
    r0 = sph_to_cart(np.array([ra0]), np.array([dec0]))[0]
    return np.degrees(np.arccos(np.clip(r1 @ r0, -1.0, 1.0)))


def assign_dist_bins(combined, sim, frame, nbins=9, phi2_max=2.0,
                     fallback_kpc=None, clamp=True):
    """Attach phi1, phi2, ``dist_bin`` and ``dist_ref_kpc`` columns.

    Stars are binned by along-stream ``phi1`` over the simulated stream's phi1
    extent (within ``|phi2| <= phi2_max``); each bin is given the *median
    simulated-stream distance* in that bin. Out-of-range phi1 clamps to the
    nearest end bin (``clamp=True``) so distant field is extrapolated along the
    stream's distance trend rather than reset to the cluster distance.
    """
    ex, ey, ez = frame
    phi1_d, phi2_d = icrs_to_phi12(_strip(combined["ra"]),
                                   _strip(combined["dec"]), ex, ey, ez)
    phi1_s, phi2_s = icrs_to_phi12(_strip(sim["RA"]), _strip(sim["DEC"]),
                                   ex, ey, ez)
    dist_s = _strip(sim["DIST"])
    band = np.abs(phi2_s) <= phi2_max
    lo, hi = float(np.nanmin(phi1_s[band])), float(np.nanmax(phi1_s[band]))
    edges = np.linspace(lo, hi, nbins + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])

    bin_dist = np.full(nbins, np.nan)
    for b in range(nbins):
        m = band & (phi1_s >= edges[b])
        m &= (phi1_s <= edges[b + 1]) if b == nbins - 1 else (phi1_s < edges[b + 1])
        if m.any():
            bin_dist[b] = np.nanmedian(dist_s[m])
    goodb = np.isfinite(bin_dist)
    if goodb.sum() >= 2:
        bin_dist = np.interp(cen, cen[goodb], bin_dist[goodb])
    elif goodb.sum() == 1:
        bin_dist[:] = bin_dist[goodb][0]
    elif fallback_kpc is not None:
        bin_dist[:] = fallback_kpc

    pc = np.clip(phi1_d, lo, hi - 1e-9) if clamp else phi1_d
    bid = np.clip(np.digitize(pc, edges) - 1, 0, nbins - 1)
    combined = combined.copy()
    combined["phi1"] = phi1_d
    combined["phi2"] = phi2_d
    combined["dist_bin"] = bid
    combined["dist_ref_kpc"] = bin_dist[bid]
    print(f"[dist bins] {nbins} phi1 bins over [{lo:.1f}, {hi:.1f}] deg, "
          f"sim distance {np.nanmin(bin_dist):.1f}-{np.nanmax(bin_dist):.1f} kpc")
    return combined


def assign_stream_regions(combined, sim, frame, gc, on_width, bg_offset,
                          bg_width, gc_mask_deg, track_nbins=30, track_smooth=2.0,
                          mag_col="gmag", mag_lim=None,
                          color_col="grcolor", color_lim=None):
    """Tag stars ``target`` / ``background`` / ``none`` from the density-profile
    phi2 bands (the curved track +/- corridor), masking the cluster core.

    Requires ``combined`` to already carry ``phi1`` / ``phi2`` (see
    :func:`assign_dist_bins`).

    Parameters
    ----------
    on_width, bg_offset, bg_width, gc_mask_deg : float
        Cross-track geometry, all in DEGREES: the on-stream band is
        ``|phi2 - track(phi1)| < on_width``, the background band runs from
        ``bg_offset`` to ``bg_offset + bg_width``, and stars within
        ``gc_mask_deg`` of the cluster centre are dropped from both.
    mag_lim, color_lim : (lo, hi) or None
        Optional limiting magnitude / colour range. Stars outside are tagged
        ``none``, i.e. they enter neither the on-stream nor the background
        sample -- so the Hess panels, their star counts and anything else built
        on these regions (e.g. a PM GMM fitted on the on-stream sample) obey the
        same photometric limits. Pass ``None`` on either end to leave that side
        open, e.g. ``mag_lim=(None, 21.5)``.
    """
    ex, ey, ez = frame
    phi1, phi2 = _strip(combined["phi1"]), _strip(combined["phi2"])
    sphi1, sphi2 = icrs_to_phi12(_strip(sim["RA"]), _strip(sim["DEC"]), ex, ey, ez)
    tx, ty = binned_track(sphi1, sphi2, nbins=track_nbins, min_count=3)
    order = np.argsort(tx)
    tx, ty = tx[order], ty[order]
    if track_smooth and len(ty) >= 3:
        ty = gaussian_filter(ty, sigma=track_smooth)
    track = lambda x: np.interp(x, tx, ty, left=ty[0], right=ty[-1])

    resid = phi2 - track(phi1)
    region = np.full(len(combined), "none", dtype="U10")
    region[np.abs(resid) < on_width] = "target"
    region[(np.abs(resid) >= bg_offset) &
           (np.abs(resid) < bg_offset + bg_width)] = "background"
    sep = _great_circle_sep(_strip(combined["ra"]), _strip(combined["dec"]),
                            gc["RA"], gc["DEC"])
    region[sep < gc_mask_deg] = "none"

    cuts = ""
    for col, lim, tag in ((mag_col, mag_lim, "mag"), (color_col, color_lim, "colour")):
        if lim is None or col not in getattr(combined, "colnames", []):
            continue
        v = _strip(combined[col])
        keep = np.isfinite(v)
        if lim[0] is not None:
            keep &= v >= lim[0]
        if lim[1] is not None:
            keep &= v <= lim[1]
        region[~keep] = "none"
        cuts += f" {tag}=[{lim[0]}, {lim[1]}]"

    combined = combined.copy()
    combined["region"] = region
    print(f"[regions] on_width={on_width:.3f} bg=[{bg_offset:.3f},"
          f"{bg_offset + bg_width:.3f}] deg gc_mask={gc_mask_deg:.3f} deg"
          f"{cuts} -> "
          f"target={int((region == 'target').sum())}  "
          f"background={int((region == 'background').sum())}")
    return combined


def _hess(tab, xcol, ycol, area, xbins=None, ybins=None):
    H, xe, ye = np.histogram2d(_strip(tab[xcol]), _strip(tab[ycol]),
                               bins=[XBINS if xbins is None else xbins,
                                     YBINS if ybins is None else ybins])
    return H.T / area, xe, ye


def tol_band_edges(base_tol, faint_tol, g_bright=18.0, g_faint=21.0):
    """`band_edges` for the magnitude-dependent colour tolerance used by
    :func:`streamcutter.select_around_isochrone` (M5 / NGC scripts)."""
    def _edges(g_app, gr_app, b):
        tol = base_tol + np.clip((g_app - g_bright) / (g_faint - g_bright),
                                 0, 1) * (faint_tol - base_tol)
        return gr_app - tol, gr_app + tol
    return _edges


def delta_band_edges(deltas, default):
    """`band_edges` for the asymmetric fixed (blueward, redward) colour band used
    by the Pal 5 / Pal 13 scripts. ``deltas`` maps bin index -> (dleft, dright)."""
    def _edges(g_app, gr_app, b):
        dleft, dright = deltas.get(int(b), default)
        return gr_app - dleft, gr_app + dright
    return _edges

# ===========================================================================
# pm_gmm: proper-motion Gaussian-mixture membership
# ===========================================================================
def gmm_ellipse(ax, mean, cov, n_std=2.0, **kw):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=theta,
                         fill=False, lw=1.5, **kw))


def fit_stream_gmm(pm_table, gc_pmra=None, gc_pmdec=None, known_components=None,
                   stream_name=None, n_components=4, pmra_col="pmra",
                   pmdec_col="pmdec", cov_inflation=1.0, min_weight=0.02,
                   random_state=42):
    """Gaussian-mixture in PM space with optional frozen (known) templates."""
    pmra = _strip(pm_table[pmra_col])
    pmdec = _strip(pm_table[pmdec_col])
    X = np.column_stack([pmra, pmdec])
    good = np.all(np.isfinite(X), axis=1)
    Xfit = X[good]
    N = Xfit.shape[0]

    known_components = known_components or []
    n_known = len(known_components)
    k_free = n_components - n_known
    if k_free < 1:
        raise ValueError(f"n_components ({n_components}) must exceed "
                         f"len(known_components) ({n_known}).")

    fixed_means, fixed_covs, fixed_names = [], [], []
    for comp in known_components:
        arr = np.column_stack([_strip(comp["pmra"]), _strip(comp["pmdec"])])
        arr = arr[np.all(np.isfinite(arr), axis=1)]
        if arr.shape[0] < 5:
            raise ValueError(f"known component '{comp.get('name', '?')}' has too "
                             f"few finite stars ({arr.shape[0]})")
        fixed_means.append(arr.mean(axis=0))
        fixed_covs.append(np.cov(arr, rowvar=False) + np.eye(2) * cov_inflation)
        fixed_names.append(comp.get("name", f"known_{len(fixed_names)}"))

    init_gmm = GaussianMixture(n_components=k_free, covariance_type="full",
                               random_state=random_state, n_init=5,
                               max_iter=500, reg_covar=1e-4).fit(Xfit)
    means = np.asarray(list(init_gmm.means_) + fixed_means)
    covs = np.asarray(list(init_gmm.covariances_) + fixed_covs)
    names = [f"free {k}" for k in range(k_free)] + fixed_names
    weights = np.concatenate([0.7 * init_gmm.weights_,
                              np.full(n_known, 0.3 / max(n_known, 1))])
    weights /= weights.sum()

    if n_known > 0:
        for _ in range(100):
            log_resp = np.empty((N, n_components))
            for k in range(n_components):
                log_resp[:, k] = (np.log(weights[k] + 1e-300)
                    + multivariate_normal(mean=means[k], cov=covs[k],
                                          allow_singular=True).logpdf(Xfit))
            log_resp -= log_resp.max(axis=1, keepdims=True)
            resp = np.exp(log_resp)
            resp /= resp.sum(axis=1, keepdims=True)
            Nk = resp.sum(axis=0)
            new_weights = Nk / N
            for j in range(n_known):
                idx = k_free + j
                if new_weights[idx] < min_weight:
                    new_weights[idx] = min_weight
            new_weights /= new_weights.sum()
            weights = new_weights
            for k in range(k_free):
                if Nk[k] < 1:
                    continue
                r = resp[:, k:k + 1]
                mu_new = (r * Xfit).sum(axis=0) / Nk[k]
                d = Xfit - mu_new
                covs[k] = (r * d).T @ d / Nk[k] + np.eye(2) * 1e-4
                means[k] = mu_new
    else:
        resp = init_gmm.predict_proba(Xfit)

    if stream_name is not None:
        stream_idx = k_free + fixed_names.index(stream_name)
        names[stream_idx] = f"stream ({stream_name})"
    else:
        if gc_pmra is None:
            raise ValueError("Need either stream_name or (gc_pmra, gc_pmdec).")
        anchor = np.array([float(_strip(gc_pmra)), float(_strip(gc_pmdec))])
        stream_idx = int(np.argmin(np.linalg.norm(means[:k_free] - anchor, axis=1)))
        names[stream_idx] = f"stream (free {stream_idx})"

    full_resp = np.full((X.shape[0], n_components), np.nan)
    full_resp[good] = resp
    return dict(means=means, covs=covs, weights=weights, names=names,
                stream_idx=stream_idx, probs=full_resp,
                stream_prob=full_resp[:, stream_idx],
                gc_anchor=(np.array([float(_strip(gc_pmra)), float(_strip(gc_pmdec))])
                           if gc_pmra is not None else means[stream_idx]))


def plot_stream_gmm(pm_table, result, extras=None, pmra_col="pmra",
                    pmdec_col="pmdec", xlim=(-10, 0), ylim=(-10, 0), title="GC"):
    pmra = _strip(pm_table[pmra_col])
    pmdec = _strip(pm_table[pmdec_col])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=120)

    ax = axes[0]
    ax.scatter(pmra, pmdec, s=0.3, alpha=0.25, c="k")
    for e in (extras or []):
        ax.scatter(_strip(e["pmra"]), _strip(e["pmdec"]), s=e.get("s", 2),
                   c=e.get("color", "blue"), label=e.get("label", ""))
    ax.scatter(*result["gc_anchor"], s=120, marker="X", c="red",
               edgecolors="k", zorder=5, label="anchor (GC)")
    for k, (mu, cov, w, nm) in enumerate(zip(result["means"], result["covs"],
                                             result["weights"], result["names"])):
        is_s = (k == result["stream_idx"])
        ax.scatter(*mu, marker="*" if is_s else "o", s=140 if is_s else 50,
                   c="yellow" if is_s else "orange", edgecolors="k", zorder=5,
                   label=f"{nm}  (w={w:.2f})")
        gmm_ellipse(ax, mu, cov, color="magenta" if is_s else "gray",
                 alpha=0.9 if is_s else 0.5)
    ax.set(xlim=xlim, ylim=ylim, xlabel="pmra [mas/yr]", ylabel="pmdec [mas/yr]",
           title=f"{title}: GMM (K={len(result['means'])})")

    ax = axes[1]
    p = result["stream_prob"]
    order = np.argsort(np.nan_to_num(p))
    sc = ax.scatter(pmra[order], pmdec[order], s=0.6, c=p[order],
                    cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="P(stream member)")
    ax.scatter(*result["gc_anchor"], s=120, marker="X", c="red",
               edgecolors="k", zorder=5)
    ax.set(xlim=xlim, ylim=ylim, xlabel="pmra [mas/yr]", ylabel="pmdec [mas/yr]",
           title="Per-star P(stream)")
    fig.tight_layout()
    legend_outside(axes[0], where="below", fontsize=7, ncol=2)
    return fig


def gmm_stream_prob(result, pmra, pmdec):
    """P(stream member) for stars that were not part of the fit.

    Applies the mixture in ``result`` (from :func:`fit_stream_gmm`) to a new
    (pmra, pmdec) sample, so a GMM fitted on e.g. the on-stream region can be
    evaluated over a whole footprint. Non-finite proper motions give NaN.
    """
    X = np.column_stack([_strip(pmra), _strip(pmdec)])
    good = np.all(np.isfinite(X), axis=1)
    means, covs, weights = result["means"], result["covs"], result["weights"]
    log_resp = np.empty((int(good.sum()), len(means)))
    for k in range(len(means)):
        log_resp[:, k] = (np.log(weights[k] + 1e-300)
                          + multivariate_normal(mean=means[k], cov=covs[k],
                                                allow_singular=True).logpdf(X[good]))
    log_resp -= log_resp.max(axis=1, keepdims=True)
    resp = np.exp(log_resp)
    resp /= resp.sum(axis=1, keepdims=True)
    out = np.full(X.shape[0], np.nan)
    out[good] = resp[:, result["stream_idx"]]
    return out


# =========================================================================== #
# NGC 5024 pipeline steps
#
# Everything above is cluster-agnostic. The helpers below are the steps of this
# particular pipeline; they take the cluster parameters and cut values as
# arguments (the notebook's CONFIG cell owns those) rather than reading globals,
# so this module can be imported and reused as-is.
# =========================================================================== #
def tidal_radius_deg(rt_pc, dist_kpc):
    """Tidal radius converted from pc to degrees at the cluster distance."""
    return np.degrees(np.arctan(rt_pc / (dist_kpc * 1000.0)))


def load_combined(result_dir, name):
    """Load and stack the target + background Tractor cutout catalogs."""
    result_dir = Path(result_dir)
    tgt = Table.read(result_dir / f"{name}_tgt_tractor_combined.fits")
    bg = Table.read(result_dir / f"{name}_bg_tractor_combined.fits")
    tgt["source"] = Column(np.full(len(tgt), "tgt"), dtype="U3")
    bg["source"] = Column(np.full(len(bg), "bg"), dtype="U3")
    combined = vstack([tgt, bg], join_type="outer", metadata_conflicts="silent")

    # Drop duplicate physical sources. The tgt/bg cutouts (and adjacent bricks)
    # overlap, so the same star can appear twice. There is no objid column, so
    # key on (ra, dec) -- exact per source -- and keep the first occurrence.
    n_before = len(combined)
    ra = np.round(np.asarray(combined["ra"], float), 8)
    dec = np.round(np.asarray(combined["dec"], float), 8)
    _, keep = np.unique(np.stack([ra, dec], axis=1), axis=0, return_index=True)
    keep.sort()  # preserve original (tgt-first) row order
    combined = combined[keep]
    print(f"combined rows: {len(combined)}  (tgt {len(tgt)} + bg {len(bg)}; "
          f"dropped {n_before - len(combined)} duplicate sources)")
    return combined


def plot_sky(combined, gc, gc_label, show=True):
    """Smoothed log-density map of the raw footprint.

    With ``show``, drawn inline. It returns the
    figure, so assign the result (``_ = plot_sky(...)``) in a notebook cell --
    a bare call makes the cell echo the figure and render it a second time.
    """
    ra = np.asarray(combined["ra"], float)
    dec = np.asarray(combined["dec"], float)
    good = np.isfinite(ra) & np.isfinite(dec)
    ra, dec = ra[good], dec[good]

    # 2D histogram of the footprint, Gaussian-smoothed, shown in log density.
    bin_deg = 1
    smooth_deg = 1
    ra_bins = np.arange(ra.min(), ra.max() + bin_deg, bin_deg)
    dec_bins = np.arange(dec.min(), dec.max() + bin_deg, bin_deg)
    H, _, _ = np.histogram2d(ra, dec, bins=[ra_bins, dec_bins])
    H = gaussian_filter(H, sigma=smooth_deg / bin_deg)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.pcolormesh(ra_bins, dec_bins, H.T,
                       norm=LogNorm(vmin=max(H.min(), 1e-1), vmax=H.max()),
                       cmap="magma", shading="auto")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("stars / bin (smoothed)")
    ax.scatter([gc["RA"]], [gc["DEC"]], marker="*", s=300, c="cyan",
               edgecolor="k", zorder=5, label=gc_label)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_title(f"{gc_label} wide footprint (PM-cut)")
    legend_outside(ax)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def apply_pm_cut(combined, pm_box, ra_lim, dec_lim):
    """Hard proper-motion box plus the RA/Dec footprint trim."""
    mask = ((combined["pmra"] > pm_box["pmra_min"]) &
            (combined["pmra"] < pm_box["pmra_max"]) &
            (combined["pmdec"] > pm_box["pmdec_min"]) &
            (combined["pmdec"] < pm_box["pmdec_max"]) &
            (combined["ra"] > ra_lim[0]) & (combined["ra"] < ra_lim[1]) &
            (combined["dec"] > dec_lim[0]) & (combined["dec"] < dec_lim[1]))
    out = combined[mask]
    print(f"PM box + footprint: kept {len(out)} / {len(combined)}")
    return out


def stellar_locus_cut(combined, gc, gc_label, fit=True,
                      aperture_deg=0.5, tol=0.1, gr_min=0.1, gr_max=1.2,
                      gmag_lim=(16.0, 22.0), show=True):
    """Point-source stellar-locus cut in (g-r) vs (g-z), before any CMD work.

    Adds the dereddened ``gmag``/``rmag``/``zmag`` photometry, both colours and
    ``delta_locus`` (the residual from the locus), then keeps stars inside the
    ``+/- tol`` band with ``gmag_lim`` and the colour window
    ``gr_min .. gr_max`` applied. With ``fit=True`` the locus slope/intercept
    are fitted on stars within ``aperture_deg`` of the cluster centre over the
    same colour window; otherwise the reference locus is used.
    """
    combined = combined.copy()
    g, r, z = mag(combined, "g"), mag(combined, "r"), mag(combined, "z")
    combined["gmag"], combined["rmag"], combined["zmag"] = g, r, z
    with np.errstate(invalid="ignore"):
        combined["grcolor"] = g - r
        combined["gzcolor"] = g - z
    gr = np.asarray(combined["grcolor"], float)
    gz = np.asarray(combined["gzcolor"], float)

    slope, intercept = LOCUS_SLOPE_REF, LOCUS_INTERCEPT_REF
    source = "reference 1.7(g-r) - 0.17"
    if fit:
        # calibration sample: stars in a small aperture on the cluster, which are
        # overwhelmingly real point sources at a single distance.
        sep = SkyCoord(ra=gc["RA"] * u.deg, dec=gc["DEC"] * u.deg).separation(
            SkyCoord(ra=np.asarray(combined["ra"], float) * u.deg,
                     dec=np.asarray(combined["dec"], float) * u.deg))
        in_ap = sep < aperture_deg * u.deg
        slope, intercept, n_fit = fit_stellar_locus(
            gr[in_ap], gz[in_ap], np.asarray(combined["gmag"], float)[in_ap],
            gr_range=(gr_min, gr_max), gmax=gmag_lim[1] - 1.0)
        source = f"{gc_label} members, N={n_fit}"
        print(f"[locus] fitted on {int(in_ap.sum())} stars within "
              f"{aperture_deg} deg of {gc_label}: "
              f"g-z = {slope:.3f} (g-r) {intercept:+.3f}  (N_fit={n_fit})")

    keep, delta = select_stellar_locus(
        gr, gz, np.asarray(combined["gmag"], float), slope=slope,
        intercept=intercept, tol=tol, g_lim=gmag_lim, gr_min=gr_min,
        gr_max=gr_max, return_delta=True)
    combined["delta_locus"] = delta

    fig = figure_stellar_locus(gr, gz, delta, slope, intercept, tol, gc_label,
                               gr_min=gr_min, gr_max=gr_max,
                               locus_source=source)
    if show:
        plt.show()
    else:
        plt.close(fig)

    out = combined[keep]
    print(f"[locus] kept {len(out)} / {len(combined)} within +/-{tol} mag, "
          f"{gmag_lim[0]} <= g < {gmag_lim[1]}, "
          f"{gr_min} <= (g-r)_0 <= {gr_max}")
    return out



def rv_weight(d_table, stream_simulated, gc):
    """Weight candidates by agreement with the predicted v_hel(RA) profile."""
    ra = np.asarray(d_table["ra"], float)
    rv = np.asarray(d_table["RV"], float)
    rve = np.asarray(d_table["RV_ERR"], float)

    xr, yr = binned_track(np.asarray(stream_simulated["RA"], float),
                          np.asarray(stream_simulated["VLOS"], float),
                          min_count=3)
    v_pred = np.interp(ra, xr, yr)
    sig_v = np.hypot(rve, gc["VLOS"])
    w_vlos = np.exp(-0.5 * ((rv - v_pred) / sig_v) ** 2)

    d_table["v_pred"] = v_pred
    d_table["P_rv"] = w_vlos
    d_table["w_vlos"] = w_vlos
    print(f"w_vlos > 0.5: {(w_vlos > 0.5).sum()} of {len(d_table)}")
    return d_table[w_vlos > 0.5]


def feh_weight(d_good, gc_feh, show=True):
    """2-component GMM on [Fe/H]; probability of the cluster-like component."""
    from scipy.stats import norm

    feh_g = np.asarray(d_good["FEH_CORRECTED"], float)
    m = np.isfinite(feh_g)
    print(f"{m.sum()} of {len(d_good)} RV-passed stars have [Fe/H]")

    gmm = GaussianMixture(n_components=2, random_state=0).fit(
        feh_g[m].reshape(-1, 1))
    means = gmm.means_.ravel()
    sigs = np.sqrt(gmm.covariances_.ravel())
    wts = gmm.weights_.ravel()
    stream_k = int(np.argmin(np.abs(means - gc_feh)))
    print(f"means: {np.round(means, 2)}  sigmas: {np.round(sigs, 2)}  "
          f"stream: {stream_k}")

    p_feh = np.full(len(d_good), np.nan)
    p_feh[m] = gmm.predict_proba(feh_g[m].reshape(-1, 1))[:, stream_k]
    d_good["P_feh"] = p_feh

    xx = np.linspace(np.nanmin(feh_g), np.nanmax(feh_g), 400)
    fig, ax = plt.subplots()
    ax.hist(feh_g[m], bins=20, density=True, alpha=0.4, color="grey")
    for k in range(2):
        ax.plot(xx, wts[k] * norm.pdf(xx, means[k], sigs[k]),
                c="crimson" if k == stream_k else "steelblue", lw=2)
    ax.axvline(gc_feh, ls=":", c="k")
    ax.set_xlabel("[Fe/H]")
    ax.set_title("RV-passed")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return d_good


# --------------------------------------------------------------------------- #
# Companion cluster (NGC 5053) spectroscopic confirmation
# --------------------------------------------------------------------------- #
def plot_rv_two_clusters(spec, gc, gc_label, gc2, gc2_label, show=True):
    """RV vs RA for the shared (M53) CMD+PM DESI sample near the M53/NGC 5053
    pair, coloured by NGC 5053 RV membership. Because the clusters overlap in
    CMD and PM, they separate here only in line-of-sight velocity: M53 forms a
    cloud around -63 km/s and NGC 5053 around +43 km/s.
    """
    ra = np.asarray(spec["ra"], float)
    rv = np.asarray(spec["RV"], float)
    p = np.asarray(spec["P_rv"], float)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(ra, rv, c=p, cmap="viridis", vmin=0, vmax=1,
                    s=28, edgecolor="k", lw=0.3, zorder=3)
    ax.axhline(gc2["VLOS"], ls="--", c="crimson", lw=1.5,
               label=f"{gc2_label} $v_{{los}}$ = {gc2['VLOS']:.0f} km/s")
    ax.axhline(gc["VLOS"], ls=":", c="navy", lw=1.5,
               label=f"{gc_label} $v_{{los}}$ = {gc['VLOS']:.0f} km/s")
    ax.invert_xaxis()
    # zoom on the two systemic velocities; bad DESI RVs (|v| up to ~1500 km/s)
    # sit far outside this window and already carry P_rv ~ 0.
    vlo = min(gc["VLOS"], gc2["VLOS"]) - 200
    vhi = max(gc["VLOS"], gc2["VLOS"]) + 200
    ax.set_ylim(vlo, vhi)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel(r"$v_{los}$ [km/s]")
    ax.set_title(f"{gc2_label}: RV confirmation (shared {gc_label} CMD+PM sample)")
    plt.colorbar(sc, ax=ax, label=f"P(RV | {gc2_label})")
    legend_outside(ax, fontsize=8)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def confirm_ngc5053(combined_iso, desi_mws, gc, gc_label, gc2,
                    gc2_label, gc2_feh, gc2_rt_pc, region_rt=20.0,
                    rv_tol=15.0, pfeh_min=None, show=True):
    """Spectroscopically confirm NGC 5053 members inside the shared M53 CMD+PM
    sample.

    NGC 5053 shares M53's CMD and proper motion, so its members already live in
    ``combined_iso``. They are isolated here by (a) a wide spatial region around
    NGC 5053 (``region_rt`` tidal radii) and (b) agreement with the NGC 5053
    systemic line-of-sight velocity, which differs from M53 by ~100 km/s.

    ``pfeh_min`` additionally requires ``P_feh > pfeh_min``, the same RV +
    [Fe/H] pair the NGC 5024 candidates are cut on (stars without a measured
    [Fe/H] have ``P_feh = NaN`` and are dropped). Left at None, RV alone
    decides and [Fe/H] is only reported.

    Returns ``(members, no_core)``: the confirmed members, and the same members
    with the cluster core (2 tidal radii) removed.
    """
    rt2_deg = tidal_radius_deg(gc2_rt_pc, gc2["DIST"])
    region_deg = region_rt * rt2_deg

    gc2_coord = SkyCoord(ra=gc2["RA"] * u.deg, dec=gc2["DEC"] * u.deg)
    obj_coord = SkyCoord(ra=np.asarray(combined_iso["ra"], float) * u.deg,
                         dec=np.asarray(combined_iso["dec"], float) * u.deg)
    region = combined_iso[gc2_coord.separation(obj_coord) < region_deg * u.deg]
    print(f"[NGC 5053] {len(region)} CMD+PM stars within {region_deg:.2f} deg "
          f"({region_rt:.0f} rt) of {gc2_label}")

    cand = crossmatch_desi(region, desi_mws)
    spec = cand[cand["has_desi"]]
    print(f"[NGC 5053] {len(spec)} of {len(region)} have DESI spectra")
    if len(spec) == 0:
        print("[NGC 5053] no DESI matches in region; skipping confirmation")
        return spec, spec

    # RV membership around the NGC 5053 systemic velocity (the discriminant).
    rv = np.asarray(spec["RV"], float)
    rve = np.asarray(spec["RV_ERR"], float)
    sig_v = np.hypot(rve, rv_tol)
    spec["P_rv"] = np.exp(-0.5 * ((rv - gc2["VLOS"]) / sig_v) ** 2)
    # [Fe/H] consistency -- both clusters are metal-poor, so this is a loose
    # sanity weight rather than the discriminant.
    feh = np.asarray(spec["FEH_CORRECTED"], float)
    spec["P_feh"] = np.where(np.isfinite(feh),
                             np.exp(-0.5 * ((feh - gc2_feh) / 0.5) ** 2), np.nan)

    plot_rv_two_clusters(spec, gc, gc_label, gc2, gc2_label, show=show)

    members = spec[spec["P_rv"] > 0.5]
    print(f"[NGC 5053] {len(members)} stars with P_rv > 0.5 around "
          f"v_los = {gc2['VLOS']:.1f} km/s")
    if pfeh_min is not None:
        keep_feh = np.asarray(members["P_feh"], float) > pfeh_min
        print(f"[NGC 5053] P_feh > {pfeh_min} around [Fe/H] = {gc2_feh:.2f}: "
              f"kept {int(keep_feh.sum())} / {len(members)} "
              f"({int(np.sum(~np.isfinite(np.asarray(members['P_feh'], float))))}"
              f" without a measured [Fe/H])")
        members = members[keep_feh]

    print(f"[NGC 5053] {len(spec)} DESI-matched in region")

    # confirmed members with the cluster core removed (for stream comparison).
    msep = gc2_coord.separation(
        SkyCoord(ra=np.asarray(members["ra"], float) * u.deg,
                 dec=np.asarray(members["dec"], float) * u.deg))
    no_core = members[msep > 2 * rt2_deg * u.deg]
    print(f"[NGC 5053] {len(no_core)} confirmed members excl. core")
    return members, no_core


# --------------------------------------------------------------------------- #
# LMS-1 (Wukong) appendix
# --------------------------------------------------------------------------- #
def load_lms1_template(streamfinder_file, ra_lim, dec_lim):
    """LMS-1 STREAMFINDER members inside the footprint, as a stream table.

    matched_filter_map / pm_weight expect uppercase RA, DEC, PMRA, PMDEC and a
    PM convention matching the observations. STREAMFINDER pmRA already includes
    cos(dec) (Gaia convention), the same as the Tractor pmra.
    """
    import pandas as pd

    df = pd.read_csv(streamfinder_file)
    lms = df[df["Name"] == "LMS-1"].copy()
    m = (lms.RAdeg.between(*ra_lim)) & (lms.DEdeg.between(*dec_lim))
    lms = lms[m].sort_values("RAdeg")
    tab = Table()
    tab["RA"] = np.asarray(lms.RAdeg, float)
    tab["DEC"] = np.asarray(lms.DEdeg, float)
    tab["PMRA"] = np.asarray(lms.pmRA, float)     # includes cos(dec)
    tab["PMDEC"] = np.asarray(lms.pmDE, float)
    print(f"LMS-1 template: {len(tab)} members in footprint "
          f"(pmRA med {np.median(tab['PMRA']):.2f}, "
          f"pmDE med {np.median(tab['PMDEC']):.2f})")
    return tab

def plot_sky_stage(tab, title, gc, gc_label, ra_lim, dec_lim, stream=None,
                   overlays=None,
                   style="density", bin_deg=0.5,
                   normalize_bg=True, bg_exclude_deg=2.0, bg_smooth_deg=5.0,
                   vmin=None, vmax=None, log=False, cmap="magma",
                   color_col=None, cbar_label=None, figsize=(9, 6.5)):
    """RA/Dec map of ONE selection stage, in the fixed footprint frame.

    ``style`` is ``"density"`` (a smoothed map, the default) or ``"scatter"``
    (one point per star). It is explicit rather than chosen from the star count,
    so two stages always render the same way and are comparable; a scatter
    silently ignores ``vmin`` / ``vmax`` and the background normalisation.

    The density map is a plain 2-D histogram on ``bin_deg`` pixels, not smoothed:
    ``bin_deg`` alone sets the resolution. Smoothing the counts would correlate
    neighbouring pixels and turn Poisson noise into blobs that read as structure. ``color_col`` colours either style by a column (e.g. a GMM
    ``P_stream``). The simulated-stream centreline and the cluster are marked,
    and the axes are pinned to ``ra_lim`` / ``dec_lim`` -- stars outside are
    dropped, so the printed count is what the map shows.

    Colour scale
    ------------
    ``vmin`` / ``vmax`` are in stars per smoothed bin. Left as None the scale
    runs from 0 to the densest bin, so nothing is clipped at either end. Set ``vmax`` by hand to bring out faint structure
    when the cluster core saturates the map, or to put two stages on a common
    scale. The printed ``[sky map]`` line reports the bin range and the limits in
    use. ``log=False`` switches to a linear stretch. For a ``color_col`` map,
    ``vmin`` / ``vmax`` default to 0/1 (probability-like columns).
    """
    ra = np.asarray(tab["ra"], float)
    dec = np.asarray(tab["dec"], float)
    good = (np.isfinite(ra) & np.isfinite(dec) &
            (ra >= ra_lim[0]) & (ra <= ra_lim[1]) &
            (dec >= dec_lim[0]) & (dec <= dec_lim[1]))
    ra, dec = ra[good], dec[good]

    fig, ax = plt.subplots(figsize=figsize)
    ra_bins = np.arange(ra_lim[0], ra_lim[1] + bin_deg, bin_deg)
    dec_bins = np.arange(dec_lim[0], dec_lim[1] + bin_deg, bin_deg)

    if color_col is not None:
        c = np.asarray(tab[color_col], float)[good]
        if style == "density":
            # too many points to scatter legibly: map the MEAN of color_col per
            # sky bin (bins with no finite value stay blank).
            fin = np.isfinite(c)
            tot, _, _ = np.histogram2d(ra[fin], dec[fin],
                                       bins=[ra_bins, dec_bins], weights=c[fin])
            cnt, _, _ = np.histogram2d(ra[fin], dec[fin], bins=[ra_bins, dec_bins])
            with np.errstate(invalid="ignore"):
                mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
            im = ax.pcolormesh(ra_bins, dec_bins, np.ma.masked_invalid(mean).T,
                               cmap="viridis",
                               vmin=0.0 if vmin is None else vmin,
                               vmax=1.0 if vmax is None else vmax,
                               shading="auto")
            fig.colorbar(im, ax=ax, shrink=0.85,
                         label=f"mean {cbar_label or color_col} / bin")
        else:
            order = np.argsort(np.nan_to_num(c))
            sc = ax.scatter(ra[order], dec[order], c=c[order], s=10,
                            cmap="viridis", vmin=0.0 if vmin is None else vmin,
                            vmax=1.0 if vmax is None else vmax, lw=0)
            fig.colorbar(sc, ax=ax, shrink=0.85, label=cbar_label or color_col)
    elif style == "density":
        H, _, _ = np.histogram2d(ra, dec, bins=[ra_bins, dec_bins])

        if normalize_bg and stream is not None and len(stream):
            # background surface from off-stream pixels only, interpolated under
            # the stream by normalised convolution (as density_profile_figure).
            tx, ty = binned_track(np.asarray(stream["RA"], float),
                                  np.asarray(stream["DEC"], float), min_count=3)
            order = np.argsort(tx)
            tx, ty = tx[order], ty[order]
            xc = 0.5 * (ra_bins[:-1] + ra_bins[1:])
            yc = 0.5 * (dec_bins[:-1] + dec_bins[1:])
            trk = np.interp(xc, tx, ty, left=ty[0], right=ty[-1])
            resid = yc[None, :] - trk[:, None]              # (n_ra, n_dec)
            bgmask = (np.abs(resid) >= bg_exclude_deg).astype(float)

            sig = bg_smooth_deg / bin_deg
            num = gaussian_filter(H * bgmask, sigma=sig)
            den = gaussian_filter(bgmask, sigma=sig)
            bg_surface = num / np.maximum(den, 1e-12)
            floor = (max(np.median(bg_surface[bg_surface > 0]) * 1e-2, 1e-9)
                     if np.any(bg_surface > 0) else 1e-9)
            Hdisp = H / np.maximum(bg_surface, floor)
            cbar = "density / background"
            print(f"[sky map] background fitted on {bgmask.mean():.0%} of pixels "
                  f"(> {bg_exclude_deg} deg off track), smoothed {bg_smooth_deg} deg")
        else:
            Hdisp = H
            cbar = "stars / bin"

        pos = Hdisp[Hdisp > 0]
        lo = vmin if vmin is not None else (0.0 if not normalize_bg else
                                            float(np.nanmin(pos)) if pos.size else 0.0)
        hi = vmax if vmax is not None else (float(Hdisp.max()) if pos.size else 1.0)
        if log:
            lo = max(lo, 1e-2)       # LogNorm cannot start at 0
        if hi <= lo:
            hi = lo * 1.5 if log else lo + 1.0
        print(f"[sky map] raw bins {H.min():.3g}-{H.max():.3g} stars "
              f"({bin_deg} deg pixels); displayed {Hdisp.min():.3g}-"
              f"{Hdisp.max():.3g}; colour scale {lo:.3g}-{hi:.3g}"
              + ("" if vmin is None and vmax is None else "  [set by hand]"))
        norm = LogNorm(vmin=lo, vmax=hi) if log else None
        im = ax.pcolormesh(ra_bins, dec_bins, Hdisp.T, cmap=cmap, norm=norm,
                           vmin=None if log else lo, vmax=None if log else hi,
                           shading="auto")
        fig.colorbar(im, ax=ax, shrink=0.85, label=cbar)
    else:
        ax.set_facecolor("0.12")
        ax.scatter(ra, dec, s=6, c="gold", alpha=0.8, lw=0)

    if stream is not None and len(stream):
        sx, sy = binned_track(np.asarray(stream["RA"], float),
                              np.asarray(stream["DEC"], float), min_count=3)
        ax.plot(sx, sy, c="deepskyblue", lw=1.4, ls="--", zorder=4,
                label="simulated stream")
    for ov in (overlays or []):
        ot = ov["tab"]
        if ot is None or not len(ot):
            continue
        ox, oy = binned_track(np.asarray(ot["RA"], float),
                              np.asarray(ot["DEC"], float), min_count=3)
        ax.plot(ox, oy, c=ov.get("color", "lime"), lw=ov.get("lw", 1.4),
                ls=ov.get("ls", "--"), zorder=4, label=ov.get("label"))

    ax.scatter([gc["RA"]], [gc["DEC"]], marker="*", s=260, c="cyan",
               edgecolor="k", zorder=5, label=gc_label)
    ax.set_xlim(ra_lim[1], ra_lim[0])      # RA increases to the left
    ax.set_ylim(*dec_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(f"{title}\n{ra.size:,} stars")
    fig.tight_layout()
    legend_outside(ax, fontsize=9)
    plt.show()
    return fig

def pm_gmm_step(sample, sim, gc, gc_label, gmax=21.0, n_components=4,
                cov_inflation=1.0, min_weight=0.01,
                xlim=(-10, 0), ylim=(-10, 0)):
    """Fit the PM GMM on the on-stream stars, score the sample, plot the plane.

    The mixture is fitted on the on-stream (``region == "target"``) stars
    brighter than ``gmax`` -- where the stream signal is and the proper motions
    are still precise -- with the simulated stream's own PM distribution frozen
    in as a known component, then evaluated over every star in ``sample``. Adds
    ``P_stream`` and returns ``(sample, result)``.

    The two-panel diagnostic shows the fitted components with their 2-sigma
    ellipses (the stream one in magenta) and the per-star P(stream).

    The fit can be fragile: including a faint tail with large PM errors, or
    changing ``n_components`` / ``cov_inflation``, can flip the stream component
    between a real weight and a collapse to ``min_weight`` (P ~ 0 everywhere).
    The stream weight is printed, with a warning when it collapses.
    """
    gm = np.asarray(sample["gmag"], float)
    fit_m = (np.asarray(sample["region"]) == "target")
    if gmax is not None:
        fit_m &= np.isfinite(gm) & (gm < gmax)
    fit_tab = sample[fit_m]
    print(f"[gmm] fitting on {len(fit_tab)} stars"
          + (f" (g < {gmax})" if gmax is not None else "")
          + f" of {len(sample)}")

    result = fit_stream_gmm(
        fit_tab,
        known_components=[{"name": f"{gc_label}_sim", "pmra": sim["PMRA"],
                           "pmdec": sim["PMDEC"]}],
        stream_name=f"{gc_label}_sim", n_components=n_components,
        cov_inflation=cov_inflation, min_weight=min_weight,
        gc_pmra=gc["PMRA"], gc_pmdec=gc["PMDEC"])

    sample = sample.copy()
    sample["P_stream"] = gmm_stream_prob(result, sample["pmra"], sample["pmdec"])
    p = np.asarray(sample["P_stream"], float)
    w_stream = float(result["weights"][result["stream_idx"]])
    print(f"[gmm] stream component weight = {w_stream:.3f}; "
          f"P_stream > 0.5: {int(np.nansum(p > 0.5)):,} / {len(sample):,}")
    if w_stream <= min_weight * 1.01:
        print("[gmm] WARNING: stream component collapsed to min_weight -- the "
              "mixture found no PM-separable stream, so P_stream is meaningless.")

    # ---- figure ---------------------------------------------------------
    pmra = np.asarray(fit_tab["pmra"], float)
    pmdec = np.asarray(fit_tab["pmdec"], float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=120)

    ax = axes[0]
    ax.scatter(pmra, pmdec, s=0.3, alpha=0.25, c="k")
    ax.scatter(np.asarray(sim["PMRA"], float), np.asarray(sim["PMDEC"], float),
               s=2, c="blue", label="simulated stream")
    ax.scatter(*result["gc_anchor"], s=120, marker="X", c="red",
               edgecolors="k", zorder=5, label="anchor (GC)")
    for k, (mu, cov, w, nm) in enumerate(zip(result["means"], result["covs"],
                                             result["weights"], result["names"])):
        is_s = (k == result["stream_idx"])
        ax.scatter(*mu, marker="*" if is_s else "o", s=140 if is_s else 50,
                   c="yellow" if is_s else "orange", edgecolors="k", zorder=5,
                   label=f"{nm}  (w={w:.2f})")
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        ax.add_patch(Ellipse(
            xy=mu, width=2 * 2.0 * np.sqrt(vals[0]),
            height=2 * 2.0 * np.sqrt(vals[1]),
            angle=np.degrees(np.arctan2(*vecs[:, 0][::-1])), fill=False, lw=1.5,
            color="magenta" if is_s else "gray", alpha=0.9 if is_s else 0.5))
    ax.set(xlim=xlim, ylim=ylim, xlabel="pmra [mas/yr]", ylabel="pmdec [mas/yr]",
           title=f"{gc_label}: GMM (K={len(result['means'])})")

    ax = axes[1]
    pfit = result["stream_prob"]
    order = np.argsort(np.nan_to_num(pfit))
    sc = ax.scatter(pmra[order], pmdec[order], s=0.6, c=pfit[order],
                    cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="P(stream member)")
    ax.scatter(*result["gc_anchor"], s=120, marker="X", c="red",
               edgecolors="k", zorder=5)
    ax.set(xlim=xlim, ylim=ylim, xlabel="pmra [mas/yr]", ylabel="pmdec [mas/yr]",
           title="Per-star P(stream)")

    fig.tight_layout()
    legend_outside(axes[0], where="below", fontsize=7, ncol=2)
    return sample, result