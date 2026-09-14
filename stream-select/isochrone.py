import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils import mist_isochrone, legend_outside

BRICK_AREA = 0.25 * 0.25


def isochrone_select(combined, mag_bright_lim, mag_faint_lim, iso_tol_minus,
                     iso_tol_plus, gc_label, age_gyr, feh,
                     filters=("DECam_g", "DECam_r"), gr_lim=(0.0, 0.8),
                     hess_bins=(180, 180), cmap="Greys", vmax=None,
                     overlays=None):
    """Distance-resolved MIST-isochrone CMD selection, and the figure of it.

    Each star is judged against the isochrone placed at its own phi1-bin
    distance (needs ``dist_ref_kpc`` from :func:`assign_dist_bins` and the
    ``gmag`` / ``grcolor`` photometry attached by :func:`stellar_locus_cut`).
    The colour band is ``-iso_tol_minus`` / ``+iso_tol_plus`` about the ridge,
    the same band the figure draws. Stars outside the g-band limits are cut
    first. Adds ``keep_iso`` and returns
    ``(tagged_table, iso_g_abs, iso_r_abs, fig)``.

    ``overlays`` draws extra isochrones that are NOT part of the cut, each a
    dict with ``g_abs``, ``r_abs``, ``dist_kpc`` and optionally ``label``,
    ``color``, ``ls``, ``lw``.
    """
    combined = combined.copy()
    g = np.asarray(combined["gmag"], float)
    r = g - np.asarray(combined["grcolor"], float)
    gr = g - r

    iso, iso_mask, iso_g_abs, iso_r_abs = mist_isochrone(age_gyr, feh, filters)

    with np.errstate(invalid="ignore"):
        finite = np.isfinite(g) & np.isfinite(r)
        in_maglim = (g > mag_bright_lim) & (g < mag_faint_lim)
        good = finite & in_maglim & (gr < 1)

    print(f"g magnitude limits [{mag_bright_lim}, {mag_faint_lim}]: "
          f"rejected {int((finite & ~in_maglim).sum())} of {int(finite.sum())} stars")

    dist_ref = np.asarray(combined["dist_ref_kpc"], float)
    keep = np.zeros(len(combined), bool)

    for d in np.unique(dist_ref[np.isfinite(dist_ref)]):
        m = dist_ref == d
        distmod = 5 * np.log10(d * 1000) - 5

        # isochrone ridge in apparent mags, sorted for interpolation
        g_iso = iso["DECam_g"][iso_mask] + distmod
        r_iso = iso["DECam_r"][iso_mask] + distmod
        order = np.argsort(g_iso)
        g_iso_s = g_iso[order]
        gr_iso_s = (g_iso - r_iso)[order]

        # brighter than the RGB tip: clamp to the tip colour rather than reject
        gr_pred = np.interp(g[m], g_iso_s, gr_iso_s, left=gr_iso_s[0], right=np.nan)
        dcolor = gr[m] - gr_pred          # signed: negative = bluer
        keep[m] = (np.isfinite(gr_pred) & (dcolor >= -iso_tol_minus)
                   & (dcolor <= iso_tol_plus))

        visible = ((iso_g_abs + distmod > mag_bright_lim) &
                   (iso_g_abs + distmod < mag_faint_lim))
        print(f"  d = {d:5.1f} kpc  (m-M = {distmod:5.2f}): "
              f"{visible.mean():5.1%} of isochrone visible, "
              f"probes to M_g = {mag_faint_lim - distmod:5.2f}")

    keep &= good
    combined["keep_iso"] = keep
    print(f"selected {int(keep.sum())} / {int(good.sum())} stars near isochrone "
          f"(distance-resolved)")

    # ---- figure ---------------------------------------------------------
    g_lim = (mag_bright_lim - 0.5, mag_faint_lim + 0.5)
    bin_ids = sorted(np.unique(combined["dist_bin"]))
    bin_dist = {b: float(np.nanmedian(dist_ref[combined["dist_bin"] == b]))
                for b in bin_ids}
    finite_d = [d for d in bin_dist.values() if np.isfinite(d)]
    mid = bin_ids[len(bin_ids) // 2]

    xedges = np.linspace(gr_lim[0], gr_lim[1], hess_bins[0] + 1)
    yedges = np.linspace(g_lim[0], g_lim[1], hess_bins[1] + 1)
    ok = np.isfinite(gr) & np.isfinite(g)
    H, _, _ = np.histogram2d(gr[ok], g[ok], bins=[xedges, yedges])

    fig, ax = plt.subplots(figsize=(6.5, 8))
    hi = float(H.max()) if vmax is None else float(vmax)
    im = ax.pcolormesh(xedges, yedges, H.T, cmap=cmap,
                       norm=LogNorm(vmin=1, vmax=max(hi, 2)))
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="stars / bin")

    for b in bin_ids:
        d = bin_dist[b]
        if not np.isfinite(d):
            continue
        dm = 5 * np.log10(d * 1000) - 5
        g_app = iso_g_abs + dm
        gr_app = g_app - (iso_r_abs + dm)
        if b == mid:
            ax.plot(gr_app, g_app, c="red", lw=2.0, zorder=6,
                    label=f"{gc_label}: MIST {age_gyr:.1f} Gyr, "
                          f"[Fe/H]={feh:.2f} @ {d:.1f} kpc")
            ax.plot(gr_app - iso_tol_minus, g_app, c="red", lw=1.2, ls="--",
                    zorder=6, label=f"selection band "
                                    f"(-{iso_tol_minus:.2f} / +{iso_tol_plus:.2f})")
            ax.plot(gr_app + iso_tol_plus, g_app, c="red", lw=1.2, ls="--", zorder=6)
        else:
            ax.plot(gr_app, g_app, c="orangered", lw=0.8, alpha=0.45, zorder=5,
                    label="other distance bins" if b == bin_ids[0] else None)

    for ov in (overlays or []):
        d = float(ov["dist_kpc"])
        dm = 5 * np.log10(d * 1000) - 5
        g_app = np.asarray(ov["g_abs"], float) + dm
        gr_app = g_app - (np.asarray(ov["r_abs"], float) + dm)
        ax.plot(gr_app, g_app, c=ov.get("color", "dodgerblue"),
                lw=ov.get("lw", 1.8), ls=ov.get("ls", "-"), zorder=7,
                label=f"{ov.get('label', 'overlay')} @ {d:.1f} kpc")

    ax.set_xlim(*gr_lim)
    ax.set_ylim(g_lim[1], g_lim[0])
    ax.set_xlabel("(g - r)$_0$")
    ax.set_ylabel("g$_0$")
    sub = (f"{len(bin_ids)} phi1 bins, "
           f"{min(finite_d):.1f}-{max(finite_d):.1f} kpc" if finite_d else "")
    ax.set_title(f"{gc_label}: distance-resolved isochrone selection\n"
                 f"{sub} -- {int(keep.sum()):,} stars kept "
                 f"of {int(ok.sum()):,} with g, r", fontsize=10)
    fig.tight_layout()
    legend_outside(ax, fontsize=8)
    plt.show()

    return combined, iso_g_abs, iso_r_abs, fig


def cmd_hess_regions(sample, iso_g_abs, iso_r_abs, gc_label, iso_tol_minus,
                     iso_tol_plus, mag_bright_lim, mag_faint_lim,
                     gr_lim=(0.0, 0.8), hess_bins=(49, 49),
                     hess_vmax=None, diff_vmin=None, diff_vmax=None,
                     fallback_kpc=None, overlays=None):
    """Per phi1-bin Hess CMD: target, background, and their difference.

    One 4-panel row per phi1 distance bin -- target CMD with the isochrone band,
    target Hess, background Hess, and the background-subtracted difference --
    each drawn with the isochrone at THAT bin's distance. Every target and
    background star enters the panels, inside the band or not, so the band is
    drawn over the full CMD; in the CMD panel the stars passing ``keep_iso``
    are black and the ones outside the band grey. Needs a ``region``
    column (target / background / none, from :func:`assign_stream_regions`) and
    the ``dist_bin`` / ``dist_ref_kpc`` columns from :func:`assign_dist_bins`.

    The colour band is the same ``-iso_tol_minus`` / ``+iso_tol_plus`` band the
    Colour scales, all in stars/deg2. ``hess_vmax`` caps the Target and
    Background greyscales; left at None each panel uses its own maximum, so the
    two are not directly comparable -- set it to compare them by eye.
    ``diff_vmin`` / ``diff_vmax`` bound the difference panel; left at None they
    are symmetric about zero at that row's largest absolute residual, so the
    diverging colormap is centred and a deficit is visible as well as an
    excess. The printed per-row line reports the ranges actually present.

    ``overlays`` takes
    the same dicts as :func:`isochrone_select` and draws each companion
    isochrone on every panel, at its own fixed distance.
    """
    reg = np.asarray(sample["region"])
    tgt_all = sample[reg == "target"]
    bg_all = sample[reg == "background"]
    print(f"[hess] target {len(tgt_all)} "
          f"({int(np.asarray(tgt_all['keep_iso'], bool).sum())} in band), "
          f"background {len(bg_all)} "
          f"({int(np.asarray(bg_all['keep_iso'], bool).sum())} in band)")
    tb = np.asarray(tgt_all["dist_bin"], int)
    bins = [b for b in sorted(np.unique(np.asarray(sample["dist_bin"], int)))
            if (tb == b).any()]
    if not bins:
        print("[hess] no target stars in any bin")
        return None

    global_dist = float(np.nanmedian(np.asarray(tgt_all["dist_ref_kpc"], float)))
    if not np.isfinite(global_dist):
        global_dist = fallback_kpc if fallback_kpc is not None else 10.0

    g_lim = (mag_bright_lim, mag_faint_lim)
    xbins = np.linspace(gr_lim[0], gr_lim[1], hess_bins[0] + 1)
    ybins = np.linspace(g_lim[0], g_lim[1], hess_bins[1] + 1)

    fig, axes = plt.subplots(len(bins), 4, figsize=(18, 4.2 * len(bins)),
                             squeeze=False)
    for grow, b in enumerate(bins):
        tgt = tgt_all[tb == b]
        bg = bg_all[np.asarray(bg_all["dist_bin"], int) == b]
        tgt_area = max(len(np.unique(tgt["brickid"])) * BRICK_AREA, 1e-9)
        bg_area = max(len(np.unique(bg["brickid"])) * BRICK_AREA, 1e-9)

        d = float(np.nanmedian(np.asarray(tgt["dist_ref_kpc"], float)))
        if not np.isfinite(d):
            d = global_dist
        dm = 5 * np.log10(d * 1000) - 5
        g_app = iso_g_abs + dm
        gr_app = g_app - (iso_r_abs + dm)

        H_tgt, _, _ = np.histogram2d(np.asarray(tgt["grcolor"], float),
                                     np.asarray(tgt["gmag"], float),
                                     bins=[xbins, ybins])
        H_bg, _, _ = np.histogram2d(np.asarray(bg["grcolor"], float),
                                    np.asarray(bg["gmag"], float),
                                    bins=[xbins, ybins])
        H_tgt = H_tgt.T / tgt_area
        H_bg = H_bg.T / bg_area
        H_diff = H_tgt - H_bg

        def band(ax, color):
            ax.plot(gr_app, g_app, color=color, lw=1.5)
            ax.plot(gr_app - iso_tol_minus, g_app, color=color, lw=0.8,
                    ls="--", alpha=0.7)
            ax.plot(gr_app + iso_tol_plus, g_app, color=color, lw=0.8,
                    ls="--", alpha=0.7)
            for ov in (overlays or []):
                ov_dm = 5 * np.log10(float(ov["dist_kpc"]) * 1000) - 5
                ov_g = np.asarray(ov["g_abs"], float) + ov_dm
                ov_gr = ov_g - (np.asarray(ov["r_abs"], float) + ov_dm)
                ax.plot(ov_gr, ov_g, color=ov.get("color", "dodgerblue"),
                        lw=ov.get("lw", 1.2), ls=ov.get("ls", "-"), alpha=0.9)

        ax = axes[grow]
        in_band = np.asarray(tgt["keep_iso"], bool)
        ax[0].scatter(tgt["grcolor"][~in_band], tgt["gmag"][~in_band],
                      s=0.05, color="0.6")
        ax[0].scatter(tgt["grcolor"][in_band], tgt["gmag"][in_band],
                      s=0.05, color="k")
        band(ax[0], "red")
        ax[0].set_title(f"CMD  bin {b}", fontsize=10)

        tgt_hi = hess_vmax if hess_vmax is not None else max(H_tgt.max(), 1)
        bg_hi = hess_vmax if hess_vmax is not None else max(H_bg.max(), 1)
        dmax = float(np.nanmax(np.abs(H_diff))) if H_diff.size else 1.0
        dlo = diff_vmin if diff_vmin is not None else 0.0
        dhi = diff_vmax if diff_vmax is not None else dmax
        print(f"  bin {b}: target max {H_tgt.max():.2f}, "
              f"background max {H_bg.max():.2f}, "
              f"diff [{H_diff.min():.2f}, {H_diff.max():.2f}] stars/deg2")

        ax[1].pcolormesh(xbins, ybins, H_tgt, cmap="Greys",
                         vmin=0, vmax=tgt_hi)
        band(ax[1], "red")
        ax[1].set_title(f"Target  ({tgt_area:.2f} deg2, {len(tgt)} st)", fontsize=10)

        ax[2].pcolormesh(xbins, ybins, H_bg, cmap="Greys",
                         vmin=0, vmax=bg_hi)
        band(ax[2], "red")
        ax[2].set_title(f"Background  ({bg_area:.2f} deg2, {len(bg)} st)", fontsize=10)

        im = ax[3].pcolormesh(xbins, ybins, H_diff, cmap="RdBu_r",
                              vmin=dlo, vmax=dhi)
        band(ax[3], "lime")
        ax[3].set_title(f"Hess diff  {d:.1f} kpc", fontsize=10)
        plt.colorbar(im, ax=ax[3], label="stars / deg2", fraction=0.046, pad=0.02)

        for a in ax:
            a.set_xlim(*gr_lim)
            a.set_ylim(g_lim[1], g_lim[0])
            a.set_xlabel("(g - r)$_0$")
            a.set_ylabel("g$_0$")

    fig.suptitle(f"{gc_label}: Hess CMD per distance bin "
                 f"(target/background from density-profile regions)",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    plt.show()
    return fig