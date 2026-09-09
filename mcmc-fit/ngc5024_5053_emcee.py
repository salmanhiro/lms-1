"""
Full 6D phase-space MCMC for NGC 5024 (M53) and NGC 5053
using values from Baumgardt's globular cluster catalog
(https://people.smp.uq.edu.au/HolgerBaumgardt/globular/orbits.html).

Samples 12 parameters (X,Y,Z,vX,vY,vZ for each cluster) with
independent Gaussian likelihoods, then derives:
  - 3D spatial separation  dr   [kpc]
  - 3D relative velocity   dv   [km/s]
  - Approx. escape velocity given combined mass (point-mass check)
  - P(pair is currently bound) under that toy approximation

Note: a proper bound test needs the Milky Way tidal field; this toy
escape-velocity check is only a crude indicator.
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner


# Baumgardt catalog values (means & 1-sigma), kpc and km/s
# Naming: <quantity>_<cluster>.  vX,vY,vZ are galactocentric Cartesian
# velocities (the catalog's U,V,W columns), NOT heliocentric UVW.

#                        X      Y      Z      vX      vY      vZ
mu_5024  = np.array([ 5.25, -1.49, 18.20, -57.27, 155.76, -72.49])
sig_5024 = np.array([ 0.03,  0.01,  0.18,   0.41,   0.41,   0.26])
mu_5053  = np.array([ 5.11, -1.38, 17.21, -52.79, 150.82,  35.30])
sig_5053 = np.array([ 0.04,  0.02,  0.23,   0.58,   0.58,   0.27])

mu  = np.concatenate([mu_5024, mu_5053])        # (12,)
sig = np.concatenate([sig_5024, sig_5053])

labels = ["X_5024", "Y_5024", "Z_5024", "vX_5024", "vY_5024", "vZ_5024",
          "X_5053", "Y_5053", "Z_5053", "vX_5053", "vY_5053", "vZ_5053"]

# Approximate cluster masses (Baumgardt & Hilker 2018), solar masses
M_5024 = 5.02e5
M_5053 = 6.28e4
G      = 4.30091e-6   # kpc (km/s)^2 / Msun


# Jacobi (tidal) radii.
# >>> REPLACE with the rJ column from Baumgardt's fundamental-parameters
# >>> table.  Values below are order-of-magnitude placeholders in pc.

rJ_5024_pc = 183.4
rJ_5053_pc = 91.64

rJ_5024 = rJ_5024_pc / 1000.0     # kpc
rJ_5053 = rJ_5053_pc / 1000.0     # kpc
rJ_max  = max(rJ_5024, rJ_5053)   # larger Jacobi radius (NGC 5024)
rJ_sum  = rJ_5024 + rJ_5053       # tidal spheres touch below this


# Posterior

def log_prob(theta):
    if np.any(np.abs(theta - mu) > 50 * sig):
        return -np.inf
    return -0.5 * np.sum(((theta - mu) / sig) ** 2)

ndim, nwalkers = 12, 64
nburn, nsteps  = 1500, 5000
rng = np.random.default_rng(1)
p0 = mu + sig * rng.normal(size=(nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
print("Burn-in ...")
state = sampler.run_mcmc(p0, nburn, progress=True)
sampler.reset()
print("Production ...")
sampler.run_mcmc(state, nsteps, progress=True)
print(f"Acceptance: {np.mean(sampler.acceptance_fraction):.3f}")

try:
    tau = sampler.get_autocorr_time(quiet=True)
    print("Autocorrelation times:")
    for lab, t in zip(labels, tau):
        print(f"  {lab:>9} : {t:6.1f}   (N/tau = {nsteps/t:6.1f})")
except Exception as e:
    print("autocorr warn:", e)

flat = sampler.get_chain(flat=True)          # (N, 12)


# Derived quantities

dr   = flat[:, :3] - flat[:, 6:9]
dv   = flat[:, 3:6] - flat[:, 9:12]
sep  = np.linalg.norm(dr, axis=1)            # kpc
vrel = np.linalg.norm(dv, axis=1)            # km/s

# toy two-body escape speed at current separation
v_esc = np.sqrt(2 * G * (M_5024 + M_5053) / sep)
bound = vrel < v_esc

def summary(x, name, unit):
    m = np.median(x)
    p16, p84   = np.percentile(x, [16, 84])
    p025, p975 = np.percentile(x, [2.5, 97.5])
    print(f"{name:>10} = {m:7.3f}  +{p84-m:.3f} / -{m-p16:.3f}  "
          f"[95%: {p025:.3f}, {p975:.3f}] {unit}")

print("\nPosterior summaries:")
summary(sep,   "dr",    "kpc")
summary(vrel,  "dv",    "km/s")
summary(v_esc, "v_esc", "km/s  (toy point-mass)")

thr_r = [0.2, 0.5, 1.0, 1.5, 2.0]
thr_v = [5, 10, 20, 50]
for thr in thr_r:
    print(f"P(dr < {thr:>4} kpc) = {(sep < thr).mean():.4f}")
for thr in thr_v:
    print(f"P(dv < {thr:>4} km/s) = {(vrel < thr).mean():.4f}")
print(f"P(currently bound, toy)  = {bound.mean():.4f}")

def tail_prob(thr):
    """Fraction of posterior samples below thr, with MC (Poisson) error."""
    k = int((sep < thr).sum())
    n = sep.size
    return k / n, np.sqrt(max(k, 1)) / n, k

p_max, e_max, k_max = tail_prob(rJ_max)
p_sum, e_sum, k_sum = tail_prob(rJ_sum)

print(f"\nTidal-overlap check (rJ):")
print(f"  rJ(NGC 5024)      = {rJ_5024:.3f} kpc")
print(f"  rJ(NGC 5053)      = {rJ_5053:.3f} kpc")
print(f"  rJ max (5024)     = {rJ_max:.3f} kpc")
print(f"  rJ sum (contact)  = {rJ_sum:.3f} kpc")
print(f"  P(dr < rJ max)    = {p_max:.5f} +/- {e_max:.5f}   ({k_max} samples)")
print(f"  P(dr < rJ sum)    = {p_sum:.5f} +/- {e_sum:.5f}   ({k_sum} samples)")


# Plots: dr histogram, dr CDF, dv histogram, joint dr-dv
# Two threshold colors, checked for colour-vision-deficient separation
# (OKLab dE: 30 normal, 23 deuteranope, 19 protanope).
C_MAX = "#1f77b4"     # dr < rJ(NGC 5024)  -> 5053 inside 5024's Jacobi radius
C_SUM = "#d95f02"     # dr < rJ_5024+rJ_5053 -> tidal spheres in contact

med = np.median(sep)
p16, p84 = np.percentile(sep, [16, 84])

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (0,0) separation histogram with median and 68% band
ax[0, 0].hist(sep, bins=120, density=True, color="steelblue", alpha=0.85)
ax[0, 0].axvline(med, color="k", ls="--", label=f"median = {med:.2f} kpc")
ax[0, 0].axvspan(p16, p84, color="orange", alpha=0.25, label="68% CI")
ax[0, 0].set(xlabel=r"$\Delta r$ (kpc)", ylabel="posterior PDF",
             title="Spatial separation")
ax[0, 0].legend()

# (0,1) cumulative probability with tidal radii marked
ss  = np.sort(sep)
cdf = np.arange(1, len(ss) + 1) / len(ss)
ax[0, 1].plot(ss, cdf, color="firebrick", lw=1.8)
ax[0, 1].set(xlabel=r"$\Delta r$ (kpc)", ylabel=r"P($\Delta r < s$)",
             title="Cumulative probability")
ax[0, 1].grid(alpha=0.3)
ax[0, 1].set_ylim(0, 1.05)

for thr in thr_r:
    ax[0, 1].axvline(thr, color="gray", ls=":", lw=0.9)
    ax[0, 1].text(thr, 0.04, f"{(sep < thr).mean():.1%}",
                  rotation=90, va="bottom", fontsize=8, color="gray")

tidal_marks = [
    (rJ_5053, f"$r_J$ 5053 = {rJ_5053:.2f}", "tab:purple"),
    (rJ_max,  f"$r_J$ max (5024) = {rJ_max:.2f}", C_MAX),
    (rJ_sum,  f"$r_J$ sum = {rJ_sum:.2f}",   C_SUM),
]
for xv, lab, c in tidal_marks:
    ax[0, 1].axvline(xv, color=c, ls="-.", lw=1.4, alpha=0.9)
    ax[0, 1].text(xv, 0.62, lab, rotation=90, va="bottom",
                  ha="right", fontsize=8, color=c)
ax[0, 1].axvspan(0, rJ_sum, color=C_SUM, alpha=0.08)
ax[0, 1].axvspan(0, rJ_max, color=C_MAX, alpha=0.10)

# (1,0) relative velocity histogram
ax[1, 0].hist(vrel, bins=120, density=True, color="seagreen", alpha=0.85)
mv = np.median(vrel)
ax[1, 0].axvline(mv, color="k", ls="--", label=f"median = {mv:.1f} km/s")
ax[1, 0].set(xlabel=r"$\Delta v$ (km/s)", ylabel="posterior PDF",
             title="Relative velocity")
ax[1, 0].legend()

# (1,1) joint dr - dv
h = ax[1, 1].hexbin(sep, vrel, gridsize=60, cmap="viridis", mincnt=1)
ax[1, 1].set(xlabel=r"$\Delta r$ (kpc)", ylabel=r"$\Delta v$ (km/s)",
             title="Joint phase-space separation")
plt.colorbar(h, ax=ax[1, 1], label="counts")

plt.tight_layout()
plt.savefig("ngc5024_5053_6d_pdfs.png", dpi=130)


# Tidal-contact probability: zoom on the small-separation tail
# Left  : posterior CDF over the tidal-radius range (log axis - the tail is small)
# Right : the two criteria as probabilities, with Monte-Carlo error bars

crit = [
    ("tidal spheres in contact\n" r"$\Delta r < r_J(5024)+r_J(5053)$",
     rJ_sum, p_sum, e_sum, C_SUM),
    ("5053 inside 5024's Jacobi radius\n" r"$\Delta r < r_J(5024)$",
     rJ_max, p_max, e_max, C_MAX),
]

figt, axt = plt.subplots(1, 2, figsize=(12.5, 4.8))

# (0) CDF of the tail, log-y so a rare overlap is still readable
xmax   = max(4 * rJ_sum, np.percentile(sep, 1))
p_floor = 1.0 / sep.size                      # one-sample resolution limit
mask   = ss <= xmax
axt[0].plot(ss[mask], np.maximum(cdf[mask], p_floor / 2),
            color="0.25", lw=2.0)
axt[0].set_yscale("log")
axt[0].set_ylim(p_floor / 2, 1.0)
axt[0].set_xlim(0, xmax)
axt[0].set(xlabel=r"separation $s$ (kpc)",
           ylabel=r"P($\Delta r < s$)",
           title="Small-separation tail of the posterior")
axt[0].grid(alpha=0.25, which="both")
axt[0].axhline(p_floor, color="0.6", ls=":", lw=1.0)
axt[0].text(xmax, p_floor, f" MC floor (1/{sep.size})", fontsize=7,
            color="0.5", va="bottom", ha="right")

for lab, xv, pv, ev, c in crit:
    axt[0].axvline(xv, color=c, ls="-.", lw=1.8, alpha=0.9)
    axt[0].plot([xv], [max(pv, p_floor / 2)], "o", ms=9, color=c,
                mec="white", mew=2, zorder=5)
    axt[0].annotate(f"{lab.splitlines()[0]}\nP = {pv:.2e}",
                    xy=(xv, max(pv, p_floor / 2)),
                    xytext=(12, -26), textcoords="offset points",
                    fontsize=8, color=c, ha="left",
                    bbox=dict(fc="white", ec=c, lw=0.6, alpha=0.9, pad=2.5),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8))

# (1) the two criteria side by side, with MC uncertainty
ypos = np.arange(len(crit))
for y, (lab, xv, pv, ev, c) in zip(ypos, crit):
    axt[1].barh(y, pv, height=0.32, color=c, alpha=0.9,
                xerr=ev, error_kw=dict(ecolor="0.3", capsize=4, lw=1.2))
    axt[1].text(pv + ev, y, f"  {pv:.3e}  ({int(round(pv*sep.size))} samples)",
                va="center", fontsize=9, color="0.2")
axt[1].set_yticks(ypos)
axt[1].set_yticklabels([lab for lab, *_ in crit], fontsize=8)
axt[1].set_ylim(len(crit) - 0.6, -0.6)
axt[1].set_xlim(0, max(p_sum + e_sum, p_floor) * 1.9)
axt[1].set(xlabel="posterior probability",
           title="Probability the tidal radii touch")
axt[1].grid(alpha=0.25, axis="x")
for sp in ("top", "right", "left"):
    axt[1].spines[sp].set_visible(False)

figt.tight_layout()
figt.savefig("ngc5024_5053_tidal_contact.png", dpi=130)


# Corner plots: raw sampled parameters, and derived quantities
fig_raw = corner.corner(flat, labels=labels, show_titles=True,
                        title_fmt=".3f", label_kwargs={"fontsize": 8},
                        title_kwargs={"fontsize": 7})
fig_raw.savefig("ngc5024_5053_6d_corner_params.png", dpi=110)

derived = np.column_stack([sep, vrel, v_esc])
fig_der = corner.corner(derived,
                        labels=[r"$\Delta r$ [kpc]",
                                r"$\Delta v$ [km/s]",
                                r"$v_{\rm esc}$ [km/s]"],
                        show_titles=True, title_fmt=".2f")
fig_der.savefig("ngc5024_5053_6d_corner_derived.png", dpi=110)
print("Saved plots.")