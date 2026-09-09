# lms-1 — LMS-1 merger history and its globular-cluster association

Do **NGC 5024 (M53)** and **NGC 5053** belong to the **LMS-1 / Wukong** accretion
event, as proposed by Yuan et al. (2020) and Malhan et al. (2021)?

The repository attacks that from two independent directions:

| | question | method | answer |
|---|---|---|---|
| **`mcmc-fit/`** | Are the two clusters a **bound binary today**? | Bayesian MCMC over their 6D phase space + three binding criteria | **No** — unbound by ~400× in energy; a transient flyby / chance projection |
| **`LMS_1_progenitor/`** | Were they **deposited together** by the LMS-1 dwarf, and does **dynamical friction** change that picture? | N-body rewind + tidal disruption of the LMS-1 progenitor with two live GCs embedded | **Yes, and DF matters** — friction (omitted by Yuan+/Malhan+) keeps *both* clusters in the LMS-1 core and drives them into a tight pair |

Together: the pair is **not self-bound now**, but a common origin in the LMS-1
core is dynamically viable — and the friction term missing from the published
models is what decides whether the two clusters are co-retained or
mass-segregated.

## Repository layout

```
lms-1/
├── data/                       # shared input catalogues (see Requirements)
├── MWPotential2014.ini         # Agama MW potential, read by both projects
├── mcmc-fit/                   # Bayesian binary-cluster analysis
│   ├── ngc5024_5053_emcee.py           spatial-only 6D MCMC
│   ├── ngc5024_5053_6d_emcee.py        full 12D phase-space MCMC
│   ├── ngc5024_5053_*.png              posteriors, corner plots, tidal contact
│   └── README.md                       detailed method + results tables
└── LMS_1_progenitor/           # N-body progenitor rewind + disruption
    ├── progenitor_rewind_disrupt.py    the simulation
    ├── analyze_final_snapshot.py       t = 0 model vs STREAMFINDER
    ├── methods_draft.tex/.pdf          methods write-up
    └── mass_<M_200>_{df,nodf}/         all results of a run (see below)
```

---

## 1. `mcmc-fit/` — is the pair bound today?

MCMC sampling of both clusters' phase space (emcee, 64 walkers, 1500 burn-in +
5000 production), then three independent binding tests. Headline posteriors:

- separation Δr = 1.01 (+0.28 / −0.29) kpc
- relative velocity Δv = 108.0 ± 0.4 km/s
- Hill radius of NGC 5024 at r_gc = 18.3 kpc: r_H ≈ 229 pc
- P(Δr < r_H | data) ≈ 10⁻³, and Δv ≫ v_esc ≈ 5.4 km/s

All three criteria agree the pair is unbound. Full tables, data sources and
references are in [`mcmc-fit/README.md`](mcmc-fit/README.md).

```bash
cd mcmc-fit
python ngc5024_5053_emcee.py        # spatial-only (6D)
python ngc5024_5053_6d_emcee.py     # full phase space (12D)
```

---

## 2. `LMS_1_progenitor/` — were they deposited together?

Modifies the earlier `7_dynfric_stream` experiment to isolate the effect of
**dynamical friction**, which is absent in Yuan+2020 / Malhan+2021.

### Experiment

1. **Rewind the LMS-1 progenitor** 5 Gyr. Its present-day phase space is the
   **mass-weighted mean** of the NGC 5024 + NGC 5053 orbits, so it tracks the
   dominant cluster NGC 5024 and stays on a physical eccentric infall orbit (a
   plain average cancels the clusters' opposite v_los/v_z and circularises it).
   The rewind is run **with** Chandrasekhar friction of the 10¹⁰ M⊙ halo against
   the MW and **without** (= Yuan/Malhan). This trajectory is the moving centre
   of the LMS-1 NFW halo.
2. **Plant two globular clusters** at the masses of NGC 5024 (5.02×10⁵ M⊙) and
   NGC 5053 (6.28×10⁴ M⊙) **near the core** at t = −5 Gyr, on coplanar bound
   orbits (r₀ = 0.25 and 0.40 kpc) — the "deposited centrally as a pair"
   picture. (An earlier offset start, r₀ = 0.8 / 1.5 kpc in a 10⁹ M⊙ halo,
   instead **mass-segregated** them: only NGC 5024 sank, NGC 5053 was stripped.)
3. **Carry everything forward to today.** MW tides disrupt the LMS-1 stellar body
   (10⁴ massless Plummer tracers) into a stream. The GCs are live softened point
   masses feeling MW + moving NFW + each other, stirring the stream, and — in
   the DF run only — feeling **Chandrasekhar friction against the LMS-1 dark
   halo** (the new physics).
4. **Diagnose** whether the GCs remain inside the bound LMS-1 core at t = 0.

The DF and no-DF runs share everything except friction.

### Result (`results_summary.txt`, at `M_HALO = 1e10`)

Distance of each cluster from the LMS-1 centre at t = 0:

| run | NGC 5024 | NGC 5053 | bound stellar frac | core radius |
|-----|----------|----------|--------------------|-------------|
| **With DF** | **0.048 kpc → in core** | **0.588 kpc → in core** | **0.392** | 1.673 kpc |
| No DF (Yuan/Malhan-like) | 0.074 kpc → in core | 0.588 kpc → in core | 0.148 | 1.382 kpc |

Because the clusters are **rewound from their observed present-day phase space**,
both land on their observed positions today by construction; the question the run
answers is whether that endpoint is compatible with them having sat in the LMS-1
core for 5 Gyr. It is — in **both** runs both clusters finish inside the bound
core. What friction changes is:

- **NGC 5024 sits ~35% closer to the core with DF** (0.048 vs 0.074 kpc), i.e.
  friction does sink the massive cluster toward the centre;
- **DF leaves 2.6× more of the stellar body bound** (0.392 vs 0.148) and a larger
  surviving core (1.673 vs 1.382 kpc) — the dwarf resists MW tides for longer.

**Whether DF co-locates or mass-segregates the pair depends on halo depth and
start radius:** in earlier configurations a shallow 10⁹ M⊙ halo with an offset
start (r₀ = 0.8 / 1.5 kpc) segregated them — only NGC 5024 sank, NGC 5053 was
stripped. Either way, the friction omitted by Yuan+2020 / Malhan+2021
qualitatively changes the cluster–core configuration and the survival of the
progenitor.

### Run

```bash
cd LMS_1_progenitor
python progenitor_rewind_disrupt.py     # simulation (ffmpeg auto-detected; GIF fallback)
python analyze_final_snapshot.py        # compare the t = 0 snapshot to STREAMFINDER
```

Both scripts resolve paths from their own directory, so they can be launched from
anywhere. `analyze_final_snapshot.py` re-reads `M_HALO` and `T_BACKWARD` out of
`progenitor_rewind_disrupt.py`, so it always looks in the folders of the mass
currently configured.

Key parameters at the top of the simulation script: `M_HALO=1e10` (also sets the
output folder name), `B_LMS1=2 kpc`, `N_STREAM=1e4`, GC masses / `r0` in the
`GCS` list, `B_GC=10 pc`, `TAU=2⁻¹¹ Gyr (~0.49 Myr)`, `T_BACKWARD=5 Gyr`.

### Where results are stored

Everything a run produces goes into a **pair of folders named after the LMS-1
subhalo mass and the friction setting**:

```
mass_<M_200>_df/      with dynamical friction
mass_<M_200>_nodf/    no friction (Yuan+2020 / Malhan+2021 like)
```

The mass tag is `M_HALO` in compact scientific form, `p` standing in for the
decimal point:

| `M_HALO` | folders |
|----------|---------|
| `1e10`   | `mass_1e10_df/`, `mass_1e10_nodf/` |
| `1e9`    | `mass_1e9_df/`, `mass_1e9_nodf/` |
| `3.2e9`  | `mass_3p2e9_df/`, `mass_3p2e9_nodf/` |

A mass scan is therefore just: edit `M_HALO`, re-run — **runs at different masses
never overwrite each other**, and the DF / no-DF results of one mass sit side by
side. The folders are created automatically.

**What lands where.** Snapshot arrays are scenario-specific and go to their own
folder. The figures, animations and summaries compare DF against no-DF inside a
single frame, so they are written once and copied into both folders — each
folder is self-contained and can be shipped on its own.

```
mass_1e10_df/
├── lms1gc_core_retention.png     ┐
├── lms1gc_core_distance.png      │
├── lms1gc_gc_separation.png      │  combined DF | no-DF products,
├── lms1gc_radec.png              │  identical copies in both folders
├── lms1gc_energy_L.png           │
├── lms1gc_{xy,xz,yz}.png         │
├── lms1gc_{xy,xz,yz}.mp4         │  (.gif if no ffmpeg)
├── results_summary.txt           ┘
├── lms1gc_pm_compare.png         ┐  written by analyze_final_snapshot.py
├── lms1gc_vlos_compare.png       │  (it also overwrites lms1gc_radec.png with
├── obs_comparison_summary.txt    ┘   the observation-overlaid version)
├── snap_stream_df_t{-5..0}gyr.npy  ┐
├── snap_gc_df_t{-5..0}gyr.npy      │  this scenario only
├── snap_stream_df_final.npy        │  (…_nodf_… in the nodf folder)
└── snap_gc_df_final.npy            ┘
```

Output descriptions:

- `lms1gc_core_retention.png` — **headline**: GC galactocentric radius vs time
  (DF solid / no-DF dashed) against the LMS-1 core orbit.
- `lms1gc_core_distance.png` — GC distance from the core, per scenario, with the
  core radius band and the t = 0 bound stellar fraction.
- `lms1gc_{xy,xz,yz}.png/.mp4` — 2 rows (DF / no-DF) × 6 epochs (t = −5 … 0 Gyr):
  stream, core, Jacobi radius, both GCs; the `.mp4` is the side-by-side animation.
- `lms1gc_gc_separation.png` — NGC 5024–NGC 5053 separation vs the observed value.
- `lms1gc_radec.png` — RA/Dec sky map at t = 0, simulated vs observed.
- `lms1gc_energy_L.png` — integrals of motion (E vs L_z, E vs |L|).
- `lms1gc_pm_compare.png`, `lms1gc_vlos_compare.png` — model debris inside the
  observed footprint vs STREAMFINDER proper motions / line-of-sight velocities.
- `results_summary.txt` — core-retention verdict, tagged with the subhalo mass.
- `obs_comparison_summary.txt` — median/scatter table and verdict vs the data.
- `snap_stream_*`, `snap_gc_*` — snapshot (xyz) and final (xyz + v) phase-space
  arrays; the `*_final.npy` pair is what `analyze_final_snapshot.py` reads.

Snapshots written before this layout existed are still read from the flat
directory as a fallback, so older runs keep working.

---

## Requirements

```bash
pip install numpy scipy matplotlib pandas astropy emcee corner
pip install agama          # potentials, DF sampling, orbit integration
# optional: ffmpeg on PATH for .mp4 animations (otherwise .gif via Pillow)
```

Both projects expect two shared inputs at the **repository root**:

| path | used by | source |
|------|---------|--------|
| `MWPotential2014.ini` | both | Bovy (2015) MW potential, Agama format |
| `data/cleaned_streamfinder_ibata24.csv` | `LMS_1_progenitor` | STREAMFINDER catalogue (Ibata et al. 2024), filtered to `Name == "LMS-1"` (358 members) |

The scripts reference them as `../MWPotential2014.ini` and
`../data/cleaned_streamfinder_ibata24.csv` relative to their own folder.

## References

- Yuan et al. 2020, ApJ 898, 26 — LMS-1 discovery
- Malhan et al. 2021, ApJ 920, 51 — LMS-1 / Wukong GC association
- Ibata et al. 2024 — STREAMFINDER stream catalogue
- Baumgardt & Hilker 2018, MNRAS 478, 1520 — GC masses and distances
- Vasiliev & Baumgardt 2021, MNRAS 505, 5978 — GC proper motions
- Bovy 2015, ApJS 216, 29 — MWPotential2014
- Vasiliev 2019, MNRAS 482, 1525 — Agama
- Foreman-Mackey et al. 2013, PASP 125, 306 — emcee
- Dutton & Macciò 2014, MNRAS 441, 3359 — c–M relation
