# 8 — Progenitor rewind + disruption with two embedded GCs

Modifies `7_dynfric_stream` to test the role of **dynamical friction**, which is
**absent in Yuan et al. (2020) and Malhan et al. (2021)**.

## Experiment

1. **Rewind the LMS-1 progenitor** 5 Gyr into the past. The progenitor's present-day
   phase space is the **mass-weighted mean of the NGC 5024 + NGC 5053** orbits
   (so it tracks the dominant cluster, NGC 5024, keeping the dwarf on its
   physical eccentric infall orbit — a plain average instead cancels the
   clusters' opposite v_los/v_z and circularises it). The rewind is done both
   **with** Chandrasekhar friction of the 10¹⁰ M⊙ halo against the MW and
   **without** (= Yuan/Malhan). This trajectory is the moving centre of the
   LMS-1 NFW halo.
2. **Plant two globular clusters** at the masses of NGC 5024 (5.02×10⁵ M⊙) and
   NGC 5053 (6.28×10⁴ M⊙) **near the core** of the LMS-1 halo at t = −5 Gyr, on
   coplanar bound orbits (r₀ = 0.25 and 0.40 kpc) — the "deposited centrally as a
   pair" picture. (An earlier offset start, r₀ = 0.8 / 1.5 kpc in a 10⁹ M⊙ halo,
   instead **mass-segregated** them: only NGC 5024 sank, NGC 5053 was stripped.)
3. **Carry everything forward to today.** MW tides disrupt the LMS-1 stellar body
   (10⁴ massless Plummer tracers) into a stream. The two GCs are live softened
   point masses that feel MW + moving NFW + each other, stir the stream, and —
   in the DF run only — feel **Chandrasekhar friction against the LMS-1 dark
   halo** (the new physics).
4. **Diagnose** whether the GCs remain inside the bound LMS-1 core at t = 0.

The DF vs no-DF runs share everything except friction.

## Result (`results_summary.txt`)

distance of each cluster from the LMS-1 centre at t = 0:

| run | NGC 5024 | NGC 5053 | bound stellar frac |
|-----|----------|----------|--------------------|
| **With DF** | **0.026 kpc → in core** | **0.026 kpc → in core** | 0.39 |
| No DF (Yuan/Malhan-like) | 0.262 kpc → in core | 0.429 kpc → in core | 0.15 |

With a deep 10¹⁰ M⊙ halo and both clusters deposited near the core, **dynamical
friction keeps BOTH NGC 5024 and NGC 5053 in the LMS-1 core and drives them into
a tight pair (separation → 0 within ~1 Gyr, see `lms1gc_gc_separation.png`)** —
in fact tighter than the observed 0.66 kpc pair. Without friction both stay in
the core but orbit independently, with their separation swinging up to the
observed ~0.66 kpc; DF also leaves more of the stellar body bound (0.39 vs 0.15).

**Whether DF co-locates or segregates the pair depends on halo depth and start
radius:** a shallow 10⁹ M⊙ halo with an offset start mass-segregates them (only
the massive NGC 5024 is retained); a deep halo with a central start co-locates
them. Either way, the friction omitted by Yuan+2020 / Malhan+2021 qualitatively
changes the cluster–core configuration.

## Run

```bash
python progenitor_rewind_disrupt.py     # ffmpeg auto-detected; GIF fallback
```

## Outputs

- `lms1gc_core_retention.png` — **headline**: GC distance from the core vs time
  (DF solid / no-DF dashed) + core radius band + galactocentric radii.
- `lms1gc_{xy,xz,yz}.png/.mp4` — 2 rows (DF / no-DF) × 5 epochs; stream + GCs.
- `lms1gc_gc_separation.png` — NGC 5024–NGC 5053 separation vs the observed value.
- `lms1gc_radec.png` — RA/Dec sky map at t = 0 (simulated vs observed positions).
- `lms1gc_energy_L.png` — integrals of motion of stream + GCs.
- `snap_stream_*`, `snap_gc_*` — snapshot/final phase-space arrays.

## Key parameters (top of the script)

`M_HALO=1e9`, `B_LMS1=2 kpc`, `N_STREAM=1e4`, GC masses/`r0` in the `GCS` list,
`TAU=2⁻¹¹ Gyr`, `T_BACKWARD=4 Gyr`.
