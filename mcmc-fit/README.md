# NGC 5024 & NGC 5053 — Bayesian + N-body Binary Cluster Analysis

A probabilistic and dynamical analysis of whether the Milky Way globular
clusters **NGC 5024 (M53)** and **NGC 5053** form a gravitationally bound
binary pair, using MCMC phase-space sampling combined with three
independent gravitational binding criteria.

---

## 1. TL;DR

- **3D spatial separation** (MCMC posterior): Δr = 1.01 +0.28 / −0.29 kpc
- **3D relative velocity** (MCMC posterior): Δv = 108.0 ± 0.4 km/s
- **Hill radius** of NGC 5024 at r_gc = 18.3 kpc (M = 10⁶ M☉): r_H ≈ 229 pc
- P(Δr < r_H | data) ≈ 10⁻³ → pair is **not tidally bound**
- Δv ≫ v_esc(300 pc) ≈ 5.4 km/s → pair is **not kinematically bound**
- E_bind / E_kin ≈ 0.0025 → unbound by ~400× in energy

All three criteria and the MCMC posterior agree: NGC 5024 and NGC 5053 are
most plausibly a **transient flyby / chance projection**, not a bound pair.

---

## 2. File Layout

| File | Description |
|------|-------------|
| `ngc5024_5053_combined_report.tex` | Main compilable LaTeX report (Overleaf-ready) |
| `ngc5024_5053_6d_emcee.py` | Full 12D phase-space MCMC script (emcee) |
| `ngc5024_5053_emcee.py`    | Spatial-only 6D MCMC script (emcee) |
| `mcmc_sep_compare.png`     | Δr PDFs: MCMC-1 vs MCMC-2 |
| `mcmc_dv_pdf.png`          | Δv PDF from MCMC-2 |
| `mcmc_joint_rv.png`        | Joint Δr–Δv hexbin |
| `mcmc_corner.png`          | Corner plot of (Δr, Δv) |
| `mcmc_hill_pdf.png`        | Δr PDF with Hill-radius thresholds overlaid |
| `mcmc_hill_cdf.png`        | Cumulative P(Δr < r_H) |
| `ngc5024_5053_6d_pdfs.png` | 3-panel (Δr, Δv, joint) figure |
| `ngc5024_5053_6d_corner.png` | Corner plot of derived quantities |

---

## 3. Data Sources

Phase-space data from the Baumgardt globular-cluster catalog:
<https://people.smp.uq.edu.au/HolgerBaumgardt/globular/orbits.html>

|                 | NGC 5024 (M53)     | NGC 5053            |
|-----------------|--------------------|---------------------|
| d_helio (kpc)   | 18.50 ± 0.18       | 17.54 ± 0.23        |
| X, Y, Z (kpc)   | 5.25, −1.49, 18.20 | 5.11, −1.38, 17.21  |
| U, V, W (km/s)  | −57.27, 155.76, −72.49 | −52.79, 150.82, 35.30 |
| v_los (km/s)    | −63.37 ± 0.25      | +42.82 ± 0.25       |

---

## 4. Method Summary

### 4.1 MCMC-1 — Spatial-only (6D)
Sample (X₁, Y₁, Z₁, X₂, Y₂, Z₂) with independent Gaussian likelihoods;
derive Δr = |r₁ − r₂|.

### 4.2 MCMC-2 — Full 6D phase space (12D)
Sample full (X, Y, Z, U, V, W) for each cluster; derive Δr and
Δv = |v₁ − v₂|.

Both runs use `emcee.EnsembleSampler` with 64 walkers, 1500 burn-in
steps, 5000 production steps. Autocorrelation τ ≈ 60 → ~5000 effective
samples. Acceptance fractions 0.39–0.52.

### 4.3 Binding criteria
1. **Hill radius**: r_H = r_gc · (M₁ / 3M_MW)^(1/3)
2. **Escape velocity**: v_esc = √(2GM₁ / d)
3. **Energy ratio**: E_bind / E_kin

### 4.4 N-body simulation
- Potential: `MWPotential2014` (Agama)
- Particles: 2 point-mass Plummer (ε=0) representing the clusters
- Integrator: scipy DOP853, rtol=1e-10, atol=1e-12
- Energy conservation: |ΔE/E| < 1e-12 over 1 Gyr

---

## 5. Reproducing the Analysis

### Dependencies

```bash
pip install numpy scipy matplotlib emcee corner
# N-body part:
pip install agama
```

### Run

```bash
# MCMC — spatial only (6D)
python ngc5024_5053_emcee.py

# MCMC — full 6D phase space (12D)
python ngc5024_5053_6d_emcee.py

# Hill-radius posterior analysis
python hill_pdf.py
```
---

## 6. Key Results

### Posterior summaries

| Quantity | Median +1σ / −1σ | 95% CI |
|----------|------------------|--------|
| Δr (kpc) | 1.012 +0.282 / −0.290 | [0.456, 1.587] |
| Δv (km/s)| 108.01 +0.38 / −0.37 | [107.27, 108.74] |

### Hill-radius binding probabilities

| Assumed M₁ (M☉) | r_H (pc) | P(Δr < r_H) |
|-----------------|----------|-------------|
| 5×10⁵ (actual) | 182 | 4.1×10⁻⁴ |
| 1×10⁶ (proxy)  | 229 | 1.5×10⁻³ |
| 2.25×10⁶       | 300 | 5.1×10⁻³ |
| 1.04×10⁷       | 500 | 3.6×10⁻² |

### Mass thresholds for binding

| d (pc) | M_Hill (M☉) | M_v_esc (M☉) |
|--------|-------------|---------------|
| 100 | 8.3×10⁴ | 1.36×10⁸ |
| 300 | 2.25×10⁶ | 4.07×10⁸ |
| 500 | 1.04×10⁷ | 6.78×10⁸ |

---

## 7. References

- Baumgardt & Hilker 2018, MNRAS 478, 1520
- Bovy 2015, ApJS 216, 29 (galpy / MWPotential2014)
- Foreman-Mackey et al. 2013, PASP 125, 306 (emcee)
- Vasiliev 2019, MNRAS 482, 1525 (Agama)
- Chun et al. 2010, AJ 139, 606

