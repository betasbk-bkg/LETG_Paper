# LETG — Reproduction Materials

Reproduction code and data for the manuscript:

**"An Empirical Latency–Geometry Scaling Relation for Optimal Agent Speed in Crowd-Sourced Continuous Control"**
BongKeun Song, Department of Chemical and Bioengineering (CBI), Friedrich-Alexander-Universität Erlangen-Nürnberg, Germany.
Submitted to *International Journal of Human-Computer Studies* (Elsevier), 2026.

This repository contains the simulation code and raw experimental outputs (JSON/CSV) supporting the manuscript. Approximately **35,600 Monte Carlo simulation runs** across six trajectory types and ten experiment groups are reproduced by the scripts below.

---

## Summary of Key Findings

- **Empirical scaling**: the optimal operating speed follows v\*(α, τ) = C(α) × τ⁻ᵝ across all six trajectories (R² ≥ 0.94) over a 10-fold latency range (τ = 100–1000 ms).
- **Within-trajectory collapse**: RMSE(v) curves from six latency conditions align onto a single master function of x = v√τ; the per-trajectory invariant x\* = v\*√τ has a CV of 6–11%.
- **γ trajectory-invariance**: the aggregated consensus strength γ is trajectory-invariant, with a cross-trajectory spread of **0.81%** in its mean across four trajectories spanning all three geometry classes (Circle, Square, Ellipse, Suzuka). This directly supports the assumption that the per-step direction error is set by the vote-quantization scheme rather than the trajectory.
- **Geometry modulation**: the exponent β is geometry-modulated within a narrow band [0.45, 0.60] (Cochran Q heterogeneity I² = 92.3%, p < 0.001); four of six trajectories remain individually consistent with β = 1/2.
- **Practical formula (regular-polygon class)**: v\*(α, τ) ≈ 1.347 × cos¹·⁶⁸⁸(α/2) × τ⁻¹ᐟ².

---

## Repository Contents

### Core simulator

| File | Role |
|---|---|
| `simulation_main.py` | Discrete-time simulation framework (Δt = 1/60 s; delayed-state retrieval; vote aggregation; exponential smoothing α_e = 0.2). All experiment scripts import from this module. |
| `paper_fullscale.py` | Full-scale vote-generation logic (`gen_votes`): agent composition for accurate / slow / other / adversarial participants under a given troll ratio. Defines the per-frame vote distribution used throughout. |

### Experiment scripts and outputs

| Script | Output | Manuscript reference |
|---|---|---|
| `unified_vstar_experiment.py` | `unified_vstar_results.json` | Table I; Table III; Fig. 1; Fig. 2 (G1 + G2 main sweep) |
| `exp1_fixed.py` | `exp1_fixed_results.json` | E1 — vote-noise σ sweep |
| `exp2_fixed_exp3_supplement.py` | `exp2_fixed_results.json`, `exp3_quadratic_fixed.json` | E2 — VOTE_INT proportional; E3 quadratic τ=100 ms supplement |
| `exp123_final.py` | `exp3_results.json` | E3 — aggregation method (Fixed/Majority/Quadratic); Fig. 3 |
| `expa_quantization.py` | `expa_results.json` | EXP-A — directional resolution n_dirs ∈ {4, 8, 16, 36, 360}; Fig. 4 |
| `theta_autocorr_fixed.py` | `theta_autocorr_results.json` | E4 — direction-error analysis (σ_θ vs N, troll, autocorrelation); Fig. 5 |
| `corner_freq_experiment.py` | `corner_freq_results.json` | CF — corner-frequency control (n = 11,700, r = +0.94) |
| `collapse_nonregular.py` | `collapse_nonregular_results.json` | **NR — non-regular trajectory collapse (Ellipse, Suzuka); Table I (6 rows); Fig. 2; Fig. 6** |
| `gamma_variance.py` | `gamma_variance_results.json`, `gamma_data_per_run.csv`, `gamma_data_summary.csv` | **Gγ — γ trajectory-invariance (0.81% spread); Fig. 6** |
| `suzuka_extract.py` | `suzuka_track.csv` | Suzuka GP centerline extraction (real-world circuit, perimeter-rescaled) |

### Note on provenance

`exp123_final.py` is retained for full provenance of `exp3_results.json`. Only its EXP3 Fixed/Majority outputs are used in the paper; its EXP1/EXP2 outputs were superseded by `exp1_fixed.py` and `exp2_fixed_exp3_supplement.py` respectively.

---

## Code → Manuscript Run Counts (Table II)

| Group | Script | Approx. runs |
|---|---|---|
| G1 — main sweep (Circle, Square) | `unified_vstar_experiment.py` | 7,020 |
| G2 — polygons (Hexagon, Triangle) | `unified_vstar_experiment.py` | 2,340 |
| E1 — noise σ | `exp1_fixed.py` | 3,510 |
| E2 — VOTE_INT | `exp2_fixed_exp3_supplement.py` | 1,170 |
| E3 — aggregation (incl. supplement) | `exp123_final.py` + `exp2_fixed_exp3_supplement.py` | 1,425 |
| EXP-A — directional resolution | `expa_quantization.py` | 5,850 |
| E4 — direction-error analysis | `theta_autocorr_fixed.py` | 195 |
| CF — corner frequency | `corner_freq_experiment.py` | 11,700 |
| NR — non-regular collapse (Ellipse, Suzuka) | `collapse_nonregular.py` | 2,340 |
| Gγ — γ trajectory-invariance | `gamma_variance.py` | 60 |
| **Total** | | **35,610** |

---

## Reproduction

### Environment

- Python ≥ 3.9
- Dependencies: `numpy`, `scipy`, `matplotlib` (diagnostic plots only); `json`, `math`, `random` (standard library). See `requirements.txt`.

### Execution order

`exp2_fixed_exp3_supplement.py`, `exp123_final.py`, `collapse_nonregular.py`, and `gamma_variance.py` read prior result JSON files as baselines. Run in the following order:

```bash
# 1. Main sweep (baseline for several downstream experiments)
python unified_vstar_experiment.py

# 2. Robustness / mechanism
python exp1_fixed.py
python exp2_fixed_exp3_supplement.py
python exp123_final.py
python expa_quantization.py
python theta_autocorr_fixed.py
python corner_freq_experiment.py

# 3. Non-regular trajectories and gamma invariance
#    (suzuka_track.csv is provided; regenerate with suzuka_extract.py if desired)
python collapse_nonregular.py
python gamma_variance.py
```

Each script writes its output to the working directory. The JSON/CSV files in this repository are the exact outputs used to generate the manuscript tables and figures.

### Simulation parameters

- Monte Carlo MC = 15 per condition; default crowd size N = 150; troll ratio 5% (unless varied).
- Aggregation window WIN = 0.3 s; vote interval VOTE_INT = 18 frames at 60 Hz.
- Exponential smoothing α_e = 0.2; RMSE computed over the full 3900-frame (65 s) run.
- 8-direction vote quantization by default; EXP-A sweeps n_dirs from 4 to 360.

---

## Related Work

This study builds on a companion preprint reporting that operating speed dominates aggregation method in CSCC:

> BongKeun Song. *Speed Over Strategy: Why Agent Velocity Dominates Aggregation Method in Crowd-Sourced Continuous Control.* Research Square, preprint (Version 1), April 2026. https://doi.org/10.21203/rs.3.rs-9326125/v1

---

## Citation

See `CITATION.cff` for machine-readable citation metadata. An archived snapshot of this repository is available on Zenodo:

> Zenodo DOI: **[to be inserted after release]**

---

## License

Released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**. See `LICENSE`.
