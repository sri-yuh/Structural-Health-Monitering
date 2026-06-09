# Engineered, Interpretable, Amplitude-Invariant SHM of a Cantilever Beam

> Vibration-based Structural Health Monitoring framework that performs **damage detection, localization, and quantification** on a clamped–free cantilever beam using a 72-dimensional engineered feature vector and a hybrid CNN + LSTM + Gradient Boosting ensemble.

A final-year project at BITS Pilani, Dubai Campus (CS F366) under the supervision of Dr. Harpreet Singh Bedi.

---

## Overview

This project addresses the first three levels of Rytter's damage-identification hierarchy:

| Level | Task | Approach |
|---|---|---|
| **1 — Detection** | Is the beam damaged? | Binary classifier (CNN + LSTM) |
| **2 — Localization** | Where is the damage? | Regression on (Dx, Dy) coordinates |
| **3 — Quantification** | How severe is it? | Regression on cutout radius r |

The beam is a 100 × 10 × 1 mm aluminium cantilever with a single circular through-thickness cutout whose position and radius vary across simulations.

---

## Key Features

- **72-dimensional engineered feature vector** — physically interpretable, mathematically amplitude-invariant by construction
- **Per-target specialists** — Bagged XGBoost for Dx, LightGBM for Dy, HistGradientBoosting for radius
- **Residual learning rule** — deep models learn residuals over gradient-boosting bases when inner-validation R² > 0.5, else absolute targets
- **Group-aware data split** — eliminates 99 % R² leakage from physical-cutout duplicates across train and test
- **Mixup at training, Test-Time Augmentation at inference** — variance reduction without changing the architecture
- **Two empirical ablations** — verify the bending–torsion coupling hypothesis (Nguyen 2014) and amplitude invariance (Worden Axiom V, 2007)

---

## Results

All metrics on the **held-out group-aware test set** (306 unseen physical cutouts):

### Detection (Level 1)

| Model | Accuracy | F1 |
|---|---|---|
| CNN classifier | 1.000 | 1.000 |
| LSTM classifier | 1.000 | 1.000 |

### Localization & Quantification (Levels 2 & 3)

| Target | Best model | MAE | R² |
|---|---|---|---|
| **Dx (length-wise)** | CNN regressor + Bagged XGBoost residual | 1.83 mm | +0.985 |
| **Dy (breadth-wise)** | LSTM regressor + LightGBM absolute | 0.95 mm | +0.513 |
| **Radius** | HistGradientBoosting + CNN residual | 0.027 mm | +0.981 |

### Ablation studies

- **Y-offset ablation:** Adding off-centreline (Y2–Y8) measurements improves Dy MAE by **+8.7 %** — confirming the bending–torsion coupling argument of Nguyen (2014)
- **Forcing-amplitude ablation:** The framework achieves **100 % F1 cross-band** (train on lowest amplitude tercile, test on highest), confirming amplitude invariance — empirical validation of Axiom V (Worden et al. 2007)

---

## Repository Structure

```
.
├── SHM_FINAL.ipynb              # Main pipeline notebook (end-to-end)
├── Ablation_Study.ipynb         # Standalone ablation notebook (companion)
├── training_data_v2.zip         # Dataset (Abaqus FEA outputs)
└── README.md
```

---

## Quick Start (Google Colab)

1. **Open the main notebook** in [Google Colab](https://colab.research.google.com/)
2. **Switch runtime to GPU** (Runtime → Change runtime type → T4 GPU)
3. **Run the upload cell** and upload `training_data_v2.zip` when prompted
4. **Run all cells top-to-bottom** (Runtime → Run all). Total runtime: ~15 minutes on a T4 GPU.

The notebook handles all dependencies via `pip install` cells.

---

## Dataset

The dataset consists of approximately **4,700 finite-element simulations** generated in Abaqus. Each sample contains:

- **Modal report** — first 10 mode shapes sampled at 256 spatial points
- **Response report** — frequency response function (FRF) magnitude at 256 frequency points
- **Frequency report** — first 10 natural frequencies

### Damage parameter ranges

| Parameter | Range |
|---|---|
| Dx (longitudinal centre) | [−47, +47] mm |
| Dy (transverse centre) | [−3.4, +3.5] mm |
| Cutout radius r | [0.05, 1.0] mm |
| Sensor Y-offset | {0, 1.25, 2.5, 3.75, 5.0} mm |
| Forcing amplitude | 1 × 10 mm to 474 × 10 mm (~3 decades) |

---

## 72-D Feature Vector

| Family | Count | Description |
|---|---|---|
| Per-mode curvature peak position | 10 | argmax \|κ(x)\| per mode |
| Per-mode curvature peak value | 10 | max \|κ(x)\| per mode |
| Per-mode curvature weighted average | 10 | ∫x\|κ\|dx / ∫\|κ\|dx |
| Overall beam-wide curvature centroid | 1 | sum across all modes |
| Sensor offset / forcing-amplitude scalar | 1 | scalar metadata |
| Per-mode frequency delta | 10 | Δfₙ = fₙ_test − fₙ_healthy |
| Per-mode TKEO peak position | 10 | argmax Ψ[φₙ(x)] |
| Per-mode L/R energy contrast | 10 | (E_R − E_L) / (E_R + E_L) |
| Per-mode baseline-residual norm | 10 | ‖φₙ − ⟨φₙ⟩_healthy‖₂ |
| **TOTAL** | **72** | |

All features are amplitude-invariant by construction (ratios, peak positions, baseline differences, and quantities computed on L2-normalized inputs).

---

## Reproducibility

All randomness is seeded to `42`:

- `train_test_split(random_state=42)` for stratified classification split
- `GroupShuffleSplit(random_state=42)` for group-aware regression split
- `np.random.default_rng(42)` for Mixup augmentation
- `tf.keras.utils.set_random_seed(42)` for deep-model initialisation
- `random_state=42` for all Optuna studies and gradient-boosting models

XGBoost and LightGBM use `n_jobs=1` during Optuna hyperparameter search to ensure deterministic tree construction.

File-listing order is fixed by `sorted()` calls during dataset loading.

---

## Key References

| Ref | Used for |
|---|---|
| Rytter 1993 | 4-level damage-identification hierarchy |
| Pandey, Biswas & Samman 1991 | Modal curvature as damage indicator |
| Kaiser 1990 | Teager–Kaiser Energy Operator |
| Nguyen 2014 | Bending–torsion coupling — justifies Y-offset design |
| Worden et al. 2007 | Fundamental axioms of SHM — Axiom V on operational variability |
| Friedman 2001 | Gradient boosting framework |
| Chen & Guestrin 2016 | XGBoost |
| Ke et al. 2017 | LightGBM |
| Breiman 1996 | Bagging (used for XGBoost ensemble) |
| Wolpert 1992 | Stacked generalization (OOF base predictions) |
| Zhang et al. 2018 | Mixup augmentation |
| Ayhan & Berens 2018 | Test-Time Augmentation |
| Abdeljaber et al. 2017 | 1-D CNN for vibration-based SHM |
| Kingma & Ba 2014 | Adam optimizer |

Full reference list with DOIs is provided in the manuscript.

---

## Manuscript

See the accompanying manuscript for the full methodology, ablation analyses, and discussion.

---

## Author

**Sriya Sanagala** — 2022A7PS0101U
BITS Pilani, Dubai Campus — Computer Science
Project supervisor: Dr. Harpreet Singh Bedi

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

- BITS Pilani Dubai Campus for the computational resources
- Dr. Satya Jaswanth, Dr. Harpreet Singh Bedi for supervision and methodological guidance
- The Abaqus team at Dassault Systèmes for the FEA platform

---

*If you use this code or framework in your work, please cite the accompanying manuscript.*
