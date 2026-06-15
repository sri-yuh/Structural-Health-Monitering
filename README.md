# SHM of a Cantilever Beam

> Vibration-based Structural Health Monitoring framework that performs **damage detection, localization, and quantification** on a cantilever beam using a 78-dimensional engineered feature vector and a hybrid CNN + LSTM + Gradient Boosting ensemble.

A final-year project at BITS Pilani, Dubai Campus (CS F366) under the supervision of Dr. Satya Jaswanth and Dr. Harpreet Singh Bedi.

---

## Overview

This project addresses the first three levels of Rytter's damage-identification hierarchy:

| Level | Task | Approach |
|---|---|---|
| **1 — Detection** | Is the beam damaged? | Binary classifier (CNN + LSTM) on Modal + Response signals |
| **2 — Localization** | Where is the damage? | Regression on (Dx, Dy) coordinates with 78-D engineered features |
| **3 — Quantification** | How severe is it? | Regression on cutout radius r |

The beam is a 100 × 10 × 1 mm aluminium cantilever with a single circular through-thickness cutout whose position and radius vary across simulations.

---

## Key Features

- **78-dimensional engineered feature vector** — physically interpretable, mathematically amplitude-invariant by construction (used by the regressors)
- **Signals-only classifier** — CNN and LSTM detectors learn directly from Modal Edge displacment and Response, providing an unbiased evaluation of the deep architectures
- **Balanced classifier pool** — equal healthy and damaged samples eliminate class-prior bias
- **Group-aware data split** — keyed on (Dx, Dy, R), eliminates >99 % R² leakage from physical-cutout duplicates across train and test
- **Per-target models** — Bagged XGBoost for Dx, LightGBM for Dy, HistGradientBoosting for radius
- **Residual learning rule** — deep models learn residuals over ML base models when inner-validation R² > 0.5, else absolute targets
- **Mixup at training, Test-Time Augmentation at inference** — variance reduction without changing the architecture noisy data generated at inputs, to create variance which then the mean is calculated for.
- **Two experiments** — verify the (offset) and (forcing displacement on the free end)

---

## Results

All metrics on the **held-out group-aware test set** :

### Detection (Level 1)

Reported under realistic noise addition:

| Model | Accuracy | F1 |
|---|---|---|
| CNN classifier (σ = 0.30·⟨\|x\|⟩) | 0.98 | 0.98 |
| LSTM classifier (σ = 0.60·⟨\|x\|⟩) | 0.97 | 0.97 |

**Adding Feature condition :** Both architectures achieve **1.000** accuracy under noise-free FEA simulations, reflecting the strong separability of cutout damage in clean modal data.

### Localization & Quantification (Levels 2 & 3)

| Target | Best model | MAE | R² |
|---|---|---|---|
| **Dx (length-wise)** | CNN, LSTM regressor, Bagged XGBoost residual | 2.09 mm | +0.986 |
| **Dy (breadth-wise)** | CNN, LSTM regressor, (LightGBM- tried) absolute | 1.03 mm | +0.41 |
| **Radius** | HistGradientBoosting + CNN, LSTM residual | 0.024 mm | +0.98 |

### Experiments

- **Y-offset :** Adding off-centreline (Y2–Y8) measurements improves Dy MAE.
- **Forcing-displacement:** The framework achieves **100 % Acc, F1 cross-band** (train on lowest amplitude, test on highest), confirming amplitude invariance.

---

## Repository Structure

```
.
├── SHM_FINAL.ipynb              # Main pipeline notebook (end-to-end)
├── training_data_v2.zip         # Dataset (Abaqus FEA outputs)
└── README.md
```

---

## Quick Start (Google Colab)

1. **Open the main notebook** in [Google Colab](https://colab.research.google.com/)
2. **Switch runtime to GPU** (Runtime → Change runtime type → T4 GPU)
3. **Run the upload cell** and upload `training_data_v2.zip` when prompted
4. **Run all cells top-to-bottom** (Runtime → Run all). 

The notebook handles all dependencies via `pip install` cells.

---

## Dataset

The dataset consists of approximately **4,700 finite-element simulations** generated in Abaqus. Each sample contains:

- **Modal report** — first 10 mode shapes sampled at 256 spatial points, L2 Norm
- **Response report** — frequency response function (FRF) magnitude at 256 frequency points, L2 Norm
- **Frequency report** — first 10 natural frequencies

### Damage parameter ranges

| Parameter | Range |
|---|---|
| Dx (longitudinal centre) | [−47, +47] mm |
| Dy (transverse centre) | [−3.4, +3.5] mm |
| Cutout radius r | [0.05, 1.0] mm |
| Sensor Y-offset | {0, 1.25, 2.5, 3.75, 5.0} mm |
| Forcing displacments | 1 × 10 mm to 474 × 10 mm |

---

## Classification Pipeline (Level 1)

The classifier is deliberately **starved of engineered features** — it sees only the Modal and Response datasets. This forces both networks to learn the damage signature purely from the dynamic response, providing an unbiased evaluation of the deep architectures rather than of the engineered features.

| Stage | Configuration |
|---|---|
| Pool construction | 580 healthy + 580 damaged = 1,160 balanced samples |
| Train/test split | Group-aware on (Dx, Dy, R); 927 train / 235 test |
| Inputs | Modal (256 × 10) + Response (256 × 1)  (no spatial branch) |
| Training augmentation | Mixup (α = 0.4, n_extra = 2) + 1 % training noise |
| Inference noise | Noisy forward pass — σ = 0.30·⟨\|x\|⟩ (CNN), σ = 0.60·⟨\|x\|⟩ (LSTM) |

---

## 78-D Feature Vector (Regression)

| Family | Count | Description |
|---|---|---|
| Per-mode curvature peak position | 10 | beam pos argmax \|κ(x)\| per mode |
| Per-mode curvature peak value | 10 | max \|κ(x)\| per mode |
| Per-mode curvature weighted average | 10 | ∫x\|κ\|dx / ∫\|κ\|dx |
| Overall beam-wide curvature weighted average | 1 | weighted average across all 10 modes |
| Sensor offset | 1 | scalar metadata |
| Per-mode frequency delta | 10 | Δfₙ = fₙ_test − fₙ_healthy |
| Per-mode TKEO peak position | 10 | argmax Ψ[φₙ(x)] |
| Per-mode L/R energy contrast | 10 | (E_R − E_L) / (E_R + E_L) |
| Per-mode baseline-residual norm | 10 | ‖φₙ − ⟨φₙ⟩_healthy‖₂ |
| Mode-1 extra features | 6 | additional Mode-1 sensitivity features |
| **TOTAL** | **78** | |

All features are amplitude-invariant by construction (ratios, peak positions, baseline differences, and quantities computed on L2-normalized inputs).

---

## Reproducibility

All randomness is seeded to `42`:

- `GroupShuffleSplit(random_state=42)` for both classification and regression splits
- `np.random.default_rng(42)` for Mixup augmentation and balanced pool subsampling
- `tf.keras.utils.set_random_seed(42)` for deep-model initialisation
- `random_state=42` for all Optuna studies and gradient-boosting models

XGBoost and LightGBM use `n_jobs=1` during Optuna hyperparameter search to ensure deterministic tree construction.

---

## Key References

| Ref | Used for |
|---|---|
| Rytter 1993 | Damage-identification hierarchy |
| Pandey, Biswas & Samman 1991 | Modal curvature as damage indicator |
| Kaiser 1990 | Teager–Kaiser Energy Operator |
| Kim et al. 2003 | Mode-1 sensitivity features |
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

Full reference list is provided in the manuscript.

---

## Manuscript

See the accompanying manuscript for the full methodology, experiments analyses, and discussion.

---

## Author

**Sriya Sanagala** — 2022A7PS0101U
BITS Pilani, Dubai Campus — Computer Science
Project supervisor: Dr. Harpreet Singh Bedi

---

## Acknowledgements

- BITS Pilani Dubai Campus for the computational resources
- Dr. Satya Jaswanth, Dr. Harpreet Singh Bedi for supervision and methodological guidance
- The Abaqus team at Dassault Systèmes for the FEA platform

---

