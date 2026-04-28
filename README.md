# Physics-Informed Neural Networks (PINNs) for Stiff PDEs

## 📌 Overview
This repository contains the official PyTorch implementation for solving stiff Partial Differential Equations (PDEs) using **Physics-Informed Neural Networks (PINNs)**. 

The project systematically investigates the performance of PINNs on the 1D Convection-Diffusion equation and the highly non-linear 1D/2D **Allen-Cahn equation**. To overcome the mathematical stiffness and efficiently resolve extremely thin phase interfaces ($\epsilon = 0.025$), we implement an **Adaptive Loss Weighting** mechanism and a **Dynamic Collocation Point Movement (AMAW)** strategy.

## ✨ Key Features
- **Baseline PINN & Analytical Validation:** Solves the 1D Convection-Diffusion equation and compares it against exact analytical solutions.
- **Adaptive Loss Weighting:** Dynamically balances the competing loss terms ($\mathcal{L}_{PDE}$, $\mathcal{L}_{IC}$, $\mathcal{L}_{BC}$) to prevent convergence to trivial solutions.
- **Dynamic Point Movement:** Periodically resamples and clusters collocation points toward high-gradient regions (phase interfaces), significantly breaking the "curse of dimensionality" in 2D spatial domains.
- **FEM Reference Data:** Benchmarked against high-fidelity Finite Element Method (FEM) simulations.

## 📂 Repository Structure
```text
.
├── data/                   # Ground truth data from FEM simulations
├── figures/                # Saved visualizations and error heatmaps
├── main_1d_cd.py           # Script for 1D Convection-Diffusion
├── main_1d_ac.py           # Script for 1D Allen-Cahn
├── main_2d_ac.py           # Script for 2D Allen-Cahn (Merging circles)
└── README.md
