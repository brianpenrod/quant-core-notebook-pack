# Quant Core: Decision-Quality Analytics (Time-Series Validation & Leakage Audits)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production-00CC96?style=flat)
![License](https://img.shields.io/badge/License-MIT-grey?style=flat)

## 1. Executive Summary
This repository houses a "from-scratch" implementation of core quantitative algorithms and validation frameworks. The objective is to decouple financial modeling logic from "black-box" libraries (like Scikit-Learn) to demonstrate a granular understanding of **mathematical mechanics**, **vectorization**, and **quantitative hygiene**.

**Primary Focus:**
* **Estimators:** Closed-form OLS and MLE implementations for factor modeling.
* **Risk Management:** Eigen-decomposition (PCA) for covariance matrix filtering and feature neutralization.
* **Validation:** Strict handling of **Non-Stationarity** and **Look-Ahead Bias** via Purged Time-Series Cross-Validation.

---

## 2. Repository Architecture

The codebase is segregated into `notebooks/` (research/visualization) and `src/` (production logic) to mimic an institutional deployment environment.

```text
quant-core-notebook-pack/
├── notebooks/                     # Research & Executive Visualization
│   ├── 01_linear_regression.ipynb # OLS Estimators & Alpha Factor Loading
│   ├── 02_logistic_calibration.ipynb # Probability Calibration (Platt Scaling)
│   ├── 03_pca_risk_factors.ipynb  # Eigen-decomposition & Variance Analysis
│   └── 04_cv_leakage_audit.ipynb  # Purged K-Fold & Leakage Detection
├── src/                           # Production-Ready Python Modules
│   ├── metrics.py                 # Custom loss functions (Sharpe, Information Ratio)
│   ├── linear_models.py           # Vectorized Regressors
│   ├── pca.py                     # SVD & Covariance Logic
│   └── ts_cv.py                   # Custom Generators for Non-Stationary Data
├── figures/                       # Output artifacts and performance plots
├── requirements.txt               # Dependency manifest
└── README.md                      # Documentation
