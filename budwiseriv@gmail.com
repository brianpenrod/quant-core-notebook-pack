# quant-core-notebook-pack

A compact “from-scratch” notebook pack covering core ML + quant hygiene (calibration, PCA, time-series CV, leakage checks).

## Repository structure

```text
README.md
requirements.txt
notebooks/
  01_linear_regression_from_scratch.ipynb
  02_logistic_regression_calibration.ipynb
  03_pca_cov_eigendecomp.ipynb
  04_time_series_cv_leakage_checks.ipynb
src/
  metrics.py
  linear_models.py
  pca.py
  ts_cv.py
data/
  (optional small sample csv)
figures/
