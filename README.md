# quant-core-notebook-pack
## Repository structure

```text
quant-core-notebook-pack/
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

### Validation check
- In GitHub, the README should render a clean monospace “tree” block.
- Don’t paste that tree into a **Python** cell (that’s what caused the `01_...` SyntaxError).

### Portfolio upgrade (tiny)
Right under the tree, add 3 bullets:
- **What this repo is**
- **How to run notebooks**
- **What each notebook teaches**
