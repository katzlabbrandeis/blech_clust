# Installation Migration Notes

## Changes in this PR

This PR consolidates the installation process to use pip for all Python packages, addressing issue #579.

### What Changed

1. **Consolidated Requirements**: Combined `conda_requirements_base.txt` and `pip_requirements_base.txt` into a single `requirements/requirements.txt` file
2. **Pip-Only Installation**: All Python packages are now installed via pip instead of mixing conda and pip
3. **Removed Unused Packages**: Removed packages that are not actually used in the codebase:
   - `bokeh==1.4.0` - No usage found in codebase
   - `dask==2021.10.0` - No usage found in codebase
   - `fastcluster` - No usage found in codebase
4. **Simplified Makefile**: Updated the `base` target to use only pip for Python packages
5. **Consolidated psutil**: Unified psutil version requirement to `>=6.1.0`

### What Stayed the Same

- **R packages**: EMG analysis still requires R packages via conda (r-base, r-polynom, r-orthopolynom)
- **BaSAR**: Still installed from local tar.gz file as it's archived on CRAN
- **External repositories**: neuRecommend and blechRNN still cloned and installed separately
- **Installation commands**: Same `make` commands work as before

### Benefits

- **Simpler dependency management**: Single requirements file for Python packages
- **Faster installation**: Fewer package conflicts between conda and pip
- **Cleaner environment**: Removed unused packages reduces installation time and disk space
- **Better reproducibility**: pip-only installation is more predictable across platforms

### Migration for Users

No action required - existing `make` commands continue to work:

```bash
make all      # Install everything
make base     # Install base environment
make emg      # Install EMG requirements
make clean    # Clean environment
```

### Files Modified

- `Makefile` - Updated base installation process
- `requirements/requirements.txt` - New consolidated requirements file
- `requirements/conda_requirements_base.txt` - Kept for reference (not used)
- `requirements/pip_requirements_base.txt` - Kept for reference (not used)

### Testing

The installation has been tested with the existing CI/CD pipeline to ensure compatibility across:
- Ubuntu 20.04, 22.04, 24.04
- Python 3.8, 3.9, 3.10, 3.11
