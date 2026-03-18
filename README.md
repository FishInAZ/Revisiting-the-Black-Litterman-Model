# Revisiting-the-Black-Litterman-Model

Robust Extensions and Data-Driven Calibration for Modern Portfolios.

![Black Litterman Model](image.png)

## Environment Setup

This repository includes a reproducible Conda environment file:

- [environment.yml](/Users/jiangzhenhao/Desktop/Revisiting-the-Black-Litterman-Model/environment.yml)

Create the environment with:

```bash
conda env create -f environment.yml
conda activate qtrade311
```

If you want to use the environment inside Jupyter Notebook, register it as a kernel:

```bash
python -m ipykernel install --user --name qtrade311 --display-name "Python (qtrade311)"
```

## Running The Notebooks

After activating the environment:

```bash
jupyter notebook
```

Then open any notebook in the repository, for example:

- [covariance_backtest2.ipynb](/Users/jiangzhenhao/Desktop/Revisiting-the-Black-Litterman-Model/covariance_backtest2.ipynb)
- [BL Model v2.ipynb](/Users/jiangzhenhao/Desktop/Revisiting-the-Black-Litterman-Model/BL%20Model%20v2.ipynb)

Inside Jupyter or VS Code, select the kernel `Python (qtrade311)`.

## Notes

- `environment.yml` was exported from the local `qtrade311` Conda environment.
- Some notebooks fetch market and macro data from external sources, so network access is required when running those cells.