
# Robustness quantification in estimation: `robest`

This repository provides tools for quantifying the robustness of parameter-varying dynamical system models, with a focus on continuous-time systems. It includes general symbolic and numerical abstractions, sensitivity analysis, and bounds for estimation error.

## Paper
Please refer to the paper for more details: 

> Ayush Pandey. "Parameter Robustness in Data-Driven Estimation of Dynamical Systems". Conference on Decision and Control (CDC) 2025. 

## File Structure

- `robustness_estimates.ipynb`: Main notebook for robustness analysis for LPV systems with parameter dependent A.
- `robustness_estimates_x0.ipynb`: Notebook for robustness analysis with respect to initial condition perturbations, including sensitivity and bound calculations.
- `robest.py`: Main module defining the `System` and `AugmentedSystem` classes for symbolic/numeric system creation, matrix stacking, lambdifying, and simulation used in the jupyter notebooks.
- `utils.py`: Utility functions for sensitivity analysis, central difference computation, Gramian bounds, and theorem-based robustness bounds.
- `requirements.txt`: Lists all required Python packages for running the notebooks and modules.

## Installation

Clone the repository and install dependencies:
```sh
git clone https://github.com/ayush9pandey/robest.git
cd robest
pip install -r requirements.txt
```

## Usage

- Open `robustness_estimates.ipynb` or `robustness_estimates_x0.ipynb` in Jupyter to explore symbolic/numeric system setup, simulation, sensitivity analysis, and bound visualization.
- Use the `System` and `AugmentedSystem` classes in `robest.py` for general parameter-varying systems.
- Utility functions in `utils.py` support sensitivity and robustness computations for research and experimentation.

## Requirements

All dependencies are listed in `requirements.txt`:
- numpy
- scipy
- sympy
- control
- matplotlib
- plotly
- pandas
- sklearn

## Questions

Please create an [issue](https://github.com/ayush9pandey/robest/issues/) or send an email to ayush pandey at ucmerced dot edu. 