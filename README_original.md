# Constrained Mechanics Solvers

This repository contains Python implementations of structure-preserving solvers for constrained mechanical systems. It includes functionality for full-order simulations as well as Model Order Reduction (MOR) using techniques like POD, DEIM, and MDEIM.

## Features

*   **Solvers**:
    *   Discrete Gradient (DG)
    *   Conformal Stormer-Verlet
    *   Conformal Implicit Midpoint
*   **Model Reduction**:
    *   Proper Orthogonal Decomposition (POD)
    *   Discrete Empirical Interpolation Method (DEIM)
    *   Matrix DEIM (MDEIM) for hyper-reduction
*   **Performance**:
    *   Parallel execution support using `dask` and `dask_jobqueue` (SLURM integration).

## Installation

Ensure you have Python 3 installed along with the following dependencies:

```bash
pip install numpy scipy sympy matplotlib dask dask-jobqueue joblib tqdm
```

## Usage

The main driver script is located in `src-4th/ode2/app8_lattice.py`.

To run the simulation:

```bash
python src-4th/ode2/app8_lattice.py
```

*Note: The script is configured to use a SLURM cluster by default. If running locally, ensure you adjust the `dask` client configuration in the `__main__` block of the script.*

## Citation

If you use this software in your research, please cite it using the metadata provided in `CITATION.cff` or via the Zenodo DOI associated with this release.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

*   Ashish Bhatt

## Acknowledgements

This repository is forked from [scipro-primer](https://github.com/hplgit/scipro-primer) by Hans Petter Langtangen.