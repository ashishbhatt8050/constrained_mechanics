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
    *   Centralized configuration for simulation parameters in `src/System.py`.

## Installation

It is highly recommended to install the project and its dependencies in a dedicated virtual environment to avoid conflicts with other projects. This project uses `conda` for environment management.

1.  Clone the repository:
    ```bash
    git clone https://github.com/ashishbhatt8050/constrained_mechanics.git
    cd constrained_mechanics
    ```

2.  Create and activate a new conda environment (e.g., named `constrained_mechanics_env`):
    ```bash
    conda create --name constrained_mechanics_env python=3.11
    conda activate constrained_mechanics_env
    ```

3.  Install the project in "editable" mode. This command reads the `pyproject.toml` file, installs all required dependencies into your active environment, and makes your local source code available on your Python path.
    ```bash
    pip install -e .
    ```

## Usage

The main driver script is `scripts/app8_lattice.py`. All core simulation parameters (number of oscillators, final time, etc.) are now centralized in the `SysConfig` class within `src/System.py`. Select between prediction and reproduction experiment therein.

### Running the Main Simulation

To run the simulation:
```bash
python scripts/app8_lattice.py
```

To run the simulation and clear any previous checkpoints:
```bash
python scripts/app8_lattice.py --clean
```

To run without using LaTeX for plot rendering:
```bash
python scripts/app8_lattice.py --no-latex
```

### Running the Symbolics Benchmark

A script for benchmarking different symbolic computation strategies is available. 
```bash
python scripts/benchmark_symbolics.py
```

## Debugging

You can selectively disable parts of the simulation for easier debugging.

### Running on a Local Machine

The script automatically detects if the `sbatch` command is available. If not, it defaults to using `dask.distributed.LocalCluster`. To force local execution on a machine that has SLURM, you can temporarily modify the check in `scripts/app8_lattice.py`:

```bash
```python
# In scripts/app8_lattice.py, change this line:
if False and shutil.which('sbatch') and not "PYTEST_CURRENT_TEST" in os.environ:
#  ^ Change to False to disable SLURM detection
```

*Note: The script is configured to use a SLURM cluster by default. Ensure you adjust the `dask` client configuration in the `__main__` block of the script according to your own cluster configuration.*

## Citation

If you use this software in your research, please cite it using the metadata provided in `CITATION.cff` or via the Zenodo DOI associated with this release.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

*   Ashish Bhatt

## Acknowledgements

This repository is forked from [scipro-primer](https://github.com/hplgit/scipro-primer) by Hans Petter Langtangen.