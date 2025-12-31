#!/bin/bash

# Run this as "sbatch submit_job_slurm.sh"

# Set a name for this run and the resource requirements,
# Exclusive node access (all CPUs), all available memory and 24 hours wall time.
# TODO: raise MaxMemoryPerUser limit, install latex on compute nodes

#SBATCH --job-name=app8_lattice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=workq
#SBATCH --account=cpu_users
#SBATCH --output=app8_lattice-%j.out
#SBATCH --qos=cpu_users

# To use GPU, uncomment the following two lines:
# SBATCH --partition=gpuq
# SBATCH --gres=gpu:1
# SBATCH --account=gpu_users

# Send an email when this job aborts, begins or ends.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashish.bhatt@vit.ac.in

# Ensure the script exits with an error if the python command fails
set -o pipefail
set -e

# Increase stack size to prevent segfaults from deep recursion in symbolic math
ulimit -s unlimited
export SLURM_CPU_BIND=none

# Change to the directory where the job was submitted from.
cd $SLURM_SUBMIT_DIR

# Activate the conda environment
source /home/bhattah/miniconda3/etc/profile.d/conda.sh
conda activate modred-dae-torch

# Set SLURM_NTASKS to all available CPUs so the python script uses the full node
# export SLURM_NTASKS=$(nproc) # No longer needed for Driver

# Set MPLCONFIGDIR to a specific directory to avoid cache locking issues on shared filesystems
export MPLCONFIGDIR=$SLURM_SUBMIT_DIR/matplotlib_cache
mkdir -p $MPLCONFIGDIR

# Verify LaTeX dependencies required by Matplotlib
echo "Checking LaTeX environment..."
which latex || echo "Warning: 'latex' not found. Matplotlib usetex=True will fail."
# which dvipng || echo "Warning: 'dvipng' not found. Matplotlib usetex=True will fail."
# which gs || echo "Warning: 'gs' (Ghostscript) not found. Matplotlib usetex=True will fail."

# Run on CPU by default. To run on GPU, add --device gpu
python -u app8_lattice.py | tee logfile.txt #--device gpu

# Move the SLURM output file to the latest data directory
LATEST_DATA_DIR=$(ls -td data/*/ | head -n 1)
if [ -d "$LATEST_DATA_DIR" ]; then
    mv "app8_lattice-${SLURM_JOB_ID}.out" "$LATEST_DATA_DIR"
fi

# sleep 3
# Call the monitor_job.sh script with the current job ID
# JOBID=$(squeue -h -o %i -u $USER -n app8_lattice | head -n 1)
# ./monitor_job.sh $JOBID &
