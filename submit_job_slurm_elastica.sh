#!/bin/bash

# ==============================================================================
# SLURM Job Submission & Monitoring Wrapper for app9_elastica
#
# This script is a backward-compatible wrapper around submit_job_slurm.sh.
# It automatically selects the 'elastica' application.
#
# Usage:
#   sbatch submit_job_slurm_elastica.sh [mode] [experiment_name]
#   ./submit_job_slurm_elastica.sh monitor [ID]
# ==============================================================================

#SBATCH --job-name=app9_elastica
#SBATCH --output=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --partition=high_cpuq
#SBATCH --account=high_cpu_acct
#SBATCH --qos=high_cpu_qos

# To use GPU, uncomment the following lines:
# SBATCH --partition=gpuq
# SBATCH --gres=gpu:1
# SBATCH --account=gpu_users

# Notification settings
# SBATCH --mail-type=ALL
# SBATCH --mail-user=

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If monitor command or ID passed, forward directly
if [ "$1" = "monitor" ] || [ "$1" = "status" ] || [[ "$1" =~ ^[0-9]{5,}$ ]]; then
    exec "${SCRIPT_DIR}/submit_job_slurm.sh" "$@"
fi

# Otherwise execute submit_job_slurm.sh with elastica as the application
exec "${SCRIPT_DIR}/submit_job_slurm.sh" elastica "$@"
