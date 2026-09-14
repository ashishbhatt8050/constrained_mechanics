#!/bin/bash

# ==============================================================================
# SLURM Job Submission & Monitoring Script for app9_elastica
#
# Usage:
#   sbatch submit_job_slurm_elastica.sh          Submit job to SLURM cluster
#   ./submit_job_slurm_elastica.sh monitor [ID]  Monitor running/recent job
# ==============================================================================

#SBATCH --job-name=app9_elastica
#SBATCH --output=app9_elastica-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --partition=high_cpuq
#SBATCH --account=high_cpu_acct
#SBATCH --qos=high_cpu_qos

monitor_job() {
    local JOBID="$1"
    
    if [ -z "$JOBID" ]; then
        JOBID=$(squeue -u "$USER" -o "%i %j" -h 2>/dev/null | grep "app9_elastica" | head -n 1 | awk '{print $1}')
        if [ -z "$JOBID" ]; then
            local LATEST_OUT=$(ls -t app9_elastica-*.out 2>/dev/null | head -n 1)
            [ -n "$LATEST_OUT" ] && JOBID=$(echo "$LATEST_OUT" | sed -E 's/app9_elastica-([0-9]+)\.out/\1/')
        fi
    fi

    if [ -z "$JOBID" ]; then
        echo "Error: No active or recent 'app9_elastica' job found."
        echo "Usage: $0 monitor <jobid>"
        exit 1
    fi

    local LOGFILE="app9_elastica-${JOBID}.out"
    [ ! -f "$LOGFILE" ] && [ -f "logfile_elastica.txt" ] && LOGFILE="logfile_elastica.txt"

    echo "🔎 Monitoring SLURM job ID: $JOBID (Press Ctrl+C to exit)"
    
    while true; do
        clear
        echo "===== SLURM Job Queue Status ====="
        squeue -j "$JOBID" 2>/dev/null || echo "Job $JOBID not in active queue."

        echo -e "\n===== Job Details ====="
        scontrol show job "$JOBID" 2>/dev/null | egrep "JobId=|JobState=|RunTime=|NodeList=|NumCPUs=|NumNodes=|MinMemoryNode=" || echo "No scontrol info available."

        echo -e "\n===== Recent Log Output ($LOGFILE) ====="
        if [ -f "$LOGFILE" ]; then
            tail -n 15 "$LOGFILE"
        else
            echo "Log file $LOGFILE not found."
        fi

        sleep 10
    done
}

if [ "$1" = "monitor" ] || [ "$1" = "status" ] || [[ "$1" =~ ^[0-9]+$ ]]; then
    JOB_ARG="$1"
    [ "$JOB_ARG" = "monitor" ] || [ "$JOB_ARG" = "status" ] && JOB_ARG="$2"
    monitor_job "$JOB_ARG"
    exit 0
fi

if [ -z "$SLURM_JOB_ID" ] && [ -t 0 ]; then
    echo "Usage:"
    echo "  sbatch submit_job_slurm_elastica.sh [YYYY-MM-DD]    Submit Elastica job to SLURM cluster"
    echo "  ./submit_job_slurm_elastica.sh monitor [ID]         Monitor running Elastica job"
    exit 0
fi

set -eo pipefail
ulimit -s unlimited
export SLURM_CPU_BIND=none
export MALLOC_TRIM_THRESHOLD_=65536

if [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    export JOB_KEEP_TIME="$1"
else
    export JOB_KEEP_TIME="${JOB_KEEP_TIME:-$(date +%Y-%m-%d)}"
fi
TARGET_DATA_DIR="data/${JOB_KEEP_TIME}/elastica"
mkdir -p "${PWD}/${TARGET_DATA_DIR}"

CONDA_SH="/home/bhattah/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate modred-dae-torch
fi

export MPLCONFIGDIR="${PWD}/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

echo "Starting Elastica job execution..."
python -u scripts/app9_elastica.py --n-workers 16
