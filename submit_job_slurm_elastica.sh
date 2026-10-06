#!/bin/bash

# ==============================================================================
# SLURM Job Submission & Monitoring Script for app9_elastica
#
# Usage:
#   sbatch submit_job_slurm_elastica.sh          Submit job to SLURM cluster
#   ./submit_job_slurm_elastica.sh monitor [ID]  Monitor running/recent job
# ==============================================================================

#SBATCH --job-name=app9_elastica
#SBATCH --output=/dev/null
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
    
    # Auto-detect job ID if not supplied
    if [ -z "$JOBID" ]; then
        JOBID=$(squeue -u "$USER" -o "%i %j" -h 2>/dev/null | grep "app9_elastica" | head -n 1 | awk '{print $1}')
        if [ -z "$JOBID" ]; then
            local LATEST_OUT=$(ls -t data/*/elastica/app9_elastica-*.out app9_elastica-*.out 2>/dev/null | head -n 1)
            [ -n "$LATEST_OUT" ] && JOBID=$(basename "$LATEST_OUT" | sed -E 's/app9_elastica-([0-9]+)\.out/\1/')
        fi
    fi

    if [ -z "$JOBID" ]; then
        echo "Error: No active or recent 'app9_elastica' job found."
        echo "Usage: $0 monitor <jobid>"
        exit 1
    fi

    # Locate logfile in the target date directory or latest data folder
    local LOGFILE="data/${JOB_KEEP_TIME:-$(date +%Y-%m-%d)}/elastica/app9_elastica-${JOBID}.out"
    if [ ! -f "$LOGFILE" ]; then
        local LATEST_DATA_DIR=$(ls -td data/*/elastica/ 2>/dev/null | grep -v "joblib_cache" | head -n 1)
        [ -n "$LATEST_DATA_DIR" ] && [ -f "${LATEST_DATA_DIR}app9_elastica-${JOBID}.out" ] && LOGFILE="${LATEST_DATA_DIR}app9_elastica-${JOBID}.out"
    fi
    [ ! -f "$LOGFILE" ] && [ -f "data/${JOB_KEEP_TIME:-$(date +%Y-%m-%d)}/elastica/logfile.txt" ] && LOGFILE="data/${JOB_KEEP_TIME:-$(date +%Y-%m-%d)}/elastica/logfile.txt"
    [ ! -f "$LOGFILE" ] && [ -f "app9_elastica-${JOBID}.out" ] && LOGFILE="app9_elastica-${JOBID}.out"
    [ ! -f "$LOGFILE" ] && [ -f "logfile_elastica.txt" ] && LOGFILE="logfile_elastica.txt"

    echo "🔎 Monitoring SLURM job ID: $JOBID (Press Ctrl+C to exit)"
    
    while true; do
        clear
        echo "===== SLURM Job Queue Status ====="
        squeue -j "$JOBID" 2>/dev/null || echo "Job $JOBID not in active queue."

        echo -e "\n===== Job Details ====="
        scontrol show job "$JOBID" 2>/dev/null | egrep "JobId=|JobState=|RunTime=|NodeList=|NumCPUs=|NumNodes=|MinMemoryNode=|GRES=" || echo "No scontrol info available."

        echo -e "\n===== Live Resource Usage ====="
        sstat -j "${JOBID}.batch" --format=JobID,MaxRSS,AveRSS,AveCPU,MaxVMSize 2>/dev/null || echo "sstat output unavailable."

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

# -----------------------------------------------------------------------------
# Batch Execution Mode (Runs when submitted via sbatch)
# -----------------------------------------------------------------------------
set -eo pipefail
ulimit -s unlimited
export SLURM_CPU_BIND=none
export MALLOC_TRIM_THRESHOLD_=65536

# Compute workers configuration
export COMPUTE_WORKERS="${COMPUTE_WORKERS:-16}"

# Define target date directory for this run (supports positional $1, JOB_KEEP_TIME env var, or current date)
if [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    export JOB_KEEP_TIME="$1"
else
    export JOB_KEEP_TIME="${JOB_KEEP_TIME:-$(date +%Y-%m-%d)}"
fi
TARGET_DATA_DIR="data/${JOB_KEEP_TIME}/elastica"
JOB_TAG="${SLURM_JOB_ID:-local}"
LOGFILE_NAME="app9_elastica-${JOB_TAG}.out"
TARGET_LOGFILE="${PWD}/${TARGET_DATA_DIR}/${LOGFILE_NAME}"
export TARGET_LOGFILE

# Create directories on master storage
mkdir -p "${PWD}/${TARGET_DATA_DIR}"

# Create convenience symlink logfile.txt -> app9_elastica-${JOB_TAG}.out in target directory
ln -sf "${LOGFILE_NAME}" "${PWD}/${TARGET_DATA_DIR}/logfile.txt"

# Conda environment activation
CONDA_SH="/home/bhattah/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate modred-dae-torch
fi

# Matplotlib cache setup
export MPLCONFIGDIR="${PWD}/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

echo "Checking LaTeX environment..." | tee -a "${TARGET_LOGFILE}"
which latex >/dev/null 2>&1 || echo "Warning: 'latex' not found. Matplotlib usetex=True will fail." | tee -a "${TARGET_LOGFILE}"

# Execute Python job with timing capture
echo "Starting Elastica job execution..." | tee -a "${TARGET_LOGFILE}"
{ time python -u scripts/app9_elastica.py --n-workers "${COMPUTE_WORKERS}"; } 2>&1 | tee -a "${TARGET_LOGFILE}"

# Calculate timing & parallelization efficiency
python3 << 'TIMING_EOF'
import os, re, sys

logfile = os.environ.get('TARGET_LOGFILE', 'logfile.txt')
try:
    with open(logfile) as f:
        content = f.read()

    times = {}
    for line in content.splitlines():
        m = re.match(r'^(real|user|sys)\s+(\d+)m([\d.]+)s', line)
        if m:
            label, mins, secs = m.groups()
            times[label] = int(mins) * 60 + float(secs)

    if len(times) == 3:
        fmt = lambda s: f"{int(s//3600)}h {int((s%3600)//60):02d}m {int(s%60):02d}s"
        real, user, sys_t = times['real'], times['user'], times['sys']
        total_cpu = user + sys_t
        eff = total_cpu / real if real > 0 else 0.0
        total_cpus = float(os.environ.get('COMPUTE_WORKERS', os.environ.get('SLURM_CPUS_PER_TASK', 16)))
        util = (eff / total_cpus) * 100.0

        summary = f"""
========== EXECUTION TIME SUMMARY ==========
Wall-clock time:            {fmt(real)}
User CPU time:              {fmt(user)}
System CPU time:            {fmt(sys_t)}
Total CPU time:             {fmt(total_cpu)}
========== PARALLELIZATION EFFICIENCY ==========
Parallelization efficiency: {eff:.2f}x
Core utilization:           {util:.1f}% of {int(total_cpus)} CPUs
============================================
"""
        print(summary)
        with open(logfile, 'a') as f:
            f.write(summary)
    else:
        print(f"Warning: Could not parse timing output from {logfile}", file=sys.stderr)
except Exception as e:
    print(f"Timing parse warning: {e}", file=sys.stderr)
TIMING_EOF
