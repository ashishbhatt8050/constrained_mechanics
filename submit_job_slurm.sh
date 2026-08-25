#!/bin/bash

# ==============================================================================
# SLURM Job Submission & Monitoring Script for app8_lattice
#
# Usage:
#   sbatch submit_job_slurm.sh          Submit job to SLURM cluster
#   ./submit_job_slurm.sh monitor [ID]  Monitor running/recent job
# ==============================================================================

#SBATCH --job-name=app8_lattice
#SBATCH --output=app8_lattice-%j.out
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

# -----------------------------------------------------------------------------
# Interactive Monitoring Mode
# -----------------------------------------------------------------------------
monitor_job() {
    local JOBID="$1"
    
    # Auto-detect job ID if not supplied
    if [ -z "$JOBID" ]; then
        JOBID=$(squeue -u "$USER" -o "%i %j" -h 2>/dev/null | grep "app8_lattice" | head -n 1 | awk '{print $1}')
        if [ -z "$JOBID" ]; then
            local LATEST_OUT=$(ls -t app8_lattice-*.out 2>/dev/null | head -n 1)
            [ -n "$LATEST_OUT" ] && JOBID=$(echo "$LATEST_OUT" | sed -E 's/app8_lattice-([0-9]+)\.out/\1/')
        fi
    fi

    if [ -z "$JOBID" ]; then
        echo "Error: No active or recent 'app8_lattice' job found."
        echo "Usage: $0 monitor <jobid>"
        exit 1
    fi

    local LOGFILE="app8_lattice-${JOBID}.out"
    [ ! -f "$LOGFILE" ] && [ -f "logfile.txt" ] && LOGFILE="logfile.txt"

    echo "🔎 Monitoring SLURM job ID: $JOBID (Press Ctrl+C to exit)"
    
    while true; do
        clear
        echo "===== SLURM Job Queue Status ====="
        squeue -j "$JOBID" 2>/dev/null || echo "Job $JOBID not in active queue."

        echo -e "\n===== Job Details ====="
        scontrol show job "$JOBID" 2>/dev/null | egrep "JobId=|JobState=|RunTime=|NodeList=|NumCPUs=|NumNodes=|MinMemoryNode=|GRES=" || echo "No scontrol info available."

        echo -e "\n===== Live Resource Usage ====="
        sstat -j "${JOBID}.batch" --format=JobID,MaxRSS,AveRSS,AveCPU,MaxVMSize 2>/dev/null || echo "sstat output unavailable."

        echo -e "\n===== GPU Usage (nvidia-smi via srun) ====="
        srun --jobid="$JOBID" --exclusive -N1 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "No GPU assigned or srun unavailable."

        echo -e "\n===== Recent Log Output ($LOGFILE) ====="
        if [ -f "$LOGFILE" ]; then
            tail -n 15 "$LOGFILE"
        else
            echo "Log file $LOGFILE not found."
        fi

        sleep 10
    done
}

# -----------------------------------------------------------------------------
# Dispatcher: Monitor command or interactive invocation
# -----------------------------------------------------------------------------
if [ "$1" = "monitor" ] || [ "$1" = "status" ] || [[ "$1" =~ ^[0-9]+$ ]]; then
    JOB_ARG="$1"
    [ "$JOB_ARG" = "monitor" ] || [ "$JOB_ARG" = "status" ] && JOB_ARG="$2"
    monitor_job "$JOB_ARG"
    exit 0
fi

if [ -z "$SLURM_JOB_ID" ] && [ -t 0 ]; then
    echo "Usage:"
    echo "  sbatch submit_job_slurm.sh [YYYY-MM-DD]    Submit job to SLURM cluster (optional target date)"
    echo "  ./submit_job_slurm.sh monitor [ID]         Monitor running job"
    exit 0
fi

# -----------------------------------------------------------------------------
# Batch Execution Mode (Runs when submitted via sbatch)
# -----------------------------------------------------------------------------
set -eo pipefail

# System & memory configuration
ulimit -s unlimited
export SLURM_CPU_BIND=none
export MALLOC_TRIM_THRESHOLD_=65536

# Local scratch directory setup for fast compute node NVMe I/O
LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/constrained_mechanics_${SLURM_JOB_ID:-$$}}"
mkdir -p "$LOCAL_SCRATCH"
export SCRATCH_DIR="$LOCAL_SCRATCH"
export DASK_TEMPORARY_DIRECTORY="${LOCAL_SCRATCH}/dask_tmp"
mkdir -p "$DASK_TEMPORARY_DIRECTORY"

# Define target date directory for this run (supports positional $1, JOB_KEEP_TIME env var, or current date)
if [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    export JOB_KEEP_TIME="$1"
else
    export JOB_KEEP_TIME="${JOB_KEEP_TIME:-$(date +%Y-%m-%d)}"
fi
TARGET_DATA_DIR="data/${JOB_KEEP_TIME}"

# Create directories on master storage and local scratch
mkdir -p "${PWD}/${TARGET_DATA_DIR}"
mkdir -p "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}"

# Automatic synchronization trap: copy target run checkpoints/data from local scratch back to master node disk
sync_scratch_to_master() {
    echo "🔄 Syncing run directory (${TARGET_DATA_DIR}) from local scratch to master node storage..."
    if [ -d "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}" ]; then
        mkdir -p "${PWD}/${TARGET_DATA_DIR}"
        rsync -a "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}/" "${PWD}/${TARGET_DATA_DIR}/" 2>/dev/null || cp -r "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}/"* "${PWD}/${TARGET_DATA_DIR}/" 2>/dev/null || true
    fi
    # Also sync any top-level symbolic expression pickles generated on scratch
    find "${LOCAL_SCRATCH}/data" -maxdepth 1 -name "symbolic_expr_*.pickle" -exec cp -t "${PWD}/data/" {} + 2>/dev/null || true
}
trap sync_scratch_to_master EXIT INT TERM

# Conda environment activation
CONDA_SH="/home/bhattah/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate modred-dae-torch
fi

# Matplotlib cache setup
export MPLCONFIGDIR="${PWD}/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

echo "Checking LaTeX environment..."
which latex >/dev/null 2>&1 || echo "Warning: 'latex' not found. Matplotlib usetex=True will fail."

# Selective pre-staging: copy ONLY the specific target date directory and symbolic pickles from master to scratch
if [ -d "${PWD}/${TARGET_DATA_DIR}" ]; then
    echo "📥 Pre-staging target date directory (${TARGET_DATA_DIR}) from master node storage to local scratch..."
    rsync -a "${PWD}/${TARGET_DATA_DIR}/" "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}/" 2>/dev/null || cp -r "${PWD}/${TARGET_DATA_DIR}/"* "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}/" 2>/dev/null || true
fi

# Pre-stage symbolic expression pickle files (small files) from master node data to local scratch
find "${PWD}/data" -maxdepth 1 -name "symbolic_expr_*.pickle" -exec cp -t "${LOCAL_SCRATCH}/data/" {} + 2>/dev/null || true

# Execute Python job with timing capture
echo "Starting job execution..."
{ time python -u scripts/app8_lattice.py; } 2>&1 | tee logfile.txt

# Calculate timing & parallelization efficiency
python3 << 'TIMING_EOF'
import re, sys

try:
    with open('logfile.txt') as f:
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
        util = (eff / 32.0) * 100.0

        summary = f"""
========== EXECUTION TIME SUMMARY ==========
Wall-clock time:            {fmt(real)}
User CPU time:              {fmt(user)}
System CPU time:            {fmt(sys_t)}
Total CPU time:             {fmt(total_cpu)}
========== PARALLELIZATION EFFICIENCY ==========
Parallelization efficiency: {eff:.2f}x
Core utilization:           {util:.1f}% of 32 CPUs
============================================
"""
        print(summary)
        with open('logfile.txt', 'a') as f:
            f.write(summary)
    else:
        print("Warning: Could not parse timing output from logfile.txt", file=sys.stderr)
except Exception as e:
    print(f"Timing parse warning: {e}", file=sys.stderr)
TIMING_EOF

# Relocate SLURM log output to data directory
LATEST_DATA_DIR=$(ls -td data/*/ 2>/dev/null | grep -v "joblib_cache" | head -n 1)
if [ -n "$SLURM_JOB_ID" ] && [ -d "$LATEST_DATA_DIR" ]; then
    [ -f "app8_lattice-${SLURM_JOB_ID}.out" ] && mv "app8_lattice-${SLURM_JOB_ID}.out" "$LATEST_DATA_DIR/"
fi
