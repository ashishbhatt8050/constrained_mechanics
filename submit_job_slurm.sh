#!/bin/bash

# ==============================================================================
# Unified SLURM Job Submission & Monitoring Script for Constrained Mechanics
#
# Supported Models:
#   - lattice  (scripts/app8_lattice.py)
#   - elastica (scripts/app9_elastica.py)
#
# Supported Modes:
#   - reproduce  (Full model reproduction sweep across tolerances & dt)
#   - predict    (Fast parameter sweep prediction on unseen samples)
#
# Usage:
#   sbatch submit_job_slurm.sh [app] [mode] [experiment_name]
#   sbatch submit_job_slurm.sh <existing_dir_or_experiment_name>
#   ./submit_job_slurm.sh monitor [ID]
#   ./monitor_job.sh [ID]
# ==============================================================================

#SBATCH --job-name=constrained_mechanics
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

# -----------------------------------------------------------------------------
# Interactive Monitoring Mode
# -----------------------------------------------------------------------------
monitor_job() {
    local JOBID="$1"
    
    # Auto-detect job ID if not supplied
    if [ -z "$JOBID" ]; then
        JOBID=$(squeue -u "$USER" -o "%i %j" -h 2>/dev/null | grep -E "app8_lattice|app9_elastica|lattice|elastica|constrained_mechanics|cm_" | head -n 1 | awk '{print $1}')
        if [ -z "$JOBID" ]; then
            local LATEST_OUT=$(ls -t data/*/*.out data/*/*/*.out *.out 2>/dev/null | head -n 1)
            [ -n "$LATEST_OUT" ] && JOBID=$(basename "$LATEST_OUT" | sed -E 's/.*-([0-9]+)\.out/\1/')
        fi
    fi

    if [ -z "$JOBID" ]; then
        echo "Error: No active or recent job found."
        echo "Usage: $0 monitor <jobid>"
        exit 1
    fi

    # Locate logfile in the target experiment directory or latest data folder
    local LOGFILE=""
    if [ -n "$EXPERIMENT_NAME" ] && [ -d "data/${EXPERIMENT_NAME}" ]; then
        LOGFILE=$(ls -t "data/${EXPERIMENT_NAME}/"*-"${JOBID}.out" "data/${EXPERIMENT_NAME}/logfile.txt" 2>/dev/null | head -n 1)
    fi
    if [ -z "$LOGFILE" ] || [ ! -f "$LOGFILE" ]; then
        LOGFILE=$(ls -t data/*/*-"${JOBID}.out" data/*/*/*-"${JOBID}.out" *-"${JOBID}.out" 2>/dev/null | head -n 1)
    fi
    if [ -z "$LOGFILE" ] || [ ! -f "$LOGFILE" ]; then
        local LATEST_DATA_DIR=$(ls -td data/*/ 2>/dev/null | grep -v "joblib_cache" | head -n 1)
        [ -n "$LATEST_DATA_DIR" ] && LOGFILE=$(ls -t "${LATEST_DATA_DIR}"*-"${JOBID}.out" "${LATEST_DATA_DIR}"*/*-"${JOBID}.out" "${LATEST_DATA_DIR}logfile.txt" 2>/dev/null | head -n 1)
    fi
    [ ! -f "$LOGFILE" ] && [ -f "logfile.txt" ] && LOGFILE="logfile.txt"

    echo "🔎 Monitoring SLURM job ID: $JOBID (Press Ctrl+C to exit)"
    
    while true; do
        clear
        echo "===== SLURM Job Queue Status ====="
        squeue -j "$JOBID" 2>/dev/null || echo "Job $JOBID not in active queue."

        echo -e "\n===== Job Details ====="
        scontrol show job "$JOBID" 2>/dev/null | egrep "JobId=|JobName=|JobState=|RunTime=|NodeList=|NumCPUs=|NumNodes=|MinMemoryNode=|GRES=" || echo "No scontrol info available."

        echo -e "\n===== Live Resource Usage ====="
        sstat -j "${JOBID}.batch" --format=JobID,MaxRSS,AveRSS,AveCPU,MaxVMSize 2>/dev/null || echo "sstat output unavailable."

        echo -e "\n===== GPU Usage (nvidia-smi via srun) ====="
        srun --jobid="$JOBID" --exclusive -N1 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "No GPU assigned or srun unavailable."

        echo -e "\n===== Recent Log Output ($LOGFILE) ====="
        if [ -n "$LOGFILE" ] && [ -f "$LOGFILE" ]; then
            tail -n 15 "$LOGFILE"
        else
            echo "Log file $LOGFILE not found."
        fi

        sleep 10
    done
}

# -----------------------------------------------------------------------------
# Dispatcher: Monitor command or interactive usage help
# -----------------------------------------------------------------------------
if [ "$1" = "monitor" ] || [ "$1" = "status" ] || [[ "$1" =~ ^[0-9]{5,}$ ]]; then
    JOB_ARG="$1"
    [ "$JOB_ARG" = "monitor" ] || [ "$JOB_ARG" = "status" ] && JOB_ARG="$2"
    monitor_job "$JOB_ARG"
    exit 0
fi

if [ -z "$SLURM_JOB_ID" ] && [ -t 0 ] && [ "$#" -eq 0 ]; then
    echo "=============================================================================="
    echo "Constrained Mechanics SLURM Submission & Monitoring Helper"
    echo "=============================================================================="
    echo "Usage (Submission):"
    echo "  sbatch submit_job_slurm.sh [app] [mode] [experiment_name]"
    echo "    app:             'lattice' (default) or 'elastica'"
    echo "    mode:            'reproduce' (default for lattice) or 'predict' (default for elastica)"
    echo "    experiment_name: optional custom folder name under data/ (default: \${DATE}_\${APP}_\${MODE})"
    echo ""
    echo "Resuming Existing Experiments:"
    echo "  sbatch submit_job_slurm.sh <experiment_name_or_dir>"
    echo "    Example: sbatch submit_job_slurm.sh 2026-10-15_lattice_reproduce"
    echo "    Example: sbatch submit_job_slurm.sh 2026-10-15"
    echo "    Example: sbatch submit_job_slurm.sh data/2026-10-15/elastica"
    echo ""
    echo "Monitoring:"
    echo "  ./submit_job_slurm.sh monitor [jobid]    Monitor running/recent job"
    echo "  ./monitor_job.sh [jobid]                 Shortcut to monitor"
    echo "=============================================================================="
    exit 0
fi

# -----------------------------------------------------------------------------
# Batch Execution Mode (Runs when submitted via sbatch or headless)
# -----------------------------------------------------------------------------
set -eo pipefail

DATE_TAG="$(date +%Y-%m-%d)"

# Argument parsing and auto-detection
APP_ARG=""
MODE_ARG=""
EXP_ARG=""

# Helper function to strip path prefixes
sanitize_exp_name() {
    local val="$1"
    val="${val#data/}"
    val="${val%/}"
    echo "$val"
}

# Auto-detect if argument 1 is an existing folder or date
if [ -n "$1" ]; then
    CLEANED_ARG1="$(sanitize_exp_name "$1")"
    
    if [ "$CLEANED_ARG1" = "lattice" ] || [ "$CLEANED_ARG1" = "elastica" ]; then
        APP_ARG="$CLEANED_ARG1"
        [ -n "$2" ] && MODE_ARG="$2"
        [ -n "$3" ] && EXP_ARG="$(sanitize_exp_name "$3")"
    elif [ -d "data/${CLEANED_ARG1}" ] || [ -d "${CLEANED_ARG1}" ] || [[ "$CLEANED_ARG1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2} ]]; then
        # Argument 1 is an experiment directory or date
        EXP_ARG="$CLEANED_ARG1"
        [ -n "$2" ] && MODE_ARG="$2"

        # Determine app from directory contents or name
        if [[ "$CLEANED_ARG1" =~ elastica ]] || [ -d "data/${CLEANED_ARG1}/elastica" ]; then
            APP_ARG="elastica"
            [ -d "data/${CLEANED_ARG1}/elastica" ] && EXP_ARG="${CLEANED_ARG1}/elastica"
        elif [[ "$CLEANED_ARG1" =~ lattice ]]; then
            APP_ARG="lattice"
        else
            APP_ARG="${APP:-lattice}"
        fi

        # Determine mode from name if not explicitly set
        if [ -z "$MODE_ARG" ]; then
            if [[ "$CLEANED_ARG1" =~ predict ]]; then
                MODE_ARG="predict"
            elif [[ "$CLEANED_ARG1" =~ reproduce ]]; then
                MODE_ARG="reproduce"
            fi
        fi
    else
        # Argument 1 is custom experiment name
        EXP_ARG="$CLEANED_ARG1"
        [ -n "$2" ] && MODE_ARG="$2"
    fi
fi

# Resolve APP (CLI arg > Env var > Default)
if [ -n "$APP_ARG" ]; then
    APP="$APP_ARG"
elif [ -n "$APP" ]; then
    APP="$APP"
else
    APP="lattice"
fi
if [ "$APP" != "lattice" ] && [ "$APP" != "elastica" ]; then
    echo "Warning: Unknown app '$APP'. Defaulting to lattice."
    APP="lattice"
fi

# Resolve MODE (CLI arg > Env var > Default)
if [ -n "$MODE_ARG" ]; then
    MODE="$MODE_ARG"
elif [ -n "$MODE" ]; then
    MODE="$MODE"
elif [ "$APP" = "elastica" ]; then
    MODE="predict"
else
    MODE="reproduce"
fi
if [ "$MODE" != "predict" ] && [ "$MODE" != "reproduce" ]; then
    echo "Warning: Unknown mode '$MODE'. Defaulting to reproduce."
    MODE="reproduce"
fi

# Resolve EXPERIMENT_NAME (CLI arg > Env var > Default)
if [ -n "$EXP_ARG" ]; then
    EXPERIMENT_NAME="$EXP_ARG"
elif [ -n "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="$(sanitize_exp_name "$EXPERIMENT_NAME")"
else
    EXPERIMENT_NAME="${DATE_TAG}_${APP}_${MODE}"
fi

export APP
export MODE
export EXPERIMENT_NAME
export EXPERIMENT_MODE="${MODE}"

# Update SLURM job name dynamically if running under SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="${APP}_${MODE}" 2>/dev/null || true
fi

# System & memory configuration
ulimit -s unlimited
export SLURM_CPU_BIND=none
export MALLOC_TRIM_THRESHOLD_=65536

# Compute workers configuration for Dask LocalCluster & task batching.
# Default: 30 workers (reserves 2 cores on 32-core node for Dask scheduler/driver/OS to prevent latency).
export COMPUTE_WORKERS="${COMPUTE_WORKERS:-30}"
export DASK_NUM_WORKERS="${COMPUTE_WORKERS}"

# Parameter space batch waves
if [ "$APP" = "elastica" ]; then
    export OMEGA2_TRAIN_WAVES="${OMEGA2_TRAIN_WAVES:-1}"
    export OMEGA2_TEST_WAVES="${OMEGA2_TEST_WAVES:-0.3334}"
else
    export OMEGA2_TRAIN_WAVES="${OMEGA2_TRAIN_WAVES:-1}"
    export OMEGA2_TEST_WAVES="${OMEGA2_TEST_WAVES:-0.3334}"
fi

# Local scratch directory setup for fast compute node NVMe I/O
LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/constrained_mechanics_${SLURM_JOB_ID:-$$}}"
mkdir -p "$LOCAL_SCRATCH"
export SCRATCH_DIR="$LOCAL_SCRATCH"
export DASK_TEMPORARY_DIRECTORY="${LOCAL_SCRATCH}/dask_tmp"
mkdir -p "$DASK_TEMPORARY_DIRECTORY"

# Target data directories
TARGET_DATA_DIR="data/${EXPERIMENT_NAME}"
JOB_TAG="${SLURM_JOB_ID:-local}"
LOGFILE_NAME="${APP}-${JOB_TAG}.out"
TARGET_LOGFILE="${PWD}/${TARGET_DATA_DIR}/${LOGFILE_NAME}"
export TARGET_LOGFILE

# Create directories on master storage and local scratch
mkdir -p "${PWD}/${TARGET_DATA_DIR}"
mkdir -p "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}"
mkdir -p "${LOCAL_SCRATCH}/data"

# Create convenience symlink logfile.txt -> ${APP}-${JOB_TAG}.out in target directory
ln -sf "${LOGFILE_NAME}" "${PWD}/${TARGET_DATA_DIR}/logfile.txt"

echo "==============================================================================" | tee -a "${TARGET_LOGFILE}"
echo "Constrained Mechanics Unified SLURM Job Execution" | tee -a "${TARGET_LOGFILE}"
echo "==============================================================================" | tee -a "${TARGET_LOGFILE}"
echo "Application:      ${APP}" | tee -a "${TARGET_LOGFILE}"
echo "Mode:             ${MODE}" | tee -a "${TARGET_LOGFILE}"
echo "Experiment Name:  ${EXPERIMENT_NAME}" | tee -a "${TARGET_LOGFILE}"
echo "Target Dir:       ${TARGET_DATA_DIR}" | tee -a "${TARGET_LOGFILE}"
echo "Local Scratch:    ${LOCAL_SCRATCH}" | tee -a "${TARGET_LOGFILE}"
echo "Job ID:           ${JOB_TAG}" | tee -a "${TARGET_LOGFILE}"
echo "Log File:         ${TARGET_LOGFILE}" | tee -a "${TARGET_LOGFILE}"
echo "Workers:          ${COMPUTE_WORKERS}" | tee -a "${TARGET_LOGFILE}"
echo "==============================================================================" | tee -a "${TARGET_LOGFILE}"

# Automatic synchronization trap: copy target run checkpoints/data from local scratch back to master node disk
sync_scratch_to_master() {
    echo "🔄 Syncing run directory (${TARGET_DATA_DIR}) from local scratch to master node storage..."
    if [ -d "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}" ]; then
        # Ensure no stale/frozen logfiles in scratch can overwrite the live master log
        rm -f "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}/"*.out "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}/"*.log "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}/logfile*.txt" 2>/dev/null || true
        mkdir -p "${PWD}/${TARGET_DATA_DIR}"
        rsync -au --exclude="*.out" --exclude="*.log" --exclude="logfile*.txt" "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}/" "${PWD}/${TARGET_DATA_DIR}/" 2>/dev/null || true
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

echo "Checking LaTeX environment..." | tee -a "${TARGET_LOGFILE}"
which latex >/dev/null 2>&1 || echo "Warning: 'latex' not found. Matplotlib usetex=True will fail." | tee -a "${TARGET_LOGFILE}"

# Selective pre-staging: copy target experiment directory from master to scratch
if [ -d "${PWD}/${TARGET_DATA_DIR}" ] && [ "$(ls -A "${PWD}/${TARGET_DATA_DIR}" 2>/dev/null)" ]; then
    echo "📥 Pre-staging target experiment directory (${TARGET_DATA_DIR}) from master storage to local scratch..." | tee -a "${TARGET_LOGFILE}"
    rsync -au --exclude="*.out" --exclude="*.log" --exclude="logfile*.txt" "${PWD}/${TARGET_DATA_DIR}/" "${LOCAL_SCRATCH}/${TARGET_DATA_DIR}/" 2>/dev/null || true
fi

# Pre-stage symbolic expression pickle files from master node data to local scratch
find "${PWD}/data" -maxdepth 1 -name "symbolic_expr_*.pickle" -exec cp -t "${LOCAL_SCRATCH}/data/" {} + 2>/dev/null || true

# Execute Python job with timing capture
echo "Starting ${APP} (${MODE}) job execution..." | tee -a "${TARGET_LOGFILE}"

if [ "$APP" = "elastica" ]; then
    SCRIPT_PATH="scripts/app9_elastica.py"
else
    SCRIPT_PATH="scripts/app8_lattice.py"
fi

{ time python -u "${SCRIPT_PATH}" --${MODE} --experiment-name "${EXPERIMENT_NAME}"; } 2>&1 | tee -a "${TARGET_LOGFILE}"

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
        total_cpus = float(os.environ.get('COMPUTE_WORKERS', os.environ.get('SLURM_CPUS_PER_TASK', 32)))
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
