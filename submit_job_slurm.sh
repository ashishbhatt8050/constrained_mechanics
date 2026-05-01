#!/bin/bash

# Run this as "sbatch submit_job_slurm.sh"

# Set a name for this run and the resource requirements,
# Exclusive node access (all CPUs), all available memory and 24 hours wall time.
# TODO: install latex on compute nodes

#SBATCH --job-name=app8_lattice
#SBATCH --output=app8_lattice-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --partition=high_cpuq
#SBATCH --account=high_cpu_acct
#SBATCH --qos=high_cpu_qos

# To use GPU, uncomment the following two lines:
# SBATCH --partition=gpuq
# SBATCH --gres=gpu:1
# SBATCH --account=gpu_users

# Send an email when this job aborts, begins or ends.
#SBATCH --mail-type=ALL
# SBATCH --mail-user=

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

# Configure Dask to spill to local scratch ($TMPDIR) to prevent OOM
# This ensures that when workers hit memory limits, they write to disk instead of crashing
export DASK_TEMPORARY_DIRECTORY=${TMPDIR:-/tmp}
export MALLOC_TRIM_THRESHOLD_=65536

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
{ time python -u scripts/app8_lattice.py --clear-cache | tee logfile.txt; } 2>>logfile.txt &
PYTHON_PID=$!

# Start monitoring in background (only useful in interactive SLURM allocations)
if [ -t 1 ]; then
    # We're in an interactive terminal, so monitor the job
    python3 << 'MONITOR_EOF' &
import subprocess
import sys
import os
import time
import re

def run_sstat(jobid):
    """Get resource usage from sstat"""
    try:
        result = subprocess.run(
            ['sstat', '-j', f'{jobid}.batch', '--format=JobID,MaxRSS,AveRSS,AveCPU,MaxVMSize'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout if result.returncode == 0 else "N/A"
    except:
        return "N/A"

def run_scontrol(jobid):
    """Get job details from scontrol"""
    try:
        result = subprocess.run(
            ['scontrol', 'show', 'job', jobid],
            capture_output=True, text=True, timeout=5
        )
        info = {}
        for line in result.stdout.split('\n'):
            if any(x in line for x in ['JobId=', 'JobState=', 'RunTime=', 'NodeList=', 'NumCPUs=', 'GRES=']):
                info[line.strip()] = True
        return info
    except:
        return {}

jobid = os.environ.get('SLURM_JOB_ID', 'unknown')
logfile = 'logfile.txt'
last_lines_shown = 0
monitor_count = 0

print(f"[Monitor] Starting job monitoring for SLURM jobid={jobid}", flush=True)
time.sleep(2)  # Give Python job time to start

while True:
    monitor_count += 1
    if monitor_count % 30 == 1:  # Every 5 minutes at 10s intervals
        print(f"\n[Monitor] Status update - {time.strftime('%H:%M:%S')}", flush=True)
        
        # Job details
        details = run_scontrol(jobid)
        for detail in details.keys():
            print(f"  {detail}", flush=True)
        
        # Resource usage
        sstat_out = run_sstat(jobid)
        if sstat_out != "N/A":
            for line in sstat_out.split('\n')[:3]:
                if line.strip():
                    print(f"  {line.strip()}", flush=True)
    
    # Show new log lines (without clearing screen in background)
    try:
        if os.path.exists(logfile):
            with open(logfile) as f:
                lines = f.readlines()
            if len(lines) > last_lines_shown:
                for line in lines[last_lines_shown:]:
                    if any(x in line for x in ['Solving trajectory:', 'Worker', 'Submitting', 'complete', 'Error']):
                        print(f"[Log] {line.rstrip()}", flush=True)
                last_lines_shown = len(lines)
    except:
        pass
    
    time.sleep(10)
MONITOR_EOF
    MONITOR_PID=$!
fi

# Wait for Python job to complete
wait $PYTHON_PID
PYTHON_EXIT=$?

# Kill monitoring process if it exists
if [ ! -z "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null
fi

if [ $PYTHON_EXIT -ne 0 ]; then
    echo "Warning: Python job exited with code $PYTHON_EXIT" >&2
fi

# Parse and format timing output with efficiency calculation
python3 << 'TIMING_EOF'
import re
import sys
import os

try:
    with open('logfile.txt') as f:
        time_output = f.read()
    
    # Extract timing values in format "XXXmYY.ZZZs" (only lines starting with real/user/sys)
    times = {}
    for line in time_output.split('\n'):
        match = re.match(r'^(real|user|sys)\s+(\d+)m([\d.]+)s', line)
        if match:
            label, mins, secs = match.groups()
            total_secs = int(mins) * 60 + float(secs)
            times[label] = total_secs
    
    if len(times) == 3:
        def secs_to_hms(seconds):
            """Convert seconds to Hours:Minutes:Seconds format"""
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h}h {m:02d}m {s:02d}s"
        
        real = times['real']
        user = times['user']
        sys_time = times['sys']
        total_cpu = user + sys_time
        efficiency = total_cpu / real if real > 0 else 0.0
        cpu_utilization = (efficiency / 32.0) * 100.0
        
        # Format output
        summary = f"""
========== EXECUTION TIME SUMMARY ==========
Wall-clock time:               {secs_to_hms(real)}
User CPU time:                 {secs_to_hms(user)}
System CPU time:               {secs_to_hms(sys_time)}
Total CPU time:                {secs_to_hms(total_cpu)}
========== PARALLELIZATION EFFICIENCY ==========
Parallelization efficiency:    {efficiency:.2f}x
Core utilization:              {cpu_utilization:.1f}% of 32 available CPUs
============================================
"""
        print(summary, end='')
        # Append to logfile for permanent record
        with open('logfile.txt', 'a') as f:
            f.write(summary)
    else:
        print("Warning: Could not parse timing output", file=sys.stderr)
        
except Exception as e:
    print(f"Error processing timing: {e}", file=sys.stderr)
TIMING_EOF

# Move the SLURM output file to the latest data directory
LATEST_DATA_DIR=$(ls -td data/*/ | head -n 1)
if [ -d "$LATEST_DATA_DIR" ]; then
    mv "app8_lattice-${SLURM_JOB_ID}.out" "$LATEST_DATA_DIR"
fi
