#!/bin/bash
# Usage: ./monitor_job.sh <jobid>
# Continuously monitor SLURM job status, resource usage, logs, and GPU usage (via srun)

if [ -z "$1" ]; then
    echo "Usage: $0 <jobid>"
    exit 1
fi

JOBID=$1
LOGFILE="app8_lattice-${JOBID}.out"

echo "🔎 Monitoring SLURM job $JOBID ..."
echo "Press Ctrl+C to exit."

while true; do
    clear
    echo "===== Job Status ====="
    squeue -j $JOBID -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

    echo -e "\n===== Job Details ====="
    scontrol show job $JOBID | egrep "JobId=|JobState=|RunTime=|NodeList=|NumCPUs=|NumNodes=|MinMemoryNode=|GRES="

    echo -e "\n===== Resource Usage (Live) ====="
    sstat -j ${JOBID}.batch --format=JobID,MaxRSS,AveRSS,AveCPU,MaxVMSize 2>/dev/null

    echo -e "\n===== GPU Usage (nvidia-smi via srun) ====="
    # Run nvidia-smi inside the allocated job
    srun --jobid=$JOBID --exclusive -N1 nvidia-smi \
        --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null

    echo -e "\n===== Recent Log Output ($LOGFILE) ====="
    tail -n 10 $LOGFILE 2>/dev/null

    sleep 10
done
