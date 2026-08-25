#!/bin/bash

# ==============================================================================
# Monitor SLURM Job Wrapper
# Usage: ./monitor_job.sh [jobid]
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/submit_job_slurm.sh" monitor "$@"
