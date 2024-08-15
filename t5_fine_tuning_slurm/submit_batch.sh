#!/bin/bash
#SBATCH --job-name=t5_optimization
#SBATCH --output=%A/slurm_out/%a.out
#SBATCH --error=%A/slurm_err/%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-34%8  # Assumes 35 total jobs (7 ACQF * 5 trials), run 8 at a time

set -e

# Define variables
# ACQF_ARRAY=(EEIPU CArBO LaMBO MS_BO EIPS EI MS_CArBO)
ACQF_ARRAY=(EEIPU CArBO)
exp_name=t5-pipe-multi-new
data_dir=${1:-./inputs}
model_name=${2:-t5-small}

# Calculate ACQF and trial based on SLURM_ARRAY_TASK_ID
acqf_index=$((SLURM_ARRAY_TASK_ID / 5))
trial=$((SLURM_ARRAY_TASK_ID % 5 + 1))
acqf=${ACQF_ARRAY[$acqf_index]}

# Create log directory
mkdir -p log/${acqf}

# Set up unique cache directory
cache_root=${3:-.cachestore}/${acqf}/${SLURM_JOB_ID}_trial_${trial}

# Log file
log_file=log/${acqf}/${exp_name}_trial_${trial}.log

# Run the Python script
srun python optimize_multi.py \
    --exp-name ${exp_name} \
    --trial ${trial} \
    --cache-root ${cache_root} \
    --acqf ${acqf} \
    --data-dir ${data_dir} \
    --model-name ${model_name} \
    --date-now $(date +"%Y-%m-%d_%H%M%S")
    2>&1 | tee ${log_file}

# Clean up cache directory
rm -rf ${cache_root}

# Check if job needs to be requeued
if [ -f "RESTART" ]; then
    run_id=$(cat RESTART)
    rm RESTART
    scontrol requeue ${SLURM_JOB_ID}
fi