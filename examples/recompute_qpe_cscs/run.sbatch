#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 06:00:00
#SBATCH --job-name=rfo1h
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

export PATH=/scratch/mch/wolfensb/rainforest_venv/bin/:$PATH
export RAINFOREST_DATAPATH=/store_new/mch/msrad/radar/rainforest_data/

# interval from env var (fix typo: INTERVAL, not INVERVAL)
interval=$INTERVAL

# Reconstruct day list
read -r -a DAYS <<< "$DAYS_STR"

# Map global task id -> (day_index, hour_index)
day_index=$(( SLURM_ARRAY_TASK_ID / TASKS_PER_DAY ))
hour_index=$(( SLURM_ARRAY_TASK_ID % TASKS_PER_DAY ))

YYJJJ="${DAYS[$day_index]}"
yy="${YYJJJ:0:2}"
DOY="${YYJJJ:2:3}"
YEAR=$((2000 + 10#$yy))

# Compute midnight of that day as epoch
DATE=$(date -d "${YEAR}-01-01 +$((10#$DOY - 1)) days" +%Y%m%d)
start_epoch=$(date -d "${DATE} 00:00" +%s)
task_start_epoch=$(( start_epoch + hour_index * interval ))
task_end_epoch=$(( task_start_epoch + interval ))

task_start=$(date -d "@${task_start_epoch}" +%Y%m%d%H%M)
task_end=$(date -d "@${task_end_epoch}" +%Y%m%d%H%M)
day=$(date -d "@${task_start_epoch}" +%Y%j)

echo "YYJJJ=$YYJJJ day_index=$day_index hour_index=$hour_index"
echo "Running QPE: ${task_start} → ${task_end}"

mkdir -p "${OUTPUT_DIR}/${day}/"
qpe_compute \
  -s "${task_start}" \
  -e "${task_end}" \
  -c config.yml \
  -m '{"RF_dualpol":"RFO_BETA_-0.5_BC_spline.p"}' \
  -o "${OUTPUT_DIR}/${day}/"
