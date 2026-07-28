#!/bin/bash
CONCURRENCY=3
INTERVAL=3600
OUTPUT_DIR=/scratch/mch/wolfensb/rfo_recompute
DAYFILE="days.txt"

# Read valid YYJJJ lines into an array
mapfile -t DAYS < <(grep -vE '^\s*($|#)' "$DAYFILE")

NDAYS=${#DAYS[@]}
if [[ $NDAYS -eq 0 ]]; then
  echo "No days found in $DAYFILE"
  exit 1
fi

# How many tasks per day (24 for 1h interval)
TASKS_PER_DAY=$(( 86400 / INTERVAL ))
TOTAL_TASKS=$(( NDAYS * TASKS_PER_DAY ))

echo "Submitting 1 array: NDAYS=$NDAYS, TASKS_PER_DAY=$TASKS_PER_DAY, TOTAL_TASKS=$TOTAL_TASKS, concurrency=$CONCURRENCY"

# Export the day list as a space-separated string
DAYS_STR="${DAYS[*]}"

sbatch \
  --export=ALL,OUTPUT_DIR="${OUTPUT_DIR}",INTERVAL="${INTERVAL}",DAYS_STR="${DAYS_STR}",TASKS_PER_DAY="${TASKS_PER_DAY}" \
  --array=0-$((TOTAL_TASKS-1))%${CONCURRENCY} \
  run.sbatch
