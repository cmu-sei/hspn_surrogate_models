#!/bin/bash
#SBATCH --account=ARLAP44862YFR
#SBATCH --job-name=hspn-hpo
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --qos=frontier

##SBATCH --exclusive
##SBATCH --ntasks=$NTASKS
##SBATCH --gres=gpu:a100:$NUM_WORKERS
##SBATCH --gpus=$NUM_WORKERS

#SBATCH --ntasks=4
##SBATCH --gres=gpu:a100:2
##SBATCH --gpus=2
#SBATCH --constraint=mla

#SBATCH --gpus-per-task=1
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
NUM_WORKERS=2               # total number of workers our queue can work with
NTASKS=$((NUM_WORKERS + 2)) # workers + redis + controller
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TEMP_SCRIPT=$(mktemp)                   # entrypoint script for container, cleaned up via trap
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")" # Parent dir

# REDIS_SLURM_SCRIPT="rq_server.slurm"
# WORKER_SLURM_SCRIPT="rq_worker.slurm"
# CONTROLLER_SLURM_SCRIPT="hpo.slurm"
# CONTROLLER_SLURM_SCRIPT="submit.slurm train --config-name=train_sweep"

VOLUMES="${VOLUMES:-$PROJECT_ROOT:/workspace}"
CONTAINER_CMD="singularity exec --bind ${VOLUMES//,/ --bind } --nv ${CONTAINER:-"project.sif"}"

RHOST_FILENAME=redis_host.txt # For sharing the assigned hostname

# Launch Queue Server
echo "Launching Redis server..."
cmd="$CONTAINER_CMD bash -c \"echo \$(hostname) > /workspace/slurm/$RHOST_FILENAME && redis-server --bind 0.0.0.0\""
REDIS_JOB_ID=$(srun --parsable --ntasks=1 --gpus=0 $cmd)
echo "Redis Job ID: $REDIS_JOB_ID"

while [ ! -f "$SCRIPT_DIR/$RHOST_FILENAME" ]; do
  echo "Waiting for Redis hostname..."
  sleep 2
done

REDIS_HOST=$(cat "$SCRIPT_DIR/$RHOST_FILENAME")
echo "Redis is running on $REDIS_HOST"

# Launch workers
WORKER_CMD="$CONTAINER_CMD bash -c \"nvidia-smi && rq worker --url redis://$REDIS_HOST:6379 $QUEUE_NAME\""
echo "Launching $NUM_WORKERS RQ workers..."
WORKER_JOB_IDS=()
for i in $(seq 1 $NUM_WORKERS); do
  JOB_ID=$(srun --parsable --ntasks=1 --gpus=1 --gres=gpu:a100:1 $WORKER_CMD)
  echo "  Worker $i Job ID: $JOB_ID"
  WORKER_JOB_IDS+=($JOB_ID)
done

# Launch HPO Controller
echo "Launching HPO controller job..."
cmd="$CONTAINER_CMD bash -c \"\
python -m hspn.train --config-name=train_sweep \
hydra/launcher=rq hydra.launcher.redis_host=$REDIS_HOST hydra.launcher.redis_port=6379 hydra.launcher.queue=$QUEUE_NAME
\""

echo "Executing: $cmd"
CONTROLLER_JOB_ID=$(srun --parsable --ntasks=1 --gpus=0 $cmd)
echo "Controller Job ID: $CONTROLLER_JOB_ID"
squeue --job "$CONTROLLER_JOB_ID"

# We could wait for this to finish?
# echo "Waiting for controller job $CONTROLLER_JOB_ID to complete..."
# squeue --job "$CONTROLLER_JOB_ID" > /dev/null 2>&1
# while [ $? -eq 0 ]; do
#     sleep 30
#     squeue --job "$CONTROLLER_JOB_ID" > /dev/null 2>&1
# done
# echo "Sweep complete. Controller job has exited."
