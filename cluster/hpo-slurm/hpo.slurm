#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -q frontier
#SBATCH --constraint=mla

setopt verbose
set -e

echo "JOB START: $(date)"

# Required environment variables; these are expected to be set by the launcher.
: "${N_WORKERS:?Number of workers not set}"
: "${N_TRIALS:?Number of trials not set}"
: "${TRAIN_OPTS:?Train options not set}"
: "${CONTAINER:=hspn.sif}"

REDIS_PORT=6379
QUEUE_NAME="hspn_hpo_queue"
echo $SHELL

module load nvidia

checkcmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "[X] No $1 cmd"
    echo 1
  else
    echo "[OK] Have $1 cmd"
    echo 0
  fi
}

if ! checkcmd apptainer; then
  echo "trying to load apptainer"
  module load apptainer
fi

echo "Checking available GPUs..."
nvidia-smi -L || {
  echo "ERROR: No GPUs available"
  exit 1
}

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

# Container mounts
VOLUMES="${VOLUMES:-$WORKDIR:/workspace}"

# Setup cleanup ----------------------------------------------------------------
at_exit() {
  set +e
  echo "Controller finished with exit code $controller_exit"
  echo -n "Cleaning up background processes..."
  set -x
  for pid in "${worker_pids[@]}"; do
    if kill -0 $pid 2>/dev/null; then
      kill $pid
    fi
  done

  if kill -0 $redis_pid 2>/dev/null; then
    kill $redis_pid
  fi
  set +x

  echo "JOB END: $(date)"
  exit $controller_exit
}
trap at_exit EXIT

mkdir -p logs/workers

# Start Queue Server -----------------------------------------------------------
redis_host=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
echo "Using node '$redis_host' as Redis host."

# Non-exclusive
echo -n "Starting Redis server on $redis_host..."
srun --ntasks=1 --cpus-per-task=1 --mem=4G \
  apptainer exec --bind "${VOLUMES//,/ --bind }" "$CONTAINER" \
  bash -c "redis-server --port $REDIS_PORT --bind 0.0.0.0" &
redis_pid=$!
echo "done"
echo "REDIS_PID=$redis_pid"

# Short pause, it's ok if this is not long enough the workers will poll
echo -n "Pausing to give queue server a moment to initialize..."
sleep 3
echo "continuing"

# Launch Worker Processes –-----------------------------------------------------
worker_pids=()
for ((i = 1; i <= N_WORKERS; i++)); do
  echo "Launching worker $i..."
  srun --exclusive --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:1 \
    apptainer exec --nv --bind "${VOLUMES//,/ --bind }" "$CONTAINER" \
    bash -c "rq worker -S cloudpickle.cloudpickle --url redis://$redis_host:$REDIS_PORT $QUEUE_NAME" &
  worker_pids+=($!)
done
echo "$N_WORKERS workers launched."

echo -n "Pausing to give workers a moment to initialize..."
sleep 5
echo "continuing"

# Launch the Controller Process ------------------------------------------------
controller_cmd="python -m hspn.train hydra/launcher=rq \
                  hydra.launcher.redis.host=$redis_host \
                  hydra.launcher.redis.port=$REDIS_PORT \
                  hydra.launcher.queue=$QUEUE_NAME \
                  $TRAIN_OPTS"
controller_cmd=$(echo "$controller_cmd" | tr '\n' ' ') # strip newlines

echo "Starting controller with command:"
echo "$controller_cmd"

apptainer exec --bind "${VOLUMES//,/ --bind }" "$CONTAINER" \
  bash -c "$controller_cmd"
controller_exit=$?
