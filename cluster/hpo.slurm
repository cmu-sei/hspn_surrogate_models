#!/bin/zsh -l
#SBATCH --job-name=hspn-hpo-ray
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    # each worker will get the whole node
#SBATCH --ntasks-per-node=1    # each worker will get the whole node
#SBATCH --mem=0                # each worker will get the whole node
#SBATCH --cpus-per-task=1     # each worker will get the whole node
#SBATCH --gpus-per-task=1     # each worker will get the whole node
#SBATCH --exclusive            # each worker will get the whole node
#SBATCH --time=01:00:00
#SBATCH --constraint=mla
#SBATCH --qos=frontier
#SBATCH --output=%x-%j-step-%s-node_%N_task_%t.out
#
##SBATCH --ntasks=2
##SBATCH --qos=debug
##SBATCH --mem=8G
##SBATCH --time=00:05:00

echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

set -uo pipefail

# Configurable -----------------------------------------------------------------
RAY_PORT=${RAY_PORT:-6400}
CONFIG_NAME=${CONFIG_NAME:-train_hpo_optuna}
GPUS_PER_TRIAL=${GPUS_PER_TRIAL:-1}

# Env Detect -------------------------------------------------------------------
# Query total available resources on the node, all nodes are the same (assuming
#   the sbatch constraints are properly managed according to your environment)
NUM_CPUS=$(lscpu | awk '/^CPU\(s\):/ { print $2 }')
NUM_GPUS=$(nvidia-smi -L | wc -l)
# Override the SLURM values
export SLURM_CPUS_PER_TASK=$NUM_CPUS
export SLURM_GPUS_PER_TASK=$NUM_GPUS
# Derive Resources -------------------------------------------------------------
# Defaults assume CPU-only single node job
TOTAL_NODES=${SLURM_JOB_NUM_NODES:-1}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
SLURM_GPUS_PER_TASK=${SLURM_GPUS_PER_TASK:-0}
TOTAL_CPUS=$((TOTAL_NODES * SLURM_CPUS_PER_TASK))
TOTAL_GPUS=$((TOTAL_NODES * SLURM_GPUS_PER_TASK))
if (( TOTAL_GPUS == 0 )); then
    N_WORKERS=1
    NV_FLAG=""
else
    NV_FLAG="--nv"
    N_WORKERS=$((TOTAL_GPUS / GPUS_PER_TRIAL))
fi
# Divide CPUs up CPUS_PER_TRIAL, some room for cluster overhead.
AVAILABLE_CPUS=$((TOTAL_CPUS - 1))
CPUS_PER_TRIAL=$((AVAILABLE_CPUS / N_WORKERS))
printf '%*s\n' 80 | tr ' ' "-"
echo "Options:"
echo "\tTOTAL_NODES=$TOTAL_NODES"
echo "\tTOTAL_CPUS=$TOTAL_CPUS"
echo "\tTOTAL_GPUS=$TOTAL_GPUS"
echo "\tGPUS_PER_TRIAL=$GPUS_PER_TRIAL"
echo "\tN_WORKERS=$N_WORKERS"
echo "\tCPUS_PER_TRIAL=$CPUS_PER_TRIAL"
echo "\tRAY_PORT=$RAY_PORT"
printf '%*s\n' 80 | tr ' ' "-"
# Exports
export N_WORKERS # used by some HPO sweep configs, like optuna. the head node is also a worker
export N_TRIALS=${N_TRIALS:-100} # used by some HPO sweep configs, like optuna to set max number of trials
export SINGULARITYENV_N_TRIALS=$N_TRIALS
export SINGULARITYENV_WORKERS=$N_WORKERS
# ------------------------------------------------------------------------------

if ! command -v apptainer > /dev/null 2>&1; then
  if module load apptainer > /dev/null 2>&1; then
    echo "Loaded apptainer $(apptainer --version)"
  else
    echo "Could not load apptainer module. Attempting to load singularity..."
    if module load singularity > /dev/null 2>&1; then
      alias apptainer=singularity
      echo "Loaded singularity $(singularity --version)"
    else
      echo "Failed to load apptainer and singularity, cannot proceed"
      exit 1
    fi
  fi
fi
module load nvidia

echo "$(apptainer --version)"

project_root="$SLURM_SUBMIT_DIR"
cd $project_root


echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_TASK=$SLURM_GPUS_PER_TASK"
printf '%*s\n' 80 | tr ' ' "-"
all_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo
head_name=$(echo "$all_nodes" | head -n1)
worker_nodes=$(echo "$all_nodes" | grep -v "$head_name")
printf '%*s\n' 80 | tr ' ' "="

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Self: $(hostname)"
echo "All nodes:"
echo "$all_nodes"
echo

# We need to use intranet IPs, not hostname, as this often resolves incorrectly and services fail to connect.
srun --ntasks-per-node=1 echo "$(hostname) -> $(hostname -I)"
sleep 1
head_ip_addr=$(srun --ntasks=1 --nodes=1 --nodelist="$head_name" apptainer exec hspn.sif bash -c "ip -j -4 addr show ib0 | jq -r '.[].addr_info[]?.local'")
echo "v2 head ib0 address: $head_ip_addr"
echo "Head node : $head_name : $head_ip_addr : $RAY_PORT"
[[ -z $head_ip_addr ]] && { echo "failed to get head_ip_addr=$head_ip_addr"; exit 1; };


export RAY_ADDRESS="${head_ip_addr}:${RAY_PORT}"
export APPTAINERENV_RAY_ADDRESS=$RAY_ADDRESS
export SINGULARITYENV_RAY_ADDRESS=$RAY_ADDRESS

echo "Starting Ray HEAD @ ${RAY_ADDRESS}..."
set -x
srun --nodes=1 --ntasks=1 -w "$head_name" --exclusive \
  apptainer exec "$NV_FLAG" --bind $project_root \
    hspn.sif \
    ray start --head --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" \
    --node-ip-address="$head_ip_addr" --port "$RAY_PORT" --block &
sleep 5
set +x

printf '%*s\n' 80 | tr ' ' "-"
for node in "${worker_nodes[@]}"; do
  if [[ -n "$node" ]]; then
    echo "Starting Ray WORKER @ $node"
    set -x
    srun --nodes=1 --ntasks=1 -w "$node" --exclusive \
      apptainer exec "$NV_FLAG" --bind $project_root \
      hspn.sif \
      ray start --address "$RAY_ADDRESS" \
      --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" \
      --block &
    sleep 5
    set +x
  fi
done
printf '%*s\n' 80 | tr ' ' "-"

sleep 5

echo "Ray cluster status:"
apptainer exec --bind $project_root \
    hspn.sif ray status

echo "Starting Redis server for Optuna store..."
optuna_db_host="$(hostname)"
OPTUNA_DB_PORT="${OPTUNA_DB_PORT:-6379}"
export OPTUNA_STORAGE_URL="redis://$optuna_db_host:$OPTUNA_DB_PORT/"
redis_dir="${project_root}/.redis"
echo "Creating $redis_dir"
mkdir -p "$redis_dir"
REDIS_PID=""

cleanup() {
    if [[ -n "$REDIS_PID" ]]; then
        echo "Flushing Redis data for Optuna before shutdown..."
        apptainer exec --bind "$redis_dir" hspn.sif redis-cli -h "$optuna_db_host" -p "$OPTUNA_DB_PORT" SAVE
        kill -TERM "$REDIS_PID"
        wait "$REDIS_PID" || true
    fi
  echo "Job ended at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
}
trap cleanup TERM INT EXIT
apptainer exec --bind "$redis_dir" hspn.sif redis-server \
    --bind 0.0.0.0 --port "$OPTUNA_DB_PORT" --appendonly yes --appendfsync no --dir "$redis_dir" &
REDIS_PID=$!
sleep 2

echo "Optuna Storage Backend Configured:"
echo "   OPTUNA_STORAGE_URL=$OPTUNA_STORAGE_URL"

echo "Launching controller..."
set -x
apptainer exec --bind $project_root \
  hspn.sif \
  python -u -m hspn.train --multirun \
  --config-name=$CONFIG_NAME \
  hydra/launcher=ray \
  +hydra.launcher.ray.remote.num_gpus=${GPUS_PER_TRIAL} \
  +hydra.launcher.ray.remote.num_cpus=${CPUS_PER_TRIAL} \
  hydra.launcher.ray.init.address=$RAY_ADDRESS "$@"
set +x
