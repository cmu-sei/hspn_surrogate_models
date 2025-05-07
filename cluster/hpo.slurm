#!/bin/zsh -l
#SBATCH --job-name=hspn-hpo-ray
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    # each worker will get the whole node
#SBATCH --mem=0                # each worker will get the whole node
#SBATCH --cpus-per-task=1     # each worker will get the whole node
#SBATCH --gpus-per-task=1     # each worker will get the whole node
#SBATCH --exclusive            # each worker will get the whole node
#SBATCH --time=00:30:00
#SBATCH --constraint=mla
#SBATCH --qos=frontier
#SBATCH --output=%x-%j-step-%s-node_%N_task_%t.out

if ! command -v apptainer > /dev/null 2>&1; then
  if module load apptainer > /dev/null 2>&1; then
    :
  else
    echo "Could not load apptainer module. Attempting to load singularity..."
    if module load singularity > /dev/null 2>&1; then
      alias apptainer=singularity
      echo "Loaded singularity"
    else
      echo "Failed to load apptainer and singularity, cannot proceed"
      exit 1
    fi
  fi
fi
set -x
project_root="$SLURM_SUBMIT_DIR"
cd $project_root
rm -rf tmp-ray
mkdir -p tmp-ray
set +x

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "HOSTNAME: $(hostname)"

all_nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "All nodes: ${all_nodes[@]}"

head_node=$(echo $all_nodes | head -n1)
echo "Head node: ${head_node}"

worker_nodes=($(echo $all_nodes | grep -v $head_node))
echo "Worker nodes: ${worker_nodes[@]}"

# Query total available resources on the node, all nodes are the same (assuming the sbatch constraints are properly managed according to your environment)
NUM_CPUS=$(lscpu | awk '/^CPU\(s\):/ { print $2 }')
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Override the SLURM values
export SLURM_CPUS_PER_TASK=$NUM_CPUS
export SLURM_GPUS_PER_TASK=$NUM_GPUS
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_TASK=$SLURM_GPUS_PER_TASK"

port=6379
export RAY_ADDRESS="${head_node}:${port}"
export APPTAINERENV_RAY_ADDRESS=$RAY_ADDRESS
export SINGULARITYENV_RAY_ADDRESS=$RAY_ADDRESS

echo "Starting Ray HEAD @ ${RAY_ADDRESS}..."
set -x
srun --nodes=1 --ntasks=1 -w "$head_node" --exclusive \
  apptainer exec --nv --bind $project_root --bind $project_root/tmp-ray:/tmp/ray \
    hspn.sif \
    ray start --head --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" \
    --block &
sleep 10
set +x

for node in "${worker_nodes[@]}"; do
  echo "Starting Ray WORKER @ $node"
  set -x
  srun --nodes=1 --ntasks=1 -w "$node" --exclusive \
    apptainer exec --nv --bind $project_root --bind $project_root/tmp-ray:/tmp/ray \
    hspn.sif \
    ray start --address "$RAY_ADDRESS" \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" \
    --block &
  sleep 5
  set +x
done

sleep 10

echo "Ray cluster status:"
apptainer exec --bind $project_root --bind $project_root/tmp-ray:/tmp/ray \
    hspn.sif ray status

TOTAL_NODES=$SLURM_JOB_NUM_NODES
TOTAL_CPUS=$((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_TASK))
N_JOBS=$TOTAL_GPUS  # One job per GPU

echo "Total nodes: $TOTAL_NODES"
echo "Total CPUs available: $TOTAL_CPUS"
echo "Total GPUs available: $TOTAL_GPUS"
echo "Number of parallel jobs: $N_JOBS"

echo "Launching controller..."
set -x
apptainer exec --bind $project_root --bind $project_root/tmp-ray:/tmp/ray \
  hspn.sif \
  python -u -m hspn.train --multirun --config-name=train_hpo_basic \
  hydra/launcher=ray \
  +hydra.launcher.ray.remote.num_gpus=1 \
  +hydra.launcher.ray.remote.num_cpus=16 \
  hydra.launcher.ray.init.address=$RAY_ADDRESS $@
set +x
echo "Done"
