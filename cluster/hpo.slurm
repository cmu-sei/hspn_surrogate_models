#!/bin/zsh -l
#SBATCH --job-name=hspn-hpo-ray
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1    # each worker will get the whole node
#SBATCH --mem=0                # each worker will get the whole node
#SBATCH --cpus-per-task=1     # each worker will get the whole node
#SBATCH --gpus-per-task=1     # each worker will get the whole node
#SBATCH --exclusive            # each worker will get the whole node
#SBATCH --time=00:30:00
#SBATCH --constraint=mla
#SBATCH --qos=frontier
#SBATCH --output=ray-hydra-%j.out

if ! command -v apptainer &> /dev/null; then
    module load apptainer
fi

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "HOSTNAME: $(hostname)"

# Get the head node (current node)
head_node=$(hostname)
# head_node_ip=$(hostname -i)
echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

# Query total available resources on the node
NUM_CPUS=$(lscpu | awk '/^CPU\(s\):/ { print $2 }')
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Override the SLURM values
export SLURM_CPUS_PER_TASK=$NUM_CPUS
export SLURM_GPUS_PER_TASK=$NUM_GPUS
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_TASK=$SLURM_GPUS_PER_TASK"

port=6379
# ip_head="${head_node_ip}:${port}"
ip_head="${head_node}:${port}"
export ip_head
echo "IP Head: $ip_head"
export RAY_ADDRESS=$ip_head

echo "Starting Ray HEAD at $head_node"
apptainer exec --nv hspn.sif ray start --head \
    --num-cpus "${SLURM_CPUS_PER_TASK}" \
    --num-gpus "${SLURM_GPUS_PER_TASK}" \
    --block &

sleep 10

all_nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "All nodes: ${all_nodes[@]}"
all_nodes=($(echo $all_nodes | grep -v $head_node))
echo "Worker nodes: ${all_nodes[@]}"

for node in "${all_nodes[@]}"; do
      echo "Starting Ray WORKER at $node"
      srun --nodes=1 --ntasks=1 -w "$node" --exclusive \
          apptainer exec --nv hspn.sif ray start \
          --address "$ip_head" \
          --num-cpus "${SLURM_CPUS_PER_TASK}" \
          --num-gpus "${SLURM_GPUS_PER_TASK}" \
          --block &
      sleep 5
done

sleep 15

echo "Ray cluster status:"
apptainer exec --nv hspn.sif ray status

TOTAL_NODES=$SLURM_JOB_NUM_NODES
TOTAL_CPUS=$((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_TASK))
N_JOBS=$TOTAL_GPUS  # One job per GPU

echo "Total nodes: $TOTAL_NODES"
echo "Total CPUs available: $TOTAL_CPUS"
echo "Total GPUs available: $TOTAL_GPUS"
echo "Number of parallel jobs: $N_JOBS"

apptainer exec --nv hspn.sif python -u -m hspn.train \
    --config-name=train_hpo_basic \
    --multirun \
    hydra/launcher=ray \
    hydra.launcher.ray.init.address=$ip_head
