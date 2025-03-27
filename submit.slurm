#!/bin/bash
## USAGE
##
## 1. Run a hspn module with `sbatch submit.slurm module_name [...]` which runs `python -m hspn.module_name [...]` inside the container
##    Submit a training job: `sbatch submit.slurm train seed=42 epochs=100`
##
##    Submit a data preprocessing job `sbatch submit.slurm prepare data_dir=/srv/hspn_data/raw_data output_path=./don_dataset.h5`
##
## 2. Open an interactive shell inside the container:
##    sbatch --interactive submit.slurm shell
##
## ENVIRONMENT VARIABLES
##
## - CONTAINER: Path to the Singularity container
##   Example: CONTAINER=/path/to/custom_container.sif sbatch submit.slurm train
##
## TIPS
##
## - For memory-intensive workloads, increase --mem=4G
## - For multi-GPU training, adjust --gres=gpu:a100:4 or adjust based on availability
## - To customize CPU core allocation, use the --cpus-per-task directive
## - If your job fails check slurm-[JOBID].out and slurm-[JOBID].err
## - You can use "sacct -j [JOBID] --format=JobID,State,ExitCode" to check job status

#SBATCH --account=ARLAP44862YFR
#SBATCH --job-name=hspn
#SBATCH --partition=general

#SBATCH --time=0:30:00
#SBATCH --qos=debug
##SBATCH --qos=hie
##SBATCH --qos=frontier

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
##SBATCH --cpus-per-task=4

## AI/ML nodes are GPU nodes with specialized hardware
## and software to support machine-learning tasks.
## To request an AI/ML node, use the directive:
#SBATCH --constraint=mla

## Request GPU nodes - select one of the options below
##SBATCH --gres=gpu:a100:2  # Same as mla
##SBATCH --gres=gpu:2
##SBATCH --gres=gpu:4

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
echo "----------------------------------------------------------"
echo "Type of node                    " $BC_NODE_TYPE
echo "CPU cores per node              " $BC_CORES_PER_NODE
echo "CPU cores per standard node     " $BC_STANDARD_NODE_CORES
echo "CPU cores per accelerator node  " $BC_ACCELERATOR_NODE_CORES
echo "CPU cores per big memory node   " $BC_BIGMEM_NODE_CORES
echo "Hostname                        " $BC_HOST
echo "Maximum memory per nodes        " $BC_MEM_PER_NODE
echo "Number of tasks allocated       " $BC_MPI_TASKS_ALLOC
echo "Number of nodes allocated       " $BC_NODE_ALLOC
echo "----------------------------------------------------------"

echo "-------------------------------------------------------"
echo "Project ID                      " $SLURM_JOB_ACCOUNT
echo "Job submission directory        " $SLURM_SUBMIT_DIR
echo "Submit host                     " $SLURM_SUBMIT_HOST
echo "Job name                        " $SLURM_JOB_NAME
echo "Job identifier (SLURM_JOB_ID)   " $SLURM_JOB_ID
echo "Job identifier (SLURM_JOBID)    " $SLURM_JOBID
echo "Working directory               " $WORKDIR
echo "Job partition                   " $SLURM_JOB_PARTITION
echo "Job queue (QOS)                 " $SLURM_JOB_QOS
echo "Job number of nodes             " $SLURM_JOB_NUM_NODES
echo "Job node list                   " $SLURM_JOB_NODELIST
echo "Number of nodes                 " $SLURM_NNODES
echo "Number of tasks                 " $SLURM_NTASKS
echo "Node list                       " $SLURM_NODELIST
echo "-------------------------------------------------------"
echo

module load slurm
module load penguin/openmpi/4.1.4/gcc
module load nvidia/openmpi/gnu/4.1
module load cuda/cuda-12.4
srun nvidia-smi # Check for GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TEMP_SCRIPT=""
WORKDIR=$(pwd)

cleanup() {
  if [ -n "$TEMP_SCRIPT" ] && [ -f "$TEMP_SCRIPT" ]; then
    rm -f "$TEMP_SCRIPT"
    echo "Removed temporary script: $TEMP_SCRIPT"
  fi
  echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

CONTAINER=${CONTAINER:-"/p/app/containers/pytorch/pytorch-22.03-py3.sif"}
if [ ! -f "$CONTAINER" ]; then
  echo "Error: Container file not found: $CONTAINER"
  exit 1
fi

VOLUMES="${WORKDIR}:/workspace"

if [ "$1" = "shell" ]; then
  cmd="srun --pty singularity shell --bind ${VOLUMES//,/ --bind } --nv $CONTAINER"
  echo "Executing: $cmd"
  eval $cmd
  exit $?
fi

pymod=$1
shift
jobcmd="python -m hspn.$pymod $@"
echo "Jobcmd: \`$jobcmd\`"

TEMP_SCRIPT=$(mktemp)
chmod +x $TEMP_SCRIPT
cat >$TEMP_SCRIPT <<EOF
#!/bin/bash
set -eu
cd /workspace
pip install -e . || { echo "Failed to install package"; exit 1; }
$jobcmd || { echo "command failed \\\`$jobcmd\\\`"; exit 1; }
EOF

cmd="srun --ntasks=$SLURM_NTASKS singularity exec --bind ${VOLUMES//,/ --bind } --nv $CONTAINER $TEMP_SCRIPT"
echo "Executing: $cmd"
eval $cmd
exit_code=$?

if [ $exit_code -ne 0 ]; then
  echo "Command failed with exit code $exit_code"
  exit $exit_code
fi

echo "Job completed ($1)"

