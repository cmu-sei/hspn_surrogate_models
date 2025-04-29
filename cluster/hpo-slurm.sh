#!/bin/bash
# @describe Launch hyperparameter optimization on a SLURM cluster

# PBS options
# @option -A --account! $ACCT
## @option -q --qos=standard $$                  SLURM Queue/QOS
## @option -p --partition=mla $$               SLURM partition (optional)
# @option -w --walltime=00:20:00 $$           Walltime

# Training + HPO config
# @option -n --n-workers=2 $$                           Number of workers
# @option -t --n-trials=2 $$                            Number of trials
# @option -o --train-opts=--config-name=train_hpo_optuna $$ Arguments passed to train cli

eval "$(argc --argc-eval "$0" "$@")"
export ACCOUNT="$argc_account"
#export QUEUE="$argc_qos"
export WALLTIME="$argc_walltime"
export N_WORKERS="$argc_n_workers"
export N_TRIALS="$argc_n_trials"
export TRAIN_OPTS="$argc_train_opts"

export CPUS_PER_TASK=$((4 * N_WORKERS + 2)) # Total CPUs: 4 per worker plus 2 extra for controller/queue.
export MEM=$((16 * N_WORKERS + 8))G         # Total memory: 16GB per worker plus 8GB extra.

env | grep -E "(ACCT|QUEUE|WORK|TRIAL|OPT|CPUS_PER|MEM|argc_.*)"

cmd="sbatch"
#cmd+=" --constraint=mla"
cmd+=" --account=$ACCOUNT"
#cmd+=" -q=$QUEUE"
cmd+=" --time=$WALLTIME"
cmd+=" --cpus-per-task=$((4 * N_WORKERS + 2))"
# cmd+=" --mem=4G"
cmd+=" --mem=$MEM"
#cmd+=" --cpus-per-task=$((4*N_WORKERS+2))"  # Total CPUs: 4 per worker plus 2 extra for controller/queue. \
#cmd+=" --mem=$((16*N_WORKERS+8))G"          # Total memory: 16GB per worker plus 8GB extra.
#cmd+=" --gres=gpu:$N_WORKERS"
cmd+=" --gres=gpu:a100:$N_WORKERS"
cmd+=" cluster/hpo-slurm/hpo.slurm"

echo "Submitting: $cmd"
eval $cmd
