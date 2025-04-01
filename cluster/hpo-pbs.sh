#!/bin/bash
# Launches hyperparameter optimization on PBS cluster
set -eu

ACCT="${ACCT:-ODEFN50352CMU}"
QUEUE="${QUEUE:-mla}"
WALLTIME="${WALLTIME:-00:20:00}"
N_WORKERS=${1:-2}
N_TRIALS=${2:-2}
TRAIN_OPTS="${TRAIN_OPTS:-}"
pbs_opts="-A $ACCT -q $QUEUE -l walltime=$WALLTIME"

worker_pbs="cluster/hpo-pbs/rq_worker.pbs"
server_pbs="cluster/hpo-pbs/rq_server.pbs"
controller_pbs="cluster/hpo-pbs/controller.pbs"
cleanup_pbs="cluster/hpo-pbs/cleanup.pbs"

assert_exists() {
  if [[ ! -f "$1" ]]; then
    echo "Error: required file '$1' does not exist. PWD=$PWD " >&2
    exit 1
  fi
}

assert_exists "$worker_pbs"
assert_exists "$server_pbs"
assert_exists "$controller_pbs"
assert_exists "$cleanup_pbs"

echo "PBS opts: $pbs_opts"
echo "Train opts: $TRAIN_OPTS"
echo "N_WORKERS=$N_WORKERS"

worker_job_id=""
server_job_id=""
controller_job_id=""

at_exit() {
  set +e
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    echo "ERROR - cleaning up"
    if [ -n "$controller_job_id" ]; then
      echo "Stopping controller $controller_job_id"
      qdel $controller_job_id
    fi
    if [ -n "$server_job_id" ]; then
      echo "Stopping server $server_job_id"
      qdel $server_job_id
    fi
    if [ -n "$worker_job_id" ]; then
      echo "Stopping worker $worker_job_id"
      qdel $worker_job_id
    fi
  else
    echo "Stop jobs with:"
    if [[ -n $controller_job_id ]]; then
      echo -n "qdel \"$controller_job_id\";"
    fi
    if [[ -n $server_job_id ]]; then
      echo -n "qdel \"$server_job_id\";"
    fi
    if [[ -n $worker_job_id ]]; then
      echo -n "qdel \"$worker_job_id\";"
    fi
    echo ""
  fi
}
trap at_exit EXIT

submit() {
  local name=$1
  local prefix="Submit : $name :"
  shift
  local cmd="qsub "$@""
  local job_id
  echo "$prefix $cmd" >&2
  if ! job_id=$($cmd 2>&1); then
    echo "$prefix ERROR Failed to submit $name job. qsub output:" >&2
    echo "$job_id" >&2
    exit 1
  fi
  echo "$prefix $job_id" >&2
  echo "$job_id"
}

# Start workers first bc they ask for the most resources
worker_job_id=$(submit "Workers" $pbs_opts -J 1-$N_WORKERS "$worker_pbs")

# Start queue server after workers
after_workers="-W depend=after:$worker_job_id"
server_job_id=$(submit "Server" $pbs_opts $after_workers "$server_pbs")

# Start controller after server (starts last)
after_server="-W depend=after:$server_job_id"
controller_job_id=$(submit "Controller" $pbs_opts $after_server -v "N_WORKERS=$N_WORKERS,N_TRIALS=$N_TRIALS,OPTS=$TRAIN_OPTS" "$controller_pbs")

# Cleanup - shutdown workers and server after controller exits
after_controller="-W depend=afterok:$controller_job_id"
shutdown_ids="${worker_job_id},${server_job_id}"
cleanup_job_id=$(submit "Cleanup" $pbs_opts -W depend=afterok:$controller_job_id -v "SHUTDOWN_JOB_IDS=$shutdown_ids" "$cleanup_pbs")
