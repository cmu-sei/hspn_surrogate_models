#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

PROCESSED_DATA_PATH=./don_dataset.h5
#TRAIN_OPTS="hydra/sweeper=optuna hydra/sweeper=grid dataloader.dataset.file_path=$PROCESSED_DATA_PATH"
TRAIN_OPTS="--config-name=train_sweep dataloader.dataset.file_path=$PROCESSED_DATA_PATH"

# Show config
HELP_FLAGS="--cfg=job --resolve"
make train OPTS="$TRAIN_OPTS $HELP_FLAGS"

# If running in a TTY, ask for confirmation with a 5s timeout for cancelling (continue by default).
if [[ -t 0 ]]; then
  echo "Proceed? (Y/n) (default: Y in 5s)"
  read -r -t 5 response || response="y"
  if [[ "$response" =~ ^[Nn]$ ]]; then
    echo "Aborting."
    exit 1
  fi
fi

# Run
make train OPTS="$TRAIN_OPTS -m"
