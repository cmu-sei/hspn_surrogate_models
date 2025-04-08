#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

PROCESSED_DATA_PATH=./don_dataset.h5
PREPARE_OPTS="output_path=$PROCESSED_DATA_PATH force=true"
TRAIN_OPTS="dataloader.dataset.file_path=$PROCESSED_DATA_PATH"

# Show config
HELP_FLAGS="--cfg=job --resolve"
make prepare DATA_DIR="$DATA_DIR" OPTS="$PREPARE_OPTS $HELP_FLAGS"
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
make prepare DATA_DIR="$DATA_DIR" OPTS="$PREPARE_OPTS"
make train OPTS="$TRAIN_OPTS"
