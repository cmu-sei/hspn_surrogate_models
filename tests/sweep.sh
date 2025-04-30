#!/bin/bash
#
# HyperSPIN code - hspn_surrogate_models
# 
# Copyright 2025 Carnegie Mellon University.
# 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# 
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
# 
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM25-0396
#

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
