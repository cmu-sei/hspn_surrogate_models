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

.PHONY: default prepare train

default:
	docker compose run --build -it --rm -v $(shell pwd):/workspace --user $(shell id -u):$(shell id -g) -e USER=$(USER) dev
 

# Usage: make prepare DATA_DIR=/path/to/input/data/dir/  [OPTS="foo=false"]
# OPTS: For additional options the user may want to add to the end of the command (e.g. OPTS="output_path=./don_dataset.h5 force=true")
prepare:
	@if [ -z "$(DATA_DIR)" ]; then echo "Error: DATA_DIR is not set. Usage: make prepare DATA_DIR=/path/to/data [OPTS=\"extra args\"]"; exit 1; fi
	hspn-preprocess format=HDF5 data_dir=$(DATA_DIR) \
        branch_files=[aoa_total.npy] trunk_files=[xyz.npy] \
        output_files=[data_total.npy] \
        branch_normalization.method=MINMAX \
        trunk_normalization.method=MINMAX trunk_normalization.axis=0 \
        output_normalization.method=MINMAX $(OPTS)

aim:
	apptainer exec hspn.sif aim up

aim-reindex:
	apptainer exec hspn.sif aim storage prune && aim storage reindex

# Usage: make train [OPTS="foo=false"]
# OPTS: For additional options the user may want to add to the end of the command (e.g. OPTS="seed=55, extra.data_dir=./data --cfg=job")
train:
	hspn-train $(OPTS)

hspn.sif:
	singularity build --sandbox --fakeroot --bind $(shell pwd):/workspace hspn.sif cluster/hspn.def
	singularity build --fakeroot hspn.sif hspn.sif/


OPENSSL_VERSION := 3.4.1
OPENSSL_TARBALL := openssl-$(OPENSSL_VERSION).tar.gz
OPENSSL_DIR := openssl-$(OPENSSL_VERSION)

# Get OpenSSL since we need to compile it during container build
$(OPENSSL_DIR): $(OPENSSL_TARBALL)
	tar -xf $(OPENSSL_TARBALL)

$(OPENSSL_TARBALL):
	@echo "Downloading OpenSSL $(OPENSSL_VERSION)"
	wget https://github.com/openssl/openssl/releases/download/openssl-$(OPENSSL_VERSION)/$(OPENSSL_TARBALL) -O $(OPENSSL_TARBALL)

# Build the base image
pytorch-2503.sif: $(OPENSSL_DIR)
	singularity build --fakeroot --bind $(shell realpath $(OPENSSL_DIR)):/tmp/$(OPENSSL_DIR) pytorch-2503.sif cluster/pytorch-2503.def

# Build
hspn-2503.sif: pytorch-2503.sif
	singularity build --fakeroot --bind $(shell pwd):/workspace hspn-2503.sif cluster/hspn-2503.def

# Staged Variant - WIP
# Build the base image - staged
pytorch-2503-staged.sif: $(OPENSSL_DIR)
	singularity build --fakeroot --bind $(shell realpath $(OPENSSL_DIR)):/tmp/$(OPENSSL_DIR) pytorch-2503-staged.sif cluster/pytorch-2503-staged.def

# Staged build
hspn-2503-staged.sif: pytorch-2503-staged.sif
	singularity build --fakeroot --bind $(shell pwd):/workspace hspn-2503-staged.sif cluster/hspn-2503.def

