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
	apptainer build --fakeroot hspn.sif cluster/hspn.def
	# In case of issue, please see the documentation on building in hspn.def
	# If you do not have apptainer, singularity is a drop in replacement.
	# Standard build:
	# apptainer build --fakeroot --bind $(shell pwd):/workspace hspn.sif cluster/hspn.def

	# In case of memory issues use two-step build process:
	# apptainer build --fakeroot --bind "$(shell pwd):/workspace" --sandbox hspn.sif/ cluster/hspn.def
	# apptainer build --fakeroot hspn.sif hspn.sif/
	#
	# For a peristent sandbox cache, name it something else:
	# apptainer build --fakeroot --bind "$(shell pwd):/workspace" --sandbox hspn.sandbox/ cluster/hspn.def
	# apptainer build --fakeroot hspn.sif hspn.sandbox/

apptainer-shell:
	apptainer shell --no-home --bind $(shell pwd) --pwd $(shell pwd) hspn.sif

