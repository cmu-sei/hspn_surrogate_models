.PHONY: default

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

# Usage: make train [OPTS="foo=false"]
# OPTS: For additional options the user may want to add to the end of the command (e.g. OPTS="seed=55, extra.data_dir=./data --cfg=job")
train:
	hspn-train $(OPTS)
