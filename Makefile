.PHONY: default

default:
	docker compose run --build -it --rm -v $(shell pwd):/workspace --user $(shell id -u):$(shell id -g) -e USER=$(USER) dev
 

# Usage: make prepare DATA_DIR=/path/to/input/data/dir/  [OPTS="foo=false"]
# OPTS: For additional options the user may want to add to the end of the command (e.g. force=true)
prepare:
	@ifndef DATA_DIR; echo "Error: DATA_DIR is not set. Usage: make prepare DATA_DIR=/path/to/data [OPTS=\"extra args\"]"; exit 1; fi
    OPTS ?=
	hspn-preprocess format=HDF5 data_dir=$(DATA_DIR) \
        branch_files=[aoa_total.npy] trunk_files=[xyz.npy] \
        output_files=[data_total.npy] \
        branch_normalization.method=MINMAX \
        trunk_normalization.method=MINMAX trunk_normalization.axis=0 \
        output_normalization.method=MINMAX $${OPTS}