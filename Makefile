.PHONY: default

default:
	docker compose run --build -it --rm -v $(shell pwd):/workspace --user $(shell id -u):$(shell id -g) -e USER=$(USER) dev
 
