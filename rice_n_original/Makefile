PROJECT = climate-coop/cpu
#VERSION ?= $(shell git describe --abbrev=0 --tags)
VERSION ?= latest
IMAGE ?= $(PROJECT):$(VERSION)
DOCKER_VARS = --user $(shell id -u):$(shell id -g) --group-add users -e GRANT_SUDO=yes
MOUNT_VARS = -v "${PWD}":/home/jovyan/work -v ${PWD}/ray_results:/home/jovyan/ray_results
X11_VARS = -e "DISPLAY" -v "/etc/group:/etc/group:ro" -v "/etc/passwd:/etc/passwd:ro" -v "/etc/shadow:/etc/shadow:ro" -v "/etc/sudoers.d:/etc/sudoers.d:ro" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
ALL_VARS = $(DOCKER_VARS) $(MOUNT_VARS) $(X11_VARS)

PYTHONPATH ?= -e PYTHONPATH=scripts

all: build run

build:
	docker build -t $(IMAGE) -f Dockerfile_CPU .

run:
	docker run -it --rm $(ALL_VARS) -p 8888:8888 $(IMAGE)

train:
	mkdir -p ray_results > /dev/null
	docker run -it --rm $(ALL_VARS) $(IMAGE) python scripts/train_with_rllib.py

evaluate: SUB=$(shell ls -r Submissions/*.zip | tr ' ' '\n' | head -1)
evaluate:
	mkdir -p .tmp/_base
	docker run -it --rm $(ALL_VARS) $(IMAGE) python scripts/evaluate_submission.py -r $(SUB)

bash:
	docker run -it --rm $(ALL_VARS) $(IMAGE) bash

python:
	docker run -it --rm $(ALL_VARS) $(PYTHONPATH) $(IMAGE) python

# View on http://localhost:6006
tensorboard:
	docker run -it --rm $(ALL_VARS) -p 6006:6006 $(IMAGE) tensorboard --logdir '~/work/ray_results'

diagnose: LOG=diagnostic.log
diagnose:
	@echo "Build clean image (no cache)..." | tee $(LOG)
	docker build --no-cache -t $(IMAGE) -f Dockerfile_CPU . | tee -a $(LOG)
	@echo "Using Docker: $(shell docker --version)" | tee -a $(LOG)
	@echo "Working dir: $(shell pwd)" | tee -a $(LOG)
	@echo "User: $(shell id -un)" | tee -a $(LOG)
	@echo "In docker group: $(shell id -Gn | tr ' ' '\n' | grep docker)" | tee -a $(LOG)
	@echo "Inspect Docker container..." | tee -a $(LOG)
	docker run --rm $(ALL_VARS) $(IMAGE) whoami | tee -a $(LOG)
	docker run --rm $(ALL_VARS) $(IMAGE) ls -l | tee -a $(LOG)

