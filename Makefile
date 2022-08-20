PROJECT = climate-coop/cpu
#VERSION ?= $(shell git describe --abbrev=0 --tags)
VERSION ?= latest
IMAGE ?= $(PROJECT):$(VERSION)
DOCKER_VARS = --user $(shell id -u):$(shell id -g) --group-add users -e GRANT_SUDO=yes

all: build run

build:
	docker build -t $(IMAGE) -f Dockerfile_CPU .

run:
	docker run -it --rm $(DOCKER_VARS) -p 8888:8888 -v "${PWD}":/home/jovyan/work $(IMAGE)

train:
	docker run -it --rm $(DOCKER_VARS) -v "${PWD}":/home/jovyan/work $(IMAGE) python scripts/train_with_rllib.py

evaluate: SUB=$(shell ls -r Submissions/*.zip | tr ' ' '\n' | head -1)
evaluate:
	docker run -it --rm $(DOCKER_VARS) -v "${PWD}":/home/jovyan/work $(IMAGE) python scripts/evaluate_submission.py -r $(SUB)

bash:
	docker run -it --rm $(DOCKER_VARS) -v "${PWD}":/home/jovyan/work $(IMAGE) bash

