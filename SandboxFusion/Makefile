HOST ?= 0.0.0.0
PORT ?= 8080
TEST_NP ?= 16
IMAGE_TAG ?= 25042026-2
BACKEND ?= docker
MODE ?= full
SANDBOX_LOG_LEVEL ?= OFF
run:
	uvicorn sandbox.server.server:app --reload --host $(HOST) --port $(PORT)

run-online:
	uvicorn sandbox.server.server:app --host $(HOST) --port $(PORT)

install-runtimes:
	cd runtime/python && bash install-python-runtime.sh
	cd runtime/node && npm ci
	cd runtime/go && go build
	cd runtime/lean && lake build

build-base-image:
	docker build . -f scripts/Dockerfile.base -t ineil77/sandbox-fusion-base:$(IMAGE_TAG)

build-server-image:
	docker build . -f scripts/Dockerfile.server -t ineil77/sandbox-fusion-server:$(IMAGE_TAG)

pull-apptainer-images:
	apptainer pull docker://ineil77/sandbox-fusion-base:$(IMAGE_TAG)
	apptainer pull docker://ineil77/sandbox-fusion-server:$(IMAGE_TAG)

# Launch a standalone apptainer server in bindroot mode for interactive use.
# Binds the host sandbox/ directory over the SIF's copy so the latest local
# code is picked up without rebuilding the image.  Drop the -B once the SIF
# has been rebuilt with the new sources.
start-apptainer-container:
	apptainer run --cleanenv --fakeroot --ignore-fakeroot-command --no-home \
		--env PORT=$(PORT) \
		--env SANDBOX_CONFIG=docker_bindroot \
		--env SANDBOX_LOG_LEVEL=OFF \
		--env SANDBOX_MAX_CONCURRENCY=$(SANDBOX_MAX_CONCURRENCY) \
		-B $(CURDIR)/sandbox:/root/sandbox/sandbox \
		"$$WORK/sandbox-fusion-server_$(IMAGE_TAG).sif"

test: test-docker-full

test-docker-full:
	pytest -m "not datalake and not stress" -n $(TEST_NP) --sandbox-backend docker --sandbox-mode full

test-docker-lite:
	pytest -m "not datalake and not stress" -n $(TEST_NP) --sandbox-backend docker --sandbox-mode lite

test-apptainer-bindroot:
	pytest -m "not datalake and not stress" -n $(TEST_NP) --sandbox-backend apptainer --sandbox-mode bindroot

# Replays an RL reward loop against the server (48 concurrent clients, mixed
# correct/TLE/poison-stdin solutions).  Runs sequentially: the tests manage
# their own concurrency.
test-rl-load:
	pytest -s -m stress --sandbox-backend apptainer --sandbox-mode bindroot sandbox/tests/test_rl_load.py

test-case:
	pytest -s -vv -k $(CASE) --sandbox-backend $(BACKEND) --sandbox-mode $(MODE)

format:
	pycln --config pyproject.toml
	isort sandbox/*
	yapf -ir sandbox/*

format-client:
	mv scripts/client/pyproject.toml scripts/faas/pyproject.toml && yapf -ir scripts/client/* && mv scripts/faas/pyproject.toml scripts/client/pyproject.toml

# mypy --explicit-package-bases sandbox
check:
	pycln --config pyproject.toml --check
	yapf --diff --recursive sandbox/*
	make test-docker-full
