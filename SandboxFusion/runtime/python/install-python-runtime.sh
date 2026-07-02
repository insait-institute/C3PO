#!/bin/bash
# ============================================================================
# install-python-runtime.sh
#
# Bootstraps the Python sandbox runtime environment used by SandboxFusion for
# executing user-submitted Python code. This script performs the following:
#
#   1. Creates a new conda environment named "sandbox-runtime" with Python 3.13.
#   2. Activates the environment and installs all packages listed in
#      requirements.txt (ignoring Python-version constraints to ensure broad
#      compatibility within the sandboxed environment).
#   3. Purges the pip download cache and conda package cache to minimise the
#      final Docker image size.
#
# This script is intended to be run once during the Docker image build.
# ============================================================================
set -o errexit

conda create -n sandbox-runtime -y python=3.13

source activate sandbox-runtime

pip install -r ./requirements.txt --ignore-requires-python

pip cache purge
conda clean --all -y
