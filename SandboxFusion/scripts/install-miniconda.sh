#!/bin/bash
# =============================================================================
# install-miniconda.sh -- Download and install Miniconda for a given Python version
# =============================================================================
# Usage: bash install-miniconda.sh <python_version>
#   e.g. bash install-miniconda.sh 3.13
#
# Downloads the Miniconda installer from repo.anaconda.com for the specified
# Python version and installs it non-interactively to /root/miniconda3.
# =============================================================================
set -o errexit

# Build the installer filename from the Python version argument ($1).
# The version dots are stripped (e.g. 3.11 -> 311) for the filename convention.
py_ver="$1"
FILENAME=Miniconda3-py${py_ver//.}_26.1.1-1-Linux-x86_64.sh

MIRROR="https://repo.anaconda.com/miniconda"

# Download, install in batch mode (-b), and clean up the installer.
wget ${MIRROR}/${FILENAME}
mkdir -p /root/.conda
bash ${FILENAME} -b
rm -f ${FILENAME}

# Accept Anaconda Terms of Service (required since Miniconda 26.x)
/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Clean conda package cache to reduce image size
/root/miniconda3/bin/conda clean -afy
rm -rf /root/miniconda3/pkgs/*
