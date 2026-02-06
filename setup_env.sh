#!/bin/bash
cd $(dirname $0)
cd verl
pip install -e .
export CUDA_HOME="/opt/modules/nvidia-cuda-12.8.0"
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh 
pip install numpy==2.2.0 transformers==4.57.6
cd ../ivon
pip install -e .
cd ../