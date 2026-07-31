#!/bin/bash
#SBATCH -J ivon-phase0-test
#SBATCH -p short
#SBATCH --gres=gpu:h100:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /mnt/vast/home/vv23qeku/ivon_gpu_test.out

set -x
PY=/mnt/vast/workspaces/BayesRL/micromamba/envs/bayesrl/bin
cd /mnt/vast/home/vv23qeku/bayesrl/ivon
$PY/python -m pytest tests/test_ivon_hardening.py -q || exit 1
$PY/torchrun --nproc_per_node=2 tests/dist_test_ivon.py || exit 1
echo "GPU PHASE0 TESTS PASSED"
