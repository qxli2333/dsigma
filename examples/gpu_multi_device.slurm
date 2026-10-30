#!/bin/bash
#SBATCH --job-name=dsigma-multi-gpu
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/q/qinxunli/dsigma_multi_gpu/slurm-%j.log
set -euo pipefail
# Submit from the repository root; create the output directory before sbatch.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUPY_CACHE_DIR=/pscratch/sd/q/qinxunli/dsigma_multi_gpu/cuda_cache
DSIGMA_PYTHON=${DSIGMA_PYTHON:-/pscratch/sd/q/qinxunli/dsigma_gpu_validation/venv/bin/python}
srun "$DSIGMA_PYTHON" -m pytest tests/test_gpu.py tests/test_multi_gpu.py -q
srun "$DSIGMA_PYTHON" examples/gpu_multi_device.py --devices 0 1 2 3 \
    --output /pscratch/sd/q/qinxunli/dsigma_multi_gpu/benchmark.json
