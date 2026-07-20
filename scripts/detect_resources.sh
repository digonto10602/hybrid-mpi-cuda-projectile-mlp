#!/usr/bin/env bash
set -euo pipefail

echo "logical_cpus=$(nproc)"
lscpu | grep -E '^(CPU\(s\)|Core\(s\) per socket|Socket\(s\)|Thread\(s\) per core|Model name):' || true
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader || true
mpirun --version | head -1 || true
