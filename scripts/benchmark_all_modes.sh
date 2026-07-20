#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
default_dataset="$root/projectile_dataset_n4000_data1_split1_dt0p02.csv"
[[ -f $default_dataset ]] || default_dataset="$root/../projectile_dataset_n4000_data1_split1_dt0p02.csv"
dataset=${DATASET:-$default_dataset}
batch_size=${BATCH_SIZE:-64}
mkdir -p "$root/validation_logs"
cd "$root"
make all hybrid
./mlp_eigen_sequential "$dataset" --batch-size "$batch_size" | tee validation_logs/benchmark_sequential.log
OMP_PROC_BIND=close OMP_PLACES=cores ./mlp_openmp "$dataset" "$(nproc)" --batch-size "$batch_size" | tee validation_logs/benchmark_openmp.log
for ranks in 1 2 4; do
    mpirun --bind-to core --map-by core -np "$ranks" ./mlp_mpi "$dataset" --batch-size "$batch_size" \
        | tee "validation_logs/benchmark_mpi_np${ranks}.log"
done
./mlp_cuda "$dataset" --batch-size "$batch_size" | tee validation_logs/benchmark_cuda.log
BATCH_SIZE="$batch_size" scripts/run_hybrid_autotune.sh
