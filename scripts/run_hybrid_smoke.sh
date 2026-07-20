#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
default_dataset="$root/projectile_dataset_n4000_data1_split1_dt0p02.csv"
[[ -f $default_dataset ]] || default_dataset="$root/../projectile_dataset_n4000_data1_split1_dt0p02.csv"
dataset=${DATASET:-$default_dataset}
batch_size=${BATCH_SIZE:-64}
validate=()
[[ ${1:-} == --validate-only ]] && validate=(--validate-only)
mkdir -p "$root/validation_logs"
cd "$root"
make hybrid

mpirun --bind-to none -np 2 ./hybrid_mpi_cuda "$dataset" \
    --topology cpu-master --cpu-threads 1 --load-balance calibrated \
    --comm reduce-bcast --batch-size "$batch_size" --smoke-test "${validate[@]}" \
    | tee validation_logs/hybrid_smoke_cpu_master_np2.log

threads=$(( $(nproc) - 2 ))
(( threads < 1 )) && threads=1
mpirun --bind-to none -np 3 ./hybrid_mpi_cuda "$dataset" \
    --topology cpu-master --cpu-threads "$threads" --load-balance calibrated \
    --comm reduce-bcast --overlap on --batch-size "$batch_size" --smoke-test "${validate[@]}" \
    | tee validation_logs/hybrid_smoke_cpu_gpu_np3.log

threads=$(( $(nproc) - 1 ))
(( threads < 1 )) && threads=1
mpirun --bind-to none -np 2 ./hybrid_mpi_cuda "$dataset" \
    --topology gpu-master --cpu-threads "$threads" --load-balance calibrated \
    --comm reduce-bcast --batch-size "$batch_size" --smoke-test "${validate[@]}" \
    | tee validation_logs/hybrid_smoke_gpu_master_np2.log
