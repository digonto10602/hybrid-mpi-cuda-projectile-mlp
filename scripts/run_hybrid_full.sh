#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
default_dataset="$root/projectile_dataset_n4000_data1_split1_dt0p02.csv"
[[ -f $default_dataset ]] || default_dataset="$root/../projectile_dataset_n4000_data1_split1_dt0p02.csv"
dataset=${DATASET:-$default_dataset}
batch_size=${BATCH_SIZE:-64}
epochs=${EPOCHS:-300}
[[ -z ${TOPOLOGY:-} && -z ${RANKS:-} && -z ${CPU_THREADS:-} && \
    -f "$root/validation_logs/hybrid_selected_topology.env" ]] && \
    source "$root/validation_logs/hybrid_selected_topology.env"
topology=${TOPOLOGY:-cpu-master}
ranks=${RANKS:-2}
threads=${CPU_THREADS:-1}
mkdir -p "$root/validation_logs"
cd "$root"
make hybrid
mpirun --report-bindings --bind-to none -np "$ranks" ./hybrid_mpi_cuda "$dataset" \
    --topology "$topology" --cpu-threads "$threads" --load-balance calibrated \
    --comm reduce-bcast --overlap on --batch-size "$batch_size" --epochs "$epochs" \
    | tee "validation_logs/hybrid_full_${topology}_np${ranks}_t${threads}.log"
