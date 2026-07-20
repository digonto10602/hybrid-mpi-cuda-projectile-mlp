#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
default_dataset="$root/projectile_dataset_n4000_data1_split1_dt0p02.csv"
[[ -f $default_dataset ]] || default_dataset="$root/../projectile_dataset_n4000_data1_split1_dt0p02.csv"
dataset=${DATASET:-$default_dataset}
cores=$(nproc)
mkdir -p "$root/validation_logs"
cd "$root"
make hybrid
results=validation_logs/hybrid_autotune_candidates.csv
echo 'topology,ranks,cpu_threads,load_balance,comm,training_seconds,throughput,validation_rmse,test_rmse,test_mae,test_r2,inference_seconds,inference_relative,status' > "$results"

run_candidate() {
    local topology=$1 ranks=$2 threads=$3
    local log="validation_logs/autotune_${topology}_np${ranks}_t${threads}.log"
    mpirun --bind-to none -np "$ranks" ./hybrid_mpi_cuda "$dataset" \
        --topology "$topology" --cpu-threads "$threads" --load-balance calibrated \
        --comm reduce-bcast --autotune | tee "$log"
    sed -n 's/^RESULT_CSV,//p' "$log" >> "$results"
}

run_candidate cpu-master 2 1
threads=$(( cores - 2 )); (( threads < 1 )) && threads=1
run_candidate cpu-master 3 "$threads"
threads=$(( (cores - 2) / 2 )); (( threads < 1 )) && threads=1
run_candidate cpu-master 4 "$threads"
threads=$(( cores - 1 )); (( threads < 1 )) && threads=1
run_candidate gpu-master 2 "$threads"
run_candidate cpu-only 2 "$threads"
run_candidate gpu-only 1 1

best=$(awk -F, 'NR>1 && $14=="PASS" && (!best || $6<time) {best=$0; time=$6} END {print best}' "$results")
IFS=, read -r selected_topology selected_ranks selected_threads _ <<< "$best"
printf 'TOPOLOGY=%q\nRANKS=%q\nCPU_THREADS=%q\n' \
    "$selected_topology" "$selected_ranks" "$selected_threads" \
    > validation_logs/hybrid_selected_topology.env
echo "Selected topology CSV: $best"
