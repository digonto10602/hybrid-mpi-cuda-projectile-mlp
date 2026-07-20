# Hybrid MPI + CUDA projectile-surrogate MLP

This repository trains a float32 neural surrogate for projectile range,
`(v0, theta, drag) -> range`, and compares sequential Eigen, OpenMP, pure MPI,
single-GPU CUDA, and synchronous heterogeneous MPI + CUDA implementations.
The fixed production workload is `3 -> 64 -> 64 -> 1`, global batch 64, 300
epochs, MSE, ReLU, and Adam (`lr=1e-3`, `beta1=0.9`, `beta2=0.999`,
`epsilon=1e-8`). Standardization uses training samples only. Stored CSV split
labels are never regenerated during training.

The benchmark is intentionally small. Communication and kernel-launch overhead
can dominate useful work, so the autotuner may correctly recommend fewer
resources or a CPU-only topology.

Batch 64 remains the production default. A separately labeled batch-256
scaling experiment is available with `--batch-size 256`; measured results are
in `BATCH_256_BENCHMARK_RESULTS.md`.

A second scaling experiment uses 20,000 generated samples, batch 256, and
1,000 epochs. OpenMP with 8 threads was fastest on the measured workstation.
See `N20000_EPOCH1000_BENCHMARK_RESULTS.md` and the matching CSV files.

## Hybrid design

The hybrid executable supports these topology names:

- `cpu-master`: rank 0 owns the authoritative CPU model and Adam state; rank 1
  is the GPU worker; later ranks are optional OpenMP CPU workers.
- `gpu-master`: rank 0 owns the GPU model and GPU Adam state; later ranks are
  OpenMP CPU workers.
- `cpu-only`: rank 0 coordinates and later ranks compute CPU gradients.
- `gpu-only`: a one-rank GPU reference through the hybrid control path.
- `gpu-allreduce`: one model replica per GPU rank. It requires at least one
  distinct visible GPU per local rank.

For each global batch, active workers receive disjoint contiguous sample ranges.
Each local gradient is divided by the configured global batch size, packed in
`W1,b1,W2,b2,W3,b3` order, and summed. The packed parameter and gradient counts
are asserted to be exactly 4,481. `reduce-bcast` performs exactly one Adam
update on the master. `allreduce` is a reference mode with identical host Adam
replicas. It must not be used with the GPU-master strategy.

Calibration times representative forward/backward work on every worker and
uses median throughput to allocate integer sample counts by largest remainder.
`--load-balance adaptive` repeats calibration periodically. CPU and GPU ranks
compute concurrently; `--overlap on` enables the validated separate-rank
concurrent path. Direct CUDA-aware MPI is deliberately not assumed: `auto` and
`off` use pinned host staging, while `on` fails clearly in this portable build.

Resource discovery uses `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`,
`sched_getaffinity`, OpenMP, scheduler environment variables, and the CUDA
runtime. The program assigns non-overlapping process affinity masks and writes
`resource_mapping.csv`. It rejects CPU oversubscription, more active workers
than batch samples, and accidental GPU sharing.

## Source layout

- `mlp_common.hpp`: shared data loading, model, metrics, and CPU math.
- `mlp_eigen_sequential.cpp`, `mlp_openmp.cpp`, `mlp_mpi.cpp`, `mlp_cuda.cu`:
  reference implementations.
- `hybrid_mpi_cuda_main.cpp`: MPI roles, training, timing, inference, and CLI.
- `hybrid_mpi_cuda_gpu.cu/.hpp`: persistent GPU worker and GPU Adam wrapper.
- `hybrid_resource_discovery.cpp/.hpp`: rank, affinity, and GPU inventory.
- `hybrid_load_balancer.cpp/.hpp`: calibrated integer allocation.
- `hybrid_validation.cpp/.hpp`: strict groupwise gradient comparison.
- `hybrid_worker_roles.hpp`: role definitions.

## Dependencies and build

Required: C++17 compiler, Eigen 3, OpenMPI, CUDA toolkit, CUDA driver, cuBLAS,
GNU Make, and OpenMP. Defaults target an RTX 3070 with `CUDA_ARCH=sm_86`; all
toolchain variables can be overridden.

```bash
make all
make hybrid
make hybrid_debug
```

Supported variables include `CXX`, `MPICXX`, `NVCC`, `EIGEN_INCLUDE`,
`CUDA_ARCH`, `CUDA_HOME`, `MPI_HOME`, `CXXFLAGS`, `NVCCFLAGS`, `OMPFLAGS`, and
`LDFLAGS`. Hybrid translation units intentionally remove `-march=native` so the
host compiler and nvcc use one Eigen ABI.

## Run and validate

The scripts find the dataset in the repository root first and then in its
parent. Set `DATASET=/absolute/path/file.csv` to override it.

```bash
make hybrid_smoke
make hybrid_validate
make hybrid_autotune
make hybrid_benchmark
scripts/detect_resources.sh
```

Direct examples:

```bash
mpirun --bind-to none -np 3 ./hybrid_mpi_cuda \
  projectile_dataset_n4000_data1_split1_dt0p02.csv \
  --topology cpu-master --cpu-threads 18 \
  --load-balance calibrated --comm reduce-bcast --overlap on --smoke-test

scripts/run_hybrid_autotune.sh
scripts/run_hybrid_full.sh
```

Production defaults are 300 epochs and three complete timed repetitions;
`--smoke-test` uses 2 epochs and one repetition. `--validate-only` performs the
fixed-batch correctness checks without a training timing repetition. The full
CLI also accepts:

```text
--topology cpu-master|gpu-master|cpu-only|gpu-only|gpu-allreduce
--cpu-threads N
--batch-size N
--epochs N
--load-balance static|calibrated|adaptive
--comm reduce-bcast|allreduce
--gpu-aware-mpi auto|on|off
--overlap off|on
--autotune
```

Run batch 256 without changing the default:

```bash
./mlp_eigen_sequential projectile_dataset_n4000_data1_split1_dt0p02.csv --batch-size 256
BATCH_SIZE=256 scripts/run_hybrid_autotune.sh
BATCH_SIZE=256 scripts/run_hybrid_full.sh
```

Run the 20,000-sample, 1,000-epoch experiment without changing the defaults:

```bash
DATASET=$PWD/projectile_dataset_n20000_data1_split1_dt0p02.csv
./mlp_openmp "$DATASET" 8 --batch-size 256 --epochs 1000
DATASET="$DATASET" BATCH_SIZE=256 EPOCHS=1000 scripts/run_hybrid_autotune.sh
```

## Timing and correctness

Warm-up and calibration occur outside production timers. Training reports the
median of complete runs using maximum rank elapsed time. The component report
separates broadcasts, CPU and GPU compute, H2D, D2H, reduction, GPU Adam, and
synchronization. Loading, validation, metrics, printing, and inference remain
outside the training timer.

Validation starts from identical CPU-initialized parameters and checks GPU
forward predictions, all six GPU gradient groups, the reduced heterogeneous
global gradient, exact sample coverage, parameter replica spread, repeated-run
reproducibility, finite metrics, baseline improvement, and gathered inference
order. See `HYBRID_MPI_CUDA_VALIDATION_REPORT.md`, `BENCHMARK_RESULTS.md`,
`BATCH_256_BENCHMARK_RESULTS.md`, and the result CSV files for the measured
workstation results.

The 20,000-sample dataset is included with SHA-256
`a2c1f2d30ab1eca79213ccc010e344d4ab1675b4c4e5ed9e8d63960e18320047`.
Its dedicated comparison is in `N20000_EPOCH1000_BENCHMARK_RESULTS.md`.

## Important limitations

- Only one GPU was available for the reported validation, so multi-GPU
  `gpu-allreduce` was not executed beyond its one-rank control-path case.
- The validated MPI path stages CUDA data through pinned host memory; direct
  CUDA-aware MPI remains untested.
- Explicit MPI communication/CUDA-stream overlap beyond concurrent CPU/GPU
  worker computation remains experimental.
- This synchronous batch-64 network is too small to amortize hybrid
  communication on the measured workstation.
