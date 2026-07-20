# Full validation and benchmark report

Date: 2026-07-20

Overall status: **PASS WITH WARNINGS**

Repository: `digonto10602/hybrid-mpi-cuda-projectile-mlp`

## Executive summary

Sequential Eigen, OpenMP, MPI, single-GPU CUDA, and hybrid MPI+CUDA
implementations were built, trained, tested, numerically compared, sanitized,
and benchmarked on the same projectile-range surrogate problem.

The most demanding completed experiment used 20,000 samples, batch 256, and
1,000 epochs. OpenMP with 8 threads was fastest at **2.51841720 seconds** and
**5,559,047 samples/second**. MPI with 4 ranks was close at 2.59276995 seconds.
Single-GPU CUDA took 4.47819356 seconds. The fastest true CPU+GPU hybrid took
8.44690897 seconds because synchronous communication consumed most of its
runtime. Using every resource was not optimal for this 4,481-parameter model.

All completed models produced finite metrics and substantially outperformed
the training-mean and no-drag baselines. CUDA-bearing builds are marked
`PASS WITH WARNINGS` only because nvcc emits warnings from Eigen headers.

## Validated environment

| Resource | Measured value |
|---|---|
| Host | SupremeCommander |
| Operating system | Ubuntu 24.04.4 LTS, Linux 6.8.0-65-generic |
| CPU | Intel Core i7-12700KF, 12 physical / 20 logical CPUs |
| GPU | NVIDIA GeForce RTX 3070, compute capability 8.6, 8 GiB |
| C++ compiler | GCC 13.3.0 |
| MPI | Open MPI 4.1.6 |
| CUDA compiler | CUDA 13.0 |
| MPI thread support | `MPI_THREAD_FUNNELED` |

Resource discovery used node-local MPI communicators, actual process affinity,
OpenMP configuration, and CUDA runtime discovery. The primary four-rank mixed
layout assigned rank 0 as coordinator, rank 1 to GPU 0, and ranks 2 and 3 to
nine disjoint logical CPUs each.

## Datasets

| Dataset | Samples | Stored split | SHA-256 | Status |
|---|---:|---|---|---|
| `projectile_dataset_n4000_data1_split1_dt0p02.csv` | 4,000 | 2,800/600/600 | `8d4c5c6ab0e138d9a60fd00895f78cc25383c38ee8bd5164fb62ae808415734d` | PASS |
| `projectile_dataset_n20000_data1_split1_dt0p02.csv` | 20,000 | 14,000/3,000/3,000 | `a2c1f2d30ab1eca79213ccc010e344d4ab1675b4c4e5ed9e8d63960e18320047` | PASS |

Both files use `split,v0,theta_deg,drag,range`. Validation confirmed five
columns per row, valid split labels, finite values, exact row counts, and
nonempty stored splits. Standardization statistics are fitted only to training
samples.

## Scientific contract

The implementations share:

- architecture `3 -> 64 -> 64 -> 1`;
- float32 model parameters and gradients;
- MSE loss and ReLU activation;
- Adam with learning rate `1e-3`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`;
- stored train/validation/test assignments;
- one Adam update per global mini-batch;
- gradient packing order `W1,b1,W2,b2,W3,b3`;
- exactly 4,481 packed parameters and gradients;
- untimed warm-up and median of three complete production repetitions;
- file I/O, reporting, validation, and metrics outside training timers.

The default workload remains 4,000 samples, batch 64, and 300 epochs. Batch
256 and the 20,000-sample/1,000-epoch run are explicitly labeled experiments
selected through command-line options.

## Implementations and hybrid algorithm

The validated reference modes are sequential Eigen, OpenMP shared-memory
gradients, MPI CPU all-reduce, and single-GPU CUDA. Hybrid modes include CPU
master plus GPU worker, GPU master plus CPU workers, CPU-only master/worker,
GPU-only control, reduce+broadcast, and allreduce comparison paths.

Workers receive disjoint pieces of one global mini-batch. Each worker divides
its gradient by the global batch size, not its local sample count. The master
sums worker gradients without dividing by rank count and performs exactly one
Adam update. Updated parameters are synchronized before the next batch.
Calibrated load balancing measures worker throughput and assigns integer sample
counts with exact coverage. Pinned host memory is used for portable GPU/MPI
staging.

## Validation results

| Validation | Result | Status |
|---|---|---|
| Strict sequential/OpenMP/MPI builds | Project sources warning-clean | PASS |
| Strict CUDA/hybrid builds | Eigen-header nvcc warnings only | PASS WITH WARNINGS |
| Sequential/OpenMP/MPI/CUDA smoke tests | Completed on both dataset scales | PASS |
| Dataset schema, counts, and finiteness | Exact expected values | PASS |
| Packed parameter and gradient count | 4,481 | PASS |
| GPU forward versus Eigen | Relative L2 approximately `1e-7` | PASS |
| GPU backward groups | Relative L2 below `3e-7` | PASS |
| Reduced heterogeneous gradient | All six groups below tolerance | PASS |
| Global-batch sample assignment | Every sample exactly once | PASS |
| Replica synchronization | Maximum parameter spread zero | PASS |
| Repeated complete training | Maximum parameter difference zero | PASS |
| Distributed inference ordering | Relative difference below `4e-7` | PASS |
| Finite loss, parameters, predictions, metrics | Verified in all completed runs | PASS |
| Training-mean baseline improvement | All full models substantially better | PASS |
| Multi-GPU execution | One GPU was available | NOT TESTED |
| Direct CUDA-aware MPI | Portable pinned-host path used | NOT TESTED |

The 20,000-sample hybrid correctness batch contained 256 unique contiguous
slots. In the four-rank calibrated run, the normal allocation was
`master=0, GPU=97, CPU=92, CPU=67`.

## Sanitizers and diagnostics

| Diagnostic | Result | Status |
|---|---|---|
| AddressSanitizer + UndefinedBehaviorSanitizer | No project-code findings | PASS |
| Compute Sanitizer memcheck | Zero errors | PASS |
| Compute Sanitizer racecheck | Zero hazards | PASS |
| Compute Sanitizer initcheck | Zero errors | PASS |
| CUDA launch blocking validation | No asynchronous launch errors | PASS |
| ThreadSanitizer with MPI/OpenMP | PMIx initialization incompatibility | NOT TESTED |
| MPI affinity validation | Nonoverlapping assigned masks | PASS |

## Benchmark methodology

Each production result is the median of three complete training repetitions
after untimed warm-up. Sequential and OpenMP use steady host timing. MPI uses
the maximum rank elapsed time. CUDA synchronizes at timer boundaries. Dataset
loading, standardization, validation, prediction transfer, metrics, and
printing are excluded from training time. Hybrid component timers separately
measure CPU/GPU compute, parameter broadcast, reduction, H2D, D2H, Adam, and
synchronization.

## Experiment 1: default production workload

Workload: 4,000 samples, batch 64, 300 epochs.

| Implementation | Best tested configuration | Median train (s) | Test RMSE | Status |
|---|---|---:|---:|---|
| Sequential Eigen | 1 thread | 0.41426635 | 4.66033 | PASS |
| OpenMP | 4 threads | **0.36042566** | 4.69281 | PASS |
| MPI CPU | 4 ranks | 0.37011332 | 4.75891 | PASS |
| CUDA C++ | RTX 3070 | 0.82811239 | 4.68703 | PASS WITH WARNINGS |
| Hybrid executable | CPU-only, 2 ranks, 19 threads | 1.06552339 | 4.84067 | PASS |
| True CPU+GPU hybrid | CPU master + GPU + CPU(18) | 1.75375705 | 4.74146 | PASS WITH WARNINGS |

OpenMP with four threads was fastest. Full worker-count results are in
`benchmark_results.csv` and `BENCHMARK_RESULTS.md`.

## Experiment 2: batch-256 scaling

Workload: 4,000 samples, batch 256, 300 epochs.

| Implementation | Best tested configuration | Median train (s) | Test RMSE | Status |
|---|---|---:|---:|---|
| Sequential Eigen | 1 thread | 0.46511502 | 5.75997 | PASS |
| OpenMP | 8 threads | **0.12788462** | 5.72867 | PASS |
| MPI CPU | 4 ranks | 0.13310524 | 5.81016 | PASS |
| CUDA C++ | RTX 3070 | 0.26852771 | 6.30641 | PASS WITH WARNINGS |
| Hybrid executable | CPU-only, 2 ranks, 19 threads | 0.36244186 | 5.70815 | PASS |
| True CPU+GPU hybrid | CPU master + GPU + 2 CPU(9) | 0.46010461 | 5.71626 | PASS WITH WARNINGS |

Batch 256 reduced synchronization overhead but made only 11 Adam updates per
epoch instead of 44, so its fixed-300-epoch accuracy was lower. Full results
are in `benchmark_results_batch256.csv` and
`BATCH_256_BENCHMARK_RESULTS.md`.

## Experiment 3: 20,000 samples and 1,000 epochs

Workload: 20,000 samples, batch 256, 1,000 epochs.

| Implementation | Configuration | Median train (s) | Samples/s | Val RMSE | Test RMSE | Test R2 | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Sequential Eigen | 1 thread | 5.35435215 | 2,614,695 | 1.29781 | 1.23281 | 0.999104 | PASS |
| OpenMP | 2 threads | 3.51498957 | 3,982,942 | 1.18691 | 1.15032 | 0.999220 | PASS |
| OpenMP | 4 threads | 2.63892620 | 5,305,188 | 1.21776 | 1.22131 | 0.999120 | PASS |
| OpenMP | **8 threads** | **2.51841720** | **5,559,047** | 1.00103 | 1.08077 | 0.999311 | PASS |
| OpenMP | 16 threads | 3.85944713 | 3,627,463 | 1.12879 | 1.06458 | 0.999332 | PASS |
| OpenMP | 20 threads | 4.09741957 | 3,416,785 | 1.00278 | **1.01243** | **0.999395** | PASS |
| MPI CPU | 1 rank | 5.56673496 | 2,514,939 | 1.29781 | 1.23281 | 0.999104 | PASS |
| MPI CPU | 2 ranks | 3.47081383 | 4,033,636 | 1.18691 | 1.15032 | 0.999220 | PASS |
| MPI CPU | 4 ranks | **2.59276995** | **5,399,631** | 1.24209 | 1.17437 | 0.999187 | PASS |
| CUDA C++ | RTX 3070 | 4.47819356 | 3,126,261 | 1.13036 | 1.14614 | 0.999225 | PASS WITH WARNINGS |
| Hybrid | GPU-only control | 8.90629132 | 1,571,923 | 1.13036 | 1.14614 | 0.999225 | PASS WITH WARNINGS |
| Hybrid | CPU master + GPU | 8.67788772 | 1,613,296 | 1.58462 | 1.57590 | 0.998535 | PASS WITH WARNINGS |
| Hybrid | CPU master + GPU + CPU(18) | 8.73268954 | 1,603,172 | 1.27204 | 1.26134 | 0.999062 | PASS WITH WARNINGS |
| Hybrid | CPU master + GPU + 2 CPU(9) | **8.44690897** | **1,657,411** | 1.08549 | 1.12244 | 0.999257 | PASS WITH WARNINGS |
| Hybrid | GPU master + CPU(19) | 8.79413016 | 1,591,971 | 1.14406 | 1.18177 | 0.999176 | PASS WITH WARNINGS |
| Hybrid | CPU-only master/worker | **5.54067876** | **2,526,766** | **0.96899** | **1.01050** | **0.999398** | PASS |
| Hybrid | CPU master + GPU, allreduce | 8.78268223 | 1,594,046 | 1.58462 | 1.57590 | 0.998535 | PASS WITH WARNINGS |

OpenMP with 8 threads was 2.126 times faster than sequential Eigen. The best
OpenMP accuracy occurred at 20 threads, but training was slower. The full
accuracy fields, including MAE, median error, P95 error, and maximum error,
are in `accuracy_results_n20000_e1000.csv`.

## Hybrid performance analysis

The short autotuner and full runs both identified CPU-only master/worker as the
fastest mode inside the hybrid executable. The fastest true mixed topology was
CPU master + GPU + two 9-thread CPU workers. Communication fractions across
the tested true hybrid layouts ranged from approximately 61% to 85%. The
4,481-parameter model does too little work between reductions and broadcasts
to amortize MPI synchronization and GPU staging on this workstation.

Reduce+broadcast slightly outperformed allreduce in the matched two-rank
comparison, but both were much slower than direct OpenMP and MPI CPU paths.
This is a valid benchmark result rather than a correctness failure.

## Problems found and fixed

1. Host and CUDA translation units initially used incompatible Eigen SIMD ABI
   flags. Hybrid compilation now uses consistent host flags.
2. CUDA library discovery could mix toolkit versions. The Makefile now derives
   matching CUDA runtime, cuBLAS paths, and rpaths.
3. The standalone CUDA entry point conflicted with hybrid kernel reuse. A
   library guard now preserves both builds.
4. Heterogeneous sample allocation lacked exact integer coverage guarantees.
   Calibrated largest-remainder assignment now audits every global batch.
5. MPI ranks inherited overlapping CPU masks. Role-aware affinity assignment
   now creates disjoint masks and rejects oversubscription.
6. Scripts assumed one dataset location and fixed batch/epoch settings. They
   now accept `DATASET`, `BATCH_SIZE`, and `EPOCHS` overrides while preserving
   original defaults.

## Reproduction

Build everything:

```bash
make clean
make all
```

Reproduce the fastest 20,000-sample result:

```bash
DATASET=$PWD/projectile_dataset_n20000_data1_split1_dt0p02.csv

OMP_PROC_BIND=close OMP_PLACES=cores \
  ./mlp_openmp "$DATASET" 8 --batch-size 256 --epochs 1000
```

Reproduce the fastest true hybrid configuration:

```bash
mpirun --bind-to none -np 4 ./hybrid_mpi_cuda "$DATASET" \
  --topology cpu-master --cpu-threads 9 \
  --batch-size 256 --epochs 1000 \
  --load-balance calibrated --comm reduce-bcast --overlap on
```

Re-run topology selection:

```bash
DATASET="$DATASET" BATCH_SIZE=256 EPOCHS=1000 \
  scripts/run_hybrid_autotune.sh
```

## Remaining limitations

- Only one GPU was visible, so multi-GPU distributed training is NOT TESTED.
- Direct CUDA-aware MPI is NOT TESTED; pinned-host staging is validated.
- ThreadSanitizer could not pass the OpenMPI/PMIx initialization layer.
- Results apply to this workstation; other machines require fresh autotuning.
- The small synchronous network is communication-bound in hybrid modes.

## Result artifacts

- `benchmark_results.csv`, `accuracy_results.csv`: default workload.
- `benchmark_results_batch256.csv`, `accuracy_results_batch256.csv`: batch 256.
- `benchmark_results_n20000_e1000.csv`, `accuracy_results_n20000_e1000.csv`:
  20,000 samples and 1,000 epochs.
- `HYBRID_MPI_CUDA_VALIDATION_REPORT.md`: detailed hybrid correctness record.
- `BENCHMARK_RESULTS.md`, `BATCH_256_BENCHMARK_RESULTS.md`, and
  `N20000_EPOCH1000_BENCHMARK_RESULTS.md`: experiment-specific analysis.

## Final conclusion

The package is fully functional on the measured single-GPU workstation. All
supported single-GPU and CPU configurations passed correctness and runtime
validation. For the largest tested workload, OpenMP with 8 threads is the
recommended training topology. Use CUDA for its low inference latency or for
larger networks that can amortize launch and transfer overhead; do not select
the MPI+CUDA hybrid solely to maximize hardware occupancy.
