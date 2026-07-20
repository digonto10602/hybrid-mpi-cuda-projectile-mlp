# Hybrid MPI + CUDA validation report

Date: 2026-07-20
Overall status: **PASS WITH WARNINGS**

## Environment and dataset

| Item | Measured value |
|---|---|
| Host | SupremeCommander |
| OS | Ubuntu 24.04.4 LTS, Linux 6.8.0-65-generic |
| CPU | Intel Core i7-12700KF, 12 physical / 20 logical CPUs |
| GPU | NVIDIA GeForce RTX 3070, compute capability 8.6, 8192 MiB |
| Driver / CUDA compiler | 580.173.02 / CUDA 13.0 |
| C++ / MPI | GCC 13.3.0 / OpenMPI 4.1.6 |
| Dataset rows | 4,000 plus header; train/validation/test = 2,800/600/600 |
| Dataset SHA-256 | `8d4c5c6ab0e138d9a60fd00895f78cc25383c38ee8bd5164fb62ae808415734d` |

The CSV schema is `split,v0,theta_deg,drag,range`. Loading verifies five finite
numeric values, valid split labels, and exact counts. Feature and target
standardizers are fitted only on training data.

## Scientific and timing audit

All reference and hybrid production rows use architecture `3->64->64->1`,
float32 model data, global batch 64, 300 epochs, MSE, ReLU, and Adam with
`lr=1e-3`, `beta1=0.9`, `beta2=0.999`, and `epsilon=1e-8`. Parameter and gradient
packing order is `W1,b1,W2,b2,W3,b3`; runtime assertions require 4,481 values.

Local gradients use `2*(prediction-target)/64`, even for smaller local slices.
Slices are disjoint and cover each global batch exactly once. Reduce+broadcast
performs one authoritative Adam update; no division by rank count follows the
sum. Allreduce uses identical optimizer replicas only as a comparison mode.

Warm-up, calibration, loading, printing, validation, metrics, and inference are
outside training timers. Each production result is the median of three complete
300-epoch runs; MPI time is the maximum over ranks. CUDA work is synchronized
at timing boundaries. Component timing includes broadcasts, CPU/GPU compute,
H2D, D2H, reduction, GPU Adam, validation, inference, and gather.

## Resource mapping and tested topologies

Discovery found one shared-memory node, 20 visible logical CPUs, one CUDA GPU,
and `MPI_THREAD_FUNNELED`. The hybrid executable assigns nonoverlapping affinity
masks after discovery. The principal four-rank layout was:

| Rank | Role | Assigned resource |
|---:|---|---|
| 0 | CPU master | one coordinator CPU |
| 1 | GPU worker | RTX 3070 plus one host CPU |
| 2 | CPU worker | 9 logical CPUs |
| 3 | CPU worker | 9 logical CPUs |

Tested topology set: sequential, OpenMP 1/2/4/8/16/20, MPI 1/2/4, single CUDA,
GPU-only hybrid, CPU master plus GPU, CPU master plus GPU plus one or two CPU
workers, GPU master plus CPU worker, CPU-only master/worker, reduce+broadcast,
allreduce, static/calibrated/adaptive balancing, and one-rank GPU-allreduce
control path. `--overlap on` exercised simultaneous CPU/GPU work on distinct
ranks. No tested layout oversubscribed its process affinity.

For normal calibrated batches, measured allocations were:

- CPU master + GPU: `0,64`.
- CPU master + GPU + CPU(18): `0,47,17`.
- CPU master + GPU + 2 CPU(9): `0,27,23,14`.
- GPU master + CPU(19): `42,22`.
- CPU-only master + CPU(19): `0,64`.

Allocation is recalibrated per training repetition, so counts may vary slightly
with system noise while retaining exact coverage.

## Correctness results

| Check | Result | Status |
|---|---|---|
| Split counts, schema, finiteness | 2,800/600/600; all valid | PASS |
| Packed parameters / gradients | 4,481 / 4,481 | PASS |
| GPU forward vs Eigen | relative L2 approximately `1.13e-7` | PASS |
| GPU backward W1 | max abs `5.96e-8`, relative L2 `1.44e-7` | PASS |
| GPU backward b1 | max abs `5.96e-8`, relative L2 `1.07e-7` | PASS |
| GPU backward W2 | max abs `1.19e-7`, relative L2 `1.11e-7` | PASS |
| GPU backward b2 | max abs `8.94e-8`, relative L2 `1.50e-7` | PASS |
| GPU backward W3 | max abs `4.77e-7`, relative L2 `1.35e-7` | PASS |
| GPU backward b3 | max abs `1.19e-7`, relative L2 `8.35e-8` | PASS |
| Mixed reduced W1 | max abs `8.94e-8`, relative L2 `1.5e-7` | PASS |
| Mixed reduced b1 | max abs `4.47e-8`, relative L2 `8e-8` | PASS |
| Mixed reduced W2 | max abs `2.38e-7`, relative L2 `1.2e-7` | PASS |
| Mixed reduced b2 | max abs `8.94e-8`, relative L2 `1.3e-7` | PASS |
| Mixed reduced W3 | max abs `7.15e-7`, relative L2 `1.7e-7` | PASS |
| Mixed reduced b3 | max abs `2.38e-7`, relative L2 `1.5e-7` | PASS |
| Fixed-batch assignment | 64 unique contiguous slots | PASS |
| Parameter replicas | maximum spread 0 | PASS |
| Three repeated complete runs | maximum parameter difference 0 | PASS |
| Gathered inference order | relative difference at most `3.5e-7` | PASS |
| Finite loss/parameters/metrics | verified during all runs | PASS |
| Beats training-mean baseline | test RMSE about 4.7 vs 38.86 | PASS |
| Beats no-drag baseline | test RMSE about 4.7 vs 213.33 | PASS |

The strict comparison initializes parameters once on CPU and compares each
named parameter group over the same fixed global batch. Tolerances are
`max_abs <= 1e-4` and `relative_L2 <= 1e-5`; observed errors are much smaller.

## Build and diagnostic status

| Validation | Status | Evidence summary |
|---|---|---|
| Strict hybrid release build | PASS WITH WARNINGS | project CPU sources warning-clean; nvcc reports Eigen-header constexpr warnings |
| Debug build | PASS | CPU `-O0 -g`, CUDA `-O0 -g -G` |
| CPU ASan + UBSan smoke | PASS WITH WARNINGS | no ASan/UBSan finding; leak detection disabled for MPI runtime |
| ThreadSanitizer | NOT TESTED | executable built, but TSan failed inside PMIx shared-memory initialization before project code |
| Compute Sanitizer memcheck | PASS | zero errors |
| Compute Sanitizer racecheck | PASS | zero hazards, errors, or warnings |
| Compute Sanitizer initcheck | PASS | zero errors |
| CUDA launch blocking validation | PASS | no asynchronous launch error |
| MPI binding / application affinity | PASS | mapping logged; application affinity masks disjoint |
| GPU activity sampling | PASS WITH WARNINGS | clocks/power activity captured; 1-second SM sample missed short kernels |

## Full training and benchmark status

All production rows in `benchmark_results.csv` completed. The primary mixed
CPU-master topology completed three repetitions with median 1.75375705 s,
validation RMSE 4.56620683, test RMSE 4.74145504, test MAE 2.86019812, and test
R2 0.98510470. Its inference end-to-end median was 0.01863152 s.

Overall fastest training was OpenMP four threads at 0.36042566 s. The short
autotuner selected the hybrid executable's CPU-only two-rank topology; among
GPU-distributed candidates it selected CPU master plus one GPU worker. Among
true CPU+GPU compute plans, CPU master + GPU + CPU(18) was best. The GPU-master
mixed plan was 23.7% slower than the analogous CPU-master mixed plan because it
adds reduced-gradient H2D and updated-parameter D2H staging around GPU Adam.

## Problems found and exact fixes

1. Optimized hybrid builds initially crashed despite debug and UBSan passing.
   Eigen objects crossed translation units compiled with different SIMD ABI
   settings: MPI C++ used `-march=native`, while nvcc host compilation did not.
   Hybrid objects now share `HYBRID_CXXFLAGS` with `-march=native` removed.
2. CUDA library discovery initially mixed an older CUDA runtime with nvcc 13.0.
   The Makefile now derives the compiler release and locates matching cudart and
   cuBLAS directories, with matching rpaths. `ldd` confirms consistent linkage.
3. CUDA implementation reuse previously conflicted with its standalone `main`.
   The standalone entry point now has a library guard; normal CUDA behavior is
   unchanged, and the hybrid GPU wrapper reuses the validated kernels.
4. Heterogeneous batches originally needed exact integer and nonempty-worker
   guarantees. The load balancer now uses calibrated largest-remainder
   assignment with audits for sum, offsets, duplication, and omissions.
5. MPI-launched ranks initially inherited the same broad CPU mask. The hybrid
   executable now reserves coordinator/GPU host CPUs and assigns disjoint masks
   to CPU worker ranks, rejecting oversubscription.
6. Dataset lookup in scripts assumed only a parent-directory dataset. Scripts
   now prefer a repository-local CSV and fall back to the package parent.

## Commands executed

Representative exact commands (all corresponding output was inspected):

```bash
make clean
make all
make hybrid_debug
make hybrid
scripts/detect_resources.sh
scripts/run_hybrid_smoke.sh
scripts/run_hybrid_autotune.sh

mpirun --bind-to none -np 3 ./hybrid_mpi_cuda DATASET \
  --topology cpu-master --cpu-threads 18 --load-balance calibrated \
  --comm reduce-bcast --overlap on --validate-only

mpirun --bind-to none -np 3 ./hybrid_mpi_cuda DATASET \
  --topology cpu-master --cpu-threads 18 --load-balance calibrated \
  --comm reduce-bcast --overlap on

mpirun --bind-to none -np 4 ./hybrid_mpi_cuda DATASET \
  --topology cpu-master --cpu-threads 9 --load-balance calibrated \
  --comm reduce-bcast --overlap on

mpirun --bind-to none -np 2 ./hybrid_mpi_cuda DATASET \
  --topology gpu-master --cpu-threads 19 --load-balance calibrated

compute-sanitizer --tool memcheck ./hybrid_mpi_cuda DATASET \
  --topology gpu-only --validate-only
compute-sanitizer --tool racecheck ./hybrid_mpi_cuda DATASET \
  --topology gpu-only --validate-only
compute-sanitizer --tool initcheck ./hybrid_mpi_cuda DATASET \
  --topology gpu-only --validate-only
```

`DATASET` denotes the CSV path listed by checksum above. Additional tested
commands cover 1/2/4-rank references, every topology in the result table,
static/adaptive balancing, reduce+broadcast/allreduce, ASan/UBSan, TSan,
`CUDA_LAUNCH_BLOCKING=1`, `mpirun --report-bindings`, and `nvidia-smi dmon`.

## Remaining limitations

- Multi-GPU distributed execution: **NOT TESTED**; only one GPU was visible.
- Direct CUDA-aware MPI: **NOT TESTED**; validated pinned-host staging is used.
- TSan race analysis: **NOT TESTED** due an OpenMPI/PMIx/TSan initialization
  incompatibility, not a project-code finding.
- External per-rank CPU utilization counters: **NOT TESTED**; internal
  component timers and affinity were recorded.
- Explicit communication/compute pipeline overlap beyond naturally concurrent
  CPU/GPU rank computation remains experimental.
- Performance results apply to this workstation and tiny batch-64 network;
  clusters with different MPI latency require fresh autotuning.
