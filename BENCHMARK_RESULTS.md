# Benchmark results

## Latest scaling result

The validated 20,000-sample, batch-256, 1,000-epoch experiment is documented
in `N20000_EPOCH1000_BENCHMARK_RESULTS.md`. OpenMP with 8 threads was fastest
at 2.51841720 s; MPI with 4 ranks took 2.59276995 s, single-GPU CUDA took
4.47819356 s, and the best true CPU+GPU topology took 8.44690897 s.

## Outcome

The best production training topology on the measured workstation was the
existing OpenMP implementation with four threads: **0.36042566 s** median for
300 epochs and **2,330,577.68 training samples/s**. The hybrid autotuner selected
its `cpu-only`, two-rank candidate in short calibration because it was the
fastest valid topology in the new executable. Among GPU-distributed candidates,
CPU master plus one GPU worker was fastest. Among genuine CPU+GPU compute
topologies, CPU master plus one GPU worker plus one 18-thread CPU worker was
fastest.

Fewer workers won. The primary mixed topology allocated a typical batch as
`master=0, GPU=47, CPU=17`, but communication/synchronization consumed 87.57% of
maximum-rank training time. Adding a second CPU worker made training slower.

## Production comparison

All rows use float32, the stored 2,800/600/600 split, batch 64, 300 epochs,
three complete repetitions, and median maximum-rank time where MPI is involved.

| Implementation | Topology / workers | Median train (s) | Samples/s | Val RMSE | Test RMSE | Test MAE | Test R2 | Inference (s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Sequential Eigen | 1 thread | 0.41426635 | 2,027,680.98 | 4.65101 | 4.66033 | 2.84558 | 0.985610 | 0.05298598 | PASS |
| OpenMP | 1 thread | 0.43621570 | 1,925,652.82 | 4.65101 | 4.66033 | 2.84558 | 0.985610 | 0.05318648 | PASS |
| OpenMP | 2 threads | 0.38740084 | 2,168,296.79 | 4.59873 | 4.71757 | 2.81816 | 0.985254 | 0.01604783 | PASS |
| OpenMP | 4 threads | **0.36042566** | **2,330,577.68** | 4.60720 | 4.69281 | 2.82745 | 0.985409 | 0.00938034 | PASS |
| OpenMP | 8 threads | 0.45029626 | 1,865,438.54 | 4.55326 | 4.68538 | 2.76860 | 0.985455 | **0.00649687** | PASS |
| OpenMP | 16 threads | 0.72642356 | 1,156,350.16 | 4.72216 | 4.71234 | 2.78985 | 0.985287 | 0.01280699 | PASS |
| OpenMP | 20 threads | 0.86873734 | 966,920.57 | 4.66620 | 4.71969 | 2.79225 | 0.985241 | 0.01289029 | PASS |
| MPI CPU | 1 rank | 0.43433551 | 1,933,988.76 | 4.65101 | 4.66033 | 2.84558 | 0.985610 | 0.05530239 | PASS |
| MPI CPU | 2 ranks | 0.37165999 | 2,260,130.28 | 4.59873 | 4.71757 | 2.81816 | 0.985254 | 0.02329604 | PASS |
| MPI CPU | 4 ranks | 0.37011332 | 2,269,575.18 | 4.60174 | 4.75891 | 2.85683 | 0.984995 | 0.00731481 | PASS |
| CUDA C++ | one RTX 3070 | 0.82811239 | 1,014,355.07 | 4.69305 | 4.68703 | 2.78168 | 0.985445 | 0.00806532 | PASS WITH WARNINGS |
| Hybrid | GPU-only control path | 1.81894306 | 461,806.65 | 4.69305 | 4.68703 | 2.78168 | 0.985445 | 0.01597742 | PASS WITH WARNINGS |
| Hybrid | CPU master + GPU, 2 ranks | **1.68557546** | **498,346.13** | 4.65426 | 4.72267 | 2.82901 | 0.985223 | 0.01733621 | PASS WITH WARNINGS |
| Hybrid | CPU master + GPU + CPU(18), 3 ranks | **1.75375705** | **478,971.70** | 4.56621 | 4.74146 | 2.86020 | 0.985105 | 0.01863152 | PASS WITH WARNINGS |
| Hybrid | CPU master + GPU + 2 CPU(9), 4 ranks | 1.83048344 | 458,895.16 | 4.54509 | 4.68452 | 2.77018 | 0.985460 | 0.02493837 | PASS WITH WARNINGS |
| Hybrid | GPU master + CPU(19), 2 ranks | 2.16969131 | 387,151.85 | 4.63714 | 4.72153 | 2.78848 | 0.985230 | 0.02664564 | PASS WITH WARNINGS |
| Hybrid | CPU-only, 2 ranks, CPU(19) | 1.06552339 | 788,344.96 | 4.72795 | 4.84067 | 2.94856 | 0.984475 | 0.08412046 | PASS |
| Hybrid | allreduce GPU worker, 2 ranks | 1.68835276 | 497,526.36 | 4.65426 | 4.72267 | 2.82901 | 0.985223 | 0.01733492 | PASS WITH WARNINGS |

CUDA rows are `PASS WITH WARNINGS` because nvcc emits warnings from Eigen
headers, not from project code. Exact values are in `benchmark_results.csv` and
`accuracy_results.csv`.

The separately labeled global-batch-256 experiment is documented in
`BATCH_256_BENCHMARK_RESULTS.md`; exact values are in
`benchmark_results_batch256.csv` and `accuracy_results_batch256.csv`.

## Scaling

Using sequential Eigen as `T1`, OpenMP speedup/efficiency is 1.069/53.5% at two
threads and 1.149/28.7% at four threads. Pure MPI is 1.115/55.7% at two ranks and
1.119/28.0% at four ranks. Scaling reverses after four CPU workers because this
network is too small for added scheduling and reduction overhead.

The CPU-master mixed strategy is 1.237 times faster than the GPU-master mixed
strategy on this machine. Reduce+broadcast is 0.16% faster than allreduce in the
matched two-rank GPU-worker comparison, which is within normal timing noise.

## Autotuning candidates

Short autotuning ranked six valid configurations. The final clean-build pass
selected `cpu-only,np=2,threads=19` at 0.00709907 s. GPU-distributed runner-up
was `cpu-master,np=2` at 0.01073664 s. The selected topology file is consumed by `run_hybrid_full.sh`.
Selection is based on measured time after correctness gates—not on resource
count.
