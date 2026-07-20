# 20,000-sample, 1,000-epoch comparison

Status: **PASS WITH WARNINGS**

This is a separately labeled scaling experiment using the stored 70/15/15
split (14,000/3,000/3,000), global batch 256, 1,000 epochs, and the unchanged
`3->64->64->1` float32 MLP. MSE, ReLU, Adam, training-only
standardization, warm-up, three complete timed repetitions, and median timing
are unchanged. The default 4,000-sample, batch-64, 300-epoch workload remains
available.

Dataset: `projectile_dataset_n20000_data1_split1_dt0p02.csv`
SHA-256: `a2c1f2d30ab1eca79213ccc010e344d4ab1675b4c4e5ed9e8d63960e18320047`

## Result

OpenMP with 8 threads was fastest overall at **2.51841720 s** and
**5,559,047 samples/s**. MPI with four ranks was second at 2.59276995 s. The
single-GPU CUDA implementation took 4.47819356 s. The best hybrid-executable
mode was CPU-only master/worker at 5.54067876 s; the best true CPU+GPU mode was
CPU master + GPU + two 9-thread CPU workers at 8.44690897 s.

The result is scientifically reasonable for this compact 4,481-parameter
network: CPU matrix work is small, while synchronous transfers and collectives
remain frequent. In the true hybrid configurations, measured communication
fractions were 61% to 85%, so adding the GPU did not improve end-to-end
training throughput.

## Full comparison

| Implementation | Workers / device | Median train (s) | Samples/s | Val RMSE | Test RMSE | Test MAE | Test R2 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Sequential Eigen | 1 thread | 5.35435215 | 2,614,695 | 1.29781 | 1.23281 | 0.65775 | 0.999104 | PASS |
| OpenMP | 1 thread | 5.44815634 | 2,569,677 | 1.29781 | 1.23281 | 0.65775 | 0.999104 | PASS |
| OpenMP | 2 threads | 3.51498957 | 3,982,942 | 1.18691 | 1.15032 | 0.58269 | 0.999220 | PASS |
| OpenMP | 4 threads | 2.63892620 | 5,305,188 | 1.21776 | 1.22131 | 0.65024 | 0.999120 | PASS |
| OpenMP | **8 threads** | **2.51841720** | **5,559,047** | 1.00103 | 1.08077 | 0.57275 | 0.999311 | PASS |
| OpenMP | 16 threads | 3.85944713 | 3,627,463 | 1.12879 | 1.06458 | 0.52791 | 0.999332 | PASS |
| OpenMP | 20 threads | 4.09741957 | 3,416,785 | 1.00278 | **1.01243** | **0.51737** | **0.999395** | PASS |
| MPI CPU | 1 rank | 5.56673496 | 2,514,939 | 1.29781 | 1.23281 | 0.65775 | 0.999104 | PASS |
| MPI CPU | 2 ranks | 3.47081383 | 4,033,636 | 1.18691 | 1.15032 | 0.58269 | 0.999220 | PASS |
| MPI CPU | 4 ranks | **2.59276995** | **5,399,631** | 1.24209 | 1.17437 | 0.62997 | 0.999187 | PASS |
| CUDA C++ | RTX 3070 | 4.47819356 | 3,126,261 | 1.13036 | 1.14614 | 0.60870 | 0.999225 | PASS WITH WARNINGS |
| Hybrid | GPU only | 8.90629132 | 1,571,923 | 1.13036 | 1.14614 | 0.60870 | 0.999225 | PASS WITH WARNINGS |
| Hybrid | CPU master + GPU | 8.67788772 | 1,613,296 | 1.58462 | 1.57590 | 1.05425 | 0.998535 | PASS WITH WARNINGS |
| Hybrid | CPU master + GPU + CPU(18) | 8.73268954 | 1,603,172 | 1.27204 | 1.26134 | 0.77615 | 0.999062 | PASS WITH WARNINGS |
| Hybrid | CPU master + GPU + 2 CPU(9) | **8.44690897** | **1,657,411** | 1.08549 | 1.12244 | 0.56358 | 0.999257 | PASS WITH WARNINGS |
| Hybrid | GPU master + CPU(19) | 8.79413016 | 1,591,971 | 1.14406 | 1.18177 | 0.65360 | 0.999176 | PASS WITH WARNINGS |
| Hybrid | CPU-only master/worker | **5.54067876** | **2,526,766** | **0.96899** | **1.01050** | 0.65267 | **0.999398** | PASS |
| Hybrid | CPU master + GPU, allreduce | 8.78268223 | 1,594,046 | 1.58462 | 1.57590 | 1.05425 | 0.998535 | PASS WITH WARNINGS |

All models beat the training-mean baseline (test RMSE 41.18066) by a wide
margin. Differences between parallel metrics are expected because batch
partitioning changes float32 summation order and therefore the 1,000-epoch
optimization trajectory.

## CPU scaling

Using each implementation's one-worker time as its reference:

| Workers | OpenMP speedup | OpenMP efficiency | MPI speedup | MPI efficiency |
|---:|---:|---:|---:|---:|
| 2 | 1.550 | 77.5% | 1.604 | 80.2% |
| 4 | 2.065 | 51.6% | 2.147 | 53.7% |
| 8 | 2.163 | 27.0% | NOT TESTED | NOT TESTED |
| 16 | 1.412 | 8.8% | NOT TESTED | NOT TESTED |
| 20 | 1.330 | 6.6% | NOT TESTED | NOT TESTED |

## Correctness and diagnostics

- Dataset schema, finiteness, and 14,000/3,000/3,000 stored splits: PASS.
- Strict builds for sequential, OpenMP, MPI, CUDA, and hybrid: PASS; nvcc
  emits Eigen-header warnings, so CUDA-bearing rows are PASS WITH WARNINGS.
- Sequential/OpenMP/MPI/CUDA smoke tests on the 20,000-row dataset: PASS.
- Hybrid sample assignment: exactly 256 unique contiguous slots: PASS.
- GPU forward and all six reduced-gradient groups: relative L2 below `3e-7`:
  PASS.
- Replica synchronization and three-run reproducibility: zero maximum
  parameter spread/difference: PASS.
- ASan/UBSan sequential smoke: PASS.
- Compute Sanitizer memcheck/initcheck: zero errors; racecheck: zero hazards:
  PASS.
- One GPU was visible, so multi-GPU mode remains NOT TESTED.

## Resource mapping and hybrid allocation

Measured system: Ubuntu 24.04.4, 20 logical CPUs, Open MPI 4.1.6, CUDA 13.0,
and one NVIDIA GeForce RTX 3070 (compute capability 8.6, 8 GiB). The four-rank
mixed run mapped rank 0 to the CPU master, rank 1 to GPU 0, and ranks 2/3 to
nine disjoint CPU threads each. A normal batch allocated
`master=0, GPU=97, CPU=92, CPU=67` after calibration.

The short autotuner also selected `cpu-only,np=2,threads=19`; the complete
1,000-epoch timings confirmed that this was the fastest hybrid-executable
topology. It still did not beat the direct OpenMP or MPI implementations.

## Reproduce

```bash
make all

DATASET=$PWD/projectile_dataset_n20000_data1_split1_dt0p02.csv

OMP_PROC_BIND=close OMP_PLACES=cores \
  ./mlp_openmp "$DATASET" 20 --batch-size 256 --epochs 1000

mpirun --bind-to core --map-by core -np 4 ./mlp_mpi \
  "$DATASET" --batch-size 256 --epochs 1000

./mlp_cuda "$DATASET" --batch-size 256 --epochs 1000

DATASET="$DATASET" BATCH_SIZE=256 EPOCHS=1000 \
  scripts/run_hybrid_autotune.sh

mpirun --bind-to none -np 4 ./hybrid_mpi_cuda "$DATASET" \
  --batch-size 256 --epochs 1000 --topology cpu-master --cpu-threads 9 \
  --load-balance calibrated --comm reduce-bcast --overlap on
```

Exact results are in `benchmark_results_n20000_e1000.csv` and
`accuracy_results_n20000_e1000.csv`.
