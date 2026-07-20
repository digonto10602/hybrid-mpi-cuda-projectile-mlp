# Batch-256 benchmark results

Status: **PASS WITH WARNINGS**

This is a separate scaling experiment. Only the global batch changes from 64
to 256. The stored 2,800/600/600 split, `3->64->64->1` float32 model, 300
epochs, MSE, ReLU, Adam settings, standardization, warm-up, and three-run median
methodology are unchanged. Batch 64 remains the default.

## Production results

| Implementation | Workers | Median train (s) | Samples/s | Val RMSE | Test RMSE | Test MAE | Test R2 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Sequential Eigen | 1 thread | 0.46511502 | 1,806,004.89 | 6.19983 | 5.75997 | 2.49329 | 0.978018 | PASS |
| OpenMP | 2 threads | 0.18609917 | 4,513,722.39 | 6.18529 | 5.79862 | 2.49800 | 0.977722 | PASS |
| OpenMP | 4 threads | 0.13462192 | 6,239,697.12 | 6.19983 | 5.75997 | 2.49329 | 0.978018 | PASS |
| OpenMP | 8 threads | **0.12788462** | **6,568,420.82** | 6.14346 | 5.72867 | 2.47359 | 0.978256 | PASS |
| MPI CPU | 2 ranks | 0.18360475 | 4,575,044.91 | 6.18529 | 5.79862 | 2.49800 | 0.977722 | PASS |
| MPI CPU | 4 ranks | **0.13310524** | **6,310,795.78** | 6.16945 | 5.81016 | 2.52334 | 0.977633 | PASS |
| CUDA C++ | RTX 3070 | 0.26852771 | 3,128,168.80 | 6.88147 | 6.30641 | 3.18083 | 0.973649 | PASS WITH WARNINGS |
| Hybrid GPU-only | RTX 3070 | 0.52998501 | 1,584,950.49 | 6.13692 | 5.74747 | 2.43921 | 0.978113 | PASS WITH WARNINGS |
| Hybrid CPU master | GPU worker | 0.47638862 | 1,763,266.30 | 6.12719 | 5.73024 | 2.47699 | 0.978244 | PASS WITH WARNINGS |
| Hybrid CPU master | GPU + CPU(18) | 0.47755851 | 1,758,946.78 | 6.19983 | 5.75997 | 2.49329 | 0.978018 | PASS WITH WARNINGS |
| Hybrid CPU master | GPU + 2 CPU(9) | **0.46010461** | **1,825,671.76** | 6.15151 | 5.71626 | 2.42829 | 0.978350 | PASS WITH WARNINGS |
| Hybrid GPU master | GPU + CPU(19) | 0.52837465 | 1,589,781.03 | 6.18097 | 5.79233 | 2.50313 | 0.977770 | PASS WITH WARNINGS |
| Hybrid CPU-only | CPU(19) | **0.36244186** | **2,317,613.07** | 6.13239 | 5.70815 | 2.45959 | 0.978412 | PASS |

OpenMP with 8 threads is fastest overall. MPI with 4 ranks is close. The best
full hybrid-executable mode is CPU-only; the best true CPU+GPU mode is CPU
master + GPU + two 9-thread CPU workers. The short autotuner selected that
four-rank mixed topology, but the complete production timing favored CPU-only.

## Correctness and diagnostics

- Fixed-batch assignment: 256 unique contiguous samples, PASS.
- GPU forward and reduced-gradient relative L2 differences: below `2.5e-7`, PASS.
- Replica spread and three-run parameter difference: zero, PASS.
- ASan/UBSan sequential smoke: PASS.
- Compute Sanitizer memcheck/initcheck: zero errors, PASS.
- Compute Sanitizer racecheck: zero hazards, PASS.
- nvcc still emits Eigen-header constexpr warnings, hence `PASS WITH WARNINGS`.

The typical four-rank heterogeneous allocation was
`master=0, GPU=107, CPU=76, CPU=73`. Communication remained 90.2% of its
maximum-rank time. Batch 256 made this topology 3.98 times faster than batch 64,
but it still did not beat OpenMP or pure MPI.

## Accuracy tradeoff

Batch 256 performs 11 Adam updates per epoch instead of 44 for batch 64. This
cuts synchronization and launch overhead, but provides four times fewer updates
over a fixed 300 epochs. The best batch-256 test RMSE is about 5.71 versus about
4.66 for batch 64. It is faster but less accurate under the fixed epoch count.

## Reproduce

```bash
make all

OMP_PROC_BIND=close OMP_PLACES=cores \
./mlp_openmp projectile_dataset_n4000_data1_split1_dt0p02.csv 20 \
  --batch-size 256

mpirun --bind-to core --map-by core -np 4 ./mlp_mpi \
  projectile_dataset_n4000_data1_split1_dt0p02.csv --batch-size 256

./mlp_cuda projectile_dataset_n4000_data1_split1_dt0p02.csv --batch-size 256

BATCH_SIZE=256 scripts/run_hybrid_autotune.sh

mpirun --bind-to none -np 4 ./hybrid_mpi_cuda \
  projectile_dataset_n4000_data1_split1_dt0p02.csv \
  --batch-size 256 --topology cpu-master --cpu-threads 9 \
  --load-balance calibrated --comm reduce-bcast --overlap on
```

Exact data is in `benchmark_results_batch256.csv` and
`accuracy_results_batch256.csv`.
