CXX ?= g++
MPICXX ?= mpicxx
NVCC ?= nvcc
EIGEN_INCLUDE ?= /usr/include/eigen3
CUDA_ARCH ?= sm_86
CUDA_HOME ?= /usr/local/cuda
MPI_HOME ?= /usr
CUDA_VERSION = $(shell $(NVCC) --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
CUDA_LIBDIR ?= $(shell dirname "$$(find $(CUDA_HOME) /opt/nvidia/hpc_sdk -path '*/cuda/$(CUDA_VERSION)/*' -name libcudart.so -print -quit 2>/dev/null)")
CUBLAS_LIBDIR ?= $(shell dirname "$$(find $(CUDA_HOME) /opt/nvidia/hpc_sdk -path '*/math_libs/$(CUDA_VERSION)/*' -name libcublas.so -print -quit 2>/dev/null)")
CXXFLAGS ?= -O3 -march=native
NVCCFLAGS ?= -O3
OMPFLAGS ?= -fopenmp
LDFLAGS ?=

WARNFLAGS = -Wall -Wextra -Wpedantic

.PHONY: all sequential openmp mpi cuda hybrid hybrid_debug hybrid_smoke \
	hybrid_validate hybrid_benchmark hybrid_autotune clean

all: sequential openmp mpi cuda hybrid

sequential: mlp_eigen_sequential

openmp: mlp_openmp

mpi: mlp_mpi

cuda: mlp_cuda

hybrid: hybrid_mpi_cuda

HYBRID_OBJECTS = hybrid_mpi_cuda_main.o hybrid_resource_discovery.o \
	hybrid_load_balancer.o hybrid_validation.o hybrid_mpi_cuda_gpu.o
HYBRID_CXXFLAGS = $(filter-out -march=native,$(CXXFLAGS))

hybrid_mpi_cuda: $(HYBRID_OBJECTS)
	$(MPICXX) $(HYBRID_CXXFLAGS) $(OMPFLAGS) $(HYBRID_OBJECTS) \
		-L$(CUDA_LIBDIR) -L$(CUBLAS_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) \
		-Wl,-rpath,$(CUBLAS_LIBDIR) -lcublas -lcudart $(LDFLAGS) -o $@

hybrid_mpi_cuda_main.o: hybrid_mpi_cuda_main.cpp mlp_common.hpp \
	hybrid_mpi_cuda_gpu.hpp hybrid_resource_discovery.hpp \
	hybrid_worker_roles.hpp hybrid_load_balancer.hpp hybrid_validation.hpp
	$(MPICXX) -std=c++17 $(HYBRID_CXXFLAGS) $(OMPFLAGS) -isystem $(EIGEN_INCLUDE) \
		$(WARNFLAGS) -Wconversion -Wshadow -c $< -o $@

hybrid_resource_discovery.o: hybrid_resource_discovery.cpp \
	hybrid_resource_discovery.hpp hybrid_mpi_cuda_gpu.hpp
	$(MPICXX) -std=c++17 $(HYBRID_CXXFLAGS) $(OMPFLAGS) -isystem $(EIGEN_INCLUDE) \
		$(WARNFLAGS) -Wconversion -Wshadow -c $< -o $@

hybrid_load_balancer.o: hybrid_load_balancer.cpp hybrid_load_balancer.hpp
	$(CXX) -std=c++17 $(HYBRID_CXXFLAGS) $(WARNFLAGS) -Wconversion -Wshadow -c $< -o $@

hybrid_validation.o: hybrid_validation.cpp hybrid_validation.hpp mlp_common.hpp
	$(CXX) -std=c++17 $(HYBRID_CXXFLAGS) -isystem $(EIGEN_INCLUDE) $(WARNFLAGS) \
		-Wconversion -Wshadow -c $< -o $@

hybrid_mpi_cuda_gpu.o: hybrid_mpi_cuda_gpu.cu hybrid_mpi_cuda_gpu.hpp \
	mlp_cuda.cu mlp_common.hpp
	$(NVCC) -std=c++17 $(NVCCFLAGS) -arch=$(CUDA_ARCH) -isystem $(EIGEN_INCLUDE) \
		-c $< -o $@

hybrid_debug:
	$(MAKE) clean
	$(MAKE) hybrid CXXFLAGS='-O0 -g' NVCCFLAGS='-O0 -g -G'

hybrid_smoke: hybrid
	scripts/run_hybrid_smoke.sh

hybrid_validate: hybrid
	scripts/run_hybrid_smoke.sh --validate-only

hybrid_benchmark: all hybrid
	scripts/benchmark_all_modes.sh

hybrid_autotune: hybrid
	scripts/run_hybrid_autotune.sh

mlp_eigen_sequential: mlp_eigen_sequential.cpp
	$(CXX) -std=c++17 $(CXXFLAGS) -isystem $(EIGEN_INCLUDE) $(WARNFLAGS) \
		mlp_eigen_sequential.cpp $(LDFLAGS) -o $@

mlp_openmp: mlp_openmp.cpp mlp_common.hpp
	$(CXX) -std=c++17 $(CXXFLAGS) $(OMPFLAGS) \
		-isystem $(EIGEN_INCLUDE) $(WARNFLAGS) mlp_openmp.cpp $(LDFLAGS) -o $@

mlp_mpi: mlp_mpi.cpp mlp_common.hpp
	$(MPICXX) -std=c++17 $(CXXFLAGS) \
		-isystem $(EIGEN_INCLUDE) $(WARNFLAGS) mlp_mpi.cpp $(LDFLAGS) -o $@

mlp_cuda: mlp_cuda.cu mlp_common.hpp
	$(NVCC) -std=c++17 $(filter-out -march=native,$(CXXFLAGS)) \
		-arch=$(CUDA_ARCH) -isystem $(EIGEN_INCLUDE) mlp_cuda.cu -lcublas $(LDFLAGS) -o $@

clean:
	rm -f mlp_eigen_sequential mlp_openmp mlp_mpi mlp_cuda \
		mlp_eigen_sequential_debug mlp_eigen_sequential_asan \
		mlp_openmp_debug mlp_openmp_tsan mlp_mpi_debug mlp_cuda_debug \
		hybrid_mpi_cuda $(HYBRID_OBJECTS)
