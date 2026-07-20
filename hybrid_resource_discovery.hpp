#pragma once

#include "hybrid_mpi_cuda_gpu.hpp"

#include <string>
#include <vector>

namespace hybrid {

struct ResourceInfo {
    int rank = 0;
    int world_size = 1;
    int local_rank = 0;
    int local_size = 1;
    std::string hostname;
    std::string cpu_affinity;
    int visible_logical_cpus = 1;
    int physical_cores = 0;
    int openmp_threads = 1;
    std::vector<GpuInfo> gpus;
    int assigned_gpu = -1;
    std::string cuda_visible_devices;
    std::string scheduler_allocation;
    int mpi_thread_support = 0;
};

ResourceInfo discover_resources(
    int rank,
    int world_size,
    int mpi_thread_support,
    int openmp_threads
);

std::string resource_csv_header();
std::string resource_csv_row(const ResourceInfo& info, const std::string& role);

}  // namespace hybrid
