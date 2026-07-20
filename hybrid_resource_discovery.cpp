#include "hybrid_resource_discovery.hpp"

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <sched.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace hybrid {
namespace {

std::string environment(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? "" : value;
}

std::string affinity(int& count) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        count = 1;
        return "unavailable";
    }
    count = CPU_COUNT(&mask);
    std::ostringstream result;
    bool first = true;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) {
            if (!first) result << ':';
            result << cpu;
            first = false;
        }
    }
    return result.str();
}

int physical_cores() {
    std::ifstream input("/proc/cpuinfo");
    std::set<std::pair<int, int>> cores;
    std::string line;
    int physical = -1;
    int core = -1;
    while (std::getline(input, line)) {
        if (line.rfind("physical id", 0) == 0) physical = std::stoi(line.substr(line.find(':') + 1));
        if (line.rfind("core id", 0) == 0) core = std::stoi(line.substr(line.find(':') + 1));
        if (line.empty() && physical >= 0 && core >= 0) {
            cores.emplace(physical, core);
            physical = core = -1;
        }
    }
    return static_cast<int>(cores.size());
}

std::string csv_escape(const std::string& value) {
    std::string result = "\"";
    for (const char character : value) {
        if (character == '\"') result += '\"';
        result += character;
    }
    return result + "\"";
}

}  // namespace

ResourceInfo discover_resources(
    int rank,
    int world_size,
    int mpi_thread_support,
    int openmp_threads
) {
    MPI_Comm local = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Comm_split_type failed.");
    }
    int local_rank = 0;
    int local_size = 1;
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_size(local, &local_size);
    MPI_Comm_free(&local);

    char host[MPI_MAX_PROCESSOR_NAME]{};
    int host_length = 0;
    MPI_Get_processor_name(host, &host_length);
    int logical = 1;
    const std::string cpu_affinity = affinity(logical);

    std::string scheduler = environment("SLURM_JOB_CPUS_PER_NODE");
    if (scheduler.empty()) scheduler = environment("PBS_NP");
    if (scheduler.empty()) scheduler = environment("NSLOTS");
    if (scheduler.empty()) scheduler = "none";

    return ResourceInfo{
        rank,
        world_size,
        local_rank,
        local_size,
        std::string(host, static_cast<std::size_t>(host_length)),
        cpu_affinity,
        logical,
        physical_cores(),
        openmp_threads,
        discover_gpus(),
        -1,
        environment("CUDA_VISIBLE_DEVICES"),
        scheduler,
        mpi_thread_support
    };
}

std::string resource_csv_header() {
    return "rank,world_size,hostname,local_rank,local_size,cpu_affinity,logical_cpus,physical_cores,omp_threads,visible_gpus,assigned_gpu,gpu_model,gpu_capability,gpu_memory_bytes,scheduler_allocation,cuda_visible_devices,mpi_thread_support,role";
}

std::string resource_csv_row(const ResourceInfo& info, const std::string& role) {
    std::string gpu_name;
    std::string capability;
    std::size_t memory = 0;
    if (info.assigned_gpu >= 0 && static_cast<std::size_t>(info.assigned_gpu) < info.gpus.size()) {
        const GpuInfo& gpu = info.gpus[static_cast<std::size_t>(info.assigned_gpu)];
        gpu_name = gpu.name;
        capability = std::to_string(gpu.compute_major) + "." + std::to_string(gpu.compute_minor);
        memory = gpu.memory_bytes;
    }
    std::ostringstream row;
    row << info.rank << ',' << info.world_size << ',' << csv_escape(info.hostname) << ','
        << info.local_rank << ',' << info.local_size << ',' << csv_escape(info.cpu_affinity) << ','
        << info.visible_logical_cpus << ',' << info.physical_cores << ',' << info.openmp_threads << ','
        << info.gpus.size() << ',' << info.assigned_gpu << ',' << csv_escape(gpu_name) << ','
        << csv_escape(capability) << ',' << memory << ',' << csv_escape(info.scheduler_allocation) << ','
        << csv_escape(info.cuda_visible_devices) << ',' << info.mpi_thread_support << ',' << role;
    return row.str();
}

}  // namespace hybrid
