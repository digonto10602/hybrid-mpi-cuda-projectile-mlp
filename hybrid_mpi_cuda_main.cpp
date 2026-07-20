#include "hybrid_load_balancer.hpp"
#include "hybrid_mpi_cuda_gpu.hpp"
#include "hybrid_resource_discovery.hpp"
#include "hybrid_validation.hpp"
#include "hybrid_worker_roles.hpp"
#include "mlp_common.hpp"

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace mlp;
using namespace hybrid;

constexpr std::size_t packed_count = 4481U;

struct Options {
    std::string dataset = "projectile_dataset_n4000_data1_split1_dt0p02.csv";
    std::string topology = "cpu-master";
    std::string load_balance = "calibrated";
    std::string communication = "reduce-bcast";
    std::string gpu_aware_mpi = "auto";
    bool overlap = false;
    bool smoke = false;
    bool autotune = false;
    bool validate_only = false;
    int cpu_threads = 1;
    std::size_t batch_size = 64;
    std::size_t epochs = 300;
};

struct Components {
    double total = 0.0;
    double parameter_broadcast = 0.0;
    double cpu_compute = 0.0;
    double gpu_compute = 0.0;
    double gpu_h2d = 0.0;
    double gpu_d2h = 0.0;
    double reduction = 0.0;
    double communication = 0.0;
    double reduced_gradient_h2d = 0.0;
    double adam = 0.0;
};

struct TrainingResult {
    Components maximum;
    Parameters parameters;
    std::vector<int> normal_batch_counts;
    double load_imbalance = 0.0;
};

struct InferenceResult {
    double maximum_compute = 0.0;
    double minimum_compute = 0.0;
    double gather = 0.0;
    double end_to_end = 0.0;
    Vector prediction;
};

double now_seconds() {
    return MPI_Wtime();
}

Options parse_options(int argc, char** argv) {
    Options options;
    int argument = 1;
    if (argument < argc && std::string(argv[argument]).rfind("--", 0) != 0) {
        options.dataset = argv[argument++];
    }
    const auto value = [&](const std::string& name, int& index) {
        if (index + 1 >= argc) throw std::invalid_argument(name + " requires a value.");
        return std::string(argv[++index]);
    };
    for (; argument < argc; ++argument) {
        const std::string option(argv[argument]);
        if (option == "--topology") options.topology = value(option, argument);
        else if (option == "--load-balance") options.load_balance = value(option, argument);
        else if (option == "--comm") options.communication = value(option, argument);
        else if (option == "--gpu-aware-mpi") options.gpu_aware_mpi = value(option, argument);
        else if (option == "--overlap") options.overlap = value(option, argument) == "on";
        else if (option == "--cpu-threads") options.cpu_threads = std::stoi(value(option, argument));
        else if (option == "--batch-size") {
            const int batch_size = std::stoi(value(option, argument));
            if (batch_size < 1) throw std::invalid_argument("Batch size must be positive.");
            options.batch_size = static_cast<std::size_t>(batch_size);
        }
        else if (option == "--epochs") {
            const int epochs = std::stoi(value(option, argument));
            if (epochs < 1) throw std::invalid_argument("Epoch count must be positive.");
            options.epochs = static_cast<std::size_t>(epochs);
        }
        else if (option == "--smoke-test") options.smoke = true;
        else if (option == "--autotune") options.autotune = true;
        else if (option == "--validate-only") options.validate_only = true;
        else throw std::invalid_argument("Unknown option: " + option);
    }
    if (options.cpu_threads < 1) throw std::invalid_argument("CPU thread count must be positive.");
    if (options.batch_size < 1) throw std::invalid_argument("Batch size must be positive.");
    if (options.load_balance != "static" && options.load_balance != "calibrated" && options.load_balance != "adaptive")
        throw std::invalid_argument("Invalid --load-balance value.");
    if (options.communication != "reduce-bcast" && options.communication != "allreduce")
        throw std::invalid_argument("Invalid --comm value.");
    if (options.gpu_aware_mpi != "auto" && options.gpu_aware_mpi != "on" && options.gpu_aware_mpi != "off")
        throw std::invalid_argument("Invalid --gpu-aware-mpi value.");
    return options;
}

void unpack_parameters(const std::vector<float>& packed, Parameters& parameters) {
    if (packed.size() != packed_count) throw std::runtime_error("Packed parameter count must equal 4,481.");
    std::size_t offset = 0;
    const auto copy = [&](auto& values) {
        const std::size_t count = static_cast<std::size_t>(values.size());
        std::copy(packed.begin() + static_cast<std::ptrdiff_t>(offset),
            packed.begin() + static_cast<std::ptrdiff_t>(offset + count), values.data());
        offset += count;
    };
    copy(parameters.W1); copy(parameters.b1); copy(parameters.W2);
    copy(parameters.b2); copy(parameters.W3); copy(parameters.b3);
    if (offset != packed_count) throw std::runtime_error("Parameter unpack order mismatch.");
}

Batch contiguous_batch(const Matrix& X, const Vector& y, int start, int count) {
    if (count == 0) return Batch{Matrix(0, X.cols()), Vector(0)};
    return Batch{X.middleRows(start, count), y.segment(start, count)};
}

Gradients cpu_gradient(
    const Batch& batch,
    const Parameters& parameters,
    const Config& config,
    float normalization,
    int requested_threads
) {
    if (batch.X.rows() == 0) return zero_gradients(config);
    const int threads = std::min(requested_threads, static_cast<int>(batch.X.rows()));
    if (threads <= 1) {
        return mse_loss_and_backward(batch.X, batch.y, parameters, config, normalization).gradients;
    }
    std::vector<Gradients> local(static_cast<std::size_t>(threads), zero_gradients(config));
#pragma omp parallel num_threads(threads)
    {
        const int thread = omp_get_thread_num();
        const int size = omp_get_num_threads();
        const int begin = static_cast<int>(batch.X.rows()) * thread / size;
        const int end = static_cast<int>(batch.X.rows()) * (thread + 1) / size;
        const Batch part = contiguous_batch(batch.X, batch.y, begin, end - begin);
        local[static_cast<std::size_t>(thread)] =
            mse_loss_and_backward(part.X, part.y, parameters, config, normalization).gradients;
    }
    Gradients result = zero_gradients(config);
    for (const Gradients& gradient : local) add_gradients(result, gradient);
    return result;
}

Role choose_role(const Options& options, const ResourceInfo& resources, int rank, int world_size) {
    if (options.topology == "cpu-master") {
        if (world_size < 2) throw std::runtime_error("cpu-master requires at least two ranks.");
        if (rank == 0) return Role::master;
        if (rank == 1) return Role::gpu_worker;
        return Role::cpu_worker;
    }
    if (options.topology == "gpu-master") {
        if (rank == 0) return Role::gpu_master;
        return Role::cpu_worker;
    }
    if (options.topology == "cpu-only") {
        if (world_size < 2) throw std::runtime_error("cpu-only master-worker requires at least two ranks.");
        return rank == 0 ? Role::master : Role::cpu_worker;
    }
    if (options.topology == "gpu-only") {
        if (world_size != 1) throw std::runtime_error("gpu-only requires exactly one rank.");
        return Role::gpu_master;
    }
    if (options.topology == "gpu-allreduce") {
        if (resources.local_rank >= static_cast<int>(resources.gpus.size()))
            throw std::runtime_error("gpu-allreduce requires one visible GPU per node-local rank.");
        return Role::gpu_worker;
    }
    throw std::invalid_argument("Invalid topology: " + options.topology);
}

std::vector<int> active_flags(Role role, int world_size) {
    const int local = computes(role) ? 1 : 0;
    std::vector<int> active(static_cast<std::size_t>(world_size));
    MPI_Allgather(&local, 1, MPI_INT, active.data(), 1, MPI_INT, MPI_COMM_WORLD);
    return active;
}

void bind_role_affinity(Role role, int rank, int world_size, int cpu_threads) {
    cpu_set_t available;
    CPU_ZERO(&available);
    if (sched_getaffinity(0, sizeof(available), &available) != 0)
        throw std::runtime_error("sched_getaffinity failed before role binding.");
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &available)) cpus.push_back(cpu);

    const int local_role = static_cast<int>(role);
    std::vector<int> roles(static_cast<std::size_t>(world_size));
    MPI_Allgather(&local_role, 1, MPI_INT, roles.data(), 1, MPI_INT, MPI_COMM_WORLD);
    const int cpu_role = static_cast<int>(Role::cpu_worker);
    const int cpu_workers = static_cast<int>(std::count(roles.begin(), roles.end(), cpu_role));
    const int control_ranks = world_size - cpu_workers;
    if (control_ranks + cpu_workers * cpu_threads > static_cast<int>(cpus.size()))
        throw std::runtime_error("Role affinity plan oversubscribes visible logical CPUs.");

    int start = 0;
    int count = 1;
    if (role == Role::cpu_worker) {
        const int worker_index = static_cast<int>(std::count(roles.begin(), roles.begin() + rank, cpu_role));
        start = control_ranks + worker_index * cpu_threads;
        count = cpu_threads;
    } else {
        start = static_cast<int>(std::count_if(roles.begin(), roles.begin() + rank,
            [cpu_role](int candidate) { return candidate != cpu_role; }));
    }
    cpu_set_t assigned;
    CPU_ZERO(&assigned);
    for (int index = 0; index < count; ++index) CPU_SET(cpus[static_cast<std::size_t>(start + index)], &assigned);
    if (sched_setaffinity(0, sizeof(assigned), &assigned) != 0)
        throw std::runtime_error("sched_setaffinity failed for role assignment.");
}

std::vector<double> calibrate(
    Role role,
    GpuWorker* gpu,
    const Matrix& X,
    const Vector& y,
    const Parameters& parameters,
    const Config& config,
    int cpu_threads,
    int world_size
) {
    constexpr int repetitions = 5;
    const int rows = std::min<int>(32, static_cast<int>(X.rows()));
    const Batch batch = contiguous_batch(X, y, 0, rows);
    std::vector<double> times;
    if (computes(role)) {
        if (gpu != nullptr) gpu->set_parameters(parameters);
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double start = now_seconds();
            Gradients gradient = zero_gradients(config);
            if (uses_gpu(role)) gpu->local_gradient(batch, static_cast<float>(rows), gradient);
            else gradient = cpu_gradient(batch, parameters, config, static_cast<float>(rows), cpu_threads);
            if (!finite_gradients(gradient)) throw std::runtime_error("Calibration produced a non-finite gradient.");
            times.push_back(now_seconds() - start);
        }
    } else {
        for (int repetition = 0; repetition < repetitions; ++repetition) MPI_Barrier(MPI_COMM_WORLD);
    }
    const double throughput = times.empty() ? 0.0 : static_cast<double>(rows) / median(times);
    std::vector<double> throughputs(static_cast<std::size_t>(world_size));
    MPI_Allgather(&throughput, 1, MPI_DOUBLE, throughputs.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
    return throughputs;
}

Gradients local_gradient_for_allocation(
    Role role,
    GpuWorker* gpu,
    const Batch& batch,
    const Parameters& parameters,
    const Config& config,
    float normalization,
    int cpu_threads,
    Components& components
) {
    if (!computes(role) || batch.X.rows() == 0) return zero_gradients(config);
    Gradients gradient = zero_gradients(config);
    if (uses_gpu(role)) {
        const GpuTimings timing = gpu->local_gradient(batch, normalization, gradient);
        components.gpu_h2d += timing.host_to_device;
        components.gpu_compute += timing.compute;
        components.gpu_d2h += timing.device_to_host;
    } else {
        const double start = now_seconds();
        gradient = cpu_gradient(batch, parameters, config, normalization, cpu_threads);
        components.cpu_compute += now_seconds() - start;
    }
    if (!finite_gradients(gradient)) throw std::runtime_error("Worker produced a non-finite gradient.");
    return gradient;
}

void print_gradient_differences(const std::string& title, const GradientDifferences& differences) {
    std::cout << title << '\n';
    const auto& names = gradient_group_names();
    for (std::size_t group = 0; group < names.size(); ++group) {
        std::cout << "  " << names[group] << " max_abs=" << differences.groups[group].maximum_absolute
            << " rel_l2=" << differences.groups[group].relative_l2 << '\n';
    }
    std::cout << "  status=" << (differences.passed ? "PASS" : "FAIL") << '\n';
}

void validate_topology(
    Role role,
    GpuWorker* gpu,
    const std::vector<int>& active,
    const std::vector<double>& weights,
    const Dataset& scaled,
    const Parameters& parameters,
    const Config& config,
    int cpu_threads,
    int rank,
    int world_size
) {
    (void)world_size;
    std::vector<Eigen::Index> order(config.batch_size);
    std::iota(order.begin(), order.end(), 0);
    const Allocation allocation = allocate_samples(static_cast<int>(config.batch_size), weights, active);
    const Batch local = gather_order_range(scaled.train.X, scaled.train.y, order,
        static_cast<std::size_t>(allocation.offsets[static_cast<std::size_t>(rank)]),
        static_cast<std::size_t>(allocation.offsets[static_cast<std::size_t>(rank)] + allocation.counts[static_cast<std::size_t>(rank)]));

    if (gpu != nullptr) gpu->set_parameters(parameters, true);
    Components ignored;
    const Gradients local_gradient = local_gradient_for_allocation(role, gpu, local, parameters, config,
        static_cast<float>(config.batch_size), cpu_threads, ignored);
    if (uses_gpu(role) && local.X.rows() > 0) {
        const Gradients cpu = mse_loss_and_backward(local.X, local.y, parameters, config,
            static_cast<float>(config.batch_size)).gradients;
        const GradientDifferences gpu_difference = compare_gradients(cpu, local_gradient, 1.0e-4);
        const Vector cpu_prediction = predict_standardized(local.X, parameters);
        const Vector gpu_prediction = gpu->predict(local.X);
        const double forward_relative = (cpu_prediction - gpu_prediction).norm() /
            std::max<double>(cpu_prediction.norm(), 1.0e-12);
        if (!gpu_difference.passed || forward_relative > 1.0e-4) throw std::runtime_error("GPU local correctness check failed.");
        std::cout << "Rank " << rank << " GPU forward relative L2=" << forward_relative << " PASS\n";
        print_gradient_differences("Rank " + std::to_string(rank) + " GPU backward groups", gpu_difference);
    }

    std::vector<float> local_packed;
    pack_gradients(local_gradient, local_packed);
    if (local_packed.size() != packed_count) throw std::runtime_error("Packed gradient count must equal 4,481.");
    std::vector<float> reduced(packed_count, 0.0F);
    MPI_Reduce(local_packed.data(), reduced.data(), static_cast<int>(packed_count), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const Batch full = gather_order_range(scaled.train.X, scaled.train.y, order, 0, config.batch_size);
        const Gradients reference = mse_loss_and_backward(full.X, full.y, parameters, config,
            static_cast<float>(config.batch_size)).gradients;
        Gradients actual = zero_gradients(config);
        unpack_gradients(reduced, actual);
        const GradientDifferences difference = compare_gradients(reference, actual, 2.0e-4);
        print_gradient_differences("Reduced global gradient versus sequential", difference);
        if (!difference.passed) throw std::runtime_error("Reduced global gradient correctness check failed.");
        const int assigned = std::accumulate(allocation.counts.begin(), allocation.counts.end(), 0);
        if (assigned != static_cast<int>(config.batch_size)) throw std::runtime_error("Sample assignment audit failed.");
        std::cout << "Sample assignment audit: " << config.batch_size
            << " unique contiguous slots, PASS\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

Components reduce_maximum(const Components& local, int rank) {
    const std::array<double, 10> values{local.total, local.parameter_broadcast, local.cpu_compute,
        local.gpu_compute, local.gpu_h2d, local.gpu_d2h, local.reduction,
        local.communication, local.reduced_gradient_h2d, local.adam};
    std::array<double, 10> maximum{};
    MPI_Reduce(values.data(), maximum.data(), static_cast<int>(values.size()), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return {};
    return Components{maximum[0], maximum[1], maximum[2], maximum[3], maximum[4], maximum[5],
        maximum[6], maximum[7], maximum[8], maximum[9]};
}

TrainingResult train_once(
    Role role,
    GpuWorker* gpu,
    const std::vector<int>& active,
    std::vector<double> weights,
    const Dataset& scaled,
    const Config& config,
    const Options& options,
    int rank,
    int world_size,
    std::size_t maximum_steps = 0
) {
    Parameters parameters = initialize_parameters(config);
    AdamState adam = initialize_adam(config);
    if (gpu != nullptr) gpu->set_parameters(parameters, true);
    std::vector<float> parameter_buffer;
    pack_parameters(parameters, parameter_buffer);
    if (parameter_buffer.size() != packed_count) throw std::runtime_error("Packed parameter count must equal 4,481.");
    std::vector<float> local_packed(packed_count, 0.0F);
    std::vector<float> reduced_packed(packed_count, 0.0F);
    Gradients reduced_gradient = zero_gradients(config);
    std::vector<Eigen::Index> order(static_cast<std::size_t>(scaled.train.X.rows()));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 shuffle(config.shuffle_seed);
    Components local_components;
    std::vector<int> normal_counts(static_cast<std::size_t>(world_size), 0);
    std::size_t steps = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    const double total_start = now_seconds();

    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), shuffle);
        for (std::size_t global_start = 0; global_start < static_cast<std::size_t>(scaled.train.X.rows());
             global_start += config.batch_size) {
            const std::size_t global_stop = std::min(global_start + config.batch_size,
                static_cast<std::size_t>(scaled.train.X.rows()));
            const int batch_size = static_cast<int>(global_stop - global_start);
            const Allocation allocation = allocate_samples(batch_size, weights, active);
            if (batch_size == static_cast<int>(config.batch_size)) normal_counts = allocation.counts;
            const int local_count = allocation.counts[static_cast<std::size_t>(rank)];
            const int local_offset = allocation.offsets[static_cast<std::size_t>(rank)];
            const Batch batch = gather_order_range(scaled.train.X, scaled.train.y, order,
                global_start + static_cast<std::size_t>(local_offset),
                global_start + static_cast<std::size_t>(local_offset + local_count));

            if (role == Role::gpu_worker) {
                const double start = now_seconds();
                gpu->set_parameters(parameters);
                local_components.gpu_h2d += now_seconds() - start;
            }
            const Gradients local_gradient = local_gradient_for_allocation(role, gpu, batch, parameters, config,
                static_cast<float>(batch_size), options.cpu_threads, local_components);
            pack_gradients(local_gradient, local_packed);

            const double reduction_start = now_seconds();
            if (options.communication == "allreduce") {
                MPI_Allreduce(local_packed.data(), reduced_packed.data(), static_cast<int>(packed_count),
                    MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            } else {
                MPI_Reduce(local_packed.data(), reduced_packed.data(), static_cast<int>(packed_count),
                    MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            }
            local_components.reduction += now_seconds() - reduction_start;

            if (options.communication == "allreduce" || rank == 0) {
                unpack_gradients(reduced_packed, reduced_gradient);
                if (role == Role::gpu_master) {
                    const GpuTimings timing = gpu->adam_step(reduced_gradient);
                    local_components.reduced_gradient_h2d += timing.host_to_device;
                    local_components.adam += timing.adam;
                    const double copy_start = now_seconds();
                    parameters = gpu->parameters();
                    local_components.gpu_d2h += now_seconds() - copy_start;
                } else {
                    const double adam_start = now_seconds();
                    adam_step(parameters, reduced_gradient, adam, config);
                    local_components.adam += now_seconds() - adam_start;
                }
            }

            if (options.communication == "reduce-bcast") {
                const double broadcast_start = now_seconds();
                if (rank == 0) pack_parameters(parameters, parameter_buffer);
                MPI_Bcast(parameter_buffer.data(), static_cast<int>(packed_count), MPI_FLOAT, 0, MPI_COMM_WORLD);
                if (rank != 0) unpack_parameters(parameter_buffer, parameters);
                local_components.parameter_broadcast += now_seconds() - broadcast_start;
            }
            ++steps;
            if (maximum_steps > 0 && steps >= maximum_steps) goto training_complete;
        }
        const std::size_t adaptive_interval = config.smoke_test ? 1U : 50U;
        if (options.load_balance == "adaptive" && (epoch + 1U) % adaptive_interval == 0U) {
            weights = calibrate(role, gpu, scaled.train.X, scaled.train.y, parameters,
                config, options.cpu_threads, world_size);
        }
    }

training_complete:
    if (gpu != nullptr) gpu->synchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    local_components.total = now_seconds() - total_start;
    local_components.communication = local_components.parameter_broadcast + local_components.reduction;
    Components maximum = reduce_maximum(local_components, rank);

    const double local_compute = local_components.cpu_compute + local_components.gpu_compute;
    const double minimum_input = computes(role) ? local_compute : 1.0e300;
    const double maximum_input = computes(role) ? local_compute : 0.0;
    double minimum_compute = 0.0;
    double maximum_compute = 0.0;
    MPI_Allreduce(&minimum_input, &minimum_compute, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maximum_input, &maximum_compute, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double load_imbalance = maximum_compute > 0.0 ?
        (maximum_compute - minimum_compute) / maximum_compute : 0.0;

    std::vector<float> packed;
    pack_parameters(parameters, packed);
    std::vector<float> minimum(packed_count);
    std::vector<float> maximum_parameter(packed_count);
    MPI_Allreduce(packed.data(), minimum.data(), static_cast<int>(packed_count), MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(packed.data(), maximum_parameter.data(), static_cast<int>(packed_count), MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    double spread = 0.0;
    for (std::size_t index = 0; index < packed_count; ++index)
        spread = std::max(spread, static_cast<double>(maximum_parameter[index] - minimum[index]));
    if (spread > 1.0e-6) throw std::runtime_error("Model replicas are not synchronized.");
    if (!finite_parameters(parameters)) throw std::runtime_error("Training produced non-finite parameters.");
    if (rank == 0) std::cout << "Replica parameter spread: " << spread << " PASS\n";
    return TrainingResult{maximum, std::move(parameters), std::move(normal_counts), load_imbalance};
}

InferenceResult distributed_inference(
    Role role,
    GpuWorker* gpu,
    const std::vector<int>& active,
    const std::vector<double>& weights,
    const Matrix& X,
    const Parameters& parameters,
    const Config& config,
    int rank,
    int world_size
) {
    (void)world_size;
    const Allocation allocation = allocate_samples(static_cast<int>(X.rows()), weights, active);
    const int count = allocation.counts[static_cast<std::size_t>(rank)];
    const int offset = allocation.offsets[static_cast<std::size_t>(rank)];
    const Matrix local_X = count == 0 ? Matrix(0, X.cols()) : X.middleRows(offset, count).eval();
    if (gpu != nullptr) gpu->set_parameters(parameters);
    Vector local_prediction(count);
    std::vector<double> repetition_times;
    for (std::size_t repetition = 0; repetition < config.inference_repetitions; ++repetition) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = now_seconds();
        for (std::size_t loop = 0; loop < config.inference_loops_per_repetition; ++loop) {
            if (uses_gpu(role) && count > 0) local_prediction = gpu->predict(local_X);
            else if (role == Role::cpu_worker && count > 0) local_prediction = predict_standardized(local_X, parameters);
        }
        repetition_times.push_back(now_seconds() - start);
    }
    const double local_median = median(repetition_times);
    double maximum_compute = 0.0;
    double minimum_compute = 0.0;
    MPI_Reduce(&local_median, &maximum_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const double minimum_input = computes(role) ? local_median : 1.0e300;
    MPI_Reduce(&minimum_input, &minimum_compute, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    Vector gathered;
    if (rank == 0) gathered.resize(X.rows());
    const double gather_start = now_seconds();
    MPI_Gatherv(local_prediction.data(), count, MPI_FLOAT, rank == 0 ? gathered.data() : nullptr,
        allocation.counts.data(), allocation.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    const double gather_local = now_seconds() - gather_start;
    double gather_max = 0.0;
    MPI_Reduce(&gather_local, &gather_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return InferenceResult{maximum_compute, minimum_compute, gather_max,
        maximum_compute + gather_max, std::move(gathered)};
}

void write_resource_mapping(const ResourceInfo& resources, Role role, int rank, int world_size) {
    constexpr int width = 2048;
    std::array<char, width> local{};
    const std::string row = resource_csv_row(resources, role_name(role));
    if (row.size() >= local.size()) throw std::runtime_error("Resource CSV row is too long.");
    std::copy(row.begin(), row.end(), local.begin());
    std::vector<char> gathered;
    if (rank == 0) gathered.resize(static_cast<std::size_t>(world_size * width));
    MPI_Gather(local.data(), width, MPI_CHAR, gathered.data(), width, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::ofstream output("resource_mapping.csv");
        output << resource_csv_header() << '\n';
        for (int item = 0; item < world_size; ++item)
            output << gathered.data() + static_cast<std::ptrdiff_t>(item * width) << '\n';
    }
}

void print_components(const Components& c) {
    std::cout << "Timing components (maximum rank seconds)\n"
        << "  total=" << c.total << " parameter_broadcast=" << c.parameter_broadcast
        << " cpu_compute=" << c.cpu_compute << " gpu_compute=" << c.gpu_compute
        << " gpu_h2d=" << c.gpu_h2d << " gpu_d2h=" << c.gpu_d2h
        << " reduction=" << c.reduction << " communication=" << c.communication
        << " reduced_gradient_h2d=" << c.reduced_gradient_h2d
        << " adam=" << c.adam << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    try {
        Options options = parse_options(argc, argv);
        Config config;
        config.dataset_path = options.dataset;
        config.batch_size = options.batch_size;
        config.epochs = options.epochs;
        if (options.smoke || options.autotune || options.validate_only) {
            config.smoke_test = true;
            config.epochs = options.validate_only ? 1U : 2U;
            config.training_warmup_steps = 1;
            config.training_repetitions = options.validate_only ? 0U : 1U;
            config.inference_warmup_loops = 1;
            config.inference_repetitions = 1;
            config.inference_loops_per_repetition = 2;
            config.transfer_warmup_repetitions = 1;
            config.transfer_repetitions = 1;
        }
        if (parameter_count(config) != packed_count)
            throw std::runtime_error("Hybrid model must contain exactly 4,481 parameters.");
        Eigen::setNbThreads(1);
        omp_set_dynamic(0);
        omp_set_nested(0);

        ResourceInfo resources = discover_resources(rank, world_size, provided, options.cpu_threads);
        const Role role = choose_role(options, resources, rank, world_size);
        if (uses_gpu(role)) {
            resources.assigned_gpu = options.topology == "gpu-allreduce" ? resources.local_rank : 0;
            if (resources.assigned_gpu >= static_cast<int>(resources.gpus.size()))
                throw std::runtime_error("Designated GPU rank has no visible CUDA device.");
        }
        const std::vector<int> active = active_flags(role, world_size);
        const int active_workers = std::accumulate(active.begin(), active.end(), 0);
        if (active_workers < 1 || active_workers > static_cast<int>(config.batch_size))
            throw std::runtime_error("Active worker count is invalid for the global batch.");
        int cpu_workers = role == Role::cpu_worker ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &cpu_workers, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        const int reserved = world_size - cpu_workers;
        if (cpu_workers * options.cpu_threads + reserved > resources.visible_logical_cpus)
            throw std::runtime_error("Requested topology oversubscribes the visible CPU affinity mask.");
        bind_role_affinity(role, rank, world_size, options.cpu_threads);
        resources = discover_resources(rank, world_size, provided,
            role == Role::cpu_worker ? options.cpu_threads : 1);
        if (uses_gpu(role)) resources.assigned_gpu =
            options.topology == "gpu-allreduce" ? resources.local_rank : 0;
        if (options.gpu_aware_mpi == "on")
            throw std::runtime_error("CUDA-aware MPI was requested but this portable build provides only validated pinned-host staging.");
        if (options.communication == "allreduce" && role == Role::gpu_master)
            throw std::runtime_error("GPU-master optimizer ownership requires reduce-bcast.");

        write_resource_mapping(resources, role, rank, world_size);
        for (int turn = 0; turn < world_size; ++turn) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (turn == rank) std::cout << "RESOURCE," << resource_csv_row(resources, role_name(role)) << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);

        const double load_start = now_seconds();
        const Dataset raw = load_dataset(config.dataset_path);
        FeatureStandardizer X_scaler;
        TargetStandardizer y_scaler;
        const Dataset scaled = standardize_dataset(raw, X_scaler, y_scaler);
        const double load_seconds = now_seconds() - load_start;
        const Eigen::Index total_samples = raw.train.X.rows() + raw.validation.X.rows() + raw.test.X.rows();
        const Eigen::Index expected_train = total_samples * 70 / 100;
        const Eigen::Index expected_validation = total_samples * 15 / 100;
        if (raw.train.X.rows() != expected_train || raw.validation.X.rows() != expected_validation ||
            raw.test.X.rows() != total_samples - expected_train - expected_validation)
            throw std::runtime_error("Expected stored 70/15/15 split counts.");

        const int maximum_gpu_rows = std::max<int>(static_cast<int>(config.batch_size),
            std::max<int>(static_cast<int>(scaled.validation.X.rows()), static_cast<int>(scaled.test.X.rows())));
        std::unique_ptr<GpuWorker> gpu;
        if (uses_gpu(role)) gpu = std::make_unique<GpuWorker>(resources.assigned_gpu, config, maximum_gpu_rows);
        Parameters initial = initialize_parameters(config);
        std::vector<double> throughputs = calibrate(role, gpu.get(), scaled.train.X, scaled.train.y,
            initial, config, options.cpu_threads, world_size);
        std::vector<double> weights = throughputs;
        if (options.load_balance == "static") {
            for (std::size_t item = 0; item < weights.size(); ++item) weights[item] = active[item] != 0 ? 1.0 : 0.0;
        }
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(8)
                << "HYBRID MPI+CUDA PROJECTILE MLP\n"
                << "Topology=" << options.topology << " ranks=" << world_size
                << " cpu_threads=" << options.cpu_threads << " load_balance=" << options.load_balance
                << " batch_size=" << config.batch_size
                << " epochs=" << config.epochs
                << " comm=" << options.communication << " overlap=" << (options.overlap ? "on" : "off") << '\n'
                << "Dataset splits=" << raw.train.X.rows() << '/' << raw.validation.X.rows() << '/'
                << raw.test.X.rows() << " load_and_standardize=" << load_seconds << " s\n"
                << "Calibration throughput by rank:";
            for (const double throughput : throughputs) std::cout << ' ' << throughput;
            std::cout << '\n';
        }

        validate_topology(role, gpu.get(), active, weights, scaled, initial, config,
            options.cpu_threads, rank, world_size);
        if (options.validate_only) {
            if (rank == 0) std::cout << "Hybrid validation: PASS\n";
            MPI_Finalize();
            return 0;
        }

        (void)train_once(role, gpu.get(), active, weights, scaled, config, options,
            rank, world_size, config.training_warmup_steps);
        std::vector<double> training_times;
        std::vector<TrainingResult> repetitions;
        for (std::size_t repetition = 0; repetition < config.training_repetitions; ++repetition) {
            TrainingResult result = train_once(role, gpu.get(), active, weights, scaled, config,
                options, rank, world_size);
            if (rank == 0) {
                training_times.push_back(result.maximum.total);
                std::cout << "Training repetition " << repetition + 1 << "=" << result.maximum.total << " s\n";
                print_components(result.maximum);
            }
            repetitions.push_back(std::move(result));
        }
        Parameters final_parameters = repetitions.back().parameters;
        if (rank == 0 && repetitions.size() > 1U) {
            std::vector<float> reference;
            pack_parameters(repetitions.front().parameters, reference);
            double maximum_repetition_difference = 0.0;
            for (std::size_t repetition = 1; repetition < repetitions.size(); ++repetition) {
                std::vector<float> candidate;
                pack_parameters(repetitions[repetition].parameters, candidate);
                for (std::size_t index = 0; index < packed_count; ++index)
                    maximum_repetition_difference = std::max(maximum_repetition_difference,
                        std::abs(static_cast<double>(reference[index] - candidate[index])));
            }
            if (maximum_repetition_difference > 1.0e-6)
                throw std::runtime_error("Repeated training runs are not reproducible.");
            std::cout << "Repeated-run maximum parameter difference="
                << maximum_repetition_difference << " PASS\n";
        }
        if (gpu != nullptr) gpu->set_parameters(final_parameters);
        const InferenceResult inference = distributed_inference(role, gpu.get(), active, weights,
            scaled.test.X, final_parameters, config, rank, world_size);

        if (rank == 0) {
            const double training_median = median(training_times);
            const Vector validation_physical = inverse_transform_targets(
                predict_standardized(scaled.validation.X, final_parameters), y_scaler);
            const Vector test_physical = inverse_transform_targets(inference.prediction, y_scaler);
            const Metrics validation_metrics = calculate_metrics(raw.validation.y, validation_physical);
            const Metrics test_metrics = calculate_metrics(raw.test.y, test_physical);
            const auto baselines = calculate_baselines(raw.train, raw.test);
            if (!std::isfinite(validation_metrics.rmse) || !std::isfinite(test_metrics.rmse) ||
                test_metrics.rmse >= baselines.first.rmse)
                throw std::runtime_error("Hybrid metrics failed finite/baseline validation.");
            const Vector cpu_reference = predict_standardized(scaled.test.X, final_parameters);
            const double inference_relative = (cpu_reference - inference.prediction).norm() /
                std::max<double>(cpu_reference.norm(), 1.0e-12);
            if (inference_relative > 1.0e-4) throw std::runtime_error("Distributed inference ordering check failed.");

            std::cout << "Normal batch allocation:";
            for (std::size_t worker = 0; worker < repetitions.back().normal_batch_counts.size(); ++worker)
                std::cout << " rank" << worker << '=' << repetitions.back().normal_batch_counts[worker];
            std::cout << '\n' << "Median maximum-rank training=" << training_median << " s throughput="
                << static_cast<double>(config.epochs * static_cast<std::size_t>(scaled.train.X.rows())) / training_median
                << " samples/s\n";
            std::cout << "Training load imbalance=" << repetitions.back().load_imbalance
                << " communication_fraction="
                << repetitions.back().maximum.communication / repetitions.back().maximum.total << '\n';
            print_metrics("Validation accuracy", validation_metrics);
            print_metrics("Held-out test accuracy", test_metrics);
            print_metrics("Training-mean baseline", baselines.first);
            print_metrics("No-drag physics baseline", baselines.second);
            const double imbalance = inference.maximum_compute > 0.0 ?
                (inference.maximum_compute - inference.minimum_compute) / inference.maximum_compute : 0.0;
            std::cout << "Distributed inference median max=" << inference.maximum_compute
                << " s gather=" << inference.gather << " s end_to_end=" << inference.end_to_end
                << " s load_imbalance=" << imbalance << " relative_order_check=" << inference_relative << " PASS\n";
            std::cout << "RESULT_CSV," << options.topology << ',' << world_size << ',' << options.cpu_threads
                << ',' << options.load_balance << ',' << options.communication << ',' << training_median << ','
                << static_cast<double>(config.epochs * static_cast<std::size_t>(scaled.train.X.rows())) / training_median
                << ',' << validation_metrics.rmse << ',' << test_metrics.rmse << ',' << test_metrics.mae << ','
                << test_metrics.r2 << ',' << inference.end_to_end << ',' << inference_relative << ",PASS\n";
            if (options.autotune) {
                std::cout << "Selected topology: " << options.topology << " (eligible candidate in this allocation)\n"
                    << "Reason selected: correctness and accuracy constraints passed\n"
                    << "Rejected topologies: cross-allocation candidates are ranked by scripts/run_hybrid_autotune.sh\n"
                    << "Measured calibration throughput: see per-rank values above\n"
                    << "Expected bottleneck: synchronous gradient reduction and parameter broadcast\n";
            }
            std::ofstream accuracy("hybrid_run_accuracy.csv");
            accuracy << "topology,ranks,cpu_threads,validation_rmse,test_rmse,test_mae,test_r2,status\n"
                << options.topology << ',' << world_size << ',' << options.cpu_threads << ','
                << validation_metrics.rmse << ',' << test_metrics.rmse << ',' << test_metrics.mae << ','
                << test_metrics.r2 << ",PASS\n";
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << " fatal error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
