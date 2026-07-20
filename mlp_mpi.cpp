// Synchronous data-parallel MPI MLP training.
//
// Every MPI rank owns a separate model replica and a disjoint slice of each
// global mini-batch. Ranks compute local gradient contributions, then one
// MPI_Allreduce sums the packed gradients. Every rank applies the same Adam
// update, so replicas remain synchronized.
//
// Build:
//   mpicxx -std=c++17 -O3 -march=native
//       -I /usr/include/eigen3 -Wall -Wextra -pedantic
//       mlp_mpi.cpp -o mlp_mpi
//
// Local runs:
//   mpirun -np 1 ./mlp_mpi dataset.csv
//   mpirun -np 2 ./mlp_mpi dataset.csv
//   mpirun -np 4 ./mlp_mpi dataset.csv
//   mpirun -np 8 ./mlp_mpi dataset.csv
//
// The global batch size remains 64 for every rank count.

#include "mlp_common.hpp"

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

namespace {

using namespace mlp;

struct Partition {
    std::size_t start = 0;
    std::size_t stop = 0;
};

Partition partition_range(
    std::size_t global_start,
    std::size_t global_stop,
    int rank,
    int world_size
) {
    const std::size_t global_count =
        global_stop -
        global_start;

    const std::size_t base_count =
        global_count /
        static_cast<std::size_t>(
            world_size
        );

    const std::size_t remainder =
        global_count %
        static_cast<std::size_t>(
            world_size
        );

    const std::size_t local_count =
        base_count +
        (
            static_cast<std::size_t>(
                rank
            ) <
            remainder
            ? 1U
            : 0U
        );

    const std::size_t local_offset =
        static_cast<std::size_t>(
            rank
        ) *
        base_count +
        std::min<std::size_t>(
            static_cast<std::size_t>(
                rank
            ),
            remainder
        );

    return Partition{
        global_start +
        local_offset,
        global_start +
        local_offset +
        local_count
    };
}

struct MPITrainResult {
    double communication_seconds = 0.0;
};

MPITrainResult train_fixed_epochs_mpi(
    Parameters& parameters,
    AdamState& adam,
    const Matrix& X_train,
    const Vector& y_train,
    const Config& config,
    std::mt19937& shuffle_generator,
    int rank,
    int world_size,
    MPI_Comm communicator
) {
    std::vector<Eigen::Index> order(
        static_cast<std::size_t>(
            X_train.rows()
        )
    );

    std::iota(
        order.begin(),
        order.end(),
        0
    );

    const std::size_t packed_count =
        parameter_count(config);

    std::vector<float> local_packed(
        packed_count,
        0.0F
    );

    std::vector<float> global_packed(
        packed_count,
        0.0F
    );

    Gradients global_gradient =
        zero_gradients(config);

    double communication_seconds =
        0.0;

    for (
        std::size_t epoch = 0;
        epoch < config.epochs;
        ++epoch
    ) {
        // Every rank uses the same seed and generator state, so all ranks
        // construct the same global shuffled sample order.
        std::shuffle(
            order.begin(),
            order.end(),
            shuffle_generator
        );

        for (
            std::size_t global_start = 0;
            global_start <
            static_cast<std::size_t>(
                X_train.rows()
            );
            global_start +=
            config.batch_size
        ) {
            const std::size_t global_stop =
                std::min(
                    global_start +
                    config.batch_size,
                    static_cast<std::size_t>(
                        X_train.rows()
                    )
                );

            const std::size_t global_batch_size =
                global_stop -
                global_start;

            const Partition local_partition =
                partition_range(
                    global_start,
                    global_stop,
                    rank,
                    world_size
                );

            const Batch local_batch =
                gather_order_range(
                    X_train,
                    y_train,
                    order,
                    local_partition.start,
                    local_partition.stop
                );

            const LossAndGradients local_result =
                mse_loss_and_backward(
                    local_batch.X,
                    local_batch.y,
                    parameters,
                    config,
                    static_cast<float>(
                        global_batch_size
                    )
                );

            pack_gradients(
                local_result.gradients,
                local_packed
            );

            const double communication_start =
                MPI_Wtime();

            const int reduce_status =
                MPI_Allreduce(
                    local_packed.data(),
                    global_packed.data(),
                    static_cast<int>(
                        packed_count
                    ),
                    MPI_FLOAT,
                    MPI_SUM,
                    communicator
                );

            communication_seconds +=
                MPI_Wtime() -
                communication_start;

            if (
                reduce_status !=
                MPI_SUCCESS
            ) {
                throw std::runtime_error(
                    "MPI_Allreduce failed while summing gradients."
                );
            }

            unpack_gradients(
                global_packed,
                global_gradient
            );

            // Since every rank receives the same global gradient and begins
            // with the same Adam state, all replicas apply the same update.
            adam_step(
                parameters,
                global_gradient,
                adam,
                config
            );
        }
    }

    return MPITrainResult{
        communication_seconds
    };
}

void warm_up_mpi(
    const Matrix& X_train,
    const Vector& y_train,
    const Config& config,
    int rank,
    int world_size,
    MPI_Comm communicator
) {
    Parameters parameters =
        initialize_parameters(config);

    AdamState adam =
        initialize_adam(config);

    std::mt19937 shuffle_generator(
        config.shuffle_seed
    );

    std::vector<Eigen::Index> order(
        static_cast<std::size_t>(
            X_train.rows()
        )
    );

    std::iota(
        order.begin(),
        order.end(),
        0
    );

    const std::size_t packed_count =
        parameter_count(config);

    std::vector<float> local_packed(
        packed_count,
        0.0F
    );

    std::vector<float> global_packed(
        packed_count,
        0.0F
    );

    Gradients global_gradient =
        zero_gradients(config);

    std::size_t completed_steps =
        0;

    while (
        completed_steps <
        config.training_warmup_steps
    ) {
        std::shuffle(
            order.begin(),
            order.end(),
            shuffle_generator
        );

        for (
            std::size_t global_start = 0;
            global_start <
            static_cast<std::size_t>(
                X_train.rows()
            );
            global_start +=
            config.batch_size
        ) {
            const std::size_t global_stop =
                std::min(
                    global_start +
                    config.batch_size,
                    static_cast<std::size_t>(
                        X_train.rows()
                    )
                );

            const std::size_t global_batch_size =
                global_stop -
                global_start;

            const Partition local_partition =
                partition_range(
                    global_start,
                    global_stop,
                    rank,
                    world_size
                );

            const Batch local_batch =
                gather_order_range(
                    X_train,
                    y_train,
                    order,
                    local_partition.start,
                    local_partition.stop
                );

            const LossAndGradients local_result =
                mse_loss_and_backward(
                    local_batch.X,
                    local_batch.y,
                    parameters,
                    config,
                    static_cast<float>(
                        global_batch_size
                    )
                );

            pack_gradients(
                local_result.gradients,
                local_packed
            );

            if (
                MPI_Allreduce(
                    local_packed.data(),
                    global_packed.data(),
                    static_cast<int>(packed_count),
                    MPI_FLOAT,
                    MPI_SUM,
                    communicator
                ) != MPI_SUCCESS
            ) {
                throw std::runtime_error(
                    "MPI_Allreduce failed during warm-up."
                );
            }

            unpack_gradients(
                global_packed,
                global_gradient
            );

            adam_step(
                parameters,
                global_gradient,
                adam,
                config
            );

            ++completed_steps;

            if (
                completed_steps >=
                config.training_warmup_steps
            ) {
                break;
            }
        }
    }

    MPI_Barrier(communicator);
}

struct MPITrainingBenchmark {
    std::vector<double> maximum_times;
    std::vector<double> maximum_communication_times;
    Parameters final_parameters;
};

MPITrainingBenchmark benchmark_training_mpi(
    const Matrix& X_train,
    const Vector& y_train,
    const Config& config,
    int rank,
    int world_size,
    MPI_Comm communicator
) {
    std::vector<double>
        maximum_times;

    std::vector<double>
        maximum_communication_times;

    Parameters final_parameters;

    warm_up_mpi(
        X_train,
        y_train,
        config,
        rank,
        world_size,
        communicator
    );

    for (
        std::size_t repetition = 0;
        repetition <
        config.training_repetitions;
        ++repetition
    ) {
        Parameters parameters =
            initialize_parameters(config);

        AdamState adam =
            initialize_adam(config);

        std::mt19937 shuffle_generator(
            config.shuffle_seed
        );

        MPI_Barrier(communicator);

        const double start =
            MPI_Wtime();

        const MPITrainResult train_result =
            train_fixed_epochs_mpi(
                parameters,
                adam,
                X_train,
                y_train,
                config,
                shuffle_generator,
                rank,
                world_size,
                communicator
            );

        MPI_Barrier(communicator);

        const double local_elapsed =
            MPI_Wtime() -
            start;

        double maximum_elapsed =
            0.0;

        double maximum_communication =
            0.0;

        MPI_Reduce(
            &local_elapsed,
            &maximum_elapsed,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            communicator
        );

        MPI_Reduce(
            &train_result.communication_seconds,
            &maximum_communication,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            communicator
        );

        if (rank == 0) {
            maximum_times.push_back(
                maximum_elapsed
            );

            maximum_communication_times.push_back(
                maximum_communication
            );
        }

        final_parameters =
            std::move(parameters);
    }

    return MPITrainingBenchmark{
        std::move(
            maximum_times
        ),
        std::move(
            maximum_communication_times
        ),
        std::move(
            final_parameters
        )
    };
}

struct DistributedInferenceResult {
    Vector root_prediction;
    std::vector<double> maximum_times;
};

DistributedInferenceResult benchmark_distributed_inference(
    const Matrix& X_test,
    const Parameters& parameters,
    const Config& config,
    int rank,
    int world_size,
    MPI_Comm communicator
) {
    const Partition local_partition =
        partition_range(
            0,
            static_cast<std::size_t>(
                X_test.rows()
            ),
            rank,
            world_size
        );

    const Eigen::Index local_rows =
        static_cast<Eigen::Index>(
            local_partition.stop -
            local_partition.start
        );

    const Matrix local_X =
        X_test.middleRows(
            static_cast<Eigen::Index>(
                local_partition.start
            ),
            local_rows
        );

    volatile float sink =
        0.0F;

    for (
        std::size_t loop = 0;
        loop <
        config.inference_warmup_loops;
        ++loop
    ) {
        const Vector local_prediction =
            predict_standardized(
                local_X,
                parameters
            );

        if (local_prediction.size() > 0) {
            sink +=
                local_prediction(0);
        }
    }

    MPI_Barrier(communicator);

    std::vector<double>
        maximum_times;

    for (
        std::size_t repetition = 0;
        repetition <
        config.inference_repetitions;
        ++repetition
    ) {
        MPI_Barrier(communicator);

        const double start =
            MPI_Wtime();

        for (
            std::size_t loop = 0;
            loop <
            config.inference_loops_per_repetition;
            ++loop
        ) {
            const Vector local_prediction =
                predict_standardized(
                    local_X,
                    parameters
                );

            if (
                local_prediction.size() >
                0
            ) {
                sink +=
                    local_prediction(0);
            }
        }

        MPI_Barrier(communicator);

        const double local_elapsed =
            MPI_Wtime() -
            start;

        double maximum_elapsed =
            0.0;

        MPI_Reduce(
            &local_elapsed,
            &maximum_elapsed,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            communicator
        );

        if (rank == 0) {
            maximum_times.push_back(
                maximum_elapsed
            );
        }
    }

    const Vector local_prediction =
        predict_standardized(
            local_X,
            parameters
        );

    std::vector<int> counts(
        static_cast<std::size_t>(
            world_size
        )
    );

    std::vector<int> displacements(
        static_cast<std::size_t>(
            world_size
        )
    );

    for (
        int process = 0;
        process < world_size;
        ++process
    ) {
        const Partition process_partition =
            partition_range(
                0,
                static_cast<std::size_t>(
                    X_test.rows()
                ),
                process,
                world_size
            );

        counts[
            static_cast<std::size_t>(
                process
            )
        ] =
            static_cast<int>(
                process_partition.stop -
                process_partition.start
            );

        displacements[
            static_cast<std::size_t>(
                process
            )
        ] =
            static_cast<int>(
                process_partition.start
            );
    }

    Vector root_prediction;

    if (rank == 0) {
        root_prediction.resize(
            X_test.rows()
        );
    }

    MPI_Gatherv(
        local_prediction.data(),
        static_cast<int>(
            local_prediction.size()
        ),
        MPI_FLOAT,
        rank == 0
            ? root_prediction.data()
            : nullptr,
        counts.data(),
        displacements.data(),
        MPI_FLOAT,
        0,
        communicator
    );

    if (
        sink ==
        std::numeric_limits<float>::infinity()
    ) {
        std::cerr
            << "Unexpected inference sink.\n";
    }

    return DistributedInferenceResult{
        std::move(
            root_prediction
        ),
        std::move(
            maximum_times
        )
    };
}

}  // namespace

int main(
    int argc,
    char** argv
) {
    MPI_Init(
        &argc,
        &argv
    );

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &world_size
    );

    try {
        Config config;

        if (argc >= 2) {
            config.dataset_path =
                argv[1];
        }

        apply_command_line_mode(config, argc, argv);

        Eigen::setNbThreads(1);

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        const double load_start =
            MPI_Wtime();

        const Dataset raw =
            load_dataset(
                config.dataset_path
            );

        FeatureStandardizer X_scaler;
        TargetStandardizer y_scaler;

        const Dataset scaled =
            standardize_dataset(
                raw,
                X_scaler,
                y_scaler
            );

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        const double local_load_time =
            MPI_Wtime() -
            load_start;

        double maximum_load_time =
            0.0;

        MPI_Reduce(
            &local_load_time,
            &maximum_load_time,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            std::cout
                << "============================================================\n"
                << "MPI EIGEN MLP PROJECTILE SURROGATE\n"
                << "============================================================\n"
                << "Dataset                 : "
                << config.dataset_path
                << "\n"
                << "Architecture            : "
                << config.n_input << "-"
                << config.hidden1 << "-"
                << config.hidden2 << "-"
                << config.n_output << "\n"
                << "Epochs                  : "
                << config.epochs
                << "\n"
                << "Global batch size       : "
                << config.batch_size
                << "\n"
                << "MPI ranks               : "
                << world_size
                << "\n"
                << "Eigen threads per rank  : "
                << Eigen::nbThreads()
                << "\n"
                << "Maximum load time       : "
                << maximum_load_time
                << " s\n";
        }

        MPITrainingBenchmark training =
            benchmark_training_mpi(
                scaled.train.X,
                scaled.train.y,
                config,
                rank,
                world_size,
                MPI_COMM_WORLD
            );

        const DistributedInferenceResult
            inference =
                benchmark_distributed_inference(
                    scaled.test.X,
                    training.final_parameters,
                    config,
                    rank,
                    world_size,
                    MPI_COMM_WORLD
                );

        // Confirm that every model parameter remains synchronized.
        std::vector<float> local_parameters;
        pack_parameters(training.final_parameters, local_parameters);
        std::vector<float> minimum_parameters(local_parameters.size());
        std::vector<float> maximum_parameters(local_parameters.size());

        MPI_Reduce(
            local_parameters.data(),
            minimum_parameters.data(),
            static_cast<int>(local_parameters.size()),
            MPI_FLOAT,
            MPI_MIN,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            local_parameters.data(),
            maximum_parameters.data(),
            static_cast<int>(local_parameters.size()),
            MPI_FLOAT,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            float maximum_replica_difference = 0.0F;
            for (std::size_t i = 0; i < local_parameters.size(); ++i) {
                maximum_replica_difference = std::max(
                    maximum_replica_difference,
                    maximum_parameters[i] - minimum_parameters[i]
                );
            }

            const double median_training =
                print_times(
                    "MPI synchronized training",
                    training.maximum_times
                );

            const double median_communication =
                print_times(
                    "MPI gradient communication",
                    training.maximum_communication_times
                );

            const double training_throughput =
                static_cast<double>(
                    scaled.train.X.rows()
                ) *
                static_cast<double>(
                    config.epochs
                ) /
                median_training;

            std::cout
                << "Training throughput      : "
                << training_throughput
                << " global samples/s\n"
                << "Communication fraction  : "
                << (
                    median_communication /
                    median_training
                )
                << "\n"
                << "Replica parameter spread: "
                << maximum_replica_difference
                << "\n";

            const double median_inference =
                print_times(
                    "MPI distributed inference",
                    inference.maximum_times
                );

            const double inference_throughput =
                static_cast<double>(
                    scaled.test.X.rows()
                ) *
                static_cast<double>(
                    config.inference_loops_per_repetition
                ) /
                median_inference;

            std::cout
                << "Inference throughput     : "
                << inference_throughput
                << " predictions/s\n";

            const Vector physical_prediction =
                inverse_transform_targets(
                    inference.root_prediction,
                    y_scaler
                );

            const Vector validation_prediction =
                inverse_transform_targets(
                    predict_standardized(
                        scaled.validation.X,
                        training.final_parameters
                    ),
                    y_scaler
                );

            print_metrics(
                "Validation accuracy",
                calculate_metrics(
                    raw.validation.y,
                    validation_prediction
                )
            );

            const Metrics test_metrics =
                calculate_metrics(
                    raw.test.y,
                    physical_prediction
                );

            print_metrics(
                "Held-out test accuracy",
                test_metrics
            );

            const auto baselines =
                calculate_baselines(
                    raw.train,
                    raw.test
                );

            print_metrics(
                "Training-mean baseline",
                baselines.first
            );

            print_metrics(
                "No-drag physics baseline",
                baselines.second
            );

            print_example_predictions(
                raw.test,
                physical_prediction
            );
        }

        MPI_Finalize();
        return 0;
    } catch (
        const std::exception& error
    ) {
        std::cerr
            << "Rank "
            << rank
            << " ERROR: "
            << error.what()
            << "\n";

        MPI_Abort(
            MPI_COMM_WORLD,
            1
        );

        return 1;
    }
}
