// Explicit OpenMP data-parallel MLP training.
//
// The global mini-batch is divided among OpenMP threads. Each thread computes
// a gradient contribution for disjoint samples. Thread-local gradients are
// summed, then one Adam update is applied. This preserves synchronous
// mini-batch training.
//
// Build:
//   g++ -std=c++17 -O3 -march=native -fopenmp
//       -I /usr/include/eigen3 -Wall -Wextra -pedantic
//       mlp_openmp.cpp -o mlp_openmp
//
// Run on up to 20 threads:
//   OMP_PROC_BIND=close OMP_PLACES=cores ./mlp_openmp dataset.csv 20

#include "mlp_common.hpp"

#include <omp.h>

#include <set>

namespace {

using namespace mlp;

std::vector<int> make_thread_counts(int maximum_threads) {
    if (maximum_threads < 1) {
        throw std::invalid_argument(
            "Maximum OpenMP thread count must be at least one."
        );
    }

    std::set<int> unique_counts;

    for (
        int threads = 1;
        threads < maximum_threads;
        threads *= 2
    ) {
        unique_counts.insert(threads);

        if (threads > maximum_threads / 2) {
            break;
        }
    }

    unique_counts.insert(maximum_threads);

    return std::vector<int>(
        unique_counts.begin(),
        unique_counts.end()
    );
}

Gradients parallel_global_batch_gradient(
    const Matrix& X_train,
    const Vector& y_train,
    const std::vector<Eigen::Index>& order,
    std::size_t batch_start,
    std::size_t batch_stop,
    const Parameters& parameters,
    const Config& config,
    int requested_threads,
    float& batch_loss
) {
    const std::size_t global_batch_size =
        batch_stop -
        batch_start;

    const int active_threads =
        std::min<int>(
            requested_threads,
            static_cast<int>(
                global_batch_size
            )
        );

    std::vector<Gradients>
        local_gradients;

    local_gradients.reserve(
        static_cast<std::size_t>(
            active_threads
        )
    );

    for (
        int thread = 0;
        thread < active_threads;
        ++thread
    ) {
        local_gradients.push_back(
            zero_gradients(config)
        );
    }

    std::vector<float> local_losses(
        static_cast<std::size_t>(
            active_threads
        ),
        0.0F
    );

    #pragma omp parallel num_threads(active_threads)
    {
        const int thread_id =
            omp_get_thread_num();

        const int team_size =
            omp_get_num_threads();

        const std::size_t base_count =
            global_batch_size /
            static_cast<std::size_t>(
                team_size
            );

        const std::size_t remainder =
            global_batch_size %
            static_cast<std::size_t>(
                team_size
            );

        const std::size_t local_count =
            base_count +
            (
                static_cast<std::size_t>(
                    thread_id
                ) <
                remainder
                ? 1U
                : 0U
            );

        const std::size_t local_offset =
            static_cast<std::size_t>(
                thread_id
            ) *
            base_count +
            std::min<std::size_t>(
                static_cast<std::size_t>(
                    thread_id
                ),
                remainder
            );

        const std::size_t local_start =
            batch_start +
            local_offset;

        const std::size_t local_stop =
            local_start +
            local_count;

        const Batch local_batch =
            gather_order_range(
                X_train,
                y_train,
                order,
                local_start,
                local_stop
            );

        LossAndGradients local_result =
            mse_loss_and_backward(
                local_batch.X,
                local_batch.y,
                parameters,
                config,
                static_cast<float>(
                    global_batch_size
                )
            );

        local_losses[
            static_cast<std::size_t>(
                thread_id
            )
        ] = local_result.loss;

        local_gradients[
            static_cast<std::size_t>(
                thread_id
            )
        ] = std::move(
            local_result.gradients
        );
    }

    Gradients global_gradient =
        zero_gradients(config);

    batch_loss = 0.0F;

    for (
        int thread = 0;
        thread < active_threads;
        ++thread
    ) {
        add_gradients(
            global_gradient,
            local_gradients[
                static_cast<std::size_t>(
                    thread
                )
            ]
        );

        batch_loss +=
            local_losses[
                static_cast<std::size_t>(
                    thread
                )
            ];
    }

    return global_gradient;
}

void train_fixed_epochs_openmp(
    Parameters& parameters,
    AdamState& adam,
    const Matrix& X_train,
    const Vector& y_train,
    const Config& config,
    std::mt19937& shuffle_generator,
    int threads
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

    for (
        std::size_t epoch = 0;
        epoch < config.epochs;
        ++epoch
    ) {
        std::shuffle(
            order.begin(),
            order.end(),
            shuffle_generator
        );

        for (
            std::size_t start = 0;
            start <
            static_cast<std::size_t>(
                X_train.rows()
            );
            start +=
            config.batch_size
        ) {
            const std::size_t stop =
                std::min(
                    start +
                    config.batch_size,
                    static_cast<std::size_t>(
                        X_train.rows()
                    )
                );

            float batch_loss =
                0.0F;

            Gradients global_gradient =
                parallel_global_batch_gradient(
                    X_train,
                    y_train,
                    order,
                    start,
                    stop,
                    parameters,
                    config,
                    threads,
                    batch_loss
                );

            (void) batch_loss;

            adam_step(
                parameters,
                global_gradient,
                adam,
                config
            );
        }
    }
}

void warm_up_openmp(
    const Matrix& X_train,
    const Vector& y_train,
    const Config& config,
    int threads
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
            std::size_t start = 0;
            start <
            static_cast<std::size_t>(
                X_train.rows()
            );
            start +=
            config.batch_size
        ) {
            const std::size_t stop =
                std::min(
                    start +
                    config.batch_size,
                    static_cast<std::size_t>(
                        X_train.rows()
                    )
                );

            float batch_loss =
                0.0F;

            Gradients gradient =
                parallel_global_batch_gradient(
                    X_train,
                    y_train,
                    order,
                    start,
                    stop,
                    parameters,
                    config,
                    threads,
                    batch_loss
                );

            adam_step(
                parameters,
                gradient,
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
}

struct OpenMPTrainingBenchmark {
    std::vector<double> times;
    Parameters final_parameters;
};

OpenMPTrainingBenchmark benchmark_training_openmp(
    const Matrix& X_train,
    const Vector& y_train,
    const Config& config,
    int threads
) {
    std::vector<double> times;
    Parameters final_parameters;

    warm_up_openmp(
        X_train,
        y_train,
        config,
        threads
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

        const auto start =
            std::chrono::steady_clock::now();

        train_fixed_epochs_openmp(
            parameters,
            adam,
            X_train,
            y_train,
            config,
            shuffle_generator,
            threads
        );

        const auto stop =
            std::chrono::steady_clock::now();

        times.push_back(
            elapsed_seconds(
                start,
                stop
            )
        );

        final_parameters =
            std::move(parameters);
    }

    return OpenMPTrainingBenchmark{
        std::move(times),
        std::move(final_parameters)
    };
}

Vector predict_openmp(
    const Matrix& X,
    const Parameters& parameters,
    int requested_threads
) {
    const int active_threads =
        std::min<int>(
            requested_threads,
            static_cast<int>(
                X.rows()
            )
        );

    Vector prediction(
        X.rows()
    );

    #pragma omp parallel num_threads(active_threads)
    {
        const int thread_id =
            omp_get_thread_num();

        const int team_size =
            omp_get_num_threads();

        const Eigen::Index base_count =
            X.rows() /
            team_size;

        const Eigen::Index remainder =
            X.rows() %
            team_size;

        const Eigen::Index local_count =
            base_count +
            (
                thread_id <
                remainder
                ? 1
                : 0
            );

        const Eigen::Index local_start =
            static_cast<Eigen::Index>(
                thread_id
            ) *
            base_count +
            std::min<Eigen::Index>(
                thread_id,
                remainder
            );

        if (local_count > 0) {
            const Matrix local_X =
                X.middleRows(
                    local_start,
                    local_count
                );

            const Vector local_prediction =
                predict_standardized(
                    local_X,
                    parameters
                );

            prediction.segment(
                local_start,
                local_count
            ) = local_prediction;
        }
    }

    return prediction;
}

std::vector<double> benchmark_inference_openmp(
    const Matrix& X_test,
    const Parameters& parameters,
    const Config& config,
    int threads
) {
    volatile float sink =
        0.0F;

    for (
        std::size_t loop = 0;
        loop <
        config.inference_warmup_loops;
        ++loop
    ) {
        const Vector prediction =
            predict_openmp(
                X_test,
                parameters,
                threads
            );

        sink +=
            prediction(0);
    }

    std::vector<double> times;

    for (
        std::size_t repetition = 0;
        repetition <
        config.inference_repetitions;
        ++repetition
    ) {
        const auto start =
            std::chrono::steady_clock::now();

        for (
            std::size_t loop = 0;
            loop <
            config.inference_loops_per_repetition;
            ++loop
        ) {
            const Vector prediction =
                predict_openmp(
                    X_test,
                    parameters,
                    threads
                );

            sink +=
                prediction(0);
        }

        const auto stop =
            std::chrono::steady_clock::now();

        times.push_back(
            elapsed_seconds(
                start,
                stop
            )
        );
    }

    if (
        sink ==
        std::numeric_limits<float>::infinity()
    ) {
        std::cerr
            << "Unexpected inference sink.\n";
    }

    return times;
}

struct ThreadResult {
    int threads = 1;
    double training_median = 0.0;
    double inference_median = 0.0;
    double validation_rmse = 0.0;
    double test_rmse = 0.0;
    Parameters parameters;
};

}  // namespace

int main(
    int argc,
    char** argv
) {
    try {
        Config config;

        if (argc >= 2) {
            config.dataset_path =
                argv[1];
        }

        apply_command_line_mode(config, argc, argv);

        int maximum_threads =
            omp_get_max_threads();

        if (argc >= 3) {
            maximum_threads =
                std::stoi(argv[2]);
        }

        maximum_threads =
            std::min(
                maximum_threads,
                omp_get_num_procs()
            );

        omp_set_dynamic(0);
        omp_set_nested(0);
        Eigen::setNbThreads(1);

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

        const auto baselines =
            calculate_baselines(
                raw.train,
                raw.test
            );

        const std::vector<int>
            thread_counts =
                make_thread_counts(
                    maximum_threads
                );

        std::cout
            << "============================================================\n"
            << "OPENMP EIGEN MLP PROJECTILE SURROGATE\n"
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
            << "Eigen internal threads  : "
            << Eigen::nbThreads()
            << "\n"
            << "OpenMP processors       : "
            << omp_get_num_procs()
            << "\n"
            << "Thread counts tested    : ";

        for (int threads : thread_counts) {
            std::cout
                << threads
                << " ";
        }

        std::cout
            << "\n";

        std::vector<ThreadResult> results;

        for (int threads : thread_counts) {
            std::cout
                << "\n============================================================\n"
                << "OPENMP THREADS: "
                << threads
                << "\n"
                << "============================================================\n";

            OpenMPTrainingBenchmark training =
                benchmark_training_openmp(
                    scaled.train.X,
                    scaled.train.y,
                    config,
                    threads
                );

            const double training_median =
                print_times(
                    "OpenMP training",
                    training.times
                );

            const Vector test_prediction =
                inverse_transform_targets(
                    predict_openmp(
                        scaled.test.X,
                        training.final_parameters,
                        threads
                    ),
                    y_scaler
                );

            const Vector validation_prediction =
                inverse_transform_targets(
                    predict_openmp(
                        scaled.validation.X,
                        training.final_parameters,
                        threads
                    ),
                    y_scaler
                );

            const Metrics validation_metrics =
                calculate_metrics(
                    raw.validation.y,
                    validation_prediction
                );

            const Metrics test_metrics =
                calculate_metrics(
                    raw.test.y,
                    test_prediction
                );

            const std::vector<double>
                inference_times =
                    benchmark_inference_openmp(
                        scaled.test.X,
                        training.final_parameters,
                        config,
                        threads
                    );

            const double inference_median =
                print_times(
                    "OpenMP inference",
                    inference_times
                );

            const double training_throughput =
                static_cast<double>(
                    scaled.train.X.rows()
                ) *
                static_cast<double>(
                    config.epochs
                ) /
                training_median;

            const double inference_throughput =
                static_cast<double>(
                    scaled.test.X.rows()
                ) *
                static_cast<double>(
                    config.inference_loops_per_repetition
                ) /
                inference_median;

            std::cout
                << "Training throughput      : "
                << training_throughput
                << " samples/s\n"
                << "Inference throughput     : "
                << inference_throughput
                << " predictions/s\n";

            print_metrics(
                "Validation accuracy",
                validation_metrics
            );

            print_metrics(
                "Held-out test accuracy",
                test_metrics
            );

            results.push_back(
                ThreadResult{
                    threads,
                    training_median,
                    inference_median,
                    validation_metrics.rmse,
                    test_metrics.rmse,
                    std::move(
                        training.final_parameters
                    )
                }
            );
        }

        const double single_thread_time =
            results.front().training_median;

        std::cout
            << "\n============================================================\n"
            << "OPENMP SCALING SUMMARY\n"
            << "============================================================\n"
            << "Threads  Train(s)   Speedup   Efficiency   Test RMSE\n";

        for (const auto& result : results) {
            const double speedup =
                single_thread_time /
                result.training_median;

            const double efficiency =
                speedup /
                static_cast<double>(
                    result.threads
                );

            std::cout
                << std::setw(7)
                << result.threads
                << "  "
                << std::setw(8)
                << std::fixed
                << std::setprecision(4)
                << result.training_median
                << "  "
                << std::setw(8)
                << speedup
                << "  "
                << std::setw(10)
                << efficiency
                << "  "
                << std::setw(10)
                << result.test_rmse
                << "\n";
        }

        print_metrics(
            "Training-mean baseline",
            baselines.first
        );

        print_metrics(
            "No-drag physics baseline",
            baselines.second
        );

        const ThreadResult& final_result =
            results.back();

        const Vector final_prediction =
            inverse_transform_targets(
                predict_openmp(
                    scaled.test.X,
                    final_result.parameters,
                    final_result.threads
                ),
                y_scaler
            );

        print_example_predictions(
            raw.test,
            final_prediction
        );

        return 0;
    } catch (
        const std::exception& error
    ) {
        std::cerr
            << "ERROR: "
            << error.what()
            << "\n";

        return 1;
    }
}
