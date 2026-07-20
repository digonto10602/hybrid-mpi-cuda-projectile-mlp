// mlp_eigen_sequential_compact.cpp
//
// Sequential, one-thread Eigen implementation of the same projectile MLP.
//
// Install Eigen on Ubuntu:
//   sudo apt update
//   sudo apt install libeigen3-dev
//
// Compile:
//   g++ -std=c++17 -O3 -march=native
//       -I /usr/include/eigen3 -Wall -Wextra -pedantic
//       mlp_eigen_sequential.cpp
//       -o mlp_eigen_sequential
//
// Run:
//   ./mlp_eigen_sequential
//
// Or:
//   ./mlp_eigen_sequential path/to/projectile_dataset_n4000_data1_split1_dt0p02.csv
//
// Do not compile with -fopenmp. This is the sequential baseline.

#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXf;

struct Config {
    std::string dataset_path =
        "projectile_dataset_n4000_data1_split1_dt0p02.csv";

    Eigen::Index n_input = 3;
    Eigen::Index hidden1 = 64;
    Eigen::Index hidden2 = 64;
    Eigen::Index n_output = 1;

    std::size_t epochs = 300;
    std::size_t batch_size = 64;
    float learning_rate = 1.0e-3F;

    float beta1 = 0.9F;
    float beta2 = 0.999F;
    float adam_epsilon = 1.0e-8F;

    std::uint32_t model_seed = 11;
    std::uint32_t shuffle_seed = 21;

    std::size_t training_warmup_steps = 10;
    std::size_t training_repetitions = 3;

    std::size_t inference_warmup_loops = 20;
    std::size_t inference_repetitions = 9;
    std::size_t inference_loops_per_repetition = 500;

    bool run_gradient_check = true;
    float gradient_check_epsilon = 1.0e-3F;

    // Paste a reference median training time here to report speedup.
    double reference_training_seconds = 0.0;
    bool smoke_test = false;
};

volatile float inference_sink = 0.0F;

// ============================================================
// Timing/statistical helpers
// ============================================================

double elapsed_seconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& stop
) {
    return std::chrono::duration<double>(stop - start).count();
}

double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("median() received an empty vector.");
    }

    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;

    if (values.size() % 2 == 1) {
        return values[middle];
    }

    return 0.5 * (values[middle - 1] + values[middle]);
}

double percentile(std::vector<double> values, double probability) {
    if (values.empty()) {
        throw std::invalid_argument("percentile() received an empty vector.");
    }

    if (probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument("Percentile probability must be in [0,1].");
    }

    std::sort(values.begin(), values.end());

    const double position =
        probability * static_cast<double>(values.size() - 1);

    const std::size_t lower =
        static_cast<std::size_t>(std::floor(position));

    const std::size_t upper =
        static_cast<std::size_t>(std::ceil(position));

    if (lower == upper) {
        return values[lower];
    }

    const double weight = position - static_cast<double>(lower);

    return
        (1.0 - weight) * values[lower] +
        weight * values[upper];
}

double print_times(
    const std::string& title,
    const std::vector<double>& values
) {
    std::cout << "\n" << title << "\n"
              << std::string(title.size(), '-') << "\n";

    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << "Repetition " << std::setw(2) << i + 1 << ": "
                  << std::fixed << std::setprecision(8)
                  << values[i] << " s\n";
    }

    const double result = median(values);

    std::cout << "Median                   : "
              << std::fixed << std::setprecision(8)
              << result << " s\n";

    return result;
}

// ============================================================
// Dataset loading
// ============================================================

struct RawRow {
    int split = -1;
    std::array<float, 3> features{};
    float target = 0.0F;
};

struct DatasetSplit {
    Matrix X;
    Vector y;
};

struct Dataset {
    DatasetSplit train;
    DatasetSplit validation;
    DatasetSplit test;
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;

    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }

    return fields;
}

std::vector<RawRow> load_csv_rows(const std::string& path) {
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error(
            "Could not open dataset: " + path +
            "\nGenerate the CSV with the Python/PyTorch version first."
        );
    }

    std::string line;

    if (!std::getline(input, line)) {
        throw std::runtime_error("Dataset has no CSV header.");
    }

    if (line != "split,v0,theta_deg,drag,range") {
        throw std::runtime_error("Unexpected CSV header: " + line);
    }

    std::vector<RawRow> rows;

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        const auto fields = split_csv_line(line);

        if (fields.size() != 5) {
            throw std::runtime_error("Malformed CSV row: " + line);
        }

        RawRow row;
        row.split = std::stoi(fields[0]);

        row.features[0] =
            static_cast<float>(std::stod(fields[1]));

        row.features[1] =
            static_cast<float>(std::stod(fields[2]));

        row.features[2] =
            static_cast<float>(std::stod(fields[3]));

        row.target =
            static_cast<float>(std::stod(fields[4]));

        if (row.split < 0 || row.split > 2) {
            throw std::runtime_error("Split must be 0, 1, or 2.");
        }

        for (float value : row.features) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Dataset contains non-finite features.");
            }
        }

        if (!std::isfinite(row.target)) {
            throw std::runtime_error("Dataset contains non-finite targets.");
        }

        rows.push_back(row);
    }

    if (rows.empty()) {
        throw std::runtime_error("Dataset contains no samples.");
    }

    return rows;
}

DatasetSplit build_split(
    const std::vector<RawRow>& rows,
    int requested_split
) {
    std::size_t count = 0;

    for (const auto& row : rows) {
        if (row.split == requested_split) {
            ++count;
        }
    }

    if (count == 0) {
        throw std::runtime_error("A dataset split is empty.");
    }

    Matrix X(static_cast<Eigen::Index>(count), 3);
    Vector y(static_cast<Eigen::Index>(count));

    Eigen::Index destination = 0;

    for (const auto& row : rows) {
        if (row.split != requested_split) {
            continue;
        }

        X(destination, 0) = row.features[0];
        X(destination, 1) = row.features[1];
        X(destination, 2) = row.features[2];
        y(destination) = row.target;

        ++destination;
    }

    return DatasetSplit{std::move(X), std::move(y)};
}

Dataset load_dataset(const std::string& path) {
    const auto rows = load_csv_rows(path);

    return Dataset{
        build_split(rows, 0),
        build_split(rows, 1),
        build_split(rows, 2)
    };
}

// ============================================================
// Train-only standardization
// ============================================================

struct FeatureStandardizer {
    Vector mean;
    Vector standard_deviation;
};

struct TargetStandardizer {
    float mean = 0.0F;
    float standard_deviation = 1.0F;
};

FeatureStandardizer fit_feature_standardizer(const Matrix& X_train) {
    Vector mean(X_train.cols());
    Vector standard_deviation(X_train.cols());

    for (Eigen::Index column = 0; column < X_train.cols(); ++column) {
        double sum = 0.0;

        for (Eigen::Index row = 0; row < X_train.rows(); ++row) {
            sum += X_train(row, column);
        }

        const double column_mean =
            sum / static_cast<double>(X_train.rows());

        double squared_difference_sum = 0.0;

        for (Eigen::Index row = 0; row < X_train.rows(); ++row) {
            const double difference =
                static_cast<double>(X_train(row, column)) -
                column_mean;

            squared_difference_sum += difference * difference;
        }

        const double variance =
            squared_difference_sum /
            static_cast<double>(X_train.rows());

        mean(column) = static_cast<float>(column_mean);
        standard_deviation(column) =
            static_cast<float>(std::sqrt(variance));

        if (standard_deviation(column) < 1.0e-12F) {
            throw std::runtime_error(
                "A feature has nearly zero standard deviation."
            );
        }
    }

    return FeatureStandardizer{
        std::move(mean),
        std::move(standard_deviation)
    };
}

TargetStandardizer fit_target_standardizer(const Vector& y_train) {
    double sum = 0.0;

    for (Eigen::Index i = 0; i < y_train.size(); ++i) {
        sum += y_train(i);
    }

    const double mean =
        sum / static_cast<double>(y_train.size());

    double squared_difference_sum = 0.0;

    for (Eigen::Index i = 0; i < y_train.size(); ++i) {
        const double difference =
            static_cast<double>(y_train(i)) - mean;

        squared_difference_sum += difference * difference;
    }

    const float standard_deviation =
        static_cast<float>(
            std::sqrt(
                squared_difference_sum /
                static_cast<double>(y_train.size())
            )
        );

    if (standard_deviation < 1.0e-12F) {
        throw std::runtime_error(
            "Target has nearly zero standard deviation."
        );
    }

    return TargetStandardizer{
        static_cast<float>(mean),
        standard_deviation
    };
}

Matrix transform_features(
    const Matrix& X,
    const FeatureStandardizer& scaler
) {
    Matrix result = X;

    for (Eigen::Index column = 0; column < result.cols(); ++column) {
        result.col(column).array() -= scaler.mean(column);
        result.col(column).array() /= scaler.standard_deviation(column);
    }

    return result;
}

Vector transform_targets(
    const Vector& y,
    const TargetStandardizer& scaler
) {
    return ((y.array() - scaler.mean) / scaler.standard_deviation).matrix();
}

Vector inverse_transform_targets(
    const Vector& standardized_y,
    const TargetStandardizer& scaler
) {
    return (
        standardized_y.array() *
        scaler.standard_deviation +
        scaler.mean
    ).matrix();
}

// ============================================================
// Model parameters, gradients, and Adam state
// ============================================================

struct Parameters {
    Matrix W1;
    Vector b1;

    Matrix W2;
    Vector b2;

    Matrix W3;
    Vector b3;
};

struct Gradients {
    Matrix W1;
    Vector b1;

    Matrix W2;
    Vector b2;

    Matrix W3;
    Vector b3;
};

struct AdamState {
    Parameters first_moment;
    Parameters second_moment;
    std::uint64_t step = 0;
};

Parameters zero_parameters(const Config& config) {
    return Parameters{
        Matrix::Zero(config.n_input, config.hidden1),
        Vector::Zero(config.hidden1),

        Matrix::Zero(config.hidden1, config.hidden2),
        Vector::Zero(config.hidden2),

        Matrix::Zero(config.hidden2, config.n_output),
        Vector::Zero(config.n_output)
    };
}

Gradients zero_gradients(const Config& config) {
    return Gradients{
        Matrix::Zero(config.n_input, config.hidden1),
        Vector::Zero(config.hidden1),

        Matrix::Zero(config.hidden1, config.hidden2),
        Vector::Zero(config.hidden2),

        Matrix::Zero(config.hidden2, config.n_output),
        Vector::Zero(config.n_output)
    };
}

void fill_he_normal(
    Matrix& weights,
    Eigen::Index fan_in,
    std::mt19937& generator
) {
    const float standard_deviation =
        std::sqrt(2.0F / static_cast<float>(fan_in));

    std::normal_distribution<float> distribution(
        0.0F,
        standard_deviation
    );

    for (Eigen::Index i = 0; i < weights.size(); ++i) {
        weights.data()[i] = distribution(generator);
    }
}

Parameters initialize_parameters(const Config& config) {
    std::mt19937 generator(config.model_seed);
    Parameters parameters = zero_parameters(config);

    fill_he_normal(parameters.W1, config.n_input, generator);
    fill_he_normal(parameters.W2, config.hidden1, generator);
    fill_he_normal(parameters.W3, config.hidden2, generator);

    return parameters;
}

AdamState initialize_adam(const Config& config) {
    return AdamState{
        zero_parameters(config),
        zero_parameters(config),
        0
    };
}

// ============================================================
// Forward propagation
// ============================================================

struct ForwardCache {
    Matrix z1;
    Matrix a1;
    Matrix z2;
    Matrix a2;
    Matrix prediction;
};

Matrix relu(const Matrix& values) {
    return values.cwiseMax(0.0F);
}

Matrix relu_derivative(const Matrix& pre_activation) {
    return (
        pre_activation.array() > 0.0F
    ).cast<float>().matrix();
}

ForwardCache forward_with_cache(
    const Matrix& X,
    const Parameters& parameters
) {
    Matrix z1 = X * parameters.W1;
    z1.rowwise() += parameters.b1.transpose();

    Matrix a1 = relu(z1);

    Matrix z2 = a1 * parameters.W2;
    z2.rowwise() += parameters.b2.transpose();

    Matrix a2 = relu(z2);

    Matrix prediction = a2 * parameters.W3;
    prediction.rowwise() += parameters.b3.transpose();

    return ForwardCache{
        std::move(z1),
        std::move(a1),
        std::move(z2),
        std::move(a2),
        std::move(prediction)
    };
}

Vector predict_standardized(
    const Matrix& X,
    const Parameters& parameters
) {
    Matrix a1 = X * parameters.W1;
    a1.rowwise() += parameters.b1.transpose();
    a1 = relu(a1);

    Matrix a2 = a1 * parameters.W2;
    a2.rowwise() += parameters.b2.transpose();
    a2 = relu(a2);

    Matrix prediction = a2 * parameters.W3;
    prediction.rowwise() += parameters.b3.transpose();

    return prediction.col(0);
}

// ============================================================
// MSE and manual backpropagation
// ============================================================

struct LossAndGradients {
    float loss = 0.0F;
    Gradients gradients;
};

float mse_loss_only(
    const Matrix& X,
    const Vector& target,
    const Parameters& parameters
) {
    const Vector prediction = predict_standardized(X, parameters);

    return
        (prediction - target).squaredNorm() /
        static_cast<float>(target.size());
}

LossAndGradients mse_loss_and_backward(
    const Matrix& X,
    const Vector& target,
    const Parameters& parameters,
    const Config& config
) {
    const ForwardCache cache =
        forward_with_cache(X, parameters);

    const float inverse_batch =
        1.0F / static_cast<float>(X.rows());

    const Vector difference =
        cache.prediction.col(0) - target;

    const float loss =
        difference.squaredNorm() * inverse_batch;

    // dL/dprediction = 2(prediction-target)/B
    const Vector d_prediction =
        2.0F * inverse_batch * difference;

    Gradients gradients = zero_gradients(config);

    // Output layer:
    // dW3 = A2^T dY
    // db3 = sum(dY)
    gradients.W3.col(0).noalias() =
        cache.a2.transpose() * d_prediction;

    gradients.b3(0) = d_prediction.sum();

    // dA2 = dY W3^T
    const Matrix d_a2 =
        d_prediction *
        parameters.W3.col(0).transpose();

    // dZ2 = dA2 ⊙ ReLU'(Z2)
    const Matrix d_z2 =
        (
            d_a2.array() *
            relu_derivative(cache.z2).array()
        ).matrix();

    // Second layer:
    // dW2 = A1^T dZ2
    // db2 = sum_rows(dZ2)
    gradients.W2.noalias() =
        cache.a1.transpose() * d_z2;

    gradients.b2 =
        d_z2.colwise().sum().transpose();

    // dA1 = dZ2 W2^T
    const Matrix d_a1 =
        d_z2 * parameters.W2.transpose();

    // dZ1 = dA1 ⊙ ReLU'(Z1)
    const Matrix d_z1 =
        (
            d_a1.array() *
            relu_derivative(cache.z1).array()
        ).matrix();

    // First layer:
    // dW1 = X^T dZ1
    // db1 = sum_rows(dZ1)
    gradients.W1.noalias() =
        X.transpose() * d_z1;

    gradients.b1 =
        d_z1.colwise().sum().transpose();

    return LossAndGradients{
        loss,
        std::move(gradients)
    };
}

// ============================================================
// Finite-difference gradient check
// ============================================================

struct GradientCheckResult {
    std::string name;
    float analytical = 0.0F;
    float numerical = 0.0F;
    float relative_error = 0.0F;
};

GradientCheckResult check_one_gradient(
    const std::string& name,
    float& parameter,
    float analytical_gradient,
    const Matrix& X,
    const Vector& y,
    Parameters& parameters,
    float epsilon
) {
    const float original = parameter;

    parameter = original + epsilon;
    const float loss_plus =
        mse_loss_only(X, y, parameters);

    parameter = original - epsilon;
    const float loss_minus =
        mse_loss_only(X, y, parameters);

    parameter = original;

    const float numerical_gradient =
        (loss_plus - loss_minus) /
        (2.0F * epsilon);

    const float relative_error =
        std::abs(
            numerical_gradient -
            analytical_gradient
        ) /
        (
            std::abs(numerical_gradient) +
            std::abs(analytical_gradient) +
            1.0e-8F
        );

    return GradientCheckResult{
        name,
        analytical_gradient,
        numerical_gradient,
        relative_error
    };
}

void run_gradient_check(
    const Matrix& X_train,
    const Vector& y_train,
    const Config& config
) {
    const Eigen::Index samples =
        std::min<Eigen::Index>(8, X_train.rows());

    const Matrix X = X_train.topRows(samples);
    const Vector y = y_train.head(samples);

    Parameters parameters =
        initialize_parameters(config);

    const LossAndGradients analytical =
        mse_loss_and_backward(
            X,
            y,
            parameters,
            config
        );

    std::vector<GradientCheckResult> results;

    results.push_back(
        check_one_gradient(
            "W1(0,0)",
            parameters.W1(0, 0),
            analytical.gradients.W1(0, 0),
            X,
            y,
            parameters,
            config.gradient_check_epsilon
        )
    );

    results.push_back(
        check_one_gradient(
            "b1(5)",
            parameters.b1(5),
            analytical.gradients.b1(5),
            X,
            y,
            parameters,
            config.gradient_check_epsilon
        )
    );

    results.push_back(
        check_one_gradient(
            "W2(3,7)",
            parameters.W2(3, 7),
            analytical.gradients.W2(3, 7),
            X,
            y,
            parameters,
            config.gradient_check_epsilon
        )
    );

    results.push_back(
        check_one_gradient(
            "b2(9)",
            parameters.b2(9),
            analytical.gradients.b2(9),
            X,
            y,
            parameters,
            config.gradient_check_epsilon
        )
    );

    results.push_back(
        check_one_gradient(
            "W3(11,0)",
            parameters.W3(11, 0),
            analytical.gradients.W3(11, 0),
            X,
            y,
            parameters,
            config.gradient_check_epsilon
        )
    );

    results.push_back(
        check_one_gradient(
            "b3(0)",
            parameters.b3(0),
            analytical.gradients.b3(0),
            X,
            y,
            parameters,
            config.gradient_check_epsilon
        )
    );

    float maximum_relative_error = 0.0F;

    std::cout
        << "\nFinite-difference gradient check\n"
        << "--------------------------------\n";

    for (const auto& result : results) {
        maximum_relative_error =
            std::max(
                maximum_relative_error,
                result.relative_error
            );

        std::cout
            << std::setw(10)
            << result.name
            << "  analytical="
            << std::scientific
            << std::setprecision(6)
            << result.analytical
            << "  numerical="
            << result.numerical
            << "  relative_error="
            << result.relative_error
            << "\n";
    }

    std::cout
        << "Maximum relative error   : "
        << std::scientific
        << maximum_relative_error
        << "\n"
        << "Gradient check           : "
        << (
            maximum_relative_error < 5.0e-2F
            ? "PASS"
            : "REVIEW REQUIRED"
        )
        << "\n";

    if (maximum_relative_error >= 5.0e-2F) {
        throw std::runtime_error("Finite-difference gradient check failed.");
    }
}

// ============================================================
// Adam optimizer
// ============================================================

void adam_update_matrix(
    Matrix& parameter,
    const Matrix& gradient,
    Matrix& first_moment,
    Matrix& second_moment,
    const Config& config,
    std::uint64_t step
) {
    first_moment.array() =
        config.beta1 * first_moment.array() +
        (1.0F - config.beta1) * gradient.array();

    second_moment.array() =
        config.beta2 * second_moment.array() +
        (1.0F - config.beta2) * gradient.array().square();

    const float first_correction =
        1.0F -
        std::pow(
            config.beta1,
            static_cast<float>(step)
        );

    const float second_correction =
        1.0F -
        std::pow(
            config.beta2,
            static_cast<float>(step)
        );

    parameter.array() -=
        config.learning_rate *
        (first_moment.array() / first_correction) /
        (
            (second_moment.array() / second_correction).sqrt() +
            config.adam_epsilon
        );
}

void adam_update_vector(
    Vector& parameter,
    const Vector& gradient,
    Vector& first_moment,
    Vector& second_moment,
    const Config& config,
    std::uint64_t step
) {
    first_moment.array() =
        config.beta1 * first_moment.array() +
        (1.0F - config.beta1) * gradient.array();

    second_moment.array() =
        config.beta2 * second_moment.array() +
        (1.0F - config.beta2) * gradient.array().square();

    const float first_correction =
        1.0F -
        std::pow(
            config.beta1,
            static_cast<float>(step)
        );

    const float second_correction =
        1.0F -
        std::pow(
            config.beta2,
            static_cast<float>(step)
        );

    parameter.array() -=
        config.learning_rate *
        (first_moment.array() / first_correction) /
        (
            (second_moment.array() / second_correction).sqrt() +
            config.adam_epsilon
        );
}

void adam_step(
    Parameters& parameters,
    const Gradients& gradients,
    AdamState& state,
    const Config& config
) {
    ++state.step;

    adam_update_matrix(
        parameters.W1,
        gradients.W1,
        state.first_moment.W1,
        state.second_moment.W1,
        config,
        state.step
    );

    adam_update_vector(
        parameters.b1,
        gradients.b1,
        state.first_moment.b1,
        state.second_moment.b1,
        config,
        state.step
    );

    adam_update_matrix(
        parameters.W2,
        gradients.W2,
        state.first_moment.W2,
        state.second_moment.W2,
        config,
        state.step
    );

    adam_update_vector(
        parameters.b2,
        gradients.b2,
        state.first_moment.b2,
        state.second_moment.b2,
        config,
        state.step
    );

    adam_update_matrix(
        parameters.W3,
        gradients.W3,
        state.first_moment.W3,
        state.second_moment.W3,
        config,
        state.step
    );

    adam_update_vector(
        parameters.b3,
        gradients.b3,
        state.first_moment.b3,
        state.second_moment.b3,
        config,
        state.step
    );
}

// ============================================================
// Mini-batching and training
// ============================================================

struct Batch {
    Matrix X;
    Vector y;
};

Batch gather_batch(
    const Matrix& X,
    const Vector& y,
    const std::vector<Eigen::Index>& order,
    std::size_t start,
    std::size_t stop
) {
    const Eigen::Index batch_size =
        static_cast<Eigen::Index>(stop - start);

    Matrix X_batch(batch_size, X.cols());
    Vector y_batch(batch_size);

    for (Eigen::Index row = 0; row < batch_size; ++row) {
        const Eigen::Index source_row =
            order[
                start +
                static_cast<std::size_t>(row)
            ];

        X_batch.row(row) = X.row(source_row);
        y_batch(row) = y(source_row);
    }

    return Batch{
        std::move(X_batch),
        std::move(y_batch)
    };
}

void train_fixed_epochs(
    Parameters& parameters,
    AdamState& adam,
    const Matrix& X_train,
    const Vector& y_train,
    const Config& config,
    std::mt19937& shuffle_generator
) {
    std::vector<Eigen::Index> order(
        static_cast<std::size_t>(X_train.rows())
    );

    std::iota(order.begin(), order.end(), 0);

    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(
            order.begin(),
            order.end(),
            shuffle_generator
        );

        for (
            std::size_t start = 0;
            start < static_cast<std::size_t>(X_train.rows());
            start += config.batch_size
        ) {
            const std::size_t stop =
                std::min(
                    start + config.batch_size,
                    static_cast<std::size_t>(X_train.rows())
                );

            Batch batch =
                gather_batch(
                    X_train,
                    y_train,
                    order,
                    start,
                    stop
                );

            const LossAndGradients result =
                mse_loss_and_backward(
                    batch.X,
                    batch.y,
                    parameters,
                    config
                );

            adam_step(
                parameters,
                result.gradients,
                adam,
                config
            );
        }
    }
}

// ============================================================
// Warm-up and repeated training benchmark
// ============================================================

void warm_up_training(
    const Matrix& X_train,
    const Vector& y_train,
    const Config& config
) {
    Parameters parameters =
        initialize_parameters(config);

    AdamState adam =
        initialize_adam(config);

    std::mt19937 shuffle_generator(
        config.shuffle_seed
    );

    std::vector<Eigen::Index> order(
        static_cast<std::size_t>(X_train.rows())
    );

    std::iota(order.begin(), order.end(), 0);

    std::size_t completed_steps = 0;

    while (completed_steps < config.training_warmup_steps) {
        std::shuffle(
            order.begin(),
            order.end(),
            shuffle_generator
        );

        for (
            std::size_t start = 0;
            start < static_cast<std::size_t>(X_train.rows());
            start += config.batch_size
        ) {
            const std::size_t stop =
                std::min(
                    start + config.batch_size,
                    static_cast<std::size_t>(X_train.rows())
                );

            Batch batch =
                gather_batch(
                    X_train,
                    y_train,
                    order,
                    start,
                    stop
                );

            const LossAndGradients result =
                mse_loss_and_backward(
                    batch.X,
                    batch.y,
                    parameters,
                    config
                );

            adam_step(
                parameters,
                result.gradients,
                adam,
                config
            );

            ++completed_steps;

            if (completed_steps >= config.training_warmup_steps) {
                break;
            }
        }
    }
}

struct TrainingBenchmark {
    std::vector<double> times;
    Parameters final_parameters;
};

TrainingBenchmark benchmark_training(
    const Matrix& X_train,
    const Vector& y_train,
    const Config& config
) {
    std::vector<double> times;
    Parameters final_parameters;

    for (
        std::size_t repetition = 0;
        repetition < config.training_repetitions;
        ++repetition
    ) {
        // Excluded from the timed region.
        Parameters parameters =
            initialize_parameters(config);

        AdamState adam =
            initialize_adam(config);

        std::mt19937 shuffle_generator(
            config.shuffle_seed
        );

        const auto start =
            std::chrono::steady_clock::now();

        train_fixed_epochs(
            parameters,
            adam,
            X_train,
            y_train,
            config,
            shuffle_generator
        );

        const auto stop =
            std::chrono::steady_clock::now();

        times.push_back(
            elapsed_seconds(start, stop)
        );

        final_parameters =
            std::move(parameters);
    }

    return TrainingBenchmark{
        std::move(times),
        std::move(final_parameters)
    };
}

// ============================================================
// Accuracy metrics and baselines
// ============================================================

struct Metrics {
    double mse = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    double r2 = 0.0;
    double median_absolute_error = 0.0;
    double p95_absolute_error = 0.0;
    double maximum_absolute_error = 0.0;
};

Metrics calculate_metrics(
    const Vector& y_true,
    const Vector& y_prediction
) {
    if (y_true.size() != y_prediction.size()) {
        throw std::invalid_argument(
            "Metric vectors have different lengths."
        );
    }

    double squared_error_sum = 0.0;
    double absolute_error_sum = 0.0;
    double target_sum = 0.0;

    std::vector<double> absolute_errors(
        static_cast<std::size_t>(y_true.size())
    );

    for (Eigen::Index i = 0; i < y_true.size(); ++i) {
        const double target = y_true(i);
        const double prediction = y_prediction(i);
        const double residual = prediction - target;
        const double absolute_error = std::abs(residual);

        squared_error_sum += residual * residual;
        absolute_error_sum += absolute_error;
        target_sum += target;

        absolute_errors[
            static_cast<std::size_t>(i)
        ] = absolute_error;
    }

    const double count =
        static_cast<double>(y_true.size());

    const double target_mean =
        target_sum / count;

    double total_sum_of_squares = 0.0;

    for (Eigen::Index i = 0; i < y_true.size(); ++i) {
        const double difference =
            static_cast<double>(y_true(i)) -
            target_mean;

        total_sum_of_squares +=
            difference * difference;
    }

    const double mse =
        squared_error_sum / count;

    return Metrics{
        mse,
        std::sqrt(mse),
        absolute_error_sum / count,
        1.0 -
        squared_error_sum /
        (total_sum_of_squares + 1.0e-12),
        percentile(absolute_errors, 0.50),
        percentile(absolute_errors, 0.95),
        *std::max_element(
            absolute_errors.begin(),
            absolute_errors.end()
        )
    };
}

void print_metrics(
    const std::string& title,
    const Metrics& metrics
) {
    std::cout
        << "\n"
        << title
        << "\n"
        << std::string(title.size(), '-')
        << "\n"
        << std::fixed
        << std::setprecision(8)
        << "MSE         : " << metrics.mse << "\n"
        << "RMSE        : " << metrics.rmse << "\n"
        << "MAE         : " << metrics.mae << "\n"
        << "R2          : " << metrics.r2 << "\n"
        << "MedianAE    : " << metrics.median_absolute_error << "\n"
        << "P95AE       : " << metrics.p95_absolute_error << "\n"
        << "MaxAE       : " << metrics.maximum_absolute_error << "\n";
}

std::pair<Metrics, Metrics> calculate_baselines(
    const DatasetSplit& training,
    const DatasetSplit& test
) {
    const float training_mean =
        training.y.mean();

    const Vector mean_prediction =
        Vector::Constant(
            test.y.size(),
            training_mean
        );

    Vector no_drag_prediction(test.y.size());

    constexpr double gravity = 9.81;
    constexpr double pi = 3.14159265358979323846;

    for (Eigen::Index i = 0; i < test.X.rows(); ++i) {
        const double speed = test.X(i, 0);

        const double angle_radians =
            static_cast<double>(test.X(i, 1)) *
            pi /
            180.0;

        no_drag_prediction(i) =
            static_cast<float>(
                speed *
                speed *
                std::sin(2.0 * angle_radians) /
                gravity
            );
    }

    return {
        calculate_metrics(
            test.y,
            mean_prediction
        ),
        calculate_metrics(
            test.y,
            no_drag_prediction
        )
    };
}

// ============================================================
// Inference warm-up and benchmark
// ============================================================

void warm_up_inference(
    const Matrix& X_test,
    const Parameters& parameters,
    const Config& config
) {
    for (
        std::size_t loop = 0;
        loop < config.inference_warmup_loops;
        ++loop
    ) {
        const Vector prediction =
            predict_standardized(
                X_test,
                parameters
            );

        inference_sink += prediction(0);
    }
}

std::vector<double> benchmark_inference(
    const Matrix& X_test,
    const Parameters& parameters,
    const Config& config
) {
    std::vector<double> times;

    for (
        std::size_t repetition = 0;
        repetition < config.inference_repetitions;
        ++repetition
    ) {
        const auto start =
            std::chrono::steady_clock::now();

        for (
            std::size_t loop = 0;
            loop < config.inference_loops_per_repetition;
            ++loop
        ) {
            const Vector prediction =
                predict_standardized(
                    X_test,
                    parameters
                );

            inference_sink += prediction(0);
        }

        const auto stop =
            std::chrono::steady_clock::now();

        times.push_back(
            elapsed_seconds(start, stop)
        );
    }

    return times;
}

// ============================================================
// Main workflow
// ============================================================

int main(int argc, char** argv) {
    try {
        Config config;

        if (argc >= 2) {
            config.dataset_path = argv[1];
        }

        for (int argument = 1; argument < argc; ++argument) {
            const std::string option(argv[argument]);

            if (option == "--smoke-test") {
                config.smoke_test = true;
                config.epochs = 3;
                config.training_warmup_steps = 1;
                config.training_repetitions = 1;
                config.inference_warmup_loops = 1;
                config.inference_repetitions = 1;
                config.inference_loops_per_repetition = 3;
            } else if (option == "--hidden-size") {
                if (argument + 1 >= argc) {
                    throw std::invalid_argument(
                        "--hidden-size requires a positive integer."
                    );
                }

                const int hidden_size = std::stoi(argv[++argument]);

                if (hidden_size < 1) {
                    throw std::invalid_argument(
                        "--hidden-size requires a positive integer."
                    );
                }

                config.hidden1 = hidden_size;
                config.hidden2 = hidden_size;
            } else if (option == "--batch-size") {
                if (argument + 1 >= argc) {
                    throw std::invalid_argument(
                        "--batch-size requires a positive integer."
                    );
                }

                const int batch_size = std::stoi(argv[++argument]);

                if (batch_size < 1) {
                    throw std::invalid_argument(
                        "--batch-size requires a positive integer."
                    );
                }

                config.batch_size = static_cast<std::size_t>(batch_size);
            }
        }

        // Force the sequential baseline.
        Eigen::setNbThreads(1);

        std::cout
            << std::fixed
            << std::setprecision(8)
            << "============================================================\n"
            << "SEQUENTIAL C++ EIGEN MLP PROJECTILE SURROGATE\n"
            << "============================================================\n"
            << "Dataset                 : " << config.dataset_path << "\n"
            << "Architecture            : "
            << config.n_input << "-"
            << config.hidden1 << "-"
            << config.hidden2 << "-"
            << config.n_output << "\n"
            << "Epochs                  : " << config.epochs << "\n"
            << "Batch size              : " << config.batch_size << "\n"
            << "Learning rate           : " << config.learning_rate << "\n"
            << "Eigen threads           : " << Eigen::nbThreads() << "\n"
            << "OpenMP enabled          : No\n"
            << "Model precision         : float32\n";

        // Load exact shared data. File I/O is outside training timing.
        const auto load_start =
            std::chrono::steady_clock::now();

        const Dataset raw =
            load_dataset(config.dataset_path);

        const auto load_stop =
            std::chrono::steady_clock::now();

        std::cout
            << "\nDataset loading\n"
            << "---------------\n"
            << "Training samples         : " << raw.train.X.rows() << "\n"
            << "Validation samples       : " << raw.validation.X.rows() << "\n"
            << "Test samples             : " << raw.test.X.rows() << "\n"
            << "CSV loading time         : "
            << elapsed_seconds(load_start, load_stop)
            << " s\n";

        // Fit scalers from training data only.
        const FeatureStandardizer X_scaler =
            fit_feature_standardizer(raw.train.X);

        const TargetStandardizer y_scaler =
            fit_target_standardizer(raw.train.y);

        const Dataset scaled{
            DatasetSplit{
                transform_features(
                    raw.train.X,
                    X_scaler
                ),
                transform_targets(
                    raw.train.y,
                    y_scaler
                )
            },
            DatasetSplit{
                transform_features(
                    raw.validation.X,
                    X_scaler
                ),
                transform_targets(
                    raw.validation.y,
                    y_scaler
                )
            },
            DatasetSplit{
                transform_features(
                    raw.test.X,
                    X_scaler
                ),
                transform_targets(
                    raw.test.y,
                    y_scaler
                )
            }
        };

        // Verify manual chain-rule implementation.
        if (config.run_gradient_check) {
            run_gradient_check(
                scaled.train.X,
                scaled.train.y,
                config
            );
        }

        // Untimed warm-up.
        warm_up_training(
            scaled.train.X,
            scaled.train.y,
            config
        );

        // Repeated complete training measurements.
        TrainingBenchmark training =
            benchmark_training(
                scaled.train.X,
                scaled.train.y,
                config
            );

        const double median_training =
            print_times(
                "Sequential Eigen training",
                training.times
            );

        const double processed_samples =
            static_cast<double>(
                scaled.train.X.rows()
            ) *
            static_cast<double>(
                config.epochs
            );

        const double training_throughput =
            processed_samples /
            median_training;

        std::cout
            << "Training throughput       : "
            << training_throughput
            << " samples/s\n";

        if (config.reference_training_seconds > 0.0) {
            std::cout
                << "Speedup versus reference  : "
                << (
                    config.reference_training_seconds /
                    median_training
                )
                << "x\n";
        } else {
            std::cout
                << "Speedup versus reference  : N/A\n";
        }

        // Accuracy outside the training timer.
        const Vector validation_prediction =
            inverse_transform_targets(
                predict_standardized(
                    scaled.validation.X,
                    training.final_parameters
                ),
                y_scaler
            );

        const Vector test_prediction =
            inverse_transform_targets(
                predict_standardized(
                    scaled.test.X,
                    training.final_parameters
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

        const auto [
            mean_baseline,
            no_drag_baseline
        ] = calculate_baselines(
            raw.train,
            raw.test
        );

        print_metrics(
            "Validation accuracy",
            validation_metrics
        );

        print_metrics(
            "Held-out test accuracy",
            test_metrics
        );

        print_metrics(
            "Training-mean baseline",
            mean_baseline
        );

        print_metrics(
            "No-drag physics baseline",
            no_drag_baseline
        );

        // Untimed inference warm-up.
        warm_up_inference(
            scaled.test.X,
            training.final_parameters,
            config
        );

        // Repeated inference timing.
        const std::vector<double> inference_times =
            benchmark_inference(
                scaled.test.X,
                training.final_parameters,
                config
            );

        const double median_inference =
            print_times(
                "Sequential Eigen inference",
                inference_times
            );

        const double measured_predictions =
            static_cast<double>(
                scaled.test.X.rows()
            ) *
            static_cast<double>(
                config.inference_loops_per_repetition
            );

        const double inference_throughput =
            measured_predictions /
            median_inference;

        std::cout
            << "Inference throughput      : "
            << inference_throughput
            << " predictions/s\n"
            << "Time per prediction       : "
            << (
                1.0e6 /
                inference_throughput
            )
            << " microseconds\n";

        // Example output outside all timers.
        std::cout
            << "\nExample predictions\n"
            << "-------------------\n"
            << "v0       theta     drag       "
            << "true_range   predicted_range\n";

        const Eigen::Index examples =
            std::min<Eigen::Index>(
                10,
                raw.test.X.rows()
            );

        for (Eigen::Index i = 0; i < examples; ++i) {
            std::cout
                << std::setw(8) << raw.test.X(i, 0) << " "
                << std::setw(9) << raw.test.X(i, 1) << " "
                << std::setw(10) << raw.test.X(i, 2) << " "
                << std::setw(12) << raw.test.y(i) << " "
                << std::setw(16) << test_prediction(i) << "\n";
        }

        std::cout
            << "\nInference sink           : "
            << inference_sink
            << "\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "ERROR: "
            << error.what()
            << "\n";

        return 1;
    }
}
