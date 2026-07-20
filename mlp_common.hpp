#pragma once

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

namespace mlp {

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

    std::size_t transfer_warmup_repetitions = 2;
    std::size_t transfer_repetitions = 7;

    double reference_training_seconds = 0.0;
    bool smoke_test = false;
};

inline void enable_smoke_test(Config& config) {
    config.smoke_test = true;
    config.epochs = 3;
    config.training_warmup_steps = 1;
    config.training_repetitions = 1;
    config.inference_warmup_loops = 1;
    config.inference_repetitions = 1;
    config.inference_loops_per_repetition = 3;
    config.transfer_warmup_repetitions = 1;
    config.transfer_repetitions = 1;
}

inline void apply_command_line_mode(
    Config& config,
    int argc,
    char** argv
) {
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option(argv[argument]);

        if (option == "--smoke-test") {
            enable_smoke_test(config);
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
        } else if (option == "--epochs") {
            if (argument + 1 >= argc) {
                throw std::invalid_argument(
                    "--epochs requires a positive integer."
                );
            }

            const int epochs = std::stoi(argv[++argument]);

            if (epochs < 1) {
                throw std::invalid_argument(
                    "--epochs requires a positive integer."
                );
            }

            config.epochs = static_cast<std::size_t>(epochs);
        }
    }
}

inline double elapsed_seconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& stop
) {
    return std::chrono::duration<double>(stop - start).count();
}

inline double median(std::vector<double> values) {
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

inline double percentile(
    std::vector<double> values,
    double probability
) {
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

    const double weight =
        position - static_cast<double>(lower);

    return
        (1.0 - weight) * values[lower] +
        weight * values[upper];
}

inline double print_times(
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

inline std::vector<std::string> split_csv_line(
    const std::string& line
) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;

    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }

    return fields;
}

inline std::vector<RawRow> load_csv_rows(
    const std::string& path
) {
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
                throw std::runtime_error(
                    "Dataset contains non-finite features."
                );
            }
        }

        if (!std::isfinite(row.target)) {
            throw std::runtime_error(
                "Dataset contains non-finite targets."
            );
        }

        rows.push_back(row);
    }

    if (rows.empty()) {
        throw std::runtime_error("Dataset contains no samples.");
    }

    return rows;
}

inline DatasetSplit build_split(
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

inline Dataset load_dataset(
    const std::string& path
) {
    const auto rows = load_csv_rows(path);

    return Dataset{
        build_split(rows, 0),
        build_split(rows, 1),
        build_split(rows, 2)
    };
}

struct FeatureStandardizer {
    Vector mean;
    Vector standard_deviation;
};

struct TargetStandardizer {
    float mean = 0.0F;
    float standard_deviation = 1.0F;
};

inline FeatureStandardizer fit_feature_standardizer(
    const Matrix& X_train
) {
    Vector mean(X_train.cols());
    Vector standard_deviation(X_train.cols());

    for (
        Eigen::Index column = 0;
        column < X_train.cols();
        ++column
    ) {
        double sum = 0.0;

        for (
            Eigen::Index row = 0;
            row < X_train.rows();
            ++row
        ) {
            sum += X_train(row, column);
        }

        const double column_mean =
            sum /
            static_cast<double>(X_train.rows());

        double squared_difference_sum = 0.0;

        for (
            Eigen::Index row = 0;
            row < X_train.rows();
            ++row
        ) {
            const double difference =
                static_cast<double>(
                    X_train(row, column)
                ) -
                column_mean;

            squared_difference_sum +=
                difference * difference;
        }

        const double variance =
            squared_difference_sum /
            static_cast<double>(X_train.rows());

        mean(column) =
            static_cast<float>(column_mean);

        standard_deviation(column) =
            static_cast<float>(
                std::sqrt(variance)
            );

        if (
            standard_deviation(column) <
            1.0e-12F
        ) {
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

inline TargetStandardizer fit_target_standardizer(
    const Vector& y_train
) {
    double sum = 0.0;

    for (
        Eigen::Index i = 0;
        i < y_train.size();
        ++i
    ) {
        sum += y_train(i);
    }

    const double mean =
        sum /
        static_cast<double>(y_train.size());

    double squared_difference_sum = 0.0;

    for (
        Eigen::Index i = 0;
        i < y_train.size();
        ++i
    ) {
        const double difference =
            static_cast<double>(
                y_train(i)
            ) -
            mean;

        squared_difference_sum +=
            difference * difference;
    }

    const float standard_deviation =
        static_cast<float>(
            std::sqrt(
                squared_difference_sum /
                static_cast<double>(
                    y_train.size()
                )
            )
        );

    if (
        standard_deviation <
        1.0e-12F
    ) {
        throw std::runtime_error(
            "Target has nearly zero standard deviation."
        );
    }

    return TargetStandardizer{
        static_cast<float>(mean),
        standard_deviation
    };
}

inline Matrix transform_features(
    const Matrix& X,
    const FeatureStandardizer& scaler
) {
    Matrix result = X;

    for (
        Eigen::Index column = 0;
        column < result.cols();
        ++column
    ) {
        result.col(column).array() -=
            scaler.mean(column);

        result.col(column).array() /=
            scaler.standard_deviation(column);
    }

    return result;
}

inline Vector transform_targets(
    const Vector& y,
    const TargetStandardizer& scaler
) {
    return (
        (
            y.array() -
            scaler.mean
        ) /
        scaler.standard_deviation
    ).matrix();
}

inline Vector inverse_transform_targets(
    const Vector& standardized_y,
    const TargetStandardizer& scaler
) {
    return (
        standardized_y.array() *
        scaler.standard_deviation +
        scaler.mean
    ).matrix();
}

inline Dataset standardize_dataset(
    const Dataset& raw,
    FeatureStandardizer& X_scaler,
    TargetStandardizer& y_scaler
) {
    X_scaler =
        fit_feature_standardizer(
            raw.train.X
        );

    y_scaler =
        fit_target_standardizer(
            raw.train.y
        );

    return Dataset{
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
}

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

inline Parameters zero_parameters(
    const Config& config
) {
    return Parameters{
        Matrix::Zero(
            config.n_input,
            config.hidden1
        ),
        Vector::Zero(config.hidden1),

        Matrix::Zero(
            config.hidden1,
            config.hidden2
        ),
        Vector::Zero(config.hidden2),

        Matrix::Zero(
            config.hidden2,
            config.n_output
        ),
        Vector::Zero(config.n_output)
    };
}

inline Gradients zero_gradients(
    const Config& config
) {
    return Gradients{
        Matrix::Zero(
            config.n_input,
            config.hidden1
        ),
        Vector::Zero(config.hidden1),

        Matrix::Zero(
            config.hidden1,
            config.hidden2
        ),
        Vector::Zero(config.hidden2),

        Matrix::Zero(
            config.hidden2,
            config.n_output
        ),
        Vector::Zero(config.n_output)
    };
}

inline void fill_he_normal(
    Matrix& weights,
    Eigen::Index fan_in,
    std::mt19937& generator
) {
    const float standard_deviation =
        std::sqrt(
            2.0F /
            static_cast<float>(fan_in)
        );

    std::normal_distribution<float> distribution(
        0.0F,
        standard_deviation
    );

    for (
        Eigen::Index i = 0;
        i < weights.size();
        ++i
    ) {
        weights.data()[i] =
            distribution(generator);
    }
}

inline Parameters initialize_parameters(
    const Config& config
) {
    std::mt19937 generator(
        config.model_seed
    );

    Parameters parameters =
        zero_parameters(config);

    fill_he_normal(
        parameters.W1,
        config.n_input,
        generator
    );

    fill_he_normal(
        parameters.W2,
        config.hidden1,
        generator
    );

    fill_he_normal(
        parameters.W3,
        config.hidden2,
        generator
    );

    return parameters;
}

inline AdamState initialize_adam(
    const Config& config
) {
    return AdamState{
        zero_parameters(config),
        zero_parameters(config),
        0
    };
}

inline Matrix relu(
    const Matrix& values
) {
    return values.cwiseMax(0.0F);
}

inline Matrix relu_derivative(
    const Matrix& pre_activation
) {
    return (
        pre_activation.array() >
        0.0F
    ).cast<float>().matrix();
}

struct ForwardCache {
    Matrix z1;
    Matrix a1;
    Matrix z2;
    Matrix a2;
    Matrix prediction;
};

inline ForwardCache forward_with_cache(
    const Matrix& X,
    const Parameters& parameters
) {
    Matrix z1 =
        X *
        parameters.W1;

    z1.rowwise() +=
        parameters.b1.transpose();

    Matrix a1 =
        relu(z1);

    Matrix z2 =
        a1 *
        parameters.W2;

    z2.rowwise() +=
        parameters.b2.transpose();

    Matrix a2 =
        relu(z2);

    Matrix prediction =
        a2 *
        parameters.W3;

    prediction.rowwise() +=
        parameters.b3.transpose();

    return ForwardCache{
        std::move(z1),
        std::move(a1),
        std::move(z2),
        std::move(a2),
        std::move(prediction)
    };
}

inline Vector predict_standardized(
    const Matrix& X,
    const Parameters& parameters
) {
    Matrix a1 =
        X *
        parameters.W1;

    a1.rowwise() +=
        parameters.b1.transpose();

    a1 =
        relu(a1);

    Matrix a2 =
        a1 *
        parameters.W2;

    a2.rowwise() +=
        parameters.b2.transpose();

    a2 =
        relu(a2);

    Matrix prediction =
        a2 *
        parameters.W3;

    prediction.rowwise() +=
        parameters.b3.transpose();

    return prediction.col(0);
}

struct LossAndGradients {
    float loss = 0.0F;
    Gradients gradients;
};

inline LossAndGradients mse_loss_and_backward(
    const Matrix& X,
    const Vector& target,
    const Parameters& parameters,
    const Config& config,
    float normalization_count
) {
    if (X.rows() == 0) {
        return LossAndGradients{
            0.0F,
            zero_gradients(config)
        };
    }

    const ForwardCache cache =
        forward_with_cache(
            X,
            parameters
        );

    const Vector difference =
        cache.prediction.col(0) -
        target;

    const float inverse_normalization =
        1.0F /
        normalization_count;

    const float loss =
        difference.squaredNorm() *
        inverse_normalization;

    // Scaling by the global batch size lets OpenMP threads or MPI ranks
    // compute disjoint local contributions that sum to the sequential
    // global-batch gradient.
    const Vector d_prediction =
        2.0F *
        inverse_normalization *
        difference;

    Gradients gradients =
        zero_gradients(config);

    gradients.W3.col(0).noalias() =
        cache.a2.transpose() *
        d_prediction;

    gradients.b3(0) =
        d_prediction.sum();

    const Matrix d_a2 =
        d_prediction *
        parameters.W3.col(0).transpose();

    const Matrix d_z2 =
        (
            d_a2.array() *
            relu_derivative(
                cache.z2
            ).array()
        ).matrix();

    gradients.W2.noalias() =
        cache.a1.transpose() *
        d_z2;

    gradients.b2 =
        d_z2.colwise().sum().transpose();

    const Matrix d_a1 =
        d_z2 *
        parameters.W2.transpose();

    const Matrix d_z1 =
        (
            d_a1.array() *
            relu_derivative(
                cache.z1
            ).array()
        ).matrix();

    gradients.W1.noalias() =
        X.transpose() *
        d_z1;

    gradients.b1 =
        d_z1.colwise().sum().transpose();

    return LossAndGradients{
        loss,
        std::move(gradients)
    };
}

inline void add_gradients(
    Gradients& destination,
    const Gradients& source
) {
    destination.W1 += source.W1;
    destination.b1 += source.b1;
    destination.W2 += source.W2;
    destination.b2 += source.b2;
    destination.W3 += source.W3;
    destination.b3 += source.b3;
}

inline std::size_t parameter_count(
    const Config& config
) {
    if (
        config.n_input < 1 ||
        config.hidden1 < 1 ||
        config.hidden2 < 1 ||
        config.n_output < 1
    ) {
        throw std::runtime_error(
            "All model dimensions must be positive."
        );
    }

    const std::size_t count = static_cast<std::size_t>(
        config.n_input *
        config.hidden1 +
        config.hidden1 +
        config.hidden1 *
        config.hidden2 +
        config.hidden2 +
        config.hidden2 *
        config.n_output +
        config.n_output
    );

    return count;
}

inline void pack_gradients(
    const Gradients& gradients,
    std::vector<float>& packed
) {
    packed.clear();

    const std::size_t expected_size =
        static_cast<std::size_t>(
            gradients.W1.size() +
            gradients.b1.size() +
            gradients.W2.size() +
            gradients.b2.size() +
            gradients.W3.size() +
            gradients.b3.size()
        );

    packed.reserve(expected_size);

    const auto append_matrix =
        [&packed](const Matrix& matrix) {
            packed.insert(
                packed.end(),
                matrix.data(),
                matrix.data() +
                matrix.size()
            );
        };

    const auto append_vector =
        [&packed](const Vector& vector) {
            packed.insert(
                packed.end(),
                vector.data(),
                vector.data() +
                vector.size()
            );
        };

    append_matrix(gradients.W1);
    append_vector(gradients.b1);
    append_matrix(gradients.W2);
    append_vector(gradients.b2);
    append_matrix(gradients.W3);
    append_vector(gradients.b3);

    if (packed.size() != expected_size) {
        throw std::runtime_error(
            "Packed gradient size does not match the model."
        );
    }
}

inline void unpack_gradients(
    const std::vector<float>& packed,
    Gradients& gradients
) {
    const std::size_t expected_size =
        static_cast<std::size_t>(
            gradients.W1.size() +
            gradients.b1.size() +
            gradients.W2.size() +
            gradients.b2.size() +
            gradients.W3.size() +
            gradients.b3.size()
        );

    if (packed.size() != expected_size) {
        throw std::runtime_error(
            "Packed gradient size does not match the model."
        );
    }

    std::size_t offset = 0;

    const auto copy_matrix =
        [&packed, &offset](Matrix& matrix) {
            std::copy(
                packed.begin() +
                static_cast<std::ptrdiff_t>(offset),
                packed.begin() +
                static_cast<std::ptrdiff_t>(
                    offset +
                    static_cast<std::size_t>(
                        matrix.size()
                    )
                ),
                matrix.data()
            );

            offset +=
                static_cast<std::size_t>(
                    matrix.size()
                );
        };

    const auto copy_vector =
        [&packed, &offset](Vector& vector) {
            std::copy(
                packed.begin() +
                static_cast<std::ptrdiff_t>(offset),
                packed.begin() +
                static_cast<std::ptrdiff_t>(
                    offset +
                    static_cast<std::size_t>(
                        vector.size()
                    )
                ),
                vector.data()
            );

            offset +=
                static_cast<std::size_t>(
                    vector.size()
                );
        };

    copy_matrix(gradients.W1);
    copy_vector(gradients.b1);
    copy_matrix(gradients.W2);
    copy_vector(gradients.b2);
    copy_matrix(gradients.W3);
    copy_vector(gradients.b3);

    if (offset != packed.size()) {
        throw std::runtime_error(
            "Packed gradient size does not match the model."
        );
    }
}

inline void pack_parameters(
    const Parameters& parameters,
    std::vector<float>& packed
) {
    packed.clear();
    const std::size_t expected_size =
        static_cast<std::size_t>(
            parameters.W1.size() +
            parameters.b1.size() +
            parameters.W2.size() +
            parameters.b2.size() +
            parameters.W3.size() +
            parameters.b3.size()
        );
    packed.reserve(expected_size);

    const auto append = [&packed](const auto& values) {
        packed.insert(
            packed.end(),
            values.data(),
            values.data() + values.size()
        );
    };

    append(parameters.W1);
    append(parameters.b1);
    append(parameters.W2);
    append(parameters.b2);
    append(parameters.W3);
    append(parameters.b3);

    if (packed.size() != expected_size) {
        throw std::runtime_error(
            "Packed parameter size does not match the model."
        );
    }
}

inline void adam_update_matrix(
    Matrix& parameter,
    const Matrix& gradient,
    Matrix& first_moment,
    Matrix& second_moment,
    const Config& config,
    std::uint64_t step
) {
    first_moment.array() =
        config.beta1 *
        first_moment.array() +
        (
            1.0F -
            config.beta1
        ) *
        gradient.array();

    second_moment.array() =
        config.beta2 *
        second_moment.array() +
        (
            1.0F -
            config.beta2
        ) *
        gradient.array().square();

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
        (
            first_moment.array() /
            first_correction
        ) /
        (
            (
                second_moment.array() /
                second_correction
            ).sqrt() +
            config.adam_epsilon
        );
}

inline void adam_update_vector(
    Vector& parameter,
    const Vector& gradient,
    Vector& first_moment,
    Vector& second_moment,
    const Config& config,
    std::uint64_t step
) {
    first_moment.array() =
        config.beta1 *
        first_moment.array() +
        (
            1.0F -
            config.beta1
        ) *
        gradient.array();

    second_moment.array() =
        config.beta2 *
        second_moment.array() +
        (
            1.0F -
            config.beta2
        ) *
        gradient.array().square();

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
        (
            first_moment.array() /
            first_correction
        ) /
        (
            (
                second_moment.array() /
                second_correction
            ).sqrt() +
            config.adam_epsilon
        );
}

inline void adam_step(
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

struct Batch {
    Matrix X;
    Vector y;
};

inline Batch gather_order_range(
    const Matrix& X,
    const Vector& y,
    const std::vector<Eigen::Index>& order,
    std::size_t start,
    std::size_t stop
) {
    const Eigen::Index rows =
        static_cast<Eigen::Index>(
            stop - start
        );

    Matrix X_batch(
        rows,
        X.cols()
    );

    Vector y_batch(rows);

    for (
        Eigen::Index local_row = 0;
        local_row < rows;
        ++local_row
    ) {
        const Eigen::Index source_row =
            order[
                start +
                static_cast<std::size_t>(
                    local_row
                )
            ];

        X_batch.row(local_row) =
            X.row(source_row);

        y_batch(local_row) =
            y(source_row);
    }

    return Batch{
        std::move(X_batch),
        std::move(y_batch)
    };
}

struct Metrics {
    double mse = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    double r2 = 0.0;
    double median_absolute_error = 0.0;
    double p95_absolute_error = 0.0;
    double maximum_absolute_error = 0.0;
};

inline Metrics calculate_metrics(
    const Vector& y_true,
    const Vector& y_prediction
) {
    if (
        y_true.size() !=
        y_prediction.size()
    ) {
        throw std::invalid_argument(
            "Metric vectors have different lengths."
        );
    }

    double squared_error_sum = 0.0;
    double absolute_error_sum = 0.0;
    double target_sum = 0.0;

    std::vector<double> absolute_errors(
        static_cast<std::size_t>(
            y_true.size()
        )
    );

    for (
        Eigen::Index i = 0;
        i < y_true.size();
        ++i
    ) {
        const double target =
            y_true(i);

        const double prediction =
            y_prediction(i);

        const double residual =
            prediction -
            target;

        const double absolute_error =
            std::abs(residual);

        squared_error_sum +=
            residual *
            residual;

        absolute_error_sum +=
            absolute_error;

        target_sum +=
            target;

        absolute_errors[
            static_cast<std::size_t>(i)
        ] = absolute_error;
    }

    const double count =
        static_cast<double>(
            y_true.size()
        );

    const double target_mean =
        target_sum /
        count;

    double total_sum_of_squares = 0.0;

    for (
        Eigen::Index i = 0;
        i < y_true.size();
        ++i
    ) {
        const double difference =
            static_cast<double>(
                y_true(i)
            ) -
            target_mean;

        total_sum_of_squares +=
            difference *
            difference;
    }

    const double mse =
        squared_error_sum /
        count;

    return Metrics{
        mse,
        std::sqrt(mse),
        absolute_error_sum / count,
        1.0 -
        squared_error_sum /
        (
            total_sum_of_squares +
            1.0e-12
        ),
        percentile(
            absolute_errors,
            0.50
        ),
        percentile(
            absolute_errors,
            0.95
        ),
        *std::max_element(
            absolute_errors.begin(),
            absolute_errors.end()
        )
    };
}

inline void print_metrics(
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
        << "MSE         : "
        << metrics.mse
        << "\n"
        << "RMSE        : "
        << metrics.rmse
        << "\n"
        << "MAE         : "
        << metrics.mae
        << "\n"
        << "R2          : "
        << metrics.r2
        << "\n"
        << "MedianAE    : "
        << metrics.median_absolute_error
        << "\n"
        << "P95AE       : "
        << metrics.p95_absolute_error
        << "\n"
        << "MaxAE       : "
        << metrics.maximum_absolute_error
        << "\n";
}

inline std::pair<Metrics, Metrics> calculate_baselines(
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

    Vector no_drag_prediction(
        test.y.size()
    );

    constexpr double gravity =
        9.81;

    constexpr double pi =
        3.14159265358979323846;

    for (
        Eigen::Index i = 0;
        i < test.X.rows();
        ++i
    ) {
        const double speed =
            test.X(i, 0);

        const double angle_radians =
            static_cast<double>(
                test.X(i, 1)
            ) *
            pi /
            180.0;

        no_drag_prediction(i) =
            static_cast<float>(
                speed *
                speed *
                std::sin(
                    2.0 *
                    angle_radians
                ) /
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

inline void print_example_predictions(
    const DatasetSplit& raw_test,
    const Vector& prediction
) {
    std::cout
        << "\nExample predictions\n"
        << "-------------------\n"
        << "v0       theta     drag       "
        << "true_range   predicted_range\n";

    const Eigen::Index examples =
        std::min<Eigen::Index>(
            10,
            raw_test.X.rows()
        );

    for (
        Eigen::Index i = 0;
        i < examples;
        ++i
    ) {
        std::cout
            << std::setw(8)
            << raw_test.X(i, 0)
            << " "
            << std::setw(9)
            << raw_test.X(i, 1)
            << " "
            << std::setw(10)
            << raw_test.X(i, 2)
            << " "
            << std::setw(12)
            << raw_test.y(i)
            << " "
            << std::setw(16)
            << prediction(i)
            << "\n";
    }
}

}  // namespace mlp
