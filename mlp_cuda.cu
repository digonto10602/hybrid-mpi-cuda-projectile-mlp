// Explicit CUDA C++ + cuBLAS MLP training.
//
// Dense matrix products use cuBLAS. Custom CUDA kernels perform:
//   * batch gathering
//   * bias addition
//   * ReLU forward/backward
//   * MSE derivative
//   * bias-gradient reductions
//   * Adam updates
//
// Build:
//   nvcc -std=c++17 -O3 -I /usr/include/eigen3
//       mlp_cuda.cu -lcublas -o mlp_cuda
//
// Run:
//   ./mlp_cuda dataset.csv
//
// The same global batch size, epoch count, architecture, float32 precision,
// warm-up, repeated timing, median reporting, and accuracy metrics are used.

#include "mlp_common.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>

namespace {

using namespace mlp;

using ColMatrix =
    Eigen::Matrix<
        float,
        Eigen::Dynamic,
        Eigen::Dynamic,
        Eigen::ColMajor
    >;

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        const cudaError_t error__ = (call);                                \
        if (error__ != cudaSuccess) {                                      \
            throw std::runtime_error(                                      \
                std::string("CUDA error at ") +                            \
                __FILE__ + ":" +                                           \
                std::to_string(__LINE__) + ": " +                          \
                cudaGetErrorString(error__)                                \
            );                                                             \
        }                                                                  \
    } while (false)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        const cublasStatus_t status__ = (call);                            \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                           \
            throw std::runtime_error(                                      \
                std::string("cuBLAS error at ") +                          \
                __FILE__ + ":" +                                           \
                std::to_string(__LINE__) +                                 \
                ", status=" +                                              \
                std::to_string(static_cast<int>(status__))                 \
            );                                                             \
        }                                                                  \
    } while (false)


template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(
        std::size_t count
    ) {
        allocate(count);
    }

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer(
        const DeviceBuffer&
    ) = delete;

    DeviceBuffer& operator=(
        const DeviceBuffer&
    ) = delete;

    DeviceBuffer(
        DeviceBuffer&& other
    ) noexcept
        : pointer_(other.pointer_),
          count_(other.count_) {
        other.pointer_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(
        DeviceBuffer&& other
    ) noexcept {
        if (this != &other) {
            release();

            pointer_ =
                other.pointer_;

            count_ =
                other.count_;

            other.pointer_ =
                nullptr;

            other.count_ =
                0;
        }

        return *this;
    }

    void allocate(
        std::size_t count
    ) {
        release();

        count_ = count;

        if (count_ > 0) {
            CUDA_CHECK(
                cudaMalloc(
                    reinterpret_cast<void**>(
                        &pointer_
                    ),
                    count_ *
                    sizeof(T)
                )
            );
        }
    }

    void zero() {
        if (count_ > 0) {
            CUDA_CHECK(
                cudaMemset(
                    pointer_,
                    0,
                    count_ *
                    sizeof(T)
                )
            );
        }
    }

    T* data() {
        return pointer_;
    }

    const T* data() const {
        return pointer_;
    }

    std::size_t size() const {
        return count_;
    }

private:
    void release() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
            pointer_ = nullptr;
            count_ = 0;
        }
    }

    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};


template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(
        std::size_t count
    ) {
        allocate(count);
    }

    ~PinnedBuffer() {
        release();
    }

    PinnedBuffer(
        const PinnedBuffer&
    ) = delete;

    PinnedBuffer& operator=(
        const PinnedBuffer&
    ) = delete;

    PinnedBuffer(
        PinnedBuffer&& other
    ) noexcept
        : pointer_(other.pointer_),
          count_(other.count_) {
        other.pointer_ = nullptr;
        other.count_ = 0;
    }

    PinnedBuffer& operator=(
        PinnedBuffer&& other
    ) noexcept {
        if (this != &other) {
            release();

            pointer_ =
                other.pointer_;

            count_ =
                other.count_;

            other.pointer_ =
                nullptr;

            other.count_ =
                0;
        }

        return *this;
    }

    void allocate(
        std::size_t count
    ) {
        release();

        count_ = count;

        if (count_ > 0) {
            CUDA_CHECK(
                cudaMallocHost(
                    reinterpret_cast<void**>(
                        &pointer_
                    ),
                    count_ *
                    sizeof(T)
                )
            );
        }
    }

    T* data() {
        return pointer_;
    }

    const T* data() const {
        return pointer_;
    }

    std::size_t size() const {
        return count_;
    }

private:
    void release() {
        if (pointer_ != nullptr) {
            cudaFreeHost(pointer_);
            pointer_ = nullptr;
            count_ = 0;
        }
    }

    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};


class CublasHandle {
public:
    CublasHandle() {
        CUBLAS_CHECK(
            cublasCreate(
                &handle_
            )
        );
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(
        const CublasHandle&
    ) = delete;

    CublasHandle& operator=(
        const CublasHandle&
    ) = delete;

    cublasHandle_t get() {
        return handle_;
    }

private:
    cublasHandle_t handle_ = nullptr;
};


// ============================================================
// CUDA kernels
// ============================================================

__global__ void gather_batch_kernel(
    const float* full_X,
    const float* full_y,
    const int* order,
    int global_start,
    int batch_rows,
    int full_rows,
    int features,
    int batch_leading_dimension,
    float* batch_X,
    float* batch_y
) {
    const int index =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    const int matrix_elements =
        batch_rows *
        features;

    if (index < matrix_elements) {
        const int local_row =
            index %
            batch_rows;

        const int feature =
            index /
            batch_rows;

        const int source_row =
            order[
                global_start +
                local_row
            ];

        batch_X[
            local_row +
            feature *
            batch_leading_dimension
        ] =
            full_X[
                source_row +
                feature *
                full_rows
            ];
    }

    if (index < batch_rows) {
        const int source_row =
            order[
                global_start +
                index
            ];

        batch_y[index] =
            full_y[source_row];
    }
}


__global__ void bias_relu_forward_kernel(
    float* pre_activation,
    float* activation,
    const float* bias,
    int rows,
    int columns,
    int leading_dimension
) {
    const int index =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    const int count =
        rows *
        columns;

    if (index < count) {
        const int row =
            index %
            rows;

        const int column =
            index /
            rows;

        const int matrix_index =
            row +
            column *
            leading_dimension;

        const float value =
            pre_activation[
                matrix_index
            ] +
            bias[column];

        pre_activation[
            matrix_index
        ] = value;

        activation[
            matrix_index
        ] =
            value >
            0.0F
            ? value
            : 0.0F;
    }
}


__global__ void add_output_bias_kernel(
    float* prediction,
    const float* bias,
    int rows
) {
    const int row =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    if (row < rows) {
        prediction[row] +=
            bias[0];
    }
}


__global__ void prediction_gradient_kernel(
    const float* prediction,
    const float* target,
    float* d_prediction,
    int rows,
    float inverse_batch
) {
    const int row =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    if (row < rows) {
        d_prediction[row] =
            2.0F *
            inverse_batch *
            (
                prediction[row] -
                target[row]
            );
    }
}


__global__ void relu_backward_kernel(
    float* gradient,
    const float* pre_activation,
    int rows,
    int columns,
    int leading_dimension
) {
    const int index =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    const int count =
        rows *
        columns;

    if (index < count) {
        const int row =
            index %
            rows;

        const int column =
            index /
            rows;

        const int matrix_index =
            row +
            column *
            leading_dimension;

        if (
            pre_activation[
                matrix_index
            ] <=
            0.0F
        ) {
            gradient[
                matrix_index
            ] = 0.0F;
        }
    }
}


__global__ void bias_gradient_kernel(
    const float* matrix_gradient,
    float* bias_gradient,
    int rows,
    int columns,
    int leading_dimension
) {
    const int column =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    if (column < columns) {
        float sum =
            0.0F;

        for (
            int row = 0;
            row < rows;
            ++row
        ) {
            sum +=
                matrix_gradient[
                    row +
                    column *
                    leading_dimension
                ];
        }

        bias_gradient[column] =
            sum;
    }
}


__global__ void adam_update_kernel(
    float* parameter,
    const float* gradient,
    float* first_moment,
    float* second_moment,
    std::size_t count,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float first_correction,
    float second_correction
) {
    const std::size_t index =
        static_cast<std::size_t>(
            blockIdx.x
        ) *
        blockDim.x +
        threadIdx.x;

    if (index < count) {
        const float gradient_value =
            gradient[index];

        const float first =
            beta1 *
            first_moment[index] +
            (
                1.0F -
                beta1
            ) *
            gradient_value;

        const float second =
            beta2 *
            second_moment[index] +
            (
                1.0F -
                beta2
            ) *
            gradient_value *
            gradient_value;

        first_moment[index] =
            first;

        second_moment[index] =
            second;

        const float corrected_first =
            first /
            first_correction;

        const float corrected_second =
            second /
            second_correction;

        parameter[index] -=
            learning_rate *
            corrected_first /
            (
                sqrtf(
                    corrected_second
                ) +
                epsilon
            );
    }
}


// ============================================================
// Host/device dataset preparation
// ============================================================

struct HostPinnedData {
    PinnedBuffer<float> X_train;
    PinnedBuffer<float> y_train;

    PinnedBuffer<float> X_validation;
    PinnedBuffer<float> y_validation;

    PinnedBuffer<float> X_test;
    PinnedBuffer<float> y_test;

    int n_train = 0;
    int n_validation = 0;
    int n_test = 0;
    int n_features = 3;
};


struct DeviceData {
    DeviceBuffer<float> X_train;
    DeviceBuffer<float> y_train;

    DeviceBuffer<float> X_validation;
    DeviceBuffer<float> y_validation;

    DeviceBuffer<float> X_test;
    DeviceBuffer<float> y_test;

    int n_train = 0;
    int n_validation = 0;
    int n_test = 0;
    int n_features = 3;
};


void copy_matrix_to_pinned_column_major(
    const Matrix& source,
    PinnedBuffer<float>& destination
) {
    const ColMatrix column_major =
        source;

    if (
        destination.size() !=
        static_cast<std::size_t>(
            column_major.size()
        )
    ) {
        throw std::runtime_error(
            "Pinned matrix destination has the wrong size."
        );
    }

    std::copy(
        column_major.data(),
        column_major.data() +
        column_major.size(),
        destination.data()
    );
}


void copy_vector_to_pinned(
    const Vector& source,
    PinnedBuffer<float>& destination
) {
    if (
        destination.size() !=
        static_cast<std::size_t>(
            source.size()
        )
    ) {
        throw std::runtime_error(
            "Pinned vector destination has the wrong size."
        );
    }

    std::copy(
        source.data(),
        source.data() +
        source.size(),
        destination.data()
    );
}


HostPinnedData make_pinned_data(
    const Dataset& scaled
) {
    HostPinnedData host{
        PinnedBuffer<float>(
            static_cast<std::size_t>(
                scaled.train.X.size()
            )
        ),
        PinnedBuffer<float>(
            static_cast<std::size_t>(
                scaled.train.y.size()
            )
        ),

        PinnedBuffer<float>(
            static_cast<std::size_t>(
                scaled.validation.X.size()
            )
        ),
        PinnedBuffer<float>(
            static_cast<std::size_t>(
                scaled.validation.y.size()
            )
        ),

        PinnedBuffer<float>(
            static_cast<std::size_t>(
                scaled.test.X.size()
            )
        ),
        PinnedBuffer<float>(
            static_cast<std::size_t>(
                scaled.test.y.size()
            )
        ),

        static_cast<int>(
            scaled.train.X.rows()
        ),
        static_cast<int>(
            scaled.validation.X.rows()
        ),
        static_cast<int>(
            scaled.test.X.rows()
        ),
        static_cast<int>(
            scaled.train.X.cols()
        )
    };

    copy_matrix_to_pinned_column_major(
        scaled.train.X,
        host.X_train
    );

    copy_vector_to_pinned(
        scaled.train.y,
        host.y_train
    );

    copy_matrix_to_pinned_column_major(
        scaled.validation.X,
        host.X_validation
    );

    copy_vector_to_pinned(
        scaled.validation.y,
        host.y_validation
    );

    copy_matrix_to_pinned_column_major(
        scaled.test.X,
        host.X_test
    );

    copy_vector_to_pinned(
        scaled.test.y,
        host.y_test
    );

    return host;
}


DeviceData allocate_device_data(
    const HostPinnedData& host
) {
    return DeviceData{
        DeviceBuffer<float>(
            host.X_train.size()
        ),
        DeviceBuffer<float>(
            host.y_train.size()
        ),

        DeviceBuffer<float>(
            host.X_validation.size()
        ),
        DeviceBuffer<float>(
            host.y_validation.size()
        ),

        DeviceBuffer<float>(
            host.X_test.size()
        ),
        DeviceBuffer<float>(
            host.y_test.size()
        ),

        host.n_train,
        host.n_validation,
        host.n_test,
        host.n_features
    };
}


void copy_all_data_to_device(
    const HostPinnedData& host,
    DeviceData& device
) {
    CUDA_CHECK(
        cudaMemcpyAsync(
            device.X_train.data(),
            host.X_train.data(),
            host.X_train.size() *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpyAsync(
            device.y_train.data(),
            host.y_train.data(),
            host.y_train.size() *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpyAsync(
            device.X_validation.data(),
            host.X_validation.data(),
            host.X_validation.size() *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpyAsync(
            device.y_validation.data(),
            host.y_validation.data(),
            host.y_validation.size() *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpyAsync(
            device.X_test.data(),
            host.X_test.data(),
            host.X_test.size() *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpyAsync(
            device.y_test.data(),
            host.y_test.data(),
            host.y_test.size() *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
}


std::vector<double> benchmark_h2d(
    const HostPinnedData& host,
    DeviceData& device,
    const Config& config
) {
    for (
        std::size_t repetition = 0;
        repetition <
        config.transfer_warmup_repetitions;
        ++repetition
    ) {
        copy_all_data_to_device(
            host,
            device
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );
    }

    std::vector<double> times;

    for (
        std::size_t repetition = 0;
        repetition <
        config.transfer_repetitions;
        ++repetition
    ) {
        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        const auto start =
            std::chrono::steady_clock::now();

        copy_all_data_to_device(
            host,
            device
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        const auto stop =
            std::chrono::steady_clock::now();

        times.push_back(
            elapsed_seconds(
                start,
                stop
            )
        );
    }

    return times;
}


// ============================================================
// Device model and workspace
// ============================================================

struct DeviceParameterArray {
    DeviceBuffer<float> value;
    DeviceBuffer<float> gradient;
    DeviceBuffer<float> first_moment;
    DeviceBuffer<float> second_moment;

    explicit DeviceParameterArray(
        std::size_t count
    )
        : value(count),
          gradient(count),
          first_moment(count),
          second_moment(count) {}
};


struct CudaModel {
    DeviceParameterArray W1;
    DeviceParameterArray b1;

    DeviceParameterArray W2;
    DeviceParameterArray b2;

    DeviceParameterArray W3;
    DeviceParameterArray b3;

    std::uint64_t step = 0;

    explicit CudaModel(
        const Config& config
    )
        : W1(
            static_cast<std::size_t>(
                config.n_input *
                config.hidden1
            )
          ),
          b1(
            static_cast<std::size_t>(
                config.hidden1
            )
          ),
          W2(
            static_cast<std::size_t>(
                config.hidden1 *
                config.hidden2
            )
          ),
          b2(
            static_cast<std::size_t>(
                config.hidden2
            )
          ),
          W3(
            static_cast<std::size_t>(
                config.hidden2 *
                config.n_output
            )
          ),
          b3(
            static_cast<std::size_t>(
                config.n_output
            )
          ) {}

    void zero_optimizer_state() {
        W1.first_moment.zero();
        W1.second_moment.zero();

        b1.first_moment.zero();
        b1.second_moment.zero();

        W2.first_moment.zero();
        W2.second_moment.zero();

        b2.first_moment.zero();
        b2.second_moment.zero();

        W3.first_moment.zero();
        W3.second_moment.zero();

        b3.first_moment.zero();
        b3.second_moment.zero();

        step = 0;
    }
};


void copy_host_matrix_to_device(
    const Matrix& host_matrix,
    DeviceBuffer<float>& device_buffer
) {
    const ColMatrix column_major =
        host_matrix;

    CUDA_CHECK(
        cudaMemcpy(
            device_buffer.data(),
            column_major.data(),
            static_cast<std::size_t>(
                column_major.size()
            ) *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
}


void copy_host_vector_to_device(
    const Vector& host_vector,
    DeviceBuffer<float>& device_buffer
) {
    CUDA_CHECK(
        cudaMemcpy(
            device_buffer.data(),
            host_vector.data(),
            static_cast<std::size_t>(
                host_vector.size()
            ) *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
}


void initialize_cuda_model(
    CudaModel& model,
    const Config& config
) {
    const Parameters host_parameters =
        initialize_parameters(config);

    copy_host_matrix_to_device(
        host_parameters.W1,
        model.W1.value
    );

    copy_host_vector_to_device(
        host_parameters.b1,
        model.b1.value
    );

    copy_host_matrix_to_device(
        host_parameters.W2,
        model.W2.value
    );

    copy_host_vector_to_device(
        host_parameters.b2,
        model.b2.value
    );

    copy_host_matrix_to_device(
        host_parameters.W3,
        model.W3.value
    );

    copy_host_vector_to_device(
        host_parameters.b3,
        model.b3.value
    );

    model.zero_optimizer_state();
}


struct Workspace {
    int maximum_rows = 0;
    int leading_dimension = 0;

    DeviceBuffer<float> X_batch;
    DeviceBuffer<float> y_batch;

    DeviceBuffer<float> z1;
    DeviceBuffer<float> a1;

    DeviceBuffer<float> z2;
    DeviceBuffer<float> a2;

    DeviceBuffer<float> prediction;
    DeviceBuffer<float> d_prediction;

    DeviceBuffer<float> d_a2;
    DeviceBuffer<float> d_z2;

    DeviceBuffer<float> d_a1;
    DeviceBuffer<float> d_z1;

    explicit Workspace(
        int maximum_rows_value,
        const Config& config
    )
        : maximum_rows(
            maximum_rows_value
          ),
          leading_dimension(
            maximum_rows_value
          ),
          X_batch(
            static_cast<std::size_t>(
                maximum_rows_value *
                config.n_input
            )
          ),
          y_batch(
            static_cast<std::size_t>(
                maximum_rows_value
            )
          ),
          z1(
            static_cast<std::size_t>(
                maximum_rows_value *
                config.hidden1
            )
          ),
          a1(
            static_cast<std::size_t>(
                maximum_rows_value *
                config.hidden1
            )
          ),
          z2(
            static_cast<std::size_t>(
                maximum_rows_value *
                config.hidden2
            )
          ),
          a2(
            static_cast<std::size_t>(
                maximum_rows_value *
                config.hidden2
            )
          ),
          prediction(
            static_cast<std::size_t>(
                maximum_rows_value
            )
          ),
          d_prediction(
            static_cast<std::size_t>(
                maximum_rows_value
            )
          ),
          d_a2(
            static_cast<std::size_t>(
                maximum_rows_value *
                config.hidden2
            )
          ),
          d_z2(
            static_cast<std::size_t>(
                maximum_rows_value *
                config.hidden2
            )
          ),
          d_a1(
            static_cast<std::size_t>(
                maximum_rows_value *
                config.hidden1
            )
          ),
          d_z1(
            static_cast<std::size_t>(
                maximum_rows_value *
                config.hidden1
            )
          ) {}
};


// ============================================================
// Forward and backward GPU operations
// ============================================================

void launch_bias_relu(
    float* z,
    float* a,
    const float* bias,
    int rows,
    int columns,
    int leading_dimension
) {
    constexpr int threads =
        256;

    const int count =
        rows *
        columns;

    const int blocks =
        (
            count +
            threads -
            1
        ) /
        threads;

    bias_relu_forward_kernel
        <<<blocks, threads>>>(
            z,
            a,
            bias,
            rows,
            columns,
            leading_dimension
        );

    CUDA_CHECK(
        cudaGetLastError()
    );
}


void launch_relu_backward(
    float* gradient,
    const float* z,
    int rows,
    int columns,
    int leading_dimension
) {
    constexpr int threads =
        256;

    const int count =
        rows *
        columns;

    const int blocks =
        (
            count +
            threads -
            1
        ) /
        threads;

    relu_backward_kernel
        <<<blocks, threads>>>(
            gradient,
            z,
            rows,
            columns,
            leading_dimension
        );

    CUDA_CHECK(
        cudaGetLastError()
    );
}


void launch_bias_gradient(
    const float* matrix_gradient,
    float* bias_gradient,
    int rows,
    int columns,
    int leading_dimension
) {
    constexpr int threads =
        128;

    const int blocks =
        (
            columns +
            threads -
            1
        ) /
        threads;

    bias_gradient_kernel
        <<<blocks, threads>>>(
            matrix_gradient,
            bias_gradient,
            rows,
            columns,
            leading_dimension
        );

    CUDA_CHECK(
        cudaGetLastError()
    );
}


void forward_gpu(
    cublasHandle_t handle,
    const float* X,
    int X_leading_dimension,
    int rows,
    const CudaModel& model,
    Workspace& workspace,
    const Config& config
) {
    const float alpha =
        1.0F;

    const float beta =
        0.0F;

    // z1 = X W1
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            rows,
            static_cast<int>(
                config.hidden1
            ),
            static_cast<int>(
                config.n_input
            ),
            &alpha,
            X,
            X_leading_dimension,
            model.W1.value.data(),
            static_cast<int>(
                config.n_input
            ),
            &beta,
            workspace.z1.data(),
            workspace.leading_dimension
        )
    );

    launch_bias_relu(
        workspace.z1.data(),
        workspace.a1.data(),
        model.b1.value.data(),
        rows,
        static_cast<int>(
            config.hidden1
        ),
        workspace.leading_dimension
    );

    // z2 = a1 W2
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            rows,
            static_cast<int>(
                config.hidden2
            ),
            static_cast<int>(
                config.hidden1
            ),
            &alpha,
            workspace.a1.data(),
            workspace.leading_dimension,
            model.W2.value.data(),
            static_cast<int>(
                config.hidden1
            ),
            &beta,
            workspace.z2.data(),
            workspace.leading_dimension
        )
    );

    launch_bias_relu(
        workspace.z2.data(),
        workspace.a2.data(),
        model.b2.value.data(),
        rows,
        static_cast<int>(
            config.hidden2
        ),
        workspace.leading_dimension
    );

    // prediction = a2 W3
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            rows,
            1,
            static_cast<int>(
                config.hidden2
            ),
            &alpha,
            workspace.a2.data(),
            workspace.leading_dimension,
            model.W3.value.data(),
            static_cast<int>(
                config.hidden2
            ),
            &beta,
            workspace.prediction.data(),
            workspace.leading_dimension
        )
    );

    constexpr int threads =
        256;

    const int blocks =
        (
            rows +
            threads -
            1
        ) /
        threads;

    add_output_bias_kernel
        <<<blocks, threads>>>(
            workspace.prediction.data(),
            model.b3.value.data(),
            rows
        );

    CUDA_CHECK(
        cudaGetLastError()
    );
}


void backward_gpu(
    cublasHandle_t handle,
    int rows,
    CudaModel& model,
    Workspace& workspace,
    const Config& config
) {
    const float alpha =
        1.0F;

    const float beta =
        0.0F;

    constexpr int threads =
        256;

    const int row_blocks =
        (
            rows +
            threads -
            1
        ) /
        threads;

    prediction_gradient_kernel
        <<<row_blocks, threads>>>(
            workspace.prediction.data(),
            workspace.y_batch.data(),
            workspace.d_prediction.data(),
            rows,
            1.0F /
            static_cast<float>(
                rows
            )
        );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // dW3 = a2^T dY
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(
                config.hidden2
            ),
            1,
            rows,
            &alpha,
            workspace.a2.data(),
            workspace.leading_dimension,
            workspace.d_prediction.data(),
            workspace.leading_dimension,
            &beta,
            model.W3.gradient.data(),
            static_cast<int>(
                config.hidden2
            )
        )
    );

    launch_bias_gradient(
        workspace.d_prediction.data(),
        model.b3.gradient.data(),
        rows,
        1,
        workspace.leading_dimension
    );

    // dA2 = dY W3^T
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            rows,
            static_cast<int>(
                config.hidden2
            ),
            1,
            &alpha,
            workspace.d_prediction.data(),
            workspace.leading_dimension,
            model.W3.value.data(),
            static_cast<int>(
                config.hidden2
            ),
            &beta,
            workspace.d_a2.data(),
            workspace.leading_dimension
        )
    );

    CUDA_CHECK(
        cudaMemcpy2D(
            workspace.d_z2.data(),
            static_cast<std::size_t>(workspace.leading_dimension) * sizeof(float),
            workspace.d_a2.data(),
            static_cast<std::size_t>(workspace.leading_dimension) * sizeof(float),
            static_cast<std::size_t>(rows) * sizeof(float),
            static_cast<std::size_t>(config.hidden2),
            cudaMemcpyDeviceToDevice
        )
    );

    launch_relu_backward(
        workspace.d_z2.data(),
        workspace.z2.data(),
        rows,
        static_cast<int>(
            config.hidden2
        ),
        workspace.leading_dimension
    );

    // dW2 = a1^T dZ2
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(
                config.hidden1
            ),
            static_cast<int>(
                config.hidden2
            ),
            rows,
            &alpha,
            workspace.a1.data(),
            workspace.leading_dimension,
            workspace.d_z2.data(),
            workspace.leading_dimension,
            &beta,
            model.W2.gradient.data(),
            static_cast<int>(
                config.hidden1
            )
        )
    );

    launch_bias_gradient(
        workspace.d_z2.data(),
        model.b2.gradient.data(),
        rows,
        static_cast<int>(
            config.hidden2
        ),
        workspace.leading_dimension
    );

    // dA1 = dZ2 W2^T
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            rows,
            static_cast<int>(
                config.hidden1
            ),
            static_cast<int>(
                config.hidden2
            ),
            &alpha,
            workspace.d_z2.data(),
            workspace.leading_dimension,
            model.W2.value.data(),
            static_cast<int>(
                config.hidden1
            ),
            &beta,
            workspace.d_a1.data(),
            workspace.leading_dimension
        )
    );

    CUDA_CHECK(
        cudaMemcpy2D(
            workspace.d_z1.data(),
            static_cast<std::size_t>(workspace.leading_dimension) * sizeof(float),
            workspace.d_a1.data(),
            static_cast<std::size_t>(workspace.leading_dimension) * sizeof(float),
            static_cast<std::size_t>(rows) * sizeof(float),
            static_cast<std::size_t>(config.hidden1),
            cudaMemcpyDeviceToDevice
        )
    );

    launch_relu_backward(
        workspace.d_z1.data(),
        workspace.z1.data(),
        rows,
        static_cast<int>(
            config.hidden1
        ),
        workspace.leading_dimension
    );

    // dW1 = X_batch^T dZ1
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(
                config.n_input
            ),
            static_cast<int>(
                config.hidden1
            ),
            rows,
            &alpha,
            workspace.X_batch.data(),
            workspace.leading_dimension,
            workspace.d_z1.data(),
            workspace.leading_dimension,
            &beta,
            model.W1.gradient.data(),
            static_cast<int>(
                config.n_input
            )
        )
    );

    launch_bias_gradient(
        workspace.d_z1.data(),
        model.b1.gradient.data(),
        rows,
        static_cast<int>(
            config.hidden1
        ),
        workspace.leading_dimension
    );
}


struct NumericalDifference {
    float maximum_absolute = 0.0F;
    float relative_l2 = 0.0F;
};


template <typename Expected, typename Actual>
NumericalDifference numerical_difference(
    const Eigen::MatrixBase<Expected>& expected,
    const Eigen::MatrixBase<Actual>& actual
) {
    const auto difference = (expected - actual).eval();

    return NumericalDifference{
        difference.cwiseAbs().maxCoeff(),
        difference.norm() /
        std::max(expected.norm(), 1.0e-12F)
    };
}


Matrix copy_device_matrix(
    const DeviceBuffer<float>& source,
    Eigen::Index rows,
    Eigen::Index columns
) {
    ColMatrix column_major(rows, columns);
    CUDA_CHECK(
        cudaMemcpy(
            column_major.data(),
            source.data(),
            static_cast<std::size_t>(column_major.size()) * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );
    return Matrix(column_major);
}


Vector copy_device_vector(
    const DeviceBuffer<float>& source,
    Eigen::Index size
) {
    Vector result(size);
    CUDA_CHECK(
        cudaMemcpy(
            result.data(),
            source.data(),
            static_cast<std::size_t>(size) * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );
    return result;
}


void run_cuda_correctness_check(
    CublasHandle& handle,
    const Dataset& scaled,
    const Config& config
) {
    constexpr Eigen::Index rows = 8;
    const Matrix X = scaled.train.X.topRows(rows);
    const Vector y = scaled.train.y.head(rows);
    const Parameters parameters = initialize_parameters(config);
    const Vector cpu_prediction = predict_standardized(X, parameters);
    const LossAndGradients cpu = mse_loss_and_backward(
        X,
        y,
        parameters,
        config,
        static_cast<float>(rows)
    );

    CudaModel model(config);
    initialize_cuda_model(model, config);
    Workspace workspace(static_cast<int>(rows), config);
    copy_host_matrix_to_device(X, workspace.X_batch);
    copy_host_vector_to_device(y, workspace.y_batch);

    forward_gpu(
        handle.get(),
        workspace.X_batch.data(),
        workspace.leading_dimension,
        static_cast<int>(rows),
        model,
        workspace,
        config
    );
    backward_gpu(
        handle.get(),
        static_cast<int>(rows),
        model,
        workspace,
        config
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    const NumericalDifference forward = numerical_difference(
        cpu_prediction,
        copy_device_vector(workspace.prediction, rows)
    );

    const std::array<std::pair<std::string, NumericalDifference>, 6> gradients{{
        {"dW1", numerical_difference(
            cpu.gradients.W1,
            copy_device_matrix(model.W1.gradient, config.n_input, config.hidden1)
        )},
        {"db1", numerical_difference(
            cpu.gradients.b1,
            copy_device_vector(model.b1.gradient, config.hidden1)
        )},
        {"dW2", numerical_difference(
            cpu.gradients.W2,
            copy_device_matrix(model.W2.gradient, config.hidden1, config.hidden2)
        )},
        {"db2", numerical_difference(
            cpu.gradients.b2,
            copy_device_vector(model.b2.gradient, config.hidden2)
        )},
        {"dW3", numerical_difference(
            cpu.gradients.W3,
            copy_device_matrix(model.W3.gradient, config.hidden2, config.n_output)
        )},
        {"db3", numerical_difference(
            cpu.gradients.b3,
            copy_device_vector(model.b3.gradient, config.n_output)
        )}
    }};

    float maximum_gradient_relative_l2 = 0.0F;
    std::cout
        << "\nCUDA/Eigen numerical comparison\n"
        << "--------------------------------\n"
        << "Forward max absolute    : " << forward.maximum_absolute << "\n"
        << "Forward relative L2     : " << forward.relative_l2 << "\n";

    for (const auto& [name, difference] : gradients) {
        maximum_gradient_relative_l2 = std::max(
            maximum_gradient_relative_l2,
            difference.relative_l2
        );
        std::cout
            << std::setw(4) << name
            << " max absolute       : " << difference.maximum_absolute
            << ", relative L2: " << difference.relative_l2 << "\n";
    }

    if (
        !std::isfinite(forward.maximum_absolute) ||
        !std::isfinite(forward.relative_l2) ||
        !std::isfinite(maximum_gradient_relative_l2) ||
        forward.maximum_absolute > 1.0e-3F ||
        maximum_gradient_relative_l2 > 1.0e-2F
    ) {
        throw std::runtime_error("CUDA/Eigen numerical comparison failed.");
    }

    std::cout << "CUDA numerical check    : PASS\n";
}


void launch_adam_update(
    DeviceParameterArray& array,
    const Config& config,
    std::uint64_t step
) {
    constexpr int threads =
        256;

    const int blocks =
        static_cast<int>(
            (
                array.value.size() +
                threads -
                1
            ) /
            threads
        );

    const float first_correction =
        1.0F -
        std::pow(
            config.beta1,
            static_cast<float>(
                step
            )
        );

    const float second_correction =
        1.0F -
        std::pow(
            config.beta2,
            static_cast<float>(
                step
            )
        );

    adam_update_kernel
        <<<blocks, threads>>>(
            array.value.data(),
            array.gradient.data(),
            array.first_moment.data(),
            array.second_moment.data(),
            array.value.size(),
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.adam_epsilon,
            first_correction,
            second_correction
        );

    CUDA_CHECK(
        cudaGetLastError()
    );
}


void adam_step_gpu(
    CudaModel& model,
    const Config& config
) {
    ++model.step;

    launch_adam_update(
        model.W1,
        config,
        model.step
    );

    launch_adam_update(
        model.b1,
        config,
        model.step
    );

    launch_adam_update(
        model.W2,
        config,
        model.step
    );

    launch_adam_update(
        model.b2,
        config,
        model.step
    );

    launch_adam_update(
        model.W3,
        config,
        model.step
    );

    launch_adam_update(
        model.b3,
        config,
        model.step
    );
}


// ============================================================
// Training and inference benchmarks
// ============================================================

void gather_training_batch(
    const DeviceData& data,
    const DeviceBuffer<int>& order,
    int global_start,
    int batch_rows,
    Workspace& workspace
) {
    constexpr int threads =
        256;

    const int work_items =
        std::max(
            batch_rows *
            data.n_features,
            batch_rows
        );

    const int blocks =
        (
            work_items +
            threads -
            1
        ) /
        threads;

    gather_batch_kernel
        <<<blocks, threads>>>(
            data.X_train.data(),
            data.y_train.data(),
            order.data(),
            global_start,
            batch_rows,
            data.n_train,
            data.n_features,
            workspace.leading_dimension,
            workspace.X_batch.data(),
            workspace.y_batch.data()
        );

    CUDA_CHECK(
        cudaGetLastError()
    );
}


void train_steps_cuda(
    CublasHandle& handle,
    CudaModel& model,
    Workspace& workspace,
    const DeviceData& data,
    const Config& config,
    PinnedBuffer<int>& host_order,
    DeviceBuffer<int>& device_order,
    std::mt19937& shuffle_generator,
    std::size_t maximum_steps,
    bool full_epochs
) {
    std::iota(
        host_order.data(),
        host_order.data() +
        host_order.size(),
        0
    );

    std::size_t completed_steps =
        0;

    for (
        std::size_t epoch = 0;
        epoch < config.epochs;
        ++epoch
    ) {
        std::shuffle(
            host_order.data(),
            host_order.data() +
            host_order.size(),
            shuffle_generator
        );

        CUDA_CHECK(
            cudaMemcpyAsync(
                device_order.data(),
                host_order.data(),
                host_order.size() *
                sizeof(int),
                cudaMemcpyHostToDevice
            )
        );

        for (
            int global_start = 0;
            global_start <
            data.n_train;
            global_start +=
            static_cast<int>(
                config.batch_size
            )
        ) {
            const int batch_rows =
                std::min(
                    static_cast<int>(
                        config.batch_size
                    ),
                    data.n_train -
                    global_start
                );

            gather_training_batch(
                data,
                device_order,
                global_start,
                batch_rows,
                workspace
            );

            forward_gpu(
                handle.get(),
                workspace.X_batch.data(),
                workspace.leading_dimension,
                batch_rows,
                model,
                workspace,
                config
            );

            backward_gpu(
                handle.get(),
                batch_rows,
                model,
                workspace,
                config
            );

            adam_step_gpu(
                model,
                config
            );

            ++completed_steps;

            if (
                !full_epochs &&
                completed_steps >=
                maximum_steps
            ) {
                return;
            }
        }
    }
}


void warm_up_training_cuda(
    CublasHandle& handle,
    const DeviceData& data,
    const Config& config,
    int maximum_rows
) {
    CudaModel model(config);
    initialize_cuda_model(
        model,
        config
    );

    Workspace workspace(
        maximum_rows,
        config
    );

    PinnedBuffer<int> host_order(
        static_cast<std::size_t>(
            data.n_train
        )
    );

    DeviceBuffer<int> device_order(
        static_cast<std::size_t>(
            data.n_train
        )
    );

    std::mt19937 shuffle_generator(
        config.shuffle_seed
    );

    train_steps_cuda(
        handle,
        model,
        workspace,
        data,
        config,
        host_order,
        device_order,
        shuffle_generator,
        config.training_warmup_steps,
        false
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}


struct CudaTrainingBenchmark {
    std::vector<double> times;
    std::unique_ptr<CudaModel> final_model;
    std::unique_ptr<Workspace> final_workspace;
};


CudaTrainingBenchmark benchmark_training_cuda(
    CublasHandle& handle,
    const DeviceData& data,
    const Config& config,
    int maximum_rows
) {
    std::vector<double> times;

    std::unique_ptr<CudaModel>
        final_model;

    std::unique_ptr<Workspace>
        final_workspace;

    warm_up_training_cuda(
        handle,
        data,
        config,
        maximum_rows
    );

    for (
        std::size_t repetition = 0;
        repetition <
        config.training_repetitions;
        ++repetition
    ) {
        auto model =
            std::make_unique<CudaModel>(
                config
            );

        initialize_cuda_model(
            *model,
            config
        );

        auto workspace =
            std::make_unique<Workspace>(
                maximum_rows,
                config
            );

        PinnedBuffer<int> host_order(
            static_cast<std::size_t>(
                data.n_train
            )
        );

        DeviceBuffer<int> device_order(
            static_cast<std::size_t>(
                data.n_train
            )
        );

        std::mt19937 shuffle_generator(
            config.shuffle_seed
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        const auto start =
            std::chrono::steady_clock::now();

        train_steps_cuda(
            handle,
            *model,
            *workspace,
            data,
            config,
            host_order,
            device_order,
            shuffle_generator,
            0,
            true
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        const auto stop =
            std::chrono::steady_clock::now();

        times.push_back(
            elapsed_seconds(
                start,
                stop
            )
        );

        final_model =
            std::move(model);

        final_workspace =
            std::move(workspace);
    }

    return CudaTrainingBenchmark{
        std::move(times),
        std::move(final_model),
        std::move(final_workspace)
    };
}


void warm_up_inference_cuda(
    CublasHandle& handle,
    const float* X,
    int X_leading_dimension,
    int rows,
    const CudaModel& model,
    Workspace& workspace,
    const Config& config
) {
    for (
        std::size_t loop = 0;
        loop <
        config.inference_warmup_loops;
        ++loop
    ) {
        forward_gpu(
            handle.get(),
            X,
            X_leading_dimension,
            rows,
            model,
            workspace,
            config
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}


std::vector<double> benchmark_inference_cuda(
    CublasHandle& handle,
    const float* X,
    int X_leading_dimension,
    int rows,
    const CudaModel& model,
    Workspace& workspace,
    const Config& config
) {
    warm_up_inference_cuda(
        handle,
        X,
        X_leading_dimension,
        rows,
        model,
        workspace,
        config
    );

    std::vector<double> times;

    for (
        std::size_t repetition = 0;
        repetition <
        config.inference_repetitions;
        ++repetition
    ) {
        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        const auto start =
            std::chrono::steady_clock::now();

        for (
            std::size_t loop = 0;
            loop <
            config.inference_loops_per_repetition;
            ++loop
        ) {
            forward_gpu(
                handle.get(),
                X,
                X_leading_dimension,
                rows,
                model,
                workspace,
                config
            );
        }

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        const auto stop =
            std::chrono::steady_clock::now();

        times.push_back(
            elapsed_seconds(
                start,
                stop
            )
        );
    }

    return times;
}


Vector copy_prediction_once(
    const DeviceBuffer<float>& prediction,
    int rows
) {
    Vector result(rows);

    CUDA_CHECK(
        cudaMemcpy(
            result.data(),
            prediction.data(),
            static_cast<std::size_t>(
                rows
            ) *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    return result;
}


std::pair<
    std::vector<double>,
    Vector
> copy_predictions_to_host(
    const DeviceBuffer<float>& prediction,
    int rows,
    const Config& config
) {
    PinnedBuffer<float> host_prediction(
        static_cast<std::size_t>(
            rows
        )
    );

    for (
        std::size_t repetition = 0;
        repetition <
        config.transfer_warmup_repetitions;
        ++repetition
    ) {
        CUDA_CHECK(
            cudaMemcpyAsync(
                host_prediction.data(),
                prediction.data(),
                static_cast<std::size_t>(
                    rows
                ) *
                sizeof(float),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );
    }

    std::vector<double> times;

    for (
        std::size_t repetition = 0;
        repetition <
        config.transfer_repetitions;
        ++repetition
    ) {
        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        const auto start =
            std::chrono::steady_clock::now();

        CUDA_CHECK(
            cudaMemcpyAsync(
                host_prediction.data(),
                prediction.data(),
                static_cast<std::size_t>(
                    rows
                ) *
                sizeof(float),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        const auto stop =
            std::chrono::steady_clock::now();

        times.push_back(
            elapsed_seconds(
                start,
                stop
            )
        );
    }

    Vector result(rows);

    std::copy(
        host_prediction.data(),
        host_prediction.data() +
        rows,
        result.data()
    );

    return {
        std::move(times),
        std::move(result)
    };
}

}  // namespace


#ifndef MLP_CUDA_LIBRARY
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

        Eigen::setNbThreads(1);

        int device_count =
            0;

        CUDA_CHECK(
            cudaGetDeviceCount(
                &device_count
            )
        );

        if (device_count < 1) {
            throw std::runtime_error(
                "No CUDA-capable GPU was found."
            );
        }

        CUDA_CHECK(
            cudaSetDevice(0)
        );

        cudaDeviceProp properties{};

        CUDA_CHECK(
            cudaGetDeviceProperties(
                &properties,
                0
            )
        );

        // Initialize the CUDA context before any benchmark.
        CUDA_CHECK(
            cudaFree(nullptr)
        );

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

        HostPinnedData host_data =
            make_pinned_data(
                scaled
            );

        DeviceData device_data =
            allocate_device_data(
                host_data
            );

        CublasHandle handle;

        if (config.smoke_test) {
            run_cuda_correctness_check(handle, scaled, config);
        }

        std::cout
            << "============================================================\n"
            << "CUDA C++ + cuBLAS MLP PROJECTILE SURROGATE\n"
            << "============================================================\n"
            << "Dataset                 : "
            << config.dataset_path
            << "\n"
            << "GPU                     : "
            << properties.name
            << "\n"
            << "Architecture            : "
            << config.n_input << "-"
            << config.hidden1 << "-"
            << config.hidden2 << "-"
            << config.n_output << "\n"
            << "Epochs                  : "
            << config.epochs
            << "\n"
            << "Batch size              : "
            << config.batch_size
            << "\n"
            << "Precision               : float32\n"
            << "Dense math              : cuBLAS SGEMM\n"
            << "Custom kernels          : ReLU, bias, gather, Adam\n";

        const std::vector<double>
            h2d_times =
                benchmark_h2d(
                    host_data,
                    device_data,
                    config
                );

        const double median_h2d =
            print_times(
                "CUDA host-to-device transfer",
                h2d_times
            );

        const std::size_t transferred_bytes =
            (
                host_data.X_train.size() +
                host_data.y_train.size() +
                host_data.X_validation.size() +
                host_data.y_validation.size() +
                host_data.X_test.size() +
                host_data.y_test.size()
            ) *
            sizeof(float);

        std::cout
            << "H2D bandwidth            : "
            << (
                static_cast<double>(
                    transferred_bytes
                ) /
                median_h2d /
                1.0e9
            )
            << " GB/s\n";

        const int maximum_rows =
            std::max(
                {
                    static_cast<int>(
                        config.batch_size
                    ),
                    device_data.n_validation,
                    device_data.n_test
                }
            );

        CudaTrainingBenchmark training =
            benchmark_training_cuda(
                handle,
                device_data,
                config,
                maximum_rows
            );

        const double median_training =
            print_times(
                "CUDA C++ training",
                training.times
            );

        const double training_throughput =
            static_cast<double>(
                device_data.n_train
            ) *
            static_cast<double>(
                config.epochs
            ) /
            median_training;

        std::cout
            << "Training throughput      : "
            << training_throughput
            << " samples/s\n"
            << "Training + median H2D    : "
            << (
                median_training +
                median_h2d
            )
            << " s\n";

        if (
            config.reference_training_seconds >
            0.0
        ) {
            std::cout
                << "Speedup versus reference : "
                << (
                    config.reference_training_seconds /
                    median_training
                )
                << "x\n";
        } else {
            std::cout
                << "Speedup versus reference : N/A\n";
        }

        // Validation accuracy is calculated outside timed training.
        forward_gpu(
            handle.get(),
            device_data.X_validation.data(),
            device_data.n_validation,
            device_data.n_validation,
            *training.final_model,
            *training.final_workspace,
            config
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        const Vector validation_standardized =
            copy_prediction_once(
                training.final_workspace->prediction,
                device_data.n_validation
            );

        const Vector validation_physical =
            inverse_transform_targets(
                validation_standardized,
                y_scaler
            );

        print_metrics(
            "Validation accuracy",
            calculate_metrics(
                raw.validation.y,
                validation_physical
            )
        );

        // One test forward pass remains GPU-resident.
        forward_gpu(
            handle.get(),
            device_data.X_test.data(),
            device_data.n_test,
            device_data.n_test,
            *training.final_model,
            *training.final_workspace,
            config
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        const auto [
            d2h_times,
            standardized_prediction
        ] =
            copy_predictions_to_host(
                training.final_workspace->prediction,
                device_data.n_test,
                config
            );

        const double median_d2h =
            print_times(
                "CUDA device-to-host prediction transfer",
                d2h_times
            );

        const Vector physical_prediction =
            inverse_transform_targets(
                standardized_prediction,
                y_scaler
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

        const std::vector<double>
            inference_times =
                benchmark_inference_cuda(
                    handle,
                    device_data.X_test.data(),
                    device_data.n_test,
                    device_data.n_test,
                    *training.final_model,
                    *training.final_workspace,
                    config
                );

        const double median_inference =
            print_times(
                "CUDA C++ inference",
                inference_times
            );

        const double inference_throughput =
            static_cast<double>(
                device_data.n_test
            ) *
            static_cast<double>(
                config.inference_loops_per_repetition
            ) /
            median_inference;

        std::cout
            << "Inference throughput     : "
            << inference_throughput
            << " predictions/s\n"
            << "Inference + median D2H   : "
            << (
                median_inference +
                median_d2h
            )
            << " s\n";

        print_example_predictions(
            raw.test,
            physical_prediction
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
#endif
