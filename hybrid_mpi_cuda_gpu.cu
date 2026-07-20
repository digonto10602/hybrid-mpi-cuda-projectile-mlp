#include "hybrid_mpi_cuda_gpu.hpp"

#define MLP_CUDA_LIBRARY 1
#include "mlp_cuda.cu"

#include <chrono>

namespace hybrid {
namespace {

__global__ void scale_values(float* values, std::size_t count, float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] *= scale;
}

template <typename Array>
void scale_gradient(Array& array, float scale) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((array.gradient.size() + threads - 1) / threads);
    scale_values<<<blocks, threads>>>(array.gradient.data(), array.gradient.size(), scale);
    CUDA_CHECK(cudaGetLastError());
}

double seconds(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

struct GpuWorker::Impl {
    int device = 0;
    Config config;
    std::unique_ptr<CublasHandle> handle;
    std::unique_ptr<CudaModel> model;
    std::unique_ptr<Workspace> workspace;
    mutable PinnedBuffer<float> staging;
    PinnedBuffer<float> batch_staging;

    Impl(int device_value, const Config& config_value, int maximum_rows)
        : device(device_value), config(config_value), staging(parameter_count(config)),
          batch_staging(static_cast<std::size_t>(maximum_rows) *
              static_cast<std::size_t>(config.n_input + 1)) {
        CUDA_CHECK(cudaSetDevice(device));
        handle = std::make_unique<CublasHandle>();
        model = std::make_unique<CudaModel>(config);
        workspace = std::make_unique<Workspace>(maximum_rows, config);
        if (parameter_count(config) != 4481U) {
            throw std::runtime_error("Hybrid GPU worker requires exactly 4,481 parameters.");
        }
    }

    template <typename Values>
    void stage_values(const Values& values, std::size_t& offset) {
        std::copy(values.data(), values.data() + values.size(), staging.data() + offset);
        offset += static_cast<std::size_t>(values.size());
    }

    void stage_matrix(const Matrix& matrix, std::size_t& offset) {
        for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
            for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
                staging.data()[offset++] = matrix(row, column);
            }
        }
    }

    template <typename DeviceArray>
    void upload(DeviceArray& destination, std::size_t& offset) {
        CUDA_CHECK(cudaMemcpyAsync(
            destination.data(), staging.data() + offset,
            destination.size() * sizeof(float), cudaMemcpyHostToDevice));
        offset += destination.size();
    }

    template <typename DeviceArray>
    void download(const DeviceArray& source, std::size_t& offset) const {
        CUDA_CHECK(cudaMemcpyAsync(
            staging.data() + offset, source.data(),
            source.size() * sizeof(float), cudaMemcpyDeviceToHost));
        offset += source.size();
    }

    void upload_parameters(const Parameters& parameters) {
        std::size_t offset = 0;
        stage_matrix(parameters.W1, offset); stage_values(parameters.b1, offset);
        stage_matrix(parameters.W2, offset); stage_values(parameters.b2, offset);
        stage_matrix(parameters.W3, offset); stage_values(parameters.b3, offset);
        if (offset != staging.size()) throw std::runtime_error("GPU parameter staging size mismatch.");
        offset = 0;
        upload(model->W1.value, offset); upload(model->b1.value, offset);
        upload(model->W2.value, offset); upload(model->b2.value, offset);
        upload(model->W3.value, offset); upload(model->b3.value, offset);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void upload_gradients(const Gradients& gradients) {
        std::size_t offset = 0;
        stage_matrix(gradients.W1, offset); stage_values(gradients.b1, offset);
        stage_matrix(gradients.W2, offset); stage_values(gradients.b2, offset);
        stage_matrix(gradients.W3, offset); stage_values(gradients.b3, offset);
        offset = 0;
        upload(model->W1.gradient, offset); upload(model->b1.gradient, offset);
        upload(model->W2.gradient, offset); upload(model->b2.gradient, offset);
        upload(model->W3.gradient, offset); upload(model->b3.gradient, offset);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    template <typename MatrixType>
    void read_matrix(MatrixType& destination, int rows, int columns, std::size_t& offset) const {
        for (int column = 0; column < columns; ++column) {
            for (int row = 0; row < rows; ++row) {
                destination(row, column) = staging.data()[offset++];
            }
        }
    }

    template <typename VectorType>
    void read_vector(VectorType& destination, int count, std::size_t& offset) const {
        for (int index = 0; index < count; ++index) destination(index) = staging.data()[offset++];
    }

    void download_gradients(Gradients& gradients) const {
        std::size_t offset = 0;
        download(model->W1.gradient, offset); download(model->b1.gradient, offset);
        download(model->W2.gradient, offset); download(model->b2.gradient, offset);
        download(model->W3.gradient, offset); download(model->b3.gradient, offset);
        CUDA_CHECK(cudaDeviceSynchronize());
        offset = 0;
        read_matrix(gradients.W1, static_cast<int>(config.n_input), static_cast<int>(config.hidden1), offset);
        read_vector(gradients.b1, static_cast<int>(config.hidden1), offset);
        read_matrix(gradients.W2, static_cast<int>(config.hidden1), static_cast<int>(config.hidden2), offset);
        read_vector(gradients.b2, static_cast<int>(config.hidden2), offset);
        read_matrix(gradients.W3, static_cast<int>(config.hidden2), static_cast<int>(config.n_output), offset);
        read_vector(gradients.b3, static_cast<int>(config.n_output), offset);
    }

    void download_parameters(Parameters& parameters) const {
        std::size_t offset = 0;
        download(model->W1.value, offset); download(model->b1.value, offset);
        download(model->W2.value, offset); download(model->b2.value, offset);
        download(model->W3.value, offset); download(model->b3.value, offset);
        CUDA_CHECK(cudaDeviceSynchronize());
        offset = 0;
        read_matrix(parameters.W1, static_cast<int>(config.n_input), static_cast<int>(config.hidden1), offset);
        read_vector(parameters.b1, static_cast<int>(config.hidden1), offset);
        read_matrix(parameters.W2, static_cast<int>(config.hidden1), static_cast<int>(config.hidden2), offset);
        read_vector(parameters.b2, static_cast<int>(config.hidden2), offset);
        read_matrix(parameters.W3, static_cast<int>(config.hidden2), static_cast<int>(config.n_output), offset);
        read_vector(parameters.b3, static_cast<int>(config.n_output), offset);
    }

    void upload_batch(const Batch& batch) {
        const std::size_t rows = static_cast<std::size_t>(batch.X.rows());
        for (Eigen::Index column = 0; column < batch.X.cols(); ++column) {
            for (Eigen::Index row = 0; row < batch.X.rows(); ++row) {
                batch_staging.data()[static_cast<std::size_t>(column) * rows +
                    static_cast<std::size_t>(row)] = batch.X(row, column);
            }
        }
        float* staged_y = batch_staging.data() + rows * static_cast<std::size_t>(config.n_input);
        std::copy(batch.y.data(), batch.y.data() + batch.y.size(), staged_y);
        CUDA_CHECK(cudaMemcpy2DAsync(
            workspace->X_batch.data(), static_cast<std::size_t>(workspace->leading_dimension) * sizeof(float),
            batch_staging.data(), rows * sizeof(float),
            rows * sizeof(float), static_cast<std::size_t>(batch.X.cols()),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(
            workspace->y_batch.data(), staged_y,
            static_cast<std::size_t>(batch.y.size()) * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

std::vector<GpuInfo> discover_gpus() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    std::vector<GpuInfo> result;
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        result.push_back(GpuInfo{device, properties.name, properties.major, properties.minor,
            static_cast<std::size_t>(properties.totalGlobalMem)});
    }
    return result;
}

GpuWorker::GpuWorker(int device, const Config& config, int maximum_rows)
    : impl_(std::make_unique<Impl>(device, config, maximum_rows)) {}

GpuWorker::~GpuWorker() = default;

void GpuWorker::set_parameters(const Parameters& parameters, bool reset_optimizer) {
    CUDA_CHECK(cudaSetDevice(impl_->device));
    impl_->upload_parameters(parameters);
    if (reset_optimizer) impl_->model->zero_optimizer_state();
}

Parameters GpuWorker::parameters() const {
    CUDA_CHECK(cudaSetDevice(impl_->device));
    Parameters result = zero_parameters(impl_->config);
    impl_->download_parameters(result);
    return result;
}

GpuTimings GpuWorker::local_gradient(
    const Batch& batch,
    float global_normalization,
    Gradients& gradients
) {
    if (batch.X.rows() == 0) {
        gradients = zero_gradients(impl_->config);
        return {};
    }
    CUDA_CHECK(cudaSetDevice(impl_->device));
    const auto h2d_start = std::chrono::steady_clock::now();
    impl_->upload_batch(batch);
    const double h2d = seconds(h2d_start);
    const int rows = static_cast<int>(batch.X.rows());
    const auto compute_start = std::chrono::steady_clock::now();
    forward_gpu(impl_->handle->get(), impl_->workspace->X_batch.data(),
        impl_->workspace->leading_dimension, rows, *impl_->model, *impl_->workspace, impl_->config);
    backward_gpu(impl_->handle->get(), rows, *impl_->model, *impl_->workspace, impl_->config);
    const float scale = static_cast<float>(rows) / global_normalization;
    scale_gradient(impl_->model->W1, scale); scale_gradient(impl_->model->b1, scale);
    scale_gradient(impl_->model->W2, scale); scale_gradient(impl_->model->b2, scale);
    scale_gradient(impl_->model->W3, scale); scale_gradient(impl_->model->b3, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double compute = seconds(compute_start);
    const auto d2h_start = std::chrono::steady_clock::now();
    impl_->download_gradients(gradients);
    return GpuTimings{h2d, compute, seconds(d2h_start), 0.0};
}

GpuTimings GpuWorker::adam_step(const Gradients& gradients) {
    CUDA_CHECK(cudaSetDevice(impl_->device));
    const auto h2d_start = std::chrono::steady_clock::now();
    impl_->upload_gradients(gradients);
    const double h2d = seconds(h2d_start);
    const auto adam_start = std::chrono::steady_clock::now();
    adam_step_gpu(*impl_->model, impl_->config);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double adam = seconds(adam_start);
    return GpuTimings{h2d, 0.0, 0.0, adam};
}

Vector GpuWorker::predict(const Matrix& X) {
    if (X.rows() > impl_->workspace->maximum_rows) throw std::runtime_error("GPU prediction batch is too large.");
    Batch batch{X, Vector::Zero(X.rows())};
    impl_->upload_batch(batch);
    forward_gpu(impl_->handle->get(), impl_->workspace->X_batch.data(), impl_->workspace->leading_dimension,
        static_cast<int>(X.rows()), *impl_->model, *impl_->workspace, impl_->config);
    CUDA_CHECK(cudaDeviceSynchronize());
    return copy_prediction_once(impl_->workspace->prediction, static_cast<int>(X.rows()));
}

void GpuWorker::synchronize() const {
    CUDA_CHECK(cudaSetDevice(impl_->device));
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace hybrid
