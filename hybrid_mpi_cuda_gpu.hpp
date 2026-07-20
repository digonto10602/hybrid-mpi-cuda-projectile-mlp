#pragma once

#include "mlp_common.hpp"

#include <memory>
#include <string>
#include <vector>

namespace hybrid {

struct GpuInfo {
    int index = -1;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t memory_bytes = 0;
};

struct GpuTimings {
    double host_to_device = 0.0;
    double compute = 0.0;
    double device_to_host = 0.0;
    double adam = 0.0;
};

std::vector<GpuInfo> discover_gpus();

class GpuWorker {
public:
    GpuWorker(int device, const mlp::Config& config, int maximum_rows);
    ~GpuWorker();
    GpuWorker(const GpuWorker&) = delete;
    GpuWorker& operator=(const GpuWorker&) = delete;

    void set_parameters(const mlp::Parameters& parameters, bool reset_optimizer = false);
    mlp::Parameters parameters() const;
    GpuTimings local_gradient(
        const mlp::Batch& batch,
        float global_normalization,
        mlp::Gradients& gradients
    );
    GpuTimings adam_step(const mlp::Gradients& gradients);
    mlp::Vector predict(const mlp::Matrix& X);
    void synchronize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hybrid
