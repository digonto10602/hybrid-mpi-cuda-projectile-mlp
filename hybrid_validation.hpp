#pragma once

#include "mlp_common.hpp"

#include <array>
#include <string>

namespace hybrid {

struct Difference {
    double maximum_absolute = 0.0;
    double relative_l2 = 0.0;
};

struct GradientDifferences {
    std::array<Difference, 6> groups{};
    bool passed = false;
};

GradientDifferences compare_gradients(
    const mlp::Gradients& reference,
    const mlp::Gradients& actual,
    double relative_tolerance
);

bool finite_parameters(const mlp::Parameters& parameters);
bool finite_gradients(const mlp::Gradients& gradients);
const std::array<const char*, 6>& gradient_group_names();

}  // namespace hybrid
