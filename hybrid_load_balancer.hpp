#pragma once

#include <vector>

namespace hybrid {

struct Allocation {
    std::vector<int> counts;
    std::vector<int> offsets;
};

Allocation allocate_samples(
    int sample_count,
    const std::vector<double>& weights,
    const std::vector<int>& active
);

}  // namespace hybrid
