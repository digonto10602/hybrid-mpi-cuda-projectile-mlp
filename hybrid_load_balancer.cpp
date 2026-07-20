#include "hybrid_load_balancer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace hybrid {

Allocation allocate_samples(
    int sample_count,
    const std::vector<double>& weights,
    const std::vector<int>& active
) {
    if (sample_count < 0 || weights.size() != active.size()) {
        throw std::invalid_argument("Invalid load-balancer input.");
    }

    const std::size_t size = weights.size();
    std::vector<int> counts(size, 0);
    std::vector<double> usable(size, 0.0);
    std::vector<std::size_t> workers;

    for (std::size_t rank = 0; rank < size; ++rank) {
        if (active[rank] != 0 && workers.size() < static_cast<std::size_t>(sample_count)) {
            workers.push_back(rank);
            usable[rank] = std::max(weights[rank], 0.0);
        }
    }

    if (sample_count > 0 && workers.empty()) {
        throw std::runtime_error("No active worker can receive samples.");
    }

    int remaining = sample_count;
    for (const std::size_t rank : workers) {
        counts[rank] = 1;
        --remaining;
    }

    double total_weight = 0.0;
    for (const std::size_t rank : workers) {
        total_weight += usable[rank];
    }
    if (total_weight <= 0.0) {
        total_weight = static_cast<double>(workers.size());
        for (const std::size_t rank : workers) usable[rank] = 1.0;
    }

    std::vector<std::pair<double, std::size_t>> remainder;
    int assigned = 0;
    for (const std::size_t rank : workers) {
        const double exact = static_cast<double>(remaining) * usable[rank] / total_weight;
        const int whole = static_cast<int>(std::floor(exact));
        counts[rank] += whole;
        assigned += whole;
        remainder.emplace_back(exact - static_cast<double>(whole), rank);
    }
    std::sort(remainder.begin(), remainder.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) return left.first > right.first;
        return left.second < right.second;
    });
    for (int extra = 0; extra < remaining - assigned; ++extra) {
        ++counts[remainder[static_cast<std::size_t>(extra)].second];
    }

    std::vector<int> offsets(size, 0);
    for (std::size_t rank = 1; rank < size; ++rank) {
        offsets[rank] = offsets[rank - 1] + counts[rank - 1];
    }
    if (std::accumulate(counts.begin(), counts.end(), 0) != sample_count) {
        throw std::runtime_error("Load balancer did not assign every sample exactly once.");
    }
    return Allocation{std::move(counts), std::move(offsets)};
}

}  // namespace hybrid
