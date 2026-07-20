#include "hybrid_validation.hpp"

#include <algorithm>
#include <cmath>

namespace hybrid {
namespace {

template <typename Left, typename Right>
Difference difference(const Left& reference, const Right& actual) {
    const auto delta = (reference - actual).eval();
    return Difference{
        static_cast<double>(delta.cwiseAbs().maxCoeff()),
        static_cast<double>(delta.norm() / std::max(reference.norm(), 1.0e-12F))
    };
}

template <typename Values>
bool finite(const Values& values) {
    return values.array().isFinite().all();
}

}  // namespace

const std::array<const char*, 6>& gradient_group_names() {
    static const std::array<const char*, 6> names{"W1", "b1", "W2", "b2", "W3", "b3"};
    return names;
}

GradientDifferences compare_gradients(
    const mlp::Gradients& reference,
    const mlp::Gradients& actual,
    double relative_tolerance
) {
    GradientDifferences result{{
        difference(reference.W1, actual.W1),
        difference(reference.b1, actual.b1),
        difference(reference.W2, actual.W2),
        difference(reference.b2, actual.b2),
        difference(reference.W3, actual.W3),
        difference(reference.b3, actual.b3)
    }, true};
    for (const Difference& group : result.groups) {
        result.passed = result.passed && std::isfinite(group.maximum_absolute) &&
            std::isfinite(group.relative_l2) && group.relative_l2 <= relative_tolerance;
    }
    return result;
}

bool finite_parameters(const mlp::Parameters& p) {
    return finite(p.W1) && finite(p.b1) && finite(p.W2) && finite(p.b2) &&
        finite(p.W3) && finite(p.b3);
}

bool finite_gradients(const mlp::Gradients& g) {
    return finite(g.W1) && finite(g.b1) && finite(g.W2) && finite(g.b2) &&
        finite(g.W3) && finite(g.b3);
}

}  // namespace hybrid
