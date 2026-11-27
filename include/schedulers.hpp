#ifndef MINIML_SCHEDULERS_HPP
#define MINIML_SCHEDULERS_HPP

// Opt-in learning-rate schedulers. Nothing in src/ includes this header, so
// including it explicitly is the feature gate; existing nn.hpp semantics are
// unchanged. Schedulers only compute a learning rate from a step index — they
// do not mutate tensors.

#include <cmath>
#include <cstddef>
#include <stdexcept>

// Same rate at every step.
class ConstantLR
{
    float lr_;

public:
    explicit ConstantLR(float lr) : lr_(lr) {}

    float base_lr() const { return lr_; }
    float get_lr(std::size_t /*step*/) const { return lr_; }
};

// lr(step) = base_lr * gamma^(floor(step / step_size)).
class StepLR
{
    float lr_;
    float gamma_;
    std::size_t step_size_;

public:
    StepLR(float lr, float gamma, std::size_t step_size)
        : lr_(lr), gamma_(gamma), step_size_(step_size)
    {
        if (step_size_ == 0)
            throw std::invalid_argument("StepLR: step_size must be positive");
    }

    float base_lr() const { return lr_; }
    float gamma() const { return gamma_; }
    std::size_t step_size() const { return step_size_; }

    float get_lr(std::size_t step) const
    {
        const std::size_t decays = step / step_size_;
        return lr_ * static_cast<float>(std::pow(static_cast<double>(gamma_),
                                                 static_cast<double>(decays)));
    }
};

#endif // MINIML_SCHEDULERS_HPP
