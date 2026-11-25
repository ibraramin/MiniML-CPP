#ifndef MINIML_GRADCHECK_HPP
#define MINIML_GRADCHECK_HPP

// Deterministic central-difference gradient checking for MiniML-CPP tensors.
//
// The library's Tensor holds a shared_ptr<vector<float>> called `data`, so a
// caller can seed a parameter deterministically and temporarily perturb a
// single float in it.  All helpers here restore the original value before
// returning; they never call Tensor::randn and never mutate the graph shape.
//
// `loss_fn` must be a callable returning a FRESHLY COMPUTED SCALAR Tensor
// (shape {1}) whose value depends on the parameter being checked.  A fresh
// graph per call matters because MiniML's autodiff is pointer-identity based.

#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>

namespace miniml_gradcheck {

// Step used for every central difference. Fixed so results are reproducible.
static constexpr float kH = 1e-3f;

// Scale-normalised relative error, following the standard gradcheck formula
// (numerator divided by the sum of magnitudes, floored to avoid zero divide).
// Returns 0 when both arguments are zero; never negative or infinite.
inline float relative_error(float a, float b)
{
    const float denom = std::max(1e-8f, std::fabs(a) + std::fabs(b));
    return std::fabs(a - b) / denom;
}

namespace detail {

// Restores the perturbed float on every exit path, including exceptions thrown
// by a loss_fn with a shape mismatch.
struct RestoreValue
{
    Tensor &tensor;
    std::size_t index;
    float original;

    ~RestoreValue() { tensor.data->at(index) = original; }
};

} // namespace detail

// Numerical derivative of sum(loss_fn) w.r.t. param.data->at(index), computed
// as (f(x+h) - f(x-h)) / (2h). The parameter is restored to its exact original
// float value before the function returns. Accumulation uses double to limit
// cancellation, although the underlying loss values are float32.
inline float numerical_gradient(Tensor &param,
                                std::size_t index,
                                float h,
                                const std::function<Tensor()> &loss_fn)
{
    const float original = param.data->at(index);
    detail::RestoreValue restore{param, index, original}; // restores on any exit path

    param.data->at(index) = original + h;
    const double loss_plus = static_cast<double>(loss_fn().data->at(0));

    param.data->at(index) = original - h;
    const double loss_minus = static_cast<double>(loss_fn().data->at(0));

    return static_cast<float>((loss_plus - loss_minus) / (2.0 * static_cast<double>(h)));
}

// Analytical derivative from reverse-mode autodiff. Gradients are cleared on
// the parameter first because Tensor::backward() only resets the root tensor's
// grad; descendant (parameter) grads would otherwise accumulate across checks.
inline float analytical_gradient(Tensor &param,
                                 std::size_t index,
                                 const std::function<Tensor()> &loss_fn)
{
    param.zero_grad();
    Tensor loss = loss_fn(); // scalar graph
    loss.backward();
    return param.grad->at(index);
}

} // namespace miniml_gradcheck

#endif // MINIML_GRADCHECK_HPP
