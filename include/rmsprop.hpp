#ifndef MINIML_RMSPROP_HPP
#define MINIML_RMSPROP_HPP

// Opt-in RMSprop optimiser, following the same construct / zero_grad() / step()
// convention as src/nn.hpp's SGD and include/adam.hpp. Including this header
// is the feature gate; src/nn.hpp is deliberately unchanged.
//
//   v_t = decay * v_{t-1} + (1 - decay) * g_t^2
//   p_t = p_{t-1} - lr * g_t / (sqrt(v_t) + eps)

#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

class RMSprop
{
    std::vector<Tensor *> params;
    float lr;
    float decay;
    float eps;
    std::vector<std::vector<float>> v; // squared-gradient accumulator per parameter

public:
    RMSprop(std::vector<Tensor *> params,
            float lr = 0.01f,
            float decay = 0.9f,
            float eps = 1e-8f)
        : params(std::move(params)), lr(lr), decay(decay), eps(eps)
    {
        if (decay < 0.0f || decay > 1.0f)
            throw std::invalid_argument("RMSprop: decay must be in [0, 1]");
        if (eps < 0.0f)
            throw std::invalid_argument("RMSprop: eps must be non-negative");

        v.reserve(this->params.size());
        for (auto *p : this->params)
            v.emplace_back(p->total_size, 0.0f);
    }

    void zero_grad()
    {
        for (auto *p : params)
            std::fill(p->grad->begin(), p->grad->end(), 0.0f);
    }

    void step()
    {
        for (std::size_t k = 0; k < params.size(); ++k)
        {
            Tensor *p = params[k];
            float *d = p->data->data();
            const float *g = p->grad->data();
            for (std::size_t i = 0; i < p->total_size; ++i)
            {
                v[k][i] = decay * v[k][i] + (1.0f - decay) * g[i] * g[i];
                d[i] -= lr * g[i] / (std::sqrt(v[k][i]) + eps);
            }
        }
    }

    // Read-only view of an accumulator, exposed for hand-table verification.
    const std::vector<float> &squared_average(std::size_t k) const { return v.at(k); }
};

#endif // MINIML_RMSPROP_HPP
