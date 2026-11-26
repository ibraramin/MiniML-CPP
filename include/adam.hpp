#ifndef MINIML_ADAM_HPP
#define MINIML_ADAM_HPP

// Adam optimiser, following the same convention as src/nn.hpp's SGD:
// construct from a vector of Tensor* parameters, then call zero_grad() and
// step() in that order. src/nn.hpp is deliberately left untouched.

#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

class Adam
{
    std::vector<Tensor *> params;
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t = 0;
    std::vector<std::vector<float>> m; // first moments, one vector per parameter
    std::vector<std::vector<float>> v; // second moments, one vector per parameter

public:
    Adam(std::vector<Tensor *> params,
         float lr = 0.001f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps = 1e-8f)
        : params(std::move(params)), lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0)
    {
        m.reserve(this->params.size());
        v.reserve(this->params.size());
        for (auto *p : this->params)
        {
            m.emplace_back(p->total_size, 0.0f);
            v.emplace_back(p->total_size, 0.0f);
        }
    }

    void zero_grad()
    {
        for (auto *p : params)
            std::fill(p->grad->begin(), p->grad->end(), 0.0f);
    }

    void step()
    {
        ++t;
        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));

        for (std::size_t k = 0; k < params.size(); ++k)
        {
            Tensor *p = params[k];
            float *d = p->data->data();
            const float *g = p->grad->data();
            for (std::size_t i = 0; i < p->total_size; ++i)
            {
                m[k][i] = beta1 * m[k][i] + (1.0f - beta1) * g[i];
                v[k][i] = beta2 * v[k][i] + (1.0f - beta2) * g[i] * g[i];

                const float m_hat = m[k][i] / bc1;
                const float v_hat = v[k][i] / bc2;
                d[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
};

#endif // MINIML_ADAM_HPP
