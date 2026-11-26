// Central-difference verification of MiniML-CPP's reverse-mode autodiff.
// Every tensor is seeded by hand: Tensor::randn is never used, so results are
// bit-for-bit reproducible across runs and machines.

#include "tensor.hpp"
#include "nn.hpp"
#include "gradcheck.hpp"

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

// Measured worst-case relative error over all cases was 1.89e-4
// (see /var/tmp/opencode/miniml-scratch.md). Threshold kept ~53x above it to
// absorb compiler/CPU differences while still catching a broken gradient.
static constexpr float kThreshold = 1e-2f;

namespace gc = miniml_gradcheck;

static float g_worst = 0.0f;

static void seed(Tensor &t, float base)
{
    for (std::size_t i = 0; i < t.total_size; ++i)
        t.data->at(i) = base + 0.13f * static_cast<float>(i);
}

static bool check_param(const std::string &label,
                        Tensor &param,
                        const std::function<Tensor()> &loss_fn)
{
    bool ok = true;
    float worst = 0.0f;
    for (std::size_t i = 0; i < param.total_size; ++i)
    {
        const float num = gc::numerical_gradient(param, i, gc::kH, loss_fn);
        const float ana = gc::analytical_gradient(param, i, loss_fn);
        const float err = gc::relative_error(num, ana);
        if (err > worst)
            worst = err;
        if (err > kThreshold || !std::isfinite(num) || !std::isfinite(ana))
        {
            std::cerr << "FAIL " << label << " index " << i
                      << ": numerical=" << num << " analytical=" << ana
                      << " rel_error=" << err << "\n";
            ok = false;
        }
    }
    if (worst > g_worst)
        g_worst = worst;
    std::cout << "  " << std::left << std::setw(24) << label
              << "worst relative error = " << std::scientific << worst << "\n"
              << std::fixed;
    return ok;
}

int main()
{
    bool ok = true;
    std::cout << "gradcheck h=" << gc::kH << " threshold=" << kThreshold << "\n";

    // 1. element-wise add reduced to a scalar: d(sum(a+b))/da == 1
    {
        Tensor a({4}), b({4});
        seed(a, 0.37f);
        seed(b, -0.21f);
        auto loss = [&]() -> Tensor { return (a + b).sum(); };
        ok = check_param("elementwise add", a, loss) && ok;
    }

    // 2. element-wise multiply: d(sum(c*d))/dc == d
    {
        Tensor c({4}), d({4});
        seed(c, 0.45f);
        seed(d, 0.80f);
        auto loss = [&]() -> Tensor { return (c * d).sum(); };
        ok = check_param("elementwise multiply", c, loss) && ok;
    }

    // 3. sigmoid activation: d(sum(sigmoid(z)))/dz == s(1-s)
    {
        Tensor z({5});
        seed(z, -0.10f);
        auto loss = [&]() -> Tensor { return z.sigmoid().sum(); };
        ok = check_param("sigmoid activation", z, loss) && ok;
    }

    // 4. a Linear layer trained through MSELoss (both weight and bias)
    {
        Linear layer(3, 2);
        seed(layer.weight, 0.25f);
        seed(layer.bias, -0.10f);

        Tensor X({4, 3}), y({4, 2});
        seed(X, -0.30f);
        seed(y, 0.15f);

        MSELoss mse;
        auto loss = [&]() -> Tensor { return mse(layer.forward(X), y); };
        ok = check_param("linear weight", layer.weight, loss) && ok;
        ok = check_param("linear bias", layer.bias, loss) && ok;
    }

    std::cout << "WORST_OBSERVED_RELATIVE_ERROR=" << std::scientific << g_worst << "\n"
              << std::fixed;
    if (ok)
        std::cout << "PASS gradcheck\n";
    else
        std::cerr << "FAIL gradcheck\n";
    return ok ? 0 : 1;
}
