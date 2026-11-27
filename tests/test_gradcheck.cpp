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

// Pre-registered in results/preregistration.md (oracle-reviewed): recorded
// worst relative error 1.889645e-04, tightened threshold 2.0e-3 (~10.6x
// margin). Do not change this value without a new preregistration entry.
static constexpr float kThreshold = 2.0e-3f;

// Falsification controls. A controlled multiplicative defect is injected into
// the analytical gradient; for the elementwise-add case the true gradient is
// exactly 1, so relative_error = delta/(2+delta) ~ delta/2.
static constexpr float kTamperDeltaFailing = 8e-3f;  // rel ~4e-3 > gate
static constexpr float kTamperDeltaPassing = 2e-3f;  // rel ~1e-3 < gate
static constexpr float kUnboundedThreshold = 1e9f;   // negative control

namespace gc = miniml_gradcheck;

static float g_worst = 0.0f;

static void seed(Tensor &t, float base)
{
    for (std::size_t i = 0; i < t.total_size; ++i)
        t.data->at(i) = base + 0.13f * static_cast<float>(i);
}

static bool check_param(const std::string &label,
                        Tensor &param,
                        const std::function<Tensor()> &loss_fn,
                        float threshold = kThreshold,
                        float gradient_delta = 0.0f)
{
    bool ok = true;
    float worst = 0.0f;
    for (std::size_t i = 0; i < param.total_size; ++i)
    {
        const float num = gc::numerical_gradient(param, i, gc::kH, loss_fn);
        // gradient_delta is zero in the honest gate; the falsification mode
        // supplies a controlled multiplicative defect to prove the gate can
        // fail. Both paths share this exact decision routine.
        const float ana = gc::analytical_gradient(param, i, loss_fn) * (1.0f + gradient_delta);
        const float err = gc::relative_error(num, ana);
        if (err > worst)
            worst = err;
        if (err > threshold || !std::isfinite(num) || !std::isfinite(ana))
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

// Negative control: the same deterministic add case and the same check_param
// decision path, but with a controlled defect injected into the analytical
// gradient. `--falsify` must be rejected (exit 1); the two controls must
// accept the same defect when it is below the gate or the gate is unbounded.
static int run_falsification(bool expect_rejection, float delta, float threshold)
{
    Tensor a({4}), b({4});
    seed(a, 0.37f);
    seed(b, -0.21f);
    auto loss = [&]() -> Tensor { return (a + b).sum(); };

    std::cout << "FALSIFICATION delta=" << std::scientific << delta
              << " threshold=" << threshold << "\n"
              << std::fixed;
    const bool gate_passed = check_param("falsification: injected defect", a, loss,
                                         threshold, delta);
    if (!gate_passed)
        std::cerr << "FAIL gradcheck\n";
    else
        std::cout << "PASS gradcheck\n";

    if (expect_rejection)
    {
        std::cout << "FALSIFICATION_DETECTED=" << (gate_passed ? 0 : 1) << "\n";
        return gate_passed ? 0 : 1; // ctest WILL_FAIL expects exit 1 here
    }
    std::cout << "FALSIFICATION_ACCEPTED=" << (gate_passed ? 1 : 0) << "\n";
    return gate_passed ? 0 : 1;
}

int main(int argc, char **argv)
{
    const std::string mode = (argc > 1) ? argv[1] : "";
    if (mode == "--falsify" || mode == "--falsify-below" || mode == "--falsify-unbounded")
    {
        const bool expect_rejection = (mode == "--falsify");
        const float delta = (mode == "--falsify-below") ? kTamperDeltaPassing
                                                        : kTamperDeltaFailing;
        const float threshold = (mode == "--falsify-unbounded") ? kUnboundedThreshold
                                                                : kThreshold;
        return run_falsification(expect_rejection, delta, threshold);
    }

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
