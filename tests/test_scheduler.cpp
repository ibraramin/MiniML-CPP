// LR scheduler checks against analytically known rate and update tables.

#include "schedulers.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

bool expect(bool condition, const char *message)
{
    if (!condition)
        std::cerr << "FAIL " << message << "\n";
    return condition;
}

bool close(float a, float b) { return std::fabs(a - b) < 1e-6f; }

} // namespace

int main()
{
    bool ok = true;

    // Constant schedule: the rate is the same at every step.
    const ConstantLR constant(0.05f);
    for (std::size_t step = 0; step < 8; ++step)
        ok = expect(close(constant.get_lr(step), 0.05f), "constant LR = 0.05 at every step") && ok;

    // Step schedule lr = 0.1 * 0.5^floor(step/3), derived by hand:
    //   steps 0-2 -> 0.1, 3-5 -> 0.05, 6-8 -> 0.025, step 9 -> 0.0125.
    const StepLR step_lr(0.1f, 0.5f, 3);
    const float expected[10] = {0.1f, 0.1f, 0.1f, 0.05f, 0.05f,
                                0.05f, 0.025f, 0.025f, 0.025f, 0.0125f};
    for (std::size_t s = 0; s < 10; ++s)
        ok = expect(close(step_lr.get_lr(s), expected[s]), "step LR table") && ok;

    // Update table 1: constant lr 0.2, fixed gradient 0.5, three steps.
    // Each step subtracts 0.2*0.5 = 0.1, so 1.0 -> 0.9 -> 0.8 -> 0.7.
    {
        ConstantLR lr(0.2f);
        float param = 1.0f;
        const float after[3] = {0.9f, 0.8f, 0.7f};
        for (std::size_t s = 0; s < 3; ++s)
        {
            param -= lr.get_lr(s) * 0.5f;
            ok = expect(close(param, after[s]), "constant-LR update table") && ok;
        }
    }

    // Update table 2: step lr 0.4 halved every 2 steps, gradient 1.0.
    // 1.0 - 0.4 - 0.4 - 0.2 - 0.2 = -0.2 after four steps.
    {
        StepLR lr(0.4f, 0.5f, 2);
        float param = 1.0f;
        const float after[4] = {0.6f, 0.2f, 0.0f, -0.2f};
        for (std::size_t s = 0; s < 4; ++s)
        {
            param -= lr.get_lr(s) * 1.0f;
            ok = expect(close(param, after[s]), "step-LR update table") && ok;
        }
    }

    bool threw = false;
    try
    {
        StepLR bad(0.1f, 0.5f, 0);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    ok = expect(threw, "zero step_size throws") && ok;

    if (ok)
        std::cout << "PASS scheduler\n";
    else
        std::cerr << "FAIL scheduler\n";
    return ok ? 0 : 1;
}
