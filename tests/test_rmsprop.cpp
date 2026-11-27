// RMSprop check against a hand-computed toy-gradient table.
//
// Setup: lr = 0.1, decay = 0.5, eps = 0, p0 = 0, gradients [2, 2, -2].
// By hand (v_t = 0.5*v_{t-1} + 0.5*g^2, update = 0.1*g/sqrt(v_t)):
//   step 1: v = 0.5*0 + 0.5*4 = 2;   update = 0.2/sqrt(2) = 0.141421356237
//           p = -0.141421356237
//   step 2: v = 0.5*2 + 0.5*4 = 3;   update = 0.2/sqrt(3) = 0.115470053838
//           p = -0.256891410075
//   step 3: v = 0.5*3 + 0.5*4 = 3.5; update = -0.2/sqrt(3.5) = -0.106904496764
//           p = -0.149986913311
// None of these expected values come from running the optimiser.

#include "rmsprop.hpp"

#include <cmath>
#include <iostream>

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

    Tensor p({1});
    p.data->at(0) = 0.0f;
    RMSprop opt({&p}, 0.1f, 0.5f, 0.0f);

    const float gradients[3] = {2.0f, 2.0f, -2.0f};
    const float expected_v[3] = {2.0f, 3.0f, 3.5f};
    const float expected_p[3] = {-0.141421356237f, -0.256891410075f, -0.149986913311f};

    for (std::size_t step = 0; step < 3; ++step)
    {
        opt.zero_grad();
        p.grad->at(0) = gradients[step];
        opt.step();

        const float v = opt.squared_average(0).at(0);
        std::cout << "step " << step + 1 << ": v=" << v << " p=" << p.data->at(0) << "\n";
        ok = expect(close(v, expected_v[step]), "accumulator matches hand table") && ok;
        ok = expect(close(p.data->at(0), expected_p[step]), "parameter matches hand table") && ok;
    }

    // zero_grad() clears gradients only; it must not reset the accumulator.
    opt.zero_grad();
    ok = expect(p.grad->at(0) == 0.0f, "zero_grad clears the gradient") && ok;
    ok = expect(close(opt.squared_average(0).at(0), 3.5f), "zero_grad keeps the accumulator") && ok;

    // A zero gradient leaves the parameter unchanged while v decays by half.
    p.grad->at(0) = 0.0f;
    opt.step();
    ok = expect(close(p.data->at(0), expected_p[2]), "zero gradient leaves parameter unchanged") && ok;
    ok = expect(close(opt.squared_average(0).at(0), 1.75f), "zero gradient decays accumulator") && ok;

    if (ok)
        std::cout << "PASS rmsprop\n";
    else
        std::cerr << "FAIL rmsprop\n";
    return ok ? 0 : 1;
}
