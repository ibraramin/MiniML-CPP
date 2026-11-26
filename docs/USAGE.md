# Usage cookbook

Every snippet below is adapted from code that is actually in this repository:
`tests/test_gradcheck.cpp`, `tests/test_serialization.cpp`,
`examples/xor_demo.cpp`, and the usage example in the root `README.md`. For
internals, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

A C++17 compiler and CMake are enough for the build, tests, and demos described
here; the core uses only the C++ standard library. `gnuplot` is needed only by
the optional plotting header in the original showcase — see the caveat at the
end of this document.

## Build and verify (exact Definition of Done)

Run from the repository root. Every command must exit 0:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/examples/xor_demo
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

`ENABLE_ASAN` is `OFF` by default; enabling it adds
`-fsanitize=address,undefined -fno-omit-frame-pointer` to compilation and
linking. `CMakeLists.txt` registers two CTest tests named `gradcheck` and
`serialization`; their binaries are written to `build/tests/`.

## Creating and seeding tensors

There is no scalar constructor and no `Tensor * float`. Allocate with a shape
and write floats into `data->at(i)`. `Tensor::randn()` is non-deterministic
(`random_device`), so the tests seed by hand.

A deterministic seed helper, as in `tests/test_gradcheck.cpp`:

```cpp
#include "tensor.hpp"

static void seed(Tensor &t, float base)
{
    for (std::size_t i = 0; i < t.total_size; ++i)
        t.data->at(i) = base + 0.13f * static_cast<float>(i);
}

Tensor a({4}), b({4});
seed(a, 0.37f);
seed(b, -0.21f);
```

For the XOR dataset, `examples/xor_demo.cpp` fills tensors element by element:

```cpp
Tensor X({4, 2}), y({4, 1});
const float xs[8] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
const float ys[4] = {0.0f, 1.0f, 1.0f, 0.0f};
for (int i = 0; i < 8; ++i) X.data->at(i) = xs[i];
for (int i = 0; i < 4; ++i) y.data->at(i) = ys[i];
```

`Linear` initializes its weights with `Tensor::randn`, so the XOR demo
overwrites all four parameter tensors with a fixed PRNG seed:

```cpp
#include <random>

std::mt19937 gen(1); // fixed seed => deterministic weights
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
for (std::size_t i = 0; i < t.total_size; ++i)
    t.data->at(i) = dist(gen);
```

Here `t` is each of `l1->weight`, `l1->bias`, `l2->weight`, and `l2->bias` in
the demo's seeding loop.

## Forward / backward / step

Build a fresh graph for every step, call `zero_grad()` before `backward()`,
and evaluate the loss again with a fresh forward pass for the final number.
This is the shape of `train()` in `examples/xor_demo.cpp`:

```cpp
#include "tensor.hpp"
#include "nn.hpp"
#include "adam.hpp"
#include "metrics.hpp"

Sequential net({std::make_shared<Linear>(2, 2), std::make_shared<Sigmoid>(),
                std::make_shared<Linear>(2, 1), std::make_shared<Sigmoid>()});
MSELoss mse;

template <typename Optimizer>
float train(Sequential &net, const Tensor &X, const Tensor &y, float lr)
{
    Optimizer optim(net.parameters(), lr);
    for (int epoch = 0; epoch < 3000; ++epoch)
    {
        optim.zero_grad();
        Tensor pred = net.forward(X);
        Tensor loss = mse(pred, y);
        loss.backward();
        optim.step();
    }
    return mse(net.forward(X), y).data->at(0); // fresh evaluation
}
```

Call `train<SGD>(net, X, y, 0.5f)` or `train<Adam>(net, X, y, 0.05f)` — the
demo uses those learning rates. Both take the `std::vector<Tensor *>` returned
by `net.parameters()` and expose `zero_grad()` / `step()`; `Adam` defaults are
`lr=0.001f`, `beta1=0.9f`, `beta2=0.999f`, `eps=1e-8f`, with bias correction.

`MSELoss` sums squared errors (not a mean), so batch size changes the effective
learning rate. Keep parameter tensors alive while an optimizer holds their raw
pointers, and never reuse tensor nodes across backward calls.

## Gradient checking

`include/gradcheck.hpp` needs `loss_fn` to return a **fresh scalar tensor**
each call. This is the core of `check_param()` in
`tests/test_gradcheck.cpp`:

```cpp
#include "gradcheck.hpp"
namespace gc = miniml_gradcheck;

Tensor param({4});
seed(param, 0.45f); // deterministic, no randn
auto loss_fn = [&]() -> Tensor { return param.sum(); };

for (std::size_t i = 0; i < param.total_size; ++i)
{
    float num = gc::numerical_gradient(param, i, gc::kH, loss_fn);
    float ana = gc::analytical_gradient(param, i, loss_fn);
    float err = gc::relative_error(num, ana); // threshold in test: 1e-2f
}
```

`kH` is `1e-3f`. The numerical helper perturbs one float, evaluates `loss_fn()`
at `+h` and `-h`, and restores it on every exit path; the analytical helper
calls `zero_grad()`, `backward()`, and reads `grad`. The test covers add,
multiply, sigmoid, and a `Linear` layer's weight and bias through `MSELoss`.
Run it with `ctest --test-dir build --output-on-failure` (the DoD command) or
`./build/tests/test_gradcheck`. The recorded worst relative error is
`1.889645e-04` against a `1e-2` gate.

## Saving and loading weights

`include/serialization.hpp` saves raw float tensors only. In
`tests/test_serialization.cpp`, two models share a topology; model A is seeded
and model B is loaded into:

```cpp
#include "serialization.hpp"
namespace ser = miniml_serialization;

auto l1a = std::make_shared<Linear>(2, 3);
auto l2a = std::make_shared<Linear>(3, 1);
Sequential a({l1a, std::make_shared<Sigmoid>(), l2a, std::make_shared<Sigmoid>()});
seed(l1a->weight, 0.31f); // seed all four parameter tensors
// ... seed(l1a->bias, ...), seed(l2a->weight, ...), seed(l2a->bias, ...)

auto l1b = std::make_shared<Linear>(2, 3);
auto l2b = std::make_shared<Linear>(3, 1);
Sequential b({l1b, std::make_shared<Sigmoid>(), l2b, std::make_shared<Sigmoid>()});

std::vector<Tensor *> weights_a = a.parameters();
if (!ser::save("weights.bin", weights_a)) { /* handle false return */ }
std::vector<Tensor *> weights_b = b.parameters();
if (!ser::load("weights.bin", weights_b)) { /* handle false return */ }
```

`load()` requires matching tensor count and shapes, validates the whole file
before writing, and returns `false` (not an exception) on bad magic,
unsupported version, truncation, trailing bytes, or shape mismatch. The test
also re-saves and compares bytes. The format is `MMLW` magic, version `1`,
little-endian integer and float fields; optimizer state is not stored.

## Running the XOR demo

```bash
./build/examples/xor_demo
```

The program trains a 2-2-1 network (`Linear(2,2), Sigmoid, Linear(2,1),
Sigmoid`) on all four XOR rows with fixed weights (seed `1`), for `3000` epochs
with SGD `lr=0.5` then Adam `lr=0.05`. Its stdout has:

- a `# mode=... lr=... seed=1 epochs=3000` line per mode, then an `epoch,loss`
  header and one loss row per epoch per mode;
- a `# results` block with both final losses, thresholds, accuracies, and the four predictions of each model;
- `PASS xor_demo` on success, `FAIL xor_demo` otherwise, with exit code 0 or 1.

The recorded run (`/var/tmp/opencode/dod/final-m4.out`) ends with:

```text
sgd_final=0.002114 threshold=0.010000
adam_final=0.000128 threshold=0.010000
sgd_accuracy=1.000000 adam_accuracy=1.000000
PASS xor_demo
```

Those are measured values for this deterministic configuration, not a promise
for other compilers; the gate requires both final losses below `0.01` and
accuracy at least `4/4`. The README GIF records the same setup (900×520,
roughly 18 seconds).

## The gnuplot showcase caveat

`src/readme.md` describes an older six-demo showcase (linear, ridge,
polynomial, logistic, SVM, MLP) whose plots require `gnuplot`; its recorded
build line is `g++ -std=c++17 main.cpp tensor.cpp -o ml_demo`. It is separate
from the DoD commands: `gnuplot` is not needed to build, test, or run anything
here. `src/plot.hpp` pipes to `gnuplot` via `popen` and is excluded from all
tests and the XOR demo. README gallery images illustrate that older showcase, not verifiability-test outputs; see README Limitations.
