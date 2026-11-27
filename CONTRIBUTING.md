# Contributing

MiniML-CPP is a small, readable C++17 codebase. Contributions are welcome, but
they must keep the project buildable, deterministic, and honestly described.
Documentation lives in the root [README.md](README.md),
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and
[docs/USAGE.md](docs/USAGE.md); read those before changing behavior.

## Branch workflow

1. Work on a topic branch and keep it focused on one change.
2. Run the full Definition of Done (below) before opening a pull request.
3. CI runs on `push`, `pull_request`, and `workflow_dispatch`; keep those jobs
   green. The workflow uses `fail-fast: false`, so both compiler jobs report.
4. Update the relevant documentation in the same change. Do not edit
   `AGENTS.md`; it is curated by humans.

## Definition of Done

Run these commands from the repository root. Each must exit 0:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/examples/xor_demo
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

`ENABLE_ASAN` defaults to `OFF`; the ASan configuration adds
`-fsanitize=address,undefined -fno-omit-frame-pointer` to both compilation and
linking. Both configurations must pass CTest; the Release run also exercises
the deterministic XOR demo.

## What CI runs

`.github/workflows/ci.yml` defines two jobs on `ubuntu-latest`:

- **build-and-test** — a matrix over `gcc` and `clang` (`CC`/`CXX` set to the
  matching compiler). Each job configures a Release build, builds with
  `cmake --build build -j`, runs `ctest --test-dir build --output-on-failure`,
  and runs `./build/examples/xor_demo`.
- **asan** — configures `build-asan` in Debug with `-DENABLE_ASAN=ON`, builds,
  and runs `ctest --test-dir build-asan --output-on-failure`.

A local run of the DoD block above covers the same checks.

## Tests must be deterministic and registered

New behavior should come with a deterministic test. The existing tests seed
every tensor by hand (`data->at(i)`) instead of using `Tensor::randn()`, which
is backed by `random_device`. A gradient test should build a fresh graph per
call, because autodiff deduplicates nodes by raw data pointer and the tape
stores tensor copies; call `zero_grad()` before `backward()` so leaf gradients
do not accumulate.

Register each new test binary with CTest in `CMakeLists.txt` using the same
pattern as the existing tests:

```cmake
add_executable(test_name tests/test_name.cpp)
target_link_libraries(test_name PRIVATE miniml_core)
set_target_properties(test_name PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_test(NAME test_name COMMAND test_name
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
```

The working directory matters: `test_serialization` writes scratch `.bin`
files in the current directory and removes them on normal exit (an early
return can skip cleanup). Tests should print a clear `PASS ...` / `FAIL ...`
line and return `0` on success, `1` on failure. A test that only passes on one
machine is not a gate. The suite currently registers seven CTest cases:
`gradcheck`, `gradcheck_falsification` (`WILL_FAIL TRUE`, so a passing negative
control is a passing CTest), `serialization`, `minibatch_loader`,
`metrics_binary`, `scheduler_lr` and `rmsprop_toy`. The last two are gated by
`ENABLE_O3_EXTRAS`, which is ON by default so a downstream build can omit them.
The tightened gradcheck threshold must not be changed without a new entry in
`results/preregistration.md`.

## Keep dependencies at zero

The build needs only a C++17 compiler and CMake; the library itself uses the
C++ standard library. Do not add third-party libraries, vendored headers, or
network-fetched dependencies to the build or tests. `gnuplot` remains optional
and is used only by `src/plot.hpp` in the original showcase, which is excluded
from tests and from the XOR demo.

## Keep the documented limitations accurate

The README's Limitations section is part of the contract. It currently states
that autodiff is pointer-identity based, that `backward()` clears only the root
gradient, that there is no scalar constructor or scalar `operator*`, that the
loss is a sum rather than a mean, that broadcasting is limited to one row case,
that the code is CPU-only with no device model, that `randn()` is not
reproducible, that serialization stores raw float weights only, that plotting
uses `popen`, that there are no thread-safety guarantees, and that no
install/export target is provided. The same section now also notes that
deterministic mini-batching with seeded shuffling (`src/data.hpp`) and opt-in
schedulers and RMSprop (`include/`) exist, while weight decay, gradient clipping
and checkpoint/resume metadata remain unsupported.

If a change alters any of these, update the bullet in the same pull request.
Do not claim support for weight decay, gradient clipping, GPU/CUDA devices,
checkpoint/resume metadata, or mean-reduced losses until the code actually
provides them. The mini-batch loader, schedulers and RMSprop are real but their
headers describe exactly what they do — schedulers only compute a rate and
RMSprop keeps its own accumulator state. Likewise, keep measured results in the
README tied to recorded outputs rather than estimates; the gradcheck gate of
`2.0e-3f` is pre-registered in `results/preregistration.md` along with its
falsification controls.
