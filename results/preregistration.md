# ADR-003 O3 / Step 7 — GradCheck honesty gate preregistration

Status: **PRE-REGISTERED before any new measurement for this change.**

The orchestrator's task brief supplies the recorded worst relative error
`1.889645e-04` measured with the old `1e-2` threshold, and states that the
baseline was 2/2 passing. No new gradcheck run has been used to choose the
value below.

## Oracle-driven revision (explicitly recorded)

An initial draft of this record proposed `1e-3` (5.29x margin). Before any
measurement, an oracle devil's-advocate review argued that this buys almost no
extra bug detection while shrinking the noise margin from ~53x to ~2.6x
relative to a *tightened* alternative, and that the recorded worst error is a
single sample on a single compiler/build/CPU. The threshold below was revised
from `1e-3` to **`2e-3`** on that evidence, still before measuring. This
section is the explicit record of that change and its rationale.

## Decision (locked before measurement)

- Tightened threshold: **2e-3** (`2.0e-3f`), replacing the old `1e-2`.
- Safety margin: `2.0e-3 / 1.889645e-4 = 10.58x` (~10.6x) above the recorded
  worst error. Rationale:
  - With `kH = 1e-3f` (`include/gradcheck.hpp:25`), a central difference
    amplifies float32 rounding by roughly `1/(2h) = 500x`. GCC
    `-ffp-contract=fast` versus Clang, and Release `-O3` versus Debug/ASan
    (`CMakeLists.txt:8,12-15`), all change that rounding. The recorded
    `1.889645e-04` is **one sample on one compiler/build/CPU**, while CI spans
    GCC/Clang/ASan on heterogeneous runners.
  - The scale-normalised `relative_error` (denominator floored at `1e-8`,
    `include/gradcheck.hpp:32`) is heavy-tailed for small gradients, so a thin
    margin risks false failures unrelated to a real gradient bug.
  - ~10.6x keeps a robust noise margin while still rejecting a 1% injected
    relative error (`~1e-2`, about **5x** the tightened threshold). Tightening
    further to `5e-4` (~2.6x) was explicitly rejected: it buys essentially zero
    additional bug detection while making the gate compiler-sensitive.
- This value is fixed for this change. If a later measurement exceeds it, the
  response is to investigate the gradient, **not** to raise the threshold
  silently. Any threshold change requires a new preregistration entry.

## Falsification design (declared before measurement)

Claim under test: *"the gradcheck gate rejects an incorrect gradient."* This
must be a genuine negative control, not a self-consistency check.

### Controlled multiplicative defect (oracle-required design)

In a **scratch copy under `/var/tmp` (never committed)**, inject a controlled
multiplicative defect into `analytical_gradient` (`include/gradcheck.hpp`) for
one index, e.g. `return grad * (1 + delta)` at `index == 1` of the
elementwise-add case, whose analytical gradient is exactly `1`. Then
`relative_error = |1 - (1+delta)| / (1 + (1+delta)) = delta/(2+delta) ~ delta/2`.

Required outcomes for the `2e-3` gate:

| delta | threshold | expected relative error | required exit | required output |
|---|---|---|---|---|
| `8e-3` | `2e-3` (gate) | `~4e-3 > 2e-3` | **1** | `FAIL gradcheck` |
| `2e-3` | `2e-3` (gate) | `~1e-3 < 2e-3` | **0** | honest pass |
| `8e-3` | large/infinite | `~4e-3` | **0** (same bug passes — negative control) | honest pass |

A committed ctest negative control `gradcheck_falsification` (`WILL_FAIL TRUE`)
runs the same `check_param` decision path with a `delta = 8e-3` tampered
analytical gradient and must exit 1, printing `FAIL gradcheck`.

### Cross-build safety check (oracle-required)

Run the **unmodified** binary in Release and in ASan/UBSan builds; record
`WORST_OBSERVED_RELATIVE_ERROR` from each. Any value above the pre-registered
`2e-3` falsifies the safety assumption and must be reported, not spun.

## Evidence to be recorded after implementation

- Verbatim output of the full baseline-equivalent configure/build/ctest.
- Verbatim output of each `-R` ctest selector (`minibatch`, `metrics`,
  `scheduler|rmsprop`).
- Verbatim output of `test -s results/preregistration.md` and the
  `grep -c '1e-2' tests/test_gradcheck.cpp` check.
- Verbatim output of the falsification commands (all three delta/threshold
  conditions) with exit codes, and of the Release/ASan worst-error readings.

---

## Appendix — post-measurement evidence log (added AFTER measurement)

This appendix was written after the experiments ran. It does **not** alter the
pre-registered threshold or design above.

### Safety check (unmodified binary)

| Build | Command | `WORST_OBSERVED_RELATIVE_ERROR` | Exit |
|---|---|---|---|
| Release | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ./build/tests/test_gradcheck` | `1.889645e-04` | 0 |
| Debug + ASan/UBSan | `cmake -S . -B /var/tmp/opencode/miniml-asan-o3 -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON && cmake --build ... && .../tests/test_gradcheck` | `1.889645e-04` | 0 |

Both readings equal the supplied baseline value and are below `2e-3`, so the
safety assumption was **not** falsified.

### Scratch-copy falsification (in `/var/tmp/opencode/miniml-falsify-o3`, never committed)

Defect injected into `analytical_gradient` at `index == 1` via
`-DMINIML_FALSIFY_DELTA=<delta>`. Observed:

| delta | threshold | worst rel error | exit | output | matches design |
|---|---|---|---|---|---|
| `8e-3` | `2e-3` | `4.077007e-03` | **1** | `FAIL gradcheck` | yes |
| `2e-3` | `2e-3` | `1.091913e-03` | **0** | `PASS gradcheck` | yes |
| `8e-3` | `1e9` (scratch threshold override) | `4.077007e-03` | **0** | `PASS gradcheck` | yes — same injected bug passes |

Committed negative control: `ctest --test-dir build -R gradcheck_falsification`
uses `WILL_FAIL TRUE` and passes because `test_gradcheck --falsify` exits 1
(`FALSIFICATION_DETECTED=1`). The two committed controls (`--falsify-below`,
`--falsify-unbounded`) both exit 0 with `FALSIFICATION_ACCEPTED=1`.
