// Deterministic mini-batch loader checks. Row 0 of X encodes the original
// sample index, so batch contents can be verified without relying on the
// loader's internal permutation.

#include "data.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

bool expect(bool condition, const char *message)
{
    if (!condition)
        std::cerr << "FAIL " << message << "\n";
    return condition;
}

std::vector<std::size_t> tags_of(const Dataset &b)
{
    std::vector<std::size_t> tags;
    for (std::size_t i = 0; i < b.X.shape[0]; ++i)
        tags.push_back(static_cast<std::size_t>(b.X.data->at(i * b.X.shape[1])));
    return tags;
}

bool same_sequences(const std::vector<std::vector<std::size_t>> &a,
                    const std::vector<std::vector<std::size_t>> &b)
{
    return a == b;
}

std::vector<std::vector<std::size_t>> all_batches(const DataLoader &loader)
{
    std::vector<std::vector<std::size_t>> out;
    for (const Dataset &batch : loader)
        out.push_back(tags_of(batch));
    return out;
}

} // namespace

int main()
{
    bool ok = true;

    constexpr std::size_t n = 10;
    Tensor X({n, 2}), y({n, 1});
    for (std::size_t i = 0; i < n; ++i)
    {
        X.data->at(i * 2 + 0) = static_cast<float>(i);     // index tag
        X.data->at(i * 2 + 1) = static_cast<float>(100 + i);
        y.data->at(i) = static_cast<float>(i * i);          // label tag
    }

    DataLoader loader(X, y, 3, 42);
    ok = expect(loader.size() == n, "loader.size() == n") && ok;
    ok = expect(loader.batch_size() == 3, "loader.batch_size() == 3") && ok;
    ok = expect(loader.seed() == 42, "loader.seed() == 42") && ok;
    ok = expect(loader.batches_per_epoch() == 4, "batches_per_epoch() == 4 (ceil)") && ok;

    const auto batches = loader.epoch_batches();
    ok = expect(batches.size() == 4, "epoch_batches() returns 4 batches") && ok;

    const std::size_t expected_sizes[4] = {3, 3, 3, 1};
    std::vector<std::size_t> seen;
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        ok = expect(batches[b].X.shape[0] == expected_sizes[b], "batch row count") && ok;
        ok = expect(batches[b].X.shape[1] == 2, "batch feature count") && ok;
        ok = expect(batches[b].y.shape[0] == expected_sizes[b], "batch label row count") && ok;
        ok = expect(batches[b].y.shape[1] == 1, "batch label column count") && ok;

        const std::vector<std::size_t> tags = tags_of(batches[b]);
        for (std::size_t i = 0; i < tags.size(); ++i)
        {
            ok = expect(tags[i] < n, "tag in range") && ok;
            const float label = batches[b].y.data->at(i);
            ok = expect(label == static_cast<float>(tags[i] * tags[i]),
                        "label row follows its X row") && ok;
        }
        seen.insert(seen.end(), tags.begin(), tags.end());
    }
    std::sort(seen.begin(), seen.end());
    std::vector<std::size_t> identity(n);
    std::iota(identity.begin(), identity.end(), std::size_t{0});
    ok = expect(seen == identity, "one epoch visits every sample exactly once") && ok;

    // Iterator and materialised epoch must agree exactly.
    ok = expect(same_sequences(all_batches(loader), [&] {
        std::vector<std::vector<std::size_t>> out;
        for (const Dataset &b : batches)
            out.push_back(tags_of(b));
        return out;
    }()), "iterator agrees with epoch_batches()") && ok;

    // Determinism: same seed -> same order, including after an epoch advance.
    DataLoader same(X, y, 3, 42);
    ok = expect(all_batches(loader) == all_batches(same), "same seed gives same epoch 0 order") && ok;
    const auto epoch0 = all_batches(loader);
    loader.next_epoch();
    same.next_epoch();
    ok = expect(loader.epoch() == 1, "next_epoch() advances epoch counter") && ok;
    ok = expect(all_batches(loader) == all_batches(same), "same seed gives same epoch 1 order") && ok;
    ok = expect(all_batches(loader) != epoch0, "next_epoch() reshuffles") && ok;

    // A different random seed must not collide with seed 42 on this fixture.
    DataLoader other(X, y, 3, 7);
    ok = expect(all_batches(other) != epoch0, "different seed gives a different order") && ok;

    // Exact division: no remainder batch.
    Tensor X9({9, 2}), y9({9, 1});
    for (std::size_t i = 0; i < 9; ++i)
    {
        X9.data->at(i * 2) = static_cast<float>(i);
        y9.data->at(i) = static_cast<float>(i);
    }
    DataLoader exact(X9, y9, 3, 42);
    ok = expect(exact.batches_per_epoch() == 3, "exact division -> 3 batches") && ok;
    for (std::size_t b = 0; b < exact.batches_per_epoch(); ++b)
        ok = expect(exact.batch(b).X.shape[0] == 3, "exact division batch size 3") && ok;

    // Invalid configurations must be rejected, not silently accepted.
    bool threw = false;
    try
    {
        DataLoader bad(X, y, 0, 42);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    ok = expect(threw, "zero batch_size throws") && ok;

    threw = false;
    try
    {
        loader.batch(loader.batches_per_epoch());
    }
    catch (const std::out_of_range &)
    {
        threw = true;
    }
    ok = expect(threw, "out-of-range batch index throws") && ok;

    if (ok)
        std::cout << "PASS minibatch\n";
    else
        std::cerr << "FAIL minibatch\n";
    return ok ? 0 : 1;
}
