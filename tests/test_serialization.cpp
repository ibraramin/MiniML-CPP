// Round-trip and corruption tests for the versioned binary weight format.

#include "tensor.hpp"
#include "nn.hpp"
#include "serialization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace ser = miniml_serialization;

// Measured max |prediction difference| after save/load was 0.0f (bit-exact)
// because float32 values are written and read verbatim. The tolerance leaves
// a small margin for that comparison.
static constexpr float kPredTolerance = 1e-6f;

static constexpr const char *kGood = "serialization_good.bin";
static constexpr const char *kTruncated = "serialization_truncated.bin";
static constexpr const char *kBadMagic = "serialization_badmagic.bin";
static constexpr const char *kBadVersion = "serialization_badversion.bin";

static void seed(Tensor &t, float base)
{
    for (std::size_t i = 0; i < t.total_size; ++i)
        t.data->at(i) = base + 0.07f * static_cast<float>(i);
}

static std::vector<uint8_t> read_bytes(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
}

static void write_bytes(const std::string &path, const std::vector<uint8_t> &b)
{
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(b.data()),
              static_cast<std::streamsize>(b.size()));
}

int main()
{
    bool ok = true;

    // Deterministic identical topology; model A is seeded, model B keeps its
    // random constructor weights until load() overwrites them.
    auto l1a = std::make_shared<Linear>(2, 3);
    auto l2a = std::make_shared<Linear>(3, 1);
    Sequential a({l1a, std::make_shared<Sigmoid>(), l2a, std::make_shared<Sigmoid>()});
    seed(l1a->weight, 0.31f);
    seed(l1a->bias, -0.12f);
    seed(l2a->weight, 0.07f);
    seed(l2a->bias, 0.23f);

    auto l1b = std::make_shared<Linear>(2, 3);
    auto l2b = std::make_shared<Linear>(3, 1);
    Sequential b({l1b, std::make_shared<Sigmoid>(), l2b, std::make_shared<Sigmoid>()});

    Tensor X({4, 2});
    const float xs[8] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    for (std::size_t i = 0; i < 8; ++i)
        X.data->at(i) = xs[i];

    const Tensor pred_a = a.forward(X);
    const std::vector<Tensor *> weights_a = a.parameters();

    if (!ser::save(kGood, weights_a))
    {
        std::cerr << "FAIL save() returned false\n";
        return 1;
    }

    std::vector<Tensor *> weights_b = b.parameters();
    if (!ser::load(kGood, weights_b))
    {
        std::cerr << "FAIL load() returned false on a valid file\n";
        return 1;
    }

    const Tensor pred_b = b.forward(X);
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < pred_a.total_size; ++i)
    {
        const float diff = std::fabs(pred_a.data->at(i) - pred_b.data->at(i));
        if (diff > max_diff)
            max_diff = diff;
    }
    std::cout << "round-trip max |pred_a - pred_b| = " << max_diff << "\n";
    if (max_diff > kPredTolerance || !std::isfinite(max_diff))
    {
        std::cerr << "FAIL predictions differ after round-trip\n";
        ok = false;
    }

    // Re-save the loaded model and compare bytes: the format must be stable.
    if (!ser::save("serialization_resaved.bin", weights_b) ||
        read_bytes("serialization_resaved.bin") != read_bytes(kGood))
    {
        std::cerr << "FAIL re-saved file is not byte-identical\n";
        ok = false;
    }

    // Corruption cases must all be rejected cleanly (no throw, false return).
    const std::vector<uint8_t> good = read_bytes(kGood);
    if (good.size() < 16)
    {
        std::cerr << "FAIL serialized file unexpectedly short\n";
        return 1;
    }

    write_bytes(kTruncated, std::vector<uint8_t>(good.begin(), good.begin() + good.size() / 2));
    std::vector<uint8_t> bad_magic = good;
    bad_magic[0] = static_cast<uint8_t>('X');
    write_bytes(kBadMagic, bad_magic);
    std::vector<uint8_t> bad_version = good;
    bad_version[4] = 9; // version low byte -> unsupported
    write_bytes(kBadVersion, bad_version);

    for (const char *path : {kTruncated, kBadMagic, kBadVersion})
    {
        std::vector<Tensor *> fresh = b.parameters();
        const bool rejected = !ser::load(path, fresh);
        std::cout << "reject " << path << ": " << (rejected ? "yes" : "NO") << "\n";
        if (!rejected)
            ok = false;
    }

    // Count mismatch must also be rejected.
    std::vector<Tensor *> too_few(weights_b.begin(), weights_b.end() - 1);
    if (ser::load(kGood, too_few))
    {
        std::cerr << "FAIL load() accepted the wrong tensor count\n";
        ok = false;
    }

    // Shape mismatch must be rejected without touching the destination.
    auto l1c = std::make_shared<Linear>(3, 2);
    auto l2c = std::make_shared<Linear>(1, 3);
    Sequential c({l1c, std::make_shared<Sigmoid>(), l2c, std::make_shared<Sigmoid>()});
    std::vector<Tensor *> wrong_shape = c.parameters();
    const std::vector<std::vector<float>> before_wrong = {
        *l1c->weight.data, *l1c->bias.data, *l2c->weight.data, *l2c->bias.data};
    if (ser::load(kGood, wrong_shape))
    {
        std::cerr << "FAIL load() accepted a destination with the wrong shapes\n";
        ok = false;
    }
    const std::vector<std::vector<float>> after_wrong = {
        *l1c->weight.data, *l1c->bias.data, *l2c->weight.data, *l2c->bias.data};
    if (before_wrong != after_wrong)
    {
        std::cerr << "FAIL a rejected load mutated the destination tensors\n";
        ok = false;
    }

    // Trailing garbage must be rejected.
    std::vector<uint8_t> trailing = good;
    trailing.push_back(0xAA);
    write_bytes("serialization_trailing.bin", trailing);
    if (ser::load("serialization_trailing.bin", weights_b))
    {
        std::cerr << "FAIL load() accepted trailing garbage\n";
        ok = false;
    }

    // Crafted huge dimensions: the truncation guard must not overflow size_t.
    // dims [0x7FFFFFFF, 0x80000001] with no payload previously wrapped
    // `pos + expected * 4` and threw std::length_error from reserve().
    std::vector<uint8_t> huge;
    for (char ch : std::string("MMLW"))
        huge.push_back(static_cast<uint8_t>(ch));
    huge.push_back(1); // version low byte
    huge.push_back(0); // version high byte
    auto append_u32 = [&huge](uint32_t v) {
        for (int shift = 0; shift < 32; shift += 8)
            huge.push_back(static_cast<uint8_t>((v >> shift) & 0xFFu));
    };
    append_u32(1);           // tensor count
    append_u32(2);           // rank
    append_u32(0x7FFFFFFFu); // dim 0
    append_u32(0x80000001u); // dim 1: pos + product*4 wraps in size_t
    write_bytes("serialization_hugedim.bin", huge);
    bool huge_rejected = false;
    try
    {
        // Count must match the target, otherwise load() rejects on count before
        // reaching the dimension/truncation guard this test exists to exercise.
        std::vector<Tensor *> target{weights_b[0]};
        huge_rejected = !ser::load("serialization_hugedim.bin", target);
    }
    catch (...)
    {
        std::cerr << "FAIL load() threw on crafted huge dimensions\n";
        ok = false;
    }
    std::cout << "reject crafted huge dims: " << (huge_rejected ? "yes" : "NO") << "\n";
    if (!huge_rejected)
        ok = false;

    // Leave no scratch files behind when the binary is invoked from the repo root.
    for (const char *path : {kGood, kTruncated, kBadMagic, kBadVersion,
                             "serialization_resaved.bin", "serialization_trailing.bin",
                             "serialization_hugedim.bin"})
        std::remove(path);

    if (ok)
        std::cout << "PASS serialization\n";
    else
        std::cerr << "FAIL serialization\n";
    return ok ? 0 : 1;
}
