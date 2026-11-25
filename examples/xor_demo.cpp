// Deterministic 2-2-1 XOR demo: trains the same seeded network with SGD and
// with Adam, emitting an epoch,loss CSV for each optimiser.

#include "tensor.hpp"
#include "nn.hpp"
#include "adam.hpp"
#include "metrics.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

// Thresholds are measured from this exact deterministic configuration; see
// /var/tmp/opencode/miniml-scratch.md for the measurement procedure and values.
static constexpr float kSgdThreshold = 0.01f;
static constexpr float kAdamThreshold = 0.01f;

static constexpr int kEpochs = 3000;
static constexpr unsigned kSeed = 1;
static constexpr float kInitScale = 1.0f;
static constexpr float kSgdLr = 0.5f;
static constexpr float kAdamLr = 0.05f;

static void seed(Tensor &t, std::mt19937 &gen)
{
    std::uniform_real_distribution<float> dist(-kInitScale, kInitScale);
    for (std::size_t i = 0; i < t.total_size; ++i)
        t.data->at(i) = dist(gen);
}

struct Dataset
{
    Tensor X;
    Tensor y;
};

static Dataset make_dataset()
{
    Tensor X({4, 2});
    Tensor y({4, 1});
    const float xs[8] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    const float ys[4] = {0.0f, 1.0f, 1.0f, 0.0f};
    for (int i = 0; i < 8; ++i)
        X.data->at(i) = xs[i];
    for (int i = 0; i < 4; ++i)
        y.data->at(i) = ys[i];
    return {X, y};
}

struct Model
{
    std::shared_ptr<Linear> l1;
    std::shared_ptr<Linear> l2;
    Sequential net;

    Model()
        : l1(std::make_shared<Linear>(2, 2)),
          l2(std::make_shared<Linear>(2, 1)),
          net({l1, std::make_shared<Sigmoid>(), l2, std::make_shared<Sigmoid>()})
    {
        std::mt19937 gen(kSeed); // fixed PRNG seed => deterministic weights
        seed(l1->weight, gen);
        seed(l1->bias, gen);
        seed(l2->weight, gen);
        seed(l2->bias, gen);
    }
};

template <typename Optimizer>
static float train(Model &model, const Tensor &X, const Tensor &y, float lr, const char *mode)
{
    Optimizer optim(model.net.parameters(), lr);
    MSELoss mse;

    std::cout << "# mode=" << mode << " lr=" << lr << " seed=" << kSeed
              << " epochs=" << kEpochs << "\n";
    std::cout << "epoch,loss\n";
    std::cout << std::fixed << std::setprecision(6);

    for (int epoch = 0; epoch < kEpochs; ++epoch)
    {
        optim.zero_grad();
        Tensor pred = model.net.forward(X);
        Tensor loss = mse(pred, y);
        loss.backward();
        optim.step();
        std::cout << epoch << "," << loss.data->at(0) << "\n";
    }

    // Final evaluation: a fresh forward pass, no gradient state involved.
    return mse(model.net.forward(X), y).data->at(0);
}

int main()
{
    const Dataset data = make_dataset();
    bool ok = true;

    Model sgd_model;
    const float sgd_final = train<SGD>(sgd_model, data.X, data.y, kSgdLr, "sgd");
    const Tensor sgd_pred = sgd_model.net.forward(data.X);

    Model adam_model;
    const float adam_final = train<Adam>(adam_model, data.X, data.y, kAdamLr, "adam");
    const Tensor adam_pred = adam_model.net.forward(data.X);
    const float sgd_acc = Metrics::accuracy(data.y, sgd_pred);
    const float adam_acc = Metrics::accuracy(data.y, adam_pred);

    std::cout << "# results\n";
    std::cout << "sgd_final=" << sgd_final << " threshold=" << kSgdThreshold << "\n";
    std::cout << "adam_final=" << adam_final << " threshold=" << kAdamThreshold << "\n";
    std::cout << "sgd_accuracy=" << sgd_acc << " adam_accuracy=" << adam_acc << "\n";
    std::cout << "sgd_predictions=";
    for (std::size_t i = 0; i < sgd_pred.total_size; ++i)
        std::cout << sgd_pred.data->at(i) << (i + 1 == sgd_pred.total_size ? '\n' : ',');
    std::cout << "adam_predictions=";
    for (std::size_t i = 0; i < adam_pred.total_size; ++i)
        std::cout << adam_pred.data->at(i) << (i + 1 == adam_pred.total_size ? '\n' : ',');

    if (!std::isfinite(sgd_final) || sgd_final >= kSgdThreshold)
    {
        std::cerr << "FAIL: SGD did not converge (final=" << sgd_final << ")\n";
        ok = false;
    }
    if (!std::isfinite(adam_final) || adam_final >= kAdamThreshold)
    {
        std::cerr << "FAIL: Adam did not converge (final=" << adam_final << ")\n";
        ok = false;
    }
    if (sgd_acc < 0.999f || adam_acc < 0.999f)
    {
        std::cerr << "FAIL: XOR classification accuracy below 4/4 (sgd=" << sgd_acc
                  << " adam=" << adam_acc << ")\n";
        ok = false;
    }

    if (ok)
        std::cout << "PASS xor_demo\n";
    else
        std::cerr << "FAIL xor_demo\n";
    return ok ? 0 : 1;
}
