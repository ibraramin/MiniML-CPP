#include "tensor.hpp"
#include "models.hpp"
#include "plot.hpp"
#include "metrics.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

struct DemoDataset
{
    Tensor X;
    Tensor y;
};

DemoDataset make_linear(size_t n = 100, float noise = 1.0f)
{
    Tensor X({n, 1});
    Tensor y({n, 1});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dx(0, 5);
    std::normal_distribution<float> dn(0, noise);
    for (size_t i = 0; i < n; ++i)
    {
        float x = dx(gen);
        X.data->at(i) = x;

        y.data->at(i) = 2.5f * x + 1.0f + dn(gen);
    }
    return {X, y};
}

DemoDataset make_curve(size_t n = 100, float noise = 0.5f)
{
    Tensor X({n, 1});
    Tensor y({n, 1});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dx(-3, 3);
    std::normal_distribution<float> dn(0, noise);
    for (size_t i = 0; i < n; ++i)
    {
        float x = dx(gen);
        X.data->at(i) = x;
        y.data->at(i) = 0.5f * x * x - 1.5f * x + 2.0f + dn(gen);
    }

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n - 1; ++j)
            if (X.data->at(j) > X.data->at(j + 1))
            {
                std::swap(X.data->at(j), X.data->at(j + 1));
                std::swap(y.data->at(j), y.data->at(j + 1));
            }
    return {X, y};
}

DemoDataset make_blobs(size_t n = 200)
{
    Tensor X({n, 2});
    Tensor y({n, 1});
    std::mt19937 gen(42);
    std::normal_distribution<float> d(0, 0.4f);
    for (size_t i = 0; i < n; ++i)
    {
        float label = (i % 2 == 0) ? 0.0f : 1.0f;
        float cx = (label == 0) ? -1.0f : 1.0f;
        X.data->at(i * 2) = cx + d(gen);
        X.data->at(i * 2 + 1) = cx + d(gen);
        y.data->at(i) = label;
    }
    return {X, y};
}

DemoDataset make_circles(size_t n = 500, float noise = 0.08f)
{
    Tensor X({n, 2});
    Tensor y({n, 1});
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> da(0, 6.28f);
    std::normal_distribution<float> dn(0, noise);
    for (size_t i = 0; i < n; ++i)
    {
        float a = da(gen);
        float r = (i % 2 == 0) ? 0.5f : 1.1f;
        r += dn(gen);
        X.data->at(i * 2) = r * std::cos(a);
        X.data->at(i * 2 + 1) = r * std::sin(a);
        y.data->at(i) = (i % 2 == 0) ? 0.0f : 1.0f;
    }
    return {X, y};
}

void standard_scale(Tensor &X)
{
    size_t N = X.shape[0];
    size_t D = X.shape[1];
    for (size_t j = 0; j < D; ++j)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i)
            sum += X.data->at(i * D + j);
        float mean = sum / N;

        float var = 0.0f;
        for (size_t i = 0; i < N; ++i)
            var += std::pow(X.data->at(i * D + j) - mean, 2);
        float std_dev = std::sqrt(var / N) + 1e-6f;

        for (size_t i = 0; i < N; ++i)
            X.data->at(i * D + j) = (X.data->at(i * D + j) - mean) / std_dev;
    }
}

void wait()
{
    std::cout << ">> Press ENTER to continue...";
    std::cin.get();
}

int main()
{
    Plotter plotter;
    std::cout << "=== C++ ML Library Final Showcase ===\n\n";

    std::cout << "[1] Linear Regression\n";
    auto d1 = make_linear(100, 1.0f);
    standard_scale(d1.X);

    LinearRegression model1(1);
    model1.fit(d1.X, d1.y, 2000, 0.001f);

    std::cout << "Weight (Scaled): " << model1.get_weight(0) << "\n";
    std::cout << "Bias: " << model1.get_bias() << "\n";
    plotter.plot_regression(d1.X, d1.y, model1.predict(d1.X), "Linear Reg (Scaled Data)");
    wait();

    std::cout << "[2] Ridge Regression\n";
    RidgeRegression model2(1, 0.5f);

    auto h2 = model2.fit(d1.X, d1.y, 2000, 0.001f);

    plotter.plot_loss(h2, "Ridge Loss");
    wait();
    plotter.plot_regression(d1.X, d1.y, model2.predict(d1.X), "Ridge Fit");
    wait();

    std::cout << "[3] Polynomial Regression\n";
    auto d3 = make_curve(100, 0.4f);
    standard_scale(d3.X);

    PolynomialRegression model3(2);

    model3.fit(d3.X, d3.y, 3000, 0.001f);

    plotter.plot_regression(d3.X, d3.y, model3.predict(d3.X), "Poly Fit");
    wait();

    std::cout << "[4] Logistic Regression\n";
    auto d4 = make_blobs(200);
    LogisticRegression model4(2);

    model4.fit(d4.X, d4.y, 1000, 0.1f);

    Tensor p4 = model4.predict(d4.X);
    std::cout << "Acc: " << Metrics::accuracy(d4.y, p4) * 100 << "%\n";
    plotter.plot_boundary(d4.X, d4.y, p4, "Logistic Boundary");
    wait();

    std::cout << "[5] Linear SVM\n";
    LinearSVM model5(2);
    model5.fit(d4.X, d4.y, 1000, 0.01f);

    Tensor p5 = model5.predict(d4.X);
    std::cout << "Acc: " << Metrics::accuracy(d4.y, p5) * 100 << "%\n";
    plotter.plot_boundary(d4.X, d4.y, p5, "SVM Boundary");
    wait();

    std::cout << "[6] MLP Classifier (Donut)\n";
    auto d6 = make_circles(500, 0.08f);

    MLPClassifier model6(2, 16, 1);
    auto h6 = model6.fit(d6.X, d6.y, 5000, 0.01f);

    Tensor p6 = model6.predict(d6.X);
    std::cout << "Acc: " << Metrics::accuracy(d6.y, p6) * 100 << "%\n";
    plotter.plot_loss(h6, "MLP Loss");
    wait();
    plotter.plot_boundary(d6.X, d6.y, p6, "MLP Donut Boundary");
    wait();

    return 0;
}