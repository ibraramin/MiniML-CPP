#ifndef MODELS_H
#define MODELS_H

#include "nn.hpp"
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>

class Estimator
{
public:
    virtual std::vector<float> fit(const Tensor &X, const Tensor &y, int epochs, float lr) = 0;
    virtual Tensor predict(const Tensor &X) = 0;
    virtual ~Estimator() = default;
};

class LinearRegression : public Estimator
{
    std::shared_ptr<Linear> layer;

public:
    LinearRegression(int n_features)
    {
        layer = std::make_shared<Linear>(n_features, 1);
    }

    std::vector<float> fit(const Tensor &X, const Tensor &y, int epochs = 1000, float lr = 0.01f) override
    {
        std::vector<float> hist;
        SGD optim(layer->parameters(), lr);
        MSELoss criterion;

        for (int i = 0; i < epochs; ++i)
        {
            optim.zero_grad();
            Tensor pred = layer->forward(X);
            Tensor loss = criterion(pred, y);
            loss.backward();
            optim.step();
            hist.push_back(loss.data->at(0));
        }
        return hist;
    }

    Tensor predict(const Tensor &X) override { return layer->forward(X); }
    float get_weight(int idx) { return layer->weight.data->at(idx); }
    float get_bias() { return layer->bias.data->at(0); }
};

class RidgeRegression : public Estimator
{
    std::shared_ptr<Linear> layer;
    float lambda;

public:
    RidgeRegression(int n_features, float lambda = 0.1f) : lambda(lambda)
    {
        layer = std::make_shared<Linear>(n_features, 1);
    }

    std::vector<float> fit(const Tensor &X, const Tensor &y, int epochs = 1000, float lr = 0.01f) override
    {
        std::vector<float> hist;
        SGD optim(layer->parameters(), lr);
        MSELoss mse;

        for (int i = 0; i < epochs; ++i)
        {
            optim.zero_grad();
            Tensor pred = layer->forward(X);
            Tensor data_loss = mse(pred, y);
            data_loss.backward();

            float *w = layer->weight.data->data();
            float *g = layer->weight.grad->data();
            for (size_t k = 0; k < layer->weight.total_size; ++k)
                g[k] += 2.0f * lambda * w[k];

            optim.step();
            hist.push_back(data_loss.data->at(0));
        }
        return hist;
    }
    Tensor predict(const Tensor &X) override { return layer->forward(X); }
};

class PolynomialRegression : public Estimator
{
    std::shared_ptr<Linear> layer;
    int degree;

    Tensor expand(const Tensor &X)
    {
        size_t N = X.shape[0];
        Tensor P({N, (size_t)degree});
        for (size_t i = 0; i < N; ++i)
            for (int d = 0; d < degree; ++d)
                P.data->at(i * degree + d) = std::pow(X.data->at(i), d + 1);
        return P;
    }

public:
    PolynomialRegression(int degree) : degree(degree)
    {
        layer = std::make_shared<Linear>(degree, 1);
    }
    std::vector<float> fit(const Tensor &X, const Tensor &y, int epochs, float lr) override
    {
        Tensor P = expand(X);
        SGD optim(layer->parameters(), lr);
        MSELoss criterion;
        std::vector<float> hist;
        for (int i = 0; i < epochs; ++i)
        {
            optim.zero_grad();
            Tensor pred = layer->forward(P);
            Tensor loss = criterion(pred, y);
            loss.backward();
            optim.step();
            hist.push_back(loss.data->at(0));
        }
        return hist;
    }
    Tensor predict(const Tensor &X) override { return layer->forward(expand(X)); }
};

class LogisticRegression : public Estimator
{
    std::shared_ptr<Linear> layer;
    std::shared_ptr<Sigmoid> act;

public:
    LogisticRegression(int n_features)
    {
        layer = std::make_shared<Linear>(n_features, 1);
        act = std::make_shared<Sigmoid>();
    }
    std::vector<float> fit(const Tensor &X, const Tensor &y, int epochs = 1000, float lr = 0.1f) override
    {
        std::vector<float> hist;
        SGD optim(layer->parameters(), lr);
        Tensor neg_one({1});
        neg_one.data->at(0) = -1.0f;

        for (int i = 0; i < epochs; ++i)
        {
            optim.zero_grad();
            Tensor pred = act->forward(layer->forward(X));

            Tensor log_p = pred.log();
            Tensor ones({X.shape[0], 1});
            std::fill(ones.data->begin(), ones.data->end(), 1.0f);

            Tensor t1 = y * log_p;
            Tensor t2 = (ones - y) * (ones - pred).log();
            Tensor loss = (t1 + t2).sum() * neg_one;

            loss.backward();
            optim.step();
            hist.push_back(loss.data->at(0));
        }
        return hist;
    }
    Tensor predict(const Tensor &X) override { return act->forward(layer->forward(X)); }
};

class LinearSVM : public Estimator
{
    std::shared_ptr<Linear> layer;

public:
    LinearSVM(int n_features)
    {
        layer = std::make_shared<Linear>(n_features, 1);
    }
    std::vector<float> fit(const Tensor &X, const Tensor &y, int epochs = 1000, float lr = 0.01f) override
    {
        std::vector<float> hist;
        SGD optim(layer->parameters(), lr);

        Tensor y_svm(y.shape);
        for (size_t i = 0; i < y.total_size; ++i)
            y_svm.data->at(i) = (y.data->at(i) < 0.5f) ? -1.0f : 1.0f;
        Tensor ones({X.shape[0], 1});
        std::fill(ones.data->begin(), ones.data->end(), 1.0f);

        for (int i = 0; i < epochs; ++i)
        {
            optim.zero_grad();
            Tensor scores = layer->forward(X);
            Tensor margin = ones - (y_svm * scores);
            Tensor loss = margin.relu().sum();

            loss.backward();
            optim.step();
            hist.push_back(loss.data->at(0));
        }
        return hist;
    }
    Tensor predict(const Tensor &X) override
    {
        Tensor s = layer->forward(X);
        Tensor p(s.shape);
        for (size_t i = 0; i < s.total_size; ++i)
            p.data->at(i) = (s.data->at(i) >= 0) ? 1.0f : 0.0f;
        return p;
    }
};

class MLPClassifier : public Estimator
{
    std::shared_ptr<Sequential> model;

public:
    MLPClassifier(int in, int h, int out)
    {
        model = std::make_shared<Sequential>(std::initializer_list<std::shared_ptr<Module>>{
            std::make_shared<Linear>(in, h), std::make_shared<ReLU>(),
            std::make_shared<Linear>(h, h), std::make_shared<ReLU>(),
            std::make_shared<Linear>(h, out), std::make_shared<Sigmoid>()});
    }
    std::vector<float> fit(const Tensor &X, const Tensor &y, int epochs, float lr) override
    {
        std::vector<float> hist;
        SGD optim(model->parameters(), lr);
        MSELoss criterion;
        for (int i = 0; i < epochs; ++i)
        {
            optim.zero_grad();
            Tensor pred = model->forward(X);
            Tensor loss = criterion(pred, y);
            loss.backward();
            optim.step();
            hist.push_back(loss.data->at(0));
        }
        return hist;
    }
    Tensor predict(const Tensor &X) override { return model->forward(X); }
};
#endif