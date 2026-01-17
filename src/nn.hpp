#ifndef NN_H
#define NN_H

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <iomanip>

class Module
{
public:
    virtual ~Module() = default;
    virtual Tensor forward(Tensor x) = 0;
    virtual std::vector<Tensor *> parameters() const = 0;

    Tensor operator()(const Tensor &x) { return forward(x); }

    void zero_grad()
    {
        for (auto *p : parameters())
            std::fill(p->grad->begin(), p->grad->end(), 0.0f);
    }
};

class Linear : public Module
{
public:
    Tensor weight, bias;
    Linear(int in, int out)
        : weight(Tensor::randn({(size_t)out, (size_t)in})),
          bias(Tensor::randn({1, (size_t)out})) {}

    Tensor forward(Tensor x) override { return x.matmul(weight.transpose()) + bias; }
    std::vector<Tensor *> parameters() const override
    {
        return {const_cast<Tensor *>(&weight), const_cast<Tensor *>(&bias)};
    }
};

class Sigmoid : public Module
{
public:
    Tensor forward(Tensor x) override { return x.sigmoid(); }
    std::vector<Tensor *> parameters() const override { return {}; }
};

class ReLU : public Module
{
public:
    Tensor forward(Tensor x) override { return x.relu(); }
    std::vector<Tensor *> parameters() const override { return {}; }
};

class Sequential : public Module
{
    std::vector<std::shared_ptr<Module>> layers;

public:
    Sequential(std::initializer_list<std::shared_ptr<Module>> list) : layers(list) {}

    Tensor forward(Tensor x) override
    {
        Tensor out = x;
        for (auto &l : layers)
            out = l->forward(out);
        return out;
    }
    std::vector<Tensor *> parameters() const override
    {
        std::vector<Tensor *> params;
        for (auto &l : layers)
        {
            auto p = l->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};

class SGD
{
    std::vector<Tensor *> params;
    float lr;

public:
    SGD(std::vector<Tensor *> params, float lr) : params(params), lr(lr) {}
    void zero_grad()
    {
        for (auto *p : params)
            std::fill(p->grad->begin(), p->grad->end(), 0.0f);
    }
    void step()
    {
        for (auto *p : params)
        {
            float *d = p->data->data();
            float *g = p->grad->data();
            for (size_t i = 0; i < p->total_size; ++i)
                d[i] -= lr * g[i];
        }
    }
};

class MSELoss
{
public:
    Tensor operator()(const Tensor &pred, const Tensor &target)
    {
        return (pred - target).pow(2.0f).sum();
    }
};

#endif