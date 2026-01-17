#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>
#include <string>

using size_vec = std::vector<size_t>;
using data_vec = std::vector<float>;

class Tensor
{
public:
    std::shared_ptr<data_vec> data;
    std::shared_ptr<data_vec> grad;

    std::vector<std::shared_ptr<Tensor>> _prev;

    std::string _op;
    std::function<void()> _backward;
    size_vec shape;
    size_vec strides;
    size_t total_size{1};

    Tensor(size_vec shape);
    void calculate_stride();

    static Tensor randn(size_vec shape);

    void backward();
    void zero_grad();

    Tensor operator+(const Tensor &t2) const;
    Tensor operator-(const Tensor &t2) const;
    Tensor operator*(const Tensor &t2) const;

    Tensor matmul(const Tensor &t2) const;
    Tensor transpose() const;
    Tensor pow(float exp) const;
    Tensor sum() const;

    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor log() const;

    void print() const;
};

#endif