#include "tensor.hpp"
#include <stdexcept>

inline float *get_raw(const std::shared_ptr<data_vec> &ptr) { return ptr->data(); }

Tensor::Tensor(std::vector<size_t> shape) : shape(shape)
{
    for (auto &s : shape)
        total_size *= s;
    data = std::make_shared<data_vec>(total_size, 0.0f);
    grad = std::make_shared<data_vec>(total_size, 0.0f);
    calculate_stride();
}

void Tensor::calculate_stride()
{
    strides.resize(shape.size());
    size_t curr_stride{1};
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        strides[i] = curr_stride;
        curr_stride *= shape[i];
    }
}

Tensor Tensor::randn(size_vec shape)
{
    Tensor t(shape);
    static thread_local std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float scale = std::sqrt(2.0f / (t.total_size + 1));
    float *ptr = get_raw(t.data);
    for (size_t i = 0; i < t.total_size; ++i)
        ptr[i] = dist(gen) * scale;
    return t;
}

void Tensor::backward()
{
    std::vector<Tensor> topo;
    std::set<float *> visited;

    std::function<void(Tensor &)> build_topo = [&](Tensor &v)
    {
        if (visited.count(get_raw(v.data)))
            return;
        visited.insert(get_raw(v.data));

        for (auto &child_ptr : v._prev)
        {
            build_topo(*child_ptr);
        }
        topo.push_back(v);
    };
    build_topo(*this);

    std::fill(grad->begin(), grad->end(), 0.0f);
    get_raw(grad)[0] = 1.0f;

    if (total_size > 1)
        std::fill(grad->begin(), grad->end(), 1.0f);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        if (it->_backward)
            it->_backward();
}

void Tensor::print() const
{
    std::cout << "Tensor(" << shape[0];
    for (size_t i = 1; i < shape.size(); ++i)
        std::cout << "x" << shape[i];
    std::cout << ", op=" << (_op.empty() ? "None" : _op) << ")\n";
}

void Tensor::zero_grad()
{
    std::fill(grad->begin(), grad->end(), 0.0f);
}

Tensor Tensor::operator+(const Tensor &t2) const
{
    bool broadcast = (shape.size() == 2 && t2.shape.size() == 2 &&
                      shape[0] != t2.shape[0] && t2.shape[0] == 1 && shape[1] == t2.shape[1]);

    if (shape != t2.shape && !broadcast)
        throw std::invalid_argument("Shape mismatch in +");

    Tensor t3(shape);
    t3._op = "+";

    t3._prev = {std::make_shared<Tensor>(*this), std::make_shared<Tensor>(t2)};

    float *d3 = get_raw(t3.data), *d1 = get_raw(data), *d2 = get_raw(t2.data);

    if (!broadcast)
    {
        for (size_t i = 0; i < total_size; ++i)
            d3[i] = d1[i] + d2[i];
    }
    else
    {
        size_t N = shape[0], M = shape[1];
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < M; ++j)
                d3[i * M + j] = d1[i * M + j] + d2[j];
    }

    t3._backward = [self = *this, other = t2, res = t3, broadcast]() mutable
    {
        float *sg = get_raw(self.grad), *og = get_raw(other.grad), *rg = get_raw(res.grad);
        if (!broadcast)
        {
            for (size_t i = 0; i < res.total_size; ++i)
            {
                sg[i] += rg[i];
                og[i] += rg[i];
            }
        }
        else
        {
            size_t N = self.shape[0], M = self.shape[1];
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < M; ++j)
                {
                    float g = rg[i * M + j];
                    sg[i * M + j] += g;
                    og[j] += g;
                }
        }
    };
    return t3;
}

Tensor Tensor::operator-(const Tensor &t2) const
{
    if (shape != t2.shape)
        throw std::invalid_argument("Shape mismatch in -");
    Tensor t3(shape);
    t3._op = "-";
    t3._prev = {std::make_shared<Tensor>(*this), std::make_shared<Tensor>(t2)};

    float *d3 = get_raw(t3.data), *d1 = get_raw(data), *d2 = get_raw(t2.data);
    for (size_t i = 0; i < total_size; ++i)
        d3[i] = d1[i] - d2[i];

    t3._backward = [self = *this, other = t2, res = t3]() mutable
    {
        float *sg = get_raw(self.grad), *og = get_raw(other.grad), *rg = get_raw(res.grad);
        for (size_t i = 0; i < res.total_size; ++i)
        {
            sg[i] += rg[i];
            og[i] -= rg[i];
        }
    };
    return t3;
}

Tensor Tensor::operator*(const Tensor &t2) const
{

    if (shape != t2.shape)
        throw std::invalid_argument("Shape mismatch in * (element-wise)");

    Tensor t3(shape);
    t3._op = "*";
    t3._prev = {std::make_shared<Tensor>(*this), std::make_shared<Tensor>(t2)};

    float *d3 = get_raw(t3.data), *d1 = get_raw(data), *d2 = get_raw(t2.data);
    for (size_t i = 0; i < total_size; ++i)
        d3[i] = d1[i] * d2[i];

    t3._backward = [self = *this, other = t2, res = t3]() mutable
    {
        float *sg = get_raw(self.grad), *og = get_raw(other.grad), *rg = get_raw(res.grad);
        float *d1 = get_raw(self.data), *d2 = get_raw(other.data);
        for (size_t i = 0; i < res.total_size; ++i)
        {
            sg[i] += d2[i] * rg[i];
            og[i] += d1[i] * rg[i];
        }
    };
    return t3;
}

Tensor Tensor::matmul(const Tensor &t2) const
{
    size_t M = shape[0], K = shape[1], N = t2.shape[1];
    Tensor t3({M, N});
    t3._op = "matmul";
    t3._prev = {std::make_shared<Tensor>(*this), std::make_shared<Tensor>(t2)};

    float *A = get_raw(data), *B = get_raw(t2.data), *C = get_raw(t3.data);

    std::fill(C, C + M * N, 0.0f);
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t k = 0; k < K; ++k)
        {
            float r = A[i * K + k];
            for (size_t j = 0; j < N; ++j)
                C[i * N + j] += r * B[k * N + j];
        }
    }

    t3._backward = [self = *this, other = t2, res = t3]() mutable
    {
        float *sg = get_raw(self.grad), *og = get_raw(other.grad), *rg = get_raw(res.grad);
        float *sd = get_raw(self.data), *od = get_raw(other.data);
        size_t M = self.shape[0], K = self.shape[1], N = other.shape[1];

        for (size_t i = 0; i < M; ++i)
            for (size_t k = 0; k < K; ++k)
            {
                float sum = 0.0f;
                for (size_t j = 0; j < N; ++j)
                    sum += rg[i * N + j] * od[k * N + j];
                sg[i * K + k] += sum;
            }
        for (size_t k = 0; k < K; ++k)
            for (size_t j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (size_t i = 0; i < M; ++i)
                    sum += sd[i * K + k] * rg[i * N + j];
                og[k * N + j] += sum;
            }
    };
    return t3;
}

Tensor Tensor::transpose() const
{
    Tensor t2({shape[1], shape[0]});
    t2._op = "T";
    t2._prev = {std::make_shared<Tensor>(*this)};

    size_t M = shape[0], N = shape[1];
    float *src = get_raw(data), *dst = get_raw(t2.data);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            dst[j * M + i] = src[i * N + j];

    t2._backward = [self = *this, res = t2]() mutable
    {
        size_t M = self.shape[0], N = self.shape[1];
        float *sg = get_raw(self.grad), *rg = get_raw(res.grad);
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                sg[i * N + j] += rg[j * M + i];
    };
    return t2;
}

Tensor Tensor::pow(float exp) const
{
    Tensor t2(shape);
    t2._op = "pow";
    t2._prev = {std::make_shared<Tensor>(*this)};

    float *d1 = get_raw(data), *d2 = get_raw(t2.data);
    for (size_t i = 0; i < total_size; ++i)
        d2[i] = std::pow(d1[i], exp);

    t2._backward = [self = *this, res = t2, exp]() mutable
    {
        float *sg = get_raw(self.grad), *rg = get_raw(res.grad), *sd = get_raw(self.data);
        for (size_t i = 0; i < res.total_size; ++i)
            sg[i] += exp * std::pow(sd[i], exp - 1.0f) * rg[i];
    };
    return t2;
}

Tensor Tensor::sum() const
{
    Tensor t2({1});
    t2._op = "sum";
    t2._prev = {std::make_shared<Tensor>(*this)};

    float *d1 = get_raw(data);
    t2.data->at(0) = std::accumulate(d1, d1 + total_size, 0.0f);

    t2._backward = [self = *this, res = t2]() mutable
    {
        float inc = res.grad->at(0);
        float *sg = get_raw(self.grad);
        for (size_t i = 0; i < self.total_size; ++i)
            sg[i] += inc;
    };
    return t2;
}

Tensor Tensor::relu() const
{
    Tensor t2(shape);
    t2._op = "relu";
    t2._prev = {std::make_shared<Tensor>(*this)};

    float *d1 = get_raw(data), *d2 = get_raw(t2.data);
    for (size_t i = 0; i < total_size; ++i)
        d2[i] = d1[i] > 0.0f ? d1[i] : 0.0f;

    t2._backward = [self = *this, res = t2]() mutable
    {
        float *sg = get_raw(self.grad), *rg = get_raw(res.grad), *sd = get_raw(self.data);
        for (size_t i = 0; i < res.total_size; ++i)
            if (sd[i] > 0.0f)
                sg[i] += rg[i];
    };
    return t2;
}

Tensor Tensor::sigmoid() const
{
    Tensor t2(shape);
    t2._op = "sigmoid";
    t2._prev = {std::make_shared<Tensor>(*this)};

    float *d1 = get_raw(data), *d2 = get_raw(t2.data);
    for (size_t i = 0; i < total_size; ++i)
        d2[i] = 1.0f / (1.0f + std::exp(-d1[i]));

    t2._backward = [self = *this, res = t2]() mutable
    {
        float *sg = get_raw(self.grad), *rg = get_raw(res.grad), *rd = get_raw(res.data);
        for (size_t i = 0; i < res.total_size; ++i)
        {
            float y = rd[i];
            sg[i] += (y * (1.0f - y)) * rg[i];
        }
    };
    return t2;
}

Tensor Tensor::log() const
{
    Tensor t2(shape);
    t2._op = "log";
    t2._prev = {std::make_shared<Tensor>(*this)};

    float *d1 = get_raw(data), *d2 = get_raw(t2.data);
    float eps = 1e-7f;
    for (size_t i = 0; i < total_size; ++i)
        d2[i] = std::log(d1[i] + eps);

    t2._backward = [self = *this, res = t2, eps]() mutable
    {
        float *sg = get_raw(self.grad), *rg = get_raw(res.grad), *sd = get_raw(self.data);
        for (size_t i = 0; i < res.total_size; ++i)
            sg[i] += (1.0f / (sd[i] + eps)) * rg[i];
    };
    return t2;
}