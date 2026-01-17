#ifndef DATA_H
#define DATA_H

#include "tensor.hpp"
#include <algorithm>
#include <numeric>

struct Dataset
{
    Tensor X;
    Tensor y;
};

class DataUtils
{
public:
    static std::pair<Dataset, Dataset> train_test_split(
        const Tensor &X, const Tensor &y, float test_size = 0.2f, int seed = 42)
    {
        size_t n_samples = X.shape[0];
        size_t n_features = X.shape[1];
        size_t n_test = static_cast<size_t>(n_samples * test_size);
        size_t n_train = n_samples - n_test;

        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 g(seed);
        std::shuffle(indices.begin(), indices.end(), g);

        Tensor Xt({n_train, n_features}), yt({n_train, y.shape[1]});
        Tensor Xv({n_test, n_features}), yv({n_test, y.shape[1]});

        auto copy_row = [](const Tensor &src, Tensor &dst, size_t s_r, size_t d_r)
        {
            std::copy(src.data->begin() + s_r * src.shape[1],
                      src.data->begin() + (s_r + 1) * src.shape[1],
                      dst.data->begin() + d_r * dst.shape[1]);
        };

        for (size_t i = 0; i < n_train; ++i)
        {
            copy_row(X, Xt, indices[i], i);
            copy_row(y, yt, indices[i], i);
        }
        for (size_t i = 0; i < n_test; ++i)
        {
            copy_row(X, Xv, indices[n_train + i], i);
            copy_row(y, yv, indices[n_train + i], i);
        }
        return {{Xt, yt}, {Xv, yv}};
    }
};
#endif