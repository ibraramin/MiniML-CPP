#ifndef DATA_H
#define DATA_H

#include "tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>

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

// Deterministic mini-batch loader. No autodiff code is involved: the loader
// only reorders and copies rows of Tensor data. Given the same seed and epoch
// index, the permutation of row indices is identical on every run.
class DataLoader
{
public:
    // Forward iterator over the batches of one epoch. Dereferencing copies the
    // selected rows into a fresh Dataset, so batches are independent objects.
    class Iterator
    {
        const DataLoader &loader_;
        std::size_t index_;

    public:
        Iterator(const DataLoader &loader, std::size_t index)
            : loader_(loader), index_(index) {}

        Dataset operator*() const { return loader_.batch(index_); }
        Iterator &operator++()
        {
            ++index_;
            return *this;
        }
        bool operator!=(const Iterator &other) const
        {
            return &loader_ != &other.loader_ || index_ != other.index_;
        }
    };

    DataLoader(const Tensor &X, const Tensor &y, std::size_t batch_size,
               unsigned int seed = 42)
        : X_(X), y_(y), batch_size_(batch_size), seed_(seed)
    {
        if (batch_size_ == 0)
            throw std::invalid_argument("DataLoader: batch_size must be positive");
        if (X_.shape.size() < 2)
            throw std::invalid_argument("DataLoader: X must be rank >= 2");
        if (y_.shape.empty() || y_.shape[0] != X_.shape[0])
            throw std::invalid_argument("DataLoader: X and y sample counts differ");

        y_width_ = (y_.shape.size() >= 2) ? y_.shape[1] : 1;
        order_.resize(X_.shape[0]);
        std::iota(order_.begin(), order_.end(), std::size_t{0});
        reshuffle(0);
    }

    std::size_t size() const { return X_.shape[0]; }
    std::size_t batch_size() const { return batch_size_; }
    unsigned int seed() const { return seed_; }
    std::size_t epoch() const { return epoch_; }

    // ceil(n / batch_size); the final batch may be smaller (remainder batch).
    std::size_t batches_per_epoch() const
    {
        return (size() + batch_size_ - 1) / batch_size_;
    }

    // Rebuild the row order deterministically for a given epoch index.
    // Uses our own Fisher-Yates loop so the permutation is fixed by the seed
    // and does not depend on std::shuffle's implementation details.
    void reshuffle(std::size_t epoch_index)
    {
        std::mt19937 g(seed_ + static_cast<unsigned int>(epoch_index));
        for (std::size_t i = order_.size(); i > 1; --i)
        {
            std::uniform_int_distribution<std::size_t> dist(0, i - 1);
            const std::size_t j = dist(g);
            std::swap(order_[i - 1], order_[j]);
        }
    }

    // Advance to the next epoch and reshuffle with seed + epoch().
    void next_epoch()
    {
        ++epoch_;
        reshuffle(epoch_);
    }

    Dataset batch(std::size_t index) const
    {
        if (index >= batches_per_epoch())
            throw std::out_of_range("DataLoader: batch index out of range");

        const std::size_t start = index * batch_size_;
        const std::size_t count = std::min(batch_size_, size() - start);
        Tensor Xb({count, X_.shape[1]});
        Tensor yb({count, y_width_});

        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t src = order_[start + i];
            std::copy_n(X_.data->begin() + src * X_.shape[1], X_.shape[1],
                        Xb.data->begin() + i * X_.shape[1]);
            std::copy_n(y_.data->begin() + src * y_width_, y_width_,
                        yb.data->begin() + i * y_width_);
        }
        return {Xb, yb};
    }

    // Materialise every batch of the current epoch (convenience for tests).
    std::vector<Dataset> epoch_batches() const
    {
        std::vector<Dataset> out;
        out.reserve(batches_per_epoch());
        for (std::size_t i = 0; i < batches_per_epoch(); ++i)
            out.push_back(batch(i));
        return out;
    }

    Iterator begin() const { return Iterator(*this, 0); }
    Iterator end() const { return Iterator(*this, batches_per_epoch()); }

private:
    Tensor X_;
    Tensor y_;
    std::size_t batch_size_;
    unsigned int seed_;
    std::size_t epoch_ = 0;
    std::size_t y_width_ = 1;
    std::vector<std::size_t> order_;
};
#endif