#ifndef METRICS_H
#define METRICS_H
#include "tensor.hpp"

#include <cstddef>
#include <stdexcept>

// Binary classification counts. `actual` is true when the label is >= 0.5,
// `predicted` is true when the score is >= the decision threshold.
struct ConfusionMatrix
{
    std::size_t tp{0};
    std::size_t fp{0};
    std::size_t tn{0};
    std::size_t fn{0};

    std::size_t total() const { return tp + fp + tn + fn; }
};

class Metrics
{
public:
    static float accuracy(const Tensor &y_true, const Tensor &y_pred, float thresh = 0.5f)
    {
        size_t correct = 0;
        for (size_t i = 0; i < y_true.total_size; ++i)
        {
            float p = y_pred.data->at(i) >= thresh ? 1.0f : 0.0f;
            if (std::abs(p - y_true.data->at(i)) < 1e-5)
                correct++;
        }
        return (float)correct / y_true.total_size;
    }

    // Counts are derived once here; precision/recall/F1 below are pure
    // functions of these four integers.
    static ConfusionMatrix confusion(const Tensor &y_true, const Tensor &y_pred,
                                     float thresh = 0.5f)
    {
        if (y_true.total_size != y_pred.total_size || y_true.total_size == 0)
            throw std::invalid_argument("Metrics: label and score sizes must match and be non-zero");

        ConfusionMatrix cm;
        for (std::size_t i = 0; i < y_true.total_size; ++i)
        {
            const bool actual = y_true.data->at(i) >= 0.5f;
            const bool predicted = y_pred.data->at(i) >= thresh;
            if (actual && predicted)
                ++cm.tp;
            else if (!actual && predicted)
                ++cm.fp;
            else if (actual && !predicted)
                ++cm.fn;
            else
                ++cm.tn;
        }
        return cm;
    }

    // Undefined ratios (zero denominator) are reported as 0.0f rather than
    // NaN, so callers can compare and log them safely.
    static float precision(const Tensor &y_true, const Tensor &y_pred, float thresh = 0.5f)
    {
        const ConfusionMatrix cm = confusion(y_true, y_pred, thresh);
        const std::size_t denom = cm.tp + cm.fp;
        return denom == 0 ? 0.0f : static_cast<float>(cm.tp) / static_cast<float>(denom);
    }

    static float recall(const Tensor &y_true, const Tensor &y_pred, float thresh = 0.5f)
    {
        const ConfusionMatrix cm = confusion(y_true, y_pred, thresh);
        const std::size_t denom = cm.tp + cm.fn;
        return denom == 0 ? 0.0f : static_cast<float>(cm.tp) / static_cast<float>(denom);
    }

    static float f1(const Tensor &y_true, const Tensor &y_pred, float thresh = 0.5f)
    {
        const ConfusionMatrix cm = confusion(y_true, y_pred, thresh);
        const float p = (cm.tp + cm.fp == 0)
                            ? 0.0f
                            : static_cast<float>(cm.tp) / static_cast<float>(cm.tp + cm.fp);
        const float r = (cm.tp + cm.fn == 0)
                            ? 0.0f
                            : static_cast<float>(cm.tp) / static_cast<float>(cm.tp + cm.fn);
        const float denom = p + r;
        return denom == 0.0f ? 0.0f : 2.0f * p * r / denom;
    }
};
#endif
