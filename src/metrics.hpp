#ifndef METRICS_H
#define METRICS_H
#include "tensor.hpp"

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
};
#endif