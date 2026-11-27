// Metrics checks against hand-computed fixtures. Every expected number below
// was derived by classifying the fixture rows by hand and applying the
// definitions directly; none is produced by calling the code under test.

#include "metrics.hpp"

#include <cmath>
#include <iostream>

namespace {

bool expect(bool condition, const char *message)
{
    if (!condition)
        std::cerr << "FAIL " << message << "\n";
    return condition;
}

bool close(float a, float b) { return std::fabs(a - b) < 1e-6f; }

} // namespace

int main()
{
    bool ok = true;

    // Hand-classified fixture (10 rows, threshold 0.5):
    //   TP: rows 1,3,5  (label 1, scores 0.9, 0.6, 0.5)  -> 3
    //   FP: rows 2,7    (label 0, scores 0.8, 0.5)      -> 2
    //   TN: rows 4,6,9,10 (label 0, scores 0.1, 0.49, 0.0, -0.2) -> 4
    //   FN: row 8       (label 1, score 0.4)            -> 1
    // Therefore, by hand:
    //   precision = 3/(3+2) = 0.6
    //   recall    = 3/(3+1) = 0.75
    //   F1        = 2*3/(2*3+2+1) = 6/9 = 0.666666666...
    //   accuracy  = (3+4)/10 = 0.7
    const float labels[10] = {1, 0, 1, 0, 1, 0, 0, 1, 0, 0};
    const float scores[10] = {0.9f, 0.8f, 0.6f, 0.1f, 0.5f,
                              0.49f, 0.5f, 0.4f, 0.0f, -0.2f};
    Tensor yt({10}), yp({10});
    for (std::size_t i = 0; i < 10; ++i)
    {
        yt.data->at(i) = labels[i];
        yp.data->at(i) = scores[i];
    }

    const ConfusionMatrix cm = Metrics::confusion(yt, yp);
    ok = expect(cm.tp == 3 && cm.fp == 2 && cm.tn == 4 && cm.fn == 1,
                "confusion matrix = (tp=3, fp=2, tn=4, fn=1)") && ok;
    ok = expect(cm.total() == 10, "confusion matrix total = 10") && ok;
    ok = expect(close(Metrics::precision(yt, yp), 0.6f), "precision = 3/5 = 0.6") && ok;
    ok = expect(close(Metrics::recall(yt, yp), 0.75f), "recall = 3/4 = 0.75") && ok;
    ok = expect(close(Metrics::f1(yt, yp), 2.0f / 3.0f), "F1 = 6/9 = 2/3") && ok;
    ok = expect(close(Metrics::accuracy(yt, yp), 0.7f), "accuracy = 7/10 = 0.7") && ok;

    // Same fixture with threshold 0.8, re-derived by hand over all 10 rows:
    //   TP: row 1 (label 1, score 0.9 >= 0.8) -> 1
    //   FP: row 2 (label 0, score 0.8 >= 0.8) -> 1
    //   FN: rows 3,5,8 (label 1, scores 0.6, 0.5, 0.4 < 0.8) -> 3
    //   TN: rows 4,6,7,9,10 (label 0, scores 0.1, 0.49, 0.5, 0.0, -0.2 < 0.8) -> 5
    //   Row 7 (label 0, score 0.5) is a TN here; it was an FP only at 0.5.
    //   Total = 1 + 1 + 5 + 3 = 10, matching cm.total().
    //   precision = 1/2 = 0.5; recall = 1/4 = 0.25; F1 = 2/(2+1+3) = 1/3
    const ConfusionMatrix cm8 = Metrics::confusion(yt, yp, 0.8f);
    ok = expect(cm8.tp == 1 && cm8.fp == 1 && cm8.tn == 5 && cm8.fn == 3,
                "threshold 0.8 confusion = (1,1,5,3)") && ok;
    ok = expect(close(Metrics::precision(yt, yp, 0.8f), 0.5f), "threshold 0.8 precision = 0.5") && ok;
    ok = expect(close(Metrics::recall(yt, yp, 0.8f), 0.25f), "threshold 0.8 recall = 0.25") && ok;
    ok = expect(close(Metrics::f1(yt, yp, 0.8f), 1.0f / 3.0f), "threshold 0.8 F1 = 1/3") && ok;

    // No positive predictions: tp=0, fp=0, fn=3 -> all ratios defined as 0.
    Tensor nt({3}), np({3});
    for (std::size_t i = 0; i < 3; ++i)
    {
        nt.data->at(i) = 1.0f;
        np.data->at(i) = 0.1f + 0.05f * static_cast<float>(i);
    }
    const ConfusionMatrix cn = Metrics::confusion(nt, np);
    ok = expect(cn.tp == 0 && cn.fp == 0 && cn.tn == 0 && cn.fn == 3,
                "all-negative fixture confusion = (0,0,0,3)") && ok;
    ok = expect(close(Metrics::precision(nt, np), 0.0f), "undefined precision -> 0") && ok;
    ok = expect(close(Metrics::recall(nt, np), 0.0f), "zero recall -> 0") && ok;
    ok = expect(close(Metrics::f1(nt, np), 0.0f), "zero F1 -> 0") && ok;

    // Perfect classifier: tp=2, tn=2 -> precision = recall = F1 = 1.
    Tensor pt({4}), pp({4});
    const float pl[4] = {0, 1, 0, 1};
    const float ps[4] = {0.2f, 0.9f, 0.1f, 0.7f};
    for (std::size_t i = 0; i < 4; ++i)
    {
        pt.data->at(i) = pl[i];
        pp.data->at(i) = ps[i];
    }
    const ConfusionMatrix cp = Metrics::confusion(pt, pp);
    ok = expect(cp.tp == 2 && cp.fp == 0 && cp.tn == 2 && cp.fn == 0,
                "perfect fixture confusion = (2,0,2,0)") && ok;
    ok = expect(close(Metrics::precision(pt, pp), 1.0f), "perfect precision = 1") && ok;
    ok = expect(close(Metrics::recall(pt, pp), 1.0f), "perfect recall = 1") && ok;
    ok = expect(close(Metrics::f1(pt, pp), 1.0f), "perfect F1 = 1") && ok;

    if (ok)
        std::cout << "PASS metrics\n";
    else
        std::cerr << "FAIL metrics\n";
    return ok ? 0 : 1;
}
