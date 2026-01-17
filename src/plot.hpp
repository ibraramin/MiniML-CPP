#ifndef PLOT_H
#define PLOT_H

#include <iostream>
#include <vector>
#include <cstdio>
#include <string>
#include <algorithm>
#include "tensor.hpp"

class Plotter
{
    FILE *pipe;

public:
    Plotter()
    {
#ifdef _WIN32
        pipe = _popen("gnuplot -persist", "w");
#else
        pipe = popen("gnuplot -persist", "w");
#endif
        if (!pipe)
            std::cerr << "Warning: Gnuplot not found.\n";
    }

    ~Plotter()
    {
        if (pipe)
        {
#ifdef _WIN32
            _pclose(pipe);
#else
            pclose(pipe);
#endif
        }
    }

    void send(const std::string &cmd)
    {
        if (pipe)
        {
            fprintf(pipe, "%s\n", cmd.c_str());
            fflush(pipe);
        }
    }

    void plot_loss(const std::vector<float> &hist, std::string title = "Training Loss")
    {
        if (!pipe)
            return;
        fprintf(pipe, "set title '%s'\n", title.c_str());
        send("set xlabel 'Epoch'");
        send("set ylabel 'Loss'");
        send("set grid");
        send("plot '-' with lines lw 2 title 'Loss'");
        for (size_t i = 0; i < hist.size(); ++i)
            fprintf(pipe, "%zu %f\n", i, hist[i]);
        send("e");
    }

    void plot_regression(const Tensor &X, const Tensor &y_true, const Tensor &y_pred, std::string title = "Regression Fit")
    {
        if (!pipe)
            return;
        fprintf(pipe, "set title '%s'\n", title.c_str());
        send("set xlabel 'Input Feature (X)'");
        send("set ylabel 'Target (Y)'");
        send("set grid");

        std::string cmd = "plot '-' using 1:2 with points pt 7 ps 0.5 lc rgb 'blue' title 'Actual Data', ";
        cmd += "'-' using 1:2 with lines lw 3 lc rgb 'red' title 'Model Prediction'";
        send(cmd);

        size_t N = X.shape[0];
        float *xd = X.data->data();
        float *yd = y_true.data->data();
        for (size_t i = 0; i < N; ++i)
            fprintf(pipe, "%f %f\n", xd[i], yd[i]);
        send("e");

        float *pd = y_pred.data->data();
        for (size_t i = 0; i < N; ++i)
            fprintf(pipe, "%f %f\n", xd[i], pd[i]);
        send("e");
    }

    void plot_boundary(const Tensor &X, const Tensor &y, const Tensor &pred, std::string title = "Decision Boundary")
    {
        if (!pipe)
            return;
        fprintf(pipe, "set title '%s'\n", title.c_str());
        send("set key right bottom");

        std::string cmd = "plot '-' using 1:2 title 'Pred 0' with points pt 7 ps 0.8 lc rgb '#4477AA', ";
        cmd += "'-' using 1:2 title 'Pred 1' with points pt 7 ps 0.8 lc rgb '#EE6677'";
        send(cmd);

        size_t N = X.shape[0];
        float *xd = X.data->data();
        float *pd = pred.data->data();

        for (size_t i = 0; i < N; ++i)
            if (pd[i] < 0.5f)
                fprintf(pipe, "%f %f\n", xd[i * 2], xd[i * 2 + 1]);
        send("e");

        for (size_t i = 0; i < N; ++i)
            if (pd[i] >= 0.5f)
                fprintf(pipe, "%f %f\n", xd[i * 2], xd[i * 2 + 1]);
        send("e");
    }
};
#endif