# MiniML-CPP: A Lightweight C++ Machine Learning Library

![Language](https://img.shields.io/badge/language-C++17-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Dependencies](https://img.shields.io/badge/dependencies-STL%20%7C%20Gnuplot-orange.svg)

**MiniML-CPP** Made my own machine learning library in c++ from scratch (because god intended it that way) only dependency is 
gnuplot cause I can't be bothered to make a custom plotter.

This project demonstrates how core ML concepts—tensors, backpropagation, optimizers, and layers—work under the hood without relying on heavy external frameworks.

---

## Features

* **Autograd Engine**: Reverse-mode automatic differentiation for scalar and tensor operations.
* **Tensor Operations**: Matrix multiplication, broadcasting, transposition, and element-wise math.
* **Modular API**: `Linear`, `ReLU`, `Sigmoid`, and `Sequential` layers similar to PyTorch/Keras.
* **Optimizers**: Stochastic Gradient Descent (SGD) with easy extensibility.
* **Visualization**: Real-time plotting pipeline using `gnuplot` (no heavy GUI libraries required).
* **Zero Bloat**: Uses only the C++ Standard Library (and Gnuplot for rendering).

---

## Getting Started

### Prerequisites

1.  **C++ Compiler**: `g++`  supporting **C++17** or higher.
    * *Note: The code is forward-compatible with C++20/23 as I compiled it with c++17 to c++23.*
2.  **Gnuplot**: Required for the visualization modules.
    * *Linux (Arch btw)*: `sudo pacman -S gnuplot`
    * *macOS*: `brew install gnuplot`
    * *Windows (please don't use that spyware anaymore)*: Install [Gnuplot](http://www.gnuplot.info/) and ensure `gnuplot.exe` is in your System PATH.

### Installation

Clone the repository:
```bash
git clone [https://github.com/ibraramin/MiniML-CPP.git](https://github.com/your-username/MiniML-CPP.git)
cd MiniML-CPP

```

### Building and Running

Compile the `main.cpp` file using your preferred compile (I used gcc, don't know if it'll compile in clang or msvc)r.

**Using g++:**

```bash
g++ -std=c++17 main.cpp tensor.cpp -o ml_demo

```

**Running the executable:**

```bash
# Linux/macOS
./ml_demo

# Windows
ml_demo.exe

```

---

## 📊 Results

The demo (`main.cpp`) runs through 6 standard machine learning scenarios.

### 1. Linear Regression

Fitting a straight line to noisy linear data using gradient descent.

### 2. Ridge Regression (L2 Regularization)

Demonstrates the training loss curve converging over epochs using L2 penalty.

### 3. Polynomial Regression

Fitting a non-linear quadratic curve () using a custom feature expansion.

### 4. Logistic Regression

Binary classification on linearly separable "blob" data.

### 5. Linear SVM

Max-margin classification using Hinge Loss on binary data.

### 6. MLP (Neural Network) on Non-Linear Data

Classifying a "Donut" dataset (concentric circles) using a Multi-Layer Perceptron (2 inputs -> 16 hidden -> 1 output) with ReLU activation.

---

## 💻 Code Example

Here is how to define a simple neural network using the library:

```cpp
#include "tensor.hpp"
#include "nn.hpp"
#include "models.hpp"

int main() {
    // 1. Define Data
    Tensor X = Tensor::randn({10, 2}); // 10 samples, 2 features
    Tensor y = Tensor::randn({10, 1}); // Targets

    // 2. Define Model (2 inputs -> 4 hidden -> 1 output)
    auto model = std::make_shared<Sequential>(std::initializer_list<std::shared_ptr<Module>>{
        std::make_shared<Linear>(2, 4),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(4, 1),
        std::make_shared<Sigmoid>()
    });

    // 3. Optimizer & Loss
    SGD optim(model->parameters(), 0.01f);
    MSELoss criterion;

    // 4. Training Loop
    for (int i = 0; i < 100; ++i) {
        optim.zero_grad();          // Reset gradients
        Tensor pred = model->forward(X);
        Tensor loss = criterion(pred, y);
        
        loss.backward();            // Autograd magic
        optim.step();               // Update weights
        
        std::cout << "Epoch " << i << " Loss: " << loss.data->at(0) << "\n";
    }
    return 0;
}

```

---

## 📂 Project Structure

* **`tensor.hpp/cpp`**: The core Tensor engine. Handles data storage, shapes, strides, and the DAG (Directed Acyclic Graph) for backpropagation.
* **`nn.hpp`**: Neural network modules (`Module` base class, `Linear`, `ReLU`, `SGD`, `MSELoss`).
* **`models.hpp`**: High-level estimator classes wrapping the NN modules (e.g., `LogisticRegression`, `MLPClassifier`).
* **`metrics.hpp`**: Utility for calculating accuracy.
* **`plot.hpp`**: Interface to `gnuplot` for piping commands and data for real-time visualization.
* **`main.cpp`**: The main showcase file containing data generation and demo logic.

---

## 🤝 Contributing

Contributions are welcome! If you want to add new activation functions (Tanh), optimizers (Adam), or support for reading CSV files, feel free to open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

```