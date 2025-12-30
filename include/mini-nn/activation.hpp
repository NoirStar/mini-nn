#pragma once
#include "matrix.hpp"
#include <concepts>

namespace nn {

// Concept 정의
template<typename T>
concept Activation = requires(Matrix m) {
    { T::apply(m) } -> std::same_as<Matrix>;
};

struct ReLU {
    static Matrix apply(const Matrix& x) {
        return relu(x);
    }
};

struct Sigmoid {
    static Matrix apply(const Matrix& x) {
        return sigmoid(x);
    }
};

struct Softmax {
    static Matrix apply(const Matrix& x) {
        return softmax(x);
    }
};


} // namespace nn