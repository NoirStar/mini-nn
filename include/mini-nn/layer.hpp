#pragma once

#include "matrix.hpp"
#include "activation.hpp"

namespace nn {

// ================================================
// Layer<InputSize, OutputSize>
// 
// 템플릿 파라미터:
//   - InputSize: 입력 뉴런 개수 (예: 784)
//   - OutputSize: 출력 뉴런 개수 (예: 128)
//
// 내부:
//   - weights_: [InputSize x OutputSize] 행렬
//   - bias_: [1 x OutputSize] 벡터
// ================================================

template<size_t InputSize, size_t OutputSize, Activation Act = ReLU>
class Layer {
public:
    // 컴파일 타임 상수로 접근 가능!
    static constexpr size_t input_size = InputSize;
    static constexpr size_t output_size = OutputSize;

    Layer() : weights_(InputSize, OutputSize), bias_(1, OutputSize, 0.0f) {
        weights_.randomize();
    }

    // forward: 입력 → 출력 계산
    // input: [batch_size x InputSize]
    // return: [batch_size x OutputSize]
    Matrix forward(const Matrix& input) {
        // 1. 행렬곱: input × weights
        // 2. bias 더하기
        lastInput_ = input;
        lastOutput_ = Act::apply(input.dot(weights_).addBias(bias_));
        return lastOutput_;
    }

    // Getter (나중에 학습할 때 사용)
    Matrix& weights() { return weights_; }
    Matrix& bias() { return bias_; }
    const Matrix& lastInput() const { return lastInput_; }
    const Matrix& lastOutput() const { return lastOutput_; }

private:
    Matrix weights_;
    Matrix bias_;
    Matrix lastInput_;   // 역전파용 (나중에)
    Matrix lastOutput_;  // 역전파용 (나중에)
};

} // namespace nn
