#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace nn {

class Matrix {
public:
    // 생성자
    Matrix() : rows_(0), cols_(0) {}
    
    Matrix(size_t rows, size_t cols, float value = 0.0f)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}
    
    // 크기 조회
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    // 요소 접근
    float& operator()(size_t r, size_t c) { return data_[r * cols_ + c]; }
    float operator()(size_t r, size_t c) const { return data_[r * cols_ + c]; }
    
    // 행렬 덧셈
    Matrix operator+(const Matrix& other) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }
    
    // 행렬 뺄셈
    Matrix operator-(const Matrix& other) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }
    
    // 스칼라 곱
    Matrix operator*(float scalar) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }
    
    // 행렬 곱셈 (A * B)
    Matrix dot(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        Matrix result(rows_, other.cols_, 0.0f);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                for (size_t k = 0; k < cols_; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }
    
    // 요소별 곱셈 (Hadamard product)
    Matrix hadamard(const Matrix& other) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        return result;
    }
    
    // 전치 행렬
    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    // 랜덤 초기화 (Xavier/He 초기화)
    void randomize(float scale = 1.0f) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        float stddev = scale * std::sqrt(2.0f / static_cast<float>(rows_));
        std::normal_distribution<float> dist(0.0f, stddev);
        for (auto& val : data_) {
            val = dist(gen);
        }
    }
    
    // 각 열에 같은 벡터 더하기 (bias 추가용)
    Matrix addBias(const Matrix& bias) const {
        Matrix result = *this;
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) += bias(0, j);
            }
        }
        return result;
    }
    
    // 최대값 인덱스 (예측용)
    size_t argmax() const {
        size_t maxIdx = 0;
        float maxVal = data_[0];
        for (size_t i = 1; i < data_.size(); ++i) {
            if (data_[i] > maxVal) {
                maxVal = data_[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    // 출력
    void print() const {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
    }

private:
    size_t rows_, cols_;
    std::vector<float> data_;
};

// ============================================
// 활성화 함수들 (여기 아래는 그냥 가져다 쓰면 됨)
// ============================================

// ReLU: 음수 → 0, 양수 → 그대로
inline Matrix relu(const Matrix& x) {
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            result(i, j) = std::max(0.0f, x(i, j));
        }
    }
    return result;
}

// ReLU 미분: 0보다 크면 1, 아니면 0
inline Matrix reluDerivative(const Matrix& x) {
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            result(i, j) = x(i, j) > 0 ? 1.0f : 0.0f;
        }
    }
    return result;
}

// Softmax: 확률 분포로 변환 (각 행별로)
inline Matrix softmax(const Matrix& x) {
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        float maxVal = x(i, 0);
        for (size_t j = 1; j < x.cols(); ++j) {
            maxVal = std::max(maxVal, x(i, j));
        }
        float sum = 0.0f;
        for (size_t j = 0; j < x.cols(); ++j) {
            result(i, j) = std::exp(x(i, j) - maxVal);
            sum += result(i, j);
        }
        for (size_t j = 0; j < x.cols(); ++j) {
            result(i, j) /= sum;
        }
    }
    return result;
}

// Sigmoid: 0~1 사이로 압축
inline Matrix sigmoid(const Matrix& x) {
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            result(i, j) = 1.0f / (1.0f + std::exp(-x(i, j)));
        }
    }
    return result;
}

// Sigmoid 미분
inline Matrix sigmoidDerivative(const Matrix& x) {
    Matrix s = sigmoid(x);
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            result(i, j) = s(i, j) * (1.0f - s(i, j));
        }
    }
    return result;
}

} // namespace nn
