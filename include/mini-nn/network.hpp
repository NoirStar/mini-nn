#pragma once

#include "layer.hpp"
#include "matrix.hpp"
#include <tuple>

namespace nn {

template<typename...>
constexpr bool is_valid_chain = true;

template<typename First, typename Second, typename... Rest>
constexpr bool is_valid_chain<First, Second, Rest...> = 
    (First::output_size == Second::input_size) &&
    is_valid_chain<Second, Rest...>;


template<typename... Layers>
class Network {
public:
    static_assert(is_valid_chain<Layers...>, "Layer sizes don't match!");

    // Layer들에 대한 output을 새로운 input으로 계속 사용, 체이닝 해야함. 
    Matrix forward(const Matrix& input) {
        return std::apply([&input] (Layers&... layers) {
            Matrix result = input;
            (result = layers.forward(result), ...); // 콤마연산자 + fold expression
            return result;
        }, layers_);
    }

private:
    std::tuple<Layers...> layers_;
};

}