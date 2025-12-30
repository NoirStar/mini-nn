#include <iostream>
#include <mini-nn/matrix.hpp>
#include <mini-nn/layer.hpp>

int main() {
    using namespace nn;
    
    std::cout << "=== Layer Test ===\n\n";
    
    // Layer<3, 2>: 3개 입력 → 2개 출력
    Layer<3, 2> layer;
    
    // 컴파일 타임 상수 확인
    std::cout << "Input size:  " << layer.input_size << "\n";
    std::cout << "Output size: " << layer.output_size << "\n\n";
    
    // 입력 데이터 (1개 샘플, 3개 특성)
    Matrix input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    
    std::cout << "Input [1x3]:\n";
    input.print();
    
    // forward
    Matrix output = layer.forward(input);
    
    std::cout << "\nOutput [1x2]:\n";
    output.print();
    
    std::cout << "\n✅ Layer test passed!\n";
    return 0;
}
