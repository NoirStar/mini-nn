# mini-nn

모던 C++ 템플릿 학습을 위한 미니 Neural Network

## 목적

- **variadic templates** 심화 학습
- **fold expressions** 실전 활용
- **concepts** 타입 제약
- 컴파일 타임 **타입 안전성** 설계

## 구현 완료

```cpp
// 이렇게 쓸 수 있어요!
using MyNetwork = Network<
    Layer<784, 128, ReLU>,    // 입력층
    Layer<128, 64, ReLU>,     // 은닉층
    Layer<64, 10, Softmax>    // 출력층
>;

MyNetwork net;
Matrix output = net.forward(input);
```

## 배운 것

| 개념 | 예시 | 사용처 |
|------|------|--------|
| **variadic templates** | `typename... Layers` | Network 다중 Layer |
| **fold expression** | `(result = layers.forward(result), ...)` | 순차 실행 |
| **pack expansion** | `Rest...` | 펼쳐서 전달 |
| **부분 특수화** | `is_valid_chain<First, Second, Rest...>` | 재귀 패턴 매칭 |
| **concepts** | `Activation` | 타입 제약 |
| **static_assert** | `is_valid_chain<Layers...>` | 컴파일 타임 검증 |

## 프로젝트 구조

```
mini-nn/
├── include/mini-nn/
│   ├── matrix.hpp      # 행렬 연산
│   ├── activation.hpp  # ReLU, Sigmoid, Softmax + Concept
│   ├── layer.hpp       # Layer<In, Out, Activation>
│   └── network.hpp     # Network<Layers...> + 연결 검증
├── test_main.cpp
└── CMakeLists.txt
```

## 빌드

```bash
mkdir build && cd build
cmake ..
cmake --build .
```
