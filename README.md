# mini-nn

모던 C++ 템플릿 학습을 위한 미니 Neural Network

## 목적

- **variadic templates** 심화 학습
- **fold expressions** 실전 활용
- **CRTP** 패턴 이해
- 컴파일 타임 **타입 안전성** 설계

## 목표

```cpp
// 이런 코드를 만들 거예요!
using MyNetwork = Network<
    Layer<784, 128, ReLU>,    // 입력층 (28x28 이미지)
    Layer<128, 64, ReLU>,     // 은닉층
    Layer<64, 10, Softmax>    // 출력층 (숫자 0~9)
>;

MyNetwork net;
net.train(data, labels, epochs);
int digit = net.predict(image);
```

## 학습 포인트

| 단계 | 주제 | C++ 기능 |
|------|------|----------|
| 1 | Matrix 클래스 | 기본 구현 |
| 2 | Layer 템플릿 | CRTP, 활성화 함수 |
| 3 | Network 템플릿 | variadic, fold expressions |
| 4 | 타입 검증 | static_assert, concepts |
| 5 | 학습 & 시각화 | 실제 MNIST 테스트 |

## 빌드

```bash
mkdir build && cd build
cmake ..
cmake --build .
```
