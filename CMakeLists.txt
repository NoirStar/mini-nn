cmake_minimum_required(VERSION 3.20)
project(mini-nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 헤더 온리 라이브러리
add_library(mini-nn INTERFACE)
target_include_directories(mini-nn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 테스트
add_executable(test_nn test_main.cpp)
target_link_libraries(test_nn PRIVATE mini-nn)

# 예제 (나중에 추가)
# add_executable(mnist_example examples/mnist/main.cpp)
# target_link_libraries(mnist_example PRIVATE mini-nn)
