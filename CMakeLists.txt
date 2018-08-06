# 基本设置
cmake_minimum_required(VERSION 3.5)
project(FBSR)

# 设置CUDA 静态库运行时关闭
OPTION(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# 设置C++ 版本为11
set(CMAKE_CXX_STANDARD 11)

# OpenCV 2
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})


# 构建example

set(EXAMPLE_SOURCE_FILE_LIST main.cpp SuperResolutionBase.h SuperResolutionBase.cpp)

add_executable(example_runtime ${EXAMPLE_SOURCE_FILE_LIST})
target_link_libraries(example_runtime ${OpenCV_LIBS})
