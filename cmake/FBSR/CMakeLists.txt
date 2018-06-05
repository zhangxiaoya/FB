# 基本设置
cmake_minimum_required(VERSION 2.8)
project(FBSR)

# 设置CUDA 静态库运行时关闭
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# 设置C++ 版本为11
set(CMAKE_CXX_STANDARD 11)

# OpenCV 2
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

# 设置库目录
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# 设置包含文件目录
include_directories(./include)

set(SOURCE_FILE_LIST src/LKOFlow/LKOFlow.cpp
        src/SuperResolution/FrameBuffer/FrameBuffer.cpp
        src/SuperResolution/SuperResolutionBase.cpp
        src/Utils/Utils.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILE_LIST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# 构建example
OPTION(BUILD_EXAMPLE ON)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/example)

set(EXAMPLE_SOURCE_FILE_LIST example/main.cpp example/MultiFrameProcesser.hpp
        example/FrameSource/ImageListFrameSource.hpp
        example/FrameSource/VideoFrameSource.hpp
        example/FrameSource/CaptureFrameSource.hpp
        example/FrameSource/EmptyFrameSource.hpp
        example/FrameSource/FrameSourceFactory.h
        example/FrameSource/ImageListFrameSource.hpp
        example/FrameSource/VideoFrameSource.hpp
        example/FrameSource/ImageListReader/ImageListReader.hpp
        example/FrameSource/ImageListReader/ImageListReaderBase.hpp)

add_executable(example ${EXAMPLE_SOURCE_FILE_LIST})
target_link_libraries(example ${OpenCV_LIBS} ${PROJECT_BINARY_DIR}/lib)