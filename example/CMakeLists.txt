cmake_minimum_required(VERSION 2.8)

# OpenCV
find_library(OpenCV REQUIRD)
include_directories(${OpenCV_INCLUDE_DIR})

add_executable(main main.cpp)