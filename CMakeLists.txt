cmake_minimum_required(VERSION 3.30)
project(i20test)
set(CMAKE_CXX_STANDARD 20)

# 设置 OpenCV 路径（Mac上的路径）
set(OpenCV_DIR /opt/homebrew/opt/opencv)

include_directories(${OpenCV_DIR}/include)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置 OpenCV 库路径
link_directories(${OpenCV_DIR}/lib)
add_executable(i20test main.cpp
        bin2mat.cpp
        breath.cpp
        circle_buf.h
        debayer.cpp
        eye.cpp
        pulse.cpp
        read_file.cpp
        utils.cpp
        findpeaks.cpp)
# linking
target_link_libraries(i20test ${OpenCV_LIBS})