cmake_minimum_required(VERSION 3.19)
project(OpenCV_DCP)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)
add_executable(OpenCV_DCP main.cpp guided-filter/guidedfilter.cpp)
target_link_libraries(OpenCV_DCP ${OpenCV_LIBS})