cmake_minimum_required(VERSION 3.23)
project(MLP)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-4")
add_executable(MLP main.cpp classes/layer.cpp classes/layer.h classes/mlp.cpp classes/mlp.h classes/utils.h classes/utils.cpp)
