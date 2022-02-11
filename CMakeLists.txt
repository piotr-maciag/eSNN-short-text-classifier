cmake_minimum_required(VERSION 3.17)
project(Classifier)

set(CMAKE_CXX_STANDARD 14)

add_executable(Classifier main.cpp eSNN.h eSNN.cpp LoadData.cpp LoadData.h)