cmake_minimum_required(VERSION 3.17)
project(eSNN-short-text-classifier)

set(CMAKE_CXX_STANDARD 14)

add_executable(eSNN-short-text-classifier main.cpp eSNN.h eSNN.cpp LoadData.cpp LoadData.h)