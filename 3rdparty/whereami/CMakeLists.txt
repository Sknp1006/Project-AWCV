cmake_minimum_required(VERSION 3.15)
project(whereami LANGUAGES CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(whereami STATIC whereami.c)
