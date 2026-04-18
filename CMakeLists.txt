cmake_minimum_required(VERSION 3.14) # version 3.14 required for FetchContent
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG        v2.4.11
)
FetchContent_MakeAvailable(doctest)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
