# Minimum version of CMake required to build this project. 3.14 is needed for
# FetchContent_MakeAvailable, which we use to download doctest.
cmake_minimum_required(VERSION 3.14)

# Names the project. This also sets the variable ${PROJECT_NAME} to
# "NeuralNetwork" and ${PROJECT_SOURCE_DIR} to the directory this file lives in.
project(NeuralNetwork)

# Tells CMake to compile all targets using the C++17 standard.
set(CMAKE_CXX_STANDARD 17)

# Loads CMake's FetchContent module, which lets us download external libraries
# from the internet at configure time.
include(FetchContent)

# Declares a dependency called "doctest" and tells CMake where to get it.
# GIT_REPOSITORY is the URL of the library's git repo. GIT_TAG pins the download
# to a specific version so your build is reproducible.
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11)

# Actually downloads doctest and makes it available to the rest of the project.
# After this line, targets like doctest::doctest can be referenced.
FetchContent_MakeAvailable(doctest)

# Enables the CTest testing framework, which lets you run all tests with the
# `ctest` command from the build directory.
enable_testing()

# Tells CMake to go into the src/ directory and process the CMakeLists.txt
# there.
add_subdirectory(src)

# Tells CMake to go into the tests/ directory and process the CMakeLists.txt
# there.
add_subdirectory(tests)

file(COPY ${PROJECT_SOURCE_DIR}/data
     DESTINATION ${PROJECT_BINARY_DIR})

file(COPY ${PROJECT_SOURCE_DIR}/models
     DESTINATION ${PROJECT_BINARY_DIR})
