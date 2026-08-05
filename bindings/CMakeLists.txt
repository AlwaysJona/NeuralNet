cmake_minimum_required(VERSION 3.15)

project(neural_bindings)

# Download pybind11 automatically
include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.4
)

FetchContent_MakeAvailable(pybind11)


# Create Python module
pybind11_add_module(neuralnet bindings.cpp)


# Link against the NeuralNet library
target_link_libraries(neuralnet PRIVATE NeuralNetworkLib)


# Include headers
target_include_directories(neuralnet PRIVATE
    ${PROJECT_SOURCE_DIR}/../include
)


# Use C++20
target_compile_features(neuralnet PRIVATE cxx_std_20)
