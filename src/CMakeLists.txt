# Creates a library target called "Tensor" compiled from Tensor.cpp.
# A library is a bundle of compiled code that other targets (like executables) can link against.
add_library(NeuralNetworkLib
    Tensor.cpp
    Module.cpp
    Linear.cpp
    Relu.cpp
    Flatten.cpp
    Serialization.cpp
    Dataloader.cpp
    Dataset.cpp
    Loss.cpp
    SGD.cpp
    Softmax.cpp
    Neural.cpp
)

# Tells CMake where to find the header files (.h) for the Tensor library.
# PUBLIC means this include path is shared with any target that links against Tensor,
# so they can also #include "Tensor.h" without specifying the path themselves.
# ${PROJECT_SOURCE_DIR} is the root of the project (where the root CMakeLists.txt lives).
target_include_directories(NeuralNetworkLib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Creates an executable target called "neural" compiled from main.cpp.
# This is the binary you actually run.
add_executable(neural main.cpp)

# Links the Tensor library into the neural executable so it can use its code.
# PRIVATE means this dependency is only for neural — it won't be propagated
# to anything that depends on neural.
target_link_libraries(neural PRIVATE NeuralNetworkLib)
