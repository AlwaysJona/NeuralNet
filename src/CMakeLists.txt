add_library(Tensor Tensor.cpp)
target_include_directories(Tensor PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(neural main.cpp)
target_link_libraries(neural PRIVATE Tensor)
