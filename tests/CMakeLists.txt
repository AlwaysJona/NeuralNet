# Creates an executable target called "tests" compiled from tests.cpp.
# This is the binary that runs all your unit tests.
add_executable(tests tests.cpp)

# Links two libraries into the tests executable:
# - Tensor: so tests can use your Tensor code
# - doctest::doctest: the testing framework itself
# PRIVATE means these dependencies are only needed to build and run tests,
# they won't be exposed to anything else.
target_link_libraries(tests PRIVATE NeuralNetworkLib doctest::doctest)

# Loads a helper script that comes bundled with doctest.
# This script provides the doctest_discover_tests() function used below.
# ${doctest_SOURCE_DIR} is automatically set by FetchContent to wherever
# doctest was downloaded.
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

# Scans the tests executable for all TEST_CASE definitions and registers
# each one individually with CTest. This means `ctest` can run, filter,
# and report on each test case separately.
doctest_discover_tests(tests)
