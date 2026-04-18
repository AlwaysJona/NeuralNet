add_executable(tests tests.cpp)
target_link_libraries(tests PRIVATE Tensor doctest::doctest)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(tests)
