enable_testing()

# SEE https://github.com/google/googletest/blob/master/googletest/README.md#visual-studio-dynamic-vs-static-runtimes
set(gtest_force_shared_crt ON)

add_subdirectory ("googletest")

include_directories("googletest/include")

set(TEST_SRCS
  "zero_test.cpp"
)

add_executable(runUnitTest ${TEST_SRCS})

target_link_libraries(runUnitTest gtest gtest_main)
target_link_libraries(runUnitTest tensor)
