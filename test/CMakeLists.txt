enable_testing()

add_subdirectory ("googletest")

include_directories("googletest/include")

set(TEST_SRCS
  "zero_test.cpp"
)

add_executable(runUnitTest ${TEST_SRCS})

target_link_libraries(runUnitTest gtest gtest_main)
target_link_libraries(runUnitTest tensor)
