find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_visual_slam test_visual_slam.cpp)
target_link_libraries(test_visual_slam
  PRIVATE
    mola_visual_slam
    GTest::GTest
    GTest::Main
)
target_include_directories(test_visual_slam PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(test_visual_slam)
