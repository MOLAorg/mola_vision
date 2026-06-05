find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_rgbd_slam test_rgbd_slam.cpp)
target_link_libraries(test_rgbd_slam
  PRIVATE
    mola_rgbd_slam
    GTest::GTest
    GTest::Main
)
target_include_directories(test_rgbd_slam PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(test_rgbd_slam)
