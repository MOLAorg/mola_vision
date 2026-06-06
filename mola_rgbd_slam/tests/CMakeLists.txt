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

# Integration test on a real TUM RGB-D sequence. Skips unless TUM_RAWLOG is set.
add_executable(test_tum_rgbd test_tum_rgbd.cpp)
target_link_libraries(test_tum_rgbd
  PRIVATE
    mola_rgbd_slam
    mrpt::mrpt_obs
    GTest::GTest
    GTest::Main
)
target_include_directories(test_tum_rgbd PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(test_tum_rgbd)

