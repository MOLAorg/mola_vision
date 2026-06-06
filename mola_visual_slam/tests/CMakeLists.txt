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

# Integration test on a real TUM RGB-D sequence (monocular). Skips at runtime
# unless TUM_RAWLOG is set; needs mrpt_obs (CRawlog) directly.
add_executable(test_tum_mono test_tum_mono.cpp)
target_link_libraries(test_tum_mono
  PRIVATE
    mola_visual_slam
    mrpt::mrpt_obs
    GTest::GTest
    GTest::Main
)
target_include_directories(test_tum_mono PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(test_tum_mono)

# Integration test on a real KITTI odometry stereo sequence. Skips at runtime
# unless KITTI_STEREO_DIR / KITTI_POSES are set.
add_executable(test_kitti_stereo test_kitti_stereo.cpp)
target_link_libraries(test_kitti_stereo
  PRIVATE
    mola_visual_slam
    GTest::GTest
    GTest::Main
)
target_include_directories(test_kitti_stereo PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(test_kitti_stereo)


