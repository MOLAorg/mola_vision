find_package(GTest REQUIRED)
include(GoogleTest)

# Helper macro matching the MOLA pattern
macro(mola_vision_add_test NAME)
  cmake_parse_arguments(_ARG "" "" "SOURCES;LINK_LIBRARIES" ${ARGN})
  add_executable(${NAME} ${_ARG_SOURCES})
  target_link_libraries(${NAME}
    PRIVATE
      mola_libvision
      GTest::GTest
      GTest::Main
      ${_ARG_LINK_LIBRARIES}
  )
  target_include_directories(${NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )
  gtest_discover_tests(${NAME})
endmacro()

# ----------------------------------------------------------------------------
# Test: image_utils (Sobel + Gaussian blur + Eigen::Map)
# ----------------------------------------------------------------------------
mola_vision_add_test(test_image_utils
  SOURCES test_image_utils.cpp
)

# ----------------------------------------------------------------------------
# Test: feature detection (goodFeaturesToTrack + GridFeatureDistributor)
# ----------------------------------------------------------------------------
mola_vision_add_test(test_feature_detection
  SOURCES test_feature_detection.cpp
)

# ----------------------------------------------------------------------------
# Test: optical flow (LK tracker + F-matrix filter)
# ----------------------------------------------------------------------------
mola_vision_add_test(test_optical_flow
  SOURCES test_optical_flow.cpp
)

# ----------------------------------------------------------------------------
# Test: geometry (triangulation + undistort + essential matrix)
# ----------------------------------------------------------------------------
mola_vision_add_test(test_geometry
  SOURCES test_geometry.cpp
)

# ----------------------------------------------------------------------------
# Test: LM solver
# ----------------------------------------------------------------------------
mola_vision_add_test(test_lm_solver
  SOURCES test_lm_solver.cpp
)

# ----------------------------------------------------------------------------
# Test: MapPoint
# ----------------------------------------------------------------------------
mola_vision_add_test(test_mappoint
  SOURCES test_mappoint.cpp
)

# ----------------------------------------------------------------------------
# Test: PnP solver
# ----------------------------------------------------------------------------
mola_vision_add_test(test_pnp
  SOURCES test_pnp.cpp
)

# ----------------------------------------------------------------------------
# Test: stereo matcher
# ----------------------------------------------------------------------------
mola_vision_add_test(test_stereo
  SOURCES test_stereo.cpp
)

# ----------------------------------------------------------------------------
# Test: sliding-window bundle adjustment
# ----------------------------------------------------------------------------
mola_vision_add_test(test_ba
  SOURCES test_ba.cpp
)

# ----------------------------------------------------------------------------
# Test: RGBD depth processing
# ----------------------------------------------------------------------------
mola_vision_add_test(test_rgbd
  SOURCES test_rgbd.cpp
)

# ----------------------------------------------------------------------------
# Test: keyframe database (covisibility) + keyframe selector
# ----------------------------------------------------------------------------
mola_vision_add_test(test_keyframe_db
  SOURCES test_keyframe_db.cpp
)

# ----------------------------------------------------------------------------
# Test: IMU preintegration
# ----------------------------------------------------------------------------
mola_vision_add_test(test_imu
  SOURCES test_imu.cpp
)

# ----------------------------------------------------------------------------
# Test: gravity estimator + place-recognition / pose-graph interfaces
# ----------------------------------------------------------------------------
mola_vision_add_test(test_gravity_and_interfaces
  SOURCES test_gravity_and_interfaces.cpp
)

# ----------------------------------------------------------------------------
# Test: marginalization prior (Schur complement)
# ----------------------------------------------------------------------------
mola_vision_add_test(test_marg_prior
  SOURCES test_marg_prior.cpp
)
