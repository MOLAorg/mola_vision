# ------------------------------------------------------------------------------
#        A Modular Optimization framework for Localization and mApping
#                               (MOLA)
#
# Copyright (C) 2018-2026, Jose Luis Blanco-Claraco, contributors (AUTHORS.md)
# All rights reserved.
# Released under the GNU GPL v3 license. See LICENSE file
# ------------------------------------------------------------------------------

# Minimum CMake version: limited by CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
cmake_minimum_required(VERSION 3.5)

# Tell CMake we'll use C++ for use in its tests/flags
project(mola_libvision LANGUAGES CXX)

# MOLA CMake scripts: "mola_xxx()"
find_package(mola_common REQUIRED)

# Dependencies: MRPT (v3) + Eigen3 + TBB.  NO OpenCV, NO Ceres.
# MRPT 3.x uses fine-grained packages "mrpt_<module>" and targets "mrpt::mrpt_<module>".
find_package(mrpt_img   REQUIRED)
find_package(mrpt_poses REQUIRED)
find_package(GTSAM REQUIRED)  # pose-graph optimizer with robust kernels (Phase 5)
find_package(TBB        REQUIRED)
find_package(Eigen3     REQUIRED)  # usually pulled in via mrpt_img already

# -----------------------
# Define library:
set(LIB_SRCS
  src/image_utils.cpp
  src/feature_detection.cpp
  src/optical_flow.cpp
  src/geometry.cpp
  src/MapPoint.cpp
  src/pnp_solver.cpp
  src/stereo_matcher.cpp
  src/sliding_window_ba.cpp
  src/rgbd_depth.cpp
  src/keyframe_database.cpp
  src/imu_preintegration.cpp
  src/gravity_estimator.cpp
  src/marg_prior.cpp
  src/gtsam_pose_graph_optimizer.cpp
  # keyframe_selector.h is header-only (policy)
  # lm_solver.h is header-only (template)
  # Keyframe.h  is header-only (POD struct)
  # ORB descriptor + loop detector (place recognition):
  src/orb_detector.cpp
  src/orb_loop_detector.cpp
  src/loop_closure.cpp
)

set(LIB_PUBLIC_HDRS
  include/mola_libvision/image_utils.h
  include/mola_libvision/feature_detection.h
  include/mola_libvision/optical_flow.h
  include/mola_libvision/geometry.h
  include/mola_libvision/MapPoint.h
  include/mola_libvision/Keyframe.h
  include/mola_libvision/lm_solver.h
  include/mola_libvision/pnp_solver.h
  include/mola_libvision/stereo_matcher.h
  include/mola_libvision/sliding_window_ba.h
  include/mola_libvision/lie_utils.h
  include/mola_libvision/rgbd_depth.h
  include/mola_libvision/keyframe_database.h
  include/mola_libvision/keyframe_selector.h
  include/mola_libvision/imu_preintegration.h
  include/mola_libvision/gravity_estimator.h
  include/mola_libvision/loop_detector.h
  include/mola_libvision/pose_graph_optimizer.h
  include/mola_libvision/gtsam_pose_graph_optimizer.h
  include/mola_libvision/marg_prior.h
  include/mola_libvision/orb_descriptor.h
  include/mola_libvision/orb_loop_detector.h
  include/mola_libvision/loop_closure.h
)

mola_add_library(
  TARGET ${PROJECT_NAME}
  SOURCES ${LIB_SRCS} ${LIB_PUBLIC_HDRS}
  PUBLIC_LINK_LIBRARIES
    mrpt::mrpt_img
    mrpt::mrpt_poses
    TBB::tbb
    Eigen3::Eigen
  CMAKE_DEPENDENCIES
    mola_common
    mrpt_img
    mrpt_poses
)

# GTSAM is an IMPLEMENTATION detail (GtsamPoseGraphOptimizer is pimpl, no GTSAM
# symbols in public headers), so link it PRIVATE: consumers of mola_libvision do
# not need to find_package(GTSAM).
target_link_libraries(${PROJECT_NAME} PRIVATE gtsam)

# C++17 required (for std::optional, structured bindings, etc.)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# -----------------------
# Unit tests:
enable_testing()
add_subdirectory(tests)
