# ------------------------------------------------------------------------------
#        A Modular Optimization framework for Localization and mApping
#                               (MOLA)
#
# Copyright (C) 2018-2026, Jose Luis Blanco-Claraco, contributors (AUTHORS.md)
# All rights reserved.
# Released under BSD 3-clause license. See LICENSE file
# ------------------------------------------------------------------------------

# Minimum CMake version: limited by CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
cmake_minimum_required(VERSION 3.5)

# Tell CMake we'll use C++ for use in its tests/flags
project(mola_libvision LANGUAGES CXX)

# MOLA CMake scripts: "mola_xxx()"
find_package(mola_common REQUIRED)

# Dependencies: MRPT (v3) + Eigen3 + TBB.  NO OpenCV, NO Ceres.
find_package(mrpt-img   REQUIRED)
find_package(mrpt-poses REQUIRED)
find_package(TBB        REQUIRED)
find_package(Eigen3     REQUIRED)  # usually pulled in via mrpt-img already

# -----------------------
# Define library:
set(LIB_SRCS
  src/image_utils.cpp
  src/feature_detection.cpp
  src/optical_flow.cpp
  src/geometry.cpp
  src/MapPoint.cpp
  # lm_solver.h is header-only (template)
  # Keyframe.h  is header-only (POD struct)
  # ORB stub (kept until Phase 5):
  src/orb_detector.cpp
)

set(LIB_PUBLIC_HDRS
  include/mola_libvision/image_utils.h
  include/mola_libvision/feature_detection.h
  include/mola_libvision/optical_flow.h
  include/mola_libvision/geometry.h
  include/mola_libvision/MapPoint.h
  include/mola_libvision/Keyframe.h
  include/mola_libvision/lm_solver.h
)

mola_add_library(
  TARGET ${PROJECT_NAME}
  SOURCES ${LIB_SRCS} ${LIB_PUBLIC_HDRS}
  PUBLIC_LINK_LIBRARIES
    mrpt::img
    mrpt::poses
    TBB::tbb
    Eigen3::Eigen
  CMAKE_DEPENDENCIES
    mola_common
    mrpt-img
    mrpt-poses
)

# C++17 required (for std::optional, structured bindings, etc.)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# -----------------------
# Unit tests:
enable_testing()
add_subdirectory(tests)
