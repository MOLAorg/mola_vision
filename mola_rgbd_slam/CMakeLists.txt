# ------------------------------------------------------------------------------
#        A Modular Optimization framework for Localization and mApping (MOLA)
#
# Copyright (C) 2018-2026, Jose Luis Blanco-Claraco, contributors (AUTHORS.md)
# All rights reserved.
# Released under the GNU GPL v3 license. See LICENSE file
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.5)
project(mola_rgbd_slam LANGUAGES CXX)

# MOLA CMake scripts: "mola_xxx()"
find_package(mola_common REQUIRED)
find_mola_package(mola_kernel)

find_package(mrpt_obs REQUIRED)
find_package(mrpt_img REQUIRED)
find_package(mrpt_maps REQUIRED)
find_package(TBB REQUIRED)
find_package(mola_libvision REQUIRED)

set(LIB_SRCS
  src/RgbdSlam.cpp
  src/register.cpp
)
set(LIB_PUBLIC_HDRS
  include/mola_rgbd_slam/RgbdSlam.h
)

mola_add_library(
  TARGET ${PROJECT_NAME}
  SOURCES ${LIB_SRCS} ${LIB_PUBLIC_HDRS}
  PUBLIC_LINK_LIBRARIES
    mola::mola_kernel
    mola::mola_libvision
    mrpt::mrpt_obs
    mrpt::mrpt_img
    mrpt::mrpt_maps
  CMAKE_DEPENDENCIES
    mola_kernel
    mola_libvision
)

# mola_add_library() expects headers under "ROOT/include":
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

# Install launch files:
install(
  DIRECTORY mola-cli-launchs
  DESTINATION share/${PROJECT_NAME}
)

# Unit tests:
enable_testing()
add_subdirectory(tests)

# -----------------------------------------------------------------------------
#  ROS2
# -----------------------------------------------------------------------------
if(DETECTED_ROS2)
  find_package(ament_cmake REQUIRED)
  ament_package()
endif()
