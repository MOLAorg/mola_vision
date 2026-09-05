# ------------------------------------------------------------------------------
#        A Modular Optimization framework for Localization and mApping (MOLA)
#
# Copyright (C) 2018-2026, Jose Luis Blanco-Claraco, contributors (AUTHORS.md)
# All rights reserved.
# Released under the GNU GPL v3 license. See LICENSE file
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.5)
project(mola_visual_slam LANGUAGES CXX)

# MOLA CMake scripts: "mola_xxx()"
find_package(mola_common REQUIRED)
find_mola_package(mola_kernel)

find_package(mrpt_obs REQUIRED)
find_package(mrpt_img REQUIRED)
find_package(mrpt_maps REQUIRED)
find_package(mrpt_viz REQUIRED)
find_package(TBB REQUIRED)
find_package(mola_libvision REQUIRED)

set(LIB_SRCS
  src/VisualSlam.cpp
  src/register.cpp
)
set(LIB_PUBLIC_HDRS
  include/mola_visual_slam/VisualSlam.h
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
    mrpt::mrpt_viz
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

# Install standalone module parameter files, for running this module as an
# additional front-end from another app (mola-lidar-odometry-cli --module):
install(
  DIRECTORY params
  DESTINATION share/${PROJECT_NAME}
)

# CLI app
# ---------------------------------
find_package(mola_input_kitti_dataset) # optional
find_package(mola_input_rosbag1) # optional
find_package(CLI11 REQUIRED)

mola_add_executable(
  TARGET  mola-visual-slam-cli
  SOURCES apps/mola-visual-slam-cli.cpp
  LINK_LIBRARIES
    ${PROJECT_NAME}
    CLI11::CLI11
)

if(mola_input_kitti_dataset_FOUND)
  message(STATUS "mola-visual-slam-cli: With Kitti = YES")
  target_link_libraries(mola-visual-slam-cli mola::mola_input_kitti_dataset)
  target_compile_definitions(mola-visual-slam-cli PRIVATE HAVE_MOLA_INPUT_KITTI)
else()
  message(STATUS "mola-visual-slam-cli: With Kitti = NO")
endif()

if(mola_input_rosbag1_FOUND)
  message(STATUS "mola-visual-slam-cli: With rosbag1 = YES")
  target_link_libraries(mola-visual-slam-cli mola::mola_input_rosbag1)
  target_compile_definitions(mola-visual-slam-cli PRIVATE HAVE_MOLA_INPUT_ROSBAG1)
else()
  message(STATUS "mola-visual-slam-cli: With rosbag1 = NO")
endif()

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
