# ------------------------------------------------------------------------------
#        A Modular Optimization framework for Localization and mApping
#                               (MOLA)
#
# Copyright (C) 2018-2024, Jose Luis Blanco-Claraco
# All rights reserved.
# Released under GNU GPL v3. See LICENSE file
# ------------------------------------------------------------------------------

# Minimum CMake vesion: limited by CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
cmake_minimum_required(VERSION 3.5)

if("$ENV{ROS_VERSION}" STREQUAL "2")
    set(DETECTED_ROS2 TRUE)
endif()

# Tell CMake we'll use C++ for use in its tests/flags
project(mola_basalt_vio LANGUAGES CXX)

# MOLA CMake scripts: "mola_xxx()"
find_package(mola_common REQUIRED)

# find dependencies:
#find_package(mrpt-tclap REQUIRED)
find_mola_package(mola_kernel)

# -----------------------
# define lib:
set(LIB_SRCS
  module/src/register.cpp
  module/src/VIO_Process.cpp
  module/src/VIO_Publish.cpp
  module/src/VIO_Initialize.cpp
  module/src/VIO_Parameters.cpp
  module/src/VIO_SensorCallbacks.cpp
)
set(LIB_PUBLIC_HDRS
  module/include/mola_basalt_vio/VisualInertialOdometry.h
)

mola_add_library(
  TARGET ${PROJECT_NAME}
  SOURCES ${LIB_SRCS} ${LIB_PUBLIC_HDRS}
  PUBLIC_LINK_LIBRARIES
    mola::mola_kernel
# PRIVATE_LINK_LIBRARIES
#  mrpt::maps
  CMAKE_DEPENDENCIES
    mola_kernel
)

# mola_add_library() expects headers under "ROOT/include", so do this manually here:
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/module/include>
  $<INSTALL_INTERFACE:include>
  #PRIVATE src
)
install(
  DIRECTORY module/include/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

# CLI apps
# ---------------------------------
#mola_add_executable(...)


# Install files:
install(
  DIRECTORY
    mola-cli-launchs
    pipelines
    #ros2-launchs
    #rviz2
  DESTINATION
    share/${PROJECT_NAME}
)

# -----------------------------------------------------------------------------
#  ROS2
# -----------------------------------------------------------------------------
if(DETECTED_ROS2)
  # find dependencies
  find_package(ament_cmake REQUIRED)

  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    find_package(ament_cmake_gtest REQUIRED)
    ament_lint_auto_find_test_dependencies()

    #add_subdirectory(test)
  endif()

  ament_package()
endif()
