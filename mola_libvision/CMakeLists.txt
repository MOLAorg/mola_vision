# ------------------------------------------------------------------------------
#        A Modular Optimization framework for Localization and mApping
#                               (MOLA)
#
# Copyright (C) 2018-2025, Jose Luis Blanco-Claraco, contributors (AUTHORS.md)
# All rights reserved.
# Released under GNU GPL v3. See LICENSE file
# ------------------------------------------------------------------------------

# Minimum CMake vesion: limited by CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
cmake_minimum_required(VERSION 3.5)

# Tell CMake we'll use C++ for use in its tests/flags
project(mola_libvision LANGUAGES CXX)

# MOLA CMake scripts: "mola_xxx()"
find_package(mola_common REQUIRED)

# find CMake dependencies:
find_package(mrpt-img)
find_package(mrpt-poses)

# -----------------------
# define lib:
set(LIB_SRCS 
  src/*.cpp
)
file(LIB_PUBLIC_HDRS
  include/*.h
)

mola_add_library(
  TARGET ${PROJECT_NAME}
  SOURCES ${LIB_SRCS} ${LIB_PUBLIC_HDRS}
  PUBLIC_LINK_LIBRARIES
    mrpt::img
    mrpt::poses
  CMAKE_DEPENDENCIES
    mola_common
    mrpt-img
    mrpt-poses
)

# -----------------------
# define tests:
enable_testing()
add_subdirectory(tests)
