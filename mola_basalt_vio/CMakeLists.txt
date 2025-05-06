# ------------------------------------------------------------------------------
#        A Modular Optimization framework for Localization and mApping
#                               (MOLA)
#
# Copyright (C) 2018-2024, Jose Luis Blanco-Claraco
# All rights reserved.
# Released under GNU GPL v3. See LICENSE file
# ------------------------------------------------------------------------------

# Minimum CMake vesion: limited by CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
cmake_minimum_required(VERSION 3.5...3.18)

if("$ENV{ROS_VERSION}" STREQUAL "2")
    set(DETECTED_ROS2 TRUE)
endif()

# Tell CMake we'll use C++ for use in its tests/flags
project(mola_basalt_vio LANGUAGES C CXX)

# MOLA CMake scripts: "mola_xxx()"
find_package(mola_common REQUIRED)

# find dependencies:
#find_package(mrpt-tclap REQUIRED)
find_mola_package(mola_kernel)

# -----------------------------------------------------------
# Start of defining the basalt imported library
# -----------------------------------------------------------
set(BASALT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/basalt")

if(NOT CXX_MARCH)
  set(CXX_MARCH native)
endif()

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(BASALT_MARCH_FLAGS "-march=${CXX_MARCH}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# base set of compile flags
set(BASALT_CXX_FLAGS "-Wall -Wextra -Wno-error=unused-parameter -ftemplate-backtrace-limit=0")
# clang-specific compile flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-exceptions -fcolor-diagnostics -frelaxed-template-template-args -Wno-error=deprecated-declarations")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
      # These are disabled to avoid lot's of warnings in Eigen code with clang 10
      set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-misleading-indentation -Wno-error=deprecated-copy")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
      # These are disabled to avoid lot's of warnings in Eigen code with clang 15
      set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=unused-but-set-variable")
  endif()

  #   - Added TBB_USE_GLIBCXX_VERSION macro to specify the version of GNU
  #     libstdc++ when it cannot be properly recognized, e.g. when used
  #     with Clang on Linux* OS. Adopted from https://github.com/wjakob/tbb
  if(NOT TBB_USE_GLIBCXX_VERSION AND UNIX AND NOT APPLE)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    endif()
    add_definitions(-DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
  endif()
else()
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=maybe-uninitialized -Wno-error=implicit-fallthrough")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
      set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=uninitialized")
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
      # These are disabled to avoid lot's of warnings in Eigen code with gcc-9
      set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=deprecated-copy")
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
    # These are disabled due to warnings in pangolin
    #set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=parentheses")
    set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-parentheses -Wno-error=array-bounds")
  endif()
endif()

if (UNIX AND NOT APPLE)
  # assume libstdc++
  set(STD_CXX_FS stdc++fs)
endif()

# Set EIGEN_DONT_PARALLELIZE to be sure it doesn't use openmp,
# just in case some dependency enables openmp without us noticing.
set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")

# Make specialization for double / float optional.
# Disabling them can be useful for better compile times during development.
option(BASALT_INSTANTIATIONS_DOUBLE "Instatiate templates for Scalar=double." ON)
option(BASALT_INSTANTIATIONS_FLOAT "Instatiate templates for Scalar=float." ON)

if(BASALT_INSTANTIATIONS_DOUBLE)
  list(APPEND BASALT_COMPILE_DEFINITIONS BASALT_INSTANTIATIONS_DOUBLE)
endif()
if(BASALT_INSTANTIATIONS_FLOAT)
  list(APPEND BASALT_COMPILE_DEFINITIONS BASALT_INSTANTIATIONS_FLOAT)
endif()

# setup combined compiler flags
set(CMAKE_CXX_FLAGS "${BASALT_CXX_FLAGS} ${BASALT_MARCH_FLAGS} ${BASALT_PASSED_CXX_FLAGS}")

find_package(Eigen3 REQUIRED) # 3.4.0 EXACT REQUIRED MODULE)
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")

find_package(TBB REQUIRED)
message(STATUS "Found TBB ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} (interface version ${TBB_INTERFACE_VERSION}) headers in: ${TBB_INCLUDE_DIRS}")

# NOTE: not specifying version, since 2, 3 or 4 is fine
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui)
message(STATUS "Found OpenCV ${OpenCV_VERSION} headers in: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV_LIBS: ${OpenCV_LIBS}")

# Add our own custom scoped opencv target since none is provided by OpenCV itself
add_library(basalt::opencv INTERFACE IMPORTED)
set_property(TARGET basalt::opencv PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
set_property(TARGET basalt::opencv PROPERTY INTERFACE_LINK_LIBRARIES ${OpenCV_LIBS})

find_package(fmt REQUIRED)
message(STATUS "Found {fmt} ${fmt_VERSION} in: ${fmt_DIR}")

set(BASALT_PANGO_TARGETS pango_display pango_image pango_windowing pango_plot pango_video)

add_subdirectory(${BASALT_DIR}/thirdparty)

# custom scoped cli11 target
add_library(basalt::cli11 INTERFACE IMPORTED)
set_property(TARGET basalt::cli11 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${BASALT_DIR}/thirdparty/CLI11/include)

# custom scoped magic_enum target
add_library(basalt::magic_enum INTERFACE IMPORTED)
set_property(TARGET basalt::magic_enum PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${BASALT_DIR}/thirdparty/magic_enum/include)


# -----------------------------------------------------------
# End of basalt library
# -----------------------------------------------------------

# define lib:
set(LIB_SRCS
  module/src/register.cpp
  module/src/VIO_Process.cpp
  module/src/VIO_Publish.cpp
  module/src/VIO_Initialize.cpp
  module/src/VIO_Parameters.cpp
  module/src/VIO_SensorCallbacks.cpp
  # -------
  ${BASALT_DIR}/src/calibration/aprilgrid.cpp
  ${BASALT_DIR}/src/calibration/calibraiton_helper.cpp
  ${BASALT_DIR}/src/calibration/vignette.cpp
  ${BASALT_DIR}/src/io/dataset_io.cpp
  ${BASALT_DIR}/src/io/marg_data_io.cpp
  ${BASALT_DIR}/src/linearization/landmark_block.cpp
  ${BASALT_DIR}/src/linearization/linearization_abs_qr.cpp
  ${BASALT_DIR}/src/linearization/linearization_abs_sc.cpp
  ${BASALT_DIR}/src/linearization/linearization_base.cpp
  ${BASALT_DIR}/src/linearization/linearization_rel_sc.cpp
  ${BASALT_DIR}/src/optical_flow/optical_flow.cpp
  ${BASALT_DIR}/src/utils/keypoints.cpp
  ${BASALT_DIR}/src/utils/system_utils.cpp
  ${BASALT_DIR}/src/utils/time_utils.cpp
  ${BASALT_DIR}/src/utils/vio_config.cpp
  ${BASALT_DIR}/src/vi_estimator/ba_base.cpp
  ${BASALT_DIR}/src/vi_estimator/landmark_database.cpp
  ${BASALT_DIR}/src/vi_estimator/marg_helper.cpp
  ${BASALT_DIR}/src/vi_estimator/nfr_mapper.cpp
  ${BASALT_DIR}/src/vi_estimator/sc_ba_base.cpp
  ${BASALT_DIR}/src/vi_estimator/sqrt_ba_base.cpp
  ${BASALT_DIR}/src/vi_estimator/sqrt_keypoint_vio.cpp
  ${BASALT_DIR}/src/vi_estimator/sqrt_keypoint_vo.cpp
  ${BASALT_DIR}/src/vi_estimator/vio_estimator.cpp
)
set(LIB_PUBLIC_HDRS
  module/include/mola_basalt_vio/VisualInertialOdometry.h
)

mola_add_library(
  TARGET ${PROJECT_NAME}
  SOURCES ${LIB_SRCS} ${LIB_PUBLIC_HDRS}
  PUBLIC_LINK_LIBRARIES
    mola::mola_kernel
  PRIVATE_LINK_LIBRARIES
    ${STD_CXX_FS} 
    basalt::opencv
    basalt::basalt-headers
    TBB::tbb
    basalt::magic_enum
    rosbag
    apriltag
    opengv
    nlohmann::json
    fmt::fmt
  CMAKE_DEPENDENCIES
    mola_kernel
)

target_include_directories(${PROJECT_NAME} PRIVATE ${BASALT_DIR}/include)
target_compile_definitions(${PROJECT_NAME} PRIVATE ${BASALT_COMPILE_DEFINITIONS})
#target_compile_definitions(basalt PUBLIC BASALT_DISABLE_ASSERTS)

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
