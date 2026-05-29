/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Adapted from: lightweight_vio (MIT License).
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_libvision/MapPoint.h>
#include <mrpt/core/Clock.h>
#include <mrpt/img/CImage.h>
#include <mrpt/img/TCamera.h>
#include <mrpt/math/TPoint2D.h>
#include <mrpt/poses/CPose3D.h>

#include <Eigen/Core>
#include <memory>
#include <optional>
#include <vector>

namespace mola::vision
{
/** IMU measurement accumulated between two keyframes */
struct ImuMeasurement
{
  double          timestamp = 0;
  Eigen::Vector3f gyro;  ///< Angular velocity  (rad/s) in body frame
  Eigen::Vector3f accel;  ///< Linear acceleration (m/s²) in body frame
};

/** A 2D feature detected and tracked in a Keyframe */
struct Feature
{
  int                   id      = -1;  ///< Unique feature id (across frames)
  int                   prev_id = -1;  ///< Feature id in previous frame (-1 if new)
  mrpt::math::TPoint2Df pixel;  ///< Distorted pixel coordinates
  mrpt::math::TPoint2Df pixel_undist;  ///< Undistorted pixel coordinates
  mrpt::math::TPoint2Df norm_coord;  ///< Normalized (z=1) undistorted coords
  int                   track_count = 1;  ///< How many frames this has been tracked
  bool                  is_outlier  = false;  ///< Marked bad by F-matrix or reprojection filter
  float                 depth       = -1.f;  ///< Metric depth (m); <0 if unknown
  float                 reprojection_error = -1.f;

  // Stereo
  std::optional<mrpt::math::TPoint2Df> right_pixel;  ///< Right-image pixel (stereo only)
  std::optional<float>                 disparity;

  MapPoint::Ptr map_point;  ///< Associated 3D landmark (null if not yet triangulated)
};

/** Keyframe: one processed image (or stereo pair) with associated pose,
 *  detected features, and optional IMU data.
 *
 *  The frame type (MONO / STEREO / RGBD) determines which image fields
 *  are valid.
 */
struct Keyframe
{
  using Ptr = std::shared_ptr<Keyframe>;

  // Type tag
  enum class Type
  {
    MONO,
    STEREO,
    RGBD
  };
  Type type = Type::MONO;

  // Identity
  int                     id        = -1;
  mrpt::Clock::time_point timestamp = {};

  // Images  (always set: left/gray; right only for STEREO; depth only for RGBD)
  mrpt::img::CImage  image;  ///< Left/gray image (grayscale, CH_GRAY)
  mrpt::img::CImage  image_right;  ///< Right image (STEREO only)
  mrpt::img::CImage  depth_image;  ///< Depth map in float meters (RGBD only)
  mrpt::img::TCamera camera;  ///< Camera intrinsics + distortion
  double             baseline = 0.0;  ///< Stereo baseline (m); 0 for MONO/RGBD

  // Pose: world←camera (or world←body if using body frame)
  mrpt::poses::CPose3D pose_world_camera;  ///< Estimated camera pose in world frame
  bool                 pose_valid = false;

  // Velocity & IMU biases (for VIO)
  Eigen::Vector3f velocity   = Eigen::Vector3f::Zero();
  Eigen::Vector3f gyro_bias  = Eigen::Vector3f::Zero();
  Eigen::Vector3f accel_bias = Eigen::Vector3f::Zero();

  // Features (filled by FeatureDetector / FeatureTracker)
  std::vector<Feature> features;

  // IMU measurements since previous keyframe (for VIO)
  std::vector<ImuMeasurement> imu_measurements;

  // Reference keyframe for relative-pose tracking (used in marginalization)
  std::weak_ptr<Keyframe> ref_keyframe;
  mrpt::poses::CPose3D    pose_relative_to_ref;  ///< pose = ref_pose ⊕ pose_relative_to_ref

  // Flag
  bool is_keyframe = false;  ///< true only for selected keyframes (not every frame)
};

}  // namespace mola::vision
