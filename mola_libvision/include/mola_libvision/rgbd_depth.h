/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/img/TCamera.h>
#include <mrpt/math/TPoint2D.h>
#include <mrpt/math/TPoint3D.h>

#include <Eigen/Core>
#include <optional>
#include <vector>

namespace mola::vision
{
/** RGBD depth processing parameters. */
struct RGBDParams
{
  /** Valid metric depth range, in meters. Depths outside are treated as
   *  invalid (no measurement). */
  float min_depth = 0.1f;
  float max_depth = 100.0f;

  /** Quadratic depth-noise model variance:  var(d) = a*d^2 + b*d + c
   *  (so the standard deviation is sqrt(var)). Defaults are typical for a
   *  structured-light / active-stereo RGBD camera (e.g. RealSense). */
  float unc_a = 0.0012f;
  float unc_b = 0.0019f;
  float unc_c = 0.0001f;
};

/** Standard deviation (meters) of a metric depth measurement, from the
 *  quadratic model sigma = sqrt(a*d^2 + b*d + c). */
[[nodiscard]] float depthStdDev(float depth, const RGBDParams& params = {});

/** Back-project one pixel with metric depth to a 3D point in the CAMERA frame:
 *    X = (u - cx) * d / fx,  Y = (v - cy) * d / fy,  Z = d.
 *
 *  \param pixel  Pixel coordinates in the (undistorted) image.
 *  \param depth  Metric depth (meters) at that pixel.
 *  \param cam    Pinhole intrinsics. The depth image is assumed registered to
 *                these (undistorted) intrinsics; lens distortion is not applied.
 *  \return The 3D point, or std::nullopt if the depth is invalid (<= 0 or
 *          outside [min_depth, max_depth]).
 */
[[nodiscard]] std::optional<mrpt::math::TPoint3Df> backprojectPixel(
    const mrpt::math::TPoint2Df& pixel, float depth, const mrpt::img::TCamera& cam,
    const RGBDParams& params = {});

/** A dense point cloud back-projected from a depth map. */
struct DepthCloud
{
  std::vector<mrpt::math::TPoint3Df> points;  ///< 3D points in the camera frame
  std::vector<mrpt::math::TPoint2Df> pixels;  ///< source pixel of each point (for coloring)
};

/** Back-project a whole depth map (meters) to a camera-frame point cloud,
 *  skipping invalid pixels.
 *
 *  \param depth_m     HxW depth matrix in meters (rows=height, cols=width).
 *                     Zero / out-of-range entries are skipped.
 *  \param cam         Pinhole intrinsics (must match the depth map resolution).
 *  \param params      Valid-depth range etc.
 *  \param decimation  Keep every Nth row and column (>=1) to subsample.
 */
[[nodiscard]] DepthCloud backprojectDepthMap(
    const Eigen::MatrixXf& depth_m, const mrpt::img::TCamera& cam, const RGBDParams& params = {},
    int decimation = 1);

}  // namespace mola::vision
