/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/img/TCamera.h>
#include <mrpt/math/TPoint2D.h>
#include <mrpt/math/TPoint3D.h>
#include <mrpt/poses/CPose3D.h>

#include <Eigen/Core>
#include <vector>

namespace mola::vision
{
/** Parameters for solvePnP(). */
struct PnPParams
{
  /** Max Gauss-Newton / LM iterations. */
  int max_iters = 20;

  /** Huber kernel width, in pixels. Residuals larger than this are
   *  down-weighted. Default ~ sqrt(5.991) (95% chi-square, 2 DoF). */
  float huber_delta = 2.448f;

  /** Squared-reprojection-error gate (pixels^2) for inlier classification.
   *  Default 5.991 = chi-square 95% quantile for 2 DoF. */
  float chi2_threshold = 5.991f;

  /** Convergence: stop when the se3 increment norm is below this. */
  float eps_step = 1e-7f;

  /** Initial LM damping (0 => pure Gauss-Newton). */
  float lambda_initial = 1e-3f;
};

/** Result of solvePnP(). */
struct PnPResult
{
  /** Estimated camera extrinsics T_cw mapping WORLD points to CAMERA
   *  coordinates: X_cam = pose.composePoint(X_world). */
  mrpt::poses::CPose3D pose;

  /** Per-correspondence inlier flag (same size as the inputs). */
  std::vector<bool> inliers;

  int   num_inliers = 0;
  bool  converged   = false;
  float final_cost  = 0.f;  ///< 0.5 * sum of squared inlier reprojection errors

  /** 6x6 covariance of the se3 pose increment (ordering [rotation; translation]),
   *  computed as the inverse of the Gauss-Newton information at the solution.
   *  Only meaningful when num_inliers is large enough. */
  Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity();
};

/** Perspective-n-Point: estimate the camera pose T_cw from 3D world points and
 *  their 2D pixel observations, via robust (Huber) iterative reweighted
 *  Gauss-Newton with Levenberg-Marquardt damping.
 *
 *  Camera model: the pinhole intrinsics (fx, fy, cx, cy) of \p cam are used;
 *  **lens distortion is ignored**. If your pixels are distorted, undistort
 *  them first (see undistortPoints()) and pass a distortion-free camera, or
 *  pass already-undistorted pixels.
 *
 *  Projection: u = fx * Xc/Zc + cx,  v = fy * Yc/Zc + cy, with
 *  Xc = pose.composePoint(Xw).
 *
 *  \param worldPts  3D points in the world frame.
 *  \param pixels    Their observed 2D pixel coordinates (same size).
 *  \param cam       Camera intrinsics (distortion ignored).
 *  \param initialPose Initial guess for T_cw (world -> camera).
 *  \param params    Solver options.
 *
 *  \note Adapted in spirit from lightweight_vio (MIT, (c) 2025 Seungwon Choi)
 *        PnP optimization; re-derived for MRPT types without Ceres.
 */
PnPResult solvePnP(
    const std::vector<mrpt::math::TPoint3Df>& worldPts,
    const std::vector<mrpt::math::TPoint2Df>& pixels, const mrpt::img::TCamera& cam,
    const mrpt::poses::CPose3D& initialPose, const PnPParams& params = {});

}  // namespace mola::vision
