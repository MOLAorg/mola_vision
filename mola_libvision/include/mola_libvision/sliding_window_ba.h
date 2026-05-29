/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/img/TCamera.h>
#include <mrpt/math/TPoint2D.h>
#include <mrpt/math/TPoint3D.h>
#include <mrpt/poses/CPose3D.h>

#include <vector>

namespace mola::vision
{
/** A single 2D observation of a landmark from a keyframe. */
struct BAObservation
{
  int                   kf_index = -1;  ///< Index into the poses vector
  int                   lm_index = -1;  ///< Index into the landmarks vector
  mrpt::math::TPoint2Df pixel;  ///< Observed pixel (already undistorted)
};

/** Options for slidingWindowBA(). */
struct BAOptions
{
  int max_iters = 10;

  /** Huber kernel width on the reprojection residual, in pixels. */
  float huber_delta = 2.448f;

  /** Initial LM damping added (additively, Levenberg style) to the pose and
   *  landmark Hessian diagonals. Adapted via the Nielsen gain-ratio rule. */
  double lambda_initial = 1e-3;

  /** Convergence: stop when the total state increment norm is below this. */
  double eps_step = 1e-7;

  /** If true and no explicit fixed mask is given, the first keyframe is held
   *  fixed to anchor the gauge. */
  bool fix_first_pose = true;
};

/** Result of slidingWindowBA(). */
struct BAResult
{
  bool   converged             = false;
  int    iterations            = 0;
  double initial_cost          = 0.0;  ///< 0.5 * sum of squared (robust-weighted) residuals
  double final_cost            = 0.0;
  int    num_observations_used = 0;
};

/** Sliding-window bundle adjustment: jointly refine a set of keyframe poses and
 *  3D landmark positions by minimizing the robust reprojection error.
 *
 *  Solver: Gauss-Newton with Levenberg-Marquardt damping. Each iteration builds
 *  the sparse normal equations and eliminates the landmark blocks via the Schur
 *  complement (each landmark is a 3x3 block), solving a reduced pose-only dense
 *  system, then back-substitutes the landmark updates. Robustified with a Huber
 *  kernel.
 *
 *  Conventions:
 *   - Poses are camera extrinsics T_cw (world -> camera): X_cam = pose o X_world.
 *   - Pose update: right-perturbation on rotation (R <- R*Exp(dw)), additive on
 *     translation. Landmark update: additive in world coordinates.
 *   - A single shared pinhole camera \p cam is used (distortion ignored; pass
 *     undistorted pixels). Gauge is anchored by fixing pose(s).
 *
 *  \param[in,out] poses      Keyframe poses T_cw, refined in place.
 *  \param[in,out] landmarks  World 3D points, refined in place.
 *  \param obs                Observations linking poses and landmarks.
 *  \param cam                Shared pinhole intrinsics (distortion ignored).
 *  \param fixedPoses         Optional per-pose "is fixed" mask (size == poses,
 *                            or empty). Fixed poses are not updated. If empty
 *                            and options.fix_first_pose, pose 0 is fixed.
 *  \param options            Solver options.
 *
 *  \note Schur-complement structure follows Basalt
 *        (BSD-3-Clause, (c) 2019 Usenko, Demmel) linearization_abs_sc; the
 *        residual/Jacobian formulas are the standard pinhole reprojection ones,
 *        shared with PnPSolver.
 */
BAResult slidingWindowBA(
    std::vector<mrpt::poses::CPose3D>& poses, std::vector<mrpt::math::TPoint3Df>& landmarks,
    const std::vector<BAObservation>& obs, const mrpt::img::TCamera& cam,
    const std::vector<bool>& fixedPoses = {}, const BAOptions& options = {});

}  // namespace mola::vision
