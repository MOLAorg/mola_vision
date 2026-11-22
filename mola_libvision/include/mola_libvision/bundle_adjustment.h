/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_libvision/camera_model.h>
#include <mrpt/poses/CPose3D.h>

#include <Eigen/Core>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/** \file bundle_adjustment.h
 *  Global bundle adjustment (GTSAM backend, not exposed in this header) with
 *  any CameraModel, distorted pixels, optional shared intrinsics as
 *  variables, points at infinity, robust losses and covariance recovery. */
namespace mola::vision
{
/** One observation of a point by a pose. */
struct BAObservationPx
{
  int             pose  = -1;  ///< index into BundleAdjustmentProblem::poses
  int             point = -1;  ///< index into BundleAdjustmentProblem::points
  Eigen::Vector2d pixel = Eigen::Vector2d::Zero();  ///< observed, DISTORTED pixel
  double          sigma = 1.0;  ///< observation standard deviation, pixels
};

/** Prior on the angle of the rotation between two poses, e.g. from a gyroscope
 *  or the odometry of a vehicle carrying the camera. The angle of a relative
 *  rotation is the same in every frame, so the rotation between that sensor
 *  and the camera is not needed. The angle's derivative is undefined at zero
 *  rotation: use pairs that rotated well above the prior's sigma. */
struct BARotationAnglePrior
{
  int    pose_a = -1;  ///< index into BundleAdjustmentProblem::poses
  int    pose_b = -1;  ///< index into BundleAdjustmentProblem::poses
  double angle  = 0.0;  ///< rotation angle from pose_a to pose_b, radians, in [0, pi]
  double sigma  = 0.0;  ///< standard deviation, radians
};

/** A bundle adjustment problem. Several poses may share one camera model
 *  (intrinsics group), which is what makes self-calibration possible. */
struct BundleAdjustmentProblem
{
  std::vector<CameraModel> cameras;
  /** Camera (intrinsics group) used by each pose; empty means camera 0. */
  std::vector<int> pose_camera;
  /** Camera poses T_cw: world to camera, X_c = pose.composePoint(X_w). */
  std::vector<mrpt::poses::CPose3D> poses;
  /** World points, or unit directions (world frame) for points at infinity. */
  std::vector<Eigen::Vector3d> points;
  /** Marks points at infinity; empty means all points are finite. A point at
   *  infinity is seen along the same direction from every camera center, so
   *  only rotations and intrinsics act on it: this keeps far or low-parallax
   *  points (and rotation-only sequences) usable for calibration. */
  std::vector<bool>            point_at_infinity;
  std::vector<BAObservationPx> observations;
  /** Optional priors on the rotation angle between pairs of poses. They are
   *  robustified with the same loss as the observations. With a focal length
   *  that the images alone determine poorly (e.g. a camera that only turns
   *  about one axis), they set it through the size of the rotations. */
  std::vector<BARotationAnglePrior> rotation_angle_priors;
};

enum class RobustLoss : uint8_t
{
  None = 0,
  Huber,
  Cauchy
};

struct BundleAdjustmentOptions
{
  RobustLoss loss = RobustLoss::Huber;
  /** Robust kernel width, in units of each observation's sigma. */
  double loss_scale = 2.0;

  int    max_iterations     = 100;
  double relative_error_tol = 1e-7;
  double absolute_error_tol = 1e-10;

  /** Per camera: which intrinsics are estimated. A missing entry or nullopt
   *  keeps that camera fixed. */
  std::vector<std::optional<IntrinsicsMask>> intrinsics;

  /** Optional Gaussian prior on each camera's free intrinsics, centered at
   *  their initial values: one standard deviation per free parameter, in the
   *  IntrinsicsParameterization order. A missing or empty entry means no
   *  prior. Useful to keep intrinsics from wandering while few views
   *  constrain them (e.g. while an incremental reconstruction grows). */
  std::vector<Eigen::VectorXd> intrinsics_prior_sigmas;

  /** Poses held fixed; empty means none. */
  std::vector<bool> fixed_poses;
  /** Hold all points fixed. */
  bool fix_points = false;
  /** Rotation-only motion: camera centers are held fixed, only orientations
   *  (and intrinsics, points) are estimated. */
  bool rotation_only = false;

  /** Scale gauge of each disconnected reconstruction when the caller fixes
   *  the gauge itself through fixed_poses: pairs (a, b) where pose `a` is
   *  fixed, and the distance from its center to the center of pose `b` is
   *  held at its initial value. */
  std::vector<std::pair<int, int>> scale_constraints;

  /** When nothing else fixes the similarity gauge of a monocular problem
   *  (no fixed pose, free points), hold a pose fixed and constrain the
   *  distance from it to the farthest camera center of its component (poses
   *  linked by finite points); other such components get their center and
   *  scale held the same way, their rotations staying free. */
  bool fix_gauge = true;

  /** The solve fails if more than this many observations that projected at
   *  the start no longer do: pushing an observation beyond the model's valid
   *  range can otherwise lower the cost. */
  int max_lost_projections = 0;

  /** Compute the marginal covariance of the free intrinsics. Directions that
   *  only the internal regularization determines are reported as NaN. */
  bool compute_covariance = false;

  bool verbose = false;
};

struct BundleAdjustmentResult
{
  bool        success      = false;
  int         iterations   = 0;
  double      initial_cost = 0.0;  ///< 0.5 * sum of robustified squared whitened errors
  double      final_cost   = 0.0;
  std::string message;

  /** Final reprojection error of each observation, in pixels; NaN when the
   *  point does not project (behind the camera, beyond the model's range). */
  std::vector<double> residuals;

  /** Per rotation angle prior: final angle minus the prior's, radians; NaN
   *  for priors left out (both poses constant). */
  std::vector<double> rotation_angle_errors;

  /** Per camera: covariance of the FULL intrinsic parameter vector (rows and
   *  columns of fixed parameters are zero), assuming the observation sigmas.
   *  Empty when not requested, or for fixed cameras, or if the information
   *  matrix was singular. */
  std::vector<Eigen::MatrixXd> intrinsics_covariance;

  /** Per camera: covariance of the free parameter vector only, in the
   *  IntrinsicsParameterization order (see its freeNames()). */
  std::vector<Eigen::MatrixXd> free_intrinsics_covariance;
};

/** Runs Levenberg-Marquardt on the problem and writes the refined cameras,
 *  poses and points back into it. */
BundleAdjustmentResult bundleAdjust(
    BundleAdjustmentProblem& problem, const BundleAdjustmentOptions& options);

}  // namespace mola::vision
