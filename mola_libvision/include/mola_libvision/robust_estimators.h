/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_libvision/minimal_solvers.h>

#include <Eigen/Core>
#include <vector>

/** \file robust_estimators.h
 *  MSAC (M-estimator sample consensus) wrappers around the minimal solvers,
 *  with adaptive iteration counts, deterministic seeding and a final
 *  non-minimal refit on the inliers. */
namespace mola::vision
{
/** Options shared by all robust estimators. */
struct RansacOptions
{
  /** Inlier threshold. Units depend on the estimator: radians for bearing
   *  based models, pixels (or the input units) for 2D models. */
  double threshold      = 1.0;
  int    max_iterations = 2000;
  int    min_iterations = 50;
  /** Probability of drawing at least one all-inlier sample. */
  double confidence = 0.9999;
  /** RNG seed: equal inputs and seed give equal outputs. */
  unsigned seed = 42;
  /** Refit the model on the inliers after sampling, keeping it only if the
   *  consensus score improves. */
  bool refine = true;
};

/** Common part of all estimation results. */
struct RansacReport
{
  bool                success     = false;
  int                 iterations  = 0;
  int                 num_inliers = 0;
  double              score       = 0.0;  ///< MSAC cost: sum of min(e^2, threshold^2)
  std::vector<bool>   inliers;
  std::vector<double> residuals;  ///< final per-correspondence error, input units
};

struct RelativePoseEstimate
{
  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  RelativePose    pose;  ///< x2 = R x1 + t, |t| = 1
  RansacReport    report;
};

/** Calibrated relative pose from bearing correspondences: five-point MSAC on
 *  the angular epipolar error (threshold in radians), refit with the linear
 *  solver on the inliers, and chirality-based decomposition. */
RelativePoseEstimate estimateRelativePose(
    const std::vector<Eigen::Vector3d>& b1, const std::vector<Eigen::Vector3d>& b2,
    const RansacOptions& options);

struct FundamentalEstimate
{
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  RansacReport    report;
};

/** Fundamental matrix from 2D correspondences: seven-point MSAC on the
 *  Sampson distance (threshold in the input units, usually pixels). */
FundamentalEstimate estimateFundamental(
    const std::vector<Eigen::Vector2d>& p1, const std::vector<Eigen::Vector2d>& p2,
    const RansacOptions& options);

struct HomographyEstimate
{
  Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
  RansacReport    report;
};

/** Homography from 2D correspondences: four-point MSAC on the symmetric
 *  transfer error (threshold in the input units). */
HomographyEstimate estimateHomography(
    const std::vector<Eigen::Vector2d>& p1, const std::vector<Eigen::Vector2d>& p2,
    const RansacOptions& options);

struct RotationEstimate
{
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  ///< b2 ~ R b1
  RansacReport    report;
};

/** Rotation-only relative motion from bearing pairs: two-point MSAC on the
 *  angle between b2 and R b1 (threshold in radians). */
RotationEstimate estimateRotation(
    const std::vector<Eigen::Vector3d>& b1, const std::vector<Eigen::Vector3d>& b2,
    const RansacOptions& options);

struct AbsolutePoseEstimate
{
  AbsolutePose pose;  ///< x_c = R X + t
  RansacReport report;
};

/** Absolute pose from 3D world points and their bearings: P3P MSAC on the
 *  angle between the observed bearing and the predicted ray (threshold in
 *  radians), with a Gauss-Newton refit on the inliers. */
AbsolutePoseEstimate estimateAbsolutePose(
    const std::vector<Eigen::Vector3d>& points_w, const std::vector<Eigen::Vector3d>& bearings,
    const RansacOptions& options);

}  // namespace mola::vision
