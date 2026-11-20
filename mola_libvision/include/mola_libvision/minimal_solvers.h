/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <Eigen/Core>
#include <optional>
#include <vector>

/** \file minimal_solvers.h
 *  Minimal and linear solvers for two-view and absolute pose geometry, in
 *  double precision.
 *
 *  Calibrated solvers work on BEARING vectors (3D ray directions in the camera
 *  frame, any norm unless stated), so they apply to any central camera model:
 *  undistort/unproject pixels with CameraModel::unproject() first. Uncalibrated
 *  solvers (fundamental matrix, homography) work on 2D points.
 *
 *  Conventions: a relative pose (R, t) maps camera-1 coordinates to camera-2
 *  coordinates, x2 = R x1 + t, and the essential matrix is E = [t]x R, so that
 *  b2^T E b1 = 0. An absolute pose (R, t) maps world to camera, x_c = R X + t.
 */
namespace mola::vision
{
// ---------------------------------------------------------------------------
// Minimal solvers
// ---------------------------------------------------------------------------

/** Five-point relative pose (essential matrix) from exactly five bearing
 *  correspondences, after Nister (2004): null space of the epipolar
 *  constraints, cubic constraints of the essential manifold, Gauss-Jordan
 *  elimination and a degree-10 polynomial in one unknown.
 *  \return Up to 10 essential matrices (unit Frobenius norm). */
std::vector<Eigen::Matrix3d> solveEssentialFivePoint(
    const Eigen::Vector3d b1[5], const Eigen::Vector3d b2[5]);

/** Seven-point fundamental matrix from exactly seven 2D correspondences
 *  (Hartley-normalized internally), x2^T F x1 = 0. Up to 3 solutions. */
std::vector<Eigen::Matrix3d> solveFundamentalSevenPoint(
    const Eigen::Vector2d p1[7], const Eigen::Vector2d p2[7]);

/** Normalized eight-point fundamental matrix from N >= 8 correspondences,
 *  with the rank-2 constraint enforced. */
std::optional<Eigen::Matrix3d> solveFundamentalLinear(
    const std::vector<Eigen::Vector2d>& p1, const std::vector<Eigen::Vector2d>& p2);

/** Linear essential matrix from N >= 8 bearing correspondences, projected onto
 *  the essential manifold (two equal singular values, the third zero) and
 *  scaled to unit Frobenius norm, as the five-point solutions. */
std::optional<Eigen::Matrix3d> solveEssentialLinear(
    const std::vector<Eigen::Vector3d>& b1, const std::vector<Eigen::Vector3d>& b2);

/** Normalized DLT homography from N >= 4 correspondences, x2 ~ H x1. */
std::optional<Eigen::Matrix3d> solveHomographyDLT(
    const std::vector<Eigen::Vector2d>& p1, const std::vector<Eigen::Vector2d>& p2);

/** Rotation-only relative pose from N >= 2 bearing pairs, b2 ~ R b1
 *  (Kabsch / orthogonal Procrustes on unit vectors). */
std::optional<Eigen::Matrix3d> solveRotationKabsch(
    const std::vector<Eigen::Vector3d>& b1, const std::vector<Eigen::Vector3d>& b2);

/** One absolute pose solution: x_c = R X + t. */
struct AbsolutePose
{
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

/** Perspective-three-point from three bearings and their world points, by
 *  Grunert's elimination into a quartic in the depth ratio, then a rigid
 *  alignment of the three recovered camera-frame points.
 *  \return Up to 4 poses. */
std::vector<AbsolutePose> solveP3P(
    const Eigen::Vector3d bearings[3], const Eigen::Vector3d points[3]);

/** Rigid transform (R, t) minimizing sum |q_i - (R p_i + t)|^2 (N >= 3). */
std::optional<AbsolutePose> alignPointSets(
    const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& q);

// ---------------------------------------------------------------------------
// Decomposition, triangulation and error measures
// ---------------------------------------------------------------------------

/** Relative pose recovered from an essential matrix. */
struct RelativePose
{
  Eigen::Matrix3d R            = Eigen::Matrix3d::Identity();  ///< x2 = R x1 + t
  Eigen::Vector3d t            = Eigen::Vector3d::Zero();  ///< unit norm
  int             num_in_front = 0;  ///< correspondences triangulated in front of both
};

/** Picks among the four (R, t) decompositions of E the one placing most of
 *  the (masked) correspondences in front of both cameras. */
std::optional<RelativePose> relativePoseFromEssential(
    const Eigen::Matrix3d& E, const std::vector<Eigen::Vector3d>& b1,
    const std::vector<Eigen::Vector3d>& b2, const std::vector<bool>* mask = nullptr);

/** Triangulates a world point from N >= 2 views by minimizing the sum of
 *  squared distances to the viewing rays (multi-view midpoint), then refines
 *  it with a few Gauss-Newton steps on the angular error.
 *
 *  \param R_cw, t_cw  World-to-camera poses of each view.
 *  \param bearings    Bearing of the point in each view (camera frame).
 *  \param min_angle   Minimum largest pairwise ray angle (radians); below it
 *                     the point is considered too far to triangulate.
 *  \return nullopt if ill-conditioned, below min_angle, or behind any view. */
std::optional<Eigen::Vector3d> triangulateBearings(
    const std::vector<Eigen::Matrix3d>& R_cw, const std::vector<Eigen::Vector3d>& t_cw,
    const std::vector<Eigen::Vector3d>& bearings, double min_angle = 0.0);

/** Largest angle (radians) between the world-frame rays of a point seen from
 *  several camera centers: the triangulation (parallax) angle. */
double triangulationAngle(
    const Eigen::Vector3d& X, const std::vector<Eigen::Vector3d>& camera_centers);

/** Symmetric angular epipolar error (radians, small-angle): the mean of the
 *  angles between each bearing and the epipolar plane of the other one. */
double epipolarAngularError(
    const Eigen::Matrix3d& E, const Eigen::Vector3d& b1, const Eigen::Vector3d& b2);

/** Sampson distance (in the units of the points) of a correspondence to F. */
double sampsonDistance(
    const Eigen::Matrix3d& F, const Eigen::Vector2d& p1, const Eigen::Vector2d& p2);

/** Symmetric transfer error of a correspondence for x2 ~ H x1. */
double homographyTransferError(
    const Eigen::Matrix3d& H, const Eigen::Vector2d& p1, const Eigen::Vector2d& p2);

/** Angle (radians) between two directions. */
double angleBetween(const Eigen::Vector3d& a, const Eigen::Vector3d& b);

/** Geometric robust information criterion (Torr, 1998) of a two-view model:
 *  GRIC = sum min(e^2/sigma^2, 2 (r - d)) + n d ln(r) + k ln(r n),
 *  with r = 4 for two-view correspondences.
 *  \param residuals  Per-correspondence GEOMETRIC errors (distance to the model
 *                    in the joint space of both images), in the units of
 *                    sigma: the Sampson distance for F/E, and for a
 *                    homography its symmetric transfer error over sqrt(2).
 *  \param d          Dimension of the model manifold (3 for F/E, 2 for H).
 *  \param k          Number of model parameters (7 F, 5 E, 8 H, 3 rotation). */
double gric(const std::vector<double>& residuals, double sigma, int d, int k, int r = 4);

}  // namespace mola::vision
