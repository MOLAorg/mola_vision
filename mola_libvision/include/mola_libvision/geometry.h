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
// ---------------------------------------------------------------------------
// Undistortion
// ---------------------------------------------------------------------------

/** Batch undistort + unproject: pixel → normalized (metric) image coordinates.
 *
 *  Output: `undistorted[i]` = normalized plane coordinate (z=1), i.e.
 *    undistorted[i] = K⁻¹ · undistort(pixel[i])
 *  where undistort removes distortion and K⁻¹ de-projects to the unit plane.
 *
 *  \note Thin float facade over MRPT 3.x's batch
 *        `mrpt::img::camera_geometry::undistort_points_to_unit_plane()`
 *        (which returns double `TPoint2D`). Supports pinhole / plumb_bob and
 *        Kannala-Brandt distortion as encoded in TCamera. The facade exists
 *        because the SLAM pipeline carries features as `std::vector<TPoint2Df>`.
 */
void undistortPoints(
    const std::vector<mrpt::math::TPoint2Df>& pixels, const mrpt::img::TCamera& camera,
    std::vector<mrpt::math::TPoint2Df>& undistorted);

// ---------------------------------------------------------------------------
// Triangulation
// ---------------------------------------------------------------------------

/** Linear (DLT) triangulation from two views.
 *
 *  For each correspondence pair (pts1[i] in view1, pts2[i] in view2),
 *  solves for the 3D point X by SVD of the 4×4 design matrix derived from
 *  the two projection equations:
 *    pts1[i] × (P1 · X) = 0
 *    pts2[i] × (P2 · X) = 0
 *
 *  pts1, pts2: normalized image coordinates (z=1 plane, already undistorted).
 *  P1, P2:     3×4 projection matrices  [R|t]  (NOT K·[R|t]; camera is
 *              assumed to be already in normalized coordinates).
 *  out_pts3d:  output 3D points in world frame.
 *  out_valid:  per-point validity flag (false if point is behind either camera
 *              or has too-high reprojection error).
 *
 *  Algorithm: Hartley & Zisserman §12.2 "Linear triangulation methods".
 *  Implementation uses Eigen::JacobiSVD on the 4×4 system.
 */
void triangulatePoints(
    const std::vector<mrpt::math::TPoint2Df>& pts1, const std::vector<mrpt::math::TPoint2Df>& pts2,
    const Eigen::Matrix<float, 3, 4>& P1, const Eigen::Matrix<float, 3, 4>& P2,
    std::vector<mrpt::math::TPoint3Df>& out_pts3d, std::vector<bool>& out_valid);

/** Triangulate a single point from two normalized image points and relative
 *  pose  T_12 = [R | t]  (pose of camera 2 expressed in camera 1 frame).
 *  Returns std::nullopt if point is behind a camera or degenerate.
 */
std::optional<mrpt::math::TPoint3Df> triangulateSinglePoint(
    const mrpt::math::TPoint2Df& pt1, const mrpt::math::TPoint2Df& pt2,
    const Eigen::Matrix<float, 3, 3>& R, const Eigen::Vector3f& t);

// ---------------------------------------------------------------------------
// Essential / Fundamental matrix utilities
// ---------------------------------------------------------------------------

/** E = K2ᵀ · F · K1
 *  Simple 3×3 matrix multiplication; provided for convenience.
 */
Eigen::Matrix3f essentialFromFundamental(
    const Eigen::Matrix3f& F, const Eigen::Matrix3f& K1, const Eigen::Matrix3f& K2);

/** Decompose essential matrix E into the four candidate (R, t) solutions.
 *
 *  Uses SVD:  E = U·Σ·Vᵀ  →  two rotation candidates × two translation signs.
 *  Selects the physically correct one via chirality check (most inlier 3D
 *  points in front of both cameras) using the provided normalized correspondences.
 *
 *  Returns the recovered rotation R and unit-scale translation t (‖t‖=1).
 *  Returns false if fewer than 10 points pass the chirality test.
 */
bool decomposeEssentialMatrix(
    const Eigen::Matrix3f& E, const std::vector<mrpt::math::TPoint2Df>& pts1_norm,
    const std::vector<mrpt::math::TPoint2Df>& pts2_norm, Eigen::Matrix3f& R_out,
    Eigen::Vector3f& t_out);

}  // namespace mola::vision
