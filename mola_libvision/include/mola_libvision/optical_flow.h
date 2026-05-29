/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/img/CImage.h>
#include <mrpt/img/CImagePyramid.h>
#include <mrpt/math/TPoint2D.h>

#include <Eigen/Core>
#include <vector>

namespace mola::vision
{
/** Output status per tracked point */
enum class TrackStatus : uint8_t
{
  OK      = 1,  ///< Successfully tracked
  LOST    = 0,  ///< Could not track (lost or OOB)
  OUTLIER = 2,  ///< Tracked but rejected by F-matrix filter
};

/** Parameters for pyramidal Lucas-Kanade optical flow */
struct LKParams
{
  int   win_size          = 21;  ///< Patch half-width in pixels (full = 2*win+1)
  int   max_levels        = 3;  ///< Number of pyramid levels (0 = no pyramid)
  int   max_iters         = 30;  ///< Max iterations per level
  float eps               = 0.01f;  ///< Convergence threshold (pixel displacement)
  float min_eig_threshold = 1e-4f;  ///< Reject patches with low eigenvalue
};

/** Pyramidal Lucas-Kanade optical flow tracker.
 *
 *  Tracks a set of 2D points from `prev` to `curr` using the KLT algorithm
 *  at multiple pyramid scales (coarse-to-fine).
 *
 *  Algorithm (per pyramid level, coarse to fine):
 *   1. Compute image gradient patch (Ix, Iy) around the predicted position.
 *   2. Compute the 2x2 spatial gradient matrix G = ΣIx²  ΣIxIy / ΣIxIy  ΣIy².
 *   3. Iterate: solve G·v = Σ(It·Ix, It·Iy) for sub-pixel displacement v.
 *   4. Reject if min eigenvalue of G < min_eig_threshold (flat patch).
 *   5. Propagate update to next (finer) level × 2.
 *
 *  Output: tracked positions `next_pts` and per-point `status`.
 *
 *  Note: builds image pyramids internally; for repeated tracking on the
 *  same frame, use the overload accepting pre-built CImagePyramid objects.
 *
 *  Adapted from: lightweight_vio (MIT License), KLT tracker core.
 *  Algorithm reference: Bouguet, "Pyramidal Implementation of the Lucas Kanade
 *  Feature Tracker", Intel Corp., 2001.
 */
void calcOpticalFlowPyrLK(
    const mrpt::img::CImage& prev, const mrpt::img::CImage& curr,
    const std::vector<mrpt::math::TPoint2Df>& prev_pts,
    std::vector<mrpt::math::TPoint2Df>& next_pts, std::vector<TrackStatus>& status,
    const LKParams& params = {});

/** Overload accepting pre-built pyramids (avoids redundant pyramid builds when
 *  the same image is used in multiple tracking calls).
 */
void calcOpticalFlowPyrLK(
    const mrpt::img::CImagePyramid& prev_pyr, const mrpt::img::CImagePyramid& curr_pyr,
    const std::vector<mrpt::math::TPoint2Df>& prev_pts,
    std::vector<mrpt::math::TPoint2Df>& next_pts, std::vector<TrackStatus>& status,
    const LKParams& params = {});

// ---------------------------------------------------------------------------

/** Parameters for fundamental matrix RANSAC filter */
struct FMatrixFilterParams
{
  float ransac_threshold = 1.0f;  ///< Sampson distance threshold (pixels)
  float confidence       = 0.999f;  ///< RANSAC confidence level
  int   max_iters        = 1000;  ///< Max RANSAC iterations
};

/** 8-point fundamental matrix estimation + RANSAC inlier filtering.
 *
 *  For each pair (prev_pts[i], next_pts[i]) with status[i] == OK,
 *  fits a fundamental matrix F using the normalized 8-point algorithm
 *  inside a RANSAC loop.  Points with Sampson distance > ransac_threshold
 *  are marked TrackStatus::OUTLIER.
 *
 *  Requires at least 8 OK correspondences; if fewer, marks all as OUTLIER.
 *
 *  The 8-point algorithm uses Eigen SVD (JacobiSVD) for robustness.
 *  Normalization follows Hartley's scheme (centroid + RMS distance = √2).
 *
 *  Adapted from: lightweight_vio (MIT License), apply_fundamental_matrix_filter().
 */
void fundamentalMatrixFilter(
    const std::vector<mrpt::math::TPoint2Df>& prev_pts,
    const std::vector<mrpt::math::TPoint2Df>& next_pts, std::vector<TrackStatus>& status,
    const FMatrixFilterParams& params = {});

}  // namespace mola::vision
