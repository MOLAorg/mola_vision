/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/img/CImage.h>
#include <mrpt/math/TPoint2D.h>

#include <Eigen/Core>
#include <vector>

namespace mola::vision
{
/** Parameters for Shi-Tomasi corner detection */
struct GoodFeaturesParams
{
  int   max_corners   = 500;  ///< Maximum number of corners to return
  float quality_level = 0.01f;  ///< Min score as fraction of best score
  float min_distance  = 8.0f;  ///< Min Euclidean distance between corners (pixels)
  int   block_size    = 3;  ///< Half-window for structure tensor (radius, e.g. 3→7x7)
  float harris_k      = 0.0f;  ///< 0 → Shi-Tomasi; >0 → Harris (k typically 0.04)
};

/** Shi-Tomasi (or Harris) corner detector.
 *
 *  Algorithm:
 *   1. Compute Sobel gradients Ix, Iy.
 *   2. Compute structure tensor entries Ix², IxIy, Iy² via box-filter (block_size).
 *   3. Score = min eigenvalue of tensor (Shi-Tomasi) or det−k·trace² (Harris).
 *   4. Non-maximum suppression in a (min_distance×min_distance) window.
 *   5. Threshold by quality_level * max_score.
 *   6. Sort by score descending, return top max_corners.
 *
 *  Input:  grayscale CH_GRAY D8U image.
 *  Output: pixel coordinates as TPoint2Df (col=x, row=y), sorted by score.
 *
 *  Adapted from: lightweight_vio (MIT License), goodFeaturesToTrack logic.
 */
std::vector<mrpt::math::TPoint2Df> goodFeaturesToTrack(
    const mrpt::img::CImage& img, const GoodFeaturesParams& params = {});

// ---------------------------------------------------------------------------

/** Grid-based feature distributor.
 *
 *  Divides the image into (grid_cols × grid_rows) cells and calls the
 *  detector independently per cell, then enforces:
 *   - max_per_cell features per cell.
 *   - Global min_distance between any two selected features.
 *   - A per-cell mask to avoid re-detecting already-tracked positions.
 *
 *  Usage:
 *  \code
 *    GridDistributorParams p;
 *    p.max_corners = 300;
 *    GridFeatureDistributor dist(p);
 *    auto pts = dist.detect(img, already_tracked_pts);
 *  \endcode
 *
 *  Adapted from: lightweight_vio (MIT License), extract_new_features() grid logic.
 */
struct GridDistributorParams
{
  int   grid_rows     = 4;
  int   grid_cols     = 6;
  int   max_corners   = 300;  ///< Total across the whole image
  float min_distance  = 10.0f;
  float quality_level = 0.01f;
  int   block_size    = 3;
};

class GridFeatureDistributor
{
 public:
  explicit GridFeatureDistributor(const GridDistributorParams& params = {});

  /** Detect new corners, avoiding positions in `existing`.
   *  Returns new corner positions (not overlapping with `existing`).
   */
  std::vector<mrpt::math::TPoint2Df> detect(
      const mrpt::img::CImage& img, const std::vector<mrpt::math::TPoint2Df>& existing = {}) const;

 private:
  GridDistributorParams params_;
};

}  // namespace mola::vision
