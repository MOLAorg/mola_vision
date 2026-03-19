/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Adapted from: lightweight_vio (MIT License).
 * ------------------------------------------------------------------------- */
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/image_utils.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace mola::vision;

// ---------------------------------------------------------------------------
// goodFeaturesToTrack
// ---------------------------------------------------------------------------
// 1. Compute structure tensor entries using Sobel gradients.
// 2. Box-filter them over a (2*block+1)² window.
// 3. Score: Shi-Tomasi = min eigenvalue = (Ixx+Iyy - sqrt((Ixx-Iyy)²+4·Ixy²)) / 2
//    Harris (if k>0): det(M) - k·trace(M)²
// 4. Non-maximum suppression within min_distance.
// 5. Threshold + sort + return top N.
// ---------------------------------------------------------------------------
std::vector<mrpt::math::TPoint2Df> mola::vision::goodFeaturesToTrack(
    const mrpt::img::CImage& img, const GoodFeaturesParams& params)
{
  const int rows = img.getHeight();
  const int cols = img.getWidth();

  // Step 1: Sobel gradients
  Eigen::MatrixXf Ix, Iy;
  sobelGradients(img, Ix, Iy);

  // Step 2: Compute Ix², IxIy, Iy² and box-filter with kernel radius block_size
  const int       b = params.block_size;
  Eigen::MatrixXf scores(rows, cols);
  scores.setZero();

  for (int r = b; r < rows - b; ++r)
  {
    for (int c = b; c < cols - b; ++c)
    {
      float Ixx = 0, Ixy = 0, Iyy = 0;
      for (int dr = -b; dr <= b; ++dr)
      {
        for (int dc = -b; dc <= b; ++dc)
        {
          const float ix = Ix(r + dr, c + dc);
          const float iy = Iy(r + dr, c + dc);
          Ixx += ix * ix;
          Ixy += ix * iy;
          Iyy += iy * iy;
        }
      }

      float score;
      if (params.harris_k <= 0.f)
      {
        // Shi-Tomasi: min eigenvalue
        const float trace = Ixx + Iyy;
        const float disc  = std::sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.f * Ixy * Ixy);
        score             = (trace - disc) * 0.5f;
      }
      else
      {
        // Harris
        const float det   = Ixx * Iyy - Ixy * Ixy;
        const float trace = Ixx + Iyy;
        score             = det - params.harris_k * trace * trace;
      }
      scores(r, c) = std::max(0.f, score);
    }
  }

  // Step 3: find max score
  const float max_score = scores.maxCoeff();
  if (max_score < 1e-10f) return {};
  const float threshold = params.quality_level * max_score;

  // Step 4: collect candidates above threshold
  struct Candidate
  {
    float score;
    int   r, c;
  };
  std::vector<Candidate> cands;
  cands.reserve(1024);
  for (int r = b; r < rows - b; ++r)
    for (int c = b; c < cols - b; ++c)
      if (scores(r, c) >= threshold) cands.push_back({scores(r, c), r, c});

  // Sort by score descending
  std::sort(
      cands.begin(), cands.end(),
      [](const Candidate& a, const Candidate& b_) { return a.score > b_.score; });

  // Step 5: greedy non-maximum suppression by min_distance
  const float                        min_dist2 = params.min_distance * params.min_distance;
  std::vector<mrpt::math::TPoint2Df> result;
  result.reserve(params.max_corners);

  for (const auto& cand : cands)
  {
    if (static_cast<int>(result.size()) >= params.max_corners) break;

    bool too_close = false;
    for (const auto& accepted : result)
    {
      const float dc = static_cast<float>(cand.c) - accepted.x;
      const float dr = static_cast<float>(cand.r) - accepted.y;
      if (dc * dc + dr * dr < min_dist2)
      {
        too_close = true;
        break;
      }
    }
    if (!too_close) result.push_back({static_cast<float>(cand.c), static_cast<float>(cand.r)});
  }

  return result;
}

// ---------------------------------------------------------------------------
// GridFeatureDistributor
// ---------------------------------------------------------------------------
GridFeatureDistributor::GridFeatureDistributor(const GridDistributorParams& params)
    : params_(params)
{
}

std::vector<mrpt::math::TPoint2Df> GridFeatureDistributor::detect(
    const mrpt::img::CImage& img, const std::vector<mrpt::math::TPoint2Df>& existing) const
{
  const int rows   = img.getHeight();
  const int cols   = img.getWidth();
  const int cell_h = rows / params_.grid_rows;
  const int cell_w = cols / params_.grid_cols;
  const int max_per_cell =
      std::max(1, params_.max_corners / (params_.grid_rows * params_.grid_cols));

  GoodFeaturesParams cell_params;
  cell_params.max_corners   = max_per_cell;
  cell_params.min_distance  = params_.min_distance;
  cell_params.quality_level = params_.quality_level;
  cell_params.block_size    = params_.block_size;

  std::vector<mrpt::math::TPoint2Df> result;
  result.reserve(params_.max_corners);

  const float min_dist2 = params_.min_distance * params_.min_distance;

  for (int gr = 0; gr < params_.grid_rows; ++gr)
  {
    for (int gc = 0; gc < params_.grid_cols; ++gc)
    {
      // Cell bounding box (clamped to image)
      const int y0 = gr * cell_h;
      const int x0 = gc * cell_w;
      const int y1 = std::min(rows, y0 + cell_h);
      const int x1 = std::min(cols, x0 + cell_w);

      // Extract cell as a patch
      mrpt::img::CImage patch;
      img.extract_patch(
          patch, {static_cast<int32_t>(x0), static_cast<int32_t>(y0)},
          {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)});

      auto cell_corners = goodFeaturesToTrack(patch, cell_params);

      // Back to full-image coordinates and check min_distance against all
      // already accepted (from previous cells + existing tracked points)
      for (auto& pt : cell_corners)
      {
        pt.x += static_cast<float>(x0);
        pt.y += static_cast<float>(y0);

        bool too_close = false;

        // Check against already-tracked existing features
        for (const auto& ex : existing)
        {
          const float dx = pt.x - ex.x, dy = pt.y - ex.y;
          if (dx * dx + dy * dy < min_dist2)
          {
            too_close = true;
            break;
          }
        }
        if (too_close) continue;

        // Check against already-accepted new detections
        for (const auto& acc : result)
        {
          const float dx = pt.x - acc.x, dy = pt.y - acc.y;
          if (dx * dx + dy * dy < min_dist2)
          {
            too_close = true;
            break;
          }
        }
        if (!too_close) result.push_back(pt);
      }
    }
  }

  // Trim to max_corners
  if (static_cast<int>(result.size()) > params_.max_corners) result.resize(params_.max_corners);

  return result;
}
