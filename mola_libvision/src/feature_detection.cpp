/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
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

  // Step 2: Compute Ix², IxIy, Iy² and box-filter with kernel radius block_size.
  // The box filter (sum over a (2b+1)x(2b+1) window) is done separably with
  // running column/row sums -> O(rows*cols) instead of O(rows*cols*block^2).
  const int b = params.block_size;

  Eigen::MatrixXf Sxx = Ix.array().square();
  Eigen::MatrixXf Sxy = Ix.array() * Iy.array();
  Eigen::MatrixXf Syy = Iy.array().square();

  // Separable box sum (valid interior [b, n-b) only; borders left at 0).
  const auto boxFilter = [&](const Eigen::MatrixXf& S) -> Eigen::MatrixXf
  {
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(rows, cols);  // horizontal sums
    for (int r = 0; r < rows; ++r)
    {
      if (cols < 2 * b + 1)
      {
        continue;
      }
      float acc = 0.f;
      for (int c = 0; c <= 2 * b; ++c)
      {
        acc += S(r, c);
      }
      H(r, b) = acc;
      for (int c = b + 1; c < cols - b; ++c)
      {
        acc += S(r, c + b) - S(r, c - b - 1);
        H(r, c) = acc;
      }
    }
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(rows, cols);  // + vertical sums
    for (int c = b; c < cols - b; ++c)
    {
      if (rows < 2 * b + 1)
      {
        continue;
      }
      float acc = 0.f;
      for (int r = 0; r <= 2 * b; ++r)
      {
        acc += H(r, c);
      }
      B(b, c) = acc;
      for (int r = b + 1; r < rows - b; ++r)
      {
        acc += H(r + b, c) - H(r - b - 1, c);
        B(r, c) = acc;
      }
    }
    return B;
  };

  const Eigen::MatrixXf Bxx = boxFilter(Sxx);
  const Eigen::MatrixXf Bxy = boxFilter(Sxy);
  const Eigen::MatrixXf Byy = boxFilter(Syy);

  Eigen::MatrixXf scores(rows, cols);
  scores.setZero();

  for (int r = b; r < rows - b; ++r)
  {
    for (int c = b; c < cols - b; ++c)
    {
      const float Ixx = Bxx(r, c);
      const float Ixy = Bxy(r, c);
      const float Iyy = Byy(r, c);

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

  // Uniform-in-angle mode: a pinhole image plane packs less angle into each
  // pixel as the radius grows, by exactly 1/(1+(r/f)^2) per axis, so a spacing
  // and a per-cell quota expressed in pixels both over-represent the
  // periphery. Both are rescaled by that factor when a focal length is given.
  const bool  angularSpacing = params_.focal_length_px > 0.f;
  const float f2             = params_.focal_length_px * params_.focal_length_px;
  const float ppx            = params_.principal_x >= 0.f ? params_.principal_x : 0.5f * cols;
  const float ppy            = params_.principal_y >= 0.f ? params_.principal_y : 0.5f * rows;

  /// Squared pixel spacing required at a point, to hold the angular spacing
  /// constant across the field.
  const auto minDist2At = [&](float x, float y) -> float
  {
    if (!angularSpacing)
    {
      return params_.min_distance * params_.min_distance;
    }
    const float dx    = x - ppx;
    const float dy    = y - ppy;
    const float scale = 1.f + (dx * dx + dy * dy) / f2;
    const float d     = params_.min_distance * scale;
    return d * d;
  };

  // Per-cell quotas follow solid angle rather than pixel area. The solid angle
  // of a patch at radius r is proportional to f*A/(f^2+r^2)^(3/2).
  std::vector<float> cellWeight(static_cast<size_t>(params_.grid_rows) * params_.grid_cols, 1.f);
  if (angularSpacing)
  {
    float wSum = 0.f;
    for (int gr = 0; gr < params_.grid_rows; ++gr)
    {
      for (int gc = 0; gc < params_.grid_cols; ++gc)
      {
        const float x  = (static_cast<float>(gc) + 0.5f) * cell_w - ppx;
        const float y  = (static_cast<float>(gr) + 0.5f) * cell_h - ppy;
        const float r2 = x * x + y * y;
        const float w  = 1.f / std::pow(1.f + r2 / f2, 1.5f);
        cellWeight[static_cast<size_t>(gr) * params_.grid_cols + gc] = w;
        wSum += w;
      }
    }
    for (auto& w : cellWeight)
    {
      w = w * static_cast<float>(params_.grid_rows * params_.grid_cols) / wSum;
    }
  }

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

      if (angularSpacing)
      {
        const size_t ci = static_cast<size_t>(gr) * params_.grid_cols + gc;
        cell_params.max_corners =
            std::max(1, static_cast<int>(std::lround(max_per_cell * cellWeight[ci])));
        // Within one cell the radius barely changes, so a single scaled
        // spacing at the cell center is enough for its internal suppression.
        const float xc           = (static_cast<float>(gc) + 0.5f) * cell_w;
        const float yc           = (static_cast<float>(gr) + 0.5f) * cell_h;
        cell_params.min_distance = std::sqrt(minDist2At(xc, yc));
      }

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
        const float ptMinDist2 = minDist2At(pt.x, pt.y);

        for (const auto& ex : existing)
        {
          const float dx = pt.x - ex.x, dy = pt.y - ex.y;
          if (dx * dx + dy * dy < ptMinDist2)
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
          if (dx * dx + dy * dy < ptMinDist2)
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
