/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Pyramidal Lucas-Kanade optical flow and fundamental matrix RANSAC filter.
 * Adapted from: lightweight_vio (MIT License) and Bouguet 2001.
 * ------------------------------------------------------------------------- */
#include <mola_libvision/image_utils.h>
#include <mola_libvision/optical_flow.h>

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <random>

using namespace mola::vision;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace
{
/** Bilinearly sample a float gradient image at sub-pixel (x, y). */
inline float sampleBilinear(const Eigen::MatrixXf& img, float x, float y)
{
  const int   cols = static_cast<int>(img.cols());
  const int   rows = static_cast<int>(img.rows());
  const int   x0   = static_cast<int>(x);
  const int   y0   = static_cast<int>(y);
  const int   x1   = std::min(x0 + 1, cols - 1);
  const int   y1   = std::min(y0 + 1, rows - 1);
  const float fx   = x - static_cast<float>(x0);
  const float fy   = y - static_cast<float>(y0);
  return (1 - fy) * ((1 - fx) * img(y0, x0) + fx * img(y0, x1)) +
         fy * ((1 - fx) * img(y1, x0) + fx * img(y1, x1));
}

/** A lightweight, read-only view over a grayscale CImage buffer: caches the
 *  base pointer, row stride and dimensions so each pixel access is plain
 *  pointer arithmetic (no per-call getWidth()/ptr<>() dispatch). This is the
 *  hot path of the LK tracker (millions of samples per frame). */
struct GrayView
{
  const uint8_t* data   = nullptr;
  int            stride = 0;  ///< bytes per row
  int            cols   = 0;
  int            rows   = 0;

  explicit GrayView(const mrpt::img::CImage& img)
      : data(img.ptr<uint8_t>(0, 0)),
        stride(static_cast<int>(img.getRowStride())),
        cols(static_cast<int>(img.getWidth())),
        rows(static_cast<int>(img.getHeight()))
  {
  }

  /** Bilinear sample (identical clamping/formula to sampleBilinearU8). */
  inline float at(float x, float y) const
  {
    const int      x0  = static_cast<int>(x);
    const int      y0  = static_cast<int>(y);
    const int      x1  = std::min(x0 + 1, cols - 1);
    const int      y1  = std::min(y0 + 1, rows - 1);
    const float    fx  = x - static_cast<float>(x0);
    const float    fy  = y - static_cast<float>(y0);
    const uint8_t* r0  = data + static_cast<size_t>(y0) * stride;
    const uint8_t* r1  = data + static_cast<size_t>(y1) * stride;
    const float    p00 = r0[x0];
    const float    p10 = r0[x1];
    const float    p01 = r1[x0];
    const float    p11 = r1[x1];
    return (1 - fy) * ((1 - fx) * p00 + fx * p10) + fy * ((1 - fx) * p01 + fx * p11);
  }
};

/** Track a single point from level 'prev' to 'curr' using iterative LK.
 *  On entry,  (px, py) = predicted position in curr at this level.
 *  On return, (px, py) = refined position.
 *  Returns false if tracking failed (low eigenvalue or OOB).
 */
bool lkTrackPoint(
    const mrpt::img::CImage& prev_img, const mrpt::img::CImage& curr_img, float px_prev,
    float  py_prev,  ///< point in prev
    float& px_curr, float& py_curr,  ///< in/out in curr
    const LKParams& params)
{
  const GrayView prev_view(prev_img);
  const GrayView curr_view(curr_img);
  const int      cols = prev_view.cols;
  const int      rows = prev_view.rows;
  const int      W    = params.win_size;

  // Precompute gradient patch around (px_prev, py_prev) in prev_img
  // using finite differences on the float-sampled image.
  float Gxx = 0, Gxy = 0, Gyy = 0;

  // Collect patch intensities and gradients in prev. Reuse thread-local buffers
  // to avoid a heap allocation on every (point, pyramid-level) call.
  const int                              patch_size = (2 * W + 1) * (2 * W + 1);
  static thread_local std::vector<float> Ix_patch;
  static thread_local std::vector<float> Iy_patch;
  static thread_local std::vector<float> I_prev_patch;
  Ix_patch.resize(patch_size);
  Iy_patch.resize(patch_size);
  I_prev_patch.resize(patch_size);

  int idx = 0;
  for (int dy = -W; dy <= W; ++dy)
  {
    for (int dx = -W; dx <= W; ++dx, ++idx)
    {
      const float x = px_prev + static_cast<float>(dx);
      const float y = py_prev + static_cast<float>(dy);

      if (x < 1 || x >= static_cast<float>(cols - 1) || y < 1 || y >= static_cast<float>(rows - 1))
      {
        Ix_patch[idx] = Iy_patch[idx] = I_prev_patch[idx] = 0.f;
        continue;
      }

      I_prev_patch[idx] = prev_view.at(x, y);
      // Central differences for gradient
      Ix_patch[idx] = 0.5f * (prev_view.at(x + 1, y) - prev_view.at(x - 1, y));
      Iy_patch[idx] = 0.5f * (prev_view.at(x, y + 1) - prev_view.at(x, y - 1));

      Gxx += Ix_patch[idx] * Ix_patch[idx];
      Gxy += Ix_patch[idx] * Iy_patch[idx];
      Gyy += Iy_patch[idx] * Iy_patch[idx];
    }
  }

  // Check minimum eigenvalue of G = [Gxx Gxy; Gxy Gyy]
  const float trace   = Gxx + Gyy;
  const float disc    = std::sqrt(std::max(0.f, (Gxx - Gyy) * (Gxx - Gyy) + 4.f * Gxy * Gxy));
  const float min_eig = (trace - disc) * 0.5f;
  if (min_eig < params.min_eig_threshold * static_cast<float>(patch_size))
  {
    return false;
  }

  // Iterative refinement
  const float det = Gxx * Gyy - Gxy * Gxy;
  if (std::abs(det) < 1e-12f)
  {
    return false;
  }
  const float inv_det = 1.f / det;

  const float fcols = static_cast<float>(cols - 1);
  const float frows = static_cast<float>(rows - 1);

  for (int iter = 0; iter < params.max_iters; ++iter)
  {
    // Build mismatch vector b = Σ(It * Ix, It * Iy) where It = Iprev - Icurr.
    float bx = 0, by = 0;
    idx = 0;

    // Fast path: the whole patch is inside the image, so we can skip the
    // per-pixel bounds test (identical result to the checked path below).
    const bool fully_in =
        px_curr - static_cast<float>(W) >= 0.f && px_curr + static_cast<float>(W) < fcols &&
        py_curr - static_cast<float>(W) >= 0.f && py_curr + static_cast<float>(W) < frows;
    if (fully_in)
    {
      for (int dy = -W; dy <= W; ++dy)
      {
        const float yc = py_curr + static_cast<float>(dy);
        for (int dx = -W; dx <= W; ++dx, ++idx)
        {
          const float xc = px_curr + static_cast<float>(dx);
          const float It = I_prev_patch[idx] - curr_view.at(xc, yc);
          bx += It * Ix_patch[idx];
          by += It * Iy_patch[idx];
        }
      }
    }
    else
    {
      for (int dy = -W; dy <= W; ++dy)
      {
        for (int dx = -W; dx <= W; ++dx, ++idx)
        {
          const float xc = px_curr + static_cast<float>(dx);
          const float yc = py_curr + static_cast<float>(dy);

          if (xc < 0 || xc >= fcols || yc < 0 || yc >= frows)
          {
            continue;
          }
          const float It = I_prev_patch[idx] - curr_view.at(xc, yc);
          bx += It * Ix_patch[idx];
          by += It * Iy_patch[idx];
        }
      }
    }

    // Solve 2×2: [Gxx Gxy; Gxy Gyy] · [vx; vy] = [bx; by]
    const float vx = inv_det * (Gyy * bx - Gxy * by);
    const float vy = inv_det * (Gxx * by - Gxy * bx);

    px_curr += vx;
    py_curr += vy;

    if (std::sqrt(vx * vx + vy * vy) < params.eps)
    {
      break;
    }
  }

  // Bounds check
  if (px_curr < 0 || px_curr >= cols || py_curr < 0 || py_curr >= rows) return false;

  return true;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// calcOpticalFlowPyrLK
// ---------------------------------------------------------------------------
void mola::vision::calcOpticalFlowPyrLK(
    const mrpt::img::CImage& prev, const mrpt::img::CImage& curr,
    const std::vector<mrpt::math::TPoint2Df>& prev_pts,
    std::vector<mrpt::math::TPoint2Df>& next_pts, std::vector<TrackStatus>& status,
    const LKParams& params)
{
  // Build pyramids (grayscale, levels 0=full, 1=half, …)
  mrpt::img::CImagePyramid prev_pyr, curr_pyr;
  mrpt::img::CImage        prev_gray, curr_gray;

  // Ensure grayscale
  if (prev.isColor())
    prev.grayscale(prev_gray);
  else
    prev_gray = prev;  // shallow copy

  if (curr.isColor())
    curr.grayscale(curr_gray);
  else
    curr_gray = curr;

  prev_pyr.buildPyramidFast(prev_gray, params.max_levels + 1, /*smooth=*/false);
  curr_pyr.buildPyramidFast(curr_gray, params.max_levels + 1, /*smooth=*/false);

  calcOpticalFlowPyrLK(prev_pyr, curr_pyr, prev_pts, next_pts, status, params);
}

void mola::vision::calcOpticalFlowPyrLK(
    const mrpt::img::CImagePyramid& prev_pyr, const mrpt::img::CImagePyramid& curr_pyr,
    const std::vector<mrpt::math::TPoint2Df>& prev_pts,
    std::vector<mrpt::math::TPoint2Df>& next_pts, std::vector<TrackStatus>& status,
    const LKParams& params)
{
  const int n = static_cast<int>(prev_pts.size());
  const int L = std::min(params.max_levels, static_cast<int>(prev_pyr.images.size()) - 1);

  next_pts.resize(n);
  status.resize(n, TrackStatus::OK);

  // Scale factor for top level
  const float scale_top = std::pow(2.f, static_cast<float>(L));

  // Initialize next_pts at coarsest level
  for (int i = 0; i < n; ++i) next_pts[i] = {prev_pts[i].x / scale_top, prev_pts[i].y / scale_top};

  // Coarse-to-fine
  for (int lv = L; lv >= 0; --lv)
  {
    const float scale    = std::pow(2.f, static_cast<float>(lv));
    const auto& prev_img = prev_pyr.images[lv];
    const auto& curr_img = curr_pyr.images[lv];

    for (int i = 0; i < n; ++i)
    {
      if (status[i] != TrackStatus::OK) continue;

      const float px_prev = prev_pts[i].x / scale;
      const float py_prev = prev_pts[i].y / scale;
      float       px_curr = next_pts[i].x;
      float       py_curr = next_pts[i].y;

      const bool ok = lkTrackPoint(prev_img, curr_img, px_prev, py_prev, px_curr, py_curr, params);

      if (ok)
      {
        // Propagate to next finer level (×2)
        if (lv > 0)
        {
          next_pts[i].x = px_curr * 2.f;
          next_pts[i].y = py_curr * 2.f;
        }
        else
        {
          next_pts[i].x = px_curr;
          next_pts[i].y = py_curr;
        }
      }
      else
      {
        status[i] = TrackStatus::LOST;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// fundamentalMatrixFilter
// ---------------------------------------------------------------------------
// Normalized 8-point algorithm inside RANSAC.
// Normalization: centroid-subtracted + RMS distance normalized to √2 (Hartley).
// ---------------------------------------------------------------------------
namespace
{
/** Normalize a set of 2D points (centroid + scale) for numerical stability.
 *  Returns the 3×3 normalization matrix T such that normalized = T · [x y 1]ᵀ.
 */
Eigen::Matrix3f normalizePoints(
    const std::vector<mrpt::math::TPoint2Df>& pts, std::vector<Eigen::Vector2f>& out_pts)
{
  float mx = 0, my = 0;
  int   n = static_cast<int>(pts.size());
  for (const auto& p : pts)
  {
    mx += p.x;
    my += p.y;
  }
  mx /= n;
  my /= n;

  float rms = 0;
  for (const auto& p : pts)
  {
    rms += (p.x - mx) * (p.x - mx) + (p.y - my) * (p.y - my);
  }
  rms               = std::sqrt(rms / n);
  const float scale = (rms > 1e-8f) ? std::sqrt(2.f) / rms : 1.f;

  out_pts.resize(n);
  for (int i = 0; i < n; ++i)
  {
    out_pts[i].x() = (pts[i].x - mx) * scale;
    out_pts[i].y() = (pts[i].y - my) * scale;
  }

  Eigen::Matrix3f T;
  T << scale, 0, -scale * mx, 0, scale, -scale * my, 0, 0, 1.f;
  return T;
}

/** Estimate F from 8 (or more) normalized correspondences via SVD. */
Eigen::Matrix3f estimateF8pt(
    const std::vector<Eigen::Vector2f>& n1, const std::vector<Eigen::Vector2f>& n2,
    const std::vector<int>& indices)
{
  Eigen::MatrixXf A(static_cast<int>(indices.size()), 9);
  for (int i = 0; i < static_cast<int>(indices.size()); ++i)
  {
    const int   k  = indices[i];
    const float x1 = n1[k].x(), y1 = n1[k].y();
    const float x2 = n2[k].x(), y2 = n2[k].y();
    A.row(i) << x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.f;
  }

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
  // The null-space vector f is ordered row-major to match the constraint row
  // [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1] = vec_row_major(F).
  // Fill F explicitly row-major (Eigen's reshaped() defaults to column-major,
  // which would yield Fᵀ).
  const Eigen::Matrix<float, 9, 1> f = svd.matrixV().col(8);
  Eigen::Matrix3f                  Fraw;
  Fraw << f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8);

  // Enforce rank-2 constraint
  Eigen::JacobiSVD<Eigen::Matrix3f> svd2(Fraw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3f                   sig = svd2.singularValues();
  sig(2)                                = 0.f;
  return svd2.matrixU() * sig.asDiagonal() * svd2.matrixV().transpose();
}

/** Sampson distance: approximate epipolar error for a correspondence. */
inline float sampsonDistance(
    const Eigen::Matrix3f& F, const Eigen::Vector2f& p1, const Eigen::Vector2f& p2)
{
  Eigen::Vector3f       x1(p1.x(), p1.y(), 1.f);
  Eigen::Vector3f       x2(p2.x(), p2.y(), 1.f);
  const float           num  = x2.dot(F * x1);
  const Eigen::Vector3f Fx1  = F * x1;
  const Eigen::Vector3f Ftx2 = F.transpose() * x2;
  const float           denom =
      Fx1.x() * Fx1.x() + Fx1.y() * Fx1.y() + Ftx2.x() * Ftx2.x() + Ftx2.y() * Ftx2.y();
  return (denom > 1e-12f) ? (num * num / denom) : std::numeric_limits<float>::max();
}

}  // anonymous namespace

void mola::vision::fundamentalMatrixFilter(
    const std::vector<mrpt::math::TPoint2Df>& prev_pts,
    const std::vector<mrpt::math::TPoint2Df>& next_pts, std::vector<TrackStatus>& status,
    const FMatrixFilterParams& params)
{
  // Collect indices of OK correspondences
  std::vector<int> ok_indices;
  for (int i = 0; i < static_cast<int>(status.size()); ++i)
    if (status[i] == TrackStatus::OK) ok_indices.push_back(i);

  if (static_cast<int>(ok_indices.size()) < 8)
  {
    for (int i : ok_indices) status[i] = TrackStatus::OUTLIER;
    return;
  }

  // Normalize
  std::vector<Eigen::Vector2f> n1, n2;
  const Eigen::Matrix3f        T1 = normalizePoints(prev_pts, n1);
  const Eigen::Matrix3f        T2 = normalizePoints(next_pts, n2);

  // Threshold in normalized space (roughly scale-invariant)
  // We use the original pixel threshold directly with Sampson distance
  // in original coordinates, so we denormalize F later.
  const float thr2 = params.ransac_threshold * params.ransac_threshold;

  const int n = static_cast<int>(ok_indices.size());
  // RANSAC iterations via Bernoulli model
  const int max_iters = std::min(
      params.max_iters,
      static_cast<int>(
          std::log(1.0 - params.confidence) /
          std::log(1.0 - std::pow(0.5, 8.0))));  // assume 50% inlier rate worst-case

  std::mt19937                       rng(42);
  std::uniform_int_distribution<int> dist(0, n - 1);

  int             best_inliers = 0;
  Eigen::Matrix3f best_F;

  for (int iter = 0; iter < max_iters; ++iter)
  {
    // Sample 8 unique indices from ok_indices
    std::vector<int> sample(8);
    for (int s = 0; s < 8; ++s)
    {
      int  idx;
      bool duplicate;
      do
      {
        idx       = ok_indices[dist(rng)];
        duplicate = false;
        for (int k = 0; k < s; ++k)
          if (sample[k] == idx)
          {
            duplicate = true;
            break;
          }
      } while (duplicate);
      sample[s] = idx;
    }

    // Estimate F in normalized coordinates
    Eigen::Matrix3f F_norm = estimateF8pt(n1, n2, sample);
    // Denormalize: F = T2ᵀ · F_norm · T1
    Eigen::Matrix3f F = T2.transpose() * F_norm * T1;

    // Count inliers
    int inliers = 0;
    for (int i : ok_indices)
    {
      Eigen::Vector2f p1(prev_pts[i].x, prev_pts[i].y);
      Eigen::Vector2f p2(next_pts[i].x, next_pts[i].y);
      if (sampsonDistance(F, p1, p2) < thr2) ++inliers;
    }

    if (inliers > best_inliers)
    {
      best_inliers = inliers;
      best_F       = F;
    }
  }

  // Mark outliers using best_F
  for (int i : ok_indices)
  {
    Eigen::Vector2f p1(prev_pts[i].x, prev_pts[i].y);
    Eigen::Vector2f p2(next_pts[i].x, next_pts[i].y);
    if (sampsonDistance(best_F, p1, p2) >= thr2) status[i] = TrackStatus::OUTLIER;
  }
}
