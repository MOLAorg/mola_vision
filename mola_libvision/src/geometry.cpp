/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/geometry.h>
#include <mrpt/img/camera_geometry.h>  // undistort_point()

#include <Eigen/LU>  // determinant()
#include <Eigen/SVD>
#include <cmath>
#include <limits>
#include <random>

using namespace mola::vision;

// ---------------------------------------------------------------------------
// undistortPoints
// ---------------------------------------------------------------------------
void mola::vision::undistortPoints(
    const std::vector<mrpt::math::TPoint2Df>& pixels, const mrpt::img::TCamera& camera,
    std::vector<mrpt::math::TPoint2Df>& undistorted)
{
  // This is a thin float facade over MRPT 3.x's batch
  // camera_geometry::undistort_points_to_unit_plane(), which removes lens
  // distortion (plumb_bob or Kannala-Brandt, per TCamera) and unprojects to
  // the z=1 normalized plane. We keep the mola::vision facade because the
  // SLAM pipeline works with std::vector<TPoint2Df> features; MRPT returns
  // TPoint2D (double).
  std::vector<mrpt::img::TPixelCoordf> distorted(pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i)
  {
    distorted[i].x = pixels[i].x;
    distorted[i].y = pixels[i].y;
  }

  std::vector<mrpt::math::TPoint2D> normalized;
  mrpt::img::camera_geometry::undistort_points_to_unit_plane(distorted, normalized, camera);

  undistorted.resize(normalized.size());
  for (size_t i = 0; i < normalized.size(); ++i)
  {
    undistorted[i].x = static_cast<float>(normalized[i].x);
    undistorted[i].y = static_cast<float>(normalized[i].y);
  }
}

// ---------------------------------------------------------------------------
// triangulatePoints  (DLT, batch)
// ---------------------------------------------------------------------------
// For each point pair, build the 4×4 DLT system:
//   A = [ x1*(P1[2]ᵀ) - P1[0]ᵀ ]   (row from view 1, x component)
//       [ y1*(P1[2]ᵀ) - P1[1]ᵀ ]   (row from view 1, y component)
//       [ x2*(P2[2]ᵀ) - P2[0]ᵀ ]
//       [ y2*(P2[2]ᵀ) - P2[1]ᵀ ]
// X = last column of V in SVD(A).
// ---------------------------------------------------------------------------
void mola::vision::triangulatePoints(
    const std::vector<mrpt::math::TPoint2Df>& pts1, const std::vector<mrpt::math::TPoint2Df>& pts2,
    const Eigen::Matrix<float, 3, 4>& P1, const Eigen::Matrix<float, 3, 4>& P2,
    std::vector<mrpt::math::TPoint3Df>& out_pts3d, std::vector<bool>& out_valid)
{
  const int n = static_cast<int>(pts1.size());
  out_pts3d.resize(n);
  out_valid.resize(n, false);

  for (int i = 0; i < n; ++i)
  {
    Eigen::Matrix4f A;
    A.row(0) = pts1[i].x * P1.row(2) - P1.row(0);
    A.row(1) = pts1[i].y * P1.row(2) - P1.row(1);
    A.row(2) = pts2[i].x * P2.row(2) - P2.row(0);
    A.row(3) = pts2[i].y * P2.row(2) - P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f                   X = svd.matrixV().col(3);

    if (std::abs(X(3)) < 1e-8f) continue;
    X /= X(3);

    // Check positive depth in both cameras: z > 0
    const Eigen::Vector3f Xw = X.head<3>();
    const Eigen::Vector3f X1 = P1.leftCols<3>() * Xw + P1.col(3);
    const Eigen::Vector3f X2 = P2.leftCols<3>() * Xw + P2.col(3);
    if (X1(2) <= 0 || X2(2) <= 0) continue;

    out_pts3d[i] = {Xw(0), Xw(1), Xw(2)};
    out_valid[i] = true;
  }
}

std::optional<mrpt::math::TPoint3Df> mola::vision::triangulateSinglePoint(
    const mrpt::math::TPoint2Df& pt1, const mrpt::math::TPoint2Df& pt2,
    const Eigen::Matrix<float, 3, 3>& R, const Eigen::Vector3f& t)
{
  // P1 = [I | 0],  P2 = [R | t]
  Eigen::Matrix<float, 3, 4> P1, P2;
  P1.leftCols<3>() = Eigen::Matrix3f::Identity();
  P1.col(3)        = Eigen::Vector3f::Zero();
  P2.leftCols<3>() = R;
  P2.col(3)        = t;

  std::vector<mrpt::math::TPoint2Df> v1{pt1}, v2{pt2};
  std::vector<mrpt::math::TPoint3Df> pts3d;
  std::vector<bool>                  valid;
  triangulatePoints(v1, v2, P1, P2, pts3d, valid);

  if (!valid[0]) return std::nullopt;
  return pts3d[0];
}

// ---------------------------------------------------------------------------
// essentialFromFundamental
// ---------------------------------------------------------------------------
Eigen::Matrix3f mola::vision::essentialFromFundamental(
    const Eigen::Matrix3f& F, const Eigen::Matrix3f& K1, const Eigen::Matrix3f& K2)
{
  return K2.transpose() * F * K1;
}

// ---------------------------------------------------------------------------
// decomposeEssentialMatrix
// ---------------------------------------------------------------------------
bool mola::vision::decomposeEssentialMatrix(
    const Eigen::Matrix3f& E, const std::vector<mrpt::math::TPoint2Df>& pts1_norm,
    const std::vector<mrpt::math::TPoint2Df>& pts2_norm, Eigen::Matrix3f& R_out,
    Eigen::Vector3f& t_out)
{
  // E = U · diag(1,1,0) · Vᵀ
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f                   U = svd.matrixU();
  Eigen::Matrix3f                   V = svd.matrixV();

  // Ensure det = +1
  if (U.determinant() < 0) U.col(2) *= -1;
  if (V.determinant() < 0) V.col(2) *= -1;

  Eigen::Matrix3f W;
  W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

  // Four candidate solutions
  const Eigen::Matrix3f R1 = U * W * V.transpose();
  const Eigen::Matrix3f R2 = U * W.transpose() * V.transpose();
  const Eigen::Vector3f t1 = U.col(2);
  const Eigen::Vector3f t2 = -U.col(2);

  // Chirality test: most 3D points positive depth in both cameras
  const std::array<std::pair<Eigen::Matrix3f, Eigen::Vector3f>, 4> candidates = {
      {{R1, t1}, {R1, t2}, {R2, t1}, {R2, t2}}};

  int             best_count = -1;
  Eigen::Matrix3f best_R;
  Eigen::Vector3f best_t;

  for (const auto& [R, t] : candidates)
  {
    Eigen::Matrix<float, 3, 4> P1, P2;
    P1.leftCols<3>() = Eigen::Matrix3f::Identity();
    P1.col(3)        = Eigen::Vector3f::Zero();
    P2.leftCols<3>() = R;
    P2.col(3)        = t;

    std::vector<mrpt::math::TPoint3Df> pts3d;
    std::vector<bool>                  valid;
    triangulatePoints(pts1_norm, pts2_norm, P1, P2, pts3d, valid);

    int cnt = 0;
    for (auto v : valid)
      if (v) ++cnt;

    if (cnt > best_count)
    {
      best_count = cnt;
      best_R     = R;
      best_t     = t;
    }
  }

  if (best_count < 10) return false;

  R_out = best_R;
  t_out = best_t;
  return true;
}

// ---------------------------------------------------------------------------
// estimateEssentialRANSAC
// ---------------------------------------------------------------------------
namespace
{
/** Hartley isotropic normalization (centroid + RMS-to-sqrt(2) scale). Returns
 *  the 3x3 conditioning matrix T with normalized = T * [x y 1]^T. */
Eigen::Matrix3f conditionPoints(
    const std::vector<mrpt::math::TPoint2Df>& pts, std::vector<Eigen::Vector2f>& out)
{
  const int n  = static_cast<int>(pts.size());
  float     mx = 0.f;
  float     my = 0.f;
  for (const auto& p : pts)
  {
    mx += p.x;
    my += p.y;
  }
  mx /= static_cast<float>(n);
  my /= static_cast<float>(n);

  float rms = 0.f;
  for (const auto& p : pts)
  {
    rms += (p.x - mx) * (p.x - mx) + (p.y - my) * (p.y - my);
  }
  rms               = std::sqrt(rms / static_cast<float>(n));
  const float scale = (rms > 1e-12f) ? std::sqrt(2.f) / rms : 1.f;

  out.resize(n);
  for (int i = 0; i < n; ++i)
  {
    out[i].x() = (pts[i].x - mx) * scale;
    out[i].y() = (pts[i].y - my) * scale;
  }

  Eigen::Matrix3f T;
  T << scale, 0.f, -scale * mx, 0.f, scale, -scale * my, 0.f, 0.f, 1.f;
  return T;
}

/** Project a 3x3 matrix onto the essential manifold (singular values {1,1,0}). */
Eigen::Matrix3f projectToEssential(const Eigen::Matrix3f& M)
{
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3f                   s(1.f, 1.f, 0.f);
  return svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();
}

/** Eight-point essential estimate from conditioned correspondences (indices into
 *  n1/n2), returned in the ORIGINAL normalized frame via T2^T * E_cond * T1. */
Eigen::Matrix3f eightPointEssential(
    const std::vector<Eigen::Vector2f>& n1, const std::vector<Eigen::Vector2f>& n2,
    const std::vector<int>& idx, const Eigen::Matrix3f& T1, const Eigen::Matrix3f& T2)
{
  Eigen::MatrixXf A(static_cast<int>(idx.size()), 9);
  for (int i = 0; i < static_cast<int>(idx.size()); ++i)
  {
    const int   k  = idx[i];
    const float x1 = n1[k].x();
    const float y1 = n1[k].y();
    const float x2 = n2[k].x();
    const float y2 = n2[k].y();
    A.row(i) << x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.f;
  }

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
  const Eigen::Matrix<float, 9, 1>  f = svd.matrixV().col(8);
  Eigen::Matrix3f                   Econd;
  Econd << f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8);

  // De-condition, then enforce the essential constraints.
  const Eigen::Matrix3f E = T2.transpose() * Econd * T1;
  return projectToEssential(E);
}

/** Squared Sampson distance for a correspondence under E. */
float sampson2(const Eigen::Matrix3f& E, const Eigen::Vector2f& p1, const Eigen::Vector2f& p2)
{
  const Eigen::Vector3f x1(p1.x(), p1.y(), 1.f);
  const Eigen::Vector3f x2(p2.x(), p2.y(), 1.f);
  const float           num  = x2.dot(E * x1);
  const Eigen::Vector3f Ex1  = E * x1;
  const Eigen::Vector3f Etx2 = E.transpose() * x2;
  const float           den =
      Ex1.x() * Ex1.x() + Ex1.y() * Ex1.y() + Etx2.x() * Etx2.x() + Etx2.y() * Etx2.y();
  return (den > 1e-15f) ? (num * num / den) : std::numeric_limits<float>::max();
}
}  // namespace

EssentialRANSACResult mola::vision::estimateEssentialRANSAC(
    const std::vector<mrpt::math::TPoint2Df>& pts1_norm,
    const std::vector<mrpt::math::TPoint2Df>& pts2_norm, const EssentialRANSACParams& params)
{
  EssentialRANSACResult result;
  const int             n = static_cast<int>(pts1_norm.size());
  if (n < 8 || static_cast<int>(pts2_norm.size()) != n)
  {
    return result;  // success stays false
  }

  std::vector<Eigen::Vector2f> n1;
  std::vector<Eigen::Vector2f> n2;
  const Eigen::Matrix3f        T1 = conditionPoints(pts1_norm, n1);
  const Eigen::Matrix3f        T2 = conditionPoints(pts2_norm, n2);

  const float thr2 = params.threshold * params.threshold;

  std::mt19937                       rng(params.seed);
  std::uniform_int_distribution<int> pick(0, n - 1);

  int             best_inliers = 0;
  Eigen::Matrix3f best_E       = Eigen::Matrix3f::Zero();

  int max_iters = std::max(1, params.max_iters);
  for (int iter = 0; iter < max_iters; ++iter)
  {
    // Sample 8 unique indices.
    std::vector<int> sample;
    sample.reserve(8);
    while (static_cast<int>(sample.size()) < 8)
    {
      const int idx = pick(rng);
      bool      dup = false;
      for (const int s : sample)
      {
        if (s == idx)
        {
          dup = true;
          break;
        }
      }
      if (!dup)
      {
        sample.push_back(idx);
      }
    }

    const Eigen::Matrix3f E = eightPointEssential(n1, n2, sample, T1, T2);

    int inliers = 0;
    for (int i = 0; i < n; ++i)
    {
      const Eigen::Vector2f p1(pts1_norm[i].x, pts1_norm[i].y);
      const Eigen::Vector2f p2(pts2_norm[i].x, pts2_norm[i].y);
      if (sampson2(E, p1, p2) < thr2)
      {
        ++inliers;
      }
    }

    if (inliers > best_inliers)
    {
      best_inliers = inliers;
      best_E       = E;

      // Adaptive iteration cap from the current inlier ratio.
      const double w = static_cast<double>(inliers) / static_cast<double>(n);
      if (w > 0.0)
      {
        const double p8    = std::pow(w, 8.0);
        const double denom = std::log(1.0 - std::min(0.999999, p8));
        if (denom < 0.0)
        {
          const int adaptive =
              static_cast<int>(std::ceil(std::log(1.0 - params.confidence) / denom));
          max_iters = std::min(max_iters, std::max(1, adaptive));
        }
      }
    }
  }

  if (best_inliers < 8)
  {
    return result;  // no usable consensus
  }

  // Final refit over all inliers of the best model.
  std::vector<int> inlier_idx;
  inlier_idx.reserve(best_inliers);
  result.inliers.assign(n, false);
  for (int i = 0; i < n; ++i)
  {
    const Eigen::Vector2f p1(pts1_norm[i].x, pts1_norm[i].y);
    const Eigen::Vector2f p2(pts2_norm[i].x, pts2_norm[i].y);
    if (sampson2(best_E, p1, p2) < thr2)
    {
      inlier_idx.push_back(i);
      result.inliers[i] = true;
    }
  }

  if (static_cast<int>(inlier_idx.size()) >= 8)
  {
    best_E = eightPointEssential(n1, n2, inlier_idx, T1, T2);
    // Re-classify inliers under the refined model.
    result.inliers.assign(n, false);
    int refined = 0;
    for (int i = 0; i < n; ++i)
    {
      const Eigen::Vector2f p1(pts1_norm[i].x, pts1_norm[i].y);
      const Eigen::Vector2f p2(pts2_norm[i].x, pts2_norm[i].y);
      if (sampson2(best_E, p1, p2) < thr2)
      {
        result.inliers[i] = true;
        ++refined;
      }
    }
    best_inliers = refined;
  }

  result.E           = best_E;
  result.num_inliers = best_inliers;
  result.success     = true;
  return result;
}
