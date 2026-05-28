/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#include <mola_libvision/geometry.h>
#include <mrpt/img/camera_geometry.h>  // undistort_point()

#include <Eigen/LU>  // determinant()
#include <Eigen/SVD>
#include <cmath>

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
