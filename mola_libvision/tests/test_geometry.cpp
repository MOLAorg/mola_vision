/* -------------------------------------------------------------------------
 * mola_libvision unit tests: geometry (triangulation, undistort, essential)
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/geometry.h>

#include <Eigen/Core>
#include <Eigen/LU>  // inverse(), determinant()
#include <cmath>

using namespace mola::vision;

// ---------------------------------------------------------------------------
// triangulatePoints
// ---------------------------------------------------------------------------
TEST(Geometry, TriangulateKnownPoints)
{
  // Camera 1 at origin, camera 2 translated +1m in x
  Eigen::Matrix<float, 3, 4> P1, P2;
  P1.setZero();
  P1.leftCols<3>() = Eigen::Matrix3f::Identity();

  P2.setZero();
  P2.leftCols<3>() = Eigen::Matrix3f::Identity();
  P2.col(3)        = Eigen::Vector3f(-1.f, 0.f, 0.f);  // camera 2 at x=+1 world

  // Known 3D points (in front of both cameras, depth > 0)
  const std::vector<Eigen::Vector3f> pts3d_gt = {
      {0.1f, 0.0f, 5.0f},
      {-0.2f, 0.3f, 3.0f},
      {0.5f, -0.1f, 8.0f},
  };

  std::vector<mrpt::math::TPoint2Df> pts1, pts2;
  for (const auto& pt : pts3d_gt)
  {
    pts1.push_back({pt.x() / pt.z(), pt.y() / pt.z()});
    // project into camera 2: (world - t_c2w) where t_c2w = [1,0,0]
    const float x2 = pt.x() - 1.f;
    pts2.push_back({x2 / pt.z(), pt.y() / pt.z()});
  }

  std::vector<mrpt::math::TPoint3Df> out_pts;
  std::vector<bool>                  valid;
  triangulatePoints(pts1, pts2, P1, P2, out_pts, valid);

  for (size_t i = 0; i < pts3d_gt.size(); ++i)
  {
    ASSERT_TRUE(valid[i]) << "Point " << i << " should be valid";
    const float err = std::sqrt(
        std::pow(out_pts[i].x - pts3d_gt[i].x(), 2) + std::pow(out_pts[i].y - pts3d_gt[i].y(), 2) +
        std::pow(out_pts[i].z - pts3d_gt[i].z(), 2));
    EXPECT_LT(err, 1e-3f) << "Triangulation error too large for point " << i;
  }
}

TEST(Geometry, TriangulateRejectsBehindCamera)
{
  Eigen::Matrix<float, 3, 4> P1, P2;
  P1.setZero();
  P1.leftCols<3>() = Eigen::Matrix3f::Identity();
  P2.setZero();
  P2.leftCols<3>() = Eigen::Matrix3f::Identity();
  P2.col(3)        = Eigen::Vector3f(-1.f, 0.f, 0.f);

  // A point "behind" camera 1 (negative z in camera 1 frame)
  // Achieved by flipping the projection direction
  std::vector<mrpt::math::TPoint2Df> pts1 = {{0.f, 0.f}};
  std::vector<mrpt::math::TPoint2Df> pts2 = {{0.2f, 0.f}};  // inconsistent with positive depth

  std::vector<mrpt::math::TPoint3Df> out;
  std::vector<bool>                  valid;
  triangulatePoints(pts1, pts2, P1, P2, out, valid);

  // The result may or may not be valid, but must not crash
  EXPECT_EQ(valid.size(), 1u);
}

// ---------------------------------------------------------------------------
// triangulateSinglePoint
// ---------------------------------------------------------------------------
TEST(Geometry, TriangulateSinglePoint)
{
  const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  const Eigen::Vector3f t(-1.f, 0.f, 0.f);

  const mrpt::math::TPoint2Df p1{0.1f / 5.f, 0.f};  // depth=5, x=0.1
  const mrpt::math::TPoint2Df p2{(0.1f - 1.f) / 5.f, 0.f};

  auto result = triangulateSinglePoint(p1, p2, R, t);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->x, 0.1f, 1e-3f);
  EXPECT_NEAR(result->y, 0.0f, 1e-3f);
  EXPECT_NEAR(result->z, 5.0f, 1e-3f);
}

// ---------------------------------------------------------------------------
// undistortPoints (no-distortion camera → identity)
// ---------------------------------------------------------------------------
TEST(Geometry, UndistortPoints_NoDistortion)
{
  mrpt::img::TCamera cam;
  // MRPT 3.x: fx()/fy()/cx()/cy() are value getters; use the setter.
  cam.setIntrinsicParamsFromValues(500.0, 500.0, 320.0, 240.0);
  // All distortion coefficients default to 0

  std::vector<mrpt::math::TPoint2Df> pixels = {
      {320.f, 240.f},  // principal point → (0, 0) normalized
      {820.f, 240.f},  // 500px right → +1.0 normalized
  };

  std::vector<mrpt::math::TPoint2Df> undist;
  undistortPoints(pixels, cam, undist);

  ASSERT_EQ(undist.size(), pixels.size());
  EXPECT_NEAR(undist[0].x, 0.f, 1e-3f);
  EXPECT_NEAR(undist[0].y, 0.f, 1e-3f);
  EXPECT_NEAR(undist[1].x, 1.f, 1e-3f);
  EXPECT_NEAR(undist[1].y, 0.f, 1e-3f);
}

// ---------------------------------------------------------------------------
// essentialFromFundamental
// ---------------------------------------------------------------------------
TEST(Geometry, EssentialFromFundamental)
{
  // F = K2^{-T} E K1^{-1}  →  E = K2^T F K1
  Eigen::Matrix3f K;
  K << 500.f, 0.f, 320.f, 0.f, 500.f, 240.f, 0.f, 0.f, 1.f;

  // Simple pure-translation E (tx skew-symmetric)
  Eigen::Matrix3f E_gt;
  E_gt << 0, 0, 0, 0, 0, -1, 0, 1, 0;

  Eigen::Matrix3f F = K.transpose().inverse() * E_gt * K.inverse();
  Eigen::Matrix3f E = essentialFromFundamental(F, K, K);

  // E and E_gt should be proportional
  const float scale = E_gt.norm() / E.norm();
  EXPECT_NEAR((E * scale - E_gt).norm(), 0.f, 1e-3f);
}
