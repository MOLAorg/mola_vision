/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/geometry.h>

#include <Eigen/Geometry>
#include <cmath>
#include <random>

using mola::vision::estimateEssentialRANSAC;
using mrpt::math::TPoint2Df;

namespace
{
// Builds a synthetic two-view scene: random 3D points in front of both cameras,
// camera 1 at the origin, camera 2 at a known relative pose (R, t). Returns the
// normalized (z=1) projections in each view.
struct Scene
{
  Eigen::Matrix3f        R;
  Eigen::Vector3f        t;  // unit norm
  std::vector<TPoint2Df> n1, n2;
};

Scene makeScene(int n_points, double yaw_deg, const Eigen::Vector3f& baseline)
{
  Scene        s;
  const double yaw = yaw_deg * M_PI / 180.0;
  s.R = Eigen::AngleAxisf(static_cast<float>(yaw), Eigen::Vector3f::UnitY()).toRotationMatrix();
  s.t = baseline.normalized();

  std::mt19937                          rng(123);
  std::uniform_real_distribution<float> ux(-1.5f, 1.5f);
  std::uniform_real_distribution<float> uy(-1.0f, 1.0f);
  std::uniform_real_distribution<float> uz(3.0f, 7.0f);

  for (int i = 0; i < n_points; ++i)
  {
    const Eigen::Vector3f X(ux(rng), uy(rng), uz(rng));  // world == cam1 frame
    const Eigen::Vector3f Xc2 = s.R * X + s.t;
    if (X.z() < 0.5f || Xc2.z() < 0.5f)
    {
      continue;
    }
    s.n1.push_back({X.x() / X.z(), X.y() / X.z()});
    s.n2.push_back({Xc2.x() / Xc2.z(), Xc2.y() / Xc2.z()});
  }
  return s;
}

double rotAngleDeg(const Eigen::Matrix3f& A, const Eigen::Matrix3f& B)
{
  const float c = ((A.transpose() * B).trace() - 1.f) / 2.f;
  return std::acos(std::max(-1.f, std::min(1.f, c))) * 180.0 / M_PI;
}
}  // namespace

TEST(EssentialRANSAC, RecoversPoseClean)
{
  const auto s = makeScene(80, 10.0, Eigen::Vector3f(1.f, 0.2f, 0.1f));
  ASSERT_GE(s.n1.size(), 60u);

  const auto res = estimateEssentialRANSAC(s.n1, s.n2);
  ASSERT_TRUE(res.success);
  EXPECT_GT(res.num_inliers, static_cast<int>(s.n1.size()) * 0.9);

  Eigen::Matrix3f R;
  Eigen::Vector3f t;
  ASSERT_TRUE(mola::vision::decomposeEssentialMatrix(res.E, s.n1, s.n2, R, t));

  // Rotation recovered within 1 degree.
  EXPECT_LT(rotAngleDeg(R, s.R), 1.0);
  // Translation direction recovered (E is scale/sign ambiguous; chirality fixes
  // the sign): |cos angle| ~ 1.
  const float cosang = std::abs(t.normalized().dot(s.t));
  EXPECT_GT(cosang, 0.99f);
}

TEST(EssentialRANSAC, RobustToOutliers)
{
  auto s = makeScene(100, 8.0, Eigen::Vector3f(1.f, 0.f, 0.f));
  ASSERT_GE(s.n1.size(), 80u);

  // Corrupt 25% of view-2 matches with random garbage.
  std::mt19937                          rng(7);
  std::uniform_real_distribution<float> noise(-0.8f, 0.8f);
  const int                             n_out = static_cast<int>(s.n2.size()) / 4;
  for (int i = 0; i < n_out; ++i)
  {
    s.n2[i].x += noise(rng);
    s.n2[i].y += noise(rng);
  }

  const auto res = estimateEssentialRANSAC(s.n1, s.n2);
  ASSERT_TRUE(res.success);

  // The corrupted matches should be flagged as outliers; the rest as inliers.
  int flagged_out = 0;
  for (int i = 0; i < n_out; ++i)
  {
    if (!res.inliers[i])
    {
      ++flagged_out;
    }
  }
  EXPECT_GE(flagged_out, n_out * 0.8);

  Eigen::Matrix3f R;
  Eigen::Vector3f t;
  ASSERT_TRUE(mola::vision::decomposeEssentialMatrix(res.E, s.n1, s.n2, R, t));
  EXPECT_LT(rotAngleDeg(R, s.R), 2.0);
}

TEST(EssentialRANSAC, TooFewPointsFails)
{
  std::vector<TPoint2Df> a(5);
  std::vector<TPoint2Df> b(5);
  const auto             res = estimateEssentialRANSAC(a, b);
  EXPECT_FALSE(res.success);
}
