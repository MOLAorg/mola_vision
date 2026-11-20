/* -------------------------------------------------------------------------
 * mola_libvision unit tests: minimal solvers and robust estimators
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/lie_utils.h>
#include <mola_libvision/minimal_solvers.h>
#include <mola_libvision/robust_estimators.h>

#include <Eigen/Dense>
#include <random>

using namespace mola::vision;

namespace
{
struct TwoViewScene
{
  Eigen::Matrix3d              R;  // x2 = R x1 + t
  Eigen::Vector3d              t;
  std::vector<Eigen::Vector3d> X;  // in camera 1
  std::vector<Eigen::Vector3d> b1;  // unit bearings
  std::vector<Eigen::Vector3d> b2;
};

Eigen::Matrix3d randomRotation(std::mt19937& rng, double max_angle)
{
  std::normal_distribution<double>       n(0.0, 1.0);
  std::uniform_real_distribution<double> u(0.0, max_angle);
  const Eigen::Vector3d axis = Eigen::Vector3d(n(rng), n(rng), n(rng)).normalized();
  return so3Exp(axis * u(rng));
}

TwoViewScene makeScene(std::mt19937& rng, int n, bool planar = false, double baseline = 1.0)
{
  TwoViewScene                           s;
  std::normal_distribution<double>       nd(0.0, 1.0);
  std::uniform_real_distribution<double> ux(-2.0, 2.0);
  std::uniform_real_distribution<double> uz(4.0, 10.0);
  s.R = randomRotation(rng, 0.4);
  s.t = baseline * Eigen::Vector3d(nd(rng), nd(rng), 0.3 * nd(rng)).normalized();
  for (int i = 0; i < n; ++i)
  {
    Eigen::Vector3d       X(ux(rng), ux(rng), planar ? 6.0 + 0.2 * ux(rng) * 0.0 : uz(rng));
    const Eigen::Vector3d X2 = s.R * X + s.t;
    if (X2.z() <= 0.5)
    {
      --i;
      continue;
    }
    s.X.push_back(X);
    s.b1.push_back(X.normalized());
    s.b2.push_back(X2.normalized());
  }
  return s;
}

Eigen::Matrix3d trueEssential(const TwoViewScene& s)
{
  const Eigen::Matrix3d E = skew(s.t) * s.R;
  return E / E.norm();
}

double essentialDistance(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B)
{
  const Eigen::Matrix3d a = A / A.norm();
  const Eigen::Matrix3d b = B / B.norm();
  return std::min((a - b).norm(), (a + b).norm());
}
}  // namespace

TEST(MinimalSolvers, FivePointRecoversEssential)
{
  std::mt19937 rng(11);
  int          ok     = 0;
  const int    trials = 100;
  for (int k = 0; k < trials; ++k)
  {
    const auto      s = makeScene(rng, 5);
    Eigen::Vector3d a[5];
    Eigen::Vector3d b[5];
    for (int i = 0; i < 5; ++i)
    {
      a[i] = s.b1[static_cast<size_t>(i)];
      b[i] = s.b2[static_cast<size_t>(i)];
    }
    const auto sols = solveEssentialFivePoint(a, b);
    double     best = 1e9;
    for (const auto& E : sols)
    {
      best = std::min(best, essentialDistance(E, trueEssential(s)));
      // Every solution satisfies the five constraints.
      for (int i = 0; i < 5; ++i)
      {
        EXPECT_NEAR(b[i].dot(E * a[i]), 0.0, 1e-8);
      }
    }
    if (best < 1e-6)
    {
      ++ok;
    }
  }
  EXPECT_GE(ok, 97);
}

TEST(MinimalSolvers, SevenAndEightPointFundamental)
{
  std::mt19937    rng(12);
  const auto      s = makeScene(rng, 40);
  Eigen::Matrix3d K;
  K << 500, 0, 320, 0, 510, 240, 0, 0, 1;
  std::vector<Eigen::Vector2d> p1;
  std::vector<Eigen::Vector2d> p2;
  for (size_t i = 0; i < s.X.size(); ++i)
  {
    const Eigen::Vector3d a = K * s.b1[i];
    const Eigen::Vector3d b = K * s.b2[i];
    p1.emplace_back(a.x() / a.z(), a.y() / a.z());
    p2.emplace_back(b.x() / b.z(), b.y() / b.z());
  }
  const Eigen::Vector2d a[7] = {p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p1[6]};
  const Eigen::Vector2d b[7] = {p2[0], p2[1], p2[2], p2[3], p2[4], p2[5], p2[6]};
  const auto            sols = solveFundamentalSevenPoint(a, b);
  ASSERT_FALSE(sols.empty());
  double best = 1e9;
  for (const auto& F : sols)
  {
    double worst = 0;
    for (size_t i = 0; i < p1.size(); ++i)
    {
      worst = std::max(worst, sampsonDistance(F, p1[i], p2[i]));
    }
    best = std::min(best, worst);
  }
  EXPECT_LT(best, 1e-6);

  const auto F8 = solveFundamentalLinear(p1, p2);
  ASSERT_TRUE(F8.has_value());
  for (size_t i = 0; i < p1.size(); ++i)
  {
    EXPECT_LT(sampsonDistance(*F8, p1[i], p2[i]), 1e-6);
  }
}

TEST(MinimalSolvers, LinearEssentialAndDecomposition)
{
  std::mt19937 rng(13);
  const auto   s = makeScene(rng, 30);
  const auto   E = solveEssentialLinear(s.b1, s.b2);
  ASSERT_TRUE(E.has_value());
  EXPECT_LT(essentialDistance(*E, trueEssential(s)), 1e-8);
  const auto pose = relativePoseFromEssential(*E, s.b1, s.b2);
  ASSERT_TRUE(pose.has_value());
  EXPECT_EQ(pose->num_in_front, 30);
  EXPECT_LT((pose->R - s.R).norm(), 1e-8);
  EXPECT_LT((pose->t - s.t.normalized()).norm(), 1e-8);
}

TEST(MinimalSolvers, HomographyOnPlane)
{
  std::mt19937                 rng(14);
  const auto                   s = makeScene(rng, 20, true);
  std::vector<Eigen::Vector2d> p1;
  std::vector<Eigen::Vector2d> p2;
  for (size_t i = 0; i < s.X.size(); ++i)
  {
    p1.emplace_back(s.b1[i].x() / s.b1[i].z(), s.b1[i].y() / s.b1[i].z());
    p2.emplace_back(s.b2[i].x() / s.b2[i].z(), s.b2[i].y() / s.b2[i].z());
  }
  const auto H = solveHomographyDLT(p1, p2);
  ASSERT_TRUE(H.has_value());
  for (size_t i = 0; i < p1.size(); ++i)
  {
    EXPECT_LT(homographyTransferError(*H, p1[i], p2[i]), 1e-8);
  }
}

TEST(MinimalSolvers, RotationAndP3P)
{
  std::mt19937 rng(15);
  for (int k = 0; k < 50; ++k)
  {
    const auto                   s = makeScene(rng, 10);
    std::vector<Eigen::Vector3d> r2;
    for (const auto& b : s.b1)
    {
      r2.push_back(s.R * b);
    }
    const auto R = solveRotationKabsch(s.b1, r2);
    ASSERT_TRUE(R.has_value());
    EXPECT_LT((*R - s.R).norm(), 1e-10);

    // P3P: world = camera 1, bearings from camera 2.
    const Eigen::Vector3d b[3] = {s.b2[0], s.b2[1], s.b2[2]};
    const Eigen::Vector3d X[3] = {s.X[0], s.X[1], s.X[2]};
    const auto            sols = solveP3P(b, X);
    double                best = 1e9;
    for (const auto& p : sols)
    {
      best = std::min(best, (p.R - s.R).norm() + (p.t - s.t).norm());
    }
    EXPECT_LT(best, 1e-6) << "trial " << k;
  }
}

TEST(MinimalSolvers, TriangulationAndAngles)
{
  std::mt19937                 rng(16);
  const auto                   s = makeScene(rng, 10);
  std::vector<Eigen::Matrix3d> R = {Eigen::Matrix3d::Identity(), s.R};
  std::vector<Eigen::Vector3d> t = {Eigen::Vector3d::Zero(), s.t};
  for (size_t i = 0; i < s.X.size(); ++i)
  {
    const auto X = triangulateBearings(R, t, {s.b1[i], s.b2[i]});
    ASSERT_TRUE(X.has_value());
    EXPECT_LT((*X - s.X[i]).norm(), 1e-8);
  }
  // A ridiculous minimum angle rejects every point.
  EXPECT_FALSE(triangulateBearings(R, t, {s.b1[0], s.b2[0]}, 1.5).has_value());
}

TEST(MinimalSolvers, GricPrefersHomographyOnPlanes)
{
  std::mt19937                     rng(17);
  const auto                       s = makeScene(rng, 200, true);
  std::vector<Eigen::Vector2d>     p1;
  std::vector<Eigen::Vector2d>     p2;
  std::normal_distribution<double> noise(0.0, 0.5 / 500.0);
  for (size_t i = 0; i < s.X.size(); ++i)
  {
    p1.emplace_back(s.b1[i].x() / s.b1[i].z() + noise(rng), s.b1[i].y() / s.b1[i].z() + noise(rng));
    p2.emplace_back(s.b2[i].x() / s.b2[i].z() + noise(rng), s.b2[i].y() / s.b2[i].z() + noise(rng));
  }
  const auto          H = solveHomographyDLT(p1, p2);
  const auto          F = solveFundamentalLinear(p1, p2);
  std::vector<double> eh;
  std::vector<double> ef;
  for (size_t i = 0; i < p1.size(); ++i)
  {
    eh.push_back(homographyTransferError(*H, p1[i], p2[i]) / std::sqrt(2.0));
    ef.push_back(sampsonDistance(*F, p1[i], p2[i]));
  }
  const double sigma = 0.5 / 500.0;
  EXPECT_LT(gric(eh, sigma, 2, 8), gric(ef, sigma, 3, 7));
}

TEST(RobustEstimators, RelativePoseWithOutliers)
{
  std::mt19937                           rng(21);
  auto                                   s = makeScene(rng, 300);
  std::normal_distribution<double>       noise(0.0, 0.5 / 500.0);
  std::uniform_real_distribution<double> u(-0.5, 0.5);
  std::vector<bool>                      outlier(s.b1.size(), false);
  for (size_t i = 0; i < s.b1.size(); ++i)
  {
    s.b2[i] = (s.b2[i] + Eigen::Vector3d(noise(rng), noise(rng), noise(rng))).normalized();
    if (i % 3 == 0)
    {
      s.b2[i]    = Eigen::Vector3d(u(rng), u(rng), 1.0).normalized();
      outlier[i] = true;
    }
  }
  RansacOptions o;
  o.threshold    = 3.0 / 500.0;
  const auto est = estimateRelativePose(s.b1, s.b2, o);
  ASSERT_TRUE(est.report.success);
  EXPECT_LT((est.pose.R - s.R).norm(), 0.01);
  EXPECT_LT((est.pose.t - s.t.normalized()).norm(), 0.05);
  int wrong = 0;
  for (size_t i = 0; i < outlier.size(); ++i)
  {
    if (outlier[i] && est.report.inliers[i])
    {
      ++wrong;
    }
  }
  EXPECT_LT(wrong, 10);
}

TEST(RobustEstimators, AbsolutePoseWithOutliers)
{
  std::mt19937                           rng(22);
  auto                                   s = makeScene(rng, 200);
  std::uniform_real_distribution<double> u(-0.5, 0.5);
  for (size_t i = 0; i < s.b2.size(); i += 4)
  {
    s.b2[i] = Eigen::Vector3d(u(rng), u(rng), 1.0).normalized();
  }
  RansacOptions o;
  o.threshold    = 2.0 / 500.0;
  const auto est = estimateAbsolutePose(s.X, s.b2, o);
  ASSERT_TRUE(est.report.success);
  EXPECT_LT((est.pose.R - s.R).norm(), 1e-6);
  EXPECT_LT((est.pose.t - s.t).norm(), 1e-6);
  EXPECT_NEAR(est.report.num_inliers, 150, 3);
}

TEST(RobustEstimators, HomographyFundamentalRotation)
{
  std::mt19937 rng(23);
  // Pure rotation: bearings related by R only.
  auto                         s = makeScene(rng, 100);
  std::vector<Eigen::Vector3d> rb;
  for (const auto& b : s.b1)
  {
    rb.push_back(s.R * b);
  }
  RansacOptions o;
  o.threshold    = 1e-3;
  const auto rot = estimateRotation(s.b1, rb, o);
  ASSERT_TRUE(rot.report.success);
  EXPECT_EQ(rot.report.num_inliers, 100);
  EXPECT_LT((rot.R - s.R).norm(), 1e-9);

  std::vector<Eigen::Vector2d> p1;
  std::vector<Eigen::Vector2d> p2;
  for (size_t i = 0; i < s.b1.size(); ++i)
  {
    p1.emplace_back(500 * s.b1[i].x() / s.b1[i].z() + 320, 500 * s.b1[i].y() / s.b1[i].z() + 240);
    p2.emplace_back(500 * s.b2[i].x() / s.b2[i].z() + 320, 500 * s.b2[i].y() / s.b2[i].z() + 240);
  }
  RansacOptions of;
  of.threshold  = 1.0;
  const auto fe = estimateFundamental(p1, p2, of);
  ASSERT_TRUE(fe.report.success);
  EXPECT_EQ(fe.report.num_inliers, 100);

  // A general 3D scene is not explained by a single homography.
  const auto he = estimateHomography(p1, p2, of);
  EXPECT_LT(he.report.num_inliers, 60);
}
