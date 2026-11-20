/* -------------------------------------------------------------------------
 * mola_libvision unit tests: distortion-aware PnP, focal initialization and
 * bundle adjustment with intrinsics as variables
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/bundle_adjustment.h>
#include <mola_libvision/focal_initialization.h>
#include <mola_libvision/lie_utils.h>
#include <mola_libvision/pnp_solver.h>

#include <Eigen/Dense>
#include <map>
#include <random>

using namespace mola::vision;

namespace
{
mrpt::poses::CPose3D makePose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
  mrpt::math::CMatrixDouble44 M;
  M.setIdentity();
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      M(r, c) = R(r, c);
    }
    M(r, 3) = t(r);
  }
  return mrpt::poses::CPose3D(M);
}

Eigen::Matrix3d rotOf(const mrpt::poses::CPose3D& p) { return p.getRotationMatrix().asEigen(); }
Eigen::Vector3d trOf(const mrpt::poses::CPose3D& p) { return {p.x(), p.y(), p.z()}; }

/** Cameras moving along a curve and rotating, looking at a cloud of points. */
struct SyntheticSequence
{
  CameraModel                       camera;
  std::vector<mrpt::poses::CPose3D> poses;  // T_cw
  std::vector<Eigen::Vector3d>      points;
  std::vector<BAObservationPx>      obs;
};

SyntheticSequence makeSequence(
    const CameraModel& cam, int n_poses, int n_points, double noise_px, unsigned seed,
    double rotation_deg = 15.0, double translation = 1.0, double wide = 1.0)
{
  SyntheticSequence s;
  s.camera = cam;
  std::mt19937                           rng(seed);
  std::uniform_real_distribution<double> ux(-4.0 * wide, 4.0 * wide);
  std::uniform_real_distribution<double> uz(3.0, 12.0);
  std::normal_distribution<double>       nd(0.0, noise_px);
  for (int j = 0; j < n_points; ++j)
  {
    s.points.emplace_back(ux(rng), 0.75 * ux(rng), uz(rng));
  }
  const double rot = rotation_deg * M_PI / 180.0;
  for (int i = 0; i < n_poses; ++i)
  {
    const double          u = static_cast<double>(i) / std::max(1, n_poses - 1);
    const Eigen::Vector3d w(
        rot * std::sin(2 * M_PI * u), rot * std::cos(3 * M_PI * u), 0.3 * rot * u);
    const Eigen::Matrix3d R_wc = so3Exp(w);
    const Eigen::Vector3d C(
        translation * (2 * u - 1), 0.3 * translation * std::sin(4 * u), 0.5 * translation * u);
    const Eigen::Matrix3d R_cw = R_wc.transpose();
    s.poses.push_back(makePose(R_cw, -R_cw * C));
  }
  for (size_t i = 0; i < s.poses.size(); ++i)
  {
    const Eigen::Matrix3d R = rotOf(s.poses[i]);
    const Eigen::Vector3d t = trOf(s.poses[i]);
    for (size_t j = 0; j < s.points.size(); ++j)
    {
      Eigen::Vector2d px;
      if (!cam.project(R * s.points[j] + t, px))
      {
        continue;
      }
      if (px.x() < 0 || px.y() < 0 || px.x() > cam.width() - 1 || px.y() > cam.height() - 1)
      {
        continue;
      }
      BAObservationPx o;
      o.pose  = static_cast<int>(i);
      o.point = static_cast<int>(j);
      o.pixel = px + Eigen::Vector2d(nd(rng), nd(rng));
      s.obs.push_back(o);
    }
  }
  return s;
}

std::vector<ViewPairMatches> pairsOf(const SyntheticSequence& s, int stride)
{
  std::vector<ViewPairMatches> out;
  for (size_t a = 0; a + static_cast<size_t>(stride) < s.poses.size();
       a += static_cast<size_t>(stride))
  {
    const size_t                   b = a + static_cast<size_t>(stride);
    std::map<int, Eigen::Vector2d> pa;
    for (const auto& o : s.obs)
    {
      if (o.pose == static_cast<int>(a))
      {
        pa[o.point] = o.pixel;
      }
    }
    ViewPairMatches m;
    for (const auto& o : s.obs)
    {
      if (o.pose == static_cast<int>(b) && pa.count(o.point) != 0)
      {
        m.p1.push_back(pa[o.point]);
        m.p2.push_back(o.pixel);
      }
    }
    out.push_back(m);
  }
  return out;
}

/** Perturbs poses and points of a sequence (keeping pose 0 and the scale
 *  reference roughly in place) to start a bundle adjustment away from the
 *  solution. */
BundleAdjustmentProblem perturbedProblem(
    const SyntheticSequence& s, const CameraModel& init_cam, unsigned seed, double sigma_rot = 0.01,
    double sigma_t = 0.03, double sigma_pt = 0.1)
{
  std::mt19937                     rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  BundleAdjustmentProblem          p;
  p.cameras = {init_cam};
  for (size_t i = 0; i < s.poses.size(); ++i)
  {
    if (i == 0)
    {
      p.poses.push_back(s.poses[i]);
      continue;
    }
    const Eigen::Matrix3d dR = so3Exp(sigma_rot * Eigen::Vector3d(nd(rng), nd(rng), nd(rng)));
    p.poses.push_back(makePose(
        dR * rotOf(s.poses[i]),
        trOf(s.poses[i]) + sigma_t * Eigen::Vector3d(nd(rng), nd(rng), nd(rng))));
  }
  for (const auto& X : s.points)
  {
    p.points.push_back(X + sigma_pt * Eigen::Vector3d(nd(rng), nd(rng), nd(rng)));
  }
  p.observations = s.obs;
  return p;
}
}  // namespace

TEST(SelfCalibration, PnPWithDistortion)
{
  for (const auto& cam :
       {CameraModel::RadTan(752, 480, 458.6, 457.3, 367.2, 248.4, -0.28, 0.07, 2e-4, 2e-5, 0.0),
        CameraModel::Equidistant(1280, 1024, 350, 350, 640, 512, -0.01, 0.02, -0.01, 0.002)})
  {
    const auto                             s = makeSequence(cam, 2, 300, 0.3, 5);
    std::vector<Eigen::Vector3d>           X;
    std::vector<Eigen::Vector2d>           z;
    std::mt19937                           rng(9);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    int                                    k = 0;
    for (const auto& o : s.obs)
    {
      if (o.pose != 1)
      {
        continue;
      }
      X.push_back(s.points[static_cast<size_t>(o.point)]);
      z.push_back(o.pixel);
      if (++k % 5 == 0)
      {
        z.back() = Eigen::Vector2d(u(rng) * cam.width(), u(rng) * cam.height());  // outlier
      }
    }
    RansacOptions ro;
    ro.threshold    = 2.0;
    const auto  res = solvePnPRansac(X, z, cam, ro);
    const auto& gt  = s.poses[1];
    EXPECT_LT((rotOf(res.pose) - rotOf(gt)).norm(), 2e-3);
    EXPECT_LT((trOf(res.pose) - trOf(gt)).norm(), 3e-2);
    EXPECT_GT(res.num_inliers, static_cast<int>(0.75 * X.size()));
  }
}

TEST(SelfCalibration, FocalFromFundamentalsAndConsensus)
{
  const auto cam   = CameraModel::Pinhole(640, 480, 500, 500, 319.5, 239.5);
  const auto s     = makeSequence(cam, 25, 400, 0.3, 6, 20.0, 1.5);
  const auto pairs = pairsOf(s, 4);

  std::vector<Eigen::Matrix3d> Fs;
  for (const auto& m : pairs)
  {
    RansacOptions ro;
    ro.threshold = 1.0;
    const auto F = estimateFundamental(m.p1, m.p2, ro);
    if (F.report.success)
    {
      Fs.push_back(F.F);
    }
  }
  ASSERT_GE(Fs.size(), 3u);
  const auto mc = estimateFocalFromFundamentals(Fs, 319.5, 239.5, 150, 3000);
  ASSERT_TRUE(mc.valid);
  EXPECT_NEAR(mc.focal, 500.0, 25.0);

  FocalSweepOptions fo;
  fo.model     = CameraModelType::Pinhole;
  fo.width     = 640;
  fo.height    = 480;
  const auto c = estimateFocalByTwoViewConsensus(pairs, fo);
  ASSERT_TRUE(c.valid);
  EXPECT_NEAR(c.focal, 500.0, 40.0);
  EXPECT_LT(c.flatness, 0.8);
}

TEST(SelfCalibration, FocalConsensusFisheye)
{
  const auto        cam = CameraModel::Equidistant(1024, 1024, 290, 290, 511.5, 511.5, 0, 0, 0, 0);
  const auto        s   = makeSequence(cam, 25, 800, 0.3, 7, 20.0, 1.5, 3.0);
  FocalSweepOptions fo;
  fo.model     = CameraModelType::Equidistant;
  fo.width     = 1024;
  fo.height    = 1024;
  const auto c = estimateFocalByTwoViewConsensus(pairsOf(s, 4), fo);
  ASSERT_TRUE(c.valid);
  EXPECT_NEAR(c.focal, 290.0, 25.0);
}

TEST(SelfCalibration, FocalFromRotationHomographies)
{
  std::mt19937    rng(8);
  Eigen::Matrix3d K;
  K << 700, 0, 400, 0, 700, 300, 0, 0, 1;
  std::vector<Eigen::Matrix3d> Hs;
  for (int k = 0; k < 6; ++k)
  {
    std::normal_distribution<double> nd(0.0, 0.15);
    const Eigen::Matrix3d            R = so3Exp(Eigen::Vector3d(nd(rng), nd(rng), nd(rng)));
    Hs.push_back(K * R * K.inverse());
  }
  const auto e = estimateFocalFromRotationHomographies(Hs, 400, 300, 200, 3000);
  ASSERT_TRUE(e.valid);
  EXPECT_NEAR(e.focal, 700.0, 1.0);
}

TEST(SelfCalibration, BundleAdjustmentRecoversRadTanIntrinsics)
{
  const auto gt =
      CameraModel::RadTan(752, 480, 458.6, 457.3, 367.2, 248.4, -0.28, 0.07, 2e-4, 2e-5, 0.0);
  const auto s = makeSequence(gt, 30, 600, 0.5, 31, 20.0, 1.5, 1.5);

  // Start: 8% wrong focal, image-center principal point, no distortion.
  const auto init = CameraModel::RadTan(752, 480, 495, 495, 375.5, 239.5, 0, 0, 0, 0, 0);
  auto       p    = perturbedProblem(s, init, 3);

  BundleAdjustmentOptions o;
  IntrinsicsMask          m;
  m.estimate_distortion = IntrinsicsMask::FirstN(5, 4);  // k1 k2 p1 p2
  o.intrinsics          = {m};
  o.compute_covariance  = true;
  o.max_iterations      = 200;
  const auto r          = bundleAdjust(p, o);
  ASSERT_TRUE(r.success) << r.message;
  const auto& est = p.cameras[0];
  EXPECT_NEAR(est.fx(), gt.fx(), 1.5);
  EXPECT_NEAR(est.fy(), gt.fy(), 1.5);
  EXPECT_NEAR(est.cx(), gt.cx(), 2.0);
  EXPECT_NEAR(est.cy(), gt.cy(), 2.0);
  EXPECT_NEAR(est.params()[4], gt.params()[4], 0.01);

  // Residuals at the noise level.
  double sq = 0;
  int    n  = 0;
  for (double e : r.residuals)
  {
    if (std::isfinite(e))
    {
      sq += e * e;
      ++n;
    }
  }
  EXPECT_LT(std::sqrt(sq / n), 0.5 * std::sqrt(2.0) * 1.1);

  // The covariance is consistent with the actual error (4 sigma).
  ASSERT_EQ(r.intrinsics_covariance.size(), 1u) << r.message;
  const Eigen::MatrixXd& C = r.intrinsics_covariance[0];
  for (int k = 0; k < 4; ++k)
  {
    const double sigma = std::sqrt(C(k, k));
    EXPECT_GT(sigma, 0.0);
    EXPECT_LT(std::abs(est.params()[k] - gt.params()[k]), 4.0 * sigma + 1e-3) << "param " << k;
  }
}

TEST(SelfCalibration, BundleAdjustmentRecoversEquidistantIntrinsics)
{
  const auto gt =
      CameraModel::Equidistant(1024, 1024, 290, 291, 515, 509, -0.02, 0.01, -0.005, 0.001);
  const auto s    = makeSequence(gt, 30, 1500, 0.5, 32, 25.0, 1.5, 3.0);
  const auto init = CameraModel::Equidistant(1024, 1024, 310, 310, 511.5, 511.5, 0, 0, 0, 0);
  auto       p    = perturbedProblem(s, init, 4);

  BundleAdjustmentOptions o;
  IntrinsicsMask          m;
  m.estimate_distortion = IntrinsicsMask::FirstN(4, 4);
  o.intrinsics          = {m};
  o.max_iterations      = 200;
  const auto r          = bundleAdjust(p, o);
  ASSERT_TRUE(r.success) << r.message;
  const auto& est = p.cameras[0];
  EXPECT_NEAR(est.fx(), gt.fx(), 1.5);
  EXPECT_NEAR(est.fy(), gt.fy(), 1.5);
  EXPECT_NEAR(est.cx(), gt.cx(), 1.5);
  EXPECT_NEAR(est.cy(), gt.cy(), 1.5);
}

TEST(SelfCalibration, RotationOnlyWithPointsAtInfinity)
{
  // A camera spinning about its center sees every point as a direction.
  const auto gt   = CameraModel::RadTan(640, 480, 520, 520, 321, 242, -0.15, 0.02, 0, 0, 0);
  auto       s    = makeSequence(gt, 20, 500, 0.3, 33, 25.0, 0.0);
  const auto init = CameraModel::RadTan(640, 480, 480, 480, 319.5, 239.5, 0, 0, 0, 0, 0);
  auto       p    = perturbedProblem(s, init, 5, 0.01, 0.0, 0.0);
  p.point_at_infinity.assign(p.points.size(), true);
  for (auto& X : p.points)
  {
    X.normalize();
  }
  BundleAdjustmentOptions o;
  IntrinsicsMask          m;
  m.fixed_aspect_ratio  = true;
  m.estimate_distortion = IntrinsicsMask::FirstN(5, 2);
  o.intrinsics          = {m};
  o.rotation_only       = true;
  o.compute_covariance  = true;
  const auto r          = bundleAdjust(p, o);
  ASSERT_TRUE(r.success) << r.message;
  EXPECT_NEAR(p.cameras[0].fx(), 520.0, 1.0);
  EXPECT_NEAR(p.cameras[0].params()[4], -0.15, 0.01);
  ASSERT_EQ(r.free_intrinsics_covariance.size(), 1u);
  EXPECT_EQ(r.free_intrinsics_covariance[0].rows(), 5);
}
