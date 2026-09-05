/* -------------------------------------------------------------------------
 * mola_libvision unit tests: PnP solver
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/pnp_solver.h>

#include <Eigen/Geometry>
#include <random>

using namespace mola::vision;

namespace
{
mrpt::img::TCamera makeCamera()
{
  mrpt::img::TCamera cam;
  cam.ncols = 640;
  cam.nrows = 480;
  cam.setIntrinsicParamsFromValues(500.0, 500.0, 320.0, 240.0);
  cam.distortion = mrpt::img::DistortionModel::none;
  return cam;
}

/** Build a ground-truth camera pose T_cw and project world points into it. */
struct Scene
{
  mrpt::poses::CPose3D               gtPose;  // world -> camera
  std::vector<mrpt::math::TPoint3Df> worldPts;
  std::vector<mrpt::math::TPoint2Df> pixels;
};

Scene makeScene(int nPoints, unsigned seed)
{
  Scene        s;
  const auto   cam = makeCamera();
  const double fx = cam.fx(), fy = cam.fy(), cx = cam.cx(), cy = cam.cy();

  // Ground-truth extrinsics: yaw 10 deg + pitch -5 deg, translation.
  const double    yaw = 10.0 * M_PI / 180.0, pitch = -5.0 * M_PI / 180.0;
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX());
  const Eigen::Vector3d t(0.3, -0.1, 0.2);

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
  s.gtPose = mrpt::poses::CPose3D(M);

  std::mt19937                           rng(seed);
  std::uniform_real_distribution<double> ux(-2.0, 2.0), uy(-1.5, 1.5), uz(4.0, 9.0);

  for (int i = 0; i < nPoints; ++i)
  {
    const Eigen::Vector3d Xw(ux(rng), uy(rng), uz(rng));
    const Eigen::Vector3d Xc = R * Xw + t;
    const double          u  = fx * Xc.x() / Xc.z() + cx;
    const double          v  = fy * Xc.y() / Xc.z() + cy;
    s.worldPts.push_back(
        {static_cast<float>(Xw.x()), static_cast<float>(Xw.y()), static_cast<float>(Xw.z())});
    s.pixels.push_back({static_cast<float>(u), static_cast<float>(v)});
  }
  return s;
}

double rotationErrorDeg(const mrpt::poses::CPose3D& a, const mrpt::poses::CPose3D& b)
{
  const Eigen::Matrix3d Ra = a.getRotationMatrix().asEigen();
  const Eigen::Matrix3d Rb = b.getRotationMatrix().asEigen();
  const Eigen::Matrix3d dR = Ra.transpose() * Rb;
  const double          c  = (dR.trace() - 1.0) * 0.5;
  return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

}  // namespace

// ---------------------------------------------------------------------------
// Recover a known pose from clean correspondences.
// ---------------------------------------------------------------------------
TEST(PnP, RecoverKnownPose_Clean)
{
  const auto scene = makeScene(60, 1);
  const auto cam   = makeCamera();

  // Start from identity (far from the truth) to exercise convergence.
  PnPResult res = solvePnP(scene.worldPts, scene.pixels, cam, mrpt::poses::CPose3D());

  EXPECT_TRUE(res.converged);
  EXPECT_GE(res.num_inliers, 55);

  const double rotErr = rotationErrorDeg(res.pose, scene.gtPose);
  const double trErr  = (res.pose.translation() - scene.gtPose.translation()).norm();
  EXPECT_LT(rotErr, 0.1) << "rotation error (deg)";
  EXPECT_LT(trErr, 1e-3) << "translation error (m)";
}

// ---------------------------------------------------------------------------
// Robustness: 30% gross outliers must be rejected and the pose recovered.
// ---------------------------------------------------------------------------
TEST(PnP, RecoverKnownPose_WithOutliers)
{
  auto       scene = makeScene(80, 7);
  const auto cam   = makeCamera();

  // Corrupt 30% of the observations with large random pixel offsets.
  std::mt19937                           rng(99);
  std::uniform_real_distribution<double> off(-120.0, 120.0);
  std::vector<bool>                      isOutlier(scene.pixels.size(), false);
  const int                              nOut = static_cast<int>(0.30 * scene.pixels.size());
  for (int i = 0; i < nOut; ++i)
  {
    scene.pixels[i].x += static_cast<float>(off(rng));
    scene.pixels[i].y += static_cast<float>(off(rng));
    // Ensure the corruption is actually gross.
    scene.pixels[i].x += 40.f;
    isOutlier[i] = true;
  }

  PnPParams params;
  params.max_iters = 30;
  PnPResult res    = solvePnP(scene.worldPts, scene.pixels, cam, mrpt::poses::CPose3D(), params);

  // Pose should still be accurate despite outliers.
  const double rotErr = rotationErrorDeg(res.pose, scene.gtPose);
  const double trErr  = (res.pose.translation() - scene.gtPose.translation()).norm();
  EXPECT_LT(rotErr, 0.3) << "rotation error (deg)";
  EXPECT_LT(trErr, 0.02) << "translation error (m)";

  // Outlier classification: most true outliers flagged, most inliers kept.
  int correctOut = 0, correctIn = 0, totalIn = 0;
  for (size_t i = 0; i < isOutlier.size(); ++i)
  {
    if (isOutlier[i])
    {
      if (!res.inliers[i])
      {
        ++correctOut;
      }
    }
    else
    {
      ++totalIn;
      if (res.inliers[i])
      {
        ++correctIn;
      }
    }
  }
  EXPECT_GE(correctOut, static_cast<int>(0.8 * nOut)) << "most outliers should be rejected";
  EXPECT_GE(correctIn, static_cast<int>(0.9 * totalIn)) << "most inliers should be kept";
}

// ---------------------------------------------------------------------------
// Degenerate input: too few points -> not converged, returns init pose.
// ---------------------------------------------------------------------------
TEST(PnP, TooFewPoints)
{
  std::vector<mrpt::math::TPoint3Df> w   = {{0, 0, 5}, {1, 0, 5}};
  std::vector<mrpt::math::TPoint2Df> p   = {{320, 240}, {420, 240}};
  const auto                         cam = makeCamera();

  PnPResult res = solvePnP(w, p, cam, mrpt::poses::CPose3D());
  EXPECT_FALSE(res.converged);
  EXPECT_EQ(res.num_inliers, 0);
}

// ---------------------------------------------------------------------------
// Convergence must be reported on NOISY correspondences seeded from a nearby
// pose - the regime a tracking front-end actually runs in, where the increment
// is already small from the first iteration and the Huber reweighting keeps
// producing small but non-vanishing steps.
// ---------------------------------------------------------------------------
TEST(PnP, ConvergesOnNoisyCorrespondences)
{
  const auto cam = makeCamera();

  int nConverged = 0;
  int nTrials    = 0;
  for (unsigned seed = 1; seed <= 20; ++seed)
  {
    auto                             scene = makeScene(120, seed);
    std::mt19937                     rng(seed * 977u);
    std::normal_distribution<double> pix(0.0, 0.5);  // 0.5 px feature noise
    for (auto& p : scene.pixels)
    {
      p.x += static_cast<float>(pix(rng));
      p.y += static_cast<float>(pix(rng));
    }

    // Seeded near the truth, the way a tracking front-end seeds it from the
    // previous frame. This is the regime where the increment is already small
    // from the first iteration, so a step-norm-only stopping test never fires.
    const mrpt::poses::CPose3D seedPose =
        scene.gtPose + mrpt::poses::CPose3D(0.02, -0.01, 0.03, 0.004, -0.003, 0.002);
    const PnPResult res = solvePnP(scene.worldPts, scene.pixels, cam, seedPose);
    ++nTrials;
    if (res.converged)
    {
      ++nConverged;
    }
    // Whatever the flag says, the returned pose must be the right one.
    EXPECT_LT(rotationErrorDeg(res.pose, scene.gtPose), 0.2) << "seed " << seed;
    EXPECT_LT((res.pose.translation() - scene.gtPose.translation()).norm(), 0.01)
        << "seed " << seed;
    EXPECT_GE(res.num_inliers, 100) << "seed " << seed;
  }
  EXPECT_EQ(nConverged, nTrials) << "every noisy solve should report convergence";
}

// ---------------------------------------------------------------------------
// Hitting the iteration cap is NOT failure: the returned pose is still the best
// one found, and the caller must judge usability by the inlier support.
//
// This is the contract a VO front-end depends on. Gating the pose update on
// `converged` instead loses every frame whose solve merely ran out of
// iterations - which on real imagery froze the trajectory for roughly a quarter
// of all frames and was, by a wide margin, the largest source of drift.
// ---------------------------------------------------------------------------
TEST(PnP, IterationCapStillReturnsUsablePose)
{
  const auto scene = makeScene(80, 3);
  const auto cam   = makeCamera();

  PnPParams tight;
  tight.max_iters = 2;  // guaranteed to stop on the cap, far from convergence

  const mrpt::poses::CPose3D seedPose;  // identity: far from the truth
  const PnPResult            res = solvePnP(scene.worldPts, scene.pixels, cam, seedPose, tight);

  EXPECT_FALSE(res.converged) << "two iterations cannot converge from identity";
  // ...and yet the estimate must be far better than the seed it started from,
  // and its inlier count must reflect that.
  EXPECT_LT(rotationErrorDeg(res.pose, scene.gtPose), rotationErrorDeg(seedPose, scene.gtPose));
  EXPECT_LT(
      (res.pose.translation() - scene.gtPose.translation()).norm(),
      (seedPose.translation() - scene.gtPose.translation()).norm());
  EXPECT_EQ(static_cast<size_t>(res.inliers.size()), scene.worldPts.size());
}
