/* -------------------------------------------------------------------------
 * mola_libvision unit tests: sliding-window bundle adjustment
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/lie_utils.h>
#include <mola_libvision/sliding_window_ba.h>

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

double rotErrDeg(const mrpt::poses::CPose3D& a, const mrpt::poses::CPose3D& b)
{
  const Eigen::Matrix3d dR =
      a.getRotationMatrix().asEigen().transpose() * b.getRotationMatrix().asEigen();
  const double c = std::max(-1.0, std::min(1.0, (dR.trace() - 1.0) * 0.5));
  return std::acos(c) * 180.0 / M_PI;
}

struct Problem
{
  std::vector<mrpt::poses::CPose3D>  gtPoses;
  std::vector<mrpt::math::TPoint3Df> gtLandmarks;
  std::vector<BAObservation>         obs;
  mrpt::img::TCamera                 cam = makeCamera();
};

/** 5 cameras moving along x with small yaw, all viewing a cloud in front. */
Problem makeProblem(unsigned seed)
{
  Problem      prob;
  const auto&  cam = prob.cam;
  const double fx = cam.fx(), fy = cam.fy(), cx = cam.cx(), cy = cam.cy();

  const int nCams = 5;
  for (int i = 0; i < nCams; ++i)
  {
    // Significant parallax (good depth observability => well-conditioned BA):
    // cameras span ~2 m laterally with a vergence yaw, viewing points 3-7 m away.
    const double    yaw = (i - 2) * 4.0 * M_PI / 180.0;
    Eigen::Matrix3d R   = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()).toRotationMatrix();
    // Camera world position; T_cw translation t = -R * C.
    const Eigen::Vector3d C(0.5 * i, 0.1 * (i % 2), 0.0);
    const Eigen::Vector3d t = -R * C;
    prob.gtPoses.push_back(makePose(R, t));
  }

  std::mt19937                           rng(seed);
  std::uniform_real_distribution<double> ux(-2.0, 2.0), uy(-1.5, 1.5), uz(3.0, 7.0);
  const int                              nLm = 40;
  for (int l = 0; l < nLm; ++l)
  {
    const Eigen::Vector3d Xw(ux(rng), uy(rng), uz(rng));
    prob.gtLandmarks.push_back(
        {static_cast<float>(Xw.x()), static_cast<float>(Xw.y()), static_cast<float>(Xw.z())});

    for (int i = 0; i < nCams; ++i)
    {
      const Eigen::Matrix3d R = prob.gtPoses[i].getRotationMatrix().asEigen();
      const Eigen::Vector3d t(prob.gtPoses[i].x(), prob.gtPoses[i].y(), prob.gtPoses[i].z());
      const Eigen::Vector3d Xc = R * Xw + t;
      BAObservation         o;
      o.kf_index = i;
      o.lm_index = l;
      o.pixel    = {
             static_cast<float>(fx * Xc.x() / Xc.z() + cx),
             static_cast<float>(fy * Xc.y() / Xc.z() + cy)};
      prob.obs.push_back(o);
    }
  }
  return prob;
}

}  // namespace

// ---------------------------------------------------------------------------
// Perturb free poses + all landmarks, then recover via BA (clean observations).
// First two poses fixed -> anchors gauge AND metric scale.
// ---------------------------------------------------------------------------
TEST(BundleAdjustment, RecoverFromPerturbation)
{
  const auto prob = makeProblem(3);

  auto poses     = prob.gtPoses;
  auto landmarks = prob.gtLandmarks;

  // Perturb poses 2,3,4 and all landmarks.
  std::mt19937                           rng(123);
  std::uniform_real_distribution<double> rotN(-0.006, 0.006), trN(-0.01, 0.01), lmN(-0.02, 0.02);
  for (size_t i = 2; i < poses.size(); ++i)
  {
    Eigen::Matrix3d R = poses[i].getRotationMatrix().asEigen();
    Eigen::Vector3d t(poses[i].x(), poses[i].y(), poses[i].z());
    R = R * so3Exp(Eigen::Vector3d(rotN(rng), rotN(rng), rotN(rng)));
    t += Eigen::Vector3d(trN(rng), trN(rng), trN(rng));
    poses[i] = makePose(R, t);
  }
  for (auto& lm : landmarks)
  {
    lm.x += static_cast<float>(lmN(rng));
    lm.y += static_cast<float>(lmN(rng));
    lm.z += static_cast<float>(lmN(rng));
  }

  std::vector<bool> fixed = {true, true, false, false, false};

  BAOptions opts;
  opts.max_iters      = 50;
  opts.lambda_initial = 1e-2;
  BAResult res        = slidingWindowBA(poses, landmarks, prob.obs, prob.cam, fixed, opts);

  // BA should massively reduce the reprojection cost (clean observations).
  EXPECT_LT(res.final_cost, res.initial_cost * 0.01) << "BA should cut cost by >100x";
  EXPECT_LT(res.final_cost, 1.0) << "near-converged reprojection cost (sub-pixel RMS)";

  // Free poses recovered close to ground truth.
  for (size_t i = 2; i < poses.size(); ++i)
  {
    EXPECT_LT(rotErrDeg(poses[i], prob.gtPoses[i]), 0.3) << "pose " << i << " rotation";
    EXPECT_LT((poses[i].translation() - prob.gtPoses[i].translation()).norm(), 0.03)
        << "pose " << i << " translation";
  }

  // Most landmarks recovered close to ground truth.
  int lmOk = 0;
  for (size_t l = 0; l < landmarks.size(); ++l)
  {
    const Eigen::Vector3d e(
        landmarks[l].x - prob.gtLandmarks[l].x, landmarks[l].y - prob.gtLandmarks[l].y,
        landmarks[l].z - prob.gtLandmarks[l].z);
    if (e.norm() < 0.05)
    {
      ++lmOk;
    }
  }
  EXPECT_GE(lmOk, static_cast<int>(0.85 * landmarks.size()));
}

// ---------------------------------------------------------------------------
// Noisy observations: BA must still reduce the reprojection cost substantially.
// ---------------------------------------------------------------------------
TEST(BundleAdjustment, ReducesCostWithNoise)
{
  auto prob = makeProblem(11);

  // Add ~0.5 px Gaussian pixel noise to observations.
  std::mt19937                     rng(55);
  std::normal_distribution<double> npx(0.0, 0.5);
  for (auto& o : prob.obs)
  {
    o.pixel.x += static_cast<float>(npx(rng));
    o.pixel.y += static_cast<float>(npx(rng));
  }

  auto poses     = prob.gtPoses;
  auto landmarks = prob.gtLandmarks;
  // Perturb the free poses a little.
  std::mt19937                           rng2(7);
  std::uniform_real_distribution<double> trN(-0.03, 0.03);
  for (size_t i = 2; i < poses.size(); ++i)
  {
    Eigen::Matrix3d R = poses[i].getRotationMatrix().asEigen();
    Eigen::Vector3d t(poses[i].x(), poses[i].y(), poses[i].z());
    t += Eigen::Vector3d(trN(rng2), trN(rng2), trN(rng2));
    poses[i] = makePose(R, t);
  }

  std::vector<bool> fixed = {true, true, false, false, false};
  BAResult          res   = slidingWindowBA(poses, landmarks, prob.obs, prob.cam, fixed);

  EXPECT_LT(res.final_cost, res.initial_cost);
  EXPECT_EQ(res.num_observations_used, static_cast<int>(prob.obs.size()));
}

// ---------------------------------------------------------------------------
// Empty problem is a no-op.
// ---------------------------------------------------------------------------
TEST(BundleAdjustment, EmptyIsNoOp)
{
  std::vector<mrpt::poses::CPose3D>  poses;
  std::vector<mrpt::math::TPoint3Df> lms;
  std::vector<BAObservation>         obs;
  BAResult                           res = slidingWindowBA(poses, lms, obs, makeCamera());
  EXPECT_FALSE(res.converged);
  EXPECT_EQ(res.iterations, 0);
}
