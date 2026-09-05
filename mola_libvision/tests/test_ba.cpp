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

// ---------------------------------------------------------------------------
// A landmark seen once, near the optical axis of an axis-aligned camera, must
// not destabilize the solve.
//
// Its 3x3 information block is rank 2 - depth along the viewing ray is
// unobservable from one view without a stereo disparity - and when that ray
// happens to line up with a world axis, the corresponding DIAGONAL entry of the
// block is ~0 too. Damping that is purely multiplicative in diag(JtJ) then adds
// nothing at all there, and the near-singular inverse propagates through the
// Schur complement into the reduced POSE system, so one such landmark wrecks the
// update for every pose in the window.
//
// This is not a contrived configuration: a VO front-end takes its world frame
// from the first camera, so its optical axis IS a world axis, and freshly
// spawned landmarks start life with a single observation.
// ---------------------------------------------------------------------------
TEST(BundleAdjustment, SingleViewLandmarkOnTheOpticalAxis)
{
  const auto   cam = makeCamera();
  const double fx = cam.fx(), fy = cam.fy(), cx = cam.cx(), cy = cam.cy();

  // Three cameras, all axis-aligned (R = I), sliding along +x.
  std::vector<mrpt::poses::CPose3D> gtPoses;
  for (int i = 0; i < 3; ++i)
  {
    const Eigen::Vector3d C(0.4 * i, 0, 0);
    gtPoses.push_back(makePose(Eigen::Matrix3d::Identity(), -C));
  }
  const int freeKf = 2;

  std::vector<mrpt::math::TPoint3Df> gtLandmarks;
  std::vector<BAObservation>         obs;

  auto addObs = [&](int lm, int kf)
  {
    const Eigen::Vector3d Xw(gtLandmarks[lm].x, gtLandmarks[lm].y, gtLandmarks[lm].z);
    const Eigen::Matrix3d R = gtPoses[kf].getRotationMatrix().asEigen();
    const Eigen::Vector3d t(gtPoses[kf].x(), gtPoses[kf].y(), gtPoses[kf].z());
    const Eigen::Vector3d Xc = R * Xw + t;
    BAObservation         o;
    o.kf_index = kf;
    o.lm_index = lm;
    o.pixel    = {
           static_cast<float>(fx * Xc.x() / Xc.z() + cx),
           static_cast<float>(fy * Xc.y() / Xc.z() + cy)};
    obs.push_back(o);
  };

  // A well-observed background, seen by all three cameras.
  std::mt19937                           rng(99);
  std::uniform_real_distribution<double> ux(-1.5, 1.5), uy(-1.2, 1.2), uz(3.0, 6.0);
  for (int l = 0; l < 30; ++l)
  {
    gtLandmarks.push_back(
        {static_cast<float>(ux(rng)), static_cast<float>(uy(rng)), static_cast<float>(uz(rng))});
    for (int kf = 0; kf < 3; ++kf)
    {
      addObs(static_cast<int>(gtLandmarks.size()) - 1, kf);
    }
  }
  // ...plus landmarks straight down the free camera's optical axis, seen ONCE.
  const double Cx = 0.4 * freeKf;
  for (int l = 0; l < 6; ++l)
  {
    gtLandmarks.push_back({static_cast<float>(Cx), 0.f, static_cast<float>(4.0 + 0.3 * l)});
    addObs(static_cast<int>(gtLandmarks.size()) - 1, freeKf);
  }

  auto poses     = gtPoses;
  auto landmarks = gtLandmarks;
  poses[freeKf]  = makePose(
       poses[freeKf].getRotationMatrix().asEigen(),
       Eigen::Vector3d(poses[freeKf].x() + 0.01, poses[freeKf].y() - 0.008, poses[freeKf].z()));

  const std::vector<bool> fixed = {true, true, false};

  BAOptions opts;
  opts.max_iters = 30;
  BAResult res   = slidingWindowBA(poses, landmarks, obs, cam, fixed, opts);

  EXPECT_LT(res.final_cost, 0.05 * res.initial_cost) << "the free pose must be recovered";
  EXPECT_LT((poses[freeKf].translation() - gtPoses[freeKf].translation()).norm(), 0.01);
  for (const auto& lm : landmarks)
  {
    EXPECT_TRUE(std::isfinite(lm.x) && std::isfinite(lm.y) && std::isfinite(lm.z));
    EXPECT_LT(std::abs(lm.z), 1e3) << "landmark flew off along its unobservable ray";
  }
}

// ---------------------------------------------------------------------------
// Landmarks seen from a single viewpoint must not destabilize the solve.
//
// Their 3x3 information block is rank 2 (depth along the viewing ray is
// unobservable without a second view or a stereo disparity). Damping that is
// purely multiplicative in diag(JtJ) leaves that direction unregularized, and
// the resulting near-singular inverse propagates through the Schur complement
// into the reduced POSE system - so one such landmark is enough to wreck the
// update for every pose in the window. A sliding window always contains some,
// since freshly spawned landmarks start with exactly one observation.
// ---------------------------------------------------------------------------
TEST(BundleAdjustment, SingleViewLandmarksStayStable)
{
  auto prob = makeProblem(3);

  auto poses     = prob.gtPoses;
  auto landmarks = prob.gtLandmarks;

  // Keep only the LAST keyframe's observation for every second landmark, so
  // half the map is observed exactly once.
  const int                  lastKf = poses.empty() ? 0 : static_cast<int>(poses.size()) - 1;
  std::vector<BAObservation> obs;
  for (const auto& o : prob.obs)
  {
    if ((o.lm_index % 2) == 1 && o.kf_index != lastKf)
    {
      continue;
    }
    obs.push_back(o);
  }

  std::mt19937                           rng(4242);
  std::uniform_real_distribution<double> trN(-0.01, 0.01), lmN(-0.02, 0.02);
  for (size_t i = 2; i < poses.size(); ++i)
  {
    Eigen::Matrix3d R = poses[i].getRotationMatrix().asEigen();
    Eigen::Vector3d t(poses[i].x(), poses[i].y(), poses[i].z());
    t += Eigen::Vector3d(trN(rng), trN(rng), trN(rng));
    poses[i] = makePose(R, t);
  }
  for (auto& lm : landmarks)
  {
    lm.x += static_cast<float>(lmN(rng));
    lm.y += static_cast<float>(lmN(rng));
    lm.z += static_cast<float>(lmN(rng));
  }

  std::vector<bool> fixed(poses.size(), false);
  fixed[0] = true;
  if (poses.size() > 1)
  {
    fixed[1] = true;
  }

  BAOptions opts;
  opts.max_iters = 30;
  BAResult res   = slidingWindowBA(poses, landmarks, obs, prob.cam, fixed, opts);

  EXPECT_LT(res.final_cost, res.initial_cost) << "BA must not make the fit worse";
  for (const auto& p : poses)
  {
    EXPECT_TRUE(std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z()));
  }
  for (const auto& lm : landmarks)
  {
    EXPECT_TRUE(std::isfinite(lm.x) && std::isfinite(lm.y) && std::isfinite(lm.z));
    EXPECT_LT(std::hypot(std::hypot(lm.x, lm.y), lm.z), 1e4) << "landmark flew off to infinity";
  }
  // The multi-view half of the map still has to be recovered properly.
  for (size_t i = 2; i < poses.size(); ++i)
  {
    EXPECT_LT((poses[i].translation() - prob.gtPoses[i].translation()).norm(), 0.03)
        << "pose " << i;
  }
}
