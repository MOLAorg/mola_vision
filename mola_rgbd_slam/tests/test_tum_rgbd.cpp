/* -------------------------------------------------------------------------
 * mola_rgbd_slam: RGB-D visual SLAM front-end for the MOLA framework.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Integration test: run the RGB-D pipeline on a real TUM RGB-D sequence and
 * report the (metric, SE3-aligned) ATE against the ground truth. Skipped unless
 * TUM_RAWLOG points to an MRPT rawlog from rgbd_dataset2rawlog (with a sibling
 * groundtruth.txt and a "<stem>_Images" external-images directory).
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_rgbd_slam/RgbdSlam.h>
#include <mrpt/core/Clock.h>
#include <mrpt/img/CImage.h>
#include <mrpt/obs/CObservation3DRangeScan.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/system/filesystem.h>

#include <Eigen/Geometry>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
struct StampedPos
{
  double          t = 0;
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

std::vector<StampedPos> loadGroundTruth(const std::string& path)
{
  std::vector<StampedPos> out;
  std::ifstream           f(path);
  std::string             line;
  while (std::getline(f, line))
  {
    if (line.empty() || line[0] == '#')
    {
      continue;
    }
    std::istringstream ss(line);
    StampedPos         s;
    double             qx, qy, qz, qw;
    if (ss >> s.t >> s.p.x() >> s.p.y() >> s.p.z() >> qx >> qy >> qz >> qw)
    {
      out.push_back(s);
    }
  }
  return out;
}

bool nearestGt(const std::vector<StampedPos>& gt, double t, double tol, Eigen::Vector3d& out)
{
  double best_dt = tol;
  bool   found   = false;
  for (const auto& g : gt)
  {
    const double dt = std::abs(g.t - t);
    if (dt < best_dt)
    {
      best_dt = dt;
      out     = g.p;
      found   = true;
    }
  }
  return found;
}
}  // namespace

TEST(RgbdSlamTUM, MetricAteOnFreiburg1Room)
{
  const char* env = std::getenv("TUM_RAWLOG");
  if (!env)
  {
    GTEST_SKIP() << "Set TUM_RAWLOG to a TUM rgbd .rawlog to run this integration test.";
  }
  const std::string rawlog_path = env;
  if (!mrpt::system::fileExists(rawlog_path))
  {
    GTEST_SKIP() << "TUM_RAWLOG not found: " << rawlog_path;
  }
  const std::string dir  = mrpt::system::extractFileDirectory(rawlog_path);
  const std::string stem = mrpt::system::extractFileName(rawlog_path);
  mrpt::img::CImage::setImagesPathBase(dir + "/" + stem + "_Images");
  const std::string gtPath = dir + "/groundtruth.txt";
  const auto        gt =
      mrpt::system::fileExists(gtPath) ? loadGroundTruth(gtPath) : std::vector<StampedPos>{};

  mrpt::obs::CRawlog rawlog;
  ASSERT_TRUE(rawlog.loadFromRawLogFile(rawlog_path));

  size_t max_frames = 100000;
  if (const char* mf = std::getenv("TUM_MAX_FRAMES"))
  {
    max_frames = static_cast<size_t>(std::atol(mf));
  }

  mola::RgbdSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_WARN);

  std::vector<double>          est_t;
  std::vector<Eigen::Vector3d> est_p;
  size_t                       n_frames = 0;

  for (size_t i = 0; i < rawlog.size() && n_frames < max_frames; ++i)
  {
    auto obs =
        std::dynamic_pointer_cast<mrpt::obs::CObservation3DRangeScan>(rawlog.getAsObservation(i));
    if (!obs)
    {
      continue;
    }
    obs->load();
    if (!obs->hasRangeImage || !obs->hasIntensityImage)
    {
      continue;
    }
    mrpt::math::CMatrixFloat depth;
    depth                        = obs->rangeImage.asEigen().cast<float>() * obs->rangeUnits;
    const mrpt::img::CImage gray = obs->intensityImage.grayscale();
    const auto pose = slam.processFrame(gray, depth, obs->cameraParams, obs->timestamp);
    ++n_frames;
    est_t.push_back(mrpt::Clock::toDouble(obs->timestamp));
    est_p.emplace_back(pose.x(), pose.y(), pose.z());
    obs->unload();
  }

  std::cout << "[TUM rgbd] processed " << n_frames << " frames, keyframes=" << slam.numKeyframes()
            << " active landmarks=" << slam.numActiveLandmarks() << "\n";
  slam.profiler().dumpAllStats();

  ASSERT_GE(slam.numKeyframes(), 3u);
  if (gt.empty())
  {
    GTEST_SKIP() << "No groundtruth.txt; ran the pipeline but cannot compute ATE.";
  }

  std::vector<Eigen::Vector3d> A;
  std::vector<Eigen::Vector3d> B;
  for (size_t i = 0; i < est_t.size(); ++i)
  {
    Eigen::Vector3d g;
    if (nearestGt(gt, est_t[i], 0.03, g))
    {
      A.push_back(est_p[i]);
      B.push_back(g);
    }
  }
  std::cout << "[TUM rgbd] associated " << A.size() << " / " << est_t.size() << " poses with GT\n";
  ASSERT_GE(A.size(), 20u);

  const Eigen::Index                       m = static_cast<Eigen::Index>(A.size());
  Eigen::Matrix<double, 3, Eigen::Dynamic> Am(3, m);
  Eigen::Matrix<double, 3, Eigen::Dynamic> Bm(3, m);
  for (Eigen::Index k = 0; k < m; ++k)
  {
    Am.col(k) = A[static_cast<size_t>(k)];
    Bm.col(k) = B[static_cast<size_t>(k)];
  }

  // RGB-D is metric: SE3 (rigid, no scaling) alignment.
  const Eigen::Matrix4d S   = Eigen::umeyama(Am, Bm, /*with_scaling=*/false);
  double                sse = 0;
  for (Eigen::Index k = 0; k < m; ++k)
  {
    const Eigen::Vector4d a(Am(0, k), Am(1, k), Am(2, k), 1.0);
    sse += ((S * a).head<3>() - Bm.col(k)).squaredNorm();
  }
  const double ate = std::sqrt(sse / static_cast<double>(m));

  // Best-fit scale should be ~1.0 (metric), unlike monocular.
  const Eigen::Matrix4d Ssc = Eigen::umeyama(Am, Bm, /*with_scaling=*/true);
  const double          sc  = Ssc.block<3, 3>(0, 0).norm() / std::sqrt(3.0);
  std::cout << "[TUM rgbd] metric ATE RMSE = " << ate << " m, best-fit scale = " << sc << "\n";

  EXPECT_NEAR(sc, 1.0, 0.2);  // RGB-D is metric
  EXPECT_LT(ate, 0.5);  // freiburg1_room is hard handheld; VO only, no loop closure
}
