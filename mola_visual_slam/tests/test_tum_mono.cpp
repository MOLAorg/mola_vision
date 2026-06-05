/* -------------------------------------------------------------------------
 * mola_visual_slam: monocular / stereo visual SLAM front-end for MOLA.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Integration test: run the monocular pipeline on a real TUM RGB-D sequence
 * (using only the RGB / intensity channel) and report the scale-aligned ATE
 * against the dataset ground truth. Skipped unless TUM_RAWLOG points to a
 * MRPT rawlog produced by rgbd_dataset2rawlog (and a sibling groundtruth.txt).
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_visual_slam/VisualSlam.h>
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

// Loads TUM-format groundtruth.txt lines: "timestamp tx ty tz qx qy qz qw".
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

// Nearest-time lookup within a tolerance (seconds).
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

TEST(VisualSlamTUM, MonoAteOnFreiburg1Room)
{
  const char* env = std::getenv("TUM_RAWLOG");
  if (!env)
  {
    GTEST_SKIP() << "Set TUM_RAWLOG to a TUM rgbd .rawlog to run this integration test.";
  }
  const std::string rawlog_path = env;
  if (!mrpt::system::fileExists(rawlog_path))
  {
    GTEST_SKIP() << "TUM_RAWLOG file not found: " << rawlog_path;
  }
  const std::string dir  = mrpt::system::extractFileDirectory(rawlog_path);
  const std::string stem = mrpt::system::extractFileName(rawlog_path);  // no extension
  // External images live in "<stem>_Images" beside the rawlog (rgbd_dataset2rawlog
  // convention); the stored filenames are bare, so the base is that directory.
  mrpt::img::CImage::setImagesPathBase(dir + "/" + stem + "_Images");
  const std::string gtPath = dir + "/groundtruth.txt";
  const auto        gt =
      mrpt::system::fileExists(gtPath) ? loadGroundTruth(gtPath) : std::vector<StampedPos>{};

  mrpt::obs::CRawlog rawlog;
  ASSERT_TRUE(rawlog.loadFromRawLogFile(rawlog_path)) << "Could not load " << rawlog_path;

  mola::VisualSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_WARN);

  // Optional cap on the number of processed frames (keeps runtime bounded).
  size_t max_frames = 100000;
  if (const char* mf = std::getenv("TUM_MAX_FRAMES"))
  {
    max_frames = static_cast<size_t>(std::atol(mf));
  }

  std::vector<double>          est_t;
  std::vector<Eigen::Vector3d> est_p;

  size_t n_frames = 0;
  for (size_t i = 0; i < rawlog.size() && n_frames < max_frames; ++i)
  {
    auto obs =
        std::dynamic_pointer_cast<mrpt::obs::CObservation3DRangeScan>(rawlog.getAsObservation(i));
    if (!obs)
    {
      continue;
    }
    // Monocular: load ONLY the RGB/intensity channel (the depth/points3D
    // external channels are not needed and avoid loading them).
    if (!obs->hasIntensityImage)
    {
      continue;
    }
    if (obs->intensityImage.isExternallyStored())
    {
      obs->intensityImage.forceLoad();
    }
    if (obs->intensityImage.isEmpty())
    {
      continue;
    }
    const auto pose = slam.processFrame(obs->intensityImage, obs->cameraParams, obs->timestamp);
    ++n_frames;

    if (slam.isInitialized())
    {
      est_t.push_back(mrpt::Clock::toDouble(obs->timestamp));
      est_p.push_back(Eigen::Vector3d(pose.x(), pose.y(), pose.z()));
    }
    obs->intensityImage.unload();
  }

  std::cout << "[TUM mono] processed " << n_frames
            << " frames, initialized=" << slam.isInitialized()
            << ", keyframes=" << slam.numKeyframes()
            << ", active landmarks=" << slam.numActiveLandmarks() << "\n";

  // Full per-stage profiling dump.
  slam.profiler().dumpAllStats();

  ASSERT_TRUE(slam.isInitialized()) << "Monocular bootstrap failed on the sequence.";
  ASSERT_GE(slam.numKeyframes(), 3u);

  if (gt.empty())
  {
    GTEST_SKIP() << "No groundtruth.txt; ran the pipeline but cannot compute ATE.";
  }

  // Associate estimated poses with ground truth by timestamp.
  std::vector<Eigen::Vector3d> A;  // estimated
  std::vector<Eigen::Vector3d> B;  // ground truth
  for (size_t i = 0; i < est_t.size(); ++i)
  {
    Eigen::Vector3d g;
    if (nearestGt(gt, est_t[i], 0.03, g))
    {
      A.push_back(est_p[i]);
      B.push_back(g);
    }
  }
  std::cout << "[TUM mono] associated " << A.size() << " / " << est_t.size()
            << " estimated poses with ground truth.\n";
  ASSERT_GE(A.size(), 20u) << "Too few timestamp matches for ATE.";

  // Sim(3) alignment (monocular scale is free): Umeyama with scaling.
  const Eigen::Index                       m = static_cast<Eigen::Index>(A.size());
  Eigen::Matrix<double, 3, Eigen::Dynamic> Am(3, m);
  Eigen::Matrix<double, 3, Eigen::Dynamic> Bm(3, m);
  for (Eigen::Index i = 0; i < m; ++i)
  {
    Am.col(i) = A[static_cast<size_t>(i)];
    Bm.col(i) = B[static_cast<size_t>(i)];
  }
  const Eigen::Matrix4d S = Eigen::umeyama(Am, Bm, true);

  double sse = 0;
  for (size_t i = 0; i < A.size(); ++i)
  {
    const Eigen::Vector4d a(A[i].x(), A[i].y(), A[i].z(), 1.0);
    const Eigen::Vector3d mapped = (S * a).head<3>();
    sse += (mapped - B[i]).squaredNorm();
  }
  const double ate_rmse = std::sqrt(sse / static_cast<double>(A.size()));
  std::cout << "[TUM mono] scale-aligned ATE RMSE = " << ate_rmse << " m\n";

  // Loose bound: this validates the pipeline produces a globally consistent
  // trajectory on real handheld data (freiburg1_room is a hard, rotation-heavy
  // sequence; monocular, no scale-drift mitigation or loop closure yet).
  // Observed ~1.0 m over the first 500 frames. Tighten once 2.6 + loop closure
  // land.
  EXPECT_LT(ate_rmse, 1.5);
}
