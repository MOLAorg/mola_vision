/* -------------------------------------------------------------------------
 * mola_visual_slam: monocular / stereo visual SLAM front-end for MOLA.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Integration test: run the stereo pipeline on a KITTI odometry sequence and
 * report the (metric, SE3-aligned) ATE against the ground-truth poses. Skipped
 * unless KITTI_STEREO_DIR points at a sequence directory (with image_0/,
 * image_1/, calib.txt) and KITTI_POSES at the matching poses/NN.txt.
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_visual_slam/VisualSlam.h>
#include <mrpt/core/Clock.h>
#include <mrpt/img/CImage.h>
#include <mrpt/system/filesystem.h>

#include <Eigen/Geometry>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace
{
// Parse a KITTI calib.txt line "P0: v0 .. v11" into the 12 values.
bool readProjection(const std::string& calibPath, const std::string& key, double out[12])
{
  std::ifstream f(calibPath);
  std::string   line;
  while (std::getline(f, line))
  {
    if (line.rfind(key, 0) != 0)
    {
      continue;
    }
    std::istringstream ss(line.substr(key.size() + 1));
    for (int i = 0; i < 12; ++i)
    {
      if (!(ss >> out[i]))
      {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::string frameName(int i)
{
  std::ostringstream ss;
  ss << std::setw(6) << std::setfill('0') << i << ".png";
  return ss.str();
}

// KITTI poses/NN.txt: each line is a 3x4 row-major matrix; translation is
// elements [3],[7],[11].
std::vector<Eigen::Vector3d> readGtPositions(const std::string& path)
{
  std::vector<Eigen::Vector3d> out;
  std::ifstream                f(path);
  std::string                  line;
  while (std::getline(f, line))
  {
    std::istringstream ss(line);
    double             m[12];
    bool               ok = true;
    for (double& v : m)
    {
      if (!(ss >> v))
      {
        ok = false;
        break;
      }
    }
    if (ok)
    {
      out.emplace_back(m[3], m[7], m[11]);
    }
  }
  return out;
}
}  // namespace

TEST(VisualSlamKITTI, StereoAteSeq00)
{
  const char* envDir = std::getenv("KITTI_STEREO_DIR");
  const char* envPos = std::getenv("KITTI_POSES");
  if (!envDir || !envPos)
  {
    GTEST_SKIP() << "Set KITTI_STEREO_DIR (sequence dir) and KITTI_POSES to run.";
  }
  const std::string dir   = envDir;
  const std::string calib = dir + "/calib.txt";
  if (!mrpt::system::fileExists(calib))
  {
    GTEST_SKIP() << "calib.txt not found in " << dir;
  }

  double P0[12];
  double P1[12];
  ASSERT_TRUE(readProjection(calib, "P0", P0));
  ASSERT_TRUE(readProjection(calib, "P1", P1));
  const double fx       = P0[0];
  const double fy       = P0[5];
  const double cx       = P0[2];
  const double cy       = P0[6];
  const double baseline = -P1[3] / fx;  // P1[3] = -fx * baseline
  std::cout << "[KITTI stereo] fx=" << fx << " baseline=" << baseline << " m\n";
  ASSERT_GT(baseline, 0.3);

  size_t max_frames = 300;
  if (const char* mf = std::getenv("KITTI_MAX_FRAMES"))
  {
    max_frames = static_cast<size_t>(std::atol(mf));
  }

  mola::VisualSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_WARN);

  mrpt::img::TCamera cam;  // filled from the first image size
  bool               camReady = false;

  std::vector<Eigen::Vector3d> est;
  std::vector<int>             est_frame;

  for (size_t i = 0; i < max_frames; ++i)
  {
    const std::string lp = dir + "/image_0/" + frameName(static_cast<int>(i));
    const std::string rp = dir + "/image_1/" + frameName(static_cast<int>(i));
    if (!mrpt::system::fileExists(lp))
    {
      break;
    }
    mrpt::img::CImage left;
    mrpt::img::CImage right;
    ASSERT_TRUE(left.loadFromFile(lp)) << lp;
    ASSERT_TRUE(right.loadFromFile(rp)) << rp;
    if (!camReady)
    {
      cam.ncols = static_cast<uint32_t>(left.getWidth());
      cam.nrows = static_cast<uint32_t>(left.getHeight());
      cam.setIntrinsicParamsFromValues(fx, fy, cx, cy);
      camReady = true;
    }
    const auto pose = slam.processStereoFrame(left, right, cam, baseline, mrpt::Clock::now());
    if (slam.isInitialized())
    {
      est.emplace_back(pose.x(), pose.y(), pose.z());
      est_frame.push_back(static_cast<int>(i));
    }
  }

  std::cout << "[KITTI stereo] processed frames, initialized=" << slam.isInitialized()
            << " keyframes=" << slam.numKeyframes()
            << " active landmarks=" << slam.numActiveLandmarks() << "\n";
  slam.profiler().dumpAllStats();

  ASSERT_TRUE(slam.isInitialized());
  ASSERT_GE(est.size(), 50u);

  const auto gt = readGtPositions(envPos);
  ASSERT_GE(gt.size(), est.size());

  // Associate by frame index, then SE3 (rigid, NO scaling) align - stereo is
  // already metric.
  const Eigen::Index                       m = static_cast<Eigen::Index>(est.size());
  Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, m);
  Eigen::Matrix<double, 3, Eigen::Dynamic> B(3, m);
  for (Eigen::Index k = 0; k < m; ++k)
  {
    A.col(k) = est[static_cast<size_t>(k)];
    B.col(k) = gt[static_cast<size_t>(est_frame[static_cast<size_t>(k)])];
  }
  const Eigen::Matrix4d S = Eigen::umeyama(A, B, /*with_scaling=*/false);

  double sse = 0;
  for (Eigen::Index k = 0; k < m; ++k)
  {
    const Eigen::Vector4d a(A(0, k), A(1, k), A(2, k), 1.0);
    const Eigen::Vector3d mapped = (S * a).head<3>();
    sse += (mapped - B.col(k)).squaredNorm();
  }
  const double ate = std::sqrt(sse / static_cast<double>(m));
  std::cout << "[KITTI stereo] metric ATE RMSE = " << ate << " m over " << m << " frames\n";

  // Also report scale: if stereo metric is right, the optimal scale ~1.0.
  const Eigen::Matrix4d Ssc = Eigen::umeyama(A, B, /*with_scaling=*/true);
  const double          sc  = Ssc.block<3, 3>(0, 0).norm() / std::sqrt(3.0);
  std::cout << "[KITTI stereo] best-fit scale = " << sc << " (1.0 = perfectly metric)\n";

  // Loose bound: this is pure stereo VO (PnP + stereo depth), with windowed BA
  // intentionally disabled until scale-anchored stereo BA lands (task 3.4), so
  // the pose drifts over the segment. ~5 m ATE over the first 300 frames
  // (~250 m driven, ~2%) on KITTI 00. The KEY validation is the metric scale:
  EXPECT_NEAR(sc, 1.0, 0.25);  // stereo is metric (mono would be arbitrary)
  EXPECT_LT(ate, 8.0);
}
