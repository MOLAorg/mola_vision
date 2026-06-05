/* -------------------------------------------------------------------------
 * mola_rgbd_slam: RGB-D visual SLAM front-end for the MOLA framework.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_rgbd_slam/RgbdSlam.h>
#include <mrpt/core/Clock.h>
#include <mrpt/img/CImage.h>
#include <mrpt/math/CMatrixF.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/system/COutputLogger.h>

#include <cmath>

namespace
{
// Synthetic scene: a textured fronto-parallel plane at world Z = D, observed by
// a camera that translates purely along world +X (no rotation). With no
// rotation the camera-frame depth of the plane is uniform (= D - tz = D), and a
// fixed world-anchored texture induces consistent, trackable optical flow.
//
// World frame == first camera frame, so the ground-truth camera-in-world pose
// of frame k is a pure translation (k*step, 0, 0). This is exactly what
// RgbdSlam should estimate.

constexpr int   kW    = 640;
constexpr int   kH    = 480;
constexpr float kFx   = 500.f;
constexpr float kFy   = 500.f;
constexpr float kCx   = 320.f;
constexpr float kCy   = 240.f;
constexpr float kD    = 3.0f;  // plane depth in the first camera frame [m]
constexpr float kStep = 0.03f;  // per-frame camera +X translation [m]

float texture(float Xw, float Yw)
{
  // Product of sines: a grid of intensity extrema => good Shi-Tomasi corners.
  const float a = 15.0f;
  const float v = 128.f + 80.f * std::sin(a * Xw) * std::sin(a * Yw);
  return std::min(255.f, std::max(0.f, v));
}

mrpt::img::TCamera makeCamera()
{
  mrpt::img::TCamera cam;
  cam.ncols = kW;
  cam.nrows = kH;
  cam.setIntrinsicParamsFromValues(kFx, kFy, kCx, kCy);
  return cam;
}

void renderFrame(float tx, mrpt::img::CImage& gray, mrpt::math::CMatrixFloat& depth)
{
  gray.resize(kW, kH, mrpt::img::CH_GRAY);
  const float Zc = kD;  // tz = 0 => uniform plane depth
  for (int v = 0; v < kH; ++v)
  {
    for (int u = 0; u < kW; ++u)
    {
      const float Xw         = tx + Zc * (static_cast<float>(u) - kCx) / kFx;
      const float Yw         = 0.f + Zc * (static_cast<float>(v) - kCy) / kFy;
      gray.at<uint8_t>(u, v) = static_cast<uint8_t>(std::lround(texture(Xw, Yw)));
    }
  }
  depth.setSize(kH, kW);
  depth.fill(Zc);
}
}  // namespace

TEST(RgbdSlam, RecoversPureTranslation)
{
  mola::RgbdSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);

  const auto cam = makeCamera();
  const int  N   = 12;

  mrpt::poses::CPose3D last;
  for (int k = 0; k < N; ++k)
  {
    const float              tx = static_cast<float>(k) * kStep;
    mrpt::img::CImage        gray;
    mrpt::math::CMatrixFloat depth;
    renderFrame(tx, gray, depth);
    last = slam.processFrame(gray, depth, cam, mrpt::Clock::now());
  }

  // Map and keyframes were created.
  EXPECT_GT(slam.numLandmarks(), 50u);
  EXPECT_GE(slam.numKeyframes(), 1u);

  // Recovered camera-in-world translation should match ground truth.
  const float gt_x = static_cast<float>(N - 1) * kStep;
  EXPECT_NEAR(last.x(), gt_x, 0.02);  // < 2 cm
  EXPECT_NEAR(last.y(), 0.0, 0.02);
  EXPECT_NEAR(last.z(), 0.0, 0.02);

  // Rotation should stay near identity.
  EXPECT_NEAR(last.yaw(), 0.0, 0.02);
  EXPECT_NEAR(last.pitch(), 0.0, 0.02);
  EXPECT_NEAR(last.roll(), 0.0, 0.02);
}

TEST(RgbdSlam, FirstFrameInitializesMap)
{
  mola::RgbdSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);

  const auto               cam = makeCamera();
  mrpt::img::CImage        gray;
  mrpt::math::CMatrixFloat depth;
  renderFrame(0.f, gray, depth);
  const auto pose = slam.processFrame(gray, depth, cam, mrpt::Clock::now());

  // First frame defines the world frame at the origin.
  EXPECT_NEAR(pose.x(), 0.0, 1e-6);
  EXPECT_NEAR(pose.y(), 0.0, 1e-6);
  EXPECT_NEAR(pose.z(), 0.0, 1e-6);
  EXPECT_GT(slam.numLandmarks(), 50u);
  EXPECT_EQ(slam.numKeyframes(), 1u);
}
