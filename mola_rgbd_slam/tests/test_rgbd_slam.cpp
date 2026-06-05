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
// Synthetic scene: a textured plane at world Z = D, observed by a camera moving
// on a known trajectory. The world frame == the first camera frame (which is at
// the origin looking down +Z), so the camera-in-world pose of frame 0 is the
// identity and RgbdSlam should recover the rest of the trajectory.
//
// For an arbitrary camera pose T_wc (rotation R, center C), each pixel ray is
// intersected with the plane to produce a fully consistent (image, depth) pair,
// with a fixed world-anchored texture that induces trackable optical flow.

constexpr int   kW  = 640;
constexpr int   kH  = 480;
constexpr float kFx = 500.f;
constexpr float kFy = 500.f;
constexpr float kCx = 320.f;
constexpr float kCy = 240.f;
constexpr float kD  = 3.0f;  // plane depth in the first camera frame [m]

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

// Renders the plane Z=kD as seen from camera-in-world pose `pose_wc`.
void renderPose(
    const mrpt::poses::CPose3D& pose_wc, mrpt::img::CImage& gray, mrpt::math::CMatrixFloat& depth)
{
  gray.resize(kW, kH, mrpt::img::CH_GRAY);
  depth.setSize(kH, kW);
  depth.fill(0.f);

  const auto   R  = pose_wc.getRotationMatrix();
  const double Cx = pose_wc.x();
  const double Cy = pose_wc.y();
  const double Cz = pose_wc.z();

  for (int v = 0; v < kH; ++v)
  {
    for (int u = 0; u < kW; ++u)
    {
      // Camera-frame ray direction (z = 1).
      const double dcx = (static_cast<double>(u) - kCx) / kFx;
      const double dcy = (static_cast<double>(v) - kCy) / kFy;
      const double dcz = 1.0;
      // World-frame ray direction R * dcam.
      const double dwx = R(0, 0) * dcx + R(0, 1) * dcy + R(0, 2) * dcz;
      const double dwy = R(1, 0) * dcx + R(1, 1) * dcy + R(1, 2) * dcz;
      const double dwz = R(2, 0) * dcx + R(2, 1) * dcy + R(2, 2) * dcz;
      if (std::abs(dwz) < 1e-9)
      {
        continue;
      }
      // Intersect with plane Zworld = kD: C.z + s*dwz = kD. Depth (cam frame) = s.
      const double s = (static_cast<double>(kD) - Cz) / dwz;
      if (s <= 0.1 || s > 20.0)
      {
        continue;  // behind camera or out of range: leave invalid depth
      }
      const double Xw        = Cx + s * dwx;
      const double Yw        = Cy + s * dwy;
      gray.at<uint8_t>(u, v) = static_cast<uint8_t>(
          std::lround(texture(static_cast<float>(Xw), static_cast<float>(Yw))));
      depth(v, u) = static_cast<float>(s);
    }
  }
}

void renderFrame(float tx, mrpt::img::CImage& gray, mrpt::math::CMatrixFloat& depth)
{
  renderPose(mrpt::poses::CPose3D(tx, 0, 0, 0, 0, 0), gray, depth);
}
}  // namespace

TEST(RgbdSlam, RecoversPureTranslation)
{
  mola::RgbdSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);

  const auto  cam  = makeCamera();
  const int   N    = 12;
  const float step = 0.03f;  // per-frame +X translation [m]

  mrpt::poses::CPose3D last;
  for (int k = 0; k < N; ++k)
  {
    const float              tx = static_cast<float>(k) * step;
    mrpt::img::CImage        gray;
    mrpt::math::CMatrixFloat depth;
    renderFrame(tx, gray, depth);
    last = slam.processFrame(gray, depth, cam, mrpt::Clock::now());
  }

  // Map and keyframes were created.
  EXPECT_GT(slam.numLandmarks(), 50u);
  EXPECT_GE(slam.numKeyframes(), 1u);
  // Active (non-culled) landmarks are a subset of all landmarks.
  EXPECT_LE(slam.numActiveLandmarks(), slam.numLandmarks());
  EXPECT_GT(slam.numActiveLandmarks(), 50u);

  // Recovered camera-in-world translation should match ground truth.
  const float gt_x = static_cast<float>(N - 1) * step;
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

TEST(RgbdSlam, RecoversRotationAndTranslation)
{
  mola::RgbdSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);

  const auto  cam      = makeCamera();
  const int   N        = 24;  // > default kf_max_frames_gap (20) to force a 2nd keyframe + BA
  const float step_x   = 0.02f;  // per-frame +X translation [m]
  const float step_yaw = 0.3 * M_PI / 180.0;  // per-frame yaw [rad]

  mrpt::poses::CPose3D last;
  mrpt::poses::CPose3D gt;
  for (int k = 0; k < N; ++k)
  {
    gt = mrpt::poses::CPose3D(
        static_cast<double>(k) * step_x, 0, 0, static_cast<double>(k) * step_yaw, 0, 0);
    mrpt::img::CImage        gray;
    mrpt::math::CMatrixFloat depth;
    renderPose(gt, gray, depth);
    last = slam.processFrame(gray, depth, cam, mrpt::Clock::now());
  }

  // Recover the final camera-in-world pose (translation < 3 cm, yaw < 0.5 deg).
  EXPECT_NEAR(last.x(), gt.x(), 0.03);
  EXPECT_NEAR(last.y(), gt.y(), 0.03);
  EXPECT_NEAR(last.z(), gt.z(), 0.03);
  EXPECT_NEAR(last.yaw(), gt.yaw(), 0.5 * M_PI / 180.0);
  EXPECT_NEAR(last.pitch(), gt.pitch(), 0.5 * M_PI / 180.0);
  EXPECT_NEAR(last.roll(), gt.roll(), 0.5 * M_PI / 180.0);
  EXPECT_GT(slam.numKeyframes(), 1u);
}
