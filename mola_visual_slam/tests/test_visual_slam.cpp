/* -------------------------------------------------------------------------
 * mola_visual_slam: monocular / stereo visual SLAM front-end for MOLA.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_visual_slam/VisualSlam.h>
#include <mrpt/core/Clock.h>
#include <mrpt/img/CImage.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/system/COutputLogger.h>

#include <cmath>
#include <random>
#include <vector>

namespace
{
// Monocular synthetic scene: a cloud of 3D points spread over TWO depth layers
// (so the structure is non-degenerate for essential-matrix estimation, unlike a
// single plane). Each visible point is rendered as a small bright blob, which
// the Shi-Tomasi detector + LK tracker follow across views. No depth is given to
// the SLAM module (monocular).

constexpr int   kW  = 640;
constexpr int   kH  = 480;
constexpr float kFx = 500.f;
constexpr float kFy = 500.f;
constexpr float kCx = 320.f;
constexpr float kCy = 240.f;

mrpt::img::TCamera makeCamera()
{
  mrpt::img::TCamera cam;
  cam.ncols = kW;
  cam.nrows = kH;
  cam.setIntrinsicParamsFromValues(kFx, kFy, kCx, kCy);
  return cam;
}

std::vector<mrpt::math::TPoint3D> makeCloud()
{
  std::vector<mrpt::math::TPoint3D>     pts;
  std::mt19937                          rng(2024);
  std::uniform_real_distribution<float> ux(-2.0f, 2.0f);
  std::uniform_real_distribution<float> uy(-1.5f, 1.5f);
  // Two depth layers => genuine depth variation (non-planar).
  for (int layer = 0; layer < 2; ++layer)
  {
    const float Z = (layer == 0) ? 3.0f : 6.0f;
    for (int i = 0; i < 120; ++i)
    {
      pts.push_back({ux(rng), uy(rng), Z});
    }
  }
  return pts;
}

// Render the cloud as seen from camera-in-world pose `pose_wc`.
mrpt::img::CImage renderView(
    const mrpt::poses::CPose3D& pose_wc, const std::vector<mrpt::math::TPoint3D>& cloud)
{
  std::vector<float> buf(static_cast<size_t>(kW) * kH, 70.f);  // dark background

  for (const auto& P : cloud)
  {
    const auto Xc = pose_wc.inverseComposePoint(P);  // world -> camera
    if (Xc.z < 0.3)
    {
      continue;
    }
    const float u  = kFx * static_cast<float>(Xc.x / Xc.z) + kCx;
    const float v  = kFy * static_cast<float>(Xc.y / Xc.z) + kCy;
    const int   cu = static_cast<int>(std::lround(u));
    const int   cv = static_cast<int>(std::lround(v));
    if (cu < 3 || cv < 3 || cu >= kW - 3 || cv >= kH - 3)
    {
      continue;
    }
    for (int dy = -3; dy <= 3; ++dy)
    {
      for (int dx = -3; dx <= 3; ++dx)
      {
        const float r2 = static_cast<float>(dx * dx + dy * dy);
        const float w  = 170.f * std::exp(-r2 / (2.f * 1.6f * 1.6f));
        float&      px = buf[static_cast<size_t>(cv + dy) * kW + (cu + dx)];
        px             = std::min(255.f, px + w);
      }
    }
  }

  mrpt::img::CImage img(kW, kH, mrpt::img::CH_GRAY);
  for (int y = 0; y < kH; ++y)
  {
    for (int x = 0; x < kW; ++x)
    {
      img.at<uint8_t>(x, y) = static_cast<uint8_t>(buf[static_cast<size_t>(y) * kW + x]);
    }
  }
  return img;
}
}  // namespace

TEST(VisualSlam, MonoBootstrapAndTrack)
{
  mola::VisualSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);

  const auto  cam   = makeCamera();
  const auto  cloud = makeCloud();
  const int   N     = 20;
  const float step  = 0.04f;  // +X translation per frame [m]

  mrpt::poses::CPose3D last;
  for (int k = 0; k < N; ++k)
  {
    const mrpt::poses::CPose3D gt(static_cast<double>(k) * step, 0, 0, 0, 0, 0);
    const auto                 img = renderView(gt, cloud);
    last                           = slam.processFrame(img, cam, mrpt::Clock::now());
  }

  // The module should have bootstrapped and built a map.
  EXPECT_TRUE(slam.isInitialized());
  EXPECT_GT(slam.numActiveLandmarks(), 40u);
  EXPECT_GE(slam.numKeyframes(), 2u);

  // Monocular scale is arbitrary, so check the trajectory DIRECTION: the camera
  // moved along +X with near-identity rotation.
  const double tx   = last.x();
  const double ty   = last.y();
  const double tz   = last.z();
  const double norm = std::sqrt(tx * tx + ty * ty + tz * tz);
  ASSERT_GT(norm, 1e-3);
  EXPECT_GT(tx / norm, 0.95);  // direction within ~18 deg of +X
  EXPECT_NEAR(last.yaw(), 0.0, 3.0 * M_PI / 180.0);
  EXPECT_NEAR(last.pitch(), 0.0, 3.0 * M_PI / 180.0);
  EXPECT_NEAR(last.roll(), 0.0, 3.0 * M_PI / 180.0);
}

// A sparse, well-spaced cloud (grid of two depth layers). Sparse so stereo
// matching along the epipolar row is unambiguous (no near blob within the
// disparity range), which a synthetic identical-blob renderer needs; real
// textured imagery does not.
static std::vector<mrpt::math::TPoint3D> makeSparseCloud()
{
  std::vector<mrpt::math::TPoint3D> pts;
  for (int iy = 0; iy < 6; ++iy)
  {
    for (int ix = 0; ix < 8; ++ix)
    {
      const float X = -1.4f + 0.4f * static_cast<float>(ix);
      const float Y = -1.0f + 0.4f * static_cast<float>(iy);
      const float Z = ((ix + iy) % 2 == 0) ? 3.0f : 5.0f;
      pts.push_back({X, Y, Z});
    }
  }
  return pts;
}

TEST(VisualSlam, StereoRecoversMetricScale)
{
  mola::VisualSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);

  const auto   cam      = makeCamera();
  const auto   cloud    = makeSparseCloud();
  const int    N        = 20;
  const double step     = 0.04;  // +X translation per frame [m]
  const double baseline = 0.10;  // stereo baseline [m]

  mrpt::poses::CPose3D last;
  for (int k = 0; k < N; ++k)
  {
    const mrpt::poses::CPose3D gt(static_cast<double>(k) * step, 0, 0, 0, 0, 0);
    // Right camera sits +baseline along the (rectified) camera x axis.
    const mrpt::poses::CPose3D gt_right = gt + mrpt::poses::CPose3D(baseline, 0, 0, 0, 0, 0);
    const auto                 left     = renderView(gt, cloud);
    const auto                 right    = renderView(gt_right, cloud);
    last = slam.processStereoFrame(left, right, cam, baseline, mrpt::Clock::now());
  }

  EXPECT_TRUE(slam.isInitialized());
  EXPECT_GT(slam.numActiveLandmarks(), 40u);

  // Stereo fixes the scale, so the ABSOLUTE translation must match (in meters),
  // unlike monocular which is only correct up to an arbitrary scale. Tolerance
  // reflects synthetic integer-pixel detection quantization (real textured
  // imagery is sub-pixel and far more precise).
  const double gt_x = static_cast<double>(N - 1) * step;  // 0.76 m
  EXPECT_NEAR(last.x(), gt_x, 0.10);  // within ~13% (metric, not arbitrary scale)
  EXPECT_NEAR(last.y(), 0.0, 0.05);
  EXPECT_NEAR(last.z(), 0.0, 0.05);
  EXPECT_NEAR(last.yaw(), 0.0, 3.0 * M_PI / 180.0);
  EXPECT_NEAR(last.pitch(), 0.0, 3.0 * M_PI / 180.0);
  EXPECT_NEAR(last.roll(), 0.0, 3.0 * M_PI / 180.0);
}

TEST(VisualSlam, StaysUninitializedWithoutParallax)
{
  mola::VisualSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);

  const auto cam   = makeCamera();
  const auto cloud = makeCloud();

  // A stationary camera: no parallax => no bootstrap.
  for (int k = 0; k < 8; ++k)
  {
    const auto img = renderView(mrpt::poses::CPose3D::Identity(), cloud);
    slam.processFrame(img, cam, mrpt::Clock::now());
  }
  EXPECT_FALSE(slam.isInitialized());
}
