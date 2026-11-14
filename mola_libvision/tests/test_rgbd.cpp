/* -------------------------------------------------------------------------
 * mola_libvision unit tests: RGBD depth processing
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/rgbd_depth.h>

#include <cmath>

using namespace mola::vision;

namespace
{
mrpt::img::TCamera makeCamera(int w = 640, int h = 480)
{
  mrpt::img::TCamera cam;
  cam.ncols = w;
  cam.nrows = h;
  cam.setIntrinsicParamsFromValues(500.0, 500.0, w * 0.5, h * 0.5);
  cam.distortion = mrpt::img::DistortionModel::none;
  return cam;
}
}  // namespace

// ---------------------------------------------------------------------------
// Quadratic uncertainty model.
// ---------------------------------------------------------------------------
TEST(RGBD, DepthStdDev)
{
  RGBDParams p;
  p.unc_a = 0.0012f;
  p.unc_b = 0.0019f;
  p.unc_c = 0.0001f;

  // sigma = sqrt(a*d^2 + b*d + c)
  const float d   = 2.0f;
  const float var = p.unc_a * d * d + p.unc_b * d + p.unc_c;
  EXPECT_NEAR(depthStdDev(d, p), std::sqrt(var), 1e-6f);

  // Monotonically increasing with depth.
  EXPECT_LT(depthStdDev(1.0f, p), depthStdDev(5.0f, p));
}

// ---------------------------------------------------------------------------
// Single-pixel back-projection.
// ---------------------------------------------------------------------------
TEST(RGBD, BackprojectPixel)
{
  const auto cam = makeCamera();

  const float cx = static_cast<float>(cam.cx());
  const float cy = static_cast<float>(cam.cy());
  const float fx = static_cast<float>(cam.fx());

  // Principal point at depth d -> (0, 0, d).
  auto c = backprojectPixel({cx, cy}, 3.0f, cam);
  ASSERT_TRUE(c.has_value());
  EXPECT_NEAR(c->x, 0.f, 1e-5f);
  EXPECT_NEAR(c->y, 0.f, 1e-5f);
  EXPECT_NEAR(c->z, 3.0f, 1e-5f);

  // One focal length to the right of cx -> X == depth.
  auto r = backprojectPixel({cx + fx, cy}, 4.0f, cam);
  ASSERT_TRUE(r.has_value());
  EXPECT_NEAR(r->x, 4.0f, 1e-4f);
  EXPECT_NEAR(r->y, 0.f, 1e-5f);
  EXPECT_NEAR(r->z, 4.0f, 1e-5f);

  // Invalid depth -> nullopt.
  EXPECT_FALSE(backprojectPixel({100.f, 100.f}, 0.f, cam).has_value());
  RGBDParams p;
  p.max_depth = 5.0f;
  EXPECT_FALSE(backprojectPixel({100.f, 100.f}, 10.f, cam, p).has_value());
}

// ---------------------------------------------------------------------------
// Dense back-projection of a fronto-parallel plane.
// ---------------------------------------------------------------------------
TEST(RGBD, BackprojectDepthMap_Plane)
{
  const int  W = 64, H = 48;
  const auto cam = makeCamera(W, H);

  const float     Z     = 2.5f;
  Eigen::MatrixXf depth = Eigen::MatrixXf::Constant(H, W, Z);
  // Mark a few pixels invalid (zero) and out-of-range.
  depth(0, 0)        = 0.f;  // invalid
  depth(1, 1)        = 1000.f;  // out of range (> max_depth default 100)
  const int nInvalid = 2;

  DepthCloud cloud = backprojectDepthMap(depth, cam, {}, 1);

  EXPECT_EQ(static_cast<int>(cloud.points.size()), W * H - nInvalid);
  EXPECT_EQ(cloud.points.size(), cloud.pixels.size());

  // All valid points lie on the Z plane.
  for (const auto& p : cloud.points)
  {
    EXPECT_NEAR(p.z, Z, 1e-5f);
  }

  // Spot-check a known pixel's back-projection.
  const float u = 10.f, v = 5.f;
  auto        expected = backprojectPixel({u, v}, Z, cam);
  ASSERT_TRUE(expected.has_value());
  bool found = false;
  for (size_t i = 0; i < cloud.pixels.size(); ++i)
  {
    if (cloud.pixels[i].x == u && cloud.pixels[i].y == v)
    {
      EXPECT_NEAR(cloud.points[i].x, expected->x, 1e-5f);
      EXPECT_NEAR(cloud.points[i].y, expected->y, 1e-5f);
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// Decimation reduces the cloud size roughly by step^2.
// ---------------------------------------------------------------------------
TEST(RGBD, Decimation)
{
  const int       W = 64, H = 64;
  const auto      cam   = makeCamera(W, H);
  Eigen::MatrixXf depth = Eigen::MatrixXf::Constant(H, W, 3.0f);

  DepthCloud full = backprojectDepthMap(depth, cam, {}, 1);
  DepthCloud dec  = backprojectDepthMap(depth, cam, {}, 4);

  EXPECT_EQ(static_cast<int>(full.points.size()), W * H);
  // step=4 over 64 -> 16x16 = 256 points.
  EXPECT_EQ(static_cast<int>(dec.points.size()), 16 * 16);
}

// ---------------------------------------------------------------------------
// Depth registration into another camera.
// ---------------------------------------------------------------------------
namespace
{
mrpt::math::CMatrix_u16 flatDepth(int w, int h, uint16_t mm)
{
  mrpt::math::CMatrix_u16 d(h, w);
  d.asEigen().setConstant(mm);
  return d;
}
}  // namespace

TEST(RGBD, RegisterDepthIdentity)
{
  const auto cam = makeCamera(64, 48);
  auto       raw = flatDepth(64, 48, 1500);
  raw(10, 20)    = 0;  // a hole stays a hole

  const auto out = registerDepthToCamera(raw, 1e-3f, cam, cam, mrpt::poses::CPose3D::Identity());
  ASSERT_EQ(out.rows(), 48);
  ASSERT_EQ(out.cols(), 64);
  EXPECT_NEAR(out(30, 40), 1.5f, 1e-5f);
  EXPECT_EQ(out(10, 20), 0.f);
}

TEST(RGBD, RegisterDepthBaselineShift)
{
  // A fronto-parallel plane at 2 m, seen by a target camera 0.1 m to the right
  // of the depth camera: it appears fx * 0.1 / 2 = 25 px to the left.
  const auto                 cam = makeCamera(640, 480);
  const auto                 raw = flatDepth(640, 480, 2000);
  const mrpt::poses::CPose3D targetFromDepth(-0.1, 0, 0, 0, 0, 0);

  const auto out = registerDepthToCamera(raw, 1e-3f, cam, cam, targetFromDepth);
  EXPECT_NEAR(out(240, 320), 2.0f, 1e-4f);
  // The right-most 25 columns are not seen by the depth camera:
  EXPECT_EQ(out(240, 630), 0.f);
  EXPECT_NEAR(out(240, 610), 2.0f, 1e-4f);
}

TEST(RGBD, RegisterDepthNoHolesWhenUpsampling)
{
  // Target camera with twice the focal length: splatting must avoid holes.
  const auto depthCam  = makeCamera(320, 240);
  auto       targetCam = makeCamera(320, 240);
  targetCam.setIntrinsicParamsFromValues(1000.0, 1000.0, 160.0, 120.0);
  const auto raw = flatDepth(320, 240, 1000);

  const auto out =
      registerDepthToCamera(raw, 1e-3f, depthCam, targetCam, mrpt::poses::CPose3D::Identity());
  for (int r = 10; r < 230; ++r)
  {
    for (int c = 10; c < 310; ++c)
    {
      ASSERT_NEAR(out(r, c), 1.0f, 1e-4f) << "r=" << r << " c=" << c;
    }
  }
}

TEST(RGBD, RegisterDepthZBuffer)
{
  // A near object occludes the far background where both project.
  const auto cam = makeCamera(64, 48);
  auto       raw = flatDepth(64, 48, 3000);
  for (int r = 20; r < 28; ++r)
  {
    for (int c = 28; c < 36; ++c)
    {
      raw(r, c) = 500;
    }
  }
  const auto out = registerDepthToCamera(raw, 1e-3f, cam, cam, mrpt::poses::CPose3D::Identity());
  EXPECT_NEAR(out(24, 32), 0.5f, 1e-5f);
  EXPECT_NEAR(out(5, 5), 3.0f, 1e-5f);
}

TEST(RGBD, OpticalFrameConvention)
{
  // The optical +Z axis is the +X axis of the "X forward" frame:
  const auto p = opticalFrameInXForwardFrame().composePoint({0, 0, 1});
  EXPECT_NEAR(p.x, 1.0, 1e-9);
  EXPECT_NEAR(p.y, 0.0, 1e-9);
  EXPECT_NEAR(p.z, 0.0, 1e-9);
  // Optical +X (right) is -Y (left) in the "X forward" frame:
  const auto q = opticalFrameInXForwardFrame().composePoint({1, 0, 0});
  EXPECT_NEAR(q.y, -1.0, 1e-9);
}
