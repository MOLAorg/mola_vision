/* -------------------------------------------------------------------------
 * mola_libvision unit tests: RGBD depth processing
 * SPDX-License-Identifier: BSD-3-Clause
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
