/* -------------------------------------------------------------------------
 * mola_libvision unit tests: camera models
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/camera_model.h>
#include <mrpt/img/camera_geometry.h>

#include <cmath>
#include <random>

using namespace mola::vision;

namespace
{
std::vector<CameraModel> testCameras()
{
  return {
      CameraModel::Pinhole(640, 480, 500.0, 505.0, 321.0, 239.0),
      CameraModel::RadTan(
          752, 480, 458.6, 457.3, 367.2, 248.4, -0.2834, 0.0740, 1.9e-4, 1.8e-5, 0.0),
      CameraModel::RadTan(1280, 720, 900.0, 900.0, 640.0, 360.0, 0.1, -0.05, -1e-3, 2e-3, 0.01),
      CameraModel::Equidistant(1280, 1024, 350.0, 350.5, 640.0, 512.0, -0.01, 0.02, -0.01, 0.002),
  };
}

Eigen::Vector3d randomPointInView(const CameraModel& cam, std::mt19937& rng)
{
  std::uniform_real_distribution<double> ux(0.1, 0.9);
  std::uniform_real_distribution<double> depth(1.0, 10.0);
  for (int tries = 0; tries < 1000; ++tries)
  {
    const Eigen::Vector2d px(ux(rng) * cam.width(), ux(rng) * cam.height());
    Eigen::Vector3d       b;
    if (cam.unproject(px, b))
    {
      return depth(rng) * b;
    }
  }
  return {0.0, 0.0, 1.0};
}
}  // namespace

TEST(CameraModel, JacobiansMatchFiniteDifferences)
{
  std::mt19937 rng(1);
  for (const auto& cam : testCameras())
  {
    for (int k = 0; k < 50; ++k)
    {
      const Eigen::Vector3d                    P = randomPointInView(cam, rng);
      Eigen::Vector2d                          px;
      Eigen::Matrix<double, 2, 3>              Jp;
      Eigen::Matrix<double, 2, Eigen::Dynamic> Jk;
      ASSERT_TRUE(cam.project(P, px, &Jp, &Jk));

      const double h = 1e-6;
      for (int c = 0; c < 3; ++c)
      {
        Eigen::Vector3d Pp = P;
        Eigen::Vector3d Pm = P;
        Pp[c] += h;
        Pm[c] -= h;
        Eigen::Vector2d a;
        Eigen::Vector2d b;
        ASSERT_TRUE(cam.project(Pp, a));
        ASSERT_TRUE(cam.project(Pm, b));
        const Eigen::Vector2d num = (a - b) / (2 * h);
        EXPECT_NEAR(num.x(), Jp(0, c), 1e-4 * (1 + std::abs(Jp(0, c))));
        EXPECT_NEAR(num.y(), Jp(1, c), 1e-4 * (1 + std::abs(Jp(1, c))));
      }
      for (int c = 0; c < static_cast<int>(cam.params().size()); ++c)
      {
        CameraModel     cp = cam;
        CameraModel     cm = cam;
        Eigen::VectorXd pp = cam.params();
        Eigen::VectorXd pm = cam.params();
        const double    hk = 1e-7 * std::max(1.0, std::abs(pp[c]));
        pp[c] += hk;
        pm[c] -= hk;
        cp.setParams(pp);
        cm.setParams(pm);
        Eigen::Vector2d a;
        Eigen::Vector2d b;
        ASSERT_TRUE(cp.project(P, a));
        ASSERT_TRUE(cm.project(P, b));
        const Eigen::Vector2d num = (a - b) / (2 * hk);
        EXPECT_NEAR(num.x(), Jk(0, c), 1e-3 * (1 + std::abs(Jk(0, c))));
        EXPECT_NEAR(num.y(), Jk(1, c), 1e-3 * (1 + std::abs(Jk(1, c))));
      }
    }
  }
}

TEST(CameraModel, UnprojectInvertsProject)
{
  std::mt19937 rng(2);
  for (const auto& cam : testCameras())
  {
    std::uniform_real_distribution<double> ux(0.0, 1.0);
    for (int k = 0; k < 200; ++k)
    {
      const Eigen::Vector2d px(ux(rng) * (cam.width() - 1), ux(rng) * (cam.height() - 1));
      Eigen::Vector3d       b;
      if (!cam.unproject(px, b))
      {
        continue;
      }
      EXPECT_NEAR(b.norm(), 1.0, 1e-12);
      Eigen::Vector2d px2;
      ASSERT_TRUE(cam.project(3.0 * b, px2));
      EXPECT_NEAR((px2 - px).norm(), 0.0, 1e-6);
    }
  }
}

TEST(CameraModel, EquidistantBeyondNinetyDegrees)
{
  const auto cam = CameraModel::Equidistant(1280, 1024, 300.0, 300.0, 640.0, 512.0, 0, 0, 0, 0);
  // 100 deg incidence, behind the image plane.
  const double          a = 100.0 * M_PI / 180.0;
  const Eigen::Vector3d P(std::sin(a), 0.0, std::cos(a));
  Eigen::Vector2d       px;
  ASSERT_TRUE(cam.project(P, px));
  EXPECT_NEAR(px.x(), 640.0 + 300.0 * a, 1e-9);
  Eigen::Vector3d b;
  ASSERT_TRUE(cam.unproject(px, b));
  EXPECT_NEAR((b - P).norm(), 0.0, 1e-9);

  // A pinhole model cannot see behind its image plane.
  const auto pin = CameraModel::Pinhole(640, 480, 300, 300, 320, 240);
  EXPECT_FALSE(pin.project(P, px));
}

TEST(CameraModel, MatchesMrptProjection)
{
  std::mt19937 rng(3);
  for (const auto& cam : testCameras())
  {
    const mrpt::img::TCamera tc   = cam.asTCamera();
    const CameraModel        back = CameraModel::FromTCamera(tc);
    EXPECT_EQ(back.type(), cam.type());
    EXPECT_NEAR((back.params() - cam.params()).norm(), 0.0, 1e-12);
    for (int k = 0; k < 50; ++k)
    {
      const Eigen::Vector3d P = randomPointInView(cam, rng);
      Eigen::Vector2d       px;
      ASSERT_TRUE(cam.project(P, px));
      mrpt::img::TPixelCoordf ref;
      mrpt::img::camera_geometry::projectPoint_with_distortion(
          mrpt::math::TPoint3D(P.x(), P.y(), P.z()), tc, ref);
      EXPECT_NEAR(px.x(), ref.x, 2e-3);
      EXPECT_NEAR(px.y(), ref.y, 2e-3);
    }
  }
}

TEST(CameraModel, MonotonicRegion)
{
  // Strong barrel distortion folds back at a finite radius.
  const auto cam = CameraModel::RadTan(640, 480, 300, 300, 320, 240, -0.5, 0.0, 0, 0, 0);
  // 1 + 3 k1 r^2 = 0 at r = sqrt(1/1.5).
  const double expected = std::atan(std::sqrt(1.0 / 1.5));
  EXPECT_NEAR(cam.maxIncidenceAngle(), expected, 0.03);
  Eigen::Vector2d px;
  EXPECT_TRUE(cam.project({0.5, 0.0, 1.0}, px));
  EXPECT_FALSE(cam.project({1.0, 0.0, 1.0}, px));
}

TEST(IntrinsicsParameterization, FreeVectorRoundTrip)
{
  const auto     cam = CameraModel::RadTan(640, 480, 500, 510, 320, 240, -0.1, 0.01, 0, 0, 0);
  IntrinsicsMask mask;
  mask.fixed_aspect_ratio  = true;
  mask.estimate_distortion = IntrinsicsMask::FirstN(5, 2);
  IntrinsicsParameterization p(cam, mask);
  ASSERT_EQ(p.dim(), 5);  // f, cx, cy, k1, k2
  EXPECT_EQ(p.freeNames()[0], "f");

  Eigen::VectorXd v = p.toFree(cam);
  v[0]              = 600.0;
  v[3]              = -0.2;
  CameraModel out   = cam;
  p.fromFree(v, out);
  EXPECT_NEAR(out.fx(), 600.0, 1e-12);
  EXPECT_NEAR(out.fy(), 600.0 * 510.0 / 500.0, 1e-9);
  EXPECT_NEAR(out.params()[4], -0.2, 1e-12);
  EXPECT_NEAR(out.params()[6], 0.0, 1e-12);  // p1 kept fixed

  // d(full)/d(free) for the shared focal.
  EXPECT_NEAR(p.jacobian()(0, 0), 1.0, 1e-12);
  EXPECT_NEAR(p.jacobian()(1, 0), 510.0 / 500.0, 1e-12);
}

TEST(CameraModel, FoldFoundExactlyNotBySampling)
{
  // Radial derivative 1 - 3s + s^2 is negative for s in (0.382, 2.618): any
  // point beyond the first fold must be rejected, even far beyond it.
  const auto      cam = CameraModel::RadTan(640, 480, 300, 300, 320, 240, -1.0, 0.2, 0, 0, 0);
  Eigen::Vector2d px;
  EXPECT_FALSE(cam.project({8.0, 0.0, 1.0}, px));
  const double s_fold = (3.0 - std::sqrt(5.0)) / 2.0;
  EXPECT_NEAR(cam.maxIncidenceAngle(), std::atan(std::sqrt(s_fold)), 1e-9);
}

TEST(CameraModel, EquidistantInverseStaysOnValidBranch)
{
  const auto   cam = CameraModel::Equidistant(1280, 1024, 200, 200, 640, 512, 0.3, -0.05, 0, 0);
  const double th  = 1.5;
  const Eigen::Vector3d P(std::sin(th), 0.0, std::cos(th));
  Eigen::Vector2d       px;
  ASSERT_TRUE(cam.project(P, px));
  Eigen::Vector3d b;
  ASSERT_TRUE(cam.unproject(px, b));
  EXPECT_NEAR(std::acos(b.z()), th, 1e-9);
}

TEST(CameraModel, RejectsRearAxisAndTangentialFolds)
{
  const auto      fe = CameraModel::Equidistant(640, 480, 300, 300, 320, 240, 0, 0, 0, 0);
  Eigen::Vector2d px;
  EXPECT_FALSE(fe.project({0.0, 0.0, -1.0}, px));

  // Strong tangential term: {1,0,1} lies on the folded side.
  const auto rt = CameraModel::RadTan(640, 480, 300, 300, 320, 240, 0, 0, 0, -0.2, 0);
  EXPECT_FALSE(rt.project({1.0, 0.0, 1.0}, px));
  ASSERT_TRUE(rt.project({2.0 / 3.0, 0.0, 1.0}, px));
  Eigen::Vector3d b;
  ASSERT_TRUE(rt.unproject(px, b));
  EXPECT_NEAR(b.x() / b.z(), 2.0 / 3.0, 1e-9);
}
