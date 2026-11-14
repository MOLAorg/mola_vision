/* -------------------------------------------------------------------------
 * mola_libvision unit tests: constant-velocity motion model
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/motion_model.h>

#include <cmath>

using mola::vision::ConstantVelocityModel;
using mrpt::poses::CPose3D;

namespace
{
// Pose after driving for t seconds at forward speed vx and yaw rate wz (a
// circular arc), in closed form:
CPose3D arc(double t, double vx, double wz)
{
  const double th = wz * t;
  return {vx / wz * std::sin(th), vx / wz * (1.0 - std::cos(th)), 0, th, 0, 0};
}

double dist(const CPose3D& a, const CPose3D& b)
{
  return (a.asVectorVal() - b.asVectorVal()).norm();
}
}  // namespace

TEST(MotionModel, EmptyAndSinglePose)
{
  ConstantVelocityModel m;
  EXPECT_FALSE(m.hasVelocity());
  EXPECT_NEAR(dist(m.predict(1.0), CPose3D::Identity()), 0.0, 1e-12);
  const CPose3D p(1, 2, 0, 0.3, 0, 0);
  m.addPose(5.0, p);
  EXPECT_FALSE(m.hasVelocity());
  EXPECT_NEAR(dist(m.predict(6.0), p), 0.0, 1e-12);
}

TEST(MotionModel, BridgesAGapWithTheElapsedTime)
{
  // 30 Hz poses along a circular arc, then a 1 s gap: the prediction must
  // follow the arc for the whole second, not for one frame period.
  ConstantVelocityModel m;
  const double          vx = 0.1;
  const double          wz = 0.2;
  for (int i = 0; i <= 30; i++)
  {
    m.addPose(i / 30.0, arc(i / 30.0, vx, wz));
  }
  ASSERT_TRUE(m.hasVelocity());
  EXPECT_NEAR(m.velocity()[0], vx, 1e-9);
  EXPECT_NEAR(m.velocity()[1], 0.0, 1e-9);
  EXPECT_NEAR(m.velocity()[5], wz, 1e-9);
  EXPECT_NEAR(dist(m.predict(2.0), arc(2.0, vx, wz)), 0.0, 1e-9);
}

TEST(MotionModel, ExtrapolationIsClamped)
{
  ConstantVelocityModel::Parameters p;
  p.max_extrapolation = 1.5;
  ConstantVelocityModel m(p);
  for (int i = 0; i <= 10; i++)
  {
    m.addPose(i * 0.1, CPose3D(0.1 * i * 0.1, 0, 0, 0, 0, 0));  // 0.1 m/s along x
  }
  EXPECT_NEAR(m.predict(1.0 + 5.0).x(), 0.1 + 0.1 * 1.5, 1e-9);
  EXPECT_NEAR(m.predict(0.5).x(), 0.1, 1e-9);  // never extrapolates backwards
}

TEST(MotionModel, VelocityAfterAGapUsesBothSides)
{
  ConstantVelocityModel m;
  for (int i = 0; i <= 30; i++)
  {
    m.addPose(i / 30.0, CPose3D(0.1 * i / 30.0, 0, 0, 0, 0, 0));
  }
  // The robot stopped during a 1 s gap: the next pose shows it.
  m.addPose(2.0, CPose3D(0.1, 0, 0, 0, 0, 0));
  EXPECT_NEAR(m.velocity()[0], 0.0, 1e-9);
  // Older poses are ignored:
  m.addPose(1.5, CPose3D(5, 0, 0, 0, 0, 0));
  EXPECT_NEAR(m.predict(3.0).x(), 0.1, 1e-9);
}

TEST(MotionModel, ScalePoseIncrement)
{
  const CPose3D inc(0.05, 0.01, 0, 0.1, 0.02, 0);
  EXPECT_NEAR(dist(mola::vision::scalePoseIncrement(inc, 3.0), inc + inc + inc), 0.0, 1e-9);
  EXPECT_NEAR(dist(mola::vision::scalePoseIncrement(inc, 0.0), CPose3D::Identity()), 0.0, 1e-12);
}
