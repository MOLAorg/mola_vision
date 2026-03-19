/* -------------------------------------------------------------------------
 * mola_libvision unit tests: MapPoint
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/MapPoint.h>

using namespace mola::vision;

TEST(MapPoint, Construction)
{
  MapPoint mp(42, {1.f, 2.f, 3.f});
  EXPECT_EQ(mp.id(), 42);
  EXPECT_FLOAT_EQ(mp.position().x, 1.f);
  EXPECT_FLOAT_EQ(mp.position().y, 2.f);
  EXPECT_FLOAT_EQ(mp.position().z, 3.f);
  EXPECT_FALSE(mp.isBad());
  EXPECT_EQ(mp.observationCount(), 0);
}

TEST(MapPoint, SetPosition)
{
  MapPoint mp(1);
  mp.setPositionEigen({4.f, 5.f, 6.f});
  const auto p = mp.positionEigen();
  EXPECT_FLOAT_EQ(p.x(), 4.f);
  EXPECT_FLOAT_EQ(p.y(), 5.f);
  EXPECT_FLOAT_EQ(p.z(), 6.f);
}

TEST(MapPoint, AddRemoveObservations)
{
  MapPoint mp(1, {0, 0, 1});

  auto kf1 = std::make_shared<Keyframe>();
  kf1->id  = 10;
  auto kf2 = std::make_shared<Keyframe>();
  kf2->id  = 20;

  mp.addObservation(kf1, 3);
  mp.addObservation(kf2, 7);
  EXPECT_EQ(mp.observationCount(), 2);

  mp.removeObservation(kf1);
  EXPECT_EQ(mp.observationCount(), 1);

  auto obs = mp.observations();
  EXPECT_EQ(obs.front().feature_index, 7);
  EXPECT_EQ(obs.front().keyframe.lock()->id, 20);
}

TEST(MapPoint, MarkBad)
{
  MapPoint mp(5, {1, 2, 3});
  EXPECT_FALSE(mp.isBad());
  mp.markBad();
  EXPECT_TRUE(mp.isBad());
}

TEST(MapPoint, CovarianceProjection)
{
  MapPoint mp(1, {0.f, 0.f, 5.f});  // directly ahead at 5m

  // Set isotropic covariance
  mp.setCovariance(Eigen::Matrix3f::Identity() * 0.01f);

  // Camera intrinsics (f=500)
  Eigen::Matrix3f K;
  K << 500.f, 0.f, 0.f, 0.f, 500.f, 0.f, 0.f, 0.f, 1.f;

  // Camera←World rotation = Identity (camera at origin looking along Z)
  const Eigen::Matrix2f cov_pixel = mp.projectCovariance(Eigen::Matrix3f::Identity(), K);

  // Must be symmetric positive definite
  EXPECT_GT(cov_pixel(0, 0), 0.f);
  EXPECT_GT(cov_pixel(1, 1), 0.f);
  EXPECT_NEAR(cov_pixel(0, 1), cov_pixel(1, 0), 1e-6f);

  // For a point at z=5 with σ_world=0.1m, σ_pixel ≈ f*σ/z = 500*0.1/5 = 10px
  // (variance ≈ 100)
  EXPECT_NEAR(cov_pixel(0, 0), 100.f, 5.f);
  EXPECT_NEAR(cov_pixel(1, 1), 100.f, 5.f);
}
