/* -------------------------------------------------------------------------
 * mola_libvision unit tests: feature_detection
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/feature_detection.h>

#include <cmath>

using namespace mola::vision;

static mrpt::img::CImage makeCheckerboard(int W, int H, int cell_size)
{
  mrpt::img::CImage img(W, H, mrpt::img::CH_GRAY);
  for (int r = 0; r < H; ++r)
  {
    uint8_t* row = img.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c)
    {
      const int cell_r = r / cell_size, cell_c = c / cell_size;
      row[c] = ((cell_r + cell_c) % 2 == 0) ? 200 : 55;
    }
  }
  return img;
}

static mrpt::img::CImage makeUniform(int W, int H, uint8_t val = 128)
{
  mrpt::img::CImage img(W, H, mrpt::img::CH_GRAY);
  for (int r = 0; r < H; ++r)
  {
    uint8_t* row = img.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c) row[c] = val;
  }
  return img;
}

// ---------------------------------------------------------------------------
// goodFeaturesToTrack
// ---------------------------------------------------------------------------
TEST(FeatureDetection, DetectsCorners)
{
  // 8×8 checker cells on a 128×128 image → ~100 internal corners
  auto img = makeCheckerboard(128, 128, 8);

  GoodFeaturesParams params;
  params.max_corners   = 200;
  params.min_distance  = 4.0f;
  params.quality_level = 0.01f;

  auto corners = goodFeaturesToTrack(img, params);
  EXPECT_GT(corners.size(), 10u) << "Should detect many corners on checkerboard";
  EXPECT_LE(corners.size(), static_cast<size_t>(params.max_corners));
}

TEST(FeatureDetection, ReturnsEmpty_FlatImage)
{
  auto img     = makeUniform(64, 64, 100);
  auto corners = goodFeaturesToTrack(img);
  EXPECT_EQ(corners.size(), 0u);
}

TEST(FeatureDetection, RespectsMaxCorners)
{
  auto img = makeCheckerboard(256, 256, 8);

  GoodFeaturesParams params;
  params.max_corners  = 20;
  params.min_distance = 2.0f;
  auto corners        = goodFeaturesToTrack(img, params);
  EXPECT_LE(corners.size(), 20u);
}

TEST(FeatureDetection, MinDistanceEnforced)
{
  auto img = makeCheckerboard(128, 128, 4);

  GoodFeaturesParams params;
  params.max_corners  = 500;
  params.min_distance = 10.0f;
  auto corners        = goodFeaturesToTrack(img, params);

  // All pairs must be at least 10 pixels apart
  for (size_t i = 0; i < corners.size(); ++i)
  {
    for (size_t j = i + 1; j < corners.size(); ++j)
    {
      const float dx   = corners[i].x - corners[j].x;
      const float dy   = corners[i].y - corners[j].y;
      const float dist = std::sqrt(dx * dx + dy * dy);
      EXPECT_GE(dist, params.min_distance - 0.5f)
          << "Corners " << i << " and " << j << " are too close: " << dist;
    }
  }
}

// ---------------------------------------------------------------------------
// GridFeatureDistributor
// ---------------------------------------------------------------------------
TEST(GridFeatureDistributor, BasicDetection)
{
  auto img = makeCheckerboard(128, 128, 8);

  GridDistributorParams params;
  params.max_corners  = 100;
  params.min_distance = 5.0f;
  GridFeatureDistributor dist(params);

  auto corners = dist.detect(img);
  EXPECT_GT(corners.size(), 5u);
  EXPECT_LE(corners.size(), static_cast<size_t>(params.max_corners));
}

TEST(GridFeatureDistributor, AvoidsExistingPoints)
{
  auto img = makeCheckerboard(128, 128, 8);

  // Pre-populate with points that cover a center cluster
  std::vector<mrpt::math::TPoint2Df> existing;
  for (int r = 40; r < 90; r += 5)
    for (int c = 40; c < 90; c += 5)
      existing.push_back({static_cast<float>(c), static_cast<float>(r)});

  GridDistributorParams params;
  params.min_distance = 5.0f;
  GridFeatureDistributor dist(params);
  auto                   new_corners = dist.detect(img, existing);

  // None of the new corners should be within 5px of existing ones
  for (const auto& nc : new_corners)
  {
    for (const auto& ex : existing)
    {
      const float dx = nc.x - ex.x, dy = nc.y - ex.y;
      EXPECT_GE(std::sqrt(dx * dx + dy * dy), params.min_distance - 0.5f);
    }
  }
}
