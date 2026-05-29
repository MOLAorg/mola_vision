/* -------------------------------------------------------------------------
 * mola_libvision unit tests: stereo matcher
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/stereo_matcher.h>

#include <cmath>
#include <random>

using namespace mola::vision;

namespace
{
/** A non-repetitive, well-textured grayscale image (smoothed random noise),
 *  good for unambiguous LK tracking. */
mrpt::img::CImage makeTexture(int W, int H, unsigned seed)
{
  std::mt19937                       rng(seed);
  std::uniform_int_distribution<int> noise(0, 255);
  std::vector<std::vector<float>>    raw(H, std::vector<float>(W));
  for (int r = 0; r < H; ++r)
  {
    for (int c = 0; c < W; ++c)
    {
      raw[r][c] = static_cast<float>(noise(rng));
    }
  }
  // 3x3 box blur a couple of times to create smooth gradients.
  for (int pass = 0; pass < 2; ++pass)
  {
    std::vector<std::vector<float>> tmp = raw;
    for (int r = 1; r < H - 1; ++r)
    {
      for (int c = 1; c < W - 1; ++c)
      {
        float acc = 0.f;
        for (int dr = -1; dr <= 1; ++dr)
        {
          for (int dc = -1; dc <= 1; ++dc)
          {
            acc += tmp[r + dr][c + dc];
          }
        }
        raw[r][c] = acc / 9.f;
      }
    }
  }
  mrpt::img::CImage img(W, H, mrpt::img::CH_GRAY);
  for (int r = 0; r < H; ++r)
  {
    uint8_t* row = img.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c)
    {
      row[c] = static_cast<uint8_t>(std::max(0.f, std::min(255.f, raw[r][c])));
    }
  }
  return img;
}

/** Build the RIGHT-camera image for a fronto-parallel plane by shifting content
 *  to the LEFT by `d` pixels. With the right camera displaced along +x, a scene
 *  point's image moves left, so a left feature at column x appears at x - d in
 *  the right image, giving positive disparity x_left - x_right = d. */
mrpt::img::CImage makeRightImage(const mrpt::img::CImage& src, int d)
{
  const int         W = src.getWidth(), H = src.getHeight();
  mrpt::img::CImage dst(W, H, mrpt::img::CH_GRAY);
  for (int r = 0; r < H; ++r)
  {
    const uint8_t* in  = src.ptrLine<uint8_t>(r);
    uint8_t*       out = dst.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c)
    {
      const int cs = c + d;
      out[c]       = (cs >= 0 && cs < W) ? in[cs] : 0;
    }
  }
  return dst;
}

}  // namespace

// ---------------------------------------------------------------------------
// Fronto-parallel plane: constant disparity -> constant known depth.
// ---------------------------------------------------------------------------
TEST(StereoMatcher, FrontoParallelPlaneDepth)
{
  const int    W = 320, H = 240;
  const int    disparity = 8;
  const double fx = 500.0, baseline = 0.12;
  const double expectedDepth = fx * baseline / disparity;  // = 7.5 m

  const auto left  = makeTexture(W, H, 42);
  const auto right = makeRightImage(left, disparity);

  // Detect features in the left image, away from borders.
  GoodFeaturesParams dp;
  dp.max_corners  = 80;
  dp.min_distance = 8.f;
  auto allPts     = goodFeaturesToTrack(left, dp);

  std::vector<mrpt::math::TPoint2Df> pts;
  const int                          margin = 20;
  for (const auto& p : allPts)
  {
    if (p.x > margin && p.x < W - margin && p.y > margin && p.y < H - margin)
    {
      pts.push_back(p);
    }
  }
  ASSERT_GE(pts.size(), 10u) << "need enough trackable features";

  StereoMatchParams params;
  params.max_y_diff     = 1.5f;
  params.min_disparity  = 1.f;
  StereoMatchResult res = matchStereo(left, right, pts, fx, baseline, params);

  ASSERT_EQ(res.depth.size(), pts.size());
  EXPECT_GE(res.num_valid, static_cast<int>(0.6 * pts.size()))
      << "at least 60% of features should match";

  int depthOk = 0;
  for (size_t i = 0; i < pts.size(); ++i)
  {
    if (!res.valid[i])
    {
      continue;
    }
    // Right match must be ~disparity px to the left, same row.
    EXPECT_NEAR(res.right_pts[i].y, pts[i].y, 1.5f);
    const float d = pts[i].x - res.right_pts[i].x;
    EXPECT_NEAR(d, static_cast<float>(disparity), 1.5f);
    if (std::abs(res.depth[i] - expectedDepth) < 0.15 * expectedDepth)
    {
      ++depthOk;
    }
  }
  EXPECT_GE(depthOk, static_cast<int>(0.6 * res.num_valid))
      << "most valid matches should yield ~correct depth (" << expectedDepth << " m)";
}

// ---------------------------------------------------------------------------
// Empty input is handled gracefully.
// ---------------------------------------------------------------------------
TEST(StereoMatcher, EmptyInput)
{
  const auto                         img = makeTexture(64, 64, 1);
  std::vector<mrpt::math::TPoint2Df> pts;
  StereoMatchResult                  res = matchStereo(img, img, pts, 500.0, 0.1);
  EXPECT_EQ(res.num_valid, 0);
  EXPECT_TRUE(res.depth.empty());
}

// ---------------------------------------------------------------------------
// Identical images -> zero disparity -> rejected by min_disparity.
// ---------------------------------------------------------------------------
TEST(StereoMatcher, ZeroDisparityRejected)
{
  const auto         img = makeTexture(160, 120, 7);
  GoodFeaturesParams dp;
  dp.max_corners = 40;
  auto pts       = goodFeaturesToTrack(img, dp);
  ASSERT_GE(pts.size(), 5u);

  StereoMatchParams params;
  params.min_disparity  = 1.0f;
  StereoMatchResult res = matchStereo(img, img, pts, 500.0, 0.1, params);
  EXPECT_EQ(res.num_valid, 0) << "zero-disparity matches must be rejected";
}
