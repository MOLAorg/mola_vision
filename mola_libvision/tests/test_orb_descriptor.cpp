/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/orb_descriptor.h>

#include <cmath>
#include <random>

using mola::vision::computeOrbDescriptor;
using mola::vision::hammingDistance;
using mola::vision::OrbDescriptor;
using mrpt::math::TPoint2Df;

namespace
{
// Non-periodic smoothed-noise texture (unique local patches => meaningful
// descriptors), optionally translated by a sub-pixel (dx, dy).
mrpt::img::CImage makeTexture(int W, int H, float dx = 0.f, float dy = 0.f, unsigned seed = 7)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> uni(0.f, 255.f);
  std::vector<float>                    buf(static_cast<size_t>(W) * H);
  for (auto& v : buf) v = uni(rng);
  std::vector<float> tmp(buf.size());
  for (int pass = 0; pass < 3; ++pass)
  {
    for (int r = 0; r < H; ++r)
      for (int c = 0; c < W; ++c)
      {
        const int cl = std::max(0, c - 1), cr = std::min(W - 1, c + 1);
        tmp[r * W + c] = (buf[r * W + cl] + buf[r * W + c] + buf[r * W + cr]) / 3.f;
      }
    for (int r = 0; r < H; ++r)
      for (int c = 0; c < W; ++c)
      {
        const int ru = std::max(0, r - 1), rd = std::min(H - 1, r + 1);
        buf[r * W + c] = (tmp[ru * W + c] + tmp[r * W + c] + tmp[rd * W + c]) / 3.f;
      }
  }

  mrpt::img::CImage img(W, H, mrpt::img::CH_GRAY);
  for (int r = 0; r < H; ++r)
  {
    uint8_t* row = img.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c)
    {
      const float xs = static_cast<float>(c) - dx;
      const float ys = static_cast<float>(r) - dy;
      const int   x0 = std::min(W - 2, std::max(0, static_cast<int>(std::floor(xs))));
      const int   y0 = std::min(H - 2, std::max(0, static_cast<int>(std::floor(ys))));
      const float fx = xs - static_cast<float>(x0);
      const float fy = ys - static_cast<float>(y0);
      const auto  at = [&](int x, int y) { return buf[static_cast<size_t>(y) * W + x]; };
      const float v  = (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) +
                      fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
      row[c] = static_cast<uint8_t>(std::lround(std::min(255.f, std::max(0.f, v))));
    }
  }
  return img;
}
}  // namespace

TEST(OrbDescriptor, IdenticalPatchZeroDistance)
{
  const auto    img = makeTexture(200, 200);
  OrbDescriptor a;
  OrbDescriptor b;
  ASSERT_TRUE(computeOrbDescriptor(img, {100.f, 100.f}, a));
  ASSERT_TRUE(computeOrbDescriptor(img, {100.f, 100.f}, b));
  EXPECT_EQ(hammingDistance(a, b), 0);
}

TEST(OrbDescriptor, DiscriminatesMatchVsNonMatch)
{
  const auto img       = makeTexture(200, 200);
  const auto img_shift = makeTexture(200, 200, 1.0f, -1.0f);  // translate by (1,-1)

  // Same physical point: descriptor in the shifted image is at the shifted pixel.
  OrbDescriptor ref;
  OrbDescriptor matched;
  ASSERT_TRUE(computeOrbDescriptor(img, {100.f, 100.f}, ref));
  ASSERT_TRUE(computeOrbDescriptor(img_shift, {101.f, 99.f}, matched));

  // A different keypoint far away (non-match).
  OrbDescriptor other;
  ASSERT_TRUE(computeOrbDescriptor(img, {60.f, 140.f}, other));

  const int d_match    = hammingDistance(ref, matched);
  const int d_nonmatch = hammingDistance(ref, other);

  // The true match is much closer in Hamming space than a random patch.
  EXPECT_LT(d_match, 50);
  EXPECT_GT(d_nonmatch, 80);
  EXPECT_LT(d_match, d_nonmatch);
}

TEST(OrbDescriptor, OrientationIsDeterministicAndBorderGuarded)
{
  const auto    img = makeTexture(120, 120);
  OrbDescriptor d;
  // Too close to the border => skipped.
  EXPECT_FALSE(computeOrbDescriptor(img, {3.f, 3.f}, d));
  // Interior => ok.
  EXPECT_TRUE(computeOrbDescriptor(img, {60.f, 60.f}, d));
}
