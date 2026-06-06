/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/orb_loop_detector.h>

#include <cmath>
#include <random>

using mola::vision::Keyframe;
using mola::vision::OrbLoopDetector;

namespace
{
// Non-periodic smoothed-noise texture; `seed` selects the scene, (dx,dy) shifts.
mrpt::img::CImage scene(int W, int H, unsigned seed, float dx = 0.f, float dy = 0.f)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> uni(0.f, 255.f);
  std::vector<float>                    buf(static_cast<size_t>(W) * H);
  for (auto& v : buf)
  {
    v = uni(rng);
  }
  std::vector<float> tmp(buf.size());
  for (int pass = 0; pass < 3; ++pass)
  {
    for (int r = 0; r < H; ++r)
    {
      for (int c = 0; c < W; ++c)
      {
        const int cl = std::max(0, c - 1), cr = std::min(W - 1, c + 1);
        tmp[r * W + c] = (buf[r * W + cl] + buf[r * W + c] + buf[r * W + cr]) / 3.f;
      }
    }
    for (int r = 0; r < H; ++r)
    {
      for (int c = 0; c < W; ++c)
      {
        const int ru = std::max(0, r - 1), rd = std::min(H - 1, r + 1);
        buf[r * W + c] = (tmp[ru * W + c] + tmp[r * W + c] + tmp[rd * W + c]) / 3.f;
      }
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

Keyframe::Ptr makeKf(int id, const mrpt::img::CImage& img)
{
  auto kf   = std::make_shared<Keyframe>();
  kf->id    = id;
  kf->image = img;
  return kf;
}
}  // namespace

TEST(OrbLoopDetector, DetectsRevisitAndRejectsNonMatch)
{
  OrbLoopDetector det;

  det.addKeyframe(makeKf(0, scene(320, 240, /*seed=*/1)));  // place A
  det.addKeyframe(makeKf(10, scene(320, 240, /*seed=*/2)));  // place B
  EXPECT_EQ(det.size(), 2u);

  // Query: revisit place A from a slightly shifted viewpoint, much later.
  const auto q     = makeKf(40, scene(320, 240, /*seed=*/1, 2.f, -1.5f));
  const auto cands = det.detect(q, /*min_id_gap=*/5);

  ASSERT_FALSE(cands.empty());
  EXPECT_EQ(cands.front().matched_kf_id, 0);  // matched place A, not B
  EXPECT_GT(cands.front().score, 25.f);
}

TEST(OrbLoopDetector, NoLoopForUnseenPlace)
{
  OrbLoopDetector det;
  det.addKeyframe(makeKf(0, scene(320, 240, 1)));
  det.addKeyframe(makeKf(10, scene(320, 240, 2)));

  // A scene never visited.
  const auto q     = makeKf(40, scene(320, 240, /*seed=*/99));
  const auto cands = det.detect(q, 5);
  EXPECT_TRUE(cands.empty());
}

TEST(OrbLoopDetector, IdGapExcludesRecentKeyframes)
{
  OrbLoopDetector det;
  const auto      A = scene(320, 240, 1);
  det.addKeyframe(makeKf(10, A));  // same place as the query, but recent

  const auto q = makeKf(12, scene(320, 240, 1, 1.f, 0.f));
  // |12 - 10| = 2 < min_id_gap(5) => the only db entry is excluded.
  EXPECT_TRUE(det.detect(q, 5).empty());
  // With a small gap it is allowed and detected.
  EXPECT_FALSE(det.detect(q, 1).empty());
}
