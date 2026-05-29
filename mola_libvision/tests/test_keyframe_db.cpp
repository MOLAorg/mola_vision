/* -------------------------------------------------------------------------
 * mola_libvision unit tests: KeyframeDatabase + KeyframeSelector
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/keyframe_database.h>
#include <mola_libvision/keyframe_selector.h>

using namespace mola::vision;

namespace
{
/** Build a keyframe observing the given map points. */
Keyframe::Ptr makeKf(int id, const std::vector<MapPoint::Ptr>& mps)
{
  auto kf = std::make_shared<Keyframe>();
  kf->id  = id;
  for (const auto& mp : mps)
  {
    Feature f;
    f.map_point = mp;
    kf->features.push_back(f);
  }
  return kf;
}
}  // namespace

// ---------------------------------------------------------------------------
// Covisibility graph from shared map points.
// ---------------------------------------------------------------------------
TEST(KeyframeDatabase, Covisibility)
{
  // 10 shared landmarks.
  std::vector<MapPoint::Ptr> mps;
  for (int i = 0; i < 10; ++i)
  {
    mps.push_back(std::make_shared<MapPoint>(i));
  }

  // KF0 sees 0..7, KF1 sees 4..9 (shared 4,5,6,7 => 4), KF2 sees 8,9 (shares
  // 0 with KF0, 2 with KF1).
  auto kf0 = makeKf(0, {mps[0], mps[1], mps[2], mps[3], mps[4], mps[5], mps[6], mps[7]});
  auto kf1 = makeKf(1, {mps[4], mps[5], mps[6], mps[7], mps[8], mps[9]});
  auto kf2 = makeKf(2, {mps[8], mps[9]});

  KeyframeDatabase db;
  db.add(kf0);
  db.add(kf1);
  db.add(kf2);
  EXPECT_EQ(db.size(), 3u);

  db.updateCovisibility(/*min_shared=*/1);

  EXPECT_EQ(db.connectionWeight(0, 1), 4);
  EXPECT_EQ(db.connectionWeight(1, 0), 4);  // symmetric
  EXPECT_EQ(db.connectionWeight(0, 2), 0);  // no shared map points
  EXPECT_EQ(db.connectionWeight(1, 2), 2);

  // Neighbors of KF1 sorted by descending weight: KF0 (4), KF2 (2).
  auto nb = db.covisibleKeyframes(1, /*min_weight=*/1);
  ASSERT_EQ(nb.size(), 2u);
  EXPECT_EQ(nb[0].first, 0);
  EXPECT_EQ(nb[0].second, 4);
  EXPECT_EQ(nb[1].first, 2);
  EXPECT_EQ(nb[1].second, 2);
}

// ---------------------------------------------------------------------------
// min_shared threshold prunes weak edges.
// ---------------------------------------------------------------------------
TEST(KeyframeDatabase, MinSharedThreshold)
{
  std::vector<MapPoint::Ptr> mps;
  for (int i = 0; i < 10; ++i)
  {
    mps.push_back(std::make_shared<MapPoint>(i));
  }
  auto kf0 = makeKf(0, {mps[0], mps[1], mps[2], mps[3], mps[4], mps[5], mps[6], mps[7]});
  auto kf1 = makeKf(1, {mps[4], mps[5], mps[6], mps[7], mps[8], mps[9]});  // shares 4

  KeyframeDatabase db;
  db.add(kf0);
  db.add(kf1);

  db.updateCovisibility(/*min_shared=*/5);  // 4 < 5 => no edge
  EXPECT_EQ(db.connectionWeight(0, 1), 0);

  db.updateCovisibility(/*min_shared=*/4);  // 4 >= 4 => edge
  EXPECT_EQ(db.connectionWeight(0, 1), 4);
}

// ---------------------------------------------------------------------------
// remove() drops the keyframe and its edges.
// ---------------------------------------------------------------------------
TEST(KeyframeDatabase, Remove)
{
  std::vector<MapPoint::Ptr> mps;
  for (int i = 0; i < 4; ++i)
  {
    mps.push_back(std::make_shared<MapPoint>(i));
  }
  KeyframeDatabase db;
  db.add(makeKf(0, {mps[0], mps[1], mps[2]}));
  db.add(makeKf(1, {mps[1], mps[2], mps[3]}));
  db.updateCovisibility(1);
  EXPECT_EQ(db.connectionWeight(0, 1), 2);

  db.remove(0);
  EXPECT_EQ(db.size(), 1u);
  EXPECT_EQ(db.get(0), nullptr);
  EXPECT_EQ(db.connectionWeight(0, 1), 0);
  EXPECT_EQ(db.connectionWeight(1, 0), 0);
}

// ---------------------------------------------------------------------------
// KeyframeSelector policy.
// ---------------------------------------------------------------------------
TEST(KeyframeSelector, Policy)
{
  KeyframeSelectorParams p;
  p.max_frames_gap    = 20;
  p.min_frames_gap    = 2;
  p.min_tracked       = 50;
  p.min_tracked_ratio = 0.5f;
  p.min_parallax_px   = 15.0f;
  KeyframeSelector sel(p);

  KeyframeFrameStats s;
  s.ref_num_features   = 200;
  s.num_tracked        = 180;
  s.median_parallax_px = 2.0f;

  // Debounce: too soon.
  s.frames_since_kf = 1;
  EXPECT_FALSE(sel.shouldBeKeyframe(s));

  // Healthy tracking, low parallax, within gap -> no keyframe.
  s.frames_since_kf = 5;
  EXPECT_FALSE(sel.shouldBeKeyframe(s));

  // Enough parallax -> keyframe.
  s.median_parallax_px = 20.0f;
  EXPECT_TRUE(sel.shouldBeKeyframe(s));

  // Weak absolute tracking -> keyframe.
  s.median_parallax_px = 1.0f;
  s.num_tracked        = 40;  // <= min_tracked
  EXPECT_TRUE(sel.shouldBeKeyframe(s));

  // Weak relative tracking -> keyframe.
  s.num_tracked = 80;  // ratio 80/200 = 0.4 < 0.5
  EXPECT_TRUE(sel.shouldBeKeyframe(s));

  // Hard frame-gap cap forces a keyframe even when everything else is fine.
  s.num_tracked        = 190;
  s.median_parallax_px = 0.f;
  s.frames_since_kf    = 25;
  EXPECT_TRUE(sel.shouldBeKeyframe(s));
}
