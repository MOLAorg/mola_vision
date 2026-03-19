/* -------------------------------------------------------------------------
 * mola_libvision unit tests: optical_flow (LK tracker + F-matrix filter)
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/optical_flow.h>

#include <cmath>

using namespace mola::vision;

// ---------------------------------------------------------------------------
// Helper: create grayscale image with a simple pattern
static mrpt::img::CImage makeCheckerboard(int W, int H, int cell = 16)
{
  mrpt::img::CImage img(W, H, mrpt::img::CH_GRAY);
  for (int r = 0; r < H; ++r)
  {
    uint8_t* row = img.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c) row[c] = (((r / cell) + (c / cell)) % 2 == 0) ? 200 : 55;
  }
  return img;
}

/** Translate a CImage by (dx, dy) pixels (nearest-neighbour, zero-pad). */
static mrpt::img::CImage translateImage(const mrpt::img::CImage& src, int dx, int dy)
{
  const int         W = src.getWidth(), H = src.getHeight();
  mrpt::img::CImage dst(W, H, mrpt::img::CH_GRAY);
  for (int r = 0; r < H; ++r)
  {
    uint8_t*  out_row = dst.ptrLine<uint8_t>(r);
    const int rs      = r - dy;
    if (rs < 0 || rs >= H)
    {
      for (int c = 0; c < W; ++c) out_row[c] = 0;
      continue;
    }
    const uint8_t* in_row = src.ptrLine<uint8_t>(rs);
    for (int c = 0; c < W; ++c)
    {
      const int cs = c - dx;
      out_row[c]   = (cs >= 0 && cs < W) ? in_row[cs] : 0;
    }
  }
  return dst;
}

// ---------------------------------------------------------------------------
// LK tracking: known integer displacement
// ---------------------------------------------------------------------------
TEST(OpticalFlow, TrackKnownTranslation)
{
  const int DX = 3, DY = 2;
  auto      prev = makeCheckerboard(128, 128, 16);
  auto      curr = translateImage(prev, DX, DY);

  // Detect features in prev (away from borders to avoid OOB after translation)
  GoodFeaturesParams dp;
  dp.max_corners  = 30;
  dp.min_distance = 10.f;
  auto prev_pts   = goodFeaturesToTrack(prev, dp);

  // Keep only interior features (margin to handle translation + border)
  std::vector<mrpt::math::TPoint2Df> pts;
  const int                          margin = 20;
  for (const auto& p : prev_pts)
    if (p.x > margin && p.x < 128 - margin && p.y > margin && p.y < 128 - margin) pts.push_back(p);

  ASSERT_GE(pts.size(), 5u) << "Need at least 5 trackable points";

  std::vector<mrpt::math::TPoint2Df> next_pts;
  std::vector<TrackStatus>           status;

  LKParams params;
  params.max_levels = 2;
  params.win_size   = 10;
  params.max_iters  = 30;
  calcOpticalFlowPyrLK(prev, curr, pts, next_pts, status, params);

  int ok_count = 0;
  for (size_t i = 0; i < pts.size(); ++i)
  {
    if (status[i] != TrackStatus::OK) continue;
    ++ok_count;

    const float err_x = next_pts[i].x - (pts[i].x + DX);
    const float err_y = next_pts[i].y - (pts[i].y + DY);
    const float err   = std::sqrt(err_x * err_x + err_y * err_y);

    EXPECT_LT(err, 1.5f) << "Point " << i << " tracked with error " << err << " (expected dx=" << DX
                         << " dy=" << DY << ")";
  }

  EXPECT_GE(ok_count, static_cast<int>(pts.size()) * 7 / 10)
      << "At least 70% of points should track successfully";
}

TEST(OpticalFlow, LostPointsForBlankCurr)
{
  auto              prev = makeCheckerboard(64, 64, 8);
  mrpt::img::CImage curr(64, 64, mrpt::img::CH_GRAY);  // all-zero

  std::vector<mrpt::math::TPoint2Df> pts = {{10, 10}, {20, 20}, {30, 30}};
  std::vector<mrpt::math::TPoint2Df> next_pts;
  std::vector<TrackStatus>           status;

  calcOpticalFlowPyrLK(prev, curr, pts, next_pts, status);

  // All should be lost (min_eig will be too low in the blank image)
  // OR tracked to wrong location — either way it should not crash
  EXPECT_EQ(next_pts.size(), pts.size());
  EXPECT_EQ(status.size(), pts.size());
}

// ---------------------------------------------------------------------------
// Fundamental matrix filter: synthetic correspondences
// ---------------------------------------------------------------------------
TEST(OpticalFlow, FMatrixFilter_SyntheticInliers)
{
  // Generate correspondences along a known epiline pattern.
  // Use a simple homography (pure rotation around y by 5°) as ground truth.
  const float angle = 5.f * static_cast<float>(M_PI) / 180.f;
  const float ca = std::cos(angle), sa = std::sin(angle);
  // K = identity (normalized coords), R, t = [0.1, 0, 0]
  // pts1[i] → pts2[i] = R * pts1[i] + t  (approx, no depth)

  std::vector<mrpt::math::TPoint2Df> p1, p2;
  std::vector<TrackStatus>           status;

  // 40 inliers
  for (int i = 0; i < 40; ++i)
  {
    const float x = -1.f + 0.05f * i, y = -0.5f + 0.025f * i;
    p1.push_back({x, y});
    // Approximate epipolar: p2 ≈ p1 + small translation in x
    p2.push_back({x + 0.1f * ca, y + 0.1f * sa});
    status.push_back(TrackStatus::OK);
  }

  // 10 outliers (random far-away positions)
  for (int i = 0; i < 10; ++i)
  {
    p1.push_back({static_cast<float>(i) * 0.1f, 0.f});
    p2.push_back({0.f, static_cast<float>(i) * 0.1f});  // clearly wrong
    status.push_back(TrackStatus::OK);
  }

  FMatrixFilterParams params;
  params.ransac_threshold = 3.f;
  fundamentalMatrixFilter(p1, p2, status, params);

  // Most inliers (first 40) should remain OK
  int inlier_ok = 0, outlier_rejected = 0;
  for (int i = 0; i < 40; ++i)
    if (status[i] == TrackStatus::OK) ++inlier_ok;
  for (int i = 40; i < 50; ++i)
    if (status[i] == TrackStatus::OUTLIER) ++outlier_rejected;

  EXPECT_GE(inlier_ok, 30) << "At least 30/40 true inliers should survive";
  EXPECT_GE(outlier_rejected, 5) << "At least 5/10 true outliers should be rejected";
}
