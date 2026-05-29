/* -------------------------------------------------------------------------
 * mola_libvision unit tests: optical_flow (LK tracker + F-matrix filter)
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/optical_flow.h>

#include <Eigen/Geometry>
#include <cmath>
#include <random>

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
  // Build a genuine (non-degenerate) two-view geometry in PIXEL coordinates.
  //  - Pinhole camera K (f=500, principal point (320,240)).
  //  - Second camera = small yaw rotation + sideways translation (baseline).
  //  - 3D points spread in X,Y at *varying* depths (avoids planar / collinear
  //    degeneracies that make the fundamental matrix unobservable).
  // Inlier matches are the projections of the same 3D point in both views;
  // outliers are random pixel pairs.
  const double f = 500.0, cx = 320.0, cy = 240.0;
  auto         project = [&](const Eigen::Vector3d& Pc) -> mrpt::math::TPoint2Df
  {
    return {
        static_cast<float>(f * Pc.x() / Pc.z() + cx), static_cast<float>(f * Pc.y() / Pc.z() + cy)};
  };

  const double    yaw = 4.0 * M_PI / 180.0;
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY());
  const Eigen::Vector3d t(0.4, 0.0, 0.0);  // baseline along x

  std::vector<mrpt::math::TPoint2Df> p1, p2;
  std::vector<TrackStatus>           status;

  std::mt19937                           rng(1234);
  std::uniform_real_distribution<double> ux(-2.0, 2.0);
  std::uniform_real_distribution<double> uy(-1.5, 1.5);
  std::uniform_real_distribution<double> uz(4.0, 9.0);  // varying depth

  const int nInliers = 40;
  for (int i = 0; i < nInliers; ++i)
  {
    const Eigen::Vector3d Pw(ux(rng), uy(rng), uz(rng));
    const Eigen::Vector3d Pc2 = R * Pw + t;
    if (Pc2.z() < 0.5)
    {
      continue;
    }
    p1.push_back(project(Pw));  // first camera at identity
    p2.push_back(project(Pc2));  // second camera
    status.push_back(TrackStatus::OK);
  }
  const int realInliers = static_cast<int>(p1.size());

  // 10 outliers: random, inconsistent pixel pairs.
  std::uniform_real_distribution<double> upx(0.0, 640.0);
  std::uniform_real_distribution<double> upy(0.0, 480.0);
  const int                              nOutliers = 10;
  for (int i = 0; i < nOutliers; ++i)
  {
    p1.push_back({static_cast<float>(upx(rng)), static_cast<float>(upy(rng))});
    p2.push_back({static_cast<float>(upx(rng)), static_cast<float>(upy(rng))});
    status.push_back(TrackStatus::OK);
  }

  FMatrixFilterParams params;
  params.ransac_threshold = 2.f;  // pixels
  fundamentalMatrixFilter(p1, p2, status, params);

  int inlier_ok = 0, outlier_rejected = 0;
  for (int i = 0; i < realInliers; ++i)
  {
    if (status[i] == TrackStatus::OK)
    {
      ++inlier_ok;
    }
  }
  for (int i = realInliers; i < realInliers + nOutliers; ++i)
  {
    if (status[i] == TrackStatus::OUTLIER)
    {
      ++outlier_rejected;
    }
  }

  EXPECT_GE(inlier_ok, static_cast<int>(0.75 * realInliers))
      << "At least 75% of true inliers should survive";
  EXPECT_GE(outlier_rejected, 5) << "At least 5/10 true outliers should be rejected";
}
