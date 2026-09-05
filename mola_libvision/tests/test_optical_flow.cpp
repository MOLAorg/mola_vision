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

/** Non-periodic, well-textured image: smoothed random noise. Non-repeating so
 *  there is no matching ambiguity, and smoothing yields trackable gradients. */
static mrpt::img::CImage makeSmoothTexture(int W, int H)
{
  std::mt19937                          rng(12345);
  std::uniform_real_distribution<float> uni(0.f, 255.f);
  std::vector<float>                    buf(static_cast<size_t>(W) * H);
  for (auto& v : buf) v = uni(rng);

  // A few separable 3x3 box blurs => smooth but still locally unique.
  std::vector<float> tmp(buf.size());
  for (int pass = 0; pass < 4; ++pass)
  {
    for (int r = 0; r < H; ++r)
    {
      for (int c = 0; c < W; ++c)
      {
        const int   cl = std::max(0, c - 1), cr = std::min(W - 1, c + 1);
        const float s  = buf[r * W + cl] + buf[r * W + c] + buf[r * W + cr];
        tmp[r * W + c] = s / 3.f;
      }
    }
    for (int r = 0; r < H; ++r)
    {
      for (int c = 0; c < W; ++c)
      {
        const int   ru = std::max(0, r - 1), rd = std::min(H - 1, r + 1);
        const float s  = tmp[ru * W + c] + tmp[r * W + c] + tmp[rd * W + c];
        buf[r * W + c] = s / 3.f;
      }
    }
  }

  // Stretch contrast so smoothed values span a usable range.
  float lo = 1e9f, hi = -1e9f;
  for (float v : buf)
  {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  const float scale = (hi > lo) ? 255.f / (hi - lo) : 1.f;

  mrpt::img::CImage img(W, H, mrpt::img::CH_GRAY);
  for (int r = 0; r < H; ++r)
  {
    uint8_t* row = img.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c)
    {
      row[c] = static_cast<uint8_t>(std::lround((buf[r * W + c] - lo) * scale));
    }
  }
  return img;
}

/** Translate a CImage by a sub-pixel (dx, dy) via bilinear resampling. */
static mrpt::img::CImage translateImageSubpixel(const mrpt::img::CImage& src, float dx, float dy)
{
  const int         W = src.getWidth(), H = src.getHeight();
  mrpt::img::CImage dst(W, H, mrpt::img::CH_GRAY);
  for (int r = 0; r < H; ++r)
  {
    uint8_t* out_row = dst.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c)
    {
      const float xs = static_cast<float>(c) - dx;
      const float ys = static_cast<float>(r) - dy;
      const int   x0 = static_cast<int>(std::floor(xs));
      const int   y0 = static_cast<int>(std::floor(ys));
      if (x0 < 0 || x0 + 1 >= W || y0 < 0 || y0 + 1 >= H)
      {
        out_row[c] = 0;
        continue;
      }
      const float fx = xs - static_cast<float>(x0);
      const float fy = ys - static_cast<float>(y0);
      const auto  p  = [&](int xx, int yy)
      { return static_cast<float>(src.ptrLine<uint8_t>(yy)[xx]); };
      const float v = (1 - fy) * ((1 - fx) * p(x0, y0) + fx * p(x0 + 1, y0)) +
                      fy * ((1 - fx) * p(x0, y0 + 1) + fx * p(x0 + 1, y0 + 1));
      out_row[c] = static_cast<uint8_t>(std::lround(v));
    }
  }
  return dst;
}

// ---------------------------------------------------------------------------
// LK tracking: sub-pixel displacement (guards tracker accuracy)
// ---------------------------------------------------------------------------
TEST(OpticalFlow, TrackSubpixelTranslation)
{
  const float DX = 1.7f, DY = -0.8f;
  auto        prev = makeSmoothTexture(160, 160);
  auto        curr = translateImageSubpixel(prev, DX, DY);

  GoodFeaturesParams dp;
  dp.max_corners  = 60;
  dp.min_distance = 8.f;
  auto prev_pts   = goodFeaturesToTrack(prev, dp);

  std::vector<mrpt::math::TPoint2Df> pts;
  const int                          margin = 24;
  for (const auto& p : prev_pts)
  {
    if (p.x > margin && p.x < 160 - margin && p.y > margin && p.y < 160 - margin) pts.push_back(p);
  }
  ASSERT_GE(pts.size(), 8u);

  std::vector<mrpt::math::TPoint2Df> next_pts;
  std::vector<TrackStatus>           status;
  LKParams                           params;
  params.max_levels = 3;
  params.win_size   = 10;
  calcOpticalFlowPyrLK(prev, curr, pts, next_pts, status, params);

  int ok_count = 0;
  for (size_t i = 0; i < pts.size(); ++i)
  {
    if (status[i] != TrackStatus::OK) continue;
    ++ok_count;
    const float ex  = next_pts[i].x - (pts[i].x + DX);
    const float ey  = next_pts[i].y - (pts[i].y + DY);
    const float err = std::sqrt(ex * ex + ey * ey);
    EXPECT_LT(err, 0.5f) << "Point " << i << " sub-pixel error " << err;
  }
  EXPECT_GE(ok_count, static_cast<int>(pts.size()) * 7 / 10);
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
  // OR tracked to wrong location; either way it should not crash
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

// ---------------------------------------------------------------------------
// A displacement too large for the pyramid to absorb from a standing start is
// still tracked when the caller supplies a predicted position.
//
// This is the case that breaks a VO front-end first on a legged robot: during a
// fast turn the true image motion exceeds what the coarsest pyramid level can
// find from "the point did not move", tracking collapses, and the pose solver
// has nothing left to work with. A pose-model prediction restores it.
// ---------------------------------------------------------------------------
TEST(OpticalFlow, InitialGuessTracksLargeDisplacement)
{
  const float DX   = 70.f;
  const float DY   = -25.f;
  auto        prev = makeSmoothTexture(320, 320);
  auto        curr = translateImageSubpixel(prev, DX, DY);

  GoodFeaturesParams dp;
  dp.max_corners      = 80;
  dp.min_distance     = 10.f;
  const auto detected = goodFeaturesToTrack(prev, dp);

  // Keep points whose displaced position is comfortably inside the image.
  std::vector<mrpt::math::TPoint2Df> pts;
  for (const auto& p : detected)
  {
    const float nx = p.x + DX;
    const float ny = p.y + DY;
    if (p.x > 40 && p.y > 40 && nx > 40 && ny > 40 && nx < 280 && ny < 280 && p.x < 280 &&
        p.y < 280)
    {
      pts.push_back(p);
    }
  }
  ASSERT_GE(pts.size(), 8u);

  LKParams params;
  params.max_levels = 2;  // deliberately too shallow for a 70 px jump

  auto countAccurate =
      [&](const std::vector<mrpt::math::TPoint2Df>& res, const std::vector<TrackStatus>& st)
  {
    int n = 0;
    for (size_t i = 0; i < pts.size(); ++i)
    {
      if (st[i] != TrackStatus::OK)
      {
        continue;
      }
      if (std::hypot(res[i].x - (pts[i].x + DX), res[i].y - (pts[i].y + DY)) < 1.0f)
      {
        ++n;
      }
    }
    return n;
  };

  std::vector<mrpt::math::TPoint2Df> plainRes;
  std::vector<TrackStatus>           plainStatus;
  calcOpticalFlowPyrLK(prev, curr, pts, plainRes, plainStatus, params);
  const int plainOk = countAccurate(plainRes, plainStatus);

  // Now the same tracking, seeded with a (deliberately imperfect) prediction.
  std::vector<mrpt::math::TPoint2Df> guessRes(pts.size());
  for (size_t i = 0; i < pts.size(); ++i)
  {
    guessRes[i] = {pts[i].x + DX + 2.f, pts[i].y + DY - 2.f};
  }
  std::vector<TrackStatus> guessStatus;
  params.use_initial_guess = true;
  calcOpticalFlowPyrLK(prev, curr, pts, guessRes, guessStatus, params);
  const int guessOk = countAccurate(guessRes, guessStatus);

  EXPECT_GT(guessOk, plainOk) << "the prediction must help, not merely not hurt";
  EXPECT_GE(guessOk, static_cast<int>(pts.size()) * 8 / 10);
}

// ---------------------------------------------------------------------------
// The recovered flow must be unbiased in MAGNITUDE, not merely small in error.
//
// LK fits a pure TRANSLATION to a patch that is really being warped, so its
// estimate is a gradient-energy-weighted average of the flow across the patch,
// not the flow at its center. On a SLANTED surface - where depth, and hence
// flow, varies across the patch - that average is systematically off, and the
// bias grows with the window. A VO front-end integrates it straight into a
// trajectory scale error, so a fraction of a percent per frame is a large error
// over a mission.
//
// The warp below is the real thing rather than a stand-in: a plane slanted in
// depth, seen by a camera translating along its optical axis, whose induced
// flow is u' = u + u * tz / Z(u) with Z varying linearly across the image.
// ---------------------------------------------------------------------------
TEST(OpticalFlow, FlowGainIsUnbiasedOnASlantedSurface)
{
  constexpr int    W   = 320;
  constexpr double kCx = 0.5 * (W - 1);
  constexpr double kCy = 0.5 * (W - 1);
  constexpr double kTz = 0.08;  // forward camera translation [m]
  // Depth of the slanted plane: 4 m at the left edge, 8 m at the right one.
  auto depthAtU = [](double u) { return 4.0 + 4.0 * (u / static_cast<double>(W - 1)); };
  // Forward motion maps (u, v) to the image point of the same 3D surface point.
  auto warp = [&](double u, double v)
  {
    const double Z  = depthAtU(u);
    const double sc = Z / (Z - kTz);
    return std::pair<double, double>{kCx + (u - kCx) * sc, kCy + (v - kCy) * sc};
  };

  const auto prev = makeSmoothTexture(W, W);

  // Build `curr` by inverse mapping: for each destination pixel, find the source
  // pixel it came from (a few fixed-point steps invert the mild warp exactly
  // enough at this magnitude).
  mrpt::img::CImage curr(W, W, mrpt::img::CH_GRAY);
  for (int r = 0; r < W; ++r)
  {
    for (int x = 0; x < W; ++x)
    {
      double su = x;
      double sv = r;
      for (int it = 0; it < 12; ++it)
      {
        const auto [fu, fv] = warp(su, sv);
        su += (x - fu);
        sv += (r - fv);
      }
      const int x0 = static_cast<int>(std::floor(su));
      const int y0 = static_cast<int>(std::floor(sv));
      if (x0 < 0 || y0 < 0 || x0 + 1 >= W || y0 + 1 >= W)
      {
        curr.at<uint8_t>(x, r) = 0;
        continue;
      }
      const float fx  = static_cast<float>(su - x0);
      const float fy  = static_cast<float>(sv - y0);
      const float p00 = prev.at<uint8_t>(x0, y0);
      const float p10 = prev.at<uint8_t>(x0 + 1, y0);
      const float p01 = prev.at<uint8_t>(x0, y0 + 1);
      const float p11 = prev.at<uint8_t>(x0 + 1, y0 + 1);
      const float v   = (1 - fy) * ((1 - fx) * p00 + fx * p10) + fy * ((1 - fx) * p01 + fx * p11);
      curr.at<uint8_t>(x, r) = static_cast<uint8_t>(std::lround(v));
    }
  }

  GoodFeaturesParams dp;
  dp.max_corners      = 250;
  dp.min_distance     = 9.f;
  const auto detected = goodFeaturesToTrack(prev, dp);

  std::vector<mrpt::math::TPoint2Df> pts;
  for (const auto& p : detected)
  {
    const double r = std::hypot(p.x - kCx, p.y - kCy);
    if (r > 55.0 && p.x > 35 && p.y > 35 && p.x < W - 35 && p.y < W - 35)
    {
      pts.push_back(p);
    }
  }
  ASSERT_GE(pts.size(), 20u);

  std::vector<mrpt::math::TPoint2Df> res;
  std::vector<TrackStatus>           status;
  LKParams                           params;
  calcOpticalFlowPyrLK(prev, curr, pts, res, status, params);

  std::vector<double> gain;
  for (size_t i = 0; i < pts.size(); ++i)
  {
    if (status[i] != TrackStatus::OK)
    {
      continue;
    }
    const auto [tu, tv]   = warp(pts[i].x, pts[i].y);
    const double trueFlow = std::hypot(tu - pts[i].x, tv - pts[i].y);
    if (trueFlow < 1.0)
    {
      continue;
    }
    const double gotFlow = std::hypot(res[i].x - pts[i].x, res[i].y - pts[i].y);
    gain.push_back(gotFlow / trueFlow);
  }
  ASSERT_GE(gain.size(), 15u);
  std::sort(gain.begin(), gain.end());
  const double medianGain = gain[gain.size() / 2];
  std::cout << "[LK gain] n=" << gain.size() << " median gain=" << medianGain << " ("
            << 100.0 * (medianGain - 1.0) << "% error)\n";
  // For reference, on this same warp: a uniformly weighted 21x21 window gives
  // ~0.15%, and a uniformly weighted 43x43 one ~0.6%.
  EXPECT_NEAR(medianGain, 1.0, 0.003);
}
