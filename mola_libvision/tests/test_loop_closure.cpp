/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/loop_closure.h>
#include <mola_libvision/orb_descriptor.h>

#include <cmath>
#include <random>
#include <vector>

using mola::vision::OrbDescriptor;
using mrpt::math::TPoint2Df;
using mrpt::math::TPoint3Df;

namespace
{
constexpr int   kW = 640, kH = 480;
constexpr float kFx = 500, kFy = 500, kCx = 320, kCy = 240;
constexpr float kD = 3.0f;  // plane depth in the candidate (identity) camera frame

mrpt::img::TCamera makeCam()
{
  mrpt::img::TCamera c;
  c.ncols = kW;
  c.nrows = kH;
  c.setIntrinsicParamsFromValues(kFx, kFy, kCx, kCy);
  return c;
}

// A large, non-periodic smoothed-noise world texture, sampled in metric world
// (X,Y) coordinates so every camera view of the plane is mutually consistent.
struct WorldTexture
{
  int                tw = 1024, th = 1024;
  std::vector<float> buf;
  float              ppm = 180.f;  // texture pixels per world meter
  WorldTexture()
  {
    std::mt19937                          rng(2024);
    std::uniform_real_distribution<float> uni(0.f, 255.f);
    buf.resize(static_cast<size_t>(tw) * th);
    for (auto& v : buf) v = uni(rng);
    std::vector<float> tmp(buf.size());
    for (int pass = 0; pass < 3; ++pass)
    {
      for (int r = 0; r < th; ++r)
        for (int c = 0; c < tw; ++c)
        {
          const int cl = std::max(0, c - 1), cr = std::min(tw - 1, c + 1);
          tmp[r * tw + c] = (buf[r * tw + cl] + buf[r * tw + c] + buf[r * tw + cr]) / 3.f;
        }
      for (int r = 0; r < th; ++r)
        for (int c = 0; c < tw; ++c)
        {
          const int ru = std::max(0, r - 1), rd = std::min(th - 1, r + 1);
          buf[r * tw + c] = (tmp[ru * tw + c] + tmp[r * tw + c] + tmp[rd * tw + c]) / 3.f;
        }
    }
  }
  float at(float Xw, float Yw) const
  {
    const float u  = Xw * ppm + tw * 0.5f;
    const float v  = Yw * ppm + th * 0.5f;
    const int   x0 = std::min(tw - 2, std::max(0, static_cast<int>(std::floor(u))));
    const int   y0 = std::min(th - 2, std::max(0, static_cast<int>(std::floor(v))));
    const float fx = u - x0, fy = v - y0;
    const auto  s = [&](int x, int y) { return buf[static_cast<size_t>(y) * tw + x]; };
    return (1 - fy) * ((1 - fx) * s(x0, y0) + fx * s(x0 + 1, y0)) +
           fy * ((1 - fx) * s(x0, y0 + 1) + fx * s(x0 + 1, y0 + 1));
  }
};

// Render the plane world Z=kD from camera-in-world pose `pose_wc`.
mrpt::img::CImage render(const WorldTexture& tex, const mrpt::poses::CPose3D& pose_wc)
{
  mrpt::img::CImage img(kW, kH, mrpt::img::CH_GRAY);
  const auto        R  = pose_wc.getRotationMatrix();
  const double      Cx = pose_wc.x(), Cy = pose_wc.y(), Cz = pose_wc.z();
  for (int v = 0; v < kH; ++v)
  {
    uint8_t* row = img.ptrLine<uint8_t>(v);
    for (int u = 0; u < kW; ++u)
    {
      const double dcx = (u - kCx) / kFx, dcy = (v - kCy) / kFy, dcz = 1.0;
      const double dwx = R(0, 0) * dcx + R(0, 1) * dcy + R(0, 2) * dcz;
      const double dwy = R(1, 0) * dcx + R(1, 1) * dcy + R(1, 2) * dcz;
      const double dwz = R(2, 0) * dcx + R(2, 1) * dcy + R(2, 2) * dcz;
      uint8_t      val = 0;
      if (std::abs(dwz) > 1e-9)
      {
        const double s = (kD - Cz) / dwz;
        if (s > 0.1)
        {
          val = static_cast<uint8_t>(std::lround(std::min(
              255.f,
              std::max(
                  0.f,
                  tex.at(static_cast<float>(Cx + s * dwx), static_cast<float>(Cy + s * dwy))))));
        }
      }
      row[u] = val;
    }
  }
  return img;
}
}  // namespace

TEST(LoopClosure, VerifyRecoversRelativePose)
{
  const WorldTexture tex;
  const auto         cam = makeCam();

  // Candidate keyframe: camera at identity; plane depth is uniform (= kD).
  const auto candImg = render(tex, mrpt::poses::CPose3D::Identity());

  mola::vision::GridDistributorParams gp;
  gp.max_corners  = 500;
  gp.min_distance = 10.f;
  mola::vision::GridFeatureDistributor dist(gp);

  const auto        candPts = dist.detect(candImg, {});
  std::vector<bool> cvalid;
  const auto        candDesc0 = mola::vision::computeOrbDescriptors(candImg, candPts, cvalid);
  std::vector<OrbDescriptor> candDesc;
  std::vector<TPoint3Df>     candXYZ;
  for (size_t i = 0; i < candPts.size(); ++i)
  {
    if (!cvalid[i]) continue;
    // Plane point in the candidate camera frame: depth = kD.
    const float X = (candPts[i].x - kCx) / kFx * kD;
    const float Y = (candPts[i].y - kCy) / kFy * kD;
    candDesc.push_back(candDesc0[i]);
    candXYZ.push_back({X, Y, kD});
  }
  ASSERT_GE(candDesc.size(), 100u);

  // Query keyframe: revisit from a different viewpoint.
  const mrpt::poses::CPose3D Q(0.15, -0.05, 0.0, 6.0 * M_PI / 180.0, 0.0, 0.0);  // cam-in-world
  const auto                 qImg = render(tex, Q);
  const auto                 qPts = dist.detect(qImg, {});
  std::vector<bool>          qvalid;
  const auto                 qDesc0 = mola::vision::computeOrbDescriptors(qImg, qPts, qvalid);
  std::vector<OrbDescriptor> qDesc;
  std::vector<TPoint2Df>     qPx;
  for (size_t i = 0; i < qPts.size(); ++i)
  {
    if (!qvalid[i]) continue;
    qDesc.push_back(qDesc0[i]);
    qPx.push_back(qPts[i]);
  }

  const auto res = mola::vision::verifyLoopPnP(qDesc, qPx, candDesc, candXYZ, cam);
  ASSERT_TRUE(res.success) << "matches=" << res.num_matches << " inliers=" << res.num_inliers;
  EXPECT_GE(res.num_inliers, 40);

  // PnP returns the query camera's pose in the candidate frame = Q^-1.
  const mrpt::poses::CPose3D gt = -Q;
  EXPECT_NEAR(res.relative_pose.x(), gt.x(), 0.03);
  EXPECT_NEAR(res.relative_pose.y(), gt.y(), 0.03);
  EXPECT_NEAR(res.relative_pose.z(), gt.z(), 0.03);
  EXPECT_NEAR(res.relative_pose.yaw(), gt.yaw(), 1.5 * M_PI / 180.0);
}

TEST(LoopClosure, RejectsNonMatchingPlace)
{
  WorldTexture texA;
  WorldTexture texB;
  for (auto& v : texB.buf) v = 255.f - v;  // a different "place"

  const auto cam     = makeCam();
  const auto candImg = render(texA, mrpt::poses::CPose3D::Identity());
  const auto qImg    = render(texB, mrpt::poses::CPose3D(0.1, 0, 0, 0, 0, 0));

  mola::vision::GridDistributorParams gp;
  gp.max_corners  = 500;
  gp.min_distance = 10.f;
  mola::vision::GridFeatureDistributor dist(gp);

  const auto                 cp = dist.detect(candImg, {});
  std::vector<bool>          cv;
  const auto                 cd0 = mola::vision::computeOrbDescriptors(candImg, cp, cv);
  std::vector<OrbDescriptor> cd;
  std::vector<TPoint3Df>     cx;
  for (size_t i = 0; i < cp.size(); ++i)
    if (cv[i])
    {
      cd.push_back(cd0[i]);
      cx.push_back({(cp[i].x - kCx) / kFx * kD, (cp[i].y - kCy) / kFy * kD, kD});
    }

  const auto                 qp = dist.detect(qImg, {});
  std::vector<bool>          qv;
  const auto                 qd0 = mola::vision::computeOrbDescriptors(qImg, qp, qv);
  std::vector<OrbDescriptor> qd;
  std::vector<TPoint2Df>     qx;
  for (size_t i = 0; i < qp.size(); ++i)
    if (qv[i])
    {
      qd.push_back(qd0[i]);
      qx.push_back(qp[i]);
    }

  const auto res = mola::vision::verifyLoopPnP(qd, qx, cd, cx, cam);
  EXPECT_FALSE(res.success);
}
