/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Note: the bundled ORB descriptor bit-pattern table in orb_point_pairs.h is
 * from OpenCV (BSD-3-Clause) and retains its original license notice.
 * ------------------------------------------------------------------------- */
#include <mola_libvision/orb_descriptor.h>

#include <cmath>

#include "./orb_point_pairs.h"

using namespace mola::vision;

namespace
{
inline float pix(const uint8_t* data, int stride, int x, int y)
{
  return static_cast<float>(data[static_cast<size_t>(y) * stride + x]);
}

inline float bilinear(const uint8_t* data, int stride, int cols, int rows, float x, float y)
{
  int x0 = static_cast<int>(std::floor(x));
  int y0 = static_cast<int>(std::floor(y));
  if (x0 < 0)
  {
    x0 = 0;
  }
  if (y0 < 0)
  {
    y0 = 0;
  }
  if (x0 >= cols - 1)
  {
    x0 = cols - 2;
  }
  if (y0 >= rows - 1)
  {
    y0 = rows - 2;
  }
  const float fx  = x - static_cast<float>(x0);
  const float fy  = y - static_cast<float>(y0);
  const float p00 = pix(data, stride, x0, y0);
  const float p10 = pix(data, stride, x0 + 1, y0);
  const float p01 = pix(data, stride, x0, y0 + 1);
  const float p11 = pix(data, stride, x0 + 1, y0 + 1);
  return (1 - fy) * ((1 - fx) * p00 + fx * p10) + fy * ((1 - fx) * p01 + fx * p11);
}
}  // namespace

float mola::vision::keypointOrientation(
    const mrpt::img::CImage& gray, const mrpt::math::TPoint2Df& pt, int radius)
{
  const uint8_t* data   = gray.ptr<uint8_t>(0, 0);
  const int      stride = static_cast<int>(gray.getRowStride());
  const int      cols   = static_cast<int>(gray.getWidth());
  const int      rows   = static_cast<int>(gray.getHeight());

  const int cx = static_cast<int>(std::lround(pt.x));
  const int cy = static_cast<int>(std::lround(pt.y));
  const int r2 = radius * radius;

  double m10 = 0.0;
  double m01 = 0.0;
  for (int dy = -radius; dy <= radius; ++dy)
  {
    const int y = cy + dy;
    if (y < 0 || y >= rows)
    {
      continue;
    }
    for (int dx = -radius; dx <= radius; ++dx)
    {
      const int x = cx + dx;
      if (x < 0 || x >= cols || dx * dx + dy * dy > r2)
      {
        continue;
      }
      const double I = pix(data, stride, x, y);
      m10 += dx * I;
      m01 += dy * I;
    }
  }
  return static_cast<float>(std::atan2(m01, m10));
}

bool mola::vision::computeOrbDescriptor(
    const mrpt::img::CImage& gray, const mrpt::math::TPoint2Df& pt, OrbDescriptor& out,
    const OrbParams& params)
{
  const int cols = static_cast<int>(gray.getWidth());
  const int rows = static_cast<int>(gray.getHeight());
  const int bcx  = static_cast<int>(std::lround(pt.x));
  const int bcy  = static_cast<int>(std::lround(pt.y));
  if (bcx < params.border || bcy < params.border || bcx >= cols - params.border ||
      bcy >= rows - params.border)
  {
    return false;
  }

  const uint8_t* data   = gray.ptr<uint8_t>(0, 0);
  const int      stride = static_cast<int>(gray.getRowStride());

  const float theta = keypointOrientation(gray, pt, params.orientation_radius);
  const float ca    = std::cos(theta);
  const float sa    = std::sin(theta);

  out.fill(0);
  for (int i = 0; i < 256; ++i)
  {
    const float x1 = orb_point_pairs[4 * i + 0];
    const float y1 = orb_point_pairs[4 * i + 1];
    const float x2 = orb_point_pairs[4 * i + 2];
    const float y2 = orb_point_pairs[4 * i + 3];

    // Steer the BRIEF test points by the keypoint orientation.
    const float rx1 = ca * x1 - sa * y1;
    const float ry1 = sa * x1 + ca * y1;
    const float rx2 = ca * x2 - sa * y2;
    const float ry2 = sa * x2 + ca * y2;

    const float i1 = bilinear(data, stride, cols, rows, pt.x + rx1, pt.y + ry1);
    const float i2 = bilinear(data, stride, cols, rows, pt.x + rx2, pt.y + ry2);

    if (i1 < i2)
    {
      out[static_cast<size_t>(i >> 3)] |= static_cast<uint8_t>(1u << (i & 7));
    }
  }
  return true;
}

std::vector<OrbDescriptor> mola::vision::computeOrbDescriptors(
    const mrpt::img::CImage& gray, const std::vector<mrpt::math::TPoint2Df>& pts,
    std::vector<bool>& valid, const OrbParams& params)
{
  std::vector<OrbDescriptor> out(pts.size());
  valid.assign(pts.size(), false);
  for (size_t i = 0; i < pts.size(); ++i)
  {
    valid[i] = computeOrbDescriptor(gray, pts[i], out[i], params);
  }
  return out;
}

int mola::vision::hammingDistance(const OrbDescriptor& a, const OrbDescriptor& b)
{
  int d = 0;
  for (size_t i = 0; i < a.size(); ++i)
  {
    d += __builtin_popcount(static_cast<unsigned>(a[i] ^ b[i]));
  }
  return d;
}

std::vector<DescriptorMatch> mola::vision::matchOrbDescriptors(
    const std::vector<OrbDescriptor>& query, const std::vector<OrbDescriptor>& train,
    int max_hamming, float lowe_ratio)
{
  std::vector<DescriptorMatch> out;
  if (train.size() < 2)
  {
    return out;
  }
  for (size_t i = 0; i < query.size(); ++i)
  {
    int best   = 1 << 30;
    int second = 1 << 30;
    int best_j = -1;
    for (size_t j = 0; j < train.size(); ++j)
    {
      const int d = hammingDistance(query[i], train[j]);
      if (d < best)
      {
        second = best;
        best   = d;
        best_j = static_cast<int>(j);
      }
      else if (d < second)
      {
        second = d;
      }
    }
    if (best <= max_hamming && static_cast<float>(best) < lowe_ratio * static_cast<float>(second))
    {
      out.push_back({static_cast<int>(i), best_j, best});
    }
  }
  return out;
}
