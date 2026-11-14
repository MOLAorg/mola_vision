/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Quadratic depth-uncertainty model and back-projection adapted from
 * lightweight_vio (MIT, (c) 2025 Seungwon Choi), Frame RGBD path; ported to
 * MRPT types without OpenCV.
 * ------------------------------------------------------------------------- */
#include <mola_libvision/rgbd_depth.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace mola::vision;

namespace
{
[[nodiscard]] bool depthIsValid(float depth, const RGBDParams& p)
{
  return depth > 0.f && depth >= p.min_depth && depth <= p.max_depth;
}

// floor() as an int, avoiding a libm call in hot loops.
[[nodiscard]] inline int floorToInt(float x)
{
  const int i = static_cast<int>(x);
  return x < static_cast<float>(i) ? i - 1 : i;
}
}  // namespace

float mola::vision::depthStdDev(float depth, const RGBDParams& params)
{
  const float var = params.unc_a * depth * depth + params.unc_b * depth + params.unc_c;
  return std::sqrt(std::max(0.f, var));
}

std::optional<mrpt::math::TPoint3Df> mola::vision::backprojectPixel(
    const mrpt::math::TPoint2Df& pixel, float depth, const mrpt::img::TCamera& cam,
    const RGBDParams& params)
{
  if (!depthIsValid(depth, params))
  {
    return std::nullopt;
  }
  const float fx = static_cast<float>(cam.fx());
  const float fy = static_cast<float>(cam.fy());
  const float cx = static_cast<float>(cam.cx());
  const float cy = static_cast<float>(cam.cy());

  mrpt::math::TPoint3Df pt;
  pt.x = (pixel.x - cx) * depth / fx;
  pt.y = (pixel.y - cy) * depth / fy;
  pt.z = depth;
  return pt;
}

DepthCloud mola::vision::backprojectDepthMap(
    const Eigen::MatrixXf& depth_m, const mrpt::img::TCamera& cam, const RGBDParams& params,
    int decimation)
{
  DepthCloud  cloud;
  const int   step = std::max(1, decimation);
  const float fx   = static_cast<float>(cam.fx());
  const float fy   = static_cast<float>(cam.fy());
  const float cx   = static_cast<float>(cam.cx());
  const float cy   = static_cast<float>(cam.cy());

  const int rows = static_cast<int>(depth_m.rows());
  const int cols = static_cast<int>(depth_m.cols());

  for (int v = 0; v < rows; v += step)
  {
    for (int u = 0; u < cols; u += step)
    {
      const float d = depth_m(v, u);
      if (!depthIsValid(d, params))
      {
        continue;
      }
      mrpt::math::TPoint3Df pt;
      pt.x = (static_cast<float>(u) - cx) * d / fx;
      pt.y = (static_cast<float>(v) - cy) * d / fy;
      pt.z = d;
      cloud.points.push_back(pt);
      cloud.pixels.push_back({static_cast<float>(u), static_cast<float>(v)});
    }
  }
  return cloud;
}

mrpt::math::CMatrixFloat mola::vision::registerDepthToCamera(
    const mrpt::math::CMatrix_u16& depth_raw, float depth_units,
    const mrpt::img::TCamera& depth_cam, const mrpt::img::TCamera& target_cam,
    const mrpt::poses::CPose3D& target_from_depth)
{
  const int Hd = static_cast<int>(depth_raw.rows());
  const int Wd = static_cast<int>(depth_raw.cols());
  const int Ht = static_cast<int>(target_cam.nrows);
  const int Wt = static_cast<int>(target_cam.ncols);

  constexpr float          kEmpty = std::numeric_limits<float>::max();
  mrpt::math::CMatrixFloat out(Ht, Wt);
  out.asEigen().setConstant(kEmpty);

  // Depth-camera ray of pixel (r,c), already rotated into the target camera:
  //   ray(r,c) = R * [xn(c), yn(r), 1] = colPart(c) + rowPart(r)
  // so that a point at depth z lands at z * ray + t.
  const Eigen::Matrix3f R = target_from_depth.getRotationMatrix().asEigen().cast<float>();
  const Eigen::Vector3f t(
      static_cast<float>(target_from_depth.x()), static_cast<float>(target_from_depth.y()),
      static_cast<float>(target_from_depth.z()));

  const float fxd = static_cast<float>(depth_cam.fx());
  const float fyd = static_cast<float>(depth_cam.fy());
  const float cxd = static_cast<float>(depth_cam.cx());
  const float cyd = static_cast<float>(depth_cam.cy());

  std::vector<Eigen::Vector3f> colPart(static_cast<size_t>(Wd));
  for (int c = 0; c < Wd; ++c)
  {
    colPart[static_cast<size_t>(c)] = R.col(0) * ((static_cast<float>(c) - cxd) / fxd);
  }
  std::vector<Eigen::Vector3f> rowPart(static_cast<size_t>(Hd));
  for (int r = 0; r < Hd; ++r)
  {
    rowPart[static_cast<size_t>(r)] = R.col(1) * ((static_cast<float>(r) - cyd) / fyd) + R.col(2);
  }

  const float fxt = static_cast<float>(target_cam.fx());
  const float fyt = static_cast<float>(target_cam.fy());
  const float cxt = static_cast<float>(target_cam.cx());
  const float cyt = static_cast<float>(target_cam.cy());

  // Size of the splatted rectangle, in target pixels, per axis (bounded, so a
  // wrong calibration cannot make it unreasonably costly):
  const auto splat = [](float ratio)
  { return std::clamp(static_cast<int>(std::ceil(ratio - 1e-3f)), 1, 8); };
  const int   nx     = splat(fxt / fxd);
  const int   ny     = splat(fyt / fyd);
  const float halfX  = 0.5f * static_cast<float>(nx - 1);
  const float halfY  = 0.5f * static_cast<float>(ny - 1);
  const float margin = static_cast<float>(std::max(nx, ny));

  for (int r = 0; r < Hd; ++r)
  {
    for (int c = 0; c < Wd; ++c)
    {
      const uint16_t raw = depth_raw(r, c);
      if (raw == 0)
      {
        continue;
      }
      const float           z = static_cast<float>(raw) * depth_units;
      const Eigen::Vector3f P =
          z * (colPart[static_cast<size_t>(c)] + rowPart[static_cast<size_t>(r)]) + t;
      if (P.z() <= 0.f)
      {
        continue;
      }
      const float invZ = 1.0f / P.z();
      // Top-left pixel of the splatted rectangle (the nearest one, if 1x1):
      const float uf = fxt * P.x() * invZ + cxt - halfX + 0.5f;
      const float vf = fyt * P.y() * invZ + cyt - halfY + 0.5f;
      // Also keeps the int conversion below in range (and rejects NaN):
      if (!(uf > -margin && vf > -margin && uf < static_cast<float>(Wt) + margin &&
            vf < static_cast<float>(Ht) + margin))
      {
        continue;
      }
      const int u0 = floorToInt(uf);
      const int v0 = floorToInt(vf);
      for (int dv = 0; dv < ny; ++dv)
      {
        const int vv = v0 + dv;
        if (vv < 0 || vv >= Ht)
        {
          continue;
        }
        for (int du = 0; du < nx; ++du)
        {
          const int uu = u0 + du;
          if (uu < 0 || uu >= Wt)
          {
            continue;
          }
          float& cell = out(vv, uu);
          cell        = std::min(cell, P.z());
        }
      }
    }
  }

  for (int r = 0; r < Ht; ++r)
  {
    for (int c = 0; c < Wt; ++c)
    {
      if (out(r, c) == kEmpty)
      {
        out(r, c) = 0.f;
      }
    }
  }
  return out;
}

const mrpt::poses::CPose3D& mola::vision::opticalFrameInXForwardFrame()
{
  static const mrpt::poses::CPose3D p(0, 0, 0, -M_PI / 2, 0, -M_PI / 2);
  return p;
}
