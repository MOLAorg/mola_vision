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

#include <cmath>

using namespace mola::vision;

namespace
{
[[nodiscard]] bool depthIsValid(float depth, const RGBDParams& p)
{
  return depth > 0.f && depth >= p.min_depth && depth <= p.max_depth;
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
