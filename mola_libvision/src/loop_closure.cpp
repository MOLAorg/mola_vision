/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/loop_closure.h>

using namespace mola::vision;

LoopVerifyResult mola::vision::verifyLoopPnP(
    const std::vector<OrbDescriptor>& queryDesc, const std::vector<mrpt::math::TPoint2Df>& queryPx,
    const std::vector<OrbDescriptor>& candDesc, const std::vector<mrpt::math::TPoint3Df>& candXYZ,
    const mrpt::img::TCamera& cam, const LoopVerifyParams& params)
{
  LoopVerifyResult result;

  const auto matches =
      matchOrbDescriptors(queryDesc, candDesc, params.max_hamming, params.lowe_ratio);
  result.num_matches = static_cast<int>(matches.size());
  if (static_cast<int>(matches.size()) < params.min_inliers)
  {
    return result;
  }

  // 3D (candidate frame) <-> 2D (query pixels) correspondences.
  std::vector<mrpt::math::TPoint3Df> worldPts;
  std::vector<mrpt::math::TPoint2Df> pixels;
  worldPts.reserve(matches.size());
  pixels.reserve(matches.size());
  for (const auto& m : matches)
  {
    worldPts.push_back(candXYZ[static_cast<size_t>(m.train_idx)]);
    pixels.push_back(queryPx[static_cast<size_t>(m.query_idx)]);
  }

  const auto pnp = solvePnP(worldPts, pixels, cam, mrpt::poses::CPose3D::Identity(), params.pnp);
  if (!pnp.converged || pnp.num_inliers < params.min_inliers)
  {
    result.num_inliers = pnp.num_inliers;
    return result;
  }

  result.relative_pose = pnp.pose;
  result.num_inliers   = pnp.num_inliers;
  result.success       = true;
  return result;
}
