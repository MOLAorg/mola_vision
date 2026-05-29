/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Adapted from lightweight_vio (MIT, (c) 2025 Seungwon Choi)
 * Frame::compute_stereo_matches(); simplified for rectified pairs and ported to
 * MRPT types without OpenCV.
 * ------------------------------------------------------------------------- */
#include <mola_libvision/stereo_matcher.h>

#include <cmath>

using namespace mola::vision;

StereoMatchResult mola::vision::matchStereo(
    const mrpt::img::CImage& left, const mrpt::img::CImage& right,
    const std::vector<mrpt::math::TPoint2Df>& left_pts, double fx, double baseline,
    const StereoMatchParams& params)
{
  StereoMatchResult res;
  const size_t      N = left_pts.size();
  res.right_pts.assign(N, {-1.f, -1.f});
  res.depth.assign(N, -1.f);
  res.valid.assign(N, false);

  if (N == 0)
  {
    return res;
  }

  // Left -> right pyramidal LK tracking. The right point is seeded at the left
  // location (zero initial disparity), which is what calcOpticalFlowPyrLK does
  // internally when next_pts is sized from prev_pts.
  std::vector<mrpt::math::TPoint2Df> right_pts;
  std::vector<TrackStatus>           status;
  calcOpticalFlowPyrLK(left, right, left_pts, right_pts, status, params.lk);

  const float max_disp =
      (params.max_disparity > 0.f) ? params.max_disparity : static_cast<float>(left.getWidth());

  for (size_t i = 0; i < N; ++i)
  {
    if (status[i] != TrackStatus::OK)
    {
      continue;
    }

    const float y_diff = std::abs(left_pts[i].y - right_pts[i].y);
    if (y_diff > params.max_y_diff)
    {
      continue;  // violates the (rectified) epipolar constraint
    }

    // Disparity is positive when the right-image match is to the LEFT of the
    // left-image feature (standard rectified stereo convention).
    const float disparity = left_pts[i].x - right_pts[i].x;
    if (disparity < params.min_disparity || disparity > max_disp)
    {
      continue;
    }

    res.right_pts[i] = right_pts[i];
    res.depth[i]     = static_cast<float>(fx * baseline / static_cast<double>(disparity));
    res.valid[i]     = true;
    ++res.num_valid;
  }

  return res;
}
