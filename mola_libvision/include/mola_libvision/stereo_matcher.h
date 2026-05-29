/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_libvision/optical_flow.h>
#include <mrpt/img/CImage.h>
#include <mrpt/math/TPoint2D.h>

#include <vector>

namespace mola::vision
{
/** Parameters for matchStereo(). */
struct StereoMatchParams
{
  /** Optical-flow parameters for the left -> right search. A wide window and a
   *  couple of pyramid levels work well for stereo. */
  LKParams lk = []
  {
    LKParams p;
    p.win_size   = 11;
    p.max_levels = 3;
    return p;
  }();

  /** Epipolar tolerance for a *rectified* pair: the matched right point must lie
   *  within +/- this many pixels of the left point's row. */
  float max_y_diff = 2.0f;

  /** Minimum valid disparity (pixels). Disparity = x_left - x_right. */
  float min_disparity = 1.0f;

  /** Maximum valid disparity (pixels). <= 0 means "no upper bound". */
  float max_disparity = 0.0f;
};

/** Result of matchStereo(): one entry per input left feature. */
struct StereoMatchResult
{
  /** Matched coordinate in the right image; (-1,-1) if no valid match. */
  std::vector<mrpt::math::TPoint2Df> right_pts;

  /** Metric depth in meters; < 0 if no valid match. */
  std::vector<float> depth;

  /** Per-feature validity flag. */
  std::vector<bool> valid;

  int num_valid = 0;
};

/** Stereo matching on a RECTIFIED pair: for each feature in the left image,
 *  find its correspondence in the right image and compute metric depth.
 *
 *  Method: pyramidal Lucas-Kanade tracking from left to right (reusing
 *  calcOpticalFlowPyrLK), then validation for a rectified geometry:
 *    - tracking succeeded,
 *    - row difference |y_left - y_right| <= max_y_diff (epipolar line),
 *    - disparity d = x_left - x_right within [min_disparity, max_disparity].
 *  Depth is then  Z = fx * baseline / d.
 *
 *  \param left,right   Rectified, undistorted grayscale images (CH_GRAY).
 *                      Rectify with mrpt::img::CStereoRectifyMap upstream.
 *  \param left_pts     Feature pixels detected in the left image.
 *  \param fx           Horizontal focal length (pixels) of the rectified camera.
 *  \param baseline     Stereo baseline (meters), > 0.
 *  \param params       Matching options.
 *
 *  \note Adapted from lightweight_vio (MIT, (c) 2025 Seungwon Choi)
 *        Frame::compute_stereo_matches(), simplified for the rectified case and
 *        ported to MRPT types without OpenCV.
 */
StereoMatchResult matchStereo(
    const mrpt::img::CImage& left, const mrpt::img::CImage& right,
    const std::vector<mrpt::math::TPoint2Df>& left_pts, double fx, double baseline,
    const StereoMatchParams& params = {});

}  // namespace mola::vision
