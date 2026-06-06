/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/img/CImage.h>
#include <mrpt/math/TPoint2D.h>

#include <array>
#include <cstdint>
#include <vector>

namespace mola::vision
{
/** A 256-bit ORB (oriented BRIEF) descriptor. */
using OrbDescriptor = std::array<uint8_t, 32>;

/** Parameters for ORB descriptor extraction. */
struct OrbParams
{
  /** Radius (pixels) of the circular patch used to estimate the keypoint
   *  orientation by intensity centroid. The standard ORB value is 15. */
  int orientation_radius = 15;

  /** Half-size needed around a keypoint to sample the (rotated) BRIEF pattern
   *  plus orientation patch; keypoints closer than this to the border are
   *  skipped. */
  int border = 20;
};

/** Keypoint orientation (radians) from the intensity centroid:
 *  theta = atan2(m01, m10) over a circular patch (Rosin moments). */
[[nodiscard]] float keypointOrientation(
    const mrpt::img::CImage& gray, const mrpt::math::TPoint2Df& pt, int radius = 15);

/** Compute the 256-bit oriented-BRIEF (ORB) descriptor at one keypoint.
 *  Returns false if the keypoint is too close to the image border.
 *
 *  \param gray  Grayscale (CH_GRAY) image.
 *  \param pt    Keypoint pixel.
 *  \param out   Output 32-byte descriptor.
 *
 *  \note BRIEF sampling pattern (orb_point_pairs) is from OpenCV (BSD-3-Clause).
 */
[[nodiscard]] bool computeOrbDescriptor(
    const mrpt::img::CImage& gray, const mrpt::math::TPoint2Df& pt, OrbDescriptor& out,
    const OrbParams& params = {});

/** Batch ORB descriptors. `valid[i]` is false for keypoints skipped at the
 *  border (their descriptor is left zero). */
[[nodiscard]] std::vector<OrbDescriptor> computeOrbDescriptors(
    const mrpt::img::CImage& gray, const std::vector<mrpt::math::TPoint2Df>& pts,
    std::vector<bool>& valid, const OrbParams& params = {});

/** Hamming distance (number of differing bits) between two ORB descriptors. */
[[nodiscard]] int hammingDistance(const OrbDescriptor& a, const OrbDescriptor& b);

}  // namespace mola::vision
