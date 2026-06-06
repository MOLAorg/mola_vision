/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_libvision/orb_descriptor.h>
#include <mola_libvision/pnp_solver.h>
#include <mrpt/img/TCamera.h>
#include <mrpt/math/TPoint2D.h>
#include <mrpt/math/TPoint3D.h>
#include <mrpt/poses/CPose3D.h>

#include <vector>

namespace mola::vision
{
/** Options for verifyLoopPnP(). */
struct LoopVerifyParams
{
  int       max_hamming = 60;  ///< descriptor match gate
  float     lowe_ratio  = 0.8f;  ///< descriptor ratio test
  int       min_inliers = 20;  ///< min PnP inliers to accept the loop
  PnPParams pnp;  ///< robust PnP options
};

/** Result of verifyLoopPnP(). */
struct LoopVerifyResult
{
  /** Pose of the QUERY camera expressed in the matched (candidate) keyframe's
   *  3D-point reference frame, i.e. the verified relative-pose loop constraint
   *  candidate -> query (X_query = relative_pose o X_candidate). */
  mrpt::poses::CPose3D relative_pose;
  int                  num_inliers = 0;
  int                  num_matches = 0;
  bool                 success     = false;
};

/** Geometrically verify a loop candidate and recover the relative pose.
 *
 *  Matches the query keyframe's ORB descriptors against the candidate
 *  keyframe's, forms 3D(candidate)-2D(query) correspondences from the matched
 *  candidate landmarks, and runs robust PnP. The candidate 3D points are in the
 *  candidate keyframe's reference frame, so the recovered camera pose is the
 *  relative-pose loop constraint between the two keyframes.
 *
 *  \param queryDesc,queryPx   Query keyframe descriptors and their pixels.
 *  \param candDesc,candXYZ    Candidate keyframe descriptors and the matching
 *                             3D landmark for each (same indexing as candDesc).
 *  \param cam                 Pinhole intrinsics (distortion ignored).
 */
[[nodiscard]] LoopVerifyResult verifyLoopPnP(
    const std::vector<OrbDescriptor>& queryDesc, const std::vector<mrpt::math::TPoint2Df>& queryPx,
    const std::vector<OrbDescriptor>& candDesc, const std::vector<mrpt::math::TPoint3Df>& candXYZ,
    const mrpt::img::TCamera& cam, const LoopVerifyParams& params = {});

}  // namespace mola::vision
