/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_libvision/Keyframe.h>

#include <memory>
#include <vector>

namespace mola::vision
{
/** A loop-closure candidate: a past keyframe similar to the query. */
struct LoopCandidate
{
  int   query_kf_id   = -1;
  int   matched_kf_id = -1;
  float score         = 0.f;  ///< similarity score (higher = better)
};

/** Abstract visual place-recognition interface (loop detection).
 *
 *  Concrete back ends (DBoW2, learned global descriptors such as NetVLAD, ...)
 *  implement keyframe insertion and candidate retrieval. Geometric verification
 *  of the returned candidates is the caller's responsibility (e.g. via PnP).
 *
 *  This is the abstract base only; concrete implementations are deferred to the
 *  loop-closure phase.
 */
class LoopDetector
{
 public:
  using Ptr = std::shared_ptr<LoopDetector>;

  LoopDetector()                               = default;
  LoopDetector(const LoopDetector&)            = default;
  LoopDetector(LoopDetector&&)                 = default;
  LoopDetector& operator=(const LoopDetector&) = default;
  LoopDetector& operator=(LoopDetector&&)      = default;
  virtual ~LoopDetector()                      = default;

  /** Insert a keyframe into the recognition database (compute + store its
   *  descriptor). */
  virtual void addKeyframe(const Keyframe::Ptr& kf) = 0;

  /** Retrieve loop candidates for `kf`, sorted by descending score. Candidates
   *  within `min_id_gap` keyframe ids of the query should be excluded
   *  (temporally adjacent frames are not loops). */
  [[nodiscard]] virtual std::vector<LoopCandidate> detect(
      const Keyframe::Ptr& kf, int min_id_gap = 30) = 0;

  /** Forget all stored keyframes. */
  virtual void clear() = 0;
};

}  // namespace mola::vision
