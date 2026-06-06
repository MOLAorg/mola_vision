/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_libvision/loop_detector.h>
#include <mola_libvision/orb_descriptor.h>

#include <map>
#include <vector>

namespace mola::vision
{
/** Parameters for OrbLoopDetector. */
struct OrbLoopParams
{
  int         max_features   = 500;  ///< grid corners detected per keyframe
  float       min_distance   = 10.f;  ///< NMS spacing for detection
  int         max_hamming    = 60;  ///< max ORB Hamming distance for a descriptor match
  float       lowe_ratio     = 0.8f;  ///< best/second-best distance ratio test
  int         min_matches    = 25;  ///< min matched descriptors to report a candidate
  std::size_t max_candidates = 3;  ///< top-K candidates returned per query
};

/** Concrete visual place-recognition by direct ORB descriptor matching
 *  ("descriptor voting"): each keyframe is summarized by a set of ORB
 *  descriptors at grid-distributed corners; two keyframes are scored by the
 *  number of mutually-consistent descriptor matches (Hamming + Lowe ratio).
 *
 *  Simple and dependency-free (no DBoW2); O(stored_keyframes) per query with a
 *  brute-force descriptor match. Geometric verification (e.g. PnP / essential)
 *  of the returned candidates remains the caller's responsibility.
 */
class OrbLoopDetector : public LoopDetector
{
 public:
  explicit OrbLoopDetector(const OrbLoopParams& params = {}) : params_(params) {}

  void                                     addKeyframe(const Keyframe::Ptr& kf) override;
  [[nodiscard]] std::vector<LoopCandidate> detect(
      const Keyframe::Ptr& kf, int min_id_gap = 30) override;
  void clear() override;

  /** Number of keyframes in the database. */
  [[nodiscard]] std::size_t size() const { return db_.size(); }

  /** Compute the descriptor set for a keyframe's image (exposed for reuse). */
  [[nodiscard]] std::vector<OrbDescriptor> describe(const Keyframe::Ptr& kf) const;

  /** Count mutually-consistent descriptor matches between two sets. */
  [[nodiscard]] int countMatches(
      const std::vector<OrbDescriptor>& a, const std::vector<OrbDescriptor>& b) const;

 private:
  OrbLoopParams                             params_;
  std::map<int, std::vector<OrbDescriptor>> db_;  ///< keyframe id -> descriptors
};

}  // namespace mola::vision
