/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_libvision/Keyframe.h>

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mola::vision
{
/** Stores keyframes and maintains a covisibility graph.
 *
 *  Two keyframes are "covisible" when they observe common map points; the
 *  connection weight between them is the number of shared map points. The graph
 *  is recomputed on demand (updateCovisibility) from the current
 *  feature -> map_point associations of the stored keyframes.
 *
 *  Used for local-window selection, loop-closure candidate search, and
 *  relocalization.
 */
class KeyframeDatabase
{
 public:
  KeyframeDatabase() = default;

  /** Insert (or replace) a keyframe, keyed by its `id`. */
  void add(const Keyframe::Ptr& kf);

  /** Remove a keyframe by id (also drops its covisibility edges). */
  void remove(int kf_id);

  [[nodiscard]] Keyframe::Ptr get(int kf_id) const;
  [[nodiscard]] size_t        size() const { return keyframes_.size(); }
  [[nodiscard]] bool          empty() const { return keyframes_.empty(); }

  /** All stored keyframes (unspecified order). */
  [[nodiscard]] std::vector<Keyframe::Ptr> keyframes() const;

  /** Recompute the covisibility graph from current map-point observations.
   *  Edges with fewer than `min_shared` shared map points are dropped. */
  void updateCovisibility(int min_shared = 15);

  /** Number of map points shared between two keyframes (0 if no edge). Valid
   *  after updateCovisibility(). */
  [[nodiscard]] int connectionWeight(int kf_a, int kf_b) const;

  /** Covisible neighbors of `kf_id` with weight >= `min_weight`, returned as
   *  (neighbor_id, weight) sorted by descending weight. */
  [[nodiscard]] std::vector<std::pair<int, int>> covisibleKeyframes(
      int kf_id, int min_weight = 1) const;

 private:
  std::map<int, Keyframe::Ptr> keyframes_;

  // Symmetric covisibility weights: covis_[a][b] == covis_[b][a].
  std::unordered_map<int, std::unordered_map<int, int>> covis_;
};

}  // namespace mola::vision
