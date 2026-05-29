/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

namespace mola::vision
{
/** Policy parameters for KeyframeSelector. */
struct KeyframeSelectorParams
{
  /** Force a keyframe if at least this many frames passed since the last one. */
  int max_frames_gap = 20;

  /** Never create a keyframe before this many frames have elapsed (debounce). */
  int min_frames_gap = 1;

  /** Create a keyframe if the number of features still tracked from the last
   *  keyframe drops at or below this absolute count (tracking getting weak). */
  int min_tracked = 50;

  /** Create a keyframe if the fraction of the reference keyframe's features
   *  still tracked drops below this ratio. */
  float min_tracked_ratio = 0.5f;

  /** Create a keyframe once the median feature parallax (pixels) since the last
   *  keyframe exceeds this (enough viewpoint change for triangulation). */
  float min_parallax_px = 15.0f;
};

/** Per-frame statistics used to decide whether to spawn a new keyframe. */
struct KeyframeFrameStats
{
  int   num_tracked        = 0;  ///< features tracked from the last keyframe
  int   ref_num_features   = 0;  ///< features in the last keyframe
  float median_parallax_px = 0.f;  ///< median pixel motion vs. last keyframe
  int   frames_since_kf    = 0;  ///< frames elapsed since the last keyframe
};

/** Decides when a new keyframe should be inserted, from a configurable policy
 *  combining time/frame gap, tracking strength, and parallax.
 *
 *  Stateless: the caller supplies the per-frame statistics. Mirrors the
 *  keyframe-decision logic of lightweight_vio's Estimator, generalized.
 */
class KeyframeSelector
{
 public:
  explicit KeyframeSelector(const KeyframeSelectorParams& params = {}) : params_(params) {}

  [[nodiscard]] const KeyframeSelectorParams& params() const { return params_; }

  /** Returns true if `s` warrants inserting a new keyframe. */
  [[nodiscard]] bool shouldBeKeyframe(const KeyframeFrameStats& s) const
  {
    // Debounce: too soon after the previous keyframe.
    if (s.frames_since_kf < params_.min_frames_gap)
    {
      return false;
    }

    // Hard cap on the time between keyframes.
    if (s.frames_since_kf >= params_.max_frames_gap)
    {
      return true;
    }

    // Tracking is getting weak (absolute count).
    if (s.num_tracked <= params_.min_tracked)
    {
      return true;
    }

    // Tracking is getting weak (fraction of the reference keyframe).
    if (s.ref_num_features > 0)
    {
      const float ratio =
          static_cast<float>(s.num_tracked) / static_cast<float>(s.ref_num_features);
      if (ratio < params_.min_tracked_ratio)
      {
        return true;
      }
    }

    // Enough viewpoint change to triangulate well.
    if (s.median_parallax_px >= params_.min_parallax_px)
    {
      return true;
    }

    return false;
  }

 private:
  KeyframeSelectorParams params_;
};

}  // namespace mola::vision
