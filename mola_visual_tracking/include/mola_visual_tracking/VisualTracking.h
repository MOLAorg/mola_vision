/* -------------------------------------------------------------------------
 * mola_visual_tracking: a MOLA demo front-end that detects and tracks image
 * features and shows them in a MolaViz subwindow (no 3D reconstruction).
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_kernel/interfaces/FrontEndBase.h>
#include <mrpt/img/CImage.h>
#include <mrpt/math/TPoint2D.h>

#include <deque>
#include <map>
#include <string>
#include <vector>

namespace mola
{
/** A minimal visual-tracking front-end for demos / early testing.
 *
 *  Consumes monocular `CObservationImage` observations, detects Shi-Tomasi
 *  corners (mola_libvision) and tracks them frame-to-frame with pyramidal
 *  Lucas-Kanade optical flow, optionally rejecting outliers with a fundamental
 *  matrix RANSAC filter. The current frame, annotated with the tracked features
 *  and their motion trails, is pushed to a MolaViz subwindow.
 *
 *  This module does NOT do any 3D reconstruction or localization: it only
 *  exercises and visualizes the feature detection + tracking pipeline.
 */
class VisualTracking : public mola::FrontEndBase
{
  DEFINE_MRPT_OBJECT(VisualTracking, mola)

 public:
  VisualTracking()           = default;
  ~VisualTracking() override = default;

  VisualTracking(const VisualTracking&)                = delete;
  VisualTracking& operator=(const VisualTracking&)     = delete;
  VisualTracking(VisualTracking&&) noexcept            = delete;
  VisualTracking& operator=(VisualTracking&&) noexcept = delete;

  // See docs in base class
  void initialize_frontend(const Yaml& cfg) override;
  void spinOnce() override;
  void onNewObservation(const CObservation::Ptr& o) override;

 private:
  // --- parameters ---
  std::string sensor_label_;  ///< only process images with this label (empty = any)
  std::string subwindow_title_ = "Feature tracking";
  std::string win_pos_;  ///< "x y width height" (optional)
  int         max_features_   = 300;
  float       min_distance_   = 12.0f;
  int         redetect_below_ = 150;  ///< re-detect when tracked count drops below this
  int         lk_win_size_    = 21;
  int         lk_max_levels_  = 3;
  int         trail_length_   = 12;
  bool        apply_fmatrix_  = true;

  // --- state ---
  mrpt::img::CImage                                prev_gray_;
  std::vector<mrpt::math::TPoint2Df>               pts_;
  std::vector<int>                                 ids_;
  std::map<int, std::deque<mrpt::math::TPoint2Df>> trails_;
  int                                              next_id_     = 0;
  bool                                             gui_created_ = false;

  void processImage(const mrpt::img::CImage& image, const mrpt::img::TCamera& cam);
  void publishViz(const mrpt::img::CImage& rgb, const std::vector<bool>& is_outlier);
};

}  // namespace mola
