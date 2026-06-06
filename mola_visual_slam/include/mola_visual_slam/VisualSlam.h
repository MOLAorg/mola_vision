/* -------------------------------------------------------------------------
 * mola_visual_slam: monocular / stereo visual SLAM front-end for MOLA.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_kernel/interfaces/FrontEndBase.h>
#include <mola_kernel/interfaces/LocalizationSourceBase.h>
#include <mola_kernel/interfaces/MapSourceBase.h>
#include <mrpt/img/CImage.h>
#include <mrpt/img/TCamera.h>
#include <mrpt/math/TPoint2D.h>
#include <mrpt/math/TPoint3D.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/system/CTimeLogger.h>

#include <deque>
#include <string>
#include <vector>

namespace mola
{
/** Monocular or stereo visual SLAM front-end (`mode` = "mono" | "stereo").
 *
 *  Consumes `CObservationImage` and estimates the camera trajectory and a
 *  sparse 3D landmark map, all in MRPT 3.x types via `mola_libvision`
 *  (Shi-Tomasi detection, pyramidal LK tracking, essential-matrix two-view
 *  initialization, stereo matching, triangulation, robust PnP, sliding-window
 *  BA). No OpenCV, no Ceres.
 *
 *  Monocular pipeline:
 *   - INITIALIZING: detect corners in the first frame, track them with LK; once
 *     enough parallax accumulates, estimate the essential matrix (RANSAC),
 *     recover the relative pose, and triangulate the inliers into the initial
 *     map (world frame = first camera frame). The scale is arbitrary, fixed by
 *     the bootstrap baseline (||t|| = 1 between the two init keyframes).
 *   - TRACKING: LK + robust PnP; new landmarks by triangulating long-tracked
 *     features across keyframes; sliding-window BA.
 *
 *  Stereo pipeline (`mode=stereo`): pairs the left/right image streams by
 *  timestamp; depth comes directly from the stereo match (mola_libvision
 *  matchStereo), so the map and trajectory are at TRUE METRIC scale from the
 *  first frame (no essential bootstrap). LK + robust PnP for tracking; new
 *  landmarks from the stereo depth of fresh features. (Scale-anchored stereo BA
 *  is pending: see task 3.4, so windowed BA is currently skipped in stereo.)
 *
 *  Publishes the pose (LocalizationSourceBase) and sparse map (MapSourceBase).
 */
class VisualSlam : public mola::FrontEndBase,
                   public mola::LocalizationSourceBase,
                   public mola::MapSourceBase
{
  DEFINE_MRPT_OBJECT(VisualSlam, mola)

 public:
  VisualSlam()           = default;
  ~VisualSlam() override = default;

  VisualSlam(const VisualSlam&)                = delete;
  VisualSlam& operator=(const VisualSlam&)     = delete;
  VisualSlam(VisualSlam&&) noexcept            = delete;
  VisualSlam& operator=(VisualSlam&&) noexcept = delete;

  // See docs in base class
  void initialize_frontend(const Yaml& cfg) override;
  void spinOnce() override;
  void onNewObservation(const CObservation::ConstPtr& o) override;

  /** \name Test / introspection helpers (not part of the MOLA interface)
   *  @{ */

  /** Feeds one monocular frame directly. Returns the estimated camera-in-world
   *  pose (T_wc), up to the global scale fixed at initialization. */
  mrpt::poses::CPose3D processFrame(
      const mrpt::img::CImage& gray, const mrpt::img::TCamera& cam,
      const mrpt::Clock::time_point& timestamp);

  /** Feeds one RECTIFIED stereo pair directly (stereo mode). Depth comes from the
   *  stereo match, so the trajectory and map are at TRUE metric scale. \p cam is
   *  the (rectified) left camera; \p baseline is the stereo baseline in meters.
   *  Returns the estimated left-camera-in-world pose (T_wc). */
  mrpt::poses::CPose3D processStereoFrame(
      const mrpt::img::CImage& left, const mrpt::img::CImage& right, const mrpt::img::TCamera& cam,
      double baseline, const mrpt::Clock::time_point& timestamp);

  mrpt::poses::CPose3D currentPose() const { return pose_wc_; }
  bool                 isInitialized() const { return state_ == State::TRACKING; }
  size_t               numLandmarks() const { return landmarks_.size(); }
  size_t               numActiveLandmarks() const;
  size_t               numKeyframes() const { return keyframes_.size(); }

  /** Per-stage timing profiler. Dumps a full table on destruction; call
   *  `profiler().dumpAllStats()` to print it on demand. */
  mrpt::system::CTimeLogger&       profiler() { return profiler_; }
  const mrpt::system::CTimeLogger& profiler() const { return profiler_; }

  /** @} */

 private:
  enum class State
  {
    INITIALIZING,
    TRACKING
  };

  // ---- parameters ----
  std::string sensor_label_;
  std::string mode_ = "mono";  ///< "mono" or "stereo"
  // stereo mode:
  std::string left_label_      = "image_0";  ///< sensor label of the left image
  std::string right_label_     = "image_1";  ///< sensor label of the right image
  double      stereo_baseline_ = 0.537;  ///< stereo baseline [m] (default: KITTI)
  int         max_features_    = 400;
  float       min_distance_    = 12.0f;
  int         redetect_below_  = 150;
  int         lk_win_size_     = 21;
  int         lk_max_levels_   = 3;
  int         min_pnp_points_  = 12;
  int         ba_window_size_  = 8;
  int         cull_min_obs_    = 2;
  // two-view initialization:
  float init_min_parallax_px_ = 30.0f;  ///< median parallax to attempt bootstrap
  int   init_min_inliers_     = 50;  ///< min essential-matrix inliers to accept init
  // new-landmark triangulation:
  float tri_min_parallax_deg_ = 2.0f;  ///< min ray angle to triangulate a new point
  // keyframe policy:
  int   kf_max_frames_gap_    = 20;
  int   kf_min_frames_gap_    = 2;
  int   kf_min_tracked_       = 80;
  float kf_min_tracked_ratio_ = 0.6f;
  float kf_min_parallax_px_   = 18.0f;
  // viz:
  bool        publish_viz_2d_ = true;
  bool        publish_viz_3d_ = true;
  std::string viz2d_title_    = "Visual SLAM tracking";
  std::string viz2d_win_pos_;

  // ---- landmark map (world frame) ----
  struct Landmark
  {
    mrpt::math::TPoint3Df pos;
    int                   observations = 0;
    bool                  bad          = false;
  };
  std::vector<Landmark> landmarks_;

  struct KeyframeRec
  {
    mrpt::poses::CPose3D               pose_cw;
    std::vector<int>                   lm_index;
    std::vector<mrpt::math::TPoint2Df> pixel;
  };
  std::deque<KeyframeRec> keyframes_;

  // ---- tracking state ----
  State                              state_ = State::INITIALIZING;
  mrpt::img::CImage                  prev_gray_;
  std::vector<mrpt::math::TPoint2Df> track_pts_;  ///< current pixel per tracked feature
  std::vector<int>                   track_lm_;  ///< landmark index, or -1 if a candidate
  std::vector<mrpt::math::TPoint2Df> track_lastkf_pix_;  ///< pixel at the last keyframe
  std::vector<bool>                  track_has_lastkf_;  ///< feature existed at last keyframe
  mrpt::poses::CPose3D               pose_cw_;
  mrpt::poses::CPose3D               pose_wc_;
  std::vector<mrpt::math::TPoint3D>  trajectory_;
  mrpt::img::TCamera                 camera_;
  int                                frame_count_     = 0;
  int                                frames_since_kf_ = 0;
  int                                ref_kf_features_ = 0;
  bool                               gui_created_     = false;

  // ---- initialization buffer ----
  std::vector<mrpt::math::TPoint2Df> init_ref_pts_;  ///< feature pixels in the first frame

  // ---- stereo pairing buffer (mode=stereo) ----
  mrpt::img::CImage       pending_left_;
  mrpt::img::CImage       pending_right_;
  mrpt::img::TCamera      pending_left_cam_;
  mrpt::Clock::time_point pending_left_ts_{};
  mrpt::Clock::time_point pending_right_ts_{};
  bool                    have_left_  = false;
  bool                    have_right_ = false;

  // ---- profiling ----
  mrpt::system::CTimeLogger profiler_{true, "VisualSlam"};

  bool tryInitialize(const mrpt::img::CImage& gray);
  void detectInitialFeatures(const mrpt::img::CImage& gray);
  void trackAndLocalize(const mrpt::img::CImage& gray);
  /** Stereo-match a set of left features and append valid ones as new metric
   *  landmarks (world frame, via the current pose). Returns count added. */
  int addStereoLandmarks(
      const mrpt::img::CImage& left, const mrpt::img::CImage& right, const mrpt::img::TCamera& cam,
      double baseline, const std::vector<mrpt::math::TPoint2Df>& left_feats);
  void spawnTriangulatedLandmarks();
  void insertCurrentKeyframe();
  void runWindowedBA();
  void publishLocalization(const mrpt::Clock::time_point& timestamp);
  void publishMap(const mrpt::Clock::time_point& timestamp);
  void publishViz2D(const mrpt::img::CImage& gray);
  void publishViz3D();
};

}  // namespace mola
