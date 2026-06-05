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
/** Monocular visual SLAM front-end (stereo mode reserved for a later phase).
 *
 *  Consumes `CObservationImage` and estimates the camera trajectory and a
 *  sparse 3D landmark map, all in MRPT 3.x types via `mola_libvision`
 *  (Shi-Tomasi detection, pyramidal LK tracking, essential-matrix two-view
 *  initialization, triangulation, robust PnP, sliding-window BA). No OpenCV,
 *  no Ceres.
 *
 *  Because a single camera gives no metric scale, the map and trajectory are
 *  expressed up to a global scale fixed by the bootstrap baseline (||t|| = 1
 *  between the two initialization keyframes).
 *
 *  Pipeline:
 *   - INITIALIZING: detect corners in the first frame, track them with LK; once
 *     enough parallax accumulates, estimate the essential matrix (RANSAC),
 *     recover the relative pose, and triangulate the inliers into the initial
 *     map (world frame = first camera frame).
 *   - TRACKING: track features with LK, recover the pose with robust PnP, spawn
 *     new landmarks by triangulating long-tracked features across keyframes, and
 *     refine recent keyframes + landmarks with sliding-window BA. Publishes the
 *     pose (LocalizationSourceBase) and sparse map (MapSourceBase).
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
  std::string mode_           = "mono";  ///< "mono" (stereo: future phase)
  int         max_features_   = 400;
  float       min_distance_   = 12.0f;
  int         redetect_below_ = 150;
  int         lk_win_size_    = 21;
  int         lk_max_levels_  = 3;
  int         min_pnp_points_ = 12;
  int         ba_window_size_ = 8;
  int         cull_min_obs_   = 2;
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

  // ---- profiling ----
  mrpt::system::CTimeLogger profiler_{true, "VisualSlam"};

  bool tryInitialize(const mrpt::img::CImage& gray);
  void detectInitialFeatures(const mrpt::img::CImage& gray);
  void trackAndLocalize(const mrpt::img::CImage& gray);
  void spawnTriangulatedLandmarks();
  void insertCurrentKeyframe();
  void runWindowedBA();
  void publishLocalization(const mrpt::Clock::time_point& timestamp);
  void publishMap(const mrpt::Clock::time_point& timestamp);
  void publishViz2D(const mrpt::img::CImage& gray);
  void publishViz3D();
};

}  // namespace mola
