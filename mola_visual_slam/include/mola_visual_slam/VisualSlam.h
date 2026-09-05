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
#include <mrpt/img/CStereoRectifyMap.h>
#include <mrpt/img/TCamera.h>
#include <mrpt/math/TPoint2D.h>
#include <mrpt/math/TPoint3D.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/system/CTimeLogger.h>

#include <deque>
#include <optional>
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
 *  landmarks from the stereo depth of fresh features. Windowed BA uses the
 *  stereo disparity residual (d = fx*baseline/Z) which directly constrains
 *  landmark depth and anchors metric scale (a pure-reprojection BA is degenerate
 *  for forward motion). Note: windowed BA only corrects LOCAL drift; global
 *  drift over long sequences needs loop closure / global BA (Phase 5).
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

  /** Estimated pose of the LEFT camera, in the frame the left camera had at
   *  the first processed frame (T_wc). When stereo rectification is active
   *  (see \c right_camera_pose) the internal estimate lives in the rectified
   *  frame, which is rotated with respect to the physical camera; this getter
   *  undoes that rotation, so the returned trajectory is always the physical
   *  left camera's, whether rectification is on or off. */
  [[nodiscard]] mrpt::poses::CPose3D currentPose() const;

  /** Estimated pose of the VEHICLE body, in the frame the body had at the
   *  first processed frame. Requires the camera-on-robot extrinsic, which is
   *  taken from the incoming observations' \c cameraPose (as filled from /tf
   *  or a fixed sensor pose by the dataset source) or from the
   *  \c camera_pose_on_robot parameter. Without it this falls back to
   *  currentPose(), i.e. the camera's own trajectory. */
  [[nodiscard]] mrpt::poses::CPose3D currentRobotPose() const;

  /** Whether the camera-on-robot extrinsic is known, i.e. whether
   *  currentRobotPose() is really a body pose and not the camera's. */
  [[nodiscard]] bool hasRobotExtrinsics() const { return camera_pose_on_robot_.has_value(); }

  bool   isInitialized() const { return state_ == State::TRACKING; }
  size_t numLandmarks() const { return landmarks_.size(); }
  size_t numActiveLandmarks() const;
  size_t numKeyframes() const { return keyframes_.size(); }

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
  /** Right camera pose relative to the left camera ("x y z yaw_deg pitch_deg
   *  roll_deg"), for rigs whose raw images are NOT pre-rectified (unlike
   *  KITTI's image_0/image_1). When set, the two cameras' own \c cameraParams
   *  (as carried by each CObservationImage, e.g. from a CameraInfo topic) are
   *  used to build a CStereoRectifyMap once, and every incoming pair is
   *  rectified before being handed to processStereoFrame(). Leave empty (the
   *  default) to keep the old behavior: raw images are assumed already
   *  rectified and sharing the left camera's intrinsics. */
  std::string right_camera_pose_str_;
  /** Undo a 180-degree rotation baked into an incoming stream. Some rigs
   *  physically mount one of the two cameras upside down (and some datasets
   *  ship one stream rotated while their calibration describes the upright
   *  image). Either way the pair cannot be rectified as-is: no rotation of the
   *  rectified frame can align two images whose rows run in opposite
   *  directions, so every stereo match fails. Rotating the affected stream back
   *  restores agreement with the calibration, which is why the intrinsics need
   *  no adjustment here. */
  bool left_image_rotate_180_  = false;
  bool right_image_rotate_180_ = false;
  /** Left camera pose on the vehicle ("x y z yaw_deg pitch_deg roll_deg"),
   *  overriding whatever the incoming observations carry in their
   *  \c cameraPose field. Only used to report body-frame poses; it has no
   *  effect on the visual estimation itself. */
  std::string camera_pose_on_robot_str_;
  int         max_features_   = 400;
  float       min_distance_   = 12.0f;
  int         redetect_below_ = 150;
  int         lk_win_size_    = 21;
  int         lk_max_levels_  = 3;
  int         min_pnp_points_ = 12;
  /** Minimum fraction of the 3D-2D correspondences that PnP must explain for
   *  its pose to be accepted. A solve that fits only a small minority of them
   *  has converged to a wrong local minimum; taking it corrupts the map on the
   *  next frame, from which nothing recovers. */
  float min_pnp_inlier_ratio_ = 0.35f;
  /** Consecutive frames without a localizable pose before the local map is
   *  rebuilt from scratch at the dead-reckoned pose. */
  int lost_max_frames_ = 2;
  int ba_window_size_  = 8;
  int cull_min_obs_    = 2;
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
    std::vector<float>                 disparity;  ///< measured stereo disparity (-1 if none)
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
  mrpt::poses::CPose3D               last_motion_;  ///< previous frame-to-frame motion (T_wc)
  bool                               have_motion_ = false;
  std::vector<mrpt::math::TPoint3D>  trajectory_;
  mrpt::img::TCamera                 camera_;
  int                                frame_count_         = 0;
  int                                frames_since_kf_     = 0;
  int                                frames_without_pose_ = 0;
  int                                ref_kf_features_     = 0;
  bool                               gui_created_         = false;
  double                             cur_baseline_ = 0.0;  ///< stereo baseline [m] (0 = mono)

  // ---- initialization buffer ----
  std::vector<mrpt::math::TPoint2Df> init_ref_pts_;  ///< feature pixels in the first frame

  // ---- stereo pairing buffer (mode=stereo) ----
  mrpt::img::CImage       pending_left_;
  mrpt::img::CImage       pending_right_;
  mrpt::img::TCamera      pending_left_cam_;
  mrpt::img::TCamera      pending_right_cam_;
  mrpt::Clock::time_point pending_left_ts_{};
  mrpt::Clock::time_point pending_right_ts_{};
  bool                    have_left_  = false;
  bool                    have_right_ = false;

  // ---- optional stereo rectification (unrectified rigs only) ----
  std::optional<mrpt::poses::CPose3D>
                               right_camera_pose_;  ///< parsed once, from right_camera_pose_str_
  mrpt::img::CStereoRectifyMap rectify_map_;
  /** Pose of the ORIGINAL left camera frame expressed in the RECTIFIED frame
   *  (pure rotation). Empty while no rectification is in use. */
  std::optional<mrpt::poses::CPose3D> left_cam_in_rectified_;

  // ---- extrinsics for reporting body-frame poses ----
  std::optional<mrpt::poses::CPose3D> camera_pose_on_robot_;  ///< T_body_leftcam

  // ---- profiling ----
  mrpt::system::CTimeLogger profiler_{true, "VisualSlam"};

  /** Latches the camera-on-robot extrinsic from an incoming observation, if it
   *  is not already known from the \c camera_pose_on_robot parameter. */
  void rememberCameraPoseOnRobot(const mrpt::poses::CPose3D& p);
  bool tryInitialize(const mrpt::img::CImage& gray);
  void detectInitialFeatures(const mrpt::img::CImage& gray);
  /** Constant-velocity extrapolation of the camera-in-world pose. */
  [[nodiscard]] mrpt::poses::CPose3D predictedPoseWc() const;

  /** Localizes the current frame from 3D-2D correspondences (robust PnP, seeded
   *  by a constant-velocity prediction) and updates pose_cw_ / pose_wc_ /
   *  last_motion_. \p corr_idx maps each correspondence back to its index in the
   *  per-feature tracking arrays, so rejected ones can be flagged in
   *  \p pnp_outlier (sized as those arrays). Returns false if the pose could not
   *  be estimated with enough support, in which case the prediction is used. */
  bool solveFramePose(
      const std::vector<mrpt::math::TPoint3Df>& worldPts,
      const std::vector<mrpt::math::TPoint2Df>& pixels, const std::vector<size_t>& corr_idx,
      const mrpt::img::TCamera& cam, std::vector<bool>& pnp_outlier);

  /** Fills \p out with, per tracked feature, where the constant-velocity
   *  prediction says it should appear in the current frame (its previous pixel
   *  for features with no 3D position yet). Used as the LK initial guess. */
  void predictTrackedPixels(std::vector<mrpt::math::TPoint2Df>& out) const;

  /** Drops the local map and rebuilds it from the current stereo pair at the
   *  current (dead-reckoned) pose, after tracking has been lost. */
  void restartMapHere(
      const mrpt::img::CImage& grayL, const mrpt::img::CImage& grayR, const mrpt::img::TCamera& cam,
      double baseline);
  void trackAndLocalize(const mrpt::img::CImage& gray);
  /** Stereo-match a set of left features and append valid ones as new metric
   *  landmarks (world frame, via the current pose). Returns count added. */
  int addStereoLandmarks(
      const mrpt::img::CImage& left, const mrpt::img::CImage& right, const mrpt::img::TCamera& cam,
      double baseline, const std::vector<mrpt::math::TPoint2Df>& left_feats);
  void spawnTriangulatedLandmarks();
  void insertCurrentKeyframe();
  /** Like insertCurrentKeyframe() but re-measures each tracked feature's stereo
   *  disparity (one matchStereo call) and stores it, so windowed BA can use the
   *  metric disparity residual. */
  void insertCurrentKeyframeStereo(
      const mrpt::img::CImage& left, const mrpt::img::CImage& right, const mrpt::img::TCamera& cam,
      double baseline);
  /** Windowed bundle adjustment over the recent keyframes. \p num_fixed_poses
   *  oldest poses are held fixed: 1 for monocular (gauge only; scale is the
   *  bootstrap's), 2 for stereo so the metric separation between two fixed
   *  camera centers also anchors the scale (a pure-reprojection BA otherwise has
   *  a free scale gauge that collapses the metric stereo scale). */
  void runWindowedBA(int num_fixed_poses = 1);
  void publishLocalization(const mrpt::Clock::time_point& timestamp);
  void publishMap(const mrpt::Clock::time_point& timestamp);
  void publishViz2D(const mrpt::img::CImage& gray);
  void publishViz3D();
};

}  // namespace mola
