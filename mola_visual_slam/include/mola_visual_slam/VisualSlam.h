/* -------------------------------------------------------------------------
 * mola_visual_slam: monocular / stereo visual SLAM front-end for MOLA.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_kernel/interfaces/FrontEndBase.h>
#include <mola_kernel/interfaces/LocalizationSourceBase.h>
#include <mola_kernel/interfaces/MapSourceBase.h>
#include <mola_kernel/interfaces/NavStateFilter.h>
#include <mola_libvision/optical_flow.h>
#include <mrpt/img/CImage.h>
#include <mrpt/img/CStereoRectifyMap.h>
#include <mrpt/img/TCamera.h>
#include <mrpt/math/CMatrixFixed.h>
#include <mrpt/math/TPoint2D.h>
#include <mrpt/math/TPoint3D.h>
#include <mrpt/obs/CObservationIMU.h>
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

  /** Whether a state-estimation module was found and is being fed with this
   *  module's pose estimates. */
  [[nodiscard]] bool isFusingIntoStateEstimator() const { return nav_state_filter_ != nullptr; }

  /** How many poses have been handed to the state estimator so far. */
  [[nodiscard]] size_t numFusedPoses() const { return num_fused_poses_; }

  /** How many frames had their rotation predicted from the gyro rather than
   *  from the constant-velocity model. */
  [[nodiscard]] size_t numGyroPredictions() const { return num_gyro_predictions_; }

  /** Mean |y_left - y_right| of the accepted stereo matches, binned by distance
   *  from the image center (bin width \c kStereoResidualBinPx pixels). After a
   *  correct rectification the two rows agree, so any growth with radius is
   *  rectification or camera-model error rather than matching noise. Empty bins
   *  come back as (0, 0). */
  static constexpr int                                 kStereoResidualBinPx = 100;
  [[nodiscard]] std::vector<std::pair<double, size_t>> stereoResidualByRadius() const;

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
  /** Size of the rectified images ("width height"), when \c right_camera_pose
   *  rectification is active. Empty (the default) keeps the input size.
   *
   *  Rectification maps to a PINHOLE model at the source camera's focal length,
   *  so a wide-angle (fisheye) lens loses everything outside the pinhole
   *  frustum that the output canvas spans: with an equidistant lens the raw
   *  half-angle is r/f while the rectified one is only atan(r/f), and the
   *  difference is discarded. Enlarging the canvas keeps the same angular
   *  resolution and recovers that periphery, at the cost of more pixels to
   *  process. Peripheral features are the ones that constrain rotation best,
   *  which is what long-horizon attitude drift is most sensitive to. */
  std::string rectify_output_size_str_;

  /** \name Fusion into a state-estimation module
   *  When a mola::NavStateFilter module exists in the same MOLA system, each
   *  localized frame's VEHICLE pose is handed to it as an independent odometry
   *  source, under its own frame_id, so it can be fused with LiDAR odometry,
   *  IMU and GNSS. Requires the camera-on-robot extrinsic: feeding the camera's
   *  own trajectory instead would make the estimator absorb the lever arm into
   *  a constant frame transform, which is wrong the moment the vehicle rotates.
   *  @{ */
  bool        fuse_into_state_estimator_ = true;
  std::string state_estimator_frame_id_  = "visual_odom";
  /** Sigmas of one fused pose. These describe the LOCAL quality of a single
   *  visual pose measurement (as the ICP covariance does for LiDAR odometry),
   *  not the accumulated drift of the whole trajectory: the estimator solves
   *  for this source's frame transform separately. */
  double fuse_sigma_xyz_        = 0.05;  //!< [m]
  double fuse_sigma_angles_deg_ = 0.5;  //!< [deg]
  /** Per-axis refinements of the two isotropic sigmas above. Each falls back to
   *  its isotropic counterpart when <= 0 (the default), so the fused covariance
   *  is unchanged unless one of these is set.
   *
   *  A forward-facing stereo pair is not equally good in every direction, and an
   *  isotropic covariance throws that structure away: the two tilt axes are read
   *  almost directly off pixel displacements, while yaw trades off against
   *  lateral translation; likewise the two image-plane translation axes are far
   *  better determined than range along the optical axis. Measured on GrandTour
   *  heap-1, per 0.1 s increment: roll 0.019 deg, pitch 0.020 deg, yaw 0.029
   *  deg; lateral 1.0 mm, vertical 0.8 mm, forward 2.4 mm. The axes are the
   *  VEHICLE body's, so which one is "forward" follows the body frame rather
   *  than the camera's mounting. */
  double fuse_sigma_yaw_deg_       = 0;  //!< [deg], about body z
  double fuse_sigma_pitchroll_deg_ = 0;  //!< [deg], about body y and x
  double fuse_sigma_forward_       = 0;  //!< [m], body x
  double fuse_sigma_lateral_       = 0;  //!< [m], body y
  double fuse_sigma_vertical_      = 0;  //!< [m], body z
  /** Feed only one out of every N localized frames. Visual odometry typically
   *  runs far faster than a LiDAR, and one keyframe per scan period is plenty
   *  to constrain the estimator. */
  int fuse_decimation_ = 1;
  /** @} */
  /** Left camera pose on the vehicle ("x y z yaw_deg pitch_deg roll_deg"),
   *  overriding whatever the incoming observations carry in their
   *  \c cameraPose field. Only used to report body-frame poses; it has no
   *  effect on the visual estimation itself. */
  std::string camera_pose_on_robot_str_;

  /** \name Gyroscope-aided inter-frame rotation prediction
   *  A constant-velocity model predicts the next frame's rotation from the
   *  previous one, so it is blind to angular ACCELERATION. On an agile platform
   *  a single sharp turn then throws the LK seed far enough that tracking
   *  collapses, and the frames bridged by dead reckoning inject a large,
   *  permanent attitude error into an otherwise well-behaved trajectory. A
   *  gyroscope measures exactly the missing quantity, so feeding it in as the
   *  rotation part of the prediction (translation stays constant-velocity)
   *  removes that failure mode at negligible cost. Purely a PREDICTION: no
   *  inertial residual enters PnP or bundle adjustment, so a missing, late or
   *  noisy IMU degrades gracefully back to constant velocity.
   *  @{ */
  /** sensorLabel of the CObservationIMU stream to use. Empty (the default)
   *  disables gyro aiding entirely. */
  std::string imu_label_;
  /** IMU pose on the vehicle ("x y z yaw_deg pitch_deg roll_deg"), overriding
   *  whatever the incoming IMU observations carry in their \c sensorPose. */
  std::string imu_pose_on_robot_str_;
  /** Longest frame interval the gyro is trusted to bridge [s]. Beyond it the
   *  prediction falls back to constant velocity. */
  double imu_max_gap_ = 0.5;
  /** @} */
  /** Farthest a stereo match may be back-projected into a new landmark [m].
   *  Stereo depth error grows as Z^2, so beyond some range a match is only a
   *  couple of pixels of disparity and its depth is barely observable, yet PnP
   *  and BA are handed the point as if it were as well determined as a nearby
   *  one. Capping the range trades a smaller map for landmarks whose 3D
   *  position the measurement actually supports. A large value keeps every
   *  match (the previous behavior). */
  float max_landmark_depth_ = 100.0f;
  /** Contrast-limited adaptive histogram equalization applied to every image
   *  before detection and tracking. <= 0 (the default) disables it; 2 to 4 is
   *  the useful range, and higher mostly amplifies noise.
   *
   *  Detection thresholds are relative to each grid cell, but the LK tracker
   *  gates on ABSOLUTE gradient energy, so a scene whose usable texture spans
   *  only a few grey levels loses its tracks however good its corners are. That
   *  is the situation in an unlit space, and a global equalization cannot fix
   *  it when part of the frame is saturated: the bright region then owns the
   *  histogram and the dark one stays just as compressed. */
  float clahe_clip_limit_ = 0.0f;
  int   clahe_tiles_x_    = 8;
  int   clahe_tiles_y_    = 8;
  int   max_features_     = 400;
  float min_distance_     = 12.0f;
  int   redetect_below_   = 150;
  int   lk_win_size_      = 21;
  int   lk_max_levels_    = 3;
  /** Forward-backward consistency gate for LK tracking [px]. Each tracked
   *  feature is re-tracked from the current image back to the previous one, and
   *  dropped when the round trip does not return within this distance. A patch
   *  with no well-defined match (aperture problem, occlusion boundary, repeated
   *  texture) slides in a direction set by the local image structure rather
   *  than by the motion, so it biases the pose systematically instead of merely
   *  adding noise, and the whole-mission attitude error is dominated by the
   *  systematic part. <= 0 disables the check, and with it its second LK pass. */
  float lk_fb_max_error_px_ = 0.0f;
  int   min_pnp_points_     = 12;
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

  // ---- gyroscope-aided rotation prediction ----
  struct GyroSample
  {
    double               t = 0;  ///< timestamp [s]
    mrpt::math::TPoint3D w;  ///< angular velocity in the IMU frame [rad/s]
  };
  std::deque<GyroSample>              gyro_;
  std::optional<mrpt::poses::CPose3D> imu_pose_on_robot_;  ///< T_body_imu
  /** Rotation taking IMU-frame vectors into the estimator's internal frame
   *  (the rectified left camera when rectification is on, the raw left camera
   *  otherwise). Built once both extrinsics are known. */
  std::optional<mrpt::math::CMatrixDouble33> rot_internal_imu_;
  /** Timestamp of the previously processed frame, for the gyro interval. */
  std::optional<double> last_frame_ts_;
  /** Rotation increment measured by the gyro over the current frame interval,
   *  expressed in the internal frame. Empty when unavailable. */
  std::optional<mrpt::math::CMatrixDouble33> gyro_delta_rot_;
  size_t                                     num_gyro_predictions_     = 0;
  bool                                       warned_no_imu_extrinsics_ = false;

  // ---- stereo epipolar residual, accumulated by image radius ----
  std::vector<double> stereo_resid_sum_;
  std::vector<size_t> stereo_resid_count_;

  // ---- fusion into a state-estimation module ----
  std::shared_ptr<mola::NavStateFilter> nav_state_filter_;
  size_t                                num_fused_poses_      = 0;
  int                                   fuse_frame_counter_   = 0;
  bool                                  warned_no_extrinsics_ = false;

  // ---- profiling ----
  mrpt::system::CTimeLogger profiler_{true, "VisualSlam"};

  /** Latches the camera-on-robot extrinsic from an incoming observation, if it
   *  is not already known from the \c camera_pose_on_robot parameter. */
  void rememberCameraPoseOnRobot(const mrpt::poses::CPose3D& p);

  /** Applies the configured contrast enhancement, or returns \p gray as-is. */
  [[nodiscard]] mrpt::img::CImage enhance(const mrpt::img::CImage& gray) const;

  /** Buffers one IMU sample's angular velocity for the rotation prediction. */
  void handleImuObservation(const mrpt::obs::CObservationIMU& o);

  /** Re-tracks \p next_pts from \p curr back to \p prev and flags as LOST every
   *  feature whose round trip does not return to its original pixel within
   *  \c lk_fb_max_error_px_. No-op when the check is disabled. */
  void rejectInconsistentTracks(
      const mrpt::img::CImage& prev, const mrpt::img::CImage& curr,
      const std::vector<mrpt::math::TPoint2Df>& next_pts,
      std::vector<mola::vision::TrackStatus>&   status) const;

  /** Integrates the buffered gyro samples over (t0, t1] and stores the result
   *  in \c gyro_delta_rot_, as a rotation increment in the internal frame.
   *  Clears it when no usable samples cover the interval. */
  void updateGyroDeltaRotation(double t0, double t1);

  /** Hands the current vehicle pose to the state-estimation module, if one was
   *  found. No-op unless the frame was actually localized this time (a
   *  dead-reckoned pose carries no new information and must not be fed back as
   *  a measurement). */
  void fuseIntoStateEstimator(const mrpt::Clock::time_point& timestamp, bool localized);
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
   *  oldest poses are held fixed: 1 for stereo, where the per-observation
   *  disparity residual already anchors the metric scale, so the fixed pose is
   *  only the gauge; 2 for monocular, which has no metric anchor at all, so
   *  fixing one pose would leave the scale gauge free for the window to drift
   *  (or collapse) along. The distance between two fixed camera centers pins
   *  it. */
  void runWindowedBA(int num_fixed_poses = 1);
  void publishLocalization(const mrpt::Clock::time_point& timestamp);
  void publishMap(const mrpt::Clock::time_point& timestamp);
  void publishViz2D(const mrpt::img::CImage& gray);
  void publishViz3D();
};

}  // namespace mola
