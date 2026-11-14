/* -------------------------------------------------------------------------
 * mola_rgbd_slam: RGB-D visual SLAM front-end for the MOLA framework.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_kernel/interfaces/FrontEndBase.h>
#include <mola_kernel/interfaces/LocalizationSourceBase.h>
#include <mola_kernel/interfaces/MapSourceBase.h>
#include <mola_libvision/motion_model.h>
#include <mola_libvision/optical_flow.h>
#include <mrpt/img/CImage.h>
#include <mrpt/img/TCamera.h>
#include <mrpt/math/TPoint2D.h>
#include <mrpt/math/TPoint3D.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/poses/CPose3DInterpolator.h>
#include <mrpt/system/CTimeLogger.h>

#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

namespace mola
{
/** RGB-D visual SLAM front-end.
 *
 *  Consumes `CObservation3DRangeScan` (depth + intensity) and estimates the
 *  live camera trajectory and a sparse 3D landmark map, all in
 *  MRPT 3.x types using the `mola_libvision` building blocks (Shi-Tomasi
 *  detection, pyramidal LK tracking, depth back-projection, robust PnP, and
 *  sliding-window bundle adjustment). No OpenCV, no Ceres.
 *
 *  Pipeline per frame:
 *   1. Extract grayscale image, metric depth map, and pinhole intrinsics.
 *      The intensity camera is the tracked one: depth that is not already
 *      registered to it (different intrinsics or a non-zero
 *      `relativePoseIntensityWRTDepth`) is reprojected into it first.
 *   2. First frame: detect corners, back-project the ones with valid depth to
 *      create the initial landmarks (world frame = first camera frame).
 *   3. Subsequent frames: track features with LK, gather the 3D-2D
 *      correspondences with known landmarks, and recover the camera pose with
 *      robust PnP. Reject outliers.
 *   4. Spawn new landmarks from untracked features that have valid depth.
 *   5. KeyframeSelector decides keyframes; on a keyframe, refine the recent
 *      keyframe poses and observed landmarks with sliding-window BA.
 *   6. Publish the vehicle pose (LocalizationSourceBase) and sparse map
 *      (MapSourceBase), both in the "map" frame, defined as the vehicle pose
 *      at the first frame. The camera-on-vehicle extrinsic comes from the
 *      `camera_pose_on_robot` parameter, or else from the observations
 *      (`sensorPose` composed with `relativePoseIntensityWRTDepth`).
 */
class RgbdSlam : public mola::FrontEndBase,
                 public mola::LocalizationSourceBase,
                 public mola::MapSourceBase
{
  DEFINE_MRPT_OBJECT(RgbdSlam, mola)

 public:
  RgbdSlam()           = default;
  ~RgbdSlam() override = default;

  RgbdSlam(const RgbdSlam&)                = delete;
  RgbdSlam& operator=(const RgbdSlam&)     = delete;
  RgbdSlam(RgbdSlam&&) noexcept            = delete;
  RgbdSlam& operator=(RgbdSlam&&) noexcept = delete;

  // See docs in base class
  void initialize_frontend(const Yaml& cfg) override;
  void spinOnce() override;
  void onNewObservation(const CObservation::ConstPtr& o) override;
  void onQuit() override;

  /** \name Test / introspection helpers (not part of the MOLA interface)
   *  @{ */

  /** Feeds one RGB-D frame directly (used by unit tests and the
   *  onNewObservation path alike). \p depth_m is a HxW metric-depth matrix
   *  (meters; <=0 means invalid). Returns the estimated camera-in-world pose. */
  mrpt::poses::CPose3D processFrame(
      const mrpt::img::CImage& gray, const mrpt::math::CMatrixFloat& depth_m,
      const mrpt::img::TCamera& cam, const mrpt::Clock::time_point& timestamp);

  /** Returns the metric-depth matrix of the current frame. */
  using DepthProvider = std::function<const mrpt::math::CMatrixFloat&()>;

  /** As above, with the depth matrix computed only if this frame needs it
   *  (to create landmarks), which saves its cost in most frames. */
  mrpt::poses::CPose3D processFrame(
      const mrpt::img::CImage& gray, const DepthProvider& depth, const mrpt::img::TCamera& cam,
      const mrpt::Clock::time_point& timestamp);

  /** Current camera-in-world pose estimate (T_wc), in the optical frame
   *  convention, with the world frame being the first camera pose. */
  mrpt::poses::CPose3D currentPose() const { return pose_wc_; }

  /** Current vehicle pose in the "map" frame (the vehicle at the first frame):
   *  T_{B0,Bt} = T_body_cam * T_{C0,Ct} * T_body_cam^-1. It equals the camera
   *  pose while the camera-on-vehicle extrinsic is unknown (no
   *  camera_pose_on_robot parameter and an identity sensorPose). */
  [[nodiscard]] mrpt::poses::CPose3D currentRobotPose() const;

  /** Camera (optical frame) pose on the vehicle, if known. */
  [[nodiscard]] const std::optional<mrpt::poses::CPose3D>& cameraPoseOnRobot() const
  {
    return camera_pose_on_robot_;
  }

  /** Number of landmarks currently in the map (including culled-but-not-erased). */
  size_t numLandmarks() const { return landmarks_.size(); }

  /** Number of non-culled (good) landmarks. */
  size_t numActiveLandmarks() const
  {
    size_t n = 0;
    for (const auto& lm : landmarks_)
    {
      if (!lm.bad)
      {
        ++n;
      }
    }
    return n;
  }

  /** Number of inserted keyframes. */
  size_t numKeyframes() const { return keyframes_.size(); }

  /** Per-stage timing profiler. Dumps a full table on destruction; call
   *  `profiler().dumpAllStats()` to print it on demand. */
  mrpt::system::CTimeLogger&       profiler() { return profiler_; }
  const mrpt::system::CTimeLogger& profiler() const { return profiler_; }

  /** @} */

 private:
  // ---- parameters ----
  std::string sensor_label_;  ///< only process observations with this label (empty = any)
  int         max_features_   = 400;
  float       min_distance_   = 12.0f;
  int         redetect_below_ = 150;  ///< spawn new landmarks when tracked count drops below this
  int         lk_win_size_    = 11;
  int         lk_max_levels_  = 3;
  int         min_pnp_points_ = 12;  ///< minimum 3D-2D matches to trust PnP
  float       min_depth_      = 0.2f;
  float       max_depth_      = 8.0f;
  int         ba_window_size_ = 8;  ///< number of recent keyframes in the BA window
  int         cull_min_obs_   = 2;  ///< cull dropped landmarks observed in fewer keyframes
  // keyframe-selection policy (mola_libvision KeyframeSelectorParams subset):
  float       velocity_window_      = 0.3f;  ///< [s] past poses that estimate the velocity
  float       max_extrapolation_    = 1.5f;  ///< [s] longest constant-velocity prediction
  int         kf_max_frames_gap_    = 20;
  int         kf_min_frames_gap_    = 2;
  int         kf_min_tracked_       = 80;
  float       kf_min_tracked_ratio_ = 0.6f;
  float       kf_min_parallax_px_   = 18.0f;
  bool        publish_viz_2d_       = true;  ///< push a 2D feature-overlay subwindow to MolaViz
  std::string viz2d_title_          = "RGB-D SLAM tracking";
  std::string viz2d_win_pos_;  ///< "x y width height" (optional)
  bool        publish_viz_3d_ = true;  ///< push a 3D scene (points, trajectory, frustum) to MolaViz
  /** "x y z yaw_deg pitch_deg roll_deg" of the camera optical frame on the
   *  vehicle; overrides the extrinsics carried by the observations. */
  std::string camera_pose_on_robot_str_;
  /** If not empty, the vehicle trajectory is saved here (TUM format) on quit. */
  std::string estimated_trajectory_file_;

  // ---- landmark map (world frame) ----
  struct Landmark
  {
    mrpt::math::TPoint3Df pos;  ///< world coordinates
    int                   observations = 0;  ///< total observation count (for coloring/culling)
    bool                  bad          = false;
  };
  std::vector<Landmark> landmarks_;

  // ---- per-keyframe record (for BA + map publish) ----
  struct KeyframeRec
  {
    mrpt::poses::CPose3D               pose_cw;  ///< world -> camera at the keyframe
    std::vector<int>                   lm_index;  ///< landmark index per observation
    std::vector<mrpt::math::TPoint2Df> pixel;  ///< observed pixel per observation
  };
  std::deque<KeyframeRec> keyframes_;

  // ---- vehicle frame ----
  std::optional<mrpt::poses::CPose3D> camera_pose_on_robot_;  ///< T_body_cam (optical)
  mrpt::poses::CPose3DInterpolator    estimated_trajectory_;  ///< vehicle poses, if saved
  std::mutex                          estimated_trajectory_mtx_;

  // ---- live tracking state ----
  mrpt::img::CImage                  prev_gray_;
  std::vector<mrpt::math::TPoint2Df> track_pts_;  ///< current pixel of each tracked feature
  std::vector<int>                   track_lm_;  ///< landmark index of each tracked feature
  mrpt::poses::CPose3D               pose_cw_;  ///< current world -> camera
  mrpt::poses::CPose3D               pose_wc_;  ///< current camera -> world (published)
  std::vector<mrpt::math::TPoint3D>  trajectory_;  ///< camera-center history (world frame)
  mrpt::img::TCamera                 camera_;
  int                                frame_count_     = 0;
  int                                frames_since_kf_ = 0;
  int                                ref_kf_features_ = 0;
  bool                               gui_created_     = false;

  mrpt::system::CTimeLogger profiler_{true, "RgbdSlam"};

  void insertCurrentKeyframe();
  void publishLocalization(const mrpt::Clock::time_point& timestamp);
  void recordPose(const mrpt::Clock::time_point& timestamp);
  void publishMap(const mrpt::Clock::time_point& timestamp);
  void publishViz2D(const mrpt::img::CImage& gray, bool just_initialized);
  void publishViz3D();
  void runWindowedBA();

  /** One tracking + localization attempt of the current frame. */
  struct TrackAttempt
  {
    std::vector<mrpt::math::TPoint2Df>     next_pts;
    std::vector<mola::vision::TrackStatus> status;
    std::vector<bool>                      outlier;  ///< PnP outliers, per tracked feature
    float                                  parallax_sum = 0.f;
    int                                    parallax_n   = 0;
    int                                    inliers      = 0;
    bool                                   localized    = false;
    mrpt::poses::CPose3D                   pose_cw;
  };

  /** Tracks the features into \p gray with LK and localizes the frame with
   *  PnP, both seeded with \p seed_wc. With \p projectSeed, the LK guess of
   *  each feature is its landmark projected from the seed; otherwise, its
   *  previous pixel. */
  TrackAttempt trackFrame(
      const mrpt::img::CImage& gray, const mrpt::poses::CPose3D& seed_wc, bool projectSeed,
      const mrpt::img::TCamera& cam);

  mola::vision::ConstantVelocityModel motion_;  ///< of the camera (pose_wc_)
  double last_frame_time_ = 0;  ///< [s] timestamp of the previous frame
  double frame_period_    = 0;  ///< [s] running average, to detect gaps in the stream
};

}  // namespace mola
