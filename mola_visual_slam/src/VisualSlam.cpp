/* -------------------------------------------------------------------------
 * mola_visual_slam: monocular / stereo visual SLAM front-end for MOLA.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/geometry.h>
#include <mola_libvision/keyframe_selector.h>
#include <mola_libvision/optical_flow.h>
#include <mola_libvision/pnp_solver.h>
#include <mola_libvision/rgbd_depth.h>
#include <mola_libvision/sliding_window_ba.h>
#include <mola_libvision/stereo_matcher.h>
#include <mola_visual_slam/VisualSlam.h>
#include <mrpt/containers/yaml.h>
#include <mrpt/core/Clock.h>
#include <mrpt/img/TColor.h>
#include <mrpt/img/TStereoCamera.h>
#include <mrpt/maps/CSimplePointsMap.h>
#include <mrpt/math/CMatrixFixed.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/poses/CPose3DQuat.h>
#include <mrpt/viz/CFrustum.h>
#include <mrpt/viz/CPointCloud.h>
#include <mrpt/viz/CSetOfLines.h>
#include <mrpt/viz/CSetOfObjects.h>

#include <algorithm>
#include <cmath>
#include <sstream>

using namespace mola;

IMPLEMENTS_MRPT_OBJECT(VisualSlam, FrontEndBase, mola)

namespace
{
/** Build a CPose3D (world -> camera) from a rotation matrix and translation. */
mrpt::poses::CPose3D poseFromRt(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
{
  mrpt::math::CMatrixDouble44 H = mrpt::math::CMatrixDouble44::Identity();
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      H(r, c) = R(r, c);
    }
    H(r, 3) = t(r);
  }
  return mrpt::poses::CPose3D(H);
}

/** Rodrigues exponential map of a rotation vector (axis * angle, radians). */
Eigen::Matrix3d expSO3(const Eigen::Vector3d& w)
{
  const double    theta = w.norm();
  Eigen::Matrix3d K;
  K << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
  if (theta < 1e-9)
  {
    // Second-order series: exact enough well below the angle where the
    // trigonometric form loses precision, and it stays a rotation.
    return Eigen::Matrix3d::Identity() + K + 0.5 * K * K;
  }
  const Eigen::Matrix3d Kn = K / theta;
  return Eigen::Matrix3d::Identity() + std::sin(theta) * Kn + (1 - std::cos(theta)) * Kn * Kn;
}

/** Rotates an image 180 degrees in place: (u,v) -> (W-1-u, H-1-v). */
void rotate180(mrpt::img::CImage& img)
{
  img.flipVertical();
  img.flipHorizontal();
}

/** Median pixel displacement between two equally-sized point sets. */
float medianParallax(
    const std::vector<mrpt::math::TPoint2Df>& a, const std::vector<mrpt::math::TPoint2Df>& b)
{
  std::vector<float> d;
  d.reserve(a.size());
  for (size_t i = 0; i < a.size(); ++i)
  {
    d.push_back(std::hypot(a[i].x - b[i].x, a[i].y - b[i].y));
  }
  if (d.empty())
  {
    return 0.f;
  }
  const auto mid = static_cast<std::ptrdiff_t>(d.size() / 2);
  std::nth_element(d.begin(), d.begin() + mid, d.end());
  return d[d.size() / 2];
}
}  // namespace

void VisualSlam::initialize_frontend(const Yaml& c)
{
  MRPT_START
  if (c.has("params"))
  {
    const auto cfg  = c["params"];
    auto       getI = [&](const char* k, int& v)
    {
      if (cfg.has(k))
      {
        v = cfg[k].as<int>();
      }
    };
    auto getF = [&](const char* k, float& v)
    {
      if (cfg.has(k))
      {
        v = cfg[k].as<float>();
      }
    };
    auto getS = [&](const char* k, std::string& v)
    {
      if (cfg.has(k))
      {
        v = cfg[k].as<std::string>();
      }
    };
    auto getB = [&](const char* k, bool& v)
    {
      if (cfg.has(k))
      {
        v = cfg[k].as<bool>();
      }
    };
    auto getD = [&](const char* k, double& v)
    {
      if (cfg.has(k))
      {
        v = cfg[k].as<double>();
      }
    };
    getS("sensor_label", sensor_label_);
    getS("mode", mode_);
    getS("left_label", left_label_);
    getS("right_label", right_label_);
    getD("stereo_baseline", stereo_baseline_);
    getS("right_camera_pose", right_camera_pose_str_);
    getS("camera_pose_on_robot", camera_pose_on_robot_str_);
    getS("imu_label", imu_label_);
    getS("imu_pose_on_robot", imu_pose_on_robot_str_);
    getD("imu_max_gap", imu_max_gap_);
    getB("left_image_rotate_180", left_image_rotate_180_);
    getB("right_image_rotate_180", right_image_rotate_180_);
    getS("rectify_output_size", rectify_output_size_str_);
    getB("fuse_into_state_estimator", fuse_into_state_estimator_);
    getS("state_estimator_frame_id", state_estimator_frame_id_);
    getD("fuse_sigma_xyz", fuse_sigma_xyz_);
    getD("fuse_sigma_angles_deg", fuse_sigma_angles_deg_);
    getI("fuse_decimation", fuse_decimation_);
    getF("max_landmark_depth", max_landmark_depth_);
    getI("max_features", max_features_);
    getF("min_distance", min_distance_);
    getI("redetect_below", redetect_below_);
    getI("lk_win_size", lk_win_size_);
    getI("lk_max_levels", lk_max_levels_);
    getF("lk_fb_max_error_px", lk_fb_max_error_px_);
    getI("min_pnp_points", min_pnp_points_);
    getF("min_pnp_inlier_ratio", min_pnp_inlier_ratio_);
    getI("lost_max_frames", lost_max_frames_);
    getI("ba_window_size", ba_window_size_);
    getI("cull_min_obs", cull_min_obs_);
    getF("init_min_parallax_px", init_min_parallax_px_);
    getI("init_min_inliers", init_min_inliers_);
    getF("tri_min_parallax_deg", tri_min_parallax_deg_);
    getI("kf_max_frames_gap", kf_max_frames_gap_);
    getI("kf_min_frames_gap", kf_min_frames_gap_);
    getI("kf_min_tracked", kf_min_tracked_);
    getF("kf_min_tracked_ratio", kf_min_tracked_ratio_);
    getF("kf_min_parallax_px", kf_min_parallax_px_);
    getB("publish_viz_2d", publish_viz_2d_);
    getB("publish_viz_3d", publish_viz_3d_);
    getS("viz2d_title", viz2d_title_);
    getS("viz2d_win_pos", viz2d_win_pos_);
  }

  if (mode_ != "mono" && mode_ != "stereo")
  {
    MRPT_LOG_WARN_STREAM("VisualSlam: unknown mode '" << mode_ << "'; using 'mono'.");
    mode_ = "mono";
  }
  if (!right_camera_pose_str_.empty())
  {
    right_camera_pose_ = mrpt::poses::CPose3D::FromString("[" + right_camera_pose_str_ + "]");
    MRPT_LOG_INFO_STREAM(
        "VisualSlam: right_camera_pose set (" << *right_camera_pose_
                                              << "); incoming stereo pairs will be rectified.");
    ASSERTMSG_(
        right_camera_pose_->x() > 0,
        "right_camera_pose is the pose of the RIGHT camera in the LEFT camera's frame, so its x "
        "must be positive for a side-by-side rig (the right camera sits to the right). A negative "
        "x describes the opposite rig and yields negative disparities.");
  }
  if (!camera_pose_on_robot_str_.empty())
  {
    camera_pose_on_robot_ = mrpt::poses::CPose3D::FromString("[" + camera_pose_on_robot_str_ + "]");
    MRPT_LOG_INFO_STREAM(
        "VisualSlam: camera_pose_on_robot set (" << *camera_pose_on_robot_
                                                 << "); poses will be reported in the body frame.");
  }
  if (!imu_pose_on_robot_str_.empty())
  {
    imu_pose_on_robot_ = mrpt::poses::CPose3D::FromString("[" + imu_pose_on_robot_str_ + "]");
  }
  if (!imu_label_.empty())
  {
    MRPT_LOG_INFO_STREAM(
        "VisualSlam: gyro aiding enabled on IMU stream '"
        << imu_label_ << "'; the inter-frame rotation prediction will come from the gyroscope.");
  }
  // Attach to a state-estimation module, if this MOLA system has one, so the
  // visual poses can be fused with the other odometry sources. Optional: this
  // module is perfectly usable standalone (tests, mola-visual-slam-cli), where
  // there is no name server at all.
  if (fuse_into_state_estimator_ && nameServer_)
  {
    auto mods = findService<mola::NavStateFilter>();
    if (!mods.empty())
    {
      nav_state_filter_ = std::dynamic_pointer_cast<mola::NavStateFilter>(mods[0]);
    }
    if (nav_state_filter_)
    {
      MRPT_LOG_INFO_STREAM(
          "VisualSlam: found a state-estimation module; visual poses will be fused as odometry "
          "source '"
          << state_estimator_frame_id_ << "' (1 of every " << std::max(1, fuse_decimation_)
          << " localized frames).");
    }
    else
    {
      MRPT_LOG_INFO(
          "VisualSlam: no mola::NavStateFilter module in this system; running standalone.");
    }
  }

  MRPT_LOG_INFO_STREAM("VisualSlam initialized (mode=" << mode_ << ").");
  MRPT_END
}

void VisualSlam::spinOnce()
{
  // All work happens in onNewObservation().
}

void VisualSlam::onNewObservation(const CObservation::ConstPtr& o)
{
  MRPT_START
  if (!o)
  {
    return;
  }
  if (!imu_label_.empty() && o->sensorLabel == imu_label_)
  {
    if (auto imu = std::dynamic_pointer_cast<const mrpt::obs::CObservationIMU>(o); imu)
    {
      handleImuObservation(*imu);
      return;
    }
  }
  auto obs = std::dynamic_pointer_cast<const mrpt::obs::CObservationImage>(o);
  if (!obs)
  {
    return;
  }

  if (mode_ == "stereo")
  {
    // Pair the left (image_0) and right (image_1) streams by timestamp.
    if (obs->sensorLabel == left_label_)
    {
      rememberCameraPoseOnRobot(obs->cameraPose);
      obs->load();
      pending_left_     = obs->image;
      pending_left_cam_ = obs->cameraParams;
      pending_left_ts_  = obs->timestamp;
      have_left_        = true;
      if (left_image_rotate_180_)
      {
        rotate180(pending_left_);
      }
    }
    else if (obs->sensorLabel == right_label_)
    {
      obs->load();
      pending_right_     = obs->image;
      pending_right_cam_ = obs->cameraParams;
      pending_right_ts_  = obs->timestamp;
      have_right_        = true;
      if (right_image_rotate_180_)
      {
        rotate180(pending_right_);
      }
    }
    else
    {
      return;
    }

    if (have_left_ && have_right_ &&
        std::abs(
            mrpt::Clock::toDouble(pending_left_ts_) - mrpt::Clock::toDouble(pending_right_ts_)) <
            0.005)
    {
      if (right_camera_pose_)
      {
        // Raw rig, not pre-rectified (e.g. a fisheye multi-camera bag with no
        // per-camera TF): build (once) and apply a CStereoRectifyMap from
        // each camera's own intrinsics/distortion plus the fixed extrinsic,
        // so processStereoFrame() keeps receiving what it always expected -
        // a rectified pair sharing one pinhole TCamera and a pure baseline.
        if (!rectify_map_.isSet())
        {
          mrpt::img::TStereoCamera stereoCam;
          stereoCam.leftCamera      = pending_left_cam_;
          stereoCam.rightCamera     = pending_right_cam_;
          stereoCam.rightCameraPose = mrpt::poses::CPose3DQuat(*right_camera_pose_).asTPose();
          if (!rectify_output_size_str_.empty())
          {
            unsigned int       w = 0;
            unsigned int       h = 0;
            std::istringstream ss(rectify_output_size_str_);
            ASSERTMSG_(
                (ss >> w) && (ss >> h) && w > 0 && h > 0,
                "rectify_output_size must be \"width height\", both positive.");
            rectify_map_.enableResizeOutput(true, w, h);
          }
          rectify_map_.setFromCamParams(stereoCam);
          // The rectified frame is rotated with respect to the physical left
          // camera; remember that rotation so the reported trajectory can be
          // expressed in the camera's own (and hence the robot's) frame.
          left_cam_in_rectified_ = mrpt::poses::CPose3D(mrpt::poses::CPose3DQuat(
              0, 0, 0, mrpt::math::CQuaternionDouble(rectify_map_.getLeftCameraRot())));

          const auto&  rc   = rectify_map_.getRectifiedLeftImageParams();
          const double hfov = 2 * std::atan(0.5 * rc.ncols / rc.fx());
          const double vfov = 2 * std::atan(0.5 * rc.nrows / rc.fy());
          MRPT_LOG_INFO_STREAM(
              "VisualSlam: rectified to "
              << rc.ncols << "x" << rc.nrows << " pinhole, fx=" << rc.fx()
              << ", FOV=" << mrpt::RAD2DEG(hfov) << "x" << mrpt::RAD2DEG(vfov) << " deg"
              << " (input " << pending_left_cam_.ncols << "x" << pending_left_cam_.nrows << ")");
        }
        mrpt::img::CImage rectLeft;
        mrpt::img::CImage rectRight;
        rectify_map_.rectify(pending_left_, pending_right_, rectLeft, rectRight);
        // Post-rectification the pair is parallel with a pure +x baseline.
        const double rectifiedBaseline = rectify_map_.getRectifiedImageParams().rightCameraPose.x;
        processStereoFrame(
            rectLeft, rectRight, rectify_map_.getRectifiedLeftImageParams(), rectifiedBaseline,
            pending_left_ts_);
      }
      else
      {
        processStereoFrame(
            pending_left_, pending_right_, pending_left_cam_, stereo_baseline_, pending_left_ts_);
      }
      have_left_  = false;
      have_right_ = false;
    }
    return;
  }

  // Monocular.
  if (!sensor_label_.empty() && obs->sensorLabel != sensor_label_)
  {
    return;
  }
  rememberCameraPoseOnRobot(obs->cameraPose);
  obs->load();
  if (obs->image.isEmpty())
  {
    return;
  }
  processFrame(obs->image, obs->cameraParams, obs->timestamp);
  MRPT_END
}

void VisualSlam::rememberCameraPoseOnRobot(const mrpt::poses::CPose3D& p)
{
  // An explicit parameter always wins; otherwise take the first non-identity
  // pose the dataset source provides (an identity one means "unknown", not
  // "the camera is exactly at the body origin looking along its x axis",
  // which no real optical frame ever is).
  if (camera_pose_on_robot_ || !camera_pose_on_robot_str_.empty())
  {
    return;
  }
  if (p.asVectorVal().norm() < 1e-9)
  {
    return;
  }
  camera_pose_on_robot_ = p;
  MRPT_LOG_INFO_STREAM(
      "VisualSlam: camera-on-robot extrinsic taken from the observations ("
      << p << "); poses will be reported in the body frame.");
}

void VisualSlam::handleImuObservation(const mrpt::obs::CObservationIMU& o)
{
  using mrpt::obs::IMU_WX;
  using mrpt::obs::IMU_WY;
  using mrpt::obs::IMU_WZ;
  if (!o.has(IMU_WX) || !o.has(IMU_WY) || !o.has(IMU_WZ))
  {
    return;  // an accelerometer-only stream carries nothing this can use
  }
  if (!imu_pose_on_robot_ && imu_pose_on_robot_str_.empty())
  {
    imu_pose_on_robot_ = o.sensorPose;
  }

  GyroSample s;
  s.t = mrpt::Clock::toDouble(o.timestamp);
  s.w = {o.get(IMU_WX), o.get(IMU_WY), o.get(IMU_WZ)};
  // Out-of-order samples would corrupt the trapezoidal integration below, and a
  // stream that is not monotonic in time is not usable for this anyway.
  if (!gyro_.empty() && s.t <= gyro_.back().t)
  {
    return;
  }
  gyro_.push_back(s);

  // Keep only what a future frame interval could still need.
  const double keepFrom = s.t - 2.0 * imu_max_gap_;
  while (gyro_.size() > 2 && gyro_.front().t < keepFrom)
  {
    gyro_.pop_front();
  }
}

void VisualSlam::rejectInconsistentTracks(
    const mrpt::img::CImage& prev, const mrpt::img::CImage& curr,
    const std::vector<mrpt::math::TPoint2Df>& next_pts,
    std::vector<mola::vision::TrackStatus>&   status) const
{
  if (lk_fb_max_error_px_ <= 0.f || next_pts.empty())
  {
    return;
  }
  // The backward pass deliberately starts from "the point did not move":
  // seeding it with the forward answer's origin would ask whether a fixed point
  // is a fixed point, which it always is.
  mola::vision::LKParams lk;
  lk.win_size          = lk_win_size_;
  lk.max_levels        = lk_max_levels_;
  lk.use_initial_guess = false;

  std::vector<mrpt::math::TPoint2Df>     back_pts;
  std::vector<mola::vision::TrackStatus> back_status;
  mola::vision::calcOpticalFlowPyrLK(curr, prev, next_pts, back_pts, back_status, lk);

  const float max_err2 = lk_fb_max_error_px_ * lk_fb_max_error_px_;
  for (size_t i = 0; i < next_pts.size(); ++i)
  {
    if (status[i] == mola::vision::TrackStatus::LOST)
    {
      continue;
    }
    if (back_status[i] == mola::vision::TrackStatus::LOST)
    {
      status[i] = mola::vision::TrackStatus::LOST;
      continue;
    }
    const float dx = back_pts[i].x - track_pts_[i].x;
    const float dy = back_pts[i].y - track_pts_[i].y;
    if (dx * dx + dy * dy > max_err2)
    {
      status[i] = mola::vision::TrackStatus::LOST;
    }
  }
}

void VisualSlam::updateGyroDeltaRotation(double t0, double t1)
{
  gyro_delta_rot_.reset();
  if (imu_label_.empty() || gyro_.size() < 2 || t1 <= t0 || (t1 - t0) > imu_max_gap_)
  {
    return;
  }

  // The rotation is measured in the IMU frame, so it only becomes usable once
  // the IMU-to-camera rotation is known. Both extrinsics are reported relative
  // to the vehicle body, which is what chains them together.
  if (!rot_internal_imu_)
  {
    if (!imu_pose_on_robot_ || !camera_pose_on_robot_)
    {
      if (!warned_no_imu_extrinsics_)
      {
        warned_no_imu_extrinsics_ = true;
        MRPT_LOG_WARN(
            "VisualSlam: gyro aiding is configured but the camera-on-robot or IMU-on-robot "
            "extrinsic is unknown, so the gyro cannot be rotated into the camera frame. "
            "Falling back to the constant-velocity rotation prediction.");
      }
      return;
    }
    // T_internal_imu = T_rect_cam * T_cam_body * T_body_imu. Without
    // rectification the internal frame IS the left camera, so the first factor
    // drops out.
    const mrpt::poses::CPose3D rectFromCam =
        left_cam_in_rectified_ ? *left_cam_in_rectified_ : mrpt::poses::CPose3D::Identity();
    const mrpt::poses::CPose3D T = rectFromCam + (-*camera_pose_on_robot_) + *imu_pose_on_robot_;
    rot_internal_imu_            = T.getRotationMatrix();
  }

  // Trapezoidal integration of the angular velocity over (t0, t1], composed on
  // the manifold. At any sane IMU rate the per-step angle is small enough that
  // a first-order exponential map is exact to well below the gyro's own noise.
  mrpt::math::CMatrixDouble33 dR      = mrpt::math::CMatrixDouble33::Identity();
  bool                        any     = false;
  double                      covered = 0;
  for (size_t i = 0; i + 1 < gyro_.size(); ++i)
  {
    const double ta = gyro_[i].t;
    const double tb = gyro_[i + 1].t;
    if (tb <= t0 || ta >= t1)
    {
      continue;
    }
    const double a  = std::max(ta, t0);
    const double b  = std::min(tb, t1);
    const double dt = b - a;
    if (dt <= 0)
    {
      continue;
    }
    // Linear interpolation of the two bracketing samples at the sub-interval
    // midpoint: the frame boundaries almost never fall on a sample.
    const double          span = tb - ta;
    const double          u    = (span > 1e-9) ? ((0.5 * (a + b) - ta) / span) : 0.0;
    const auto&           wa   = gyro_[i].w;
    const auto&           wb   = gyro_[i + 1].w;
    const Eigen::Vector3d w(
        wa.x + u * (wb.x - wa.x), wa.y + u * (wb.y - wa.y), wa.z + u * (wb.z - wa.z));
    dR = (dR.asEigen() * expSO3(w * dt)).eval();
    covered += dt;
    any = true;
  }
  // A partially covered interval would silently under-rotate the prediction,
  // which is worse than not using the gyro at all.
  if (!any || covered < 0.9 * (t1 - t0))
  {
    return;
  }

  gyro_delta_rot_ = mrpt::math::CMatrixDouble33(
      (rot_internal_imu_->asEigen() * dR.asEigen() * rot_internal_imu_->asEigen().transpose())
          .eval());
  ++num_gyro_predictions_;
}

void VisualSlam::fuseIntoStateEstimator(const mrpt::Clock::time_point& timestamp, bool localized)
{
  if (!nav_state_filter_ || state_ != State::TRACKING || !localized)
  {
    return;
  }
  if (!camera_pose_on_robot_)
  {
    // Fusing the CAMERA trajectory into an estimator whose other sources are
    // body-frame would let it absorb the camera lever arm into the constant
    // {map}->{visual_odom} transform, which is only valid while the vehicle
    // does not rotate. Refuse rather than corrupt the estimate silently.
    if (!warned_no_extrinsics_)
    {
      warned_no_extrinsics_ = true;
      MRPT_LOG_ERROR(
          "VisualSlam: cannot fuse into the state estimator without the camera-on-robot "
          "extrinsic. Provide it via the dataset's /tf, a fixed_sensor_pose, or the "
          "'camera_pose_on_robot' parameter.");
    }
    return;
  }
  const int dec = std::max(1, fuse_decimation_);
  if ((fuse_frame_counter_++ % dec) != 0)
  {
    return;
  }

  mrpt::poses::CPose3DPDFGaussian p;
  p.mean = currentRobotPose();
  p.cov.setZero();
  const double s2xyz = fuse_sigma_xyz_ * fuse_sigma_xyz_;
  const double s2ang = mrpt::square(mrpt::DEG2RAD(fuse_sigma_angles_deg_));
  for (int i = 0; i < 3; ++i)
  {
    p.cov(i, i)         = s2xyz;
    p.cov(i + 3, i + 3) = s2ang;
  }
  nav_state_filter_->fuse_pose(timestamp, p, state_estimator_frame_id_);
  ++num_fused_poses_;
}

mrpt::poses::CPose3D VisualSlam::currentPose() const
{
  if (!left_cam_in_rectified_)
  {
    return pose_wc_;
  }
  // pose_wc_ is T_{R0,Rt} (rectified frame). Conjugating by the constant
  // rectified<-camera rotation turns it into T_{L0,Lt}.
  return (-*left_cam_in_rectified_) + pose_wc_ + *left_cam_in_rectified_;
}

mrpt::poses::CPose3D VisualSlam::currentRobotPose() const
{
  const auto camTraj = currentPose();
  if (!camera_pose_on_robot_)
  {
    return camTraj;
  }
  // T_{B0,Bt} = T_body_cam * T_{L0,Lt} * T_body_cam^-1.
  return *camera_pose_on_robot_ + camTraj + (-*camera_pose_on_robot_);
}

size_t VisualSlam::numActiveLandmarks() const
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

void VisualSlam::detectInitialFeatures(const mrpt::img::CImage& gray)
{
  mola::vision::GridDistributorParams gp;
  gp.max_corners  = max_features_;
  gp.min_distance = min_distance_;
  mola::vision::GridFeatureDistributor dist(gp);
  init_ref_pts_ = dist.detect(gray, {});
  track_pts_    = init_ref_pts_;
}

mrpt::poses::CPose3D VisualSlam::processFrame(
    const mrpt::img::CImage& gray_in, const mrpt::img::TCamera& cam,
    const mrpt::Clock::time_point& timestamp)
{
  mrpt::system::CTimeLoggerEntry tle(profiler_, "processFrame");

  camera_ = cam;
  profiler_.enter("grayscale");
  const mrpt::img::CImage gray = gray_in.grayscale();
  profiler_.leave("grayscale");
  ++frame_count_;

  const double frame_ts = mrpt::Clock::toDouble(timestamp);
  if (last_frame_ts_)
  {
    updateGyroDeltaRotation(*last_frame_ts_, frame_ts);
  }
  else
  {
    gyro_delta_rot_.reset();
  }
  last_frame_ts_ = frame_ts;

  if (prev_gray_.isEmpty())
  {
    // Very first frame: seed reference features for the bootstrap.
    detectInitialFeatures(gray);
    prev_gray_ = gray;
    publishViz2D(gray);
    return currentPose();
  }

  if (state_ == State::INITIALIZING)
  {
    // Track reference features into this frame; keep matched pairs aligned.
    std::vector<mrpt::math::TPoint2Df>     next_pts;
    std::vector<mola::vision::TrackStatus> status;
    mola::vision::LKParams                 lk;
    lk.win_size   = lk_win_size_;
    lk.max_levels = lk_max_levels_;
    profiler_.enter("init.trackLK");
    mola::vision::calcOpticalFlowPyrLK(prev_gray_, gray, track_pts_, next_pts, status, lk);
    profiler_.leave("init.trackLK");

    std::vector<mrpt::math::TPoint2Df> kept_ref;
    std::vector<mrpt::math::TPoint2Df> kept_cur;
    for (size_t i = 0; i < next_pts.size(); ++i)
    {
      if (status[i] == mola::vision::TrackStatus::LOST)
      {
        continue;
      }
      kept_ref.push_back(init_ref_pts_[i]);
      kept_cur.push_back(next_pts[i]);
    }
    init_ref_pts_ = kept_ref;
    track_pts_    = kept_cur;
    prev_gray_    = gray;

    // Attempt a two-view bootstrap once parallax is sufficient.
    if (static_cast<int>(track_pts_.size()) >= init_min_inliers_ &&
        medianParallax(init_ref_pts_, track_pts_) >= init_min_parallax_px_)
    {
      if (tryInitialize(gray))
      {
        publishLocalization(timestamp);
        publishMap(timestamp);
        publishViz2D(gray);
        publishViz3D();
      }
    }
    return currentPose();
  }

  // TRACKING.
  {
    mrpt::system::CTimeLoggerEntry t2(profiler_, "track.localize");
    trackAndLocalize(gray);
  }

  mola::vision::KeyframeSelectorParams sp;
  sp.max_frames_gap    = kf_max_frames_gap_;
  sp.min_frames_gap    = kf_min_frames_gap_;
  sp.min_tracked       = kf_min_tracked_;
  sp.min_tracked_ratio = kf_min_tracked_ratio_;
  sp.min_parallax_px   = kf_min_parallax_px_;
  mola::vision::KeyframeSelector selector(sp);

  mola::vision::KeyframeFrameStats stats;
  stats.num_tracked        = static_cast<int>(track_pts_.size());
  stats.ref_num_features   = ref_kf_features_;
  stats.median_parallax_px = 0.f;  // tracking-strength + frame-gap drive KF here
  stats.frames_since_kf    = frames_since_kf_;
  ++frames_since_kf_;

  if (selector.shouldBeKeyframe(stats))
  {
    {
      mrpt::system::CTimeLoggerEntry t3(profiler_, "keyframe.spawnLandmarks");
      spawnTriangulatedLandmarks();
    }
    {
      mrpt::system::CTimeLoggerEntry t4(profiler_, "keyframe.insert");
      insertCurrentKeyframe();
    }
    {
      mrpt::system::CTimeLoggerEntry t5(profiler_, "keyframe.windowedBA");
      // Two fixed poses: monocular BA has no metric anchor, so fixing only one
      // pose leaves the SCALE gauge free and the window can drift (or collapse)
      // along it. The distance between two fixed camera centers pins it.
      runWindowedBA(2);
    }
    frames_since_kf_ = 0;
    publishMap(timestamp);
  }

  prev_gray_ = gray;
  trajectory_.push_back(pose_wc_.translation());
  publishLocalization(timestamp);
  // Only a frame that was actually localized carries new information; a
  // dead-reckoned one would feed the estimator its own prediction back.
  fuseIntoStateEstimator(timestamp, frames_without_pose_ == 0);
  {
    mrpt::system::CTimeLoggerEntry t6(profiler_, "viz");
    publishViz2D(gray);
    publishViz3D();
  }
  return currentPose();
}

bool VisualSlam::tryInitialize(const mrpt::img::CImage& gray)
{
  using mrpt::math::TPoint2Df;
  using mrpt::math::TPoint3Df;

  // Undistort + de-project to normalized coordinates.
  std::vector<TPoint2Df> n1;
  std::vector<TPoint2Df> n2;
  mola::vision::undistortPoints(init_ref_pts_, camera_, n1);
  mola::vision::undistortPoints(track_pts_, camera_, n2);

  profiler_.enter("init.essentialRANSAC");
  const auto er = mola::vision::estimateEssentialRANSAC(n1, n2);
  profiler_.leave("init.essentialRANSAC");
  if (!er.success || er.num_inliers < init_min_inliers_)
  {
    return false;
  }

  // Keep only the inlier correspondences.
  std::vector<TPoint2Df> in1;
  std::vector<TPoint2Df> in2;
  std::vector<TPoint2Df> inPixRef;
  std::vector<TPoint2Df> inPixCur;
  for (size_t i = 0; i < er.inliers.size(); ++i)
  {
    if (er.inliers[i])
    {
      in1.push_back(n1[i]);
      in2.push_back(n2[i]);
      inPixRef.push_back(init_ref_pts_[i]);
      inPixCur.push_back(track_pts_[i]);
    }
  }

  Eigen::Matrix3f R;
  Eigen::Vector3f t;
  if (!mola::vision::decomposeEssentialMatrix(er.E, in1, in2, R, t))
  {
    return false;
  }

  // World frame = first (reference) camera. Reference pose is identity; the
  // current camera pose (world -> camera) is [R|t] with ||t|| = 1 (sets scale).
  const mrpt::poses::CPose3D pose_ref_cw = mrpt::poses::CPose3D::Identity();
  const mrpt::poses::CPose3D pose_cur_cw = poseFromRt(R, t);

  // Triangulate the inliers in the world frame.
  Eigen::Matrix<float, 3, 4> P1;
  Eigen::Matrix<float, 3, 4> P2;
  P1.leftCols<3>() = Eigen::Matrix3f::Identity();
  P1.col(3)        = Eigen::Vector3f::Zero();
  P2.leftCols<3>() = R;
  P2.col(3)        = t;
  std::vector<TPoint3Df> pts3d;
  std::vector<bool>      valid;
  profiler_.enter("init.triangulate");
  mola::vision::triangulatePoints(in1, in2, P1, P2, pts3d, valid);
  profiler_.leave("init.triangulate");

  // Build the map and the two bootstrap keyframes.
  landmarks_.clear();
  track_pts_.clear();
  track_lm_.clear();
  KeyframeRec kf_ref;
  kf_ref.pose_cw = pose_ref_cw;
  KeyframeRec kf_cur;
  kf_cur.pose_cw = pose_cur_cw;

  for (size_t i = 0; i < pts3d.size(); ++i)
  {
    if (!valid[i])
    {
      continue;
    }
    Landmark lm;
    lm.pos           = pts3d[i];
    lm.observations  = 2;
    const int lm_idx = static_cast<int>(landmarks_.size());
    landmarks_.push_back(lm);

    kf_ref.lm_index.push_back(lm_idx);
    kf_ref.pixel.push_back(inPixRef[i]);
    kf_cur.lm_index.push_back(lm_idx);
    kf_cur.pixel.push_back(inPixCur[i]);

    track_pts_.push_back(inPixCur[i]);
    track_lm_.push_back(lm_idx);
  }

  if (landmarks_.size() < static_cast<size_t>(init_min_inliers_) / 2)
  {
    landmarks_.clear();
    return false;
  }

  keyframes_.clear();
  keyframes_.push_back(std::move(kf_ref));
  keyframes_.push_back(std::move(kf_cur));

  pose_cw_ = pose_cur_cw;
  pose_wc_ = -pose_cw_;
  trajectory_.push_back(mrpt::poses::CPose3D::Identity().translation());
  trajectory_.push_back(pose_wc_.translation());

  track_lastkf_pix_ = track_pts_;
  track_has_lastkf_.assign(track_pts_.size(), true);
  ref_kf_features_ = static_cast<int>(track_pts_.size());
  frames_since_kf_ = 0;
  prev_gray_       = gray;
  state_           = State::TRACKING;

  MRPT_LOG_INFO_STREAM(
      "VisualSlam bootstrapped: " << landmarks_.size() << " landmarks from " << er.num_inliers
                                  << " essential inliers.");
  return true;
}

mrpt::poses::CPose3D VisualSlam::predictedPoseWc() const
{
  // Constant-velocity prediction: the previous frame-to-frame motion is a far
  // better starting point than "the camera did not move", both for LK and for
  // PnP, and it is the only sane fallback when localization fails outright.
  if (!gyro_delta_rot_)
  {
    return have_motion_ ? (pose_wc_ + last_motion_) : pose_wc_;
  }
  // With a gyro, the rotation is measured rather than extrapolated. That is the
  // half of the increment a constant-velocity model gets badly wrong under
  // angular acceleration, and the half the LK seed is most sensitive to (a
  // rotation moves every pixel in the image, a small translation barely moves
  // the distant ones).
  const auto translation =
      have_motion_ ? last_motion_.translation() : mrpt::math::TPoint3D(0, 0, 0);
  mrpt::math::CVectorFixedDouble<3> t;
  t[0] = translation.x;
  t[1] = translation.y;
  t[2] = translation.z;
  const mrpt::poses::CPose3D inc(*gyro_delta_rot_, t);
  return pose_wc_ + inc;
}

bool VisualSlam::solveFramePose(
    const std::vector<mrpt::math::TPoint3Df>& worldPts,
    const std::vector<mrpt::math::TPoint2Df>& pixels, const std::vector<size_t>& corr_idx,
    const mrpt::img::TCamera& cam, std::vector<bool>& pnp_outlier)
{
  const mrpt::poses::CPose3D prev_wc = pose_wc_;
  const mrpt::poses::CPose3D pred_wc = predictedPoseWc();

  bool pose_updated = false;
  if (static_cast<int>(worldPts.size()) >= min_pnp_points_)
  {
    mrpt::system::CTimeLoggerEntry tle(profiler_, "track.PnP");
    const auto                     res = mola::vision::solvePnP(worldPts, pixels, cam, -pred_wc);

    // Trust the estimate on inlier SUPPORT, not on the solver's convergence
    // flag: hitting the iteration cap still leaves the best pose found, and
    // dropping it silently freezes the trajectory for that frame. Conversely a
    // converged solve that explains only a small fraction of the
    // correspondences is a wrong local minimum, and accepting it corrupts the
    // map beyond recovery on the very next frame.
    const int minInliers = std::max(
        min_pnp_points_,
        static_cast<int>(std::ceil(min_pnp_inlier_ratio_ * static_cast<float>(worldPts.size()))));
    const bool sane = res.pose.asVectorVal().array().isFinite().all();
    if (res.num_inliers >= minInliers && sane)
    {
      pose_cw_     = res.pose;
      pose_wc_     = -pose_cw_;
      pose_updated = true;
      for (size_t k = 0; k < res.inliers.size(); ++k)
      {
        if (!res.inliers[k])
        {
          pnp_outlier[corr_idx[k]] = true;
        }
      }
    }
    else
    {
      MRPT_LOG_THROTTLE_WARN_STREAM(
          2.0, "VisualSlam: PnP kept only " << res.num_inliers << " of " << worldPts.size()
                                            << " correspondences (need " << minInliers
                                            << "); dead-reckoning this frame.");
    }
    MRPT_LOG_DEBUG_STREAM(
        "PnP frame=" << frame_count_ << " tracked=" << track_pts_.size()
                     << " corr=" << worldPts.size() << " inliers=" << res.num_inliers
                     << " converged=" << res.converged << " iters=" << res.iterations
                     << " cost=" << res.final_cost);
  }
  else
  {
    MRPT_LOG_THROTTLE_WARN_STREAM(
        2.0, "VisualSlam: only " << worldPts.size()
                                 << " mapped features tracked into this frame (need "
                                 << min_pnp_points_ << "); dead-reckoning.");
  }

  frames_without_pose_ = pose_updated ? 0 : frames_without_pose_ + 1;

  if (!pose_updated)
  {
    // A short dropout is worth bridging with the motion model rather than
    // standing still. A sustained one is NOT: the map is then no longer
    // consistent with the images, so extrapolation runs the pose away from a
    // map that stays put, new landmarks get triangulated from a wildly wrong
    // baseline, and the state diverges (to infinity, and then to NaN). Past
    // the same threshold that triggers a stereo map restart, hold the pose and
    // drop the velocity instead.
    if (frames_without_pose_ <= lost_max_frames_)
    {
      pose_wc_ = pred_wc;
      pose_cw_ = -pose_wc_;
    }
    else
    {
      have_motion_ = false;
      last_motion_ = mrpt::poses::CPose3D::Identity();
      return false;
    }
  }

  last_motion_ = pose_wc_ - prev_wc;
  have_motion_ = true;
  return pose_updated;
}

void VisualSlam::predictTrackedPixels(std::vector<mrpt::math::TPoint2Df>& out) const
{
  out = track_pts_;
  if (!have_motion_)
  {
    return;
  }
  const auto   pred_cw = -predictedPoseWc();
  const double fx      = camera_.fx();
  const double fy      = camera_.fy();
  const double cx      = camera_.cx();
  const double cy      = camera_.cy();
  const double maxX    = static_cast<double>(camera_.ncols) - 1.0;
  const double maxY    = static_cast<double>(camera_.nrows) - 1.0;

  for (size_t i = 0; i < track_pts_.size(); ++i)
  {
    const int lm = track_lm_[i];
    if (lm < 0 || landmarks_[lm].bad)
    {
      continue;  // no 3D position to project: keep "did not move"
    }
    const auto Xc = pred_cw.composePoint(mrpt::math::TPoint3D(landmarks_[lm].pos));
    if (Xc.z < 0.1)
    {
      continue;
    }
    const double u = fx * Xc.x / Xc.z + cx;
    const double v = fy * Xc.y / Xc.z + cy;
    if (u < 0 || v < 0 || u > maxX || v > maxY || !std::isfinite(u) || !std::isfinite(v))
    {
      continue;
    }
    out[i] = {static_cast<float>(u), static_cast<float>(v)};
  }
}

void VisualSlam::restartMapHere(
    const mrpt::img::CImage& grayL, const mrpt::img::CImage& grayR, const mrpt::img::TCamera& cam,
    double baseline)
{
  MRPT_LOG_WARN_STREAM(
      "VisualSlam: tracking lost for " << frames_without_pose_
                                       << " frames; restarting the local map at the current "
                                          "dead-reckoned pose (frame "
                                       << frame_count_ << ").");
  landmarks_.clear();
  keyframes_.clear();
  track_pts_.clear();
  track_lm_.clear();
  track_lastkf_pix_.clear();
  track_has_lastkf_.clear();
  // The gap invalidates the velocity estimate: extrapolating across it is
  // exactly what turns a short dropout into a diverging trajectory.
  have_motion_ = false;
  last_motion_ = mrpt::poses::CPose3D::Identity();

  mola::vision::GridDistributorParams gp;
  gp.max_corners   = max_features_;
  gp.min_distance  = min_distance_;
  const auto feats = mola::vision::GridFeatureDistributor(gp).detect(grayL, {});
  {
    mrpt::system::CTimeLoggerEntry tle(profiler_, "stereo.match");
    addStereoLandmarks(grayL, grayR, cam, baseline, feats);
  }
  if (landmarks_.size() < 20)
  {
    return;  // too little texture here; try again next frame
  }
  insertCurrentKeyframeStereo(grayL, grayR, cam, baseline);
  frames_since_kf_     = 0;
  frames_without_pose_ = 0;
}

void VisualSlam::trackAndLocalize(const mrpt::img::CImage& gray)
{
  using mrpt::math::TPoint2Df;
  using mrpt::math::TPoint3Df;

  std::vector<TPoint2Df>                 next_pts;
  std::vector<mola::vision::TrackStatus> status;
  mola::vision::LKParams                 lk;
  lk.win_size   = lk_win_size_;
  lk.max_levels = lk_max_levels_;
  profiler_.enter("track.LK");
  mola::vision::calcOpticalFlowPyrLK(prev_gray_, gray, track_pts_, next_pts, status, lk);
  profiler_.leave("track.LK");
  {
    mrpt::system::CTimeLoggerEntry tfb(profiler_, "track.LKforwardBackward");
    rejectInconsistentTracks(prev_gray_, gray, next_pts, status);
  }

  // 3D-2D correspondences for PnP.
  std::vector<TPoint3Df> worldPts;
  std::vector<TPoint2Df> pixels;
  std::vector<size_t>    corr_idx;
  for (size_t i = 0; i < next_pts.size(); ++i)
  {
    if (status[i] == mola::vision::TrackStatus::LOST)
    {
      continue;
    }
    const int lm = track_lm_[i];
    if (lm < 0 || landmarks_[lm].bad)
    {
      continue;
    }
    worldPts.push_back(landmarks_[lm].pos);
    pixels.push_back(next_pts[i]);
    corr_idx.push_back(i);
  }

  std::vector<bool> pnp_outlier(next_pts.size(), false);
  solveFramePose(worldPts, pixels, corr_idx, camera_, pnp_outlier);

  // Compact: drop lost / rejected, cull spurious untriangulated candidates.
  std::vector<TPoint2Df> kept_pts;
  std::vector<int>       kept_lm;
  std::vector<TPoint2Df> kept_lastkf;
  std::vector<bool>      kept_has;
  for (size_t i = 0; i < next_pts.size(); ++i)
  {
    if (status[i] == mola::vision::TrackStatus::LOST || pnp_outlier[i])
    {
      const int lm = track_lm_[i];
      if (lm >= 0 && landmarks_[lm].observations < cull_min_obs_)
      {
        landmarks_[lm].bad = true;
      }
      continue;
    }
    kept_pts.push_back(next_pts[i]);
    kept_lm.push_back(track_lm_[i]);
    kept_lastkf.push_back(track_has_lastkf_[i] ? track_lastkf_pix_[i] : next_pts[i]);
    kept_has.push_back(track_has_lastkf_[i]);
  }
  track_pts_        = std::move(kept_pts);
  track_lm_         = std::move(kept_lm);
  track_lastkf_pix_ = std::move(kept_lastkf);
  track_has_lastkf_ = std::move(kept_has);

  // Replenish features when tracking gets sparse (new candidates, lm = -1).
  if (static_cast<int>(track_pts_.size()) < redetect_below_)
  {
    mrpt::system::CTimeLoggerEntry      tle(profiler_, "track.redetect");
    mola::vision::GridDistributorParams gp;
    gp.max_corners  = max_features_;
    gp.min_distance = min_distance_;
    mola::vision::GridFeatureDistributor dist(gp);
    const auto                           fresh = dist.detect(gray, track_pts_);
    for (const auto& p : fresh)
    {
      if (static_cast<int>(track_pts_.size()) >= max_features_)
      {
        break;
      }
      track_pts_.push_back(p);
      track_lm_.push_back(-1);
      track_lastkf_pix_.push_back(p);
      track_has_lastkf_.push_back(false);
    }
  }
}

int VisualSlam::addStereoLandmarks(
    const mrpt::img::CImage& left, const mrpt::img::CImage& right, const mrpt::img::TCamera& cam,
    double baseline, const std::vector<mrpt::math::TPoint2Df>& left_feats)
{
  if (left_feats.empty())
  {
    return 0;
  }
  const auto sm = mola::vision::matchStereo(left, right, left_feats, cam.fx(), baseline);

  mola::vision::RGBDParams dp;
  dp.min_depth = 0.3f;
  dp.max_depth = max_landmark_depth_;

  int added = 0;
  for (size_t i = 0; i < left_feats.size(); ++i)
  {
    if (!sm.valid[i])
    {
      continue;
    }
    const auto Xc = mola::vision::backprojectPixel(left_feats[i], sm.depth[i], cam, dp);
    if (!Xc)
    {
      continue;
    }
    const auto Xw = pose_wc_.composePoint(mrpt::math::TPoint3D(*Xc));
    if (!std::isfinite(Xw.x) || !std::isfinite(Xw.y) || !std::isfinite(Xw.z))
    {
      continue;
    }
    Landmark lm;
    lm.pos          = mrpt::math::TPoint3Df(Xw);
    lm.observations = 1;
    track_pts_.push_back(left_feats[i]);
    track_lm_.push_back(static_cast<int>(landmarks_.size()));
    track_lastkf_pix_.push_back(left_feats[i]);
    track_has_lastkf_.push_back(false);
    landmarks_.push_back(lm);
    ++added;
  }
  return added;
}

mrpt::poses::CPose3D VisualSlam::processStereoFrame(
    const mrpt::img::CImage& left, const mrpt::img::CImage& right, const mrpt::img::TCamera& cam,
    double baseline, const mrpt::Clock::time_point& timestamp)
{
  using mrpt::math::TPoint2Df;
  using mrpt::math::TPoint3Df;

  mrpt::system::CTimeLoggerEntry tle(profiler_, "processStereoFrame");

  camera_                       = cam;
  cur_baseline_                 = baseline;
  const mrpt::img::CImage grayL = left.grayscale();
  const mrpt::img::CImage grayR = right.grayscale();
  ++frame_count_;

  // Measured inter-frame rotation, if a gyro stream is configured and covers
  // this interval. Everything downstream that predicts the pose picks it up
  // through predictedPoseWc(): the LK seed, the PnP seed, and the dead-reckoned
  // fallback when PnP fails.
  const double frame_ts = mrpt::Clock::toDouble(timestamp);
  if (last_frame_ts_)
  {
    updateGyroDeltaRotation(*last_frame_ts_, frame_ts);
  }
  else
  {
    gyro_delta_rot_.reset();
  }
  last_frame_ts_ = frame_ts;

  mola::vision::GridDistributorParams gp;
  gp.max_corners  = max_features_;
  gp.min_distance = min_distance_;
  mola::vision::GridFeatureDistributor dist(gp);

  // -------- First frame: initialize a metric map directly from stereo --------
  if (prev_gray_.isEmpty())
  {
    pose_cw_ = mrpt::poses::CPose3D::Identity();
    pose_wc_ = mrpt::poses::CPose3D::Identity();
    track_pts_.clear();
    track_lm_.clear();
    track_lastkf_pix_.clear();
    track_has_lastkf_.clear();

    const auto feats = dist.detect(grayL, {});
    profiler_.enter("stereo.match");
    addStereoLandmarks(grayL, grayR, cam, baseline, feats);
    profiler_.leave("stereo.match");

    if (landmarks_.size() < 20)
    {
      prev_gray_ = grayL;
      return currentPose();  // not enough stereo matches yet; wait
    }
    insertCurrentKeyframeStereo(grayL, grayR, cam, baseline);
    state_           = State::TRACKING;
    prev_gray_       = grayL;
    frames_since_kf_ = 0;
    trajectory_.push_back(pose_wc_.translation());
    publishLocalization(timestamp);
    publishMap(timestamp);
    publishViz2D(grayL);
    publishViz3D();
    return currentPose();
  }

  // -------- Tracking: LK (left t-1 -> left t) + PnP --------
  // Seed the LK search with where the constant-velocity model says each mapped
  // feature should land. Starting from "the point did not move" is what breaks
  // first under the fast turns of a legged robot: the true displacement then
  // exceeds the coarsest pyramid level's basin of attraction.
  std::vector<TPoint2Df>                 next_pts;
  std::vector<mola::vision::TrackStatus> status;
  mola::vision::LKParams                 lk;
  lk.win_size   = lk_win_size_;
  lk.max_levels = lk_max_levels_;
  predictTrackedPixels(next_pts);
  lk.use_initial_guess = true;
  profiler_.enter("track.LK");
  mola::vision::calcOpticalFlowPyrLK(prev_gray_, grayL, track_pts_, next_pts, status, lk);
  profiler_.leave("track.LK");
  {
    mrpt::system::CTimeLoggerEntry tfb(profiler_, "track.LKforwardBackward");
    rejectInconsistentTracks(prev_gray_, grayL, next_pts, status);
  }

  std::vector<TPoint3Df> worldPts;
  std::vector<TPoint2Df> pixels;
  std::vector<size_t>    corr_idx;
  for (size_t i = 0; i < next_pts.size(); ++i)
  {
    if (status[i] == mola::vision::TrackStatus::LOST)
    {
      continue;
    }
    const int lm = track_lm_[i];
    if (lm < 0 || landmarks_[lm].bad)
    {
      continue;
    }
    worldPts.push_back(landmarks_[lm].pos);
    pixels.push_back(next_pts[i]);
    corr_idx.push_back(i);
  }

  std::vector<bool> pnp_outlier(next_pts.size(), false);
  solveFramePose(worldPts, pixels, corr_idx, cam, pnp_outlier);

  if (frames_without_pose_ > lost_max_frames_)
  {
    // The map and the images no longer agree, and every further frame would be
    // built on a pose nothing supports. Rebuild the local map here instead of
    // dead-reckoning indefinitely, which diverges without bound.
    restartMapHere(grayL, grayR, cam, baseline);
    prev_gray_ = grayL;
    trajectory_.push_back(pose_wc_.translation());
    publishLocalization(timestamp);
    publishMap(timestamp);
    return currentPose();
  }

  // Compact tracking arrays (drop lost / rejected; cull weak landmarks).
  std::vector<TPoint2Df> kept_pts;
  std::vector<int>       kept_lm;
  std::vector<TPoint2Df> kept_lastkf;
  std::vector<bool>      kept_has;
  for (size_t i = 0; i < next_pts.size(); ++i)
  {
    if (status[i] == mola::vision::TrackStatus::LOST || pnp_outlier[i])
    {
      const int lm = track_lm_[i];
      if (lm >= 0 && landmarks_[lm].observations < cull_min_obs_)
      {
        landmarks_[lm].bad = true;
      }
      continue;
    }
    kept_pts.push_back(next_pts[i]);
    kept_lm.push_back(track_lm_[i]);
    kept_lastkf.push_back(track_lastkf_pix_[i]);
    kept_has.push_back(track_has_lastkf_[i]);
  }
  track_pts_        = std::move(kept_pts);
  track_lm_         = std::move(kept_lm);
  track_lastkf_pix_ = std::move(kept_lastkf);
  track_has_lastkf_ = std::move(kept_has);

  // Spawn new metric landmarks from fresh left features when tracking is sparse.
  if (static_cast<int>(track_pts_.size()) < redetect_below_)
  {
    const auto fresh = dist.detect(grayL, track_pts_);
    profiler_.enter("stereo.match");
    addStereoLandmarks(grayL, grayR, cam, baseline, fresh);
    profiler_.leave("stereo.match");
  }

  // Keyframe decision.
  ++frames_since_kf_;
  mola::vision::KeyframeSelectorParams sp;
  sp.max_frames_gap    = kf_max_frames_gap_;
  sp.min_frames_gap    = kf_min_frames_gap_;
  sp.min_tracked       = kf_min_tracked_;
  sp.min_tracked_ratio = kf_min_tracked_ratio_;
  sp.min_parallax_px   = kf_min_parallax_px_;
  mola::vision::KeyframeSelector selector(sp);

  mola::vision::KeyframeFrameStats stats;
  stats.num_tracked      = static_cast<int>(track_pts_.size());
  stats.ref_num_features = ref_kf_features_;
  stats.frames_since_kf  = frames_since_kf_;

  if (selector.shouldBeKeyframe(stats))
  {
    insertCurrentKeyframeStereo(grayL, grayR, cam, baseline);
    {
      mrpt::system::CTimeLoggerEntry tlb(profiler_, "keyframe.windowedBA");
      runWindowedBA();  // stereo disparity residual anchors scale (see BA below)
    }
    frames_since_kf_ = 0;
    publishMap(timestamp);
  }

  prev_gray_ = grayL;
  trajectory_.push_back(pose_wc_.translation());
  publishLocalization(timestamp);
  fuseIntoStateEstimator(timestamp, frames_without_pose_ == 0);
  {
    mrpt::system::CTimeLoggerEntry tlv(profiler_, "viz");
    publishViz2D(grayL);
    publishViz3D();
  }
  return currentPose();
}

void VisualSlam::spawnTriangulatedLandmarks()
{
  using mrpt::math::TPoint2Df;
  using mrpt::math::TPoint3Df;
  if (keyframes_.empty())
  {
    return;
  }

  const mrpt::poses::CPose3D& prev_pose_cw = keyframes_.back().pose_cw;

  // Projection matrices [R|t] (world -> camera) for the previous and current KF.
  auto poseToP = [](const mrpt::poses::CPose3D& p) -> Eigen::Matrix<float, 3, 4>
  {
    const auto                 H = p.getHomogeneousMatrixVal<mrpt::math::CMatrixDouble44>();
    Eigen::Matrix<float, 3, 4> P;
    for (int r = 0; r < 3; ++r)
    {
      for (int c = 0; c < 4; ++c)
      {
        P(r, c) = static_cast<float>(H(r, c));
      }
    }
    return P;
  };
  const Eigen::Matrix<float, 3, 4> Pprev = poseToP(prev_pose_cw);
  const Eigen::Matrix<float, 3, 4> Pcur  = poseToP(pose_cw_);

  // Camera centers (world) for the parallax check.
  const auto Cprev = (-prev_pose_cw).translation();
  const auto Ccur  = pose_wc_.translation();

  const float cos_min = std::cos(tri_min_parallax_deg_ * static_cast<float>(M_PI) / 180.f);

  for (size_t i = 0; i < track_pts_.size(); ++i)
  {
    if (track_lm_[i] >= 0 || !track_has_lastkf_[i])
    {
      continue;  // already a landmark, or no prior keyframe view
    }
    std::vector<TPoint2Df> a{track_lastkf_pix_[i]};
    std::vector<TPoint2Df> b{track_pts_[i]};
    std::vector<TPoint2Df> na;
    std::vector<TPoint2Df> nb;
    mola::vision::undistortPoints(a, camera_, na);
    mola::vision::undistortPoints(b, camera_, nb);

    std::vector<TPoint3Df> pts3d;
    std::vector<bool>      valid;
    mola::vision::triangulatePoints(na, nb, Pprev, Pcur, pts3d, valid);
    if (valid.empty() || !valid[0])
    {
      continue;
    }

    // Parallax: angle between the two viewing rays at the 3D point.
    const auto      X = pts3d[0];
    Eigen::Vector3f r1(
        static_cast<float>(X.x - Cprev.x), static_cast<float>(X.y - Cprev.y),
        static_cast<float>(X.z - Cprev.z));
    Eigen::Vector3f r2(
        static_cast<float>(X.x - Ccur.x), static_cast<float>(X.y - Ccur.y),
        static_cast<float>(X.z - Ccur.z));
    const float n1 = r1.norm();
    const float n2 = r2.norm();
    if (n1 < 1e-6f || n2 < 1e-6f)
    {
      continue;
    }
    if (r1.dot(r2) / (n1 * n2) > cos_min)
    {
      continue;  // insufficient parallax
    }

    if (!std::isfinite(X.x) || !std::isfinite(X.y) || !std::isfinite(X.z))
    {
      continue;
    }
    Landmark lm;
    lm.pos          = X;
    lm.observations = 1;
    track_lm_[i]    = static_cast<int>(landmarks_.size());
    landmarks_.push_back(lm);
  }
}

void VisualSlam::insertCurrentKeyframe()
{
  KeyframeRec kf;
  kf.pose_cw = pose_cw_;
  for (size_t i = 0; i < track_pts_.size(); ++i)
  {
    const int lm = track_lm_[i];
    if (lm < 0 || landmarks_[lm].bad)
    {
      continue;
    }
    kf.lm_index.push_back(lm);
    kf.pixel.push_back(track_pts_[i]);
    landmarks_[lm].observations++;
  }
  ref_kf_features_ = static_cast<int>(track_pts_.size());
  keyframes_.push_back(std::move(kf));

  // Snapshot per-feature pixels for next keyframe's triangulation.
  track_lastkf_pix_ = track_pts_;
  track_has_lastkf_.assign(track_pts_.size(), true);
}

void VisualSlam::insertCurrentKeyframeStereo(
    const mrpt::img::CImage& left, const mrpt::img::CImage& right, const mrpt::img::TCamera& cam,
    double baseline)
{
  // Re-measure each tracked feature's disparity at this keyframe (one match).
  const auto sm = mola::vision::matchStereo(left, right, track_pts_, cam.fx(), baseline);

  KeyframeRec kf;
  kf.pose_cw = pose_cw_;
  for (size_t i = 0; i < track_pts_.size(); ++i)
  {
    const int lm = track_lm_[i];
    if (lm < 0 || landmarks_[lm].bad)
    {
      continue;
    }
    kf.lm_index.push_back(lm);
    kf.pixel.push_back(track_pts_[i]);
    // Observed disparity = x_left - x_right (>=0); -1 if no valid stereo match.
    const float disp = sm.valid[i] ? (track_pts_[i].x - sm.right_pts[i].x) : -1.f;
    kf.disparity.push_back(disp);
    landmarks_[lm].observations++;
  }
  ref_kf_features_ = static_cast<int>(track_pts_.size());
  keyframes_.push_back(std::move(kf));

  track_lastkf_pix_ = track_pts_;
  track_has_lastkf_.assign(track_pts_.size(), true);
}

void VisualSlam::runWindowedBA(int num_fixed_poses)
{
  const int W = std::min<int>(ba_window_size_, static_cast<int>(keyframes_.size()));
  if (W < 2)
  {
    return;
  }
  const size_t first = keyframes_.size() - W;

  std::vector<int>                         lm_global;
  std::vector<int>                         lm_local(landmarks_.size(), -1);
  std::vector<mrpt::poses::CPose3D>        poses;
  std::vector<mola::vision::BAObservation> obs;

  for (int w = 0; w < W; ++w)
  {
    const auto& kf = keyframes_[first + w];
    poses.push_back(kf.pose_cw);
    for (size_t j = 0; j < kf.lm_index.size(); ++j)
    {
      const int g = kf.lm_index[j];
      if (landmarks_[g].bad)
      {
        continue;
      }
      if (lm_local[g] < 0)
      {
        lm_local[g] = static_cast<int>(lm_global.size());
        lm_global.push_back(g);
      }
      mola::vision::BAObservation o;
      o.kf_index = w;
      o.lm_index = lm_local[g];
      o.pixel    = kf.pixel[j];
      if (j < kf.disparity.size())
      {
        o.disparity = kf.disparity[j];
      }
      obs.push_back(o);
    }
  }
  if (lm_global.empty() || obs.empty())
  {
    return;
  }

  std::vector<mrpt::math::TPoint3Df> lms(lm_global.size());
  for (size_t k = 0; k < lm_global.size(); ++k)
  {
    lms[k] = landmarks_[lm_global[k]].pos;
  }

  std::vector<bool> fixed(poses.size(), false);
  const int         nfix = std::min<int>(std::max(1, num_fixed_poses), W - 1);
  for (int w = 0; w < nfix; ++w)
  {
    fixed[w] = true;
  }
  mola::vision::BAOptions baopts;
  // Stereo: the disparity residual (per observation) directly constrains depth
  // and anchors metric scale; harmless for mono (no observation carries one).
  baopts.stereo_baseline = cur_baseline_;
  mola::vision::slidingWindowBA(poses, lms, obs, camera_, fixed, baopts);

  for (int w = 0; w < W; ++w)
  {
    keyframes_[first + w].pose_cw = poses[w];
  }
  for (size_t k = 0; k < lm_global.size(); ++k)
  {
    landmarks_[lm_global[k]].pos = lms[k];
  }
  pose_cw_ = keyframes_.back().pose_cw;
  pose_wc_ = -pose_cw_;
}

void VisualSlam::publishLocalization(const mrpt::Clock::time_point& timestamp)
{
  if (!anyUpdateLocalizationSubscriber())
  {
    return;
  }
  LocalizationUpdate lu;
  lu.timestamp       = timestamp;
  lu.method          = "visual_slam";
  lu.reference_frame = "map";
  // Only claim "base_link" when the camera-on-robot extrinsic is actually
  // known; otherwise this is the camera's own trajectory and saying otherwise
  // silently corrupts anything that fuses it.
  lu.child_frame = camera_pose_on_robot_ ? "base_link" : "camera";
  lu.pose        = currentRobotPose().asTPose();
  advertiseUpdatedLocalization(lu);
}

void VisualSlam::publishMap(const mrpt::Clock::time_point& timestamp)
{
  if (!anyUpdateMapSubscriber())
  {
    return;
  }
  auto cloud = mrpt::maps::CSimplePointsMap::Create();
  cloud->reserve(landmarks_.size());
  for (const auto& lm : landmarks_)
  {
    if (!lm.bad)
    {
      cloud->insertPoint(lm.pos.x, lm.pos.y, lm.pos.z);
    }
  }
  MapUpdate mu;
  mu.timestamp       = timestamp;
  mu.method          = "visual_slam";
  mu.reference_frame = "map";
  mu.map_name        = "sparse_landmarks";
  mu.map             = cloud;
  advertiseUpdatedMap(mu);
}

void VisualSlam::publishViz2D(const mrpt::img::CImage& gray)
{
  if (!publish_viz_2d_ || !visualizer_)
  {
    return;
  }
  using mrpt::img::TColor;
  mrpt::img::CImage img = gray.colorImage();

  const TColor colMapped(0, 220, 0);  // feature with a triangulated landmark
  const TColor colCand(40, 160, 255);  // candidate (not yet triangulated)
  for (size_t i = 0; i < track_pts_.size(); ++i)
  {
    const bool mapped = (i < track_lm_.size()) && track_lm_[i] >= 0;
    img.drawCircle(
        {static_cast<int>(track_pts_[i].x), static_cast<int>(track_pts_[i].y)}, 3,
        mapped ? colMapped : colCand, 1);
  }

  std::ostringstream txt;
  txt << (state_ == State::TRACKING ? "TRACKING" : "INITIALIZING") << "  frame " << frame_count_
      << "  tracked: " << track_pts_.size() << "  landmarks: " << numActiveLandmarks()
      << "  kfs: " << keyframes_.size();
  img.textOut({8, 8}, txt.str(), TColor(255, 255, 255));

  auto annotated         = mrpt::obs::CObservationImage::Create();
  annotated->sensorLabel = viz2d_title_;
  annotated->image       = std::move(img);

  if (!gui_created_)
  {
    mola::gui::WindowDescription desc;
    desc.title = viz2d_title_;
    if (!viz2d_win_pos_.empty())
    {
      std::string cleaned = viz2d_win_pos_;
      for (char& ch : cleaned)
      {
        if (ch == '[' || ch == ']' || ch == ',')
        {
          ch = ' ';
        }
      }
      int                x = 0, y = 0, w = 0, h = 0;
      std::istringstream ss(cleaned);
      if ((ss >> x) && (ss >> y) && (ss >> w) && (ss >> h) && w > 0 && h > 0)
      {
        desc.position = {x, y};
        desc.size     = {w, h};
      }
    }
    visualizer_->create_subwindow_from_description(desc).get();
    gui_created_ = true;
  }
  visualizer_->subwindow_update_visualization(annotated, viz2d_title_);
}

void VisualSlam::publishViz3D()
{
  if (!publish_viz_3d_ || !visualizer_)
  {
    return;
  }
  auto scene = mrpt::viz::CSetOfObjects::Create();

  auto cloud = mrpt::viz::CPointCloud::Create();
  cloud->setPointSize(2.0f);
  cloud->setColor_u8(mrpt::img::TColor(200, 200, 200));
  for (const auto& lm : landmarks_)
  {
    if (!lm.bad)
    {
      cloud->insertPoint(lm.pos.x, lm.pos.y, lm.pos.z);
    }
  }
  scene->insert(cloud);

  if (trajectory_.size() >= 2)
  {
    auto traj = mrpt::viz::CSetOfLines::Create();
    traj->setColor_u8(mrpt::img::TColor(40, 200, 40));
    traj->setLineWidth(2.0f);
    for (size_t i = 1; i < trajectory_.size(); ++i)
    {
      const auto& a = trajectory_[i - 1];
      const auto& b = trajectory_[i];
      traj->appendLine(a.x, a.y, a.z, b.x, b.y, b.z);
    }
    scene->insert(traj);
  }

  for (const auto& kf : keyframes_)
  {
    auto fr = mrpt::viz::CFrustum::Create();
    fr->setColor_u8(mrpt::img::TColor(120, 120, 120, 180));
    fr->setPose(-kf.pose_cw);
    scene->insert(fr);
  }
  auto cur = mrpt::viz::CFrustum::Create();
  cur->setColor_u8(mrpt::img::TColor(40, 200, 40));
  cur->setPose(pose_wc_);
  scene->insert(cur);

  visualizer_->update_3d_object("visual_slam", scene);
}
