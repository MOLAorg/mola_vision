/* -------------------------------------------------------------------------
 * mola_rgbd_slam: RGB-D visual SLAM front-end for the MOLA framework.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/keyframe_selector.h>
#include <mola_libvision/motion_model.h>
#include <mola_libvision/optical_flow.h>
#include <mola_libvision/pnp_solver.h>
#include <mola_libvision/rgbd_depth.h>
#include <mola_libvision/sliding_window_ba.h>
#include <mola_rgbd_slam/RgbdSlam.h>
#include <mrpt/containers/yaml.h>
#include <mrpt/core/lock_helper.h>
#include <mrpt/img/TColor.h>
#include <mrpt/maps/CSimplePointsMap.h>
#include <mrpt/obs/CObservation3DRangeScan.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/viz/CFrustum.h>
#include <mrpt/viz/CPointCloud.h>
#include <mrpt/viz/CSetOfLines.h>
#include <mrpt/viz/CSetOfObjects.h>
#include <mrpt/viz/stock_objects.h>

#include <algorithm>
#include <cmath>
#include <sstream>

using namespace mola;

IMPLEMENTS_MRPT_OBJECT(RgbdSlam, FrontEndBase, mola)

namespace
{
/** Looks up the metric depth (meters) at an integer-rounded pixel, returning
 *  <=0 if out of bounds. */
float depthAt(const mrpt::math::CMatrixFloat& depth_m, const mrpt::math::TPoint2Df& px)
{
  const int c = static_cast<int>(std::lround(px.x));
  const int r = static_cast<int>(std::lround(px.y));
  if (r < 0 || c < 0 || r >= static_cast<int>(depth_m.rows()) ||
      c >= static_cast<int>(depth_m.cols()))
  {
    return -1.f;
  }
  return depth_m(r, c);
}

bool sameIntrinsics(const mrpt::img::TCamera& a, const mrpt::img::TCamera& b)
{
  return (a.intrinsicParams.asEigen() - b.intrinsicParams.asEigen()).cwiseAbs().maxCoeff() < 1e-6;
}

/** True if the depth image is already registered to the intensity camera. */
bool depthIsRegistered(const mrpt::obs::CObservation3DRangeScan& obs)
{
  return obs.doDepthAndIntensityCamerasCoincide() &&
         sameIntrinsics(obs.cameraParams, obs.cameraParamsIntensity);
}
}  // namespace

void RgbdSlam::initialize_frontend(const Yaml& c)
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
    if (cfg.has("sensor_label"))
    {
      sensor_label_ = cfg["sensor_label"].as<std::string>();
    }
    getI("max_features", max_features_);
    getF("min_distance", min_distance_);
    getI("redetect_below", redetect_below_);
    getI("lk_win_size", lk_win_size_);
    getI("lk_max_levels", lk_max_levels_);
    getI("min_pnp_points", min_pnp_points_);
    getF("min_depth", min_depth_);
    getF("max_depth", max_depth_);
    getF("velocity_window", velocity_window_);
    getF("max_extrapolation", max_extrapolation_);
    getI("ba_window_size", ba_window_size_);
    getI("cull_min_obs", cull_min_obs_);
    getI("kf_max_frames_gap", kf_max_frames_gap_);
    getI("kf_min_frames_gap", kf_min_frames_gap_);
    getI("kf_min_tracked", kf_min_tracked_);
    getF("kf_min_tracked_ratio", kf_min_tracked_ratio_);
    getF("kf_min_parallax_px", kf_min_parallax_px_);
    if (cfg.has("publish_viz_2d"))
    {
      publish_viz_2d_ = cfg["publish_viz_2d"].as<bool>();
    }
    if (cfg.has("viz2d_title"))
    {
      viz2d_title_ = cfg["viz2d_title"].as<std::string>();
    }
    if (cfg.has("viz2d_win_pos"))
    {
      viz2d_win_pos_ = cfg["viz2d_win_pos"].as<std::string>();
    }
    if (cfg.has("publish_viz_3d"))
    {
      publish_viz_3d_ = cfg["publish_viz_3d"].as<bool>();
    }
    if (cfg.has("estimated_trajectory_file"))
    {
      estimated_trajectory_file_ = cfg["estimated_trajectory_file"].as<std::string>();
    }
    if (cfg.has("camera_pose_on_robot"))
    {
      camera_pose_on_robot_str_ = cfg["camera_pose_on_robot"].as<std::string>();
    }
  }

  mola::vision::ConstantVelocityModel::Parameters mp;
  mp.velocity_window   = velocity_window_;
  mp.max_extrapolation = max_extrapolation_;
  motion_.setParameters(mp);

  if (!camera_pose_on_robot_str_.empty())
  {
    camera_pose_on_robot_ = mrpt::poses::CPose3D::FromString("[" + camera_pose_on_robot_str_ + "]");
  }

  MRPT_LOG_INFO_STREAM(
      "RgbdSlam initialized. sensor_label='" << sensor_label_
                                             << "' max_features=" << max_features_);
  MRPT_END
}

void RgbdSlam::spinOnce()
{
  // All work happens in onNewObservation().
}

void RgbdSlam::onNewObservation(const CObservation::ConstPtr& o)
{
  MRPT_START
  if (!o)
  {
    return;
  }

  auto obs = std::dynamic_pointer_cast<const mrpt::obs::CObservation3DRangeScan>(o);
  if (!obs)
  {
    return;  // not an RGB-D observation
  }
  if (!sensor_label_.empty() && obs->sensorLabel != sensor_label_)
  {
    return;
  }

  obs->load();
  if (!obs->hasRangeImage)
  {
    return;
  }

  if (!obs->hasIntensityImage || obs->intensityImage.isEmpty())
  {
    return;  // no texture to track
  }
  if (!obs->range_is_depth)
  {
    MRPT_LOG_WARN_STREAM("Observation has range_is_depth=false; treating ranges as depths.");
  }

  // The intensity camera is the tracked one. Observations without intensity
  // intrinsics are assumed to carry depth already registered to the image.
  const bool         hasIntensityCalib = obs->cameraParamsIntensity.fx() > 0;
  mrpt::img::TCamera cam = hasIntensityCalib ? obs->cameraParamsIntensity : obs->cameraParams;
  // The image size wins over the declared one, which some datasets get wrong:
  cam.ncols = static_cast<uint32_t>(obs->intensityImage.getWidth());
  cam.nrows = static_cast<uint32_t>(obs->intensityImage.getHeight());

  // Metric depth map (meters) in the intensity camera. It is only needed to
  // create landmarks, which happens in few frames, so build it on demand:
  std::optional<mrpt::math::CMatrixFloat> depth_m;
  const auto                              depthProvider = [&]() -> const mrpt::math::CMatrixFloat&
  {
    if (depth_m)
    {
      return *depth_m;
    }
    if (!hasIntensityCalib || depthIsRegistered(*obs))
    {
      depth_m = obs->rangeImage.asEigen().cast<float>() * obs->rangeUnits;
      return *depth_m;
    }
    mrpt::system::CTimeLoggerEntry tle(profiler_, "registerDepth");
    // Points from the depth optical frame into the intensity optical frame:
    const auto intensityFromDepth =
        (-obs->relativePoseIntensityWRTDepth) + mola::vision::opticalFrameInXForwardFrame();
    depth_m = mola::vision::registerDepthToCamera(
        obs->rangeImage, obs->rangeUnits, obs->cameraParams, cam, intensityFromDepth);
    return *depth_m;
  };

  // An identity sensorPose means "unknown", as in VisualSlam: poses are then
  // reported for the camera itself.
  if (!camera_pose_on_robot_ && obs->sensorPose.asVectorVal().norm() > 1e-9)
  {
    // Pose of the tracked camera's optical frame on the robot:
    camera_pose_on_robot_ =
        obs->sensorPose + (hasIntensityCalib ? obs->relativePoseIntensityWRTDepth
                                             : mola::vision::opticalFrameInXForwardFrame());
    MRPT_LOG_INFO_STREAM(
        "Camera-on-robot extrinsic taken from the observations: " << *camera_pose_on_robot_);
  }

  const mrpt::img::CImage gray = obs->intensityImage.grayscale();
  processFrame(gray, depthProvider, cam, obs->timestamp);

  MRPT_END
}

mrpt::poses::CPose3D RgbdSlam::processFrame(
    const mrpt::img::CImage& gray, const mrpt::math::CMatrixFloat& depth_m,
    const mrpt::img::TCamera& cam, const mrpt::Clock::time_point& timestamp)
{
  return processFrame(
      gray, [&]() -> const mrpt::math::CMatrixFloat& { return depth_m; }, cam, timestamp);
}

mrpt::poses::CPose3D RgbdSlam::processFrame(
    const mrpt::img::CImage& gray_in, const DepthProvider& depth, const mrpt::img::TCamera& cam,
    const mrpt::Clock::time_point& timestamp)
{
  using mrpt::math::TPoint2Df;
  using mrpt::math::TPoint3Df;

  mrpt::system::CTimeLoggerEntry tle(profiler_, "processFrame");

  camera_ = cam;
  profiler_.enter("grayscale");
  const mrpt::img::CImage gray = gray_in.grayscale();
  profiler_.leave("grayscale");

  mola::vision::RGBDParams depthParams;
  depthParams.min_depth = min_depth_;
  depthParams.max_depth = max_depth_;

  // -----------------------------------------------------------------------
  // First frame: initialize the map.
  // -----------------------------------------------------------------------
  if (prev_gray_.isEmpty())
  {
    pose_cw_ = mrpt::poses::CPose3D::Identity();
    pose_wc_ = mrpt::poses::CPose3D::Identity();
    motion_.reset();
    motion_.addPose(mrpt::Clock::toDouble(timestamp), pose_wc_);
    last_frame_time_ = mrpt::Clock::toDouble(timestamp);
    frame_period_    = 0;

    mola::vision::GridDistributorParams gp;
    gp.max_corners  = max_features_;
    gp.min_distance = min_distance_;
    mola::vision::GridFeatureDistributor dist(gp);
    const auto                           fresh = dist.detect(gray, {});

    track_pts_.clear();
    track_lm_.clear();
    for (const auto& p : fresh)
    {
      const float d = depthAt(depth(), p);
      const auto  X = mola::vision::backprojectPixel(p, d, cam, depthParams);
      if (!X)
      {
        continue;
      }
      // world == first camera frame, so world point == camera point.
      Landmark lm;
      lm.pos          = *X;
      lm.observations = 1;
      track_pts_.push_back(p);
      track_lm_.push_back(static_cast<int>(landmarks_.size()));
      landmarks_.push_back(lm);
    }

    insertCurrentKeyframe();
    prev_gray_       = gray;
    frame_count_     = 1;
    frames_since_kf_ = 0;
    trajectory_.push_back(pose_wc_.translation());
    recordPose(timestamp);
    publishLocalization(timestamp);
    publishMap(timestamp);
    publishViz2D(gray, true);
    publishViz3D();
    return pose_wc_;
  }

  // -----------------------------------------------------------------------
  // Track features into the new frame (LK) and localize it (robust PnP).
  // Both are seeded with the constant-velocity prediction over the real time
  // since the last localized frame, so missing frames are bridged by the
  // expected motion. If that fails, the vehicle may have stopped meanwhile:
  // retry assuming no motion. If both fail right after a gap in the frame
  // stream, the predicted pose is used (dead reckoning).
  // -----------------------------------------------------------------------
  const double t         = mrpt::Clock::toDouble(timestamp);
  const auto   pred_wc   = motion_.hasVelocity() ? motion_.predict(t) : pose_wc_;
  const bool   predMoves = (pred_wc.asVectorVal() - pose_wc_.asVectorVal()).norm() > 1e-4;

  const double frameDt  = t - last_frame_time_;
  const bool   afterGap = frame_period_ > 0 && frameDt > 2.5 * frame_period_;
  if (frameDt > 0 && !afterGap)
  {
    frame_period_ = frame_period_ > 0 ? 0.9 * frame_period_ + 0.1 * frameDt : frameDt;
  }
  last_frame_time_ = t;

  TrackAttempt trk = trackFrame(gray, pred_wc, predMoves, cam);
  if (!trk.localized && predMoves)
  {
    TrackAttempt still = trackFrame(gray, pose_wc_, false, cam);
    if (still.localized || still.inliers > trk.inliers)
    {
      trk = std::move(still);
    }
  }
  if (trk.localized)
  {
    pose_cw_ = trk.pose_cw;
    pose_wc_ = -pose_cw_;
    motion_.addPose(t, pose_wc_);
  }
  else if (afterGap)
  {
    // Frames are missing, and the vehicle kept moving meanwhile:
    pose_wc_ = pred_wc;
    pose_cw_ = -pose_wc_;
    MRPT_LOG_THROTTLE_WARN_STREAM(
        2.0, "Frame after a " << frameDt << " s gap not localized (" << trk.inliers
                              << " PnP inliers); dead-reckoning with the constant-velocity model.");
  }
  else
  {
    // A failure within a regular stream (e.g. motion blur): extrapolating
    // would only run the pose away from the map, so it is held instead.
    MRPT_LOG_THROTTLE_WARN_STREAM(
        2.0, "Frame not localized (" << trk.inliers << " PnP inliers, need " << min_pnp_points_
                                     << "); keeping the last pose.");
  }
  const auto& next_pts     = trk.next_pts;
  const auto& status       = trk.status;
  const auto& pnp_outlier  = trk.outlier;
  const float parallax_sum = trk.parallax_sum;
  const int   parallax_n   = trk.parallax_n;

  // -----------------------------------------------------------------------
  // Compact the tracking arrays: drop lost and PnP-rejected features.
  // -----------------------------------------------------------------------
  std::vector<TPoint2Df> kept_pts;
  std::vector<int>       kept_lm;
  for (size_t i = 0; i < next_pts.size(); ++i)
  {
    if (status[i] == mola::vision::TrackStatus::LOST || pnp_outlier[i])
    {
      // Cull spurious landmarks: a dropped feature whose landmark never made it
      // into a second keyframe (observation count below the threshold) is most
      // likely a one-shot false detection. Mark it bad so it is excluded from
      // PnP, BA, and the published map.
      const int lm = track_lm_[i];
      if (lm >= 0 && landmarks_[lm].observations < cull_min_obs_)
      {
        landmarks_[lm].bad = true;
      }
      continue;
    }
    kept_pts.push_back(next_pts[i]);
    kept_lm.push_back(track_lm_[i]);
  }
  track_pts_ = std::move(kept_pts);
  track_lm_  = std::move(kept_lm);

  // -----------------------------------------------------------------------
  // Spawn new landmarks from untracked, valid-depth features when tracking
  // gets sparse.
  // -----------------------------------------------------------------------
  if (static_cast<int>(track_pts_.size()) < redetect_below_)
  {
    mrpt::system::CTimeLoggerEntry      tlr(profiler_, "track.redetectBackproject");
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
      const float d  = depthAt(depth(), p);
      const auto  Xc = mola::vision::backprojectPixel(p, d, cam, depthParams);
      if (!Xc)
      {
        continue;
      }
      // Camera-frame point -> world via current camera pose.
      const auto Xw = pose_wc_.composePoint(mrpt::math::TPoint3D(*Xc));
      Landmark   lm;
      lm.pos          = TPoint3Df(Xw);
      lm.observations = 1;
      track_pts_.push_back(p);
      track_lm_.push_back(static_cast<int>(landmarks_.size()));
      landmarks_.push_back(lm);
    }
  }

  // -----------------------------------------------------------------------
  // Keyframe decision.
  // -----------------------------------------------------------------------
  ++frame_count_;
  ++frames_since_kf_;

  // Mean feature parallax over tracked features (proxy for viewpoint change):
  float median_parallax = 0.f;
  if (parallax_n > 0)
  {
    median_parallax = parallax_sum / static_cast<float>(parallax_n);
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
  stats.median_parallax_px = median_parallax;
  stats.frames_since_kf    = frames_since_kf_;

  if (selector.shouldBeKeyframe(stats))
  {
    {
      mrpt::system::CTimeLoggerEntry tlk(profiler_, "keyframe.insert");
      insertCurrentKeyframe();
    }
    {
      mrpt::system::CTimeLoggerEntry tlb(profiler_, "keyframe.windowedBA");
      runWindowedBA();
    }
    frames_since_kf_ = 0;
    publishMap(timestamp);
  }

  prev_gray_ = gray;
  trajectory_.push_back(pose_wc_.translation());
  recordPose(timestamp);
  publishLocalization(timestamp);
  {
    mrpt::system::CTimeLoggerEntry tlv(profiler_, "viz");
    publishViz2D(gray, false);
    publishViz3D();
  }
  return pose_wc_;
}

RgbdSlam::TrackAttempt RgbdSlam::trackFrame(
    const mrpt::img::CImage& gray, const mrpt::poses::CPose3D& seed_wc, bool projectSeed,
    const mrpt::img::TCamera& cam)
{
  using mrpt::math::TPoint2Df;
  using mrpt::math::TPoint3Df;

  TrackAttempt a;
  const auto   seed_cw = -seed_wc;

  // LK initial guess: where each landmark projects from the seed pose, or the
  // previous pixel for the "no motion" hypothesis.
  a.next_pts     = track_pts_;
  const float fx = static_cast<float>(cam.fx());
  const float fy = static_cast<float>(cam.fy());
  const float cx = static_cast<float>(cam.cx());
  const float cy = static_cast<float>(cam.cy());
  const float W  = static_cast<float>(gray.getWidth());
  const float H  = static_cast<float>(gray.getHeight());
  for (size_t i = 0; projectSeed && i < track_pts_.size(); ++i)
  {
    const int lm = track_lm_[i];
    if (lm < 0)
    {
      continue;
    }
    const auto Xc = seed_cw.composePoint(mrpt::math::TPoint3D(landmarks_[lm].pos));
    if (Xc.z < 0.05)
    {
      continue;
    }
    const float u = fx * static_cast<float>(Xc.x / Xc.z) + cx;
    const float v = fy * static_cast<float>(Xc.y / Xc.z) + cy;
    if (u >= 0.f && v >= 0.f && u < W && v < H)
    {
      a.next_pts[i] = {u, v};
    }
  }

  mola::vision::LKParams lk;
  lk.win_size          = lk_win_size_;
  lk.max_levels        = lk_max_levels_;
  lk.use_initial_guess = true;
  profiler_.enter("track.LK");
  mola::vision::calcOpticalFlowPyrLK(prev_gray_, gray, track_pts_, a.next_pts, a.status, lk);
  profiler_.leave("track.LK");

  // Gather 3D-2D correspondences with known landmarks for PnP.
  std::vector<TPoint3Df> worldPts;
  std::vector<TPoint2Df> pixels;
  std::vector<size_t>    corr_track_idx;  // back-reference into track arrays
  for (size_t i = 0; i < a.next_pts.size(); ++i)
  {
    if (a.status[i] == mola::vision::TrackStatus::LOST)
    {
      continue;
    }
    a.parallax_sum +=
        std::hypot(a.next_pts[i].x - track_pts_[i].x, a.next_pts[i].y - track_pts_[i].y);
    ++a.parallax_n;

    const int lm = track_lm_[i];
    if (lm < 0 || landmarks_[lm].bad)
    {
      continue;
    }
    worldPts.push_back(landmarks_[lm].pos);
    pixels.push_back(a.next_pts[i]);
    corr_track_idx.push_back(i);
  }

  a.outlier.assign(a.next_pts.size(), false);
  if (static_cast<int>(worldPts.size()) < min_pnp_points_)
  {
    return a;
  }
  mrpt::system::CTimeLoggerEntry tlp(profiler_, "track.PnP");
  const mola::vision::PnPParams  pp;
  const auto                     res = mola::vision::solvePnP(worldPts, pixels, cam, seed_cw, pp);
  a.inliers                          = res.num_inliers;
  a.localized                        = res.converged && res.num_inliers >= min_pnp_points_;
  a.pose_cw                          = res.pose;
  for (size_t k = 0; k < res.inliers.size(); ++k)
  {
    if (!res.inliers[k])
    {
      a.outlier[corr_track_idx[k]] = true;
    }
  }
  return a;
}

void RgbdSlam::insertCurrentKeyframe()
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
}

void RgbdSlam::runWindowedBA()
{
  const int W = std::min<int>(ba_window_size_, static_cast<int>(keyframes_.size()));
  if (W < 2)
  {
    return;  // need at least two keyframes to constrain anything useful
  }
  const size_t first = keyframes_.size() - W;

  // Collect involved landmarks and build a compact local indexing.
  std::vector<int>                         lm_global;  // local -> global
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
  fixed[0] = true;  // anchor gauge on the oldest window keyframe
  mola::vision::BAOptions opts;
  mola::vision::slidingWindowBA(poses, lms, obs, camera_, fixed, opts);

  // Write back refined poses and landmarks.
  for (int w = 0; w < W; ++w)
  {
    keyframes_[first + w].pose_cw = poses[w];
  }
  for (size_t k = 0; k < lm_global.size(); ++k)
  {
    landmarks_[lm_global[k]].pos = lms[k];
  }

  // The newest keyframe is the current frame: sync the live pose.
  pose_cw_ = keyframes_.back().pose_cw;
  pose_wc_ = -pose_cw_;
}

void RgbdSlam::publishLocalization(const mrpt::Clock::time_point& timestamp)
{
  if (!anyUpdateLocalizationSubscriber())
  {
    return;
  }
  LocalizationUpdate lu;
  lu.timestamp       = timestamp;
  lu.method          = "rgbd_slam";
  lu.reference_frame = "map";
  // Only claim "base_link" when the camera-on-robot extrinsic is known:
  lu.child_frame = camera_pose_on_robot_ ? "base_link" : "camera";
  lu.pose        = currentRobotPose().asTPose();
  advertiseUpdatedLocalization(lu);
}

void RgbdSlam::recordPose(const mrpt::Clock::time_point& timestamp)
{
  if (!estimated_trajectory_file_.empty())
  {
    auto lck = mrpt::lockHelper(estimated_trajectory_mtx_);
    estimated_trajectory_.insert(timestamp, currentRobotPose());
  }
}

void RgbdSlam::onQuit()
{
  if (estimated_trajectory_file_.empty())
  {
    return;
  }
  // Observations arrive in the data source's thread, while this runs in the
  // module's own one:
  mrpt::poses::CPose3DInterpolator traj;
  {
    auto lck = mrpt::lockHelper(estimated_trajectory_mtx_);
    traj     = estimated_trajectory_;
  }
  MRPT_LOG_INFO_STREAM(
      "Saving estimated trajectory (" << traj.size() << " poses) to '" << estimated_trajectory_file_
                                      << "' in TUM format.");
  if (!traj.saveToTextFile_TUM(estimated_trajectory_file_))
  {
    MRPT_LOG_ERROR_STREAM("Could not write '" << estimated_trajectory_file_ << "'");
  }
}

mrpt::poses::CPose3D RgbdSlam::currentRobotPose() const
{
  if (!camera_pose_on_robot_)
  {
    return pose_wc_;
  }
  return *camera_pose_on_robot_ + pose_wc_ + (-*camera_pose_on_robot_);
}

void RgbdSlam::publishViz2D(const mrpt::img::CImage& gray, bool just_initialized)
{
  if (!publish_viz_2d_ || !visualizer_)
  {
    return;
  }

  using mrpt::img::TColor;

  // Color base image so we can draw colored overlays.
  mrpt::img::CImage img = gray.colorImage();

  const TColor colTrack(0, 220, 0);  // tracked feature (treated as inlier)
  const TColor colNew(40, 160, 255);  // freshly initialized features
  const TColor col = just_initialized ? colNew : colTrack;

  for (const auto& p : track_pts_)
  {
    img.drawCircle({static_cast<int>(p.x), static_cast<int>(p.y)}, 3, col, 1);
  }

  std::ostringstream txt;
  txt << "frame " << frame_count_ << "  tracked: " << track_pts_.size()
      << "  landmarks: " << landmarks_.size() << "  kfs: " << keyframes_.size();
  img.textOut({8, 8}, txt.str(), TColor(255, 255, 255));

  // Wrap into a CObservationImage so the built-in MolaViz image handler shows it.
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

void RgbdSlam::publishViz3D()
{
  if (!publish_viz_3d_ || !visualizer_)
  {
    return;
  }

  auto scene = mrpt::viz::CSetOfObjects::Create();

  // --- sparse landmark cloud ---
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

  // --- estimated trajectory ---
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

  // --- past keyframe frustums (gray) ---
  for (const auto& kf : keyframes_)
  {
    auto fr = mrpt::viz::CFrustum::Create();
    fr->setColor_u8(mrpt::img::TColor(120, 120, 120, 180));
    // CFrustum points along +X, cameras along their optical +Z:
    fr->setPose((-kf.pose_cw) + (-mola::vision::opticalFrameInXForwardFrame()));
    scene->insert(fr);
  }

  // --- current camera frustum (green) ---
  auto cur = mrpt::viz::CFrustum::Create();
  cur->setColor_u8(mrpt::img::TColor(40, 200, 40));
  cur->setPose(pose_wc_ + (-mola::vision::opticalFrameInXForwardFrame()));
  scene->insert(cur);

  // Show everything in the "map" frame (the vehicle at the first frame):
  if (camera_pose_on_robot_)
  {
    scene->setPose(*camera_pose_on_robot_);
  }

  visualizer_->update_3d_object("rgbd_slam", scene);
}

void RgbdSlam::publishMap(const mrpt::Clock::time_point& timestamp)
{
  if (!anyUpdateMapSubscriber())
  {
    return;
  }
  // Landmarks live in the first camera frame; "map" is the first vehicle frame:
  const auto mapFromWorld = camera_pose_on_robot_.value_or(mrpt::poses::CPose3D::Identity());

  auto cloud = mrpt::maps::CSimplePointsMap::Create();
  cloud->reserve(landmarks_.size());
  for (const auto& lm : landmarks_)
  {
    if (lm.bad)
    {
      continue;
    }
    const auto p = mapFromWorld.composePoint(mrpt::math::TPoint3D(lm.pos));
    cloud->insertPoint(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
  }

  MapUpdate mu;
  mu.timestamp       = timestamp;
  mu.method          = "rgbd_slam";
  mu.reference_frame = "map";
  mu.map_name        = "sparse_landmarks";
  mu.map             = cloud;
  advertiseUpdatedMap(mu);
}
