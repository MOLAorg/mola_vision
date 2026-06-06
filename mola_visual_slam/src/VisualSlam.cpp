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
#include <mrpt/maps/CSimplePointsMap.h>
#include <mrpt/math/CMatrixFixed.h>
#include <mrpt/obs/CObservationImage.h>
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
    getI("max_features", max_features_);
    getF("min_distance", min_distance_);
    getI("redetect_below", redetect_below_);
    getI("lk_win_size", lk_win_size_);
    getI("lk_max_levels", lk_max_levels_);
    getI("min_pnp_points", min_pnp_points_);
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
      obs->load();
      pending_left_     = obs->image;
      pending_left_cam_ = obs->cameraParams;
      pending_left_ts_  = obs->timestamp;
      have_left_        = true;
    }
    else if (obs->sensorLabel == right_label_)
    {
      obs->load();
      pending_right_    = obs->image;
      pending_right_ts_ = obs->timestamp;
      have_right_       = true;
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
      processStereoFrame(
          pending_left_, pending_right_, pending_left_cam_, stereo_baseline_, pending_left_ts_);
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
  obs->load();
  if (obs->image.isEmpty())
  {
    return;
  }
  processFrame(obs->image, obs->cameraParams, obs->timestamp);
  MRPT_END
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

  if (prev_gray_.isEmpty())
  {
    // Very first frame: seed reference features for the bootstrap.
    detectInitialFeatures(gray);
    prev_gray_ = gray;
    publishViz2D(gray);
    return pose_wc_;
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
    return pose_wc_;
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
      runWindowedBA();
    }
    frames_since_kf_ = 0;
    publishMap(timestamp);
  }

  prev_gray_ = gray;
  trajectory_.push_back(pose_wc_.translation());
  publishLocalization(timestamp);
  {
    mrpt::system::CTimeLoggerEntry t6(profiler_, "viz");
    publishViz2D(gray);
    publishViz3D();
  }
  return pose_wc_;
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
  if (static_cast<int>(worldPts.size()) >= min_pnp_points_)
  {
    mrpt::system::CTimeLoggerEntry tle(profiler_, "track.PnP");
    const auto res = mola::vision::solvePnP(worldPts, pixels, camera_, pose_cw_);
    if (res.converged)
    {
      pose_cw_ = res.pose;
      pose_wc_ = -pose_cw_;
    }
    for (size_t k = 0; k < res.inliers.size(); ++k)
    {
      if (!res.inliers[k])
      {
        pnp_outlier[corr_idx[k]] = true;
      }
    }
  }

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
  dp.max_depth = 100.0f;

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
    Landmark   lm;
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
      return pose_wc_;  // not enough stereo matches yet; wait
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
    return pose_wc_;
  }

  // -------- Tracking: LK (left t-1 -> left t) + PnP --------
  std::vector<TPoint2Df>                 next_pts;
  std::vector<mola::vision::TrackStatus> status;
  mola::vision::LKParams                 lk;
  lk.win_size   = lk_win_size_;
  lk.max_levels = lk_max_levels_;
  profiler_.enter("track.LK");
  mola::vision::calcOpticalFlowPyrLK(prev_gray_, grayL, track_pts_, next_pts, status, lk);
  profiler_.leave("track.LK");

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
  if (static_cast<int>(worldPts.size()) >= min_pnp_points_)
  {
    mrpt::system::CTimeLoggerEntry tlp(profiler_, "track.PnP");
    const auto                     res = mola::vision::solvePnP(worldPts, pixels, cam, pose_cw_);
    if (res.converged)
    {
      pose_cw_ = res.pose;
      pose_wc_ = -pose_cw_;
    }
    for (size_t k = 0; k < res.inliers.size(); ++k)
    {
      if (!res.inliers[k])
      {
        pnp_outlier[corr_idx[k]] = true;
      }
    }
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
  {
    mrpt::system::CTimeLoggerEntry tlv(profiler_, "viz");
    publishViz2D(grayL);
    publishViz3D();
  }
  return pose_wc_;
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
  lu.child_frame     = "base_link";
  lu.pose            = pose_wc_.asTPose();
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
