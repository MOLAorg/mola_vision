/* -------------------------------------------------------------------------
 * mola_visual_slam: monocular / stereo visual SLAM front-end for MOLA.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Ground-truth-exact stereo tests. A textured "room" (five planes carrying a
 * band-limited procedural texture) is ray-traced analytically for each camera,
 * so every pixel comes from an exact projection of an exact surface: there is
 * no correspondence noise beyond 8-bit quantization, and the world frame is the
 * first left-camera frame, which is exactly what VisualSlam estimates. Absolute
 * pose error can therefore be checked directly, with no trajectory alignment
 * and no dataset needed.
 *
 * These are the tests that pin the metric scale and the recovered attitude of
 * the stereo pipeline; a real-data regression (KITTI, GrandTour) can then be
 * read as "dataset difficulty", not "unknown geometry bug".
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/optical_flow.h>
#include <mola_libvision/stereo_matcher.h>
#include <mola_visual_slam/VisualSlam.h>
#include <mrpt/core/Clock.h>
#include <mrpt/core/format.h>
#include <mrpt/img/CImage.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/poses/CPose3D.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace
{
constexpr int   kW  = 480;
constexpr int   kH  = 360;
constexpr float kFx = 400.f;
constexpr float kFy = 400.f;
constexpr float kCx = 239.5f;
constexpr float kCy = 179.5f;

mrpt::img::TCamera makeCamera()
{
  mrpt::img::TCamera cam;
  cam.ncols = kW;
  cam.nrows = kH;
  cam.setIntrinsicParamsFromValues(kFx, kFy, kCx, kCy);
  return cam;
}

/** A closed, axis-aligned textured room, ray-traced exactly.
 *
 *  Camera convention: X right, Y down, Z forward. The camera always sits inside
 *  the box, so every ray leaves through exactly one face: the smallest positive
 *  intersection parameter is the visible surface, no depth sorting needed.
 *
 *  The texture is band-limited on purpose: white noise smoothed to a feature
 *  size that stays above ~4 image pixels at the far wall and below the LK window
 *  at the near floor, so neither aliasing nor a featureless patch is what a test
 *  ends up measuring.
 */
class TexturedRoom
{
 public:
  TexturedRoom()
  {
    buildTexture();
    // {axis, value, sign}: axis 0=x, 1=y (down), 2=z.
    faces_ = {{
        {0, -3.0f},  // left wall
        {0, +3.0f},  // right wall
        {1, -2.0f},  // ceiling
        {1, +2.0f},  // floor
        {2, +10.0f},  // front wall
        {2, -2.5f},  // back wall
    }};
  }

  /** Renders the room as seen from a camera at \p pose_wc (camera-in-world),
   *  with the intrinsics of \p cam (pinhole, distortion ignored). */
  [[nodiscard]] mrpt::img::CImage render(
      const mrpt::poses::CPose3D& pose_wc, const mrpt::img::TCamera& cam) const
  {
    const auto            R = pose_wc.getRotationMatrix();
    const auto            T = pose_wc.translation();
    const Eigen::Vector3d C(T.x, T.y, T.z);
    const double          fx = cam.fx();
    const double          fy = cam.fy();
    const double          cx = cam.cx();
    const double          cy = cam.cy();
    const int             W  = static_cast<int>(cam.ncols);
    const int             H  = static_cast<int>(cam.nrows);

    mrpt::img::CImage img(W, H, mrpt::img::CH_GRAY);
    // 2x2 supersampling: the projected texture is magnified at the near floor
    // and minified at the far wall, and averaging four samples keeps the latter
    // from aliasing into the sub-pixel accuracy the tests measure.
    constexpr int kSS      = 2;
    const double  ss_step  = 1.0 / kSS;
    const double  ss_start = 0.5 * ss_step - 0.5;

    for (int v = 0; v < H; ++v)
    {
      for (int u = 0; u < W; ++u)
      {
        double acc = 0;
        for (int sy = 0; sy < kSS; ++sy)
        {
          for (int sx = 0; sx < kSS; ++sx)
          {
            const double          uu = u + ss_start + sx * ss_step;
            const double          vv = v + ss_start + sy * ss_step;
            const Eigen::Vector3d d_cam((uu - cx) / fx, (vv - cy) / fy, 1.0);
            const Eigen::Vector3d d_world = R.asEigen() * d_cam;
            acc += traceRay(C, d_world);
          }
        }
        acc /= (kSS * kSS);
        img.at<uint8_t>(u, v) =
            static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(acc)), 0, 255));
      }
    }
    return img;
  }

  /** Exact depth (camera Z, meters) of the surface seen at pixel (u, v). */
  [[nodiscard]] double depthAt(
      const mrpt::poses::CPose3D& pose_wc, const mrpt::img::TCamera& cam, double u, double v) const
  {
    const auto            T = pose_wc.translation();
    const Eigen::Vector3d C(T.x, T.y, T.z);
    const Eigen::Vector3d d_cam((u - cam.cx()) / cam.fx(), (v - cam.cy()) / cam.fy(), 1.0);
    const Eigen::Vector3d d_world = pose_wc.getRotationMatrix().asEigen() * d_cam;
    double                best_t  = std::numeric_limits<double>::max();
    for (const auto& f : faces_)
    {
      const double dn = d_world(f.axis);
      if (std::abs(dn) < 1e-12)
      {
        continue;
      }
      const double t = (f.value - C(f.axis)) / dn;
      if (t > 1e-6 && t < best_t)
      {
        best_t = t;
      }
    }
    // d_cam.z() == 1, so the ray parameter is already the depth along Z.
    return best_t;
  }

 private:
  struct Face
  {
    int   axis;
    float value;
  };

  /** Intensity seen along the ray from \p C in direction \p d. */
  [[nodiscard]] double traceRay(const Eigen::Vector3d& C, const Eigen::Vector3d& d) const
  {
    double best_t    = std::numeric_limits<double>::max();
    int    best_face = -1;
    for (size_t f = 0; f < faces_.size(); ++f)
    {
      const int    a  = faces_[f].axis;
      const double dn = d(a);
      if (std::abs(dn) < 1e-12)
      {
        continue;
      }
      const double t = (faces_[f].value - C(a)) / dn;
      if (t > 1e-6 && t < best_t)
      {
        best_t    = t;
        best_face = static_cast<int>(f);
      }
    }
    if (best_face < 0)
    {
      return 0.0;
    }
    const Eigen::Vector3d X = C + best_t * d;
    // In-plane coordinates: the two axes that are not the face normal. A
    // per-face offset keeps the six faces from sharing the same texture patch,
    // which would make stereo matching ambiguous across a corner.
    const int    a  = faces_[best_face].axis;
    const int    i0 = (a + 1) % 3;
    const int    i1 = (a + 2) % 3;
    const double su = X(i0) + 7.3 * best_face;
    const double sv = X(i1) - 4.1 * best_face;
    return sampleTexture(su, sv);
  }

  /** Bilinear, wrapping lookup into the procedural texture. */
  [[nodiscard]] double sampleTexture(double su, double sv) const
  {
    const double x    = su * kTexPxPerMeter;
    const double y    = sv * kTexPxPerMeter;
    const int    N    = kTexN;
    auto         wrap = [N](int i) { return ((i % N) + N) % N; };
    const int    x0   = static_cast<int>(std::floor(x));
    const int    y0   = static_cast<int>(std::floor(y));
    const double fx   = x - x0;
    const double fy   = y - y0;
    const int    xa   = wrap(x0);
    const int    ya   = wrap(y0);
    const int    xb   = wrap(x0 + 1);
    const int    yb   = wrap(y0 + 1);
    const double p00  = tex_[ya * N + xa];
    const double p10  = tex_[ya * N + xb];
    const double p01  = tex_[yb * N + xa];
    const double p11  = tex_[yb * N + xb];
    return (1 - fy) * ((1 - fx) * p00 + fx * p10) + fy * ((1 - fx) * p01 + fx * p11);
  }

  void buildTexture()
  {
    const int                             N = kTexN;
    std::mt19937                          rng(20260905);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    std::vector<float>                    a(static_cast<size_t>(N) * N);
    for (auto& v : a)
    {
      v = u01(rng);
    }
    // Two separable box passes ~ a Gaussian: turns white noise into a
    // band-limited pattern whose feature size is set by kBlurRadius.
    std::vector<float> b(a.size());
    for (int pass = 0; pass < 2; ++pass)
    {
      for (int y = 0; y < N; ++y)
      {
        for (int x = 0; x < N; ++x)
        {
          float s = 0;
          for (int k = -kBlurRadius; k <= kBlurRadius; ++k)
          {
            s += a[static_cast<size_t>(y) * N + ((x + k) % N + N) % N];
          }
          b[static_cast<size_t>(y) * N + x] = s / (2 * kBlurRadius + 1);
        }
      }
      for (int y = 0; y < N; ++y)
      {
        for (int x = 0; x < N; ++x)
        {
          float s = 0;
          for (int k = -kBlurRadius; k <= kBlurRadius; ++k)
          {
            s += b[static_cast<size_t>(((y + k) % N + N) % N) * N + x];
          }
          a[static_cast<size_t>(y) * N + x] = s / (2 * kBlurRadius + 1);
        }
      }
    }
    // Stretch to a wide, well-conditioned intensity range.
    float mn = a[0];
    float mx = a[0];
    for (const float v : a)
    {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    tex_.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
      tex_[i] = 20.f + 215.f * (a[i] - mn) / std::max(1e-6f, mx - mn);
    }
  }

  static constexpr int   kTexN          = 512;
  static constexpr int   kBlurRadius    = 2;
  static constexpr float kTexPxPerMeter = 40.f;

  std::vector<float>  tex_;
  std::array<Face, 6> faces_{};
};

/** Runs a whole stereo trajectory through VisualSlam and returns the last
 *  estimated camera-in-world pose. \p gt is filled with the true poses. */
mrpt::poses::CPose3D runTrajectory(
    mola::VisualSlam& slam, const TexturedRoom& room, const std::vector<mrpt::poses::CPose3D>& traj,
    double baseline)
{
  const auto           cam = makeCamera();
  mrpt::poses::CPose3D last;
  for (const auto& pose_wc : traj)
  {
    // The right camera sits +baseline along the left camera's own x axis
    // (an ideal, already-rectified pair).
    const mrpt::poses::CPose3D right_wc = pose_wc + mrpt::poses::CPose3D(baseline, 0, 0, 0, 0, 0);
    const auto                 L        = room.render(pose_wc, cam);
    const auto                 R        = room.render(right_wc, cam);
    last = slam.processStereoFrame(L, R, cam, baseline, mrpt::Clock::now());
  }
  return last;
}

double rotationErrorDeg(const mrpt::poses::CPose3D& a, const mrpt::poses::CPose3D& b)
{
  const auto   d     = (-a) + b;  // a^-1 * b
  const auto   R     = d.getRotationMatrix();
  const double trace = R(0, 0) + R(1, 1) + R(2, 2);
  return mrpt::RAD2DEG(std::acos(std::clamp(0.5 * (trace - 1.0), -1.0, 1.0)));
}

double positionErrorM(const mrpt::poses::CPose3D& a, const mrpt::poses::CPose3D& b)
{
  return (a.translation() - b.translation()).norm();
}

constexpr double kBaseline = 0.15;  // [m]
}  // namespace

// ---------------------------------------------------------------------------
// Component check 1: is the stereo depth itself biased?
//
// Metric scale in a stereo VO is exactly as good as the disparity that produced
// the map, so a systematic depth bias here becomes a systematic trajectory
// scale error downstream. The room's exact ray-traced depth is the reference.
// ---------------------------------------------------------------------------
TEST(VisualSlamSynthetic, StereoDepthIsUnbiased)
{
  const TexturedRoom         room;
  const auto                 cam = makeCamera();
  const mrpt::poses::CPose3D pose_wc(0, 0, 0, 0, 0, 0);
  const mrpt::poses::CPose3D right_wc(kBaseline, 0, 0, 0, 0, 0);
  const auto                 L = room.render(pose_wc, cam);
  const auto                 R = room.render(right_wc, cam);

  mola::vision::GridDistributorParams gp;
  gp.max_corners   = 400;
  gp.min_distance  = 12.0f;
  const auto feats = mola::vision::GridFeatureDistributor(gp).detect(L, {});
  ASSERT_GT(feats.size(), 100u);

  const auto sm = mola::vision::matchStereo(L, R, feats, cam.fx(), kBaseline);
  ASSERT_GT(sm.num_valid, 80);

  std::vector<double> rel_err;
  for (size_t i = 0; i < feats.size(); ++i)
  {
    if (!sm.valid[i])
    {
      continue;
    }
    const double Ztrue = room.depthAt(pose_wc, cam, feats[i].x, feats[i].y);
    rel_err.push_back((sm.depth[i] - Ztrue) / Ztrue);
  }
  ASSERT_GT(rel_err.size(), 80u);
  std::sort(rel_err.begin(), rel_err.end());
  const double median = rel_err[rel_err.size() / 2];
  double       mean   = 0;
  for (const double e : rel_err)
  {
    mean += e;
  }
  mean /= static_cast<double>(rel_err.size());
  std::cout << "[synthetic depth] n=" << rel_err.size() << "  median rel. error=" << 100 * median
            << "%  mean=" << 100 * mean << "%  p10=" << 100 * rel_err[rel_err.size() / 10]
            << "%  p90=" << 100 * rel_err[9 * rel_err.size() / 10] << "%\n";

  EXPECT_LT(std::abs(median), 0.005);  // < 0.5% systematic depth bias
}

// ---------------------------------------------------------------------------
// Component check 2: is the temporal LK flow biased?
//
// Forward motion makes every patch grow, which a pure-translation LK model does
// not represent; if that produced a systematic under- or over-shoot, the whole
// trajectory would inherit it. The reference flow is computed by projecting the
// exactly ray-traced 3D point of each feature into the next frame.
// ---------------------------------------------------------------------------
TEST(VisualSlamSynthetic, TemporalFlowIsUnbiased)
{
  const TexturedRoom         room;
  const mrpt::poses::CPose3D pose0(0, 0, 0, 0, 0, 0);
  const mrpt::poses::CPose3D pose1(0, 0, 0.06, 0, 0, 0);
  const auto                 cam = makeCamera();
  const auto                 I0  = room.render(pose0, cam);
  const auto                 I1  = room.render(pose1, cam);

  mola::vision::GridDistributorParams gp;
  gp.max_corners   = 400;
  gp.min_distance  = 12.0f;
  const auto feats = mola::vision::GridFeatureDistributor(gp).detect(I0, {});
  ASSERT_GT(feats.size(), 100u);

  std::vector<mrpt::math::TPoint2Df>     tracked;
  std::vector<mola::vision::TrackStatus> status;
  mola::vision::LKParams                 lk;
  if (const char* e = std::getenv("LK_EPS"))
  {
    lk.eps = std::atof(e);
  }
  if (const char* e = std::getenv("LK_ITERS"))
  {
    lk.max_iters = std::atoi(e);
  }
  if (const char* e = std::getenv("LK_WIN"))
  {
    lk.win_size = std::atoi(e);
  }
  mola::vision::calcOpticalFlowPyrLK(I0, I1, feats, tracked, status, lk);

  std::vector<double> rel_along;
  std::vector<double> err_along;  // signed error projected on the true flow
  std::vector<double> err_norm;
  for (size_t i = 0; i < feats.size(); ++i)
  {
    if (status[i] != mola::vision::TrackStatus::OK)
    {
      continue;
    }
    const double Z = room.depthAt(pose0, cam, feats[i].x, feats[i].y);
    // Exact 3D point in frame 0, then its exact projection in frame 1.
    const mrpt::math::TPoint3D X0((feats[i].x - kCx) * Z / kFx, (feats[i].y - kCy) * Z / kFy, Z);
    const auto                 X1 = (-pose1 + pose0).composePoint(X0);
    const double               u1 = kFx * X1.x / X1.z + kCx;
    const double               v1 = kFy * X1.y / X1.z + kCy;

    const double fu = u1 - feats[i].x;
    const double fv = v1 - feats[i].y;
    const double fn = std::hypot(fu, fv);
    if (fn < 0.5)
    {
      continue;  // no measurable flow at the focus of expansion
    }
    const double du = tracked[i].x - u1;
    const double dv = tracked[i].y - v1;
    err_along.push_back((du * fu + dv * fv) / fn);
    rel_along.push_back(((du * fu + dv * fv) / fn) / fn);
    err_norm.push_back(std::hypot(du, dv));
  }
  ASSERT_GT(err_along.size(), 60u);

  std::sort(err_along.begin(), err_along.end());
  std::sort(err_norm.begin(), err_norm.end());
  std::sort(rel_along.begin(), rel_along.end());
  const double median_along = err_along[err_along.size() / 2];
  const double median_norm  = err_norm[err_norm.size() / 2];
  const double median_rel   = rel_along[rel_along.size() / 2];
  std::cout << "[synthetic flow] n=" << err_along.size()
            << "  median signed error along flow=" << median_along
            << " px  median |error|=" << median_norm
            << " px  median relative gain error=" << 100 * median_rel << "%\n";

  EXPECT_LT(std::abs(median_along), 0.1);
  EXPECT_LT(median_norm, 0.3);
}

// ---------------------------------------------------------------------------
// Pure forward translation: the classic stereo-VO scale test. Depth comes only
// from disparity, so any systematic disparity or tracking bias shows up here as
// a trajectory that is uniformly too short or too long.
// ---------------------------------------------------------------------------
TEST(VisualSlamSynthetic, ForwardTranslationIsMetric)
{
  mola::VisualSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);
  const TexturedRoom room;

  std::vector<mrpt::poses::CPose3D> traj;
  constexpr int                     N    = 40;
  constexpr double                  step = 0.06;  // [m] per frame, along +Z
  traj.reserve(N);
  for (int k = 0; k < N; ++k)
  {
    traj.emplace_back(0.0, 0.0, k * step, 0.0, 0.0, 0.0);
  }

  const auto est = runTrajectory(slam, room, traj, kBaseline);
  ASSERT_TRUE(slam.isInitialized());

  const auto&  gt        = traj.back();
  const double travelled = gt.translation().norm();
  ASSERT_GT(travelled, 2.0);

  const double perr = positionErrorM(est, gt);
  const double rerr = rotationErrorDeg(est, gt);
  std::cout << "[synthetic fwd] travelled=" << travelled << " m  position error=" << perr << " m ("
            << 100.0 * perr / travelled << "%)  attitude error=" << rerr << " deg\n";

  EXPECT_LT(perr, 0.02 * travelled);  // < 2% of the distance travelled
  EXPECT_LT(rerr, 0.5);
}

// ---------------------------------------------------------------------------
// Translation + a large, continuously changing yaw. This is the case the
// GrandTour missions actually exercise, and the one that a wrong rotation
// branch or a left/right pose-composition mistake breaks first.
// ---------------------------------------------------------------------------
TEST(VisualSlamSynthetic, YawWhileTranslatingRecoversAttitude)
{
  mola::VisualSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);
  const TexturedRoom room;

  std::vector<mrpt::poses::CPose3D> traj;
  constexpr int                     N = 40;
  traj.reserve(N);
  for (int k = 0; k < N; ++k)
  {
    // Camera-frame yaw is a rotation about the camera's Y (down) axis, which in
    // MRPT's CPose3D(x,y,z,yaw,pitch,roll) is the "pitch" slot.
    const double turn = mrpt::DEG2RAD(0.5 * k);  // up to 19.5 deg
    traj.emplace_back(0.0, 0.0, 0.05 * k, 0.0, turn, 0.0);
  }

  const auto est = runTrajectory(slam, room, traj, kBaseline);
  ASSERT_TRUE(slam.isInitialized());

  const auto&  gt        = traj.back();
  const double travelled = gt.translation().norm();
  const double perr      = positionErrorM(est, gt);
  const double rerr      = rotationErrorDeg(est, gt);
  std::cout << "[synthetic yaw] final GT yaw=" << mrpt::RAD2DEG(gt.pitch())
            << " deg  est=" << mrpt::RAD2DEG(est.pitch()) << " deg  position error=" << perr
            << " m  attitude error=" << rerr << " deg\n";

  // The recovered turn must have the right SIGN and magnitude, not just a
  // plausible size: this is what a wrong essential/PnP branch gets backwards.
  EXPECT_GT(est.pitch(), 0.5 * gt.pitch());
  EXPECT_LT(rerr, 1.0);
  EXPECT_LT(perr, 0.02 * travelled + 0.02);
}

// ---------------------------------------------------------------------------
// Pivot in place: no translation at all, only yaw. This is the legged-robot
// motion that makes monocular essential-matrix bootstrap degenerate; stereo
// PnP against a metric map must handle it, and must NOT invent translation.
// ---------------------------------------------------------------------------
TEST(VisualSlamSynthetic, PureRotationInPlace)
{
  mola::VisualSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);
  const TexturedRoom room;

  std::vector<mrpt::poses::CPose3D> traj;
  constexpr int                     N = 30;
  traj.reserve(N);
  for (int k = 0; k < N; ++k)
  {
    traj.emplace_back(0.0, 0.0, 0.0, 0.0, mrpt::DEG2RAD(0.4 * k), 0.0);
  }

  const auto est = runTrajectory(slam, room, traj, kBaseline);
  ASSERT_TRUE(slam.isInitialized());

  const auto& gt = traj.back();
  std::cout << "[synthetic pivot] GT yaw=" << mrpt::RAD2DEG(gt.pitch())
            << " deg  est=" << mrpt::RAD2DEG(est.pitch())
            << " deg  spurious translation=" << est.translation().norm() << " m\n";

  EXPECT_LT(rotationErrorDeg(est, gt), 1.0);
  EXPECT_LT(est.translation().norm(), 0.05);  // no invented motion
}

// ---------------------------------------------------------------------------
// A raw, NOT pre-rectified rig, driven through the real onNewObservation()
// entry point: the two cameras have different intrinsics AND a non-zero
// relative rotation, exactly like GrandTour's alphasense pair. VisualSlam must
// build a CStereoRectifyMap from the given `right_camera_pose`, rectify each
// incoming pair, and still recover the true metric trajectory - reported in the
// PHYSICAL camera frame, not the rotated rectified one.
//
// This also pins the sign convention: `right_camera_pose` is the pose of the
// right camera in the left camera's frame, so a side-by-side rig has a
// POSITIVE x. Getting that backwards used to be invisible (both rectified
// images flip together, so they stay epipolar-aligned) while silently negating
// every disparity.
// ---------------------------------------------------------------------------
TEST(VisualSlamSynthetic, UnrectifiedRigWithRelativeRotation)
{
  const TexturedRoom room;

  // Left and right cameras differ in focal length and principal point, and the
  // right one is yawed/pitched by ~1 deg, as on a real hand-mounted rig.
  mrpt::img::TCamera camL;
  camL.ncols = kW;
  camL.nrows = kH;
  camL.setIntrinsicParamsFromValues(400.0, 400.0, 239.5, 179.5);
  mrpt::img::TCamera camR;
  camR.ncols = kW;
  camR.nrows = kH;
  camR.setIntrinsicParamsFromValues(406.0, 406.0, 244.0, 176.0);

  const mrpt::poses::CPose3D rightInLeft(
      kBaseline, 0.0005, -0.001, mrpt::DEG2RAD(-1.2), mrpt::DEG2RAD(0.2), mrpt::DEG2RAD(0.05));

  mola::VisualSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);
  slam.initialize(mola::Yaml::FromText(mrpt::format(
      "params:\n  mode: stereo\n  left_label: image_0\n  right_label: image_1\n"
      "  publish_viz_2d: false\n  publish_viz_3d: false\n"
      "  right_camera_pose: \"%.9f %.9f %.9f %.9f %.9f %.9f\"\n",
      rightInLeft.x(), rightInLeft.y(), rightInLeft.z(), mrpt::RAD2DEG(rightInLeft.yaw()),
      mrpt::RAD2DEG(rightInLeft.pitch()), mrpt::RAD2DEG(rightInLeft.roll()))));

  std::vector<mrpt::poses::CPose3D> traj;
  constexpr int                     N = 35;
  traj.reserve(N);
  for (int k = 0; k < N; ++k)
  {
    traj.emplace_back(0.0, 0.0, 0.05 * k, 0.0, mrpt::DEG2RAD(0.3 * k), 0.0);
  }

  auto   t  = mrpt::Clock::now();
  size_t nf = 0;
  for (const auto& pose_wc : traj)
  {
    const auto right_wc = pose_wc + rightInLeft;

    auto obsL          = mrpt::obs::CObservationImage::Create();
    obsL->sensorLabel  = "image_0";
    obsL->cameraParams = camL;
    obsL->timestamp    = t;
    obsL->image        = room.render(pose_wc, camL);

    auto obsR          = mrpt::obs::CObservationImage::Create();
    obsR->sensorLabel  = "image_1";
    obsR->cameraParams = camR;
    obsR->timestamp    = t;
    obsR->image        = room.render(right_wc, camR);

    slam.onNewObservation(obsL);
    slam.onNewObservation(obsR);
    t += std::chrono::milliseconds(100);
    ++nf;
  }
  ASSERT_EQ(nf, static_cast<size_t>(N));
  ASSERT_TRUE(slam.isInitialized());

  const auto   est       = slam.currentPose();
  const auto&  gt        = traj.back();
  const double travelled = gt.translation().norm();
  const double perr      = positionErrorM(est, gt);
  const double rerr      = rotationErrorDeg(est, gt);
  std::cout << "[synthetic unrect] travelled=" << travelled << " m  position error=" << perr
            << " m (" << 100.0 * perr / travelled << "%)  attitude error=" << rerr << " deg\n";

  // The whole point: the estimate stays in the physical left-camera frame, so
  // it is directly comparable to the ground truth with no extra rotation.
  EXPECT_LT(rerr, 1.0);
  EXPECT_LT(perr, 0.05 * travelled);
  EXPECT_GT(est.pitch(), 0.5 * gt.pitch());
}
