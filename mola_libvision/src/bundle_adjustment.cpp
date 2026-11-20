/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtsam/config.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <mola_libvision/bundle_adjustment.h>
#include <mrpt/math/CMatrixFixed.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

// GTSAM 4.3 replaced boost::optional Jacobian arguments by raw pointers.
#if GTSAM_VERSION_NUMERIC >= 40300
#define MOLA_GTSAM_JACOBIANS gtsam::OptionalMatrixVecType H = nullptr
#else
#define MOLA_GTSAM_JACOBIANS boost::optional<std::vector<gtsam::Matrix>&> H = boost::none
#endif

using namespace mola::vision;

namespace
{
// Error (in sigma units) assigned to an observation that cannot be projected.
// Constant, with zero Jacobians: such terms cannot pull the solution anywhere.
constexpr double kInvalidError = 50.0;

enum class PoseVar : uint8_t
{
  Constant,
  Full,
  RotationOnly
};

/** Reprojection factor of one observation. Each of the pose, point and
 *  intrinsics may be a variable or a constant; the keys hold the variables,
 *  in that order. */
class CalibReprojectionFactor : public gtsam::NoiseModelFactor
{
 public:
  struct Setup
  {
    PoseVar         pose_var          = PoseVar::Full;
    bool            point_var         = true;
    bool            point_at_infinity = false;
    bool            intrinsics_var    = false;
    gtsam::Pose3    pose_const;  ///< T_wc, used when constant (or its center when rotation-only)
    Eigen::Vector3d point_const = Eigen::Vector3d::Zero();
    CameraModel     camera;  ///< reference camera (constant intrinsics, or fixed part)
    std::shared_ptr<const IntrinsicsParameterization> param;
    Eigen::Vector2d                                   z     = Eigen::Vector2d::Zero();
    double                                            sigma = 1.0;  ///< pixels
  };

  CalibReprojectionFactor(
      const gtsam::SharedNoiseModel& noise, const gtsam::KeyVector& keys, Setup setup)
      : gtsam::NoiseModelFactor(noise, keys), s_(std::move(setup))
  {
  }

  gtsam::Vector unwhitenedError(const gtsam::Values& x, MOLA_GTSAM_JACOBIANS) const override
  {
    size_t k = 0;
    // Camera pose (camera in world, T_wc).
    gtsam::Pose3 T  = s_.pose_const;
    gtsam::Key   kp = 0;
    if (s_.pose_var == PoseVar::Full)
    {
      kp = keys()[k++];
      T  = x.at<gtsam::Pose3>(kp);
    }
    else if (s_.pose_var == PoseVar::RotationOnly)
    {
      kp = keys()[k++];
      T  = gtsam::Pose3(x.at<gtsam::Rot3>(kp), s_.pose_const.translation());
    }
    const size_t pose_slot  = (s_.pose_var == PoseVar::Constant) ? SIZE_MAX : 0;
    const size_t point_slot = s_.point_var ? k++ : SIZE_MAX;
    const size_t intr_slot  = s_.intrinsics_var ? k++ : SIZE_MAX;

    // Point in the camera frame, with Jacobians wrt pose and point.
    Eigen::Matrix<double, 3, 6> Hpose = Eigen::Matrix<double, 3, 6>::Zero();
    Eigen::Matrix3d             Hrot  = Eigen::Matrix3d::Zero();
    Eigen::MatrixXd             Hpt;
    gtsam::Point3               pc;
    if (!s_.point_at_infinity)
    {
      const gtsam::Point3 Xw =
          s_.point_var ? x.at<gtsam::Point3>(keys()[point_slot]) : gtsam::Point3(s_.point_const);
      Eigen::Matrix3d Hp;
      if (s_.pose_var == PoseVar::RotationOnly)
      {
        pc = T.rotation().unrotate(Xw - T.translation(), Hrot, Hp);
      }
      else
      {
        pc = T.transformTo(Xw, Hpose, Hp);
      }
      Hpt = Hp;
    }
    else
    {
      Eigen::Matrix<double, 3, 2> Hd = Eigen::Matrix<double, 3, 2>::Zero();
      Eigen::Vector3d             d  = s_.point_const.normalized();
      if (s_.point_var)
      {
        d = x.at<gtsam::Unit3>(keys()[point_slot]).unitVector(Hd);
      }
      Eigen::Matrix<double, 3, 6> HR_pose;
      const gtsam::Rot3           R = T.rotation(HR_pose);
      Eigen::Matrix3d             H1;
      Eigen::Matrix3d             H2;
      pc = R.unrotate(gtsam::Point3(d), H1, H2);
      if (s_.pose_var == PoseVar::RotationOnly)
      {
        Hrot = H1;
      }
      else
      {
        Hpose = H1 * HR_pose;
      }
      Hpt = H2 * Hd;
    }

    CameraModel cam = s_.camera;
    if (s_.intrinsics_var)
    {
      s_.param->fromFree(x.at<gtsam::Vector>(keys()[intr_slot]), cam);
    }

    Eigen::Vector2d                          px;
    Eigen::Matrix<double, 2, 3>              Jp;
    Eigen::Matrix<double, 2, Eigen::Dynamic> Jk;
    const bool                               want = static_cast<bool>(H);
    const bool                               ok =
        cam.project(pc, px, want ? &Jp : nullptr, (want && s_.intrinsics_var) ? &Jk : nullptr);

    if (!ok)
    {
      const double e = kInvalidError * s_.sigma;
      if (H)
      {
        auto& Hs = *H;
        for (size_t i = 0; i < size(); ++i)
        {
          Hs[i] = gtsam::Matrix::Zero(2, static_cast<Eigen::Index>(dimOf(i)));
        }
      }
      return gtsam::Vector2(e, e);
    }

    if (H)
    {
      auto& Hs = *H;
      if (pose_slot != SIZE_MAX)
      {
        if (s_.pose_var == PoseVar::RotationOnly)
        {
          Hs[pose_slot] = Jp * Hrot;
        }
        else
        {
          Hs[pose_slot] = Jp * Hpose;
        }
      }
      if (point_slot != SIZE_MAX)
      {
        Hs[point_slot] = Jp * Hpt;
      }
      if (intr_slot != SIZE_MAX)
      {
        Hs[intr_slot] = Jk * s_.param->jacobian();
      }
    }
    return px - s_.z;
  }

 private:
  Setup s_;

  // Tangent-space dimension of the i-th variable of this factor.
  size_t dimOf(size_t i) const
  {
    size_t k = 0;
    if (s_.pose_var != PoseVar::Constant)
    {
      if (i == k)
      {
        return s_.pose_var == PoseVar::Full ? 6 : 3;
      }
      ++k;
    }
    if (s_.point_var)
    {
      if (i == k)
      {
        return s_.point_at_infinity ? 2 : 3;
      }
      ++k;
    }
    return static_cast<size_t>(s_.param ? s_.param->dim() : 0);
  }
};

/** Unary factor on the distance from a fixed point to a camera center: fixes
 *  the scale gauge of a monocular reconstruction. */
class CenterDistanceFactor : public gtsam::NoiseModelFactor
{
 public:
  CenterDistanceFactor(
      const gtsam::SharedNoiseModel& noise, gtsam::Key key, const Eigen::Vector3d& origin,
      double distance)
      : gtsam::NoiseModelFactor(noise, gtsam::KeyVector{key}), origin_(origin), distance_(distance)
  {
  }

  gtsam::Vector unwhitenedError(const gtsam::Values& x, MOLA_GTSAM_JACOBIANS) const override
  {
    const gtsam::Pose3          T = x.at<gtsam::Pose3>(keys()[0]);
    Eigen::Matrix<double, 3, 6> Ht;
    const gtsam::Point3         c = T.translation(Ht);
    const Eigen::Vector3d       d = c - origin_;
    const double                n = d.norm();
    if (H)
    {
      Eigen::Matrix<double, 1, 6> J = Eigen::Matrix<double, 1, 6>::Zero();
      if (n > 1e-15)
      {
        J = (d / n).transpose() * Ht;
      }
      (*H)[0] = J;
    }
    return gtsam::Vector1(n - distance_);
  }

 private:
  Eigen::Vector3d origin_;
  double          distance_;
};

gtsam::SharedNoiseModel makeNoise(double sigma, const BundleAdjustmentOptions& o)
{
  auto base = gtsam::noiseModel::Isotropic::Sigma(2, sigma);
  switch (o.loss)
  {
    case RobustLoss::Huber:
      return gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(o.loss_scale), base);
    case RobustLoss::Cauchy:
      return gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Cauchy::Create(o.loss_scale), base);
    case RobustLoss::None:
    default:
      return base;
  }
}

gtsam::Key poseKey(size_t i) { return gtsam::Symbol('x', i); }

// Conversions through the rotation matrix (no Euler angles involved).
gtsam::Pose3 toGtsam(const mrpt::poses::CPose3D& p)
{
  return gtsam::Pose3(
      gtsam::Rot3(p.getRotationMatrix().asEigen()), gtsam::Point3(p.x(), p.y(), p.z()));
}

mrpt::poses::CPose3D fromGtsam(const gtsam::Pose3& p)
{
  return mrpt::poses::CPose3D(mrpt::math::CMatrixDouble44(p.matrix()));
}
gtsam::Key pointKey(size_t i) { return gtsam::Symbol('l', i); }
gtsam::Key intrKey(size_t i) { return gtsam::Symbol('k', i); }

}  // namespace

BundleAdjustmentResult mola::vision::bundleAdjust(
    BundleAdjustmentProblem& problem, const BundleAdjustmentOptions& options)
{
  BundleAdjustmentResult out;
  const size_t           nPoses   = problem.poses.size();
  const size_t           nPoints  = problem.points.size();
  const size_t           nCameras = problem.cameras.size();
  if (nCameras == 0 || nPoses == 0)
  {
    out.message = "empty problem";
    return out;
  }
  // Validate the optional per-pose and per-point maps.
  if (!problem.pose_camera.empty())
  {
    if (problem.pose_camera.size() != nPoses)
    {
      out.message = "pose_camera must be empty or have one entry per pose";
      return out;
    }
    for (const int c : problem.pose_camera)
    {
      if (c < 0 || static_cast<size_t>(c) >= nCameras)
      {
        out.message = "pose_camera refers to a camera that does not exist";
        return out;
      }
    }
  }
  if (!problem.point_at_infinity.empty() && problem.point_at_infinity.size() != nPoints)
  {
    out.message = "point_at_infinity must be empty or have one entry per point";
    return out;
  }
  auto cameraOf = [&](size_t pose) -> size_t
  { return problem.pose_camera.empty() ? 0 : static_cast<size_t>(problem.pose_camera[pose]); };

  std::vector<gtsam::Pose3> T_wc(nPoses);
  for (size_t i = 0; i < nPoses; ++i)
  {
    T_wc[i] = toGtsam(-problem.poses[i]);
  }

  // Observations that take part: valid indices and, for variable points, at
  // least two of them (a single observation is always matched exactly by
  // moving the point: no information, and an undetermined depth).
  std::vector<int> obs_count(nPoints, 0);
  for (const auto& ob : problem.observations)
  {
    if (ob.point >= 0 && static_cast<size_t>(ob.point) < nPoints && ob.pose >= 0 &&
        static_cast<size_t>(ob.pose) < nPoses)
    {
      ++obs_count[static_cast<size_t>(ob.point)];
    }
  }
  std::vector<size_t> admitted;
  for (size_t o = 0; o < problem.observations.size(); ++o)
  {
    const auto& ob = problem.observations[o];
    if (ob.pose < 0 || ob.point < 0 || static_cast<size_t>(ob.pose) >= nPoses ||
        static_cast<size_t>(ob.point) >= nPoints)
    {
      continue;
    }
    if (!options.fix_points && obs_count[static_cast<size_t>(ob.point)] < 2)
    {
      continue;
    }
    admitted.push_back(o);
  }

  // Points handled as directions: those given at infinity, and, in
  // rotation-only mode, finite variable points whose observing centers
  // coincide (their depth is unobservable): those keep their range.
  std::vector<bool>            inf(nPoints, false);
  std::vector<double>          keep_range(nPoints, -1.0);
  std::vector<Eigen::Vector3d> range_center(nPoints, Eigen::Vector3d::Zero());
  for (size_t j = 0; j < nPoints; ++j)
  {
    inf[j] = !problem.point_at_infinity.empty() && problem.point_at_infinity[j];
  }
  if (options.rotation_only && !options.fix_points)
  {
    std::vector<int>  first_pose(nPoints, -1);
    std::vector<bool> common(nPoints, true);
    for (const size_t o : admitted)
    {
      const auto pi = static_cast<size_t>(problem.observations[o].pose);
      const auto li = static_cast<size_t>(problem.observations[o].point);
      if (inf[li])
      {
        continue;
      }
      if (first_pose[li] < 0)
      {
        first_pose[li] = static_cast<int>(pi);
      }
      else if (
          (T_wc[pi].translation() - T_wc[static_cast<size_t>(first_pose[li])].translation())
              .norm() > 1e-9)
      {
        common[li] = false;
      }
    }
    for (size_t j = 0; j < nPoints; ++j)
    {
      if (first_pose[j] >= 0 && common[j])
      {
        const Eigen::Vector3d C = T_wc[static_cast<size_t>(first_pose[j])].translation();
        const double          r = (problem.points[j] - C).norm();
        if (r > 0.0)
        {
          inf[j]          = true;
          keep_range[j]   = r;
          range_center[j] = C;
        }
      }
    }
  }
  // Direction of a point handled at infinity.
  auto direction = [&](size_t j) -> Eigen::Vector3d
  {
    if (keep_range[j] > 0.0)
    {
      return (problem.points[j] - range_center[j]).normalized();
    }
    return problem.points[j].normalized();
  };

  // Which poses take part, and which have finite points to fix their center.
  std::vector<bool> pose_obs(nPoses, false);
  std::vector<bool> pose_finite(nPoses, false);
  for (const size_t o : admitted)
  {
    const auto pi   = static_cast<size_t>(problem.observations[o].pose);
    const auto li   = static_cast<size_t>(problem.observations[o].point);
    pose_obs[pi]    = true;
    pose_finite[pi] = pose_finite[pi] || !inf[li];
  }

  std::vector<bool> pose_fixed(nPoses, false);
  bool              any_fixed = false;
  for (size_t i = 0; i < nPoses && i < options.fixed_poses.size(); ++i)
  {
    pose_fixed[i] = options.fixed_poses[i];
    any_fixed     = any_fixed || (pose_fixed[i] && pose_obs[i]);
  }

  // Automatic gauge, anchored on poses that take part.
  const bool gauge_free = !any_fixed && !options.fix_points;
  int        anchor     = -1;
  int        scale_pose = -1;
  if (gauge_free && options.fix_gauge)
  {
    for (size_t i = 0; i < nPoses && anchor < 0; ++i)
    {
      if (pose_obs[i])
      {
        anchor = static_cast<int>(i);
      }
    }
    if (anchor >= 0)
    {
      pose_fixed[static_cast<size_t>(anchor)] = true;
      if (!options.rotation_only)
      {
        const Eigen::Vector3d C0   = T_wc[static_cast<size_t>(anchor)].translation();
        double                best = 0.0;
        for (size_t i = 0; i < nPoses; ++i)
        {
          if (!pose_finite[i] || static_cast<int>(i) == anchor)
          {
            continue;
          }
          const double d = (T_wc[i].translation() - C0).norm();
          if (d > best)
          {
            best       = d;
            scale_pose = static_cast<int>(i);
          }
        }
        if (best < 1e-12)
        {
          scale_pose = -1;
        }
      }
    }
  }

  // Pose variable type: a pose whose observations are all directions has an
  // unconstrained center, so only its rotation is estimated.
  std::vector<PoseVar> pose_var(nPoses, PoseVar::Constant);
  for (size_t i = 0; i < nPoses; ++i)
  {
    if (pose_fixed[i] || !pose_obs[i])
    {
      continue;
    }
    pose_var[i] =
        (options.rotation_only || !pose_finite[i]) ? PoseVar::RotationOnly : PoseVar::Full;
  }

  std::vector<std::shared_ptr<const IntrinsicsParameterization>> params(nCameras);
  for (size_t c = 0; c < nCameras; ++c)
  {
    if (c < options.intrinsics.size() && options.intrinsics[c].has_value())
    {
      auto p =
          std::make_shared<IntrinsicsParameterization>(problem.cameras[c], *options.intrinsics[c]);
      if (p->dim() > 0)
      {
        params[c] = p;
      }
    }
  }

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values               values;
  std::vector<bool>           pose_used(nPoses, false);
  std::vector<bool>           point_used(nPoints, false);
  std::vector<bool>           cam_used(nCameras, false);
  int                         projectable = 0;

  for (const size_t o : admitted)
  {
    const auto& ob = problem.observations[o];
    const auto  pi = static_cast<size_t>(ob.pose);
    const auto  li = static_cast<size_t>(ob.point);
    const auto  ci = cameraOf(pi);

    CalibReprojectionFactor::Setup s;
    gtsam::KeyVector               keys;
    s.pose_var = pose_var[pi];
    if (s.pose_var != PoseVar::Constant)
    {
      keys.push_back(poseKey(pi));
      pose_used[pi] = true;
    }
    s.pose_const        = T_wc[pi];
    s.point_at_infinity = inf[li];
    s.point_const       = inf[li] ? direction(li) : problem.points[li];
    s.point_var         = !options.fix_points;
    if (s.point_var)
    {
      keys.push_back(pointKey(li));
      point_used[li] = true;
    }
    s.camera = problem.cameras[ci];
    if (params[ci])
    {
      s.intrinsics_var = true;
      s.param          = params[ci];
      keys.push_back(intrKey(ci));
      cam_used[ci] = true;
    }
    s.z     = ob.pixel;
    s.sigma = ob.sigma;
    if (keys.empty())
    {
      continue;
    }
    // At least one observation must be projectable at the start, or the
    // optimizer would see only constant penalties.
    {
      const Eigen::Vector3d pc =
          inf[li] ? Eigen::Vector3d(T_wc[pi].rotation().unrotate(gtsam::Point3(s.point_const)))
                  : Eigen::Vector3d(T_wc[pi].transformTo(s.point_const));
      Eigen::Vector2d px;
      if (s.camera.project(pc, px))
      {
        ++projectable;
      }
    }
    graph.emplace_shared<CalibReprojectionFactor>(makeNoise(ob.sigma, options), keys, s);
  }
  if (!graph.empty() && projectable == 0)
  {
    out.message = "no observation can be projected with the initial values";
    return out;
  }

  for (size_t i = 0; i < nPoses; ++i)
  {
    if (!pose_used[i])
    {
      continue;
    }
    if (pose_var[i] == PoseVar::RotationOnly)
    {
      values.insert(poseKey(i), T_wc[i].rotation());
    }
    else
    {
      values.insert(poseKey(i), T_wc[i]);
    }
  }
  for (size_t j = 0; j < nPoints; ++j)
  {
    if (!point_used[j])
    {
      continue;
    }
    if (inf[j])
    {
      values.insert(pointKey(j), gtsam::Unit3(gtsam::Point3(direction(j))));
    }
    else
    {
      values.insert(pointKey(j), gtsam::Point3(problem.points[j]));
    }
  }
  gtsam::KeyVector intr_keys;
  for (size_t c = 0; c < nCameras; ++c)
  {
    if (cam_used[c])
    {
      const gtsam::Vector v0 = params[c]->toFree(problem.cameras[c]);
      values.insert(intrKey(c), v0);
      intr_keys.push_back(intrKey(c));
      if (c < options.intrinsics_prior_sigmas.size() &&
          options.intrinsics_prior_sigmas[c].size() == v0.size())
      {
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector>>(
            intrKey(c), v0,
            gtsam::noiseModel::Diagonal::Sigmas(options.intrinsics_prior_sigmas[c]));
      }
    }
  }
  // Scale constraints act on camera centers: only for full pose variables.
  auto addScale = [&](size_t ka, size_t kb)
  {
    if (!pose_used[kb] || pose_var[kb] != PoseVar::Full)
    {
      return;
    }
    const double d0 = (T_wc[kb].translation() - T_wc[ka].translation()).norm();
    if (d0 < 1e-12)
    {
      return;
    }
    graph.emplace_shared<CenterDistanceFactor>(
        gtsam::noiseModel::Isotropic::Sigma(1, 1e-4 * d0), poseKey(kb),
        Eigen::Vector3d(T_wc[ka].translation()), d0);
  };
  if (!options.rotation_only)
  {
    for (const auto& [a, b] : options.scale_constraints)
    {
      if (a < 0 || b < 0 || static_cast<size_t>(a) >= nPoses || static_cast<size_t>(b) >= nPoses ||
          !pose_fixed[static_cast<size_t>(a)])
      {
        continue;
      }
      addScale(static_cast<size_t>(a), static_cast<size_t>(b));
    }
    if (scale_pose >= 0 && anchor >= 0)
    {
      addScale(static_cast<size_t>(anchor), static_cast<size_t>(scale_pose));
    }
  }

  if (graph.empty() || values.empty())
  {
    out.message = "nothing to optimize";
    return out;
  }

  try
  {
    const gtsam::Ordering           ordering = intr_keys.empty()
                                                   ? gtsam::Ordering::Colamd(graph)
                                                   : gtsam::Ordering::ColamdConstrainedLast(graph, intr_keys);
    gtsam::LevenbergMarquardtParams lm;
    lm.maxIterations    = options.max_iterations;
    lm.relativeErrorTol = options.relative_error_tol;
    lm.absoluteErrorTol = options.absolute_error_tol;
    lm.ordering         = ordering;
    if (options.verbose)
    {
      lm.setVerbosityLM("SUMMARY");
    }
    out.initial_cost = graph.error(values);
    gtsam::LevenbergMarquardtOptimizer opt(graph, values, lm);
    const gtsam::Values                result = opt.optimize();
    out.final_cost                            = graph.error(result);
    out.iterations                            = static_cast<int>(opt.iterations());
    out.success                               = std::isfinite(out.final_cost);

    // Write back.
    for (size_t i = 0; i < nPoses; ++i)
    {
      if (!pose_used[i])
      {
        continue;
      }
      gtsam::Pose3 T;
      if (pose_var[i] == PoseVar::RotationOnly)
      {
        T = gtsam::Pose3(result.at<gtsam::Rot3>(poseKey(i)), T_wc[i].translation());
      }
      else
      {
        T = result.at<gtsam::Pose3>(poseKey(i));
      }
      problem.poses[i] = -fromGtsam(T);
    }
    for (size_t j = 0; j < nPoints; ++j)
    {
      if (!point_used[j])
      {
        continue;
      }
      if (inf[j])
      {
        const Eigen::Vector3d d = result.at<gtsam::Unit3>(pointKey(j)).unitVector();
        problem.points[j] =
            keep_range[j] > 0.0 ? Eigen::Vector3d(range_center[j] + keep_range[j] * d) : d;
      }
      else
      {
        problem.points[j] = result.at<gtsam::Point3>(pointKey(j));
      }
    }
    for (size_t c = 0; c < nCameras; ++c)
    {
      if (cam_used[c])
      {
        params[c]->fromFree(result.at<gtsam::Vector>(intrKey(c)), problem.cameras[c]);
      }
    }

    if (options.compute_covariance && !intr_keys.empty())
    {
      out.intrinsics_covariance.assign(nCameras, Eigen::MatrixXd());
      out.free_intrinsics_covariance.assign(nCameras, Eigen::MatrixXd());
      try
      {
        // Very weak priors on every pose and point: they only remove exact
        // null spaces (e.g. the depth of a point seen from one center), which
        // would otherwise make the whole marginal computation fail, and leave
        // well-determined directions untouched.
        double scale = 1.0;
        {
          std::vector<double> d;
          for (size_t j = 0; j < nPoints; ++j)
          {
            if (point_used[j] && !inf[j])
            {
              d.push_back((problem.points[j] -
                           T_wc[anchor >= 0 ? static_cast<size_t>(anchor) : 0].translation())
                              .norm());
            }
          }
          if (!d.empty())
          {
            std::nth_element(
                d.begin(), d.begin() + static_cast<std::ptrdiff_t>(d.size() / 2), d.end());
            scale = std::max(1e-9, d[d.size() / 2]);
          }
        }
        gtsam::NonlinearFactorGraph cov_graph = graph;
        const double                weak      = 1e3;  // prior sigma, in scene and radian units
        for (size_t i = 0; i < nPoses; ++i)
        {
          if (!pose_used[i])
          {
            continue;
          }
          if (pose_var[i] == PoseVar::RotationOnly)
          {
            cov_graph.emplace_shared<gtsam::PriorFactor<gtsam::Rot3>>(
                poseKey(i), result.at<gtsam::Rot3>(poseKey(i)),
                gtsam::noiseModel::Isotropic::Sigma(3, weak));
          }
          else
          {
            gtsam::Vector6 sg;
            sg << weak, weak, weak, weak * scale, weak * scale, weak * scale;
            cov_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                poseKey(i), result.at<gtsam::Pose3>(poseKey(i)),
                gtsam::noiseModel::Diagonal::Sigmas(sg));
          }
        }
        for (size_t j = 0; j < nPoints; ++j)
        {
          if (!point_used[j])
          {
            continue;
          }
          if (inf[j])
          {
            cov_graph.emplace_shared<gtsam::PriorFactor<gtsam::Unit3>>(
                pointKey(j), result.at<gtsam::Unit3>(pointKey(j)),
                gtsam::noiseModel::Isotropic::Sigma(2, weak));
          }
          else
          {
            cov_graph.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(
                pointKey(j), result.at<gtsam::Point3>(pointKey(j)),
                gtsam::noiseModel::Isotropic::Sigma(3, weak * scale));
          }
        }
        const gtsam::Marginals marg(cov_graph, result, ordering);
        for (size_t c = 0; c < nCameras; ++c)
        {
          if (!cam_used[c])
          {
            continue;
          }
          const Eigen::MatrixXd  C          = marg.marginalCovariance(intrKey(c));
          const Eigen::MatrixXd& J          = params[c]->jacobian();
          out.free_intrinsics_covariance[c] = C;
          out.intrinsics_covariance[c]      = J * C * J.transpose();
        }
      }
      catch (const std::exception& e)
      {
        out.message = std::string("covariance not available: ") + e.what();
        out.intrinsics_covariance.clear();
        out.free_intrinsics_covariance.clear();
      }
      catch (...)
      {
        out.message = "covariance not available";
        out.intrinsics_covariance.clear();
        out.free_intrinsics_covariance.clear();
      }
    }
  }
  catch (const std::exception& e)
  {
    out.success = false;
    out.message = std::string("optimization failed: ") + e.what();
    return out;
  }
  catch (...)
  {
    out.success = false;
    out.message = "optimization failed: unknown exception";
    return out;
  }

  // Final per-observation reprojection errors.
  out.residuals.assign(problem.observations.size(), std::numeric_limits<double>::quiet_NaN());
  for (size_t o = 0; o < problem.observations.size(); ++o)
  {
    const auto& ob = problem.observations[o];
    if (ob.pose < 0 || ob.point < 0 || static_cast<size_t>(ob.pose) >= nPoses ||
        static_cast<size_t>(ob.point) >= nPoints)
    {
      continue;
    }
    const auto                  pi = static_cast<size_t>(ob.pose);
    const auto                  li = static_cast<size_t>(ob.point);
    const mrpt::poses::CPose3D& P  = problem.poses[pi];
    const Eigen::Matrix3d       R  = P.getRotationMatrix().asEigen();
    const Eigen::Vector3d       t(P.x(), P.y(), P.z());
    const bool given_inf     = !problem.point_at_infinity.empty() && problem.point_at_infinity[li];
    const Eigen::Vector3d pc = given_inf ? Eigen::Vector3d(R * problem.points[li])
                                         : Eigen::Vector3d(R * problem.points[li] + t);
    Eigen::Vector2d       px;
    if (problem.cameras[cameraOf(pi)].project(pc, px))
    {
      out.residuals[o] = (px - ob.pixel).norm();
    }
  }
  if (out.success &&
      std::none_of(
          out.residuals.begin(), out.residuals.end(), [](double e) { return std::isfinite(e); }))
  {
    out.success = false;
    out.message = "no observation can be projected with the solution";
  }
  return out;
}
