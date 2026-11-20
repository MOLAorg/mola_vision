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
#include <mola_libvision/bundle_adjustment.h>
#include <mrpt/math/CMatrixFixed.h>

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
    Eigen::Vector2d                                   z = Eigen::Vector2d::Zero();
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
      const double e = kInvalidError * noiseModel()->sigmas()[0];
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
  auto cameraOf = [&](size_t pose) -> size_t
  { return problem.pose_camera.empty() ? 0 : static_cast<size_t>(problem.pose_camera[pose]); };
  auto atInf = [&](size_t pt)
  { return !problem.point_at_infinity.empty() && problem.point_at_infinity[pt]; };

  // Which variables exist.
  std::vector<bool> pose_fixed(nPoses, false);
  bool              any_fixed = false;
  for (size_t i = 0; i < nPoses && i < options.fixed_poses.size(); ++i)
  {
    pose_fixed[i] = options.fixed_poses[i];
    any_fixed     = any_fixed || pose_fixed[i];
  }
  std::vector<gtsam::Pose3> T_wc(nPoses);
  for (size_t i = 0; i < nPoses; ++i)
  {
    T_wc[i] = toGtsam(-problem.poses[i]);
  }

  const bool gauge_free = !any_fixed && !options.fix_points;
  int        scale_pose = -1;
  if (gauge_free && options.fix_gauge)
  {
    pose_fixed[0] = true;
    if (!options.rotation_only)
    {
      double best = 0.0;
      for (size_t i = 1; i < nPoses; ++i)
      {
        const double d = (T_wc[i].translation() - T_wc[0].translation()).norm();
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

  // A variable point needs two observations: a single one is always matched
  // exactly by moving the point, adds no information, and leaves its depth
  // undetermined (a singular system for the covariance).
  std::vector<int> obs_count(nPoints, 0);
  for (const auto& ob : problem.observations)
  {
    if (ob.point >= 0 && static_cast<size_t>(ob.point) < nPoints && ob.pose >= 0 &&
        static_cast<size_t>(ob.pose) < nPoses)
    {
      ++obs_count[static_cast<size_t>(ob.point)];
    }
  }

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values               values;
  std::vector<bool>           pose_used(nPoses, false);
  std::vector<bool>           point_used(nPoints, false);
  std::vector<bool>           cam_used(nCameras, false);

  for (const auto& ob : problem.observations)
  {
    if (ob.pose < 0 || ob.point < 0 || static_cast<size_t>(ob.pose) >= nPoses ||
        static_cast<size_t>(ob.point) >= nPoints)
    {
      continue;
    }
    const auto pi = static_cast<size_t>(ob.pose);
    const auto li = static_cast<size_t>(ob.point);
    const auto ci = cameraOf(pi);
    if (!options.fix_points && obs_count[li] < 2)
    {
      continue;
    }

    CalibReprojectionFactor::Setup s;
    gtsam::KeyVector               keys;
    if (!pose_fixed[pi])
    {
      s.pose_var = options.rotation_only ? PoseVar::RotationOnly : PoseVar::Full;
      keys.push_back(poseKey(pi));
      pose_used[pi] = true;
    }
    else
    {
      s.pose_var = PoseVar::Constant;
    }
    s.pose_const        = T_wc[pi];
    s.point_at_infinity = atInf(li);
    s.point_const       = problem.points[li];
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
    s.z = ob.pixel;
    if (keys.empty())
    {
      continue;
    }
    graph.emplace_shared<CalibReprojectionFactor>(makeNoise(ob.sigma, options), keys, s);
  }

  for (size_t i = 0; i < nPoses; ++i)
  {
    if (!pose_used[i])
    {
      continue;
    }
    if (options.rotation_only)
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
    if (atInf(j))
    {
      values.insert(pointKey(j), gtsam::Unit3(gtsam::Point3(problem.points[j])));
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
      values.insert(intrKey(c), gtsam::Vector(params[c]->toFree(problem.cameras[c])));
      intr_keys.push_back(intrKey(c));
    }
  }
  if (scale_pose >= 0 && pose_used[static_cast<size_t>(scale_pose)])
  {
    const auto   k  = static_cast<size_t>(scale_pose);
    const double d0 = (T_wc[k].translation() - T_wc[0].translation()).norm();
    graph.emplace_shared<CenterDistanceFactor>(
        gtsam::noiseModel::Isotropic::Sigma(1, 1e-4 * d0), poseKey(k),
        Eigen::Vector3d(T_wc[0].translation()), d0);
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
      if (options.rotation_only)
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
      if (atInf(j))
      {
        problem.points[j] = result.at<gtsam::Unit3>(pointKey(j)).unitVector();
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
        const gtsam::Marginals marg(graph, result, ordering);
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
    }
  }
  catch (const std::exception& e)
  {
    out.success = false;
    out.message = std::string("optimization failed: ") + e.what();
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
    const Eigen::Vector3d       pc = atInf(li) ? Eigen::Vector3d(R * problem.points[li])
                                               : Eigen::Vector3d(R * problem.points[li] + t);
    Eigen::Vector2d             px;
    if (problem.cameras[cameraOf(pi)].project(pc, px))
    {
      out.residuals[o] = (px - ob.pixel).norm();
    }
  }
  return out;
}
