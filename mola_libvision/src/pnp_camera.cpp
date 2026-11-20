/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/lie_utils.h>
#include <mola_libvision/pnp_solver.h>

#include <Eigen/Dense>
#include <cmath>

using namespace mola::vision;

namespace
{
double huberCost(double r, double delta)
{
  return (r <= delta) ? 0.5 * r * r : delta * (r - 0.5 * delta);
}

mrpt::poses::CPose3D toCPose3D(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
  mrpt::math::CMatrixDouble44 M;
  M.setIdentity();
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      M(r, c) = R(r, c);
    }
    M(r, 3) = t(r);
  }
  return mrpt::poses::CPose3D(M);
}

double robustCost(
    const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const std::vector<Eigen::Vector3d>& X,
    const std::vector<Eigen::Vector2d>& z, const CameraModel& cam, double delta)
{
  double cost = 0.0;
  for (size_t i = 0; i < X.size(); ++i)
  {
    Eigen::Vector2d px;
    if (!cam.project(R * X[i] + t, px))
    {
      // Unprojectable points pay a constant, bounded price.
      cost += huberCost(10.0 * delta, delta);
      continue;
    }
    cost += huberCost((px - z[i]).norm(), delta);
  }
  return cost;
}
}  // namespace

PnPResult mola::vision::solvePnP(
    const std::vector<Eigen::Vector3d>& worldPts, const std::vector<Eigen::Vector2d>& pixels,
    const CameraModel& cam, const mrpt::poses::CPose3D& initialPose, const PnPParams& params)
{
  PnPResult    result;
  const size_t N = worldPts.size();
  result.inliers.assign(N, false);
  result.pose = initialPose;
  if (N != pixels.size() || N < 4)
  {
    return result;
  }

  Eigen::Matrix3d R      = initialPose.getRotationMatrix().asEigen();
  Eigen::Vector3d t      = Eigen::Vector3d(initialPose.x(), initialPose.y(), initialPose.z());
  const double    delta  = params.huber_delta;
  double          cost   = robustCost(R, t, worldPts, pixels, cam, delta);
  double          lambda = std::max(static_cast<double>(params.lambda_initial), 1e-9);

  for (int it = 0; it < params.max_iters; ++it)
  {
    result.iterations             = it + 1;
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    for (size_t i = 0; i < N; ++i)
    {
      const Eigen::Vector3d       p = R * worldPts[i] + t;
      Eigen::Vector2d             px;
      Eigen::Matrix<double, 2, 3> Jp;
      if (!cam.project(p, px, &Jp))
      {
        continue;
      }
      const Eigen::Vector2d r  = px - pixels[i];
      const double          rn = r.norm();
      const double          w  = rn <= delta ? 1.0 : delta / rn;
      // Left perturbation on T_cw: p' = Exp(w) p + v.
      Eigen::Matrix<double, 3, 6> Jx;
      Jx.leftCols<3>()                    = -skew(p);
      Jx.rightCols<3>()                   = Eigen::Matrix3d::Identity();
      const Eigen::Matrix<double, 2, 6> J = Jp * Jx;
      H += w * J.transpose() * J;
      g += w * J.transpose() * r;
    }
    if (g.lpNorm<Eigen::Infinity>() < 1e-12)
    {
      result.converged = true;
      break;
    }
    bool accepted = false;
    for (int tries = 0; tries < 10 && !accepted; ++tries)
    {
      Eigen::Matrix<double, 6, 6> Hd = H;
      for (int k = 0; k < 6; ++k)
      {
        Hd(k, k) += lambda * std::max(H(k, k), 1e-9);
      }
      const Eigen::Matrix<double, 6, 1> dx = -Hd.ldlt().solve(g);
      if (!dx.allFinite())
      {
        lambda *= 10.0;
        continue;
      }
      const Eigen::Matrix3d dR    = so3Exp(dx.head<3>());
      const Eigen::Matrix3d R_new = dR * R;
      const Eigen::Vector3d t_new = dR * t + dx.tail<3>();
      const double          c_new = robustCost(R_new, t_new, worldPts, pixels, cam, delta);
      if (c_new < cost)
      {
        const double rel = (cost - c_new) / std::max(cost, 1e-300);
        R                = R_new;
        t                = t_new;
        cost             = c_new;
        lambda           = std::max(lambda / 3.0, 1e-12);
        accepted         = true;
        if (dx.norm() < params.eps_step || rel < params.eps_cost)
        {
          result.converged = true;
        }
      }
      else
      {
        lambda *= 10.0;
      }
    }
    if (!accepted || result.converged)
    {
      // No decrease possible: at a minimum to numerical precision.
      result.converged = true;
      break;
    }
  }

  // Inliers, final cost and covariance (unit pixel noise).
  Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
  double                      sq   = 0.0;
  for (size_t i = 0; i < N; ++i)
  {
    const Eigen::Vector3d       p = R * worldPts[i] + t;
    Eigen::Vector2d             px;
    Eigen::Matrix<double, 2, 3> Jp;
    if (!cam.project(p, px, &Jp))
    {
      continue;
    }
    const double e2 = (px - pixels[i]).squaredNorm();
    if (e2 < params.chi2_threshold)
    {
      result.inliers[i] = true;
      ++result.num_inliers;
      sq += e2;
      Eigen::Matrix<double, 3, 6> Jx;
      Jx.leftCols<3>()                    = -skew(p);
      Jx.rightCols<3>()                   = Eigen::Matrix3d::Identity();
      const Eigen::Matrix<double, 2, 6> J = Jp * Jx;
      info += J.transpose() * J;
    }
  }
  result.final_cost = static_cast<float>(0.5 * sq);
  result.pose       = toCPose3D(R, t);
  if (result.num_inliers >= 3)
  {
    Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(info);
    if (lu.isInvertible())
    {
      result.covariance = lu.inverse();
    }
  }
  return result;
}

PnPResult mola::vision::solvePnPRansac(
    const std::vector<Eigen::Vector3d>& worldPts, const std::vector<Eigen::Vector2d>& pixels,
    const CameraModel& cam, const RansacOptions& ransac, const PnPParams& params)
{
  PnPResult    result;
  const size_t N = worldPts.size();
  result.inliers.assign(N, false);
  if (N != pixels.size() || N < 4)
  {
    return result;
  }
  // Unproject; pixels outside the model's valid region are left out.
  std::vector<Eigen::Vector3d> X;
  std::vector<Eigen::Vector3d> b;
  X.reserve(N);
  b.reserve(N);
  for (size_t i = 0; i < N; ++i)
  {
    Eigen::Vector3d bi;
    if (cam.unproject(pixels[i], bi))
    {
      X.push_back(worldPts[i]);
      b.push_back(bi);
    }
  }
  RansacOptions ro = ransac;
  ro.threshold     = ransac.threshold / (0.5 * (cam.fx() + cam.fy()));
  const auto est   = estimateAbsolutePose(X, b, ro);
  if (!est.report.success)
  {
    return result;
  }
  const mrpt::poses::CPose3D init = toCPose3D(est.pose.R, est.pose.t);

  // Refine on the MSAC consensus only, then classify every correspondence at
  // the refined pose (outliers must not pull the refinement).
  std::vector<Eigen::Vector3d> Xin;
  std::vector<Eigen::Vector2d> zin;
  for (size_t i = 0, k = 0; i < N; ++i)
  {
    Eigen::Vector3d bi;
    if (!cam.unproject(pixels[i], bi))
    {
      continue;
    }
    if (est.report.inliers[k])
    {
      Xin.push_back(worldPts[i]);
      zin.push_back(pixels[i]);
    }
    ++k;
  }
  const PnPResult refined = solvePnP(Xin, zin, cam, init, params);
  result                  = refined;
  result.inliers.assign(N, false);
  result.num_inliers      = 0;
  const Eigen::Matrix3d R = refined.pose.getRotationMatrix().asEigen();
  const Eigen::Vector3d t(refined.pose.x(), refined.pose.y(), refined.pose.z());
  for (size_t i = 0; i < N; ++i)
  {
    Eigen::Vector2d px;
    if (cam.project(R * worldPts[i] + t, px) &&
        (px - pixels[i]).squaredNorm() < params.chi2_threshold)
    {
      result.inliers[i] = true;
      ++result.num_inliers;
    }
  }
  return result;
}
