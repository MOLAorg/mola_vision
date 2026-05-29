/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Adapted in spirit from lightweight_vio (MIT, (c) 2025 Seungwon Choi);
 * re-derived for MRPT types without Ceres.
 * ------------------------------------------------------------------------- */
#include <mola_libvision/lie_utils.h>  // skew(), so3Exp()
#include <mola_libvision/lm_solver.h>  // huberWeight()
#include <mola_libvision/pnp_solver.h>

#include <Eigen/Dense>
#include <cmath>

using namespace mola::vision;

PnPResult mola::vision::solvePnP(
    const std::vector<mrpt::math::TPoint3Df>& worldPts,
    const std::vector<mrpt::math::TPoint2Df>& pixels, const mrpt::img::TCamera& cam,
    const mrpt::poses::CPose3D& initialPose, const PnPParams& params)
{
  PnPResult    result;
  const size_t N = worldPts.size();
  result.inliers.assign(N, false);

  if (N != pixels.size() || N < 4)
  {
    // Not enough correspondences for a unique pose.
    result.pose = initialPose;
    return result;
  }

  const double fx = cam.fx();
  const double fy = cam.fy();
  const double cx = cam.cx();
  const double cy = cam.cy();

  // Current estimate as (R, t):  Xc = R * Xw + t.
  Eigen::Matrix3d R = initialPose.getRotationMatrix().asEigen();
  Eigen::Vector3d t(initialPose.x(), initialPose.y(), initialPose.z());

  // Pre-convert points to double Eigen.
  std::vector<Eigen::Vector3d> Xw(N);
  std::vector<Eigen::Vector2d> z(N);
  for (size_t i = 0; i < N; ++i)
  {
    Xw[i] = Eigen::Vector3d(worldPts[i].x, worldPts[i].y, worldPts[i].z);
    z[i]  = Eigen::Vector2d(pixels[i].x, pixels[i].y);
  }

  const double chi2        = static_cast<double>(params.chi2_threshold);
  const double huber_delta = static_cast<double>(params.huber_delta);
  double       lambda      = static_cast<double>(params.lambda_initial);

  // Robust Gauss-Newton with LM damping. During optimization ALL points are
  // used with Huber down-weighting (gross outliers get a small but non-zero
  // weight). Hard inlier/outlier classification by the chi-square gate is done
  // only AFTER convergence: gating from a poor initial pose would discard every
  // correspondence and stall the solver.
  for (int iter = 0; iter < params.max_iters; ++iter)
  {
    Eigen::Matrix<double, 6, 6> H      = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g      = Eigen::Matrix<double, 6, 1>::Zero();
    int                         n_used = 0;

    for (size_t i = 0; i < N; ++i)
    {
      const Eigen::Vector3d Xc = R * Xw[i] + t;
      if (Xc.z() <= 1e-6)
      {
        continue;  // behind the camera: skip this iteration
      }
      ++n_used;

      const double inv_z = 1.0 / Xc.z();
      const double u     = fx * Xc.x() * inv_z + cx;
      const double v     = fy * Xc.y() * inv_z + cy;

      const Eigen::Vector2d e(u - z[i].x(), v - z[i].y());  // predicted - observed

      // d(proj)/d(Xc)  (2x3)
      Eigen::Matrix<double, 2, 3> Jproj;
      // clang-format off
      Jproj << fx * inv_z,          0, -fx * Xc.x() * inv_z * inv_z,
                        0, fy * inv_z, -fy * Xc.y() * inv_z * inv_z;
      // clang-format on

      // d(Xc)/d(delta), delta = [d_omega (body-frame rot); d_t (world transl)]
      //   R <- R * Exp(d_omega):  d(Xc)/d_omega = -R [Xw]_x
      //   t <- t + d_t:           d(Xc)/d_t     = I
      Eigen::Matrix<double, 3, 6> Jx;
      Jx.leftCols<3>()  = -R * skew(Xw[i]);
      Jx.rightCols<3>() = Eigen::Matrix3d::Identity();

      const Eigen::Matrix<double, 2, 6> J = Jproj * Jx;

      // Huber reweighting on the residual magnitude.
      const double w = static_cast<double>(
          huberWeight(static_cast<float>(e.norm()), static_cast<float>(huber_delta)));

      H += w * J.transpose() * J;
      g += w * J.transpose() * e;
    }

    if (n_used < 4)
    {
      break;  // too few visible points
    }

    // LM damping on the diagonal.
    Eigen::Matrix<double, 6, 6> H_lm = H;
    for (int d = 0; d < 6; ++d)
    {
      H_lm(d, d) += lambda * H_lm(d, d);
    }

    const Eigen::Matrix<double, 6, 1> delta = H_lm.ldlt().solve(-g);

    // Retract: rotation right-perturbation, translation additive.
    R = R * so3Exp(delta.head<3>());
    t = t + delta.tail<3>();

    if (delta.norm() < params.eps_step)
    {
      result.converged = true;
      break;
    }
    // Mild LM annealing: shrink damping as we proceed.
    lambda = std::max(lambda * 0.5, 1e-8);
  }

  // Final inlier classification + information matrix (inliers only).
  Eigen::Matrix<double, 6, 6> H_inliers   = Eigen::Matrix<double, 6, 6>::Zero();
  double                      inlier_cost = 0.0;
  int                         n_in        = 0;
  for (size_t i = 0; i < N; ++i)
  {
    const Eigen::Vector3d Xc = R * Xw[i] + t;
    if (Xc.z() <= 1e-6)
    {
      result.inliers[i] = false;
      continue;
    }
    const double          inv_z = 1.0 / Xc.z();
    const double          u     = fx * Xc.x() * inv_z + cx;
    const double          v     = fy * Xc.y() * inv_z + cy;
    const Eigen::Vector2d e(u - z[i].x(), v - z[i].y());
    const double          e2 = e.squaredNorm();

    const bool is_inlier = e2 <= chi2;
    result.inliers[i]    = is_inlier;
    if (!is_inlier)
    {
      continue;
    }
    ++n_in;
    inlier_cost += e2;

    Eigen::Matrix<double, 2, 3> Jproj;
    // clang-format off
    Jproj << fx * inv_z,          0, -fx * Xc.x() * inv_z * inv_z,
                      0, fy * inv_z, -fy * Xc.y() * inv_z * inv_z;
    // clang-format on
    Eigen::Matrix<double, 3, 6> Jx;
    Jx.leftCols<3>()                    = -R * skew(Xw[i]);
    Jx.rightCols<3>()                   = Eigen::Matrix3d::Identity();
    const Eigen::Matrix<double, 2, 6> J = Jproj * Jx;
    H_inliers += J.transpose() * J;
  }
  result.num_inliers = n_in;
  result.final_cost  = static_cast<float>(0.5 * inlier_cost);

  // Re-orthonormalize R to counter accumulated drift, then pack into CPose3D.
  {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0)
    {
      Eigen::Matrix3d Vt = svd.matrixV().transpose();
      Vt.row(2) *= -1;
      R = svd.matrixU() * Vt;
    }
  }

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
  result.pose = mrpt::poses::CPose3D(M);

  // Pose covariance ~ inverse Gauss-Newton information (inliers only).
  if (n_in >= 6)
  {
    const Eigen::Matrix<double, 6, 6> Hinv =
        H_inliers.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());
    if (Hinv.allFinite())
    {
      result.covariance = Hinv;
    }
  }

  return result;
}
