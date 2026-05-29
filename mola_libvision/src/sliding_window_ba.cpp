/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Schur-complement structure inspired by Basalt (BSD-3-Clause,
 * (c) 2019 Usenko, Demmel), linearization_abs_sc. Reprojection residual and
 * Jacobians are the standard pinhole ones, shared with PnPSolver.
 * ------------------------------------------------------------------------- */
#include <mola_libvision/lie_utils.h>  // skew(), so3Exp()
#include <mola_libvision/lm_solver.h>  // huberWeight()
#include <mola_libvision/sliding_window_ba.h>

#include <Eigen/Dense>
#include <cmath>
#include <map>

using namespace mola::vision;

namespace
{
using Mat6  = Eigen::Matrix<double, 6, 6>;
using Vec6  = Eigen::Matrix<double, 6, 1>;
using Mat63 = Eigen::Matrix<double, 6, 3>;
using Mat23 = Eigen::Matrix<double, 2, 3>;
using Mat26 = Eigen::Matrix<double, 2, 6>;

// Eigen fixed-size matrices >= 16 bytes are over-aligned; storing them in STL
// containers requires Eigen::aligned_allocator to avoid undefined behavior
// (heap corruption). Matrix3d / Vector3d are not over-aligned, so plain
// std::vector is fine for those.
template <typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;
using HplMap =
    std::map<int, Mat63, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Mat63>>>;

struct Intr
{
  double fx, fy, cx, cy;
};

/** Reprojection error e = predicted - observed for landmark Xw seen from
 *  pose (R,t) (T_cw). Returns false if the point is behind the camera. */
bool reproj(
    const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::Vector3d& Xw, const Intr& K,
    const Eigen::Vector2d& obs, Eigen::Vector2d& e, Eigen::Vector3d& Xc)
{
  Xc = R * Xw + t;
  if (Xc.z() <= 1e-6)
  {
    return false;
  }
  const double inv_z = 1.0 / Xc.z();
  e.x()              = K.fx * Xc.x() * inv_z + K.cx - obs.x();
  e.y()              = K.fy * Xc.y() * inv_z + K.cy - obs.y();
  return true;
}

/** Robust (Huber) cost summed over all observations at the current state. */
double robustCost(
    const std::vector<Eigen::Matrix3d>& Rs, const std::vector<Eigen::Vector3d>& ts,
    const std::vector<Eigen::Vector3d>& lms, const std::vector<BAObservation>& obs, const Intr& K,
    double delta)
{
  double cost = 0.0;
  for (const auto& o : obs)
  {
    Eigen::Vector2d e, z(o.pixel.x, o.pixel.y);
    Eigen::Vector3d Xc;
    if (!reproj(Rs[o.kf_index], ts[o.kf_index], lms[o.lm_index], K, z, e, Xc))
    {
      continue;
    }
    const double r = e.norm();
    cost += (r <= delta) ? 0.5 * r * r : delta * (r - 0.5 * delta);
  }
  return cost;
}

}  // namespace

BAResult mola::vision::slidingWindowBA(
    std::vector<mrpt::poses::CPose3D>& poses, std::vector<mrpt::math::TPoint3Df>& landmarks,
    const std::vector<BAObservation>& obs, const mrpt::img::TCamera& cam,
    const std::vector<bool>& fixedPoses, const BAOptions& options)
{
  BAResult  result;
  const int nPoses = static_cast<int>(poses.size());
  const int nLm    = static_cast<int>(landmarks.size());
  if (nPoses == 0 || nLm == 0 || obs.empty())
  {
    return result;
  }

  const Intr   K{cam.fx(), cam.fy(), cam.cx(), cam.cy()};
  const double delta = static_cast<double>(options.huber_delta);

  // Gauge: which poses are held fixed.
  std::vector<bool> fixed(nPoses, false);
  if (!fixedPoses.empty())
  {
    for (int i = 0; i < nPoses && i < static_cast<int>(fixedPoses.size()); ++i)
    {
      fixed[i] = fixedPoses[i];
    }
  }
  else if (options.fix_first_pose)
  {
    fixed[0] = true;
  }

  // Map each free pose to a column index in the reduced system.
  std::vector<int> poseCol(nPoses, -1);
  int              nFree = 0;
  for (int i = 0; i < nPoses; ++i)
  {
    if (!fixed[i])
    {
      poseCol[i] = nFree++;
    }
  }
  const int dimP = 6 * nFree;

  // Working state in Eigen.
  std::vector<Eigen::Matrix3d> Rs(nPoses);
  std::vector<Eigen::Vector3d> ts(nPoses);
  for (int i = 0; i < nPoses; ++i)
  {
    Rs[i] = poses[i].getRotationMatrix().asEigen();
    ts[i] = Eigen::Vector3d(poses[i].x(), poses[i].y(), poses[i].z());
  }
  std::vector<Eigen::Vector3d> lms(nLm);
  for (int i = 0; i < nLm; ++i)
  {
    lms[i] = Eigen::Vector3d(landmarks[i].x, landmarks[i].y, landmarks[i].z);
  }

  double lambda       = options.lambda_initial;
  double nu           = 2.0;  // Nielsen damping-increase factor
  result.initial_cost = robustCost(Rs, ts, lms, obs, K, delta);
  double cost         = result.initial_cost;

  for (int iter = 0; iter < options.max_iters; ++iter)
  {
    result.iterations = iter + 1;

    // --- Accumulate the sparse normal equations. ---
    AlignedVector<Mat6>          Hpp(nFree, Mat6::Zero());
    AlignedVector<Vec6>          bp(nFree, Vec6::Zero());
    std::vector<Eigen::Matrix3d> Hll(nLm, Eigen::Matrix3d::Zero());
    std::vector<Eigen::Vector3d> bl(nLm, Eigen::Vector3d::Zero());
    // Per-landmark off-diagonal blocks, keyed by free-pose column.
    std::vector<HplMap> Hpl(nLm);
    int                 n_used = 0;

    for (const auto& o : obs)
    {
      const int       p = o.kf_index;
      const int       l = o.lm_index;
      Eigen::Vector2d e, z(o.pixel.x, o.pixel.y);
      Eigen::Vector3d Xc;
      if (!reproj(Rs[p], ts[p], lms[l], K, z, e, Xc))
      {
        continue;
      }
      ++n_used;

      const double inv_z = 1.0 / Xc.z();
      Mat23        Jproj;
      // clang-format off
      Jproj << K.fx * inv_z,           0, -K.fx * Xc.x() * inv_z * inv_z,
                         0, K.fy * inv_z, -K.fy * Xc.y() * inv_z * inv_z;
      // clang-format on

      // Jacobian wrt landmark: d(Xc)/d(Xw) = R.
      const Mat23 Jl = Jproj * Rs[p];

      // Huber IRLS weight.
      const double w =
          static_cast<double>(huberWeight(static_cast<float>(e.norm()), static_cast<float>(delta)));

      Hll[l] += w * Jl.transpose() * Jl;
      bl[l] += w * Jl.transpose() * e;

      if (poseCol[p] >= 0)
      {
        const int c = poseCol[p];
        // Jacobian wrt pose: d(Xc)/d[dw;dt] = [-R [Xw]_x | I].
        Mat26 Jp;
        Jp.leftCols<3>()  = Jproj * (-Rs[p] * skew(lms[l]));
        Jp.rightCols<3>() = Jproj;  // = Jproj * I
        Hpp[c] += w * Jp.transpose() * Jp;
        bp[c] += w * Jp.transpose() * e;
        Hpl[l][c] += w * Jp.transpose() * Jl;
      }
    }
    result.num_observations_used = n_used;

    // --- Schur complement: eliminate landmarks, build reduced pose system. ---
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(dimP, dimP);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(dimP);
    for (int c = 0; c < nFree; ++c)
    {
      Mat6 Hd = Hpp[c];
      for (int d = 0; d < 6; ++d)
      {
        // Multiplicative (Marquardt) damping D = diag(JᵀJ): scale-aware, so the
        // huge-curvature rotation block and the weak-curvature depth block are
        // damped proportionally. This matches the predicted-reduction model
        // exactly (D = diag), which keeps the gain ratio well-calibrated.
        Hd(d, d) += lambda * Hd(d, d);
      }
      S.block<6, 6>(6 * c, 6 * c) = Hd;
      g.segment<6>(6 * c)         = bp[c];
    }

    std::vector<Eigen::Matrix3d> HllInv(nLm, Eigen::Matrix3d::Zero());
    for (int l = 0; l < nLm; ++l)
    {
      if (Hpl[l].empty() && Hll[l].isZero())
      {
        continue;  // unobserved landmark
      }
      Eigen::Matrix3d Hd = Hll[l];
      for (int d = 0; d < 3; ++d)
      {
        Hd(d, d) += lambda * Hd(d, d);
      }
      // Tiny ridge to keep the 3x3 invertible for weakly-constrained landmarks.
      Hd += 1e-9 * Eigen::Matrix3d::Identity();
      HllInv[l] = Hd.inverse();

      for (const auto& [cp, Wp] : Hpl[l])
      {
        const Eigen::Matrix<double, 6, 3> WpHinv = Wp * HllInv[l];
        g.segment<6>(6 * cp) -= WpHinv * bl[l];
        for (const auto& [cq, Wq] : Hpl[l])
        {
          S.block<6, 6>(6 * cp, 6 * cq) -= WpHinv * Wq.transpose();
        }
      }
    }

    // --- Solve reduced pose system, back-substitute landmarks. ---
    Eigen::VectorXd dxp = Eigen::VectorXd::Zero(dimP);
    if (dimP > 0)
    {
      dxp = S.ldlt().solve(-g);
      if (!dxp.allFinite())
      {
        lambda = std::min(lambda * 4.0, 1e8);
        continue;
      }
    }

    std::vector<Eigen::Vector3d> dxl(nLm, Eigen::Vector3d::Zero());
    for (int l = 0; l < nLm; ++l)
    {
      if (HllInv[l].isZero())
      {
        continue;
      }
      Eigen::Vector3d acc = bl[l];
      for (const auto& [cp, Wp] : Hpl[l])
      {
        acc += Wp.transpose() * dxp.segment<6>(6 * cp);
      }
      dxl[l] = -HllInv[l] * acc;
    }

    // --- Tentative update on copies, then LM accept/reject. ---
    std::vector<Eigen::Matrix3d> Rs_new  = Rs;
    std::vector<Eigen::Vector3d> ts_new  = ts;
    std::vector<Eigen::Vector3d> lms_new = lms;
    for (int i = 0; i < nPoses; ++i)
    {
      if (poseCol[i] < 0)
      {
        continue;
      }
      const Vec6 d = dxp.segment<6>(6 * poseCol[i]);
      Rs_new[i]    = Rs[i] * so3Exp(d.head<3>());
      ts_new[i]    = ts[i] + d.tail<3>();
    }
    for (int l = 0; l < nLm; ++l)
    {
      lms_new[l] += dxl[l];
    }

    const double cost_new = robustCost(Rs_new, ts_new, lms_new, obs, K, delta);

    // Total state increment.
    double step2 = dxp.squaredNorm();
    for (const auto& d : dxl)
    {
      step2 += d.squaredNorm();
    }
    const double step = std::sqrt(step2);

    // Predicted reduction of the damped quadratic model (Nielsen):
    //   pred = 0.5 * ( lambda * dx^T D dx  -  g^T dx ),  D = diag(JᵀJ).
    double gdx = 0.0;  // g^T dx
    double dDd = 0.0;  // dx^T D dx
    for (int c = 0; c < nFree; ++c)
    {
      const Vec6 d = dxp.segment<6>(6 * c);
      gdx += bp[c].dot(d);
      for (int k = 0; k < 6; ++k)
      {
        dDd += Hpp[c](k, k) * d(k) * d(k);
      }
    }
    for (int l = 0; l < nLm; ++l)
    {
      gdx += bl[l].dot(dxl[l]);
      for (int k = 0; k < 3; ++k)
      {
        dDd += Hll[l](k, k) * dxl[l](k) * dxl[l](k);
      }
    }
    const double predicted = 0.5 * (lambda * dDd - gdx);
    const double rho       = (predicted > 0.0) ? (cost - cost_new) / predicted : -1.0;

    if (rho > 0.0 && cost_new < cost)
    {
      Rs   = Rs_new;
      ts   = ts_new;
      lms  = lms_new;
      cost = cost_new;
      // Nielsen damping decrease, with a floor: never let damping vanish, or a
      // weakly-observed landmark-depth direction can produce an exploding step
      // on the next iteration.
      const double f = 2.0 * rho - 1.0;
      lambda *= std::max(1.0 / 3.0, 1.0 - f * f * f);
      lambda = std::max(lambda, 1e-3);
      nu     = 2.0;
      if (step < options.eps_step)
      {
        result.converged = true;
        break;
      }
    }
    else
    {
      // Reject: keep current state, increase damping geometrically.
      lambda = std::min(lambda * nu, 1e12);
      nu *= 2.0;
    }
  }

  // --- Write results back, re-orthonormalizing rotations. ---
  for (int i = 0; i < nPoses; ++i)
  {
    Eigen::Matrix3d                   R = Rs[i];
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0)
    {
      Eigen::Matrix3d Vt = svd.matrixV().transpose();
      Vt.row(2) *= -1;
      R = svd.matrixU() * Vt;
    }
    mrpt::math::CMatrixDouble44 M;
    M.setIdentity();
    for (int r = 0; r < 3; ++r)
    {
      for (int c = 0; c < 3; ++c)
      {
        M(r, c) = R(r, c);
      }
      M(r, 3) = ts[i](r);
    }
    poses[i] = mrpt::poses::CPose3D(M);
  }
  for (int l = 0; l < nLm; ++l)
  {
    landmarks[l] = {
        static_cast<float>(lms[l].x()), static_cast<float>(lms[l].y()),
        static_cast<float>(lms[l].z())};
  }

  result.final_cost = cost;
  return result;
}
