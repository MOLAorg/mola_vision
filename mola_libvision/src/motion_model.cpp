/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/lie_utils.h>
#include <mola_libvision/motion_model.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

using namespace mola::vision;

namespace
{
using Vec6 = mrpt::math::CVectorFixedDouble<6>;

// Exact SE(3) exponential and logarithm, tangent [rho, omega]: a constant body
// twist (e.g. driving along an arc) maps to exp(t * twist). MRPT's
// Lie::SE<3> uses a pseudo-exponential (translation not coupled with the
// rotation), which does not model constant-velocity motion while turning.
// The left Jacobian of SO(3) is Jl(w) = Jr(-w).
Eigen::Matrix3d leftJacobianSO3(const Eigen::Vector3d& w) { return rightJacobianSO3(-w); }

mrpt::poses::CPose3D se3Exp(const Vec6& x)
{
  const Eigen::Vector3d             rho(x[0], x[1], x[2]);
  const Eigen::Vector3d             w(x[3], x[4], x[5]);
  const Eigen::Vector3d             t = leftJacobianSO3(w) * rho;
  mrpt::math::CMatrixDouble33       R(so3Exp(w));
  mrpt::math::CVectorFixedDouble<3> tt;
  tt[0] = t.x();
  tt[1] = t.y();
  tt[2] = t.z();
  return {R, tt};
}

Vec6 se3Log(const mrpt::poses::CPose3D& P)
{
  const Eigen::Matrix3d   R = P.getRotationMatrix().asEigen();
  const Eigen::AngleAxisd aa(R);
  const Eigen::Vector3d   w = aa.angle() * aa.axis();
  const Eigen::Vector3d   t(P.x(), P.y(), P.z());
  const Eigen::Vector3d   rho = leftJacobianSO3(w).inverse() * t;
  Vec6                    x;
  for (int i = 0; i < 3; i++)
  {
    x[i]     = rho[i];
    x[i + 3] = w[i];
  }
  return x;
}
}  // namespace

mrpt::poses::CPose3D mola::vision::scalePoseIncrement(const mrpt::poses::CPose3D& inc, double s)
{
  auto x = se3Log(inc);
  for (int i = 0; i < 6; i++)
  {
    x[i] *= s;
  }
  return se3Exp(x);
}

void ConstantVelocityModel::addPose(double t, const mrpt::poses::CPose3D& pose)
{
  if (!history_.empty() && t <= history_.back().first)
  {
    return;
  }
  history_.emplace_back(t, pose);
  // Keep the newest pose older than the window too: it is the velocity
  // reference whenever no other pose falls inside the window.
  while (history_.size() > 2 && t - history_[1].first >= params_.velocity_window)
  {
    history_.pop_front();
  }
}

mrpt::math::CVectorFixedDouble<6> ConstantVelocityModel::velocity() const
{
  mrpt::math::CVectorFixedDouble<6> v;
  v.setZero();
  if (!hasVelocity())
  {
    return v;
  }
  const auto& [t0, p0] = history_.front();
  const auto& [t1, p1] = history_.back();
  v                    = se3Log(p1 - p0);
  const double dt      = t1 - t0;
  for (int i = 0; i < 6; i++)
  {
    v[i] /= dt;
  }
  return v;
}

mrpt::poses::CPose3D ConstantVelocityModel::predict(double t) const
{
  if (history_.empty())
  {
    return mrpt::poses::CPose3D::Identity();
  }
  const auto& [tl, pl] = history_.back();
  if (!hasVelocity())
  {
    return pl;
  }
  const double dt = std::clamp(t - tl, 0.0, params_.max_extrapolation);
  auto         x  = velocity();
  for (int i = 0; i < 6; i++)
  {
    x[i] *= dt;
  }
  return pl + se3Exp(x);
}
