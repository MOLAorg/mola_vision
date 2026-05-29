/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#pragma once

#include <Eigen/Core>
#include <cmath>

namespace mola::vision
{
/** Skew-symmetric matrix [v]_x such that [v]_x * w = v x w (cross product). */
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
  Eigen::Matrix3d S;
  // clang-format off
  S <<      0, -v.z(),  v.y(),
        v.z(),      0, -v.x(),
       -v.y(),  v.x(),      0;
  // clang-format on
  return S;
}

/** SO(3) exponential map (Rodrigues) of a rotation vector w (axis*angle). */
inline Eigen::Matrix3d so3Exp(const Eigen::Vector3d& w)
{
  const double theta = w.norm();
  if (theta < 1e-10)
  {
    // First-order approximation near identity.
    return Eigen::Matrix3d::Identity() + skew(w);
  }
  const Eigen::Vector3d axis = w / theta;
  const Eigen::Matrix3d K    = skew(axis);
  return Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
}

/** Right Jacobian of SO(3) at rotation vector w:
 *    Jr = I - (1-cos θ)/θ² [w]_x + (θ-sin θ)/θ³ [w]_x²,   θ = ‖w‖.
 *  Satisfies Exp(w + δ) ≈ Exp(w) · Exp(Jr·δ). */
inline Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d& w)
{
  const double          theta2 = w.squaredNorm();
  const Eigen::Matrix3d W      = skew(w);
  if (theta2 < 1e-12)
  {
    return Eigen::Matrix3d::Identity() - 0.5 * W;
  }
  const double theta = std::sqrt(theta2);
  const double a     = (1.0 - std::cos(theta)) / theta2;
  const double b     = (theta - std::sin(theta)) / (theta2 * theta);
  return Eigen::Matrix3d::Identity() - a * W + b * W * W;
}

}  // namespace mola::vision
