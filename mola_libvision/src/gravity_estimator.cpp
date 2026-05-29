/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#include <mola_libvision/gravity_estimator.h>
#include <mola_libvision/lie_utils.h>  // so3Exp(), skew()

#include <Eigen/Geometry>  // cross(), unitOrthogonal()
#include <cmath>

using namespace mola::vision;

GravityEstimate mola::vision::estimateGravityStatic(
    const std::vector<Eigen::Vector3d>& accel_samples, double g_mag)
{
  GravityEstimate est;
  if (accel_samples.empty())
  {
    return est;
  }
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto& a : accel_samples)
  {
    mean += a;
  }
  mean /= static_cast<double>(accel_samples.size());

  // Stationary specific force f = -g  =>  g = -mean(accel), rescaled to g_mag.
  const double norm = mean.norm();
  if (norm < 1e-6)
  {
    return est;  // no dominant direction (free-fall or no signal)
  }
  est.gravity = -mean / norm * g_mag;
  est.ok      = true;
  return est;
}

Eigen::Matrix3d mola::vision::gravityAlignmentRotation(
    const Eigen::Vector3d& gravity, const Eigen::Vector3d& down_dir)
{
  const double gn = gravity.norm();
  const double dn = down_dir.norm();
  if (gn < 1e-9 || dn < 1e-9)
  {
    return Eigen::Matrix3d::Identity();
  }
  const Eigen::Vector3d u = gravity / gn;  // current direction
  const Eigen::Vector3d v = down_dir / dn;  // target direction

  const double          c    = u.dot(v);
  const Eigen::Vector3d axis = u.cross(v);
  const double          s    = axis.norm();

  if (s < 1e-9)
  {
    // Parallel or anti-parallel.
    if (c > 0.0)
    {
      return Eigen::Matrix3d::Identity();  // already aligned
    }
    // 180 deg: rotate about any axis orthogonal to u.
    Eigen::Vector3d ortho = u.unitOrthogonal();
    return so3Exp(ortho * M_PI);
  }

  const double angle = std::atan2(s, c);
  return so3Exp(axis / s * angle);
}
