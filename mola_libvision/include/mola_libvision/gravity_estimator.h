/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#pragma once

#include <Eigen/Core>
#include <vector>

namespace mola::vision
{
/** Result of a gravity estimation. */
struct GravityEstimate
{
  /** Gravity vector in the input (sensor/body) frame, with magnitude ~ g_mag. */
  Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
  bool            ok      = false;
};

/** Estimate the gravity vector from (near-)stationary accelerometer samples.
 *
 *  A stationary accelerometer measures specific force f = a - g = -g (since the
 *  true acceleration a ~ 0), so gravity ~ -mean(accel), rescaled to the known
 *  magnitude. This is the standard bootstrap used to gravity-align a VIO at
 *  start-up while the platform is still.
 *
 *  \param accel_samples Raw accelerometer readings (m/s^2) in the sensor frame.
 *  \param g_mag         Expected gravity magnitude (m/s^2).
 *  \return GravityEstimate with ok=false if there are no samples.
 *
 *  \note Adapted in spirit from lightweight_vio (MIT, (c) 2025 Seungwon Choi).
 */
[[nodiscard]] GravityEstimate estimateGravityStatic(
    const std::vector<Eigen::Vector3d>& accel_samples, double g_mag = 9.81);

/** Rotation R that aligns the gravity direction to a target "down" axis:
 *  R * normalize(gravity) == normalize(down_dir).
 *
 *  Used to rotate an arbitrarily-oriented start frame into a gravity-aligned
 *  world frame (e.g. with gravity along -Z). Handles the parallel and
 *  anti-parallel degenerate cases.
 */
[[nodiscard]] Eigen::Matrix3d gravityAlignmentRotation(
    const Eigen::Vector3d& gravity, const Eigen::Vector3d& down_dir = {0.0, 0.0, -1.0});

}  // namespace mola::vision
