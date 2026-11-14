/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/math/CVectorFixed.h>
#include <mrpt/poses/CPose3D.h>

#include <deque>
#include <utility>

namespace mola::vision
{
/** Scales a pose increment in time on SE(3): Exp(s * Log(inc)). E.g. the motion
 *  over one frame period scaled to the actual time between two frames. */
[[nodiscard]] mrpt::poses::CPose3D scalePoseIncrement(const mrpt::poses::CPose3D& inc, double s);

/** Constant-velocity motion model driven by timestamps.
 *
 *  The velocity is a constant SE(3) twist per SECOND, estimated from the poses
 *  added over the last `velocity_window` seconds (longer than one frame
 *  period, so frame-to-frame jitter is not amplified), and predictions
 *  extrapolate it by the real time elapsed since the last pose. So a gap of
 *  missing frames is bridged by the motion expected over that time, not by
 *  the motion of a single frame. */
class ConstantVelocityModel
{
 public:
  struct Parameters
  {
    /** Past poses within this span [s] estimate the velocity. */
    double velocity_window = 0.3;
    /** Predictions never extrapolate further than this [s] from the last pose. */
    double max_extrapolation = 1.5;
  };

  ConstantVelocityModel() = default;
  /** Throws if a parameter is negative. */
  explicit ConstantVelocityModel(const Parameters& p) { setParameters(p); }

  /** Throws if a parameter is negative. */
  void                            setParameters(const Parameters& p);
  [[nodiscard]] const Parameters& parameters() const { return params_; }

  void reset() { history_.clear(); }

  /** Adds an estimated pose (e.g. camera in world) at time \p t [s]. Poses
   *  must come in increasing time order; older ones are ignored. */
  void addPose(double t, const mrpt::poses::CPose3D& pose);

  /** True once there are two poses to estimate a velocity from. */
  [[nodiscard]] bool hasVelocity() const { return history_.size() >= 2; }

  /** Velocity as a twist per second [rho, omega] (the exact SE(3) tangent:
   *  linear part first, then angular), in the local frame of the poses.
   *  Zero if unknown. */
  [[nodiscard]] mrpt::math::CVectorFixedDouble<6> velocity() const;

  /** Pose predicted at time \p t: the last pose moved at the constant velocity
   *  for (t - t_last), clamped to [0, max_extrapolation]. The last pose if
   *  there is no velocity yet; identity if there are no poses at all. */
  [[nodiscard]] mrpt::poses::CPose3D predict(double t) const;

  /** Time of the last added pose [s], or 0 if none. */
  [[nodiscard]] double lastTime() const { return history_.empty() ? 0.0 : history_.back().first; }

 private:
  Parameters                                          params_;
  std::deque<std::pair<double, mrpt::poses::CPose3D>> history_;
};

}  // namespace mola::vision
