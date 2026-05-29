/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <Eigen/Core>

namespace mola::vision
{
/** On-manifold IMU preintegration between two keyframes (Forster et al.,
 *  "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry", T-RO
 *  2017). Accumulates the relative rotation/velocity/position increments
 *  ΔR, ΔV, ΔP from raw gyro/accel measurements, together with the first-order
 *  Jacobians w.r.t. the gyro/accel biases used for the linearization, so the
 *  preintegrated quantities can be cheaply bias-corrected without re-integrating.
 *
 *  Conventions: ΔR is body-to-body (frame i to frame j), gravity is added at
 *  predict() time (raw accelerometer specific force is integrated, not
 *  gravity-compensated). Math is in double precision.
 *
 *  \note Adapted from lightweight_vio (MIT, (c) 2025 Seungwon Choi) IMUHandler.
 */
class ImuPreintegrator
{
 public:
  ImuPreintegrator() = default;
  ImuPreintegrator(const Eigen::Vector3d& bias_gyro, const Eigen::Vector3d& bias_accel)
      : bias_gyro_(bias_gyro), bias_accel_(bias_accel)
  {
  }

  /** Clear all increments / Jacobians (keeps the biases). */
  void reset();

  /** Set the linearization biases and reset the integration. */
  void setBias(const Eigen::Vector3d& bias_gyro, const Eigen::Vector3d& bias_accel);

  /** Integrate one IMU sample over interval dt (seconds).
   *  \param gyro  Angular velocity (rad/s), raw (bias not yet removed).
   *  \param accel Linear acceleration / specific force (m/s^2), raw. */
  void integrate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, double dt);

  // --- Preintegrated increments ---
  [[nodiscard]] const Eigen::Matrix3d& deltaR() const { return dR_; }
  [[nodiscard]] const Eigen::Vector3d& deltaV() const { return dV_; }
  [[nodiscard]] const Eigen::Vector3d& deltaP() const { return dP_; }
  [[nodiscard]] double                 dt() const { return dt_total_; }

  [[nodiscard]] const Eigen::Vector3d& biasGyro() const { return bias_gyro_; }
  [[nodiscard]] const Eigen::Vector3d& biasAccel() const { return bias_accel_; }

  // --- First-order bias-update Jacobians ---
  [[nodiscard]] const Eigen::Matrix3d& dR_dbg() const { return J_R_bg_; }
  [[nodiscard]] const Eigen::Matrix3d& dV_dbg() const { return J_V_bg_; }
  [[nodiscard]] const Eigen::Matrix3d& dV_dba() const { return J_V_ba_; }
  [[nodiscard]] const Eigen::Matrix3d& dP_dbg() const { return J_P_bg_; }
  [[nodiscard]] const Eigen::Matrix3d& dP_dba() const { return J_P_ba_; }

  /** Predict the state at frame j from the state at frame i and gravity:
   *    R_j = R_i * ΔR
   *    v_j = v_i + g*Δt + R_i*ΔV
   *    p_j = p_i + v_i*Δt + 0.5*g*Δt^2 + R_i*ΔP
   */
  void predict(
      const Eigen::Matrix3d& Ri, const Eigen::Vector3d& vi, const Eigen::Vector3d& pi,
      const Eigen::Vector3d& gravity, Eigen::Matrix3d& Rj, Eigen::Vector3d& vj,
      Eigen::Vector3d& pj) const;

 private:
  Eigen::Vector3d bias_gyro_  = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_accel_ = Eigen::Vector3d::Zero();

  Eigen::Matrix3d dR_       = Eigen::Matrix3d::Identity();
  Eigen::Vector3d dV_       = Eigen::Vector3d::Zero();
  Eigen::Vector3d dP_       = Eigen::Vector3d::Zero();
  double          dt_total_ = 0.0;

  Eigen::Matrix3d J_R_bg_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_V_bg_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_V_ba_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_P_bg_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_P_ba_ = Eigen::Matrix3d::Zero();
};

}  // namespace mola::vision
