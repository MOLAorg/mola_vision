/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * On-manifold IMU preintegration (Forster et al., T-RO 2017). Adapted from
 * lightweight_vio (MIT, (c) 2025 Seungwon Choi) IMUHandler.
 * ------------------------------------------------------------------------- */
#include <mola_libvision/imu_preintegration.h>
#include <mola_libvision/lie_utils.h>  // skew(), so3Exp(), rightJacobianSO3()

using namespace mola::vision;

void ImuPreintegrator::reset()
{
  dR_       = Eigen::Matrix3d::Identity();
  dV_       = Eigen::Vector3d::Zero();
  dP_       = Eigen::Vector3d::Zero();
  dt_total_ = 0.0;
  J_R_bg_   = Eigen::Matrix3d::Zero();
  J_V_bg_   = Eigen::Matrix3d::Zero();
  J_V_ba_   = Eigen::Matrix3d::Zero();
  J_P_bg_   = Eigen::Matrix3d::Zero();
  J_P_ba_   = Eigen::Matrix3d::Zero();
}

void ImuPreintegrator::setBias(const Eigen::Vector3d& bias_gyro, const Eigen::Vector3d& bias_accel)
{
  bias_gyro_  = bias_gyro;
  bias_accel_ = bias_accel;
  reset();
}

void ImuPreintegrator::integrate(
    const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, double dt)
{
  if (dt <= 0.0)
  {
    return;
  }

  // Bias-corrected measurements.
  const Eigen::Vector3d w = gyro - bias_gyro_;
  const Eigen::Vector3d a = accel - bias_accel_;

  const Eigen::Matrix3d dR  = so3Exp(w * dt);
  const Eigen::Matrix3d Jr  = rightJacobianSO3(w * dt);
  const Eigen::Matrix3d aSk = skew(a);

  // The position/velocity updates and their bias Jacobians must use the values
  // *before* this step's rotation update (Forster eqs. A.7-A.13). Update the
  // position-block Jacobians before the velocity-block ones (they depend on the
  // old velocity-block Jacobians), and update the rotation Jacobian last.
  const double dt2 = dt * dt;

  J_P_ba_ += J_V_ba_ * dt - 0.5 * dR_ * dt2;
  J_P_bg_ += J_V_bg_ * dt - 0.5 * dR_ * aSk * J_R_bg_ * dt2;
  J_V_ba_ += -dR_ * dt;
  J_V_bg_ += -dR_ * aSk * J_R_bg_ * dt;

  // Position and velocity increments (use the old ΔR, ΔV).
  dP_ += dV_ * dt + 0.5 * dR_ * a * dt2;
  dV_ += dR_ * a * dt;

  // Rotation Jacobian and increment (last).
  J_R_bg_ = dR.transpose() * J_R_bg_ - Jr * dt;
  dR_     = dR_ * dR;

  dt_total_ += dt;
}

void ImuPreintegrator::predict(
    const Eigen::Matrix3d& Ri, const Eigen::Vector3d& vi, const Eigen::Vector3d& pi,
    const Eigen::Vector3d& gravity, Eigen::Matrix3d& Rj, Eigen::Vector3d& vj,
    Eigen::Vector3d& pj) const
{
  const double t = dt_total_;
  Rj             = Ri * dR_;
  vj             = vi + gravity * t + Ri * dV_;
  pj             = pi + vi * t + 0.5 * gravity * t * t + Ri * dP_;
}
