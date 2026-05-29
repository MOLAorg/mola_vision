/* -------------------------------------------------------------------------
 * mola_libvision unit tests: IMU preintegration
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/imu_preintegration.h>
#include <mola_libvision/lie_utils.h>

using namespace mola::vision;

// ---------------------------------------------------------------------------
// Constant angular velocity, zero acceleration -> ΔR = Exp(w*T), ΔV=ΔP=0.
// ---------------------------------------------------------------------------
TEST(ImuPreintegration, ConstantAngularVelocity)
{
  const Eigen::Vector3d w(0.3, -0.2, 0.5);  // rad/s
  const double          dt = 1e-3;
  const int             N  = 1000;  // 1 second total

  ImuPreintegrator pre;
  for (int i = 0; i < N; ++i)
  {
    pre.integrate(w, Eigen::Vector3d::Zero(), dt);
  }

  const Eigen::Matrix3d R_expected = so3Exp(w * (N * dt));
  EXPECT_LT((pre.deltaR() - R_expected).norm(), 1e-6);
  EXPECT_LT(pre.deltaV().norm(), 1e-9);
  EXPECT_LT(pre.deltaP().norm(), 1e-9);
  EXPECT_NEAR(pre.dt(), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Zero gyro, constant accel -> ΔR=I, ΔV=a*T, ΔP=0.5*a*T^2 (exact for const a).
// ---------------------------------------------------------------------------
TEST(ImuPreintegration, ConstantAcceleration)
{
  const Eigen::Vector3d a(1.0, -2.0, 0.5);  // m/s^2
  const double          dt = 1e-3;
  const int             N  = 2000;  // 2 seconds
  const double          T  = N * dt;

  ImuPreintegrator pre;
  for (int i = 0; i < N; ++i)
  {
    pre.integrate(Eigen::Vector3d::Zero(), a, dt);
  }

  EXPECT_LT((pre.deltaR() - Eigen::Matrix3d::Identity()).norm(), 1e-9);
  EXPECT_LT((pre.deltaV() - a * T).norm(), 1e-6);
  // Discrete forward integration of ΔP accumulates an O(dt) error vs the exact
  // 0.5*a*T^2; with dt=1ms it is tiny.
  EXPECT_LT((pre.deltaP() - 0.5 * a * T * T).norm(), 1e-2);
}

// ---------------------------------------------------------------------------
// predict(): from rest with zero gravity, constant accel -> p = 0.5 a T^2.
// ---------------------------------------------------------------------------
TEST(ImuPreintegration, PredictNoGravity)
{
  const Eigen::Vector3d a(0.5, 0.0, -1.0);
  const double          dt = 1e-3;
  const int             N  = 1000;
  const double          T  = N * dt;

  ImuPreintegrator pre;
  for (int i = 0; i < N; ++i)
  {
    pre.integrate(Eigen::Vector3d::Zero(), a, dt);
  }

  Eigen::Matrix3d Rj;
  Eigen::Vector3d vj, pj;
  pre.predict(
      Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Rj, vj, pj);

  EXPECT_LT((vj - a * T).norm(), 1e-6);
  EXPECT_LT((pj - 0.5 * a * T * T).norm(), 1e-2);
}

// ---------------------------------------------------------------------------
// Bias-update Jacobians: first-order prediction vs. re-integration with a
// perturbed bias.
// ---------------------------------------------------------------------------
TEST(ImuPreintegration, BiasJacobians)
{
  // A non-trivial motion: rotating + accelerating.
  const Eigen::Vector3d w(0.2, 0.1, -0.15);
  const Eigen::Vector3d a(0.8, -0.3, 9.81);
  const double          dt = 2e-3;
  const int             N  = 500;

  auto integrateWith = [&](const Eigen::Vector3d& bg, const Eigen::Vector3d& ba)
  {
    ImuPreintegrator p(bg, ba);
    for (int i = 0; i < N; ++i)
    {
      p.integrate(w, a, dt);
    }
    return p;
  };

  const Eigen::Vector3d bg0  = Eigen::Vector3d::Zero();
  const Eigen::Vector3d ba0  = Eigen::Vector3d::Zero();
  ImuPreintegrator      base = integrateWith(bg0, ba0);

  // Perturb gyro bias.
  const Eigen::Vector3d dbg(0.01, -0.008, 0.012);
  ImuPreintegrator      pertG = integrateWith(bg0 + dbg, ba0);

  // First-order predicted ΔR change: ΔR(bg+dbg) ≈ ΔR(bg) * Exp(J_R_bg * dbg).
  const Eigen::Matrix3d R_pred = base.deltaR() * so3Exp(base.dR_dbg() * dbg);
  EXPECT_LT((pertG.deltaR() - R_pred).norm(), 5e-4) << "dR/dbg";

  // Velocity / position first-order corrections.
  const Eigen::Vector3d V_pred = base.deltaV() + base.dV_dbg() * dbg;
  const Eigen::Vector3d P_pred = base.deltaP() + base.dP_dbg() * dbg;
  EXPECT_LT((pertG.deltaV() - V_pred).norm(), 5e-3) << "dV/dbg";
  EXPECT_LT((pertG.deltaP() - P_pred).norm(), 5e-3) << "dP/dbg";

  // Perturb accel bias (affects V and P linearly => first order is near exact).
  const Eigen::Vector3d dba(0.05, 0.02, -0.03);
  ImuPreintegrator      pertA   = integrateWith(bg0, ba0 + dba);
  const Eigen::Vector3d V_predA = base.deltaV() + base.dV_dba() * dba;
  const Eigen::Vector3d P_predA = base.deltaP() + base.dP_dba() * dba;
  EXPECT_LT((pertA.deltaV() - V_predA).norm(), 1e-6) << "dV/dba";
  EXPECT_LT((pertA.deltaP() - P_predA).norm(), 1e-6) << "dP/dba";
}
