/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Custom Levenberg-Marquardt optimizer — no Ceres, no external solver.
 * Inspired by Basalt (BSD-3-Clause, Usenko et al.):
 *   the damping strategy and model-cost-decrease acceptance test follow
 *   Basalt's sqrt_keypoint_vio.cpp approach.
 * ------------------------------------------------------------------------- */
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <optional>

namespace mola::vision
{
/** Outcome of a single LM solve call */
struct LMResult
{
  int   iterations   = 0;
  float final_cost   = 0;
  float initial_cost = 0;
  bool  converged    = false;
  enum class Reason
  {
    MaxIters,
    SmallStep,
    SmallCostChange,
    SmallGradient,
    Failed
  } reason;
};

/** Configuration for the LM solver */
struct LMConfig
{
  float lambda_initial = 1e-4f;  ///< Initial damping factor
  float lambda_min     = 1e-10f;
  float lambda_max     = 1e8f;
  float lambda_factor  = 2.0f;  ///< Multiply λ on rejection
  float eps_step       = 1e-6f;  ///< Convergence: ‖Δx‖ < eps_step
  float eps_cost       = 1e-8f;  ///< Convergence: |ΔCost/Cost| < eps_cost
  float eps_grad       = 1e-8f;  ///< Convergence: ‖Jᵀr‖_inf < eps_grad (at optimum)
  int   max_iters      = 50;
  bool  verbose        = false;
};

/** Huber robust kernel.
 *  rho(r) = r² / 2            if |r| ≤ delta
 *           delta·|r| - δ²/2  if |r| >  delta
 *  Returns the weight w such that the reweighted squared cost = w·r².
 */
inline float huberWeight(float residual, float delta)
{
  const float abs_r = std::abs(residual);
  return (abs_r <= delta) ? 1.0f : delta / abs_r;
}

/** Generic Levenberg-Marquardt solver.
 *
 *  The problem is defined by a callable `cost_and_jacobian` that fills:
 *    - residuals `r`   (N×1)
 *    - Jacobian  `J`   (N×D)
 *  given the current parameter vector `x` (D×1).
 *
 *  The normal equations are:
 *    (JᵀJ + λ·diag(JᵀJ)) · Δx = -Jᵀr
 *  solved via Eigen LDLT.
 *
 *  Update: x ← x + Δx  (caller handles manifold retractions if needed).
 *  The `retract` callable applies Δx to x and returns the new x;
 *  for Euclidean spaces, simply pass `[](auto x, auto dx){ return x+dx; }`.
 *
 *  Template parameters:
 *    D = parameter dimension (use Eigen::Dynamic for runtime size)
 *
 *  \code
 *  // Example: minimize ‖Ax - b‖²
 *  Eigen::VectorXf x = Eigen::VectorXf::Zero(3);
 *  auto fun = [&](const Eigen::VectorXf& xv,
 *                 Eigen::VectorXf& r, Eigen::MatrixXf& J) {
 *      r = A * xv - b;
 *      J = A;
 *  };
 *  auto retract = [](const Eigen::VectorXf& x, const Eigen::VectorXf& dx) {
 *      return x + dx;
 *  };
 *  auto result = levenbergMarquardt(x, fun, retract);
 *  \endcode
 */
template <int D = Eigen::Dynamic>
LMResult levenbergMarquardt(
    Eigen::Matrix<float, D, 1>&                              x,
    const std::function<void(
        const Eigen::Matrix<float, D, 1>& x, Eigen::VectorXf& residuals,
        Eigen::Matrix<float, Eigen::Dynamic, D>& jacobian)>& cost_and_jacobian,
    const std::function<Eigen::Matrix<float, D, 1>(
        const Eigen::Matrix<float, D, 1>& x, const Eigen::Matrix<float, D, 1>& delta)>& retract,
    const LMConfig&                                                                     cfg = {})
{
  LMResult result;
  float    lambda = cfg.lambda_initial;

  Eigen::VectorXf                         r;
  Eigen::Matrix<float, Eigen::Dynamic, D> J;

  // Evaluate at initial x
  cost_and_jacobian(x, r, J);
  float cost          = 0.5f * r.squaredNorm();
  result.initial_cost = cost;

  for (int iter = 0; iter < cfg.max_iters; ++iter)
  {
    result.iterations = iter + 1;

    // Normal equations:  H = JᵀJ + λ·diag(JᵀJ),  b = -Jᵀr
    Eigen::Matrix<float, D, D> H = J.transpose() * J;
    Eigen::Matrix<float, D, 1> b = -(J.transpose() * r);

    // Gradient-norm convergence: at the optimum b = -Jᵀr -> 0. This check
    // must precede the gain-ratio accept/reject logic, because near the
    // optimum the model decrease vanishes and every step would be rejected,
    // leaving the solver looping until max_iters without ever converging.
    if (b.template lpNorm<Eigen::Infinity>() < cfg.eps_grad)
    {
      result.converged = true;
      result.reason    = LMResult::Reason::SmallGradient;
      break;
    }

    // Damping: add λ to diagonal (Marquardt variant)
    for (int i = 0; i < H.rows(); ++i)
    {
      H(i, i) += lambda * H(i, i);
    }

    // Solve with LDLT
    Eigen::Matrix<float, D, 1> dx = H.template selfadjointView<Eigen::Upper>().ldlt().solve(b);

    // Compute model cost decrease: ΔCost_model = bᵀΔx + ½ΔxᵀHΔx (should be > 0)
    const float model_decrease = b.dot(dx) - 0.5f * dx.dot(H * dx);

    // Try the step
    Eigen::Matrix<float, D, 1> x_new = retract(x, dx);

    Eigen::VectorXf                         r_new;
    Eigen::Matrix<float, Eigen::Dynamic, D> J_new;
    cost_and_jacobian(x_new, r_new, J_new);
    const float cost_new = 0.5f * r_new.squaredNorm();

    const float actual_decrease = cost - cost_new;
    const float relative_decrease =
        (model_decrease > 1e-12f) ? actual_decrease / model_decrease : -1.f;

    if (relative_decrease > 0)
    {
      // Accept step
      x    = x_new;
      r    = r_new;
      J    = J_new;
      cost = cost_new;

      // Decrease λ (Nielsen update). Keep all arithmetic in float: std::pow
      // with an int exponent would promote to double and make std::max
      // ambiguous between its float args.
      const float nielsen = 2.f * relative_decrease - 1.f;
      lambda *= std::max(1.f / 3.f, 1.f - nielsen * nielsen * nielsen);
      lambda = std::max(lambda, cfg.lambda_min);

      // Convergence checks
      if (dx.norm() < cfg.eps_step)
      {
        result.converged = true;
        result.reason    = LMResult::Reason::SmallStep;
        break;
      }
      if (cost > 1e-12f && std::abs(actual_decrease) / cost < cfg.eps_cost)
      {
        result.converged = true;
        result.reason    = LMResult::Reason::SmallCostChange;
        break;
      }
    }
    else
    {
      // Reject step — increase λ
      lambda = std::min(lambda * cfg.lambda_factor, cfg.lambda_max);
    }
  }

  if (!result.converged && result.iterations >= cfg.max_iters)
  {
    result.reason = LMResult::Reason::MaxIters;
  }

  result.final_cost = cost;
  return result;
}

}  // namespace mola::vision
