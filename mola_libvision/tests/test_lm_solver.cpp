/* -------------------------------------------------------------------------
 * mola_libvision unit tests: lm_solver (Levenberg-Marquardt)
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/lm_solver.h>

#include <cmath>

using namespace mola::vision;

// ---------------------------------------------------------------------------
// Simple linear least squares: min ‖Ax - b‖²  (exact solution exists)
// ---------------------------------------------------------------------------
TEST(LMSolver, LinearLeastSquares)
{
  // A is 4×2, b is 4×1
  Eigen::Matrix<float, 4, 2> A;
  A << 1, 0, 0, 1, 1, 1, 2, -1;
  Eigen::Vector4f b(1, 2, 3, 0);

  // Ground truth via normal equations
  Eigen::Vector2f x_gt = (A.transpose() * A).ldlt().solve(A.transpose() * b);

  Eigen::Vector2f x = Eigen::Vector2f::Zero();

  auto cost_jac =
      [&](const Eigen::Vector2f& xv, Eigen::VectorXf& r, Eigen::Matrix<float, Eigen::Dynamic, 2>& J)
  {
    r = A * xv - b;
    J = A.cast<float>();
  };

  auto retract = [](const Eigen::Vector2f& x, const Eigen::Vector2f& dx) { return x + dx; };

  LMConfig cfg;
  cfg.max_iters = 100;
  auto result   = levenbergMarquardt<2>(x, cost_jac, retract, cfg);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(x(0), x_gt(0), 1e-3f);
  EXPECT_NEAR(x(1), x_gt(1), 1e-3f);
}

// ---------------------------------------------------------------------------
// Rosenbrock function: f(x,y) = 100(y - x²)² + (1-x)²
// Minimum at (1, 1) with f=0
// Reformulated as least-squares: r1 = 10(y - x²), r2 = (1-x)
// J = [ -20x  10 ]
//     [ -1     0  ]
// ---------------------------------------------------------------------------
TEST(LMSolver, Rosenbrock)
{
  Eigen::Vector2f x(-1.2f, 1.f);  // standard starting point

  auto cost_jac =
      [](const Eigen::Vector2f& xv, Eigen::VectorXf& r, Eigen::Matrix<float, Eigen::Dynamic, 2>& J)
  {
    r.resize(2);
    r(0) = 10.f * (xv(1) - xv(0) * xv(0));
    r(1) = 1.f - xv(0);

    J.resize(2, 2);
    J(0, 0) = -20.f * xv(0);
    J(0, 1) = 10.f;
    J(1, 0) = -1.f;
    J(1, 1) = 0.f;
  };

  auto retract = [](const Eigen::Vector2f& x, const Eigen::Vector2f& dx) { return x + dx; };

  LMConfig cfg;
  cfg.max_iters      = 500;
  cfg.lambda_initial = 1e-3f;
  cfg.eps_step       = 1e-7f;
  cfg.eps_cost       = 1e-10f;
  auto result        = levenbergMarquardt<2>(x, cost_jac, retract, cfg);

  EXPECT_NEAR(x(0), 1.f, 1e-3f) << "Rosenbrock: x should converge to 1";
  EXPECT_NEAR(x(1), 1.f, 1e-3f) << "Rosenbrock: y should converge to 1";
  EXPECT_LT(result.final_cost, 1e-6f);
}

// ---------------------------------------------------------------------------
// Dynamic-size (Eigen::Dynamic) variant
// ---------------------------------------------------------------------------
TEST(LMSolver, DynamicSize)
{
  // Fit a line y = a*x + b to 5 data points
  // Residual: r[i] = a*x[i] + b - y[i]
  std::vector<float> xs = {0.f, 1.f, 2.f, 3.f, 4.f};
  std::vector<float> ys = {1.f, 3.f, 5.f, 7.f, 9.f};  // y = 2x + 1

  Eigen::VectorXf params(2);  // [a, b]
  params.setZero();

  auto cost_jac = [&](const Eigen::VectorXf& pv, Eigen::VectorXf& r,
                      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& J)
  {
    const int n = static_cast<int>(xs.size());
    r.resize(n);
    J.resize(n, 2);
    for (int i = 0; i < n; ++i)
    {
      r(i)    = pv(0) * xs[i] + pv(1) - ys[i];
      J(i, 0) = xs[i];
      J(i, 1) = 1.f;
    }
  };

  auto retract = [](const Eigen::VectorXf& p, const Eigen::VectorXf& dp) { return p + dp; };

  LMConfig cfg;
  levenbergMarquardt<Eigen::Dynamic>(params, cost_jac, retract, cfg);

  EXPECT_NEAR(params(0), 2.f, 1e-3f) << "slope should be 2";
  EXPECT_NEAR(params(1), 1.f, 1e-3f) << "intercept should be 1";
}

// ---------------------------------------------------------------------------
// Huber weight
// ---------------------------------------------------------------------------
TEST(LMSolver, HuberWeight)
{
  const float delta = 1.0f;
  // Inside δ → weight=1
  EXPECT_FLOAT_EQ(huberWeight(0.5f, delta), 1.f);
  EXPECT_FLOAT_EQ(huberWeight(-0.9f, delta), 1.f);
  // Outside δ → weight = δ/|r| < 1
  EXPECT_FLOAT_EQ(huberWeight(2.f, delta), 0.5f);
  EXPECT_FLOAT_EQ(huberWeight(-4.f, delta), 0.25f);
}
