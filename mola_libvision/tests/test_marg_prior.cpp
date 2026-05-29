/* -------------------------------------------------------------------------
 * mola_libvision unit tests: marginalization prior (Schur complement)
 * SPDX-License-Identifier: BSD-3-Clause
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/marg_prior.h>

#include <Eigen/Dense>
#include <random>

using namespace mola::vision;

namespace
{
// Build a random symmetric positive-definite matrix of size n.
Eigen::MatrixXd randomSPD(int n, std::mt19937& rng)
{
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd                  A(n, n);
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      A(i, j) = nd(rng);
    }
  }
  return A.transpose() * A + n * Eigen::MatrixXd::Identity(n, n);
}
}  // namespace

// ---------------------------------------------------------------------------
// Defining property: minimizing the reduced (marginalized) system gives the
// same values for the kept variables as minimizing the full system.
// ---------------------------------------------------------------------------
TEST(MargPrior, EquivalentSolution)
{
  std::mt19937 rng(123);

  // Full quadratic 0.5 xᵀ H x + bᵀ x over 9 variables (think 3 frames x 3).
  const int                        N = 9;
  Eigen::MatrixXd                  H = randomSPD(N, rng);
  Eigen::VectorXd                  b(N);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int i = 0; i < N; ++i)
  {
    b(i) = nd(rng);
  }

  // Full solution: H x = -b.
  const Eigen::VectorXd x_full = H.ldlt().solve(-b);

  // Marginalize the "middle frame" variables {3,4,5}.
  const std::vector<int> marg = {3, 4, 5};
  const MargResult       r    = schurMarginalize(H, b, marg);
  ASSERT_TRUE(r.ok);

  // Kept variables are {0,1,2,6,7,8} in ascending order.
  EXPECT_EQ(r.H.rows(), 6);
  EXPECT_EQ(r.b.size(), 6);

  // Reduced solution should equal the kept components of the full solution.
  const Eigen::VectorXd  x_red = r.H.ldlt().solve(-r.b);
  const std::vector<int> keep  = {0, 1, 2, 6, 7, 8};
  for (int a = 0; a < 6; ++a)
  {
    EXPECT_NEAR(x_red(a), x_full(keep[a]), 1e-9) << "kept var " << keep[a];
  }

  // Reduced H stays symmetric positive-definite.
  EXPECT_NEAR((r.H - r.H.transpose()).norm(), 0.0, 1e-9);
  Eigen::LLT<Eigen::MatrixXd> llt(r.H);
  EXPECT_EQ(llt.info(), Eigen::Success);
}

// ---------------------------------------------------------------------------
// MargPrior wrapper: by global variable id, with cost() evaluation.
// ---------------------------------------------------------------------------
TEST(MargPrior, ByGlobalId)
{
  std::mt19937    rng(7);
  const int       N = 6;
  Eigen::MatrixXd H = randomSPD(N, rng);
  Eigen::VectorXd b = Eigen::VectorXd::Ones(N);

  // Global ids 10..15; marginalize ids 12,13 (indices 2,3).
  const std::vector<int> var_ids  = {10, 11, 12, 13, 14, 15};
  const std::vector<int> marg_ids = {12, 13};

  MargPrior prior;
  ASSERT_TRUE(prior.build(H, b, var_ids, marg_ids));
  EXPECT_TRUE(prior.valid());

  const std::vector<int> expected_kept = {10, 11, 14, 15};
  EXPECT_EQ(prior.keptIds(), expected_kept);
  EXPECT_EQ(prior.H().rows(), 4);

  // cost(0) == 0; cost is quadratic and matches a manual evaluation.
  EXPECT_NEAR(prior.cost(Eigen::VectorXd::Zero(4)), 0.0, 1e-12);
  Eigen::VectorXd dx(4);
  dx << 0.1, -0.2, 0.3, 0.05;
  const double expected = 0.5 * dx.dot(prior.H() * dx) + prior.b().dot(dx);
  EXPECT_NEAR(prior.cost(dx), expected, 1e-12);
}

// ---------------------------------------------------------------------------
// Invalid inputs.
// ---------------------------------------------------------------------------
TEST(MargPrior, InvalidInputs)
{
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(4, 4);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(4);

  EXPECT_FALSE(schurMarginalize(H, b, {}).ok);  // nothing to marginalize
  EXPECT_FALSE(schurMarginalize(H, b, {7}).ok);  // out-of-range index
  EXPECT_FALSE(schurMarginalize(H, b, {0, 1, 2, 3}).ok);  // nothing kept
}
