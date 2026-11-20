/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * MSAC: P.H.S. Torr and A. Zisserman, "MLESAC: A new robust estimator with
 * application to estimating image geometry", CVIU 78(1), 2000.
 * ------------------------------------------------------------------------- */
#include <mola_libvision/lie_utils.h>
#include <mola_libvision/robust_estimators.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>

using namespace mola::vision;

namespace
{
/** Draws `k` distinct indices in [0, n). */
void drawSample(std::mt19937& rng, size_t n, int k, std::vector<size_t>& out)
{
  out.clear();
  std::uniform_int_distribution<size_t> dist(0, n - 1);
  while (static_cast<int>(out.size()) < k)
  {
    const size_t i = dist(rng);
    if (std::find(out.begin(), out.end(), i) == out.end())
    {
      out.push_back(i);
    }
  }
}

int requiredIterations(int num_inliers, size_t n, int sample_size, double confidence)
{
  // Probability that a sample of distinct correspondences is all inliers.
  double p = 1.0;
  for (int j = 0; j < sample_size; ++j)
  {
    p *= std::max(0.0, static_cast<double>(num_inliers - j)) /
         static_cast<double>(n - static_cast<size_t>(j));
  }
  if (p >= 1.0 - 1e-12)
  {
    return 1;
  }
  if (p <= 0.0)
  {
    return std::numeric_limits<int>::max();
  }
  const double iters = std::log(1.0 - confidence) / std::log(1.0 - p);
  if (!std::isfinite(iters) || iters > 1e7)
  {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(std::ceil(iters));
}

/** Generic MSAC loop. */
template <typename Model, typename Minimal, typename Refit, typename Error>
RansacReport runMsac(
    size_t N, int sample_size, const RansacOptions& opt, Minimal minimal, Refit refit, Error error,
    Model& best_model)
{
  RansacReport rep;
  rep.inliers.assign(N, false);
  rep.residuals.assign(N, std::numeric_limits<double>::infinity());
  if (N < static_cast<size_t>(sample_size))
  {
    return rep;
  }
  const double t2 = opt.threshold * opt.threshold;

  auto evaluate = [&](const Model& m, double& score, int& n_in)
  {
    score = 0.0;
    n_in  = 0;
    for (size_t i = 0; i < N; ++i)
    {
      const double e  = error(m, i);
      const double e2 = std::isfinite(e) ? e * e : t2;
      score += std::min(e2, t2);
      if (e2 < t2)
      {
        ++n_in;
      }
    }
  };

  std::mt19937        rng(opt.seed);
  std::vector<size_t> sample;
  double              best_score = std::numeric_limits<double>::infinity();
  int                 best_in    = 0;
  bool                have_model = false;
  int                 needed     = opt.max_iterations;
  int                 it         = 0;
  for (; it < std::min(needed, opt.max_iterations); ++it)
  {
    drawSample(rng, N, sample_size, sample);
    for (const Model& m : minimal(sample))
    {
      double score = 0;
      int    n_in  = 0;
      evaluate(m, score, n_in);
      if (score < best_score)
      {
        best_score = score;
        best_in    = n_in;
        best_model = m;
        have_model = true;
        needed =
            std::max(opt.min_iterations, requiredIterations(n_in, N, sample_size, opt.confidence));
      }
    }
  }
  rep.iterations = it;
  if (!have_model)
  {
    return rep;
  }

  auto inliersOf = [&](const Model& m)
  {
    std::vector<bool> in(N, false);
    for (size_t i = 0; i < N; ++i)
    {
      const double e = error(m, i);
      in[i]          = std::isfinite(e) && e < opt.threshold;
    }
    return in;
  };

  if (opt.refine)
  {
    for (int round = 0; round < 3; ++round)
    {
      const auto in = inliersOf(best_model);
      const auto m  = refit(in);
      if (!m)
      {
        break;
      }
      double score = 0;
      int    n_in  = 0;
      evaluate(*m, score, n_in);
      if (score < best_score)
      {
        const bool small_gain = best_score - score < 1e-9 * std::max(1.0, best_score);
        best_score            = score;
        best_in               = n_in;
        best_model            = *m;
        if (small_gain)
        {
          break;
        }
      }
      else
      {
        break;
      }
    }
  }

  rep.success     = best_in >= sample_size;
  rep.score       = best_score;
  rep.num_inliers = best_in;
  rep.inliers     = inliersOf(best_model);
  for (size_t i = 0; i < N; ++i)
  {
    rep.residuals[i] = error(best_model, i);
  }
  return rep;
}

template <typename T>
std::vector<T> selectMasked(const std::vector<T>& v, const std::vector<bool>& mask)
{
  std::vector<T> out;
  out.reserve(v.size());
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (mask[i])
    {
      out.push_back(v[i]);
    }
  }
  return out;
}

/** Gauss-Newton refinement of an absolute pose on the angular error. */
AbsolutePose refineAbsolutePose(
    const std::vector<Eigen::Vector3d>& X, const std::vector<Eigen::Vector3d>& b,
    const AbsolutePose& init)
{
  AbsolutePose pose = init;
  for (int it = 0; it < 10; ++it)
  {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    for (size_t i = 0; i < X.size(); ++i)
    {
      const Eigen::Vector3d p  = pose.R * X[i] + pose.t;
      const double          np = p.norm();
      if (np < 1e-12)
      {
        continue;
      }
      const Eigen::Vector3d u = p / np;
      const Eigen::Vector3d r = u - b[i].normalized();
      // Left perturbation: p' = Exp(w) p + v -> dp/dw = -[p]x, dp/dv = I.
      Eigen::Matrix<double, 3, 6> Jp;
      Jp.leftCols<3>()                     = -skew(p);
      Jp.rightCols<3>()                    = Eigen::Matrix3d::Identity();
      const Eigen::Matrix3d             Ju = (Eigen::Matrix3d::Identity() - u * u.transpose()) / np;
      const Eigen::Matrix<double, 3, 6> J  = Ju * Jp;
      H += J.transpose() * J;
      g += J.transpose() * r;
    }
    H.diagonal().array() += 1e-12;
    const Eigen::Matrix<double, 6, 1> dx = -H.ldlt().solve(g);
    if (!dx.allFinite())
    {
      break;
    }
    const Eigen::Matrix3d dR = so3Exp(dx.head<3>());
    pose.R                   = dR * pose.R;
    pose.t                   = dR * pose.t + dx.tail<3>();
    if (dx.norm() < 1e-12)
    {
      break;
    }
  }
  return pose;
}

}  // namespace

RelativePoseEstimate mola::vision::estimateRelativePose(
    const std::vector<Eigen::Vector3d>& b1, const std::vector<Eigen::Vector3d>& b2,
    const RansacOptions& options)
{
  RelativePoseEstimate out;
  const size_t         N = b1.size();
  if (N != b2.size())
  {
    return out;
  }
  std::vector<Eigen::Vector3d> u1(N);
  std::vector<Eigen::Vector3d> u2(N);
  for (size_t i = 0; i < N; ++i)
  {
    u1[i] = b1[i].normalized();
    u2[i] = b2[i].normalized();
  }
  auto minimal = [&](const std::vector<size_t>& s)
  {
    Eigen::Vector3d a[5];
    Eigen::Vector3d b[5];
    for (int k = 0; k < 5; ++k)
    {
      a[k] = u1[s[static_cast<size_t>(k)]];
      b[k] = u2[s[static_cast<size_t>(k)]];
    }
    return solveEssentialFivePoint(a, b);
  };
  auto refit = [&](const std::vector<bool>& in) -> std::optional<Eigen::Matrix3d>
  { return solveEssentialLinear(selectMasked(u1, in), selectMasked(u2, in)); };
  auto error = [&](const Eigen::Matrix3d& E, size_t i)
  { return epipolarAngularError(E, u1[i], u2[i]); };
  out.report = runMsac<Eigen::Matrix3d>(N, 5, options, minimal, refit, error, out.E);
  if (!out.report.success)
  {
    return out;
  }
  const auto pose = relativePoseFromEssential(out.E, u1, u2, &out.report.inliers);
  if (!pose)
  {
    out.report.success = false;
    return out;
  }
  out.pose = *pose;
  return out;
}

FundamentalEstimate mola::vision::estimateFundamental(
    const std::vector<Eigen::Vector2d>& p1, const std::vector<Eigen::Vector2d>& p2,
    const RansacOptions& options)
{
  FundamentalEstimate out;
  const size_t        N = p1.size();
  if (N != p2.size())
  {
    return out;
  }
  auto minimal = [&](const std::vector<size_t>& s)
  {
    Eigen::Vector2d a[7];
    Eigen::Vector2d b[7];
    for (int k = 0; k < 7; ++k)
    {
      a[k] = p1[s[static_cast<size_t>(k)]];
      b[k] = p2[s[static_cast<size_t>(k)]];
    }
    return solveFundamentalSevenPoint(a, b);
  };
  auto refit = [&](const std::vector<bool>& in)
  { return solveFundamentalLinear(selectMasked(p1, in), selectMasked(p2, in)); };
  auto error = [&](const Eigen::Matrix3d& F, size_t i) { return sampsonDistance(F, p1[i], p2[i]); };
  out.report = runMsac<Eigen::Matrix3d>(N, 7, options, minimal, refit, error, out.F);
  return out;
}

HomographyEstimate mola::vision::estimateHomography(
    const std::vector<Eigen::Vector2d>& p1, const std::vector<Eigen::Vector2d>& p2,
    const RansacOptions& options)
{
  HomographyEstimate out;
  const size_t       N = p1.size();
  if (N != p2.size())
  {
    return out;
  }
  auto minimal = [&](const std::vector<size_t>& s)
  {
    std::vector<Eigen::Vector2d> a(4);
    std::vector<Eigen::Vector2d> b(4);
    for (size_t k = 0; k < 4; ++k)
    {
      a[k] = p1[s[k]];
      b[k] = p2[s[k]];
    }
    std::vector<Eigen::Matrix3d> v;
    if (const auto H = solveHomographyDLT(a, b); H && std::abs(H->determinant()) > 1e-12)
    {
      v.push_back(*H);
    }
    return v;
  };
  auto refit = [&](const std::vector<bool>& in)
  { return solveHomographyDLT(selectMasked(p1, in), selectMasked(p2, in)); };
  auto error = [&](const Eigen::Matrix3d& H, size_t i)
  { return homographyTransferError(H, p1[i], p2[i]); };
  out.report = runMsac<Eigen::Matrix3d>(N, 4, options, minimal, refit, error, out.H);
  return out;
}

RotationEstimate mola::vision::estimateRotation(
    const std::vector<Eigen::Vector3d>& b1, const std::vector<Eigen::Vector3d>& b2,
    const RansacOptions& options)
{
  RotationEstimate out;
  const size_t     N = b1.size();
  if (N != b2.size())
  {
    return out;
  }
  auto minimal = [&](const std::vector<size_t>& s)
  {
    std::vector<Eigen::Matrix3d>       v;
    const std::vector<Eigen::Vector3d> a = {b1[s[0]], b1[s[1]]};
    const std::vector<Eigen::Vector3d> b = {b2[s[0]], b2[s[1]]};
    if (const auto R = solveRotationKabsch(a, b); R)
    {
      v.push_back(*R);
    }
    return v;
  };
  auto refit = [&](const std::vector<bool>& in)
  { return solveRotationKabsch(selectMasked(b1, in), selectMasked(b2, in)); };
  auto error = [&](const Eigen::Matrix3d& R, size_t i) { return angleBetween(R * b1[i], b2[i]); };
  out.report = runMsac<Eigen::Matrix3d>(N, 2, options, minimal, refit, error, out.R);
  return out;
}

AbsolutePoseEstimate mola::vision::estimateAbsolutePose(
    const std::vector<Eigen::Vector3d>& points_w, const std::vector<Eigen::Vector3d>& bearings,
    const RansacOptions& options)
{
  AbsolutePoseEstimate out;
  const size_t         N = points_w.size();
  if (N != bearings.size())
  {
    return out;
  }
  auto minimal = [&](const std::vector<size_t>& s)
  {
    Eigen::Vector3d b[3];
    Eigen::Vector3d X[3];
    for (int k = 0; k < 3; ++k)
    {
      b[k] = bearings[s[static_cast<size_t>(k)]];
      X[k] = points_w[s[static_cast<size_t>(k)]];
    }
    return solveP3P(b, X);
  };
  auto error = [&](const AbsolutePose& P, size_t i)
  { return angleBetween(P.R * points_w[i] + P.t, bearings[i]); };
  AbsolutePose current;
  auto         refit = [&](const std::vector<bool>& in) -> std::optional<AbsolutePose>
  {
    const auto X = selectMasked(points_w, in);
    const auto b = selectMasked(bearings, in);
    if (X.size() < 4)
    {
      return std::nullopt;
    }
    return refineAbsolutePose(X, b, current);
  };
  // The refit needs the current model as its starting point: wrap it so the
  // generic loop passes the best model through `current`.
  auto refit_from_best = [&](const std::vector<bool>& in)
  {
    current = out.pose;
    return refit(in);
  };
  out.report = runMsac<AbsolutePose>(N, 3, options, minimal, refit_from_best, error, out.pose);
  return out;
}
