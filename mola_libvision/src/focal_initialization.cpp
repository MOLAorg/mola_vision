/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Self-calibration costs: P.R.S. Mendonca and R. Cipolla, "A simple technique
 * for self-calibration", CVPR 1999; R. Hartley, "Self-calibration of
 * stationary cameras", IJCV 22(1), 1997 (rotation-only case).
 * ------------------------------------------------------------------------- */
#include <mola_libvision/focal_initialization.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

using namespace mola::vision;

namespace
{
Eigen::Matrix3d makeK(double f, double cx, double cy)
{
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  K(0, 0)           = f;
  K(1, 1)           = f;
  K(0, 2)           = cx;
  K(1, 2)           = cy;
  return K;
}

double median(std::vector<double> v)
{
  if (v.empty())
  {
    return std::numeric_limits<double>::infinity();
  }
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
  return v[mid];
}

/** Log-spaced sweep plus golden-section refinement of a 1D cost. */
FocalEstimate sweep(
    const std::function<double(double)>& cost, double f_min, double f_max, int num_samples)
{
  FocalEstimate out;
  if (!(f_min > 0.0) || !(f_max > f_min) || num_samples < 3)
  {
    return out;
  }
  const double la   = std::log(f_min);
  const double lb   = std::log(f_max);
  int          best = -1;
  for (int i = 0; i < num_samples; ++i)
  {
    const double f = std::exp(la + (lb - la) * i / (num_samples - 1));
    const double c = cost(f);
    out.focals.push_back(f);
    out.costs.push_back(c);
    if (std::isfinite(c) && (best < 0 || c < out.costs[static_cast<size_t>(best)]))
    {
      best = i;
    }
  }
  if (best < 0)
  {
    return out;
  }
  // Golden-section search in log(f) between the neighbors of the best sample.
  double       a  = std::log(out.focals[static_cast<size_t>(std::max(best - 1, 0))]);
  double       b  = std::log(out.focals[static_cast<size_t>(std::min(best + 1, num_samples - 1))]);
  const double gr = 0.5 * (std::sqrt(5.0) - 1.0);
  double       c  = b - gr * (b - a);
  double       d  = a + gr * (b - a);
  double       fc = cost(std::exp(c));
  double       fd = cost(std::exp(d));
  for (int it = 0; it < 40 && (b - a) > 1e-6; ++it)
  {
    if (fc < fd)
    {
      b  = d;
      d  = c;
      fd = fc;
      c  = b - gr * (b - a);
      fc = cost(std::exp(c));
    }
    else
    {
      a  = c;
      c  = d;
      fc = fd;
      d  = a + gr * (b - a);
      fd = cost(std::exp(d));
    }
  }
  const double f_best = std::exp(0.5 * (a + b));
  const double c_best = cost(f_best);
  if (c_best <= out.costs[static_cast<size_t>(best)])
  {
    out.focal = f_best;
    out.cost  = c_best;
  }
  else
  {
    out.focal = out.focals[static_cast<size_t>(best)];
    out.cost  = out.costs[static_cast<size_t>(best)];
  }
  std::vector<double> finite;
  for (const double v : out.costs)
  {
    if (std::isfinite(v))
    {
      finite.push_back(v);
    }
  }
  const double med = median(finite);
  out.flatness     = med > 0.0 ? out.cost / med : 1.0;
  out.valid        = std::isfinite(out.cost);
  return out;
}

double defaultMinFocal(const FocalSweepOptions& o)
{
  const double half_w = 0.5 * static_cast<double>(o.width);
  if (o.model == CameraModelType::Equidistant)
  {
    return half_w / (95.0 * M_PI / 180.0);
  }
  return half_w / std::tan(60.0 * M_PI / 180.0);
}

double defaultMaxFocal(const FocalSweepOptions& o)
{
  return 0.5 * static_cast<double>(o.width) / std::tan(10.0 * M_PI / 180.0);
}

CameraModel sweepModel(const FocalSweepOptions& o, double f)
{
  const double cx = 0.5 * (static_cast<double>(o.width) - 1.0);
  const double cy = 0.5 * (static_cast<double>(o.height) - 1.0);
  auto         d  = [&](int i) { return i < o.distortion.size() ? o.distortion[i] : 0.0; };
  switch (o.model)
  {
    case CameraModelType::Equidistant:
      return CameraModel::Equidistant(o.width, o.height, f, f, cx, cy, d(0), d(1), d(2), d(3));
    case CameraModelType::RadTan:
      return CameraModel::RadTan(o.width, o.height, f, f, cx, cy, d(0), d(1), d(2), d(3), d(4));
    case CameraModelType::Pinhole:
    default:
      return CameraModel::Pinhole(o.width, o.height, f, f, cx, cy);
  }
}

/** Local image scale of a camera model at a bearing, in pixels per radian:
 *  geometric mean of the singular values of the projection Jacobian restricted
 *  to the tangent plane of the bearing. Converts angular errors to pixels
 *  without favoring wide (peripherally compressed) fields of view. */
double pixelsPerRadian(const CameraModel& cam, const Eigen::Vector3d& b)
{
  const Eigen::Vector3d       u = b.normalized();
  Eigen::Vector2d             px;
  Eigen::Matrix<double, 2, 3> J;
  if (!cam.project(u, px, &J))
  {
    return 0.5 * (cam.fx() + cam.fy());
  }
  const Eigen::Vector3d a =
      std::abs(u.x()) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
  const Eigen::Vector3d       t1 = u.cross(a).normalized();
  const Eigen::Vector3d       t2 = u.cross(t1);
  Eigen::Matrix<double, 3, 2> T;
  T.col(0) = t1;
  T.col(1) = t2;
  return std::sqrt(std::abs((J * T).determinant()));
}

/** Per-view angular errors of a correspondence to an essential matrix. */
void epipolarErrors(
    const Eigen::Matrix3d& E, const Eigen::Vector3d& b1, const Eigen::Vector3d& b2, double& e1,
    double& e2)
{
  const Eigen::Vector3d u1 = b1.normalized();
  const Eigen::Vector3d u2 = b2.normalized();
  const Eigen::Vector3d n2 = E * u1;
  const Eigen::Vector3d n1 = E.transpose() * u2;
  e1 = n1.norm() > 1e-15 ? std::asin(std::min(1.0, std::abs(u1.dot(n1)) / n1.norm())) : 0.0;
  e2 = n2.norm() > 1e-15 ? std::asin(std::min(1.0, std::abs(u2.dot(n2)) / n2.norm())) : 0.0;
}

/** Error of a correspondence, in pixels, for an essential matrix or (when
 *  rotation_only) a rotation b2 ~ M b1. */
double pixelError(
    const CameraModel& cam, const Eigen::Matrix3d& M, const Eigen::Vector3d& b1,
    const Eigen::Vector3d& b2, bool rotation_only)
{
  if (rotation_only)
  {
    return angleBetween(M * b1, b2) * pixelsPerRadian(cam, b2);
  }
  double e1 = 0;
  double e2 = 0;
  epipolarErrors(M, b1, b2, e1, e2);
  return 0.5 * (e1 * pixelsPerRadian(cam, b1) + e2 * pixelsPerRadian(cam, b2));
}

/** Bearings of the correspondences that both views can unproject. */
void pairBearings(
    const CameraModel& cam, const ViewPairMatches& m, std::vector<Eigen::Vector3d>& b1,
    std::vector<Eigen::Vector3d>& b2, std::vector<size_t>& idx)
{
  b1.clear();
  b2.clear();
  idx.clear();
  for (size_t i = 0; i < m.p1.size(); ++i)
  {
    Eigen::Vector3d a;
    Eigen::Vector3d b;
    if (cam.unproject(m.p1[i], a) && cam.unproject(m.p2[i], b))
    {
      b1.push_back(a);
      b2.push_back(b);
      idx.push_back(i);
    }
  }
}

}  // namespace

FocalEstimate mola::vision::estimateFocalFromFundamentals(
    const std::vector<Eigen::Matrix3d>& F, double cx, double cy, double f_min, double f_max,
    int num_samples)
{
  auto cost = [&](double f)
  {
    const Eigen::Matrix3d K = makeK(f, cx, cy);
    std::vector<double>   c;
    c.reserve(F.size());
    for (const auto& Fi : F)
    {
      const Eigen::Matrix3d             E = K.transpose() * Fi * K;
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(E);
      const auto&                       s = svd.singularValues();
      if (s[0] + s[1] > 0.0)
      {
        c.push_back((s[0] - s[1]) / (s[0] + s[1]));
      }
    }
    return median(c);
  };
  return sweep(cost, f_min, f_max, num_samples);
}

FocalEstimate mola::vision::estimateFocalFromRotationHomographies(
    const std::vector<Eigen::Matrix3d>& H, double cx, double cy, double f_min, double f_max,
    int num_samples)
{
  auto cost = [&](double f)
  {
    const Eigen::Matrix3d K    = makeK(f, cx, cy);
    const Eigen::Matrix3d Kinv = K.inverse();
    std::vector<double>   c;
    c.reserve(H.size());
    for (const auto& Hi : H)
    {
      const Eigen::Matrix3d             M = Kinv * Hi * K;
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(M);
      const auto&                       s = svd.singularValues();
      if (s[1] > 0.0)
      {
        // A scaled rotation has three equal singular values.
        c.push_back((s[0] - s[2]) / s[1]);
      }
    }
    return median(c);
  };
  return sweep(cost, f_min, f_max, num_samples);
}

FocalEstimate mola::vision::estimateFocalByTwoViewConsensus(
    const std::vector<ViewPairMatches>& pairs, const FocalSweepOptions& options)
{
  FocalEstimate out;
  if (pairs.empty() || options.width == 0 || options.height == 0)
  {
    return out;
  }
  const double f_min = options.f_min > 0.0 ? options.f_min : defaultMinFocal(options);
  const double f_max = options.f_max > 0.0 ? options.f_max : defaultMaxFocal(options);
  const double tau   = options.threshold_px;

  // Coarse, sampled cost. Keeps the inliers of each pair at each sample so
  // the fine search can refit deterministically.
  std::vector<std::vector<std::vector<bool>>> inliers_at;  // [sample][pair][match]
  auto                                        coarse_cost = [&](double f)
  {
    const CameraModel              cam   = sweepModel(options, f);
    double                         total = 0.0;
    std::vector<std::vector<bool>> in_pairs;
    for (size_t k = 0; k < pairs.size(); ++k)
    {
      std::vector<Eigen::Vector3d> b1;
      std::vector<Eigen::Vector3d> b2;
      std::vector<size_t>          idx;
      pairBearings(cam, pairs[k], b1, b2, idx);
      std::vector<bool> in(pairs[k].p1.size(), false);
      // Matches that cannot be unprojected pay the outlier cost.
      total += tau * tau * static_cast<double>(pairs[k].p1.size() - b1.size());
      RansacOptions ro;
      ro.threshold      = tau / f;
      ro.max_iterations = options.max_iterations;
      ro.min_iterations = std::min(options.max_iterations, 100);
      ro.seed           = options.seed + static_cast<unsigned>(k);
      RansacReport    rep;
      Eigen::Matrix3d M = Eigen::Matrix3d::Identity();
      if (options.rotation_only)
      {
        const auto est = estimateRotation(b1, b2, ro);
        rep            = est.report;
        M              = est.R;
      }
      else
      {
        const auto est = estimateRelativePose(b1, b2, ro);
        rep            = est.report;
        M              = est.E;
      }
      if (!rep.success)
      {
        total += tau * tau * static_cast<double>(b1.size());
      }
      else
      {
        for (size_t i = 0; i < idx.size(); ++i)
        {
          const double e = pixelError(cam, M, b1[i], b2[i], options.rotation_only);
          total += std::min(e * e, tau * tau);
          in[idx[i]] = e < tau;
        }
      }
      in_pairs.push_back(in);
    }
    inliers_at.push_back(in_pairs);
    return total;
  };

  const int n = std::max(options.num_samples, 5);
  for (int i = 0; i < n; ++i)
  {
    const double f = std::exp(std::log(f_min) + (std::log(f_max) - std::log(f_min)) * i / (n - 1));
    out.focals.push_back(f);
    out.costs.push_back(coarse_cost(f));
  }
  const auto  best_it      = std::min_element(out.costs.begin(), out.costs.end());
  const auto  best         = static_cast<size_t>(std::distance(out.costs.begin(), best_it));
  const auto& best_inliers = inliers_at[best];

  // Fine search: deterministic refit on the best sample's inliers, so the
  // cost is a smooth function of the focal length.
  auto fine_cost = [&](double f)
  {
    const CameraModel cam   = sweepModel(options, f);
    double            total = 0.0;
    for (size_t k = 0; k < pairs.size(); ++k)
    {
      std::vector<Eigen::Vector3d> b1;
      std::vector<Eigen::Vector3d> b2;
      std::vector<size_t>          idx;
      pairBearings(cam, pairs[k], b1, b2, idx);
      total += tau * tau * static_cast<double>(pairs[k].p1.size() - b1.size());
      std::vector<Eigen::Vector3d> s1;
      std::vector<Eigen::Vector3d> s2;
      for (size_t i = 0; i < idx.size(); ++i)
      {
        if (best_inliers[k][idx[i]])
        {
          s1.push_back(b1[i]);
          s2.push_back(b2[i]);
        }
      }
      const auto Mo =
          options.rotation_only ? solveRotationKabsch(s1, s2) : solveEssentialLinear(s1, s2);
      if (!Mo)
      {
        total += tau * tau * static_cast<double>(b1.size());
        continue;
      }
      for (size_t i = 0; i < b1.size(); ++i)
      {
        const double e = pixelError(cam, *Mo, b1[i], b2[i], options.rotation_only);
        total += std::min(e * e, tau * tau);
      }
    }
    return total;
  };
  const double  lo   = out.focals[best > 0 ? best - 1 : 0];
  const double  hi   = out.focals[std::min(best + 1, out.focals.size() - 1)];
  FocalEstimate fine = sweep(fine_cost, lo, hi, 9);
  out.valid          = fine.valid || std::isfinite(*best_it);
  out.focal          = fine.valid ? fine.focal : out.focals[best];
  out.cost           = *best_it;

  std::vector<double> sorted = out.costs;
  const double        med    = median(sorted);
  out.flatness               = med > 0.0 ? out.cost / med : 1.0;
  return out;
}
