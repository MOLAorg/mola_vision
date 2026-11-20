/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/camera_model.h>
#include <mrpt/core/exceptions.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

using namespace mola::vision;

namespace
{
// Below this distance to the optical axis (relative to the depth), the
// equidistant mapping is evaluated with its pinhole limit.
constexpr double kSmallRadius = 1e-10;

// Number of samples used to check the monotonicity of the radial mapping.
constexpr int kMonotonicSamples = 16;

// Radial derivative of the RadTan mapping as a function of s = r^2:
// d/dr [ r (1 + k1 r^2 + k2 r^4 + k3 r^6) ] = 1 + 3 k1 s + 5 k2 s^2 + 7 k3 s^3
double radTanRadialDerivative(const Eigen::VectorXd& p, double s)
{
  const double k1 = p[4];
  const double k2 = p[5];
  const double k3 = p[8];
  return 1.0 + s * (3.0 * k1 + s * (5.0 * k2 + s * 7.0 * k3));
}

// Derivative d(theta_d)/d(theta) of the equidistant mapping, t = theta^2.
double equidistantDerivative(const Eigen::VectorXd& p, double t)
{
  const double k1 = p[4];
  const double k2 = p[5];
  const double k3 = p[6];
  const double k4 = p[7];
  return 1.0 + t * (3.0 * k1 + t * (5.0 * k2 + t * (7.0 * k3 + t * 9.0 * k4)));
}

// Checks that f(x) > 0 on [0, x_max] by dense sampling (the functions are
// low-degree polynomials, so a few samples catch any fold in practice).
template <typename F>
bool positiveOn(F f, double x_max)
{
  for (int i = 1; i <= kMonotonicSamples; ++i)
  {
    const double x = x_max * static_cast<double>(i) / kMonotonicSamples;
    if (f(x) <= 0.0)
    {
      return false;
    }
  }
  return true;
}

// Distortion of normalized coordinates (RadTan) with its 2x2 Jacobian.
void radTanDistort(
    const Eigen::VectorXd& p, double x, double y, double& xd, double& yd,
    Eigen::Matrix2d* J = nullptr)
{
  const double k1     = p[4];
  const double k2     = p[5];
  const double p1     = p[6];
  const double p2     = p[7];
  const double k3     = p[8];
  const double r2     = x * x + y * y;
  const double radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
  xd                  = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  yd                  = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
  if (J != nullptr)
  {
    const double dr = k1 + r2 * (2.0 * k2 + r2 * 3.0 * k3);  // d(radial)/d(r2)
    (*J)(0, 0)      = radial + 2.0 * x * x * dr + 2.0 * p1 * y + 6.0 * p2 * x;
    (*J)(0, 1)      = 2.0 * x * y * dr + 2.0 * p1 * x + 2.0 * p2 * y;
    (*J)(1, 0)      = 2.0 * x * y * dr + 2.0 * p1 * x + 2.0 * p2 * y;
    (*J)(1, 1)      = radial + 2.0 * y * y * dr + 6.0 * p1 * y + 2.0 * p2 * x;
  }
}

double equidistantThetaD(const Eigen::VectorXd& p, double theta)
{
  const double t = theta * theta;
  return theta * (1.0 + t * (p[4] + t * (p[5] + t * (p[6] + t * p[7]))));
}

}  // namespace

// ---------------------------------------------------------------------------
// CameraModel
// ---------------------------------------------------------------------------

CameraModel::CameraModel(
    CameraModelType type, uint32_t width, uint32_t height, const Eigen::VectorXd& params)
    : type_(type), width_(width), height_(height)
{
  setParams(params);
}

CameraModel CameraModel::Pinhole(
    uint32_t width, uint32_t height, double fx, double fy, double cx, double cy)
{
  Eigen::VectorXd p(4);
  p << fx, fy, cx, cy;
  return {CameraModelType::Pinhole, width, height, p};
}

CameraModel CameraModel::RadTan(
    uint32_t width, uint32_t height, double fx, double fy, double cx, double cy, double k1,
    double k2, double p1, double p2, double k3)
{
  Eigen::VectorXd p(9);
  p << fx, fy, cx, cy, k1, k2, p1, p2, k3;
  return {CameraModelType::RadTan, width, height, p};
}

CameraModel CameraModel::Equidistant(
    uint32_t width, uint32_t height, double fx, double fy, double cx, double cy, double k1,
    double k2, double k3, double k4)
{
  Eigen::VectorXd p(8);
  p << fx, fy, cx, cy, k1, k2, k3, k4;
  return {CameraModelType::Equidistant, width, height, p};
}

CameraModel CameraModel::FromTCamera(const mrpt::img::TCamera& cam)
{
  switch (cam.distortion)
  {
    case mrpt::img::DistortionModel::none:
      return Pinhole(cam.ncols, cam.nrows, cam.fx(), cam.fy(), cam.cx(), cam.cy());
    case mrpt::img::DistortionModel::plumb_bob:
      ASSERTMSG_(
          cam.k4() == 0.0 && cam.k5() == 0.0 && cam.k6() == 0.0,
          "The rational plumb_bob coefficients k4..k6 are not supported");
      return RadTan(
          cam.ncols, cam.nrows, cam.fx(), cam.fy(), cam.cx(), cam.cy(), cam.k1(), cam.k2(),
          cam.p1(), cam.p2(), cam.k3());
    case mrpt::img::DistortionModel::kannala_brandt:
      return Equidistant(
          cam.ncols, cam.nrows, cam.fx(), cam.fy(), cam.cx(), cam.cy(), cam.k1(), cam.k2(),
          cam.k3(), cam.k4());
    default:
      THROW_EXCEPTION("Unsupported distortion model");
  }
}

mrpt::img::TCamera CameraModel::asTCamera() const
{
  mrpt::img::TCamera cam;
  cam.ncols = width_;
  cam.nrows = height_;
  cam.setIntrinsicParamsFromValues(fx(), fy(), cx(), cy());
  switch (type_)
  {
    case CameraModelType::Pinhole:
      cam.distortion = mrpt::img::DistortionModel::none;
      break;
    case CameraModelType::RadTan:
      cam.setDistortionPlumbBob(params_[4], params_[5], params_[6], params_[7], params_[8]);
      break;
    case CameraModelType::Equidistant:
      cam.setDistortionKannalaBrandt(params_[4], params_[5], params_[6], params_[7]);
      break;
  }
  return cam;
}

int CameraModel::NumParams(CameraModelType type)
{
  switch (type)
  {
    case CameraModelType::Pinhole:
      return 4;
    case CameraModelType::RadTan:
      return 9;
    case CameraModelType::Equidistant:
      return 8;
  }
  return 4;
}

std::vector<std::string> CameraModel::ParamNames(CameraModelType type)
{
  switch (type)
  {
    case CameraModelType::Pinhole:
      return {"fx", "fy", "cx", "cy"};
    case CameraModelType::RadTan:
      return {"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"};
    case CameraModelType::Equidistant:
      return {"fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"};
  }
  return {};
}

std::string CameraModel::RosDistortionModelName(CameraModelType type)
{
  return type == CameraModelType::Equidistant ? "equidistant" : "plumb_bob";
}

std::string CameraModel::TypeName(CameraModelType type)
{
  switch (type)
  {
    case CameraModelType::Pinhole:
      return "pinhole";
    case CameraModelType::RadTan:
      return "radtan";
    case CameraModelType::Equidistant:
      return "equidistant";
  }
  return "unknown";
}

void CameraModel::setImageSize(uint32_t width, uint32_t height)
{
  width_  = width;
  height_ = height;
}

void CameraModel::setParams(const Eigen::VectorXd& params)
{
  ASSERT_EQUAL_(static_cast<int>(params.size()), NumParams(type_));
  params_ = params;
}

Eigen::VectorXd CameraModel::distortion() const
{
  const Eigen::Index n = params_.size() - 4;
  if (type_ == CameraModelType::RadTan)
  {
    Eigen::VectorXd d(5);
    d << params_[4], params_[5], params_[6], params_[7], params_[8];
    return d;
  }
  return params_.tail(n);
}

bool CameraModel::isMonotonicAt(double incidence_angle) const
{
  if (incidence_angle < 0.0)
  {
    return false;
  }
  switch (type_)
  {
    case CameraModelType::Pinhole:
      return incidence_angle < M_PI_2;
    case CameraModelType::RadTan:
    {
      if (incidence_angle >= M_PI_2)
      {
        return false;
      }
      const double r = std::tan(incidence_angle);
      return positiveOn([&](double s) { return radTanRadialDerivative(params_, s); }, r * r);
    }
    case CameraModelType::Equidistant:
    {
      if (incidence_angle >= M_PI)
      {
        return false;
      }
      return positiveOn(
          [&](double t) { return equidistantDerivative(params_, t); },
          incidence_angle * incidence_angle);
    }
  }
  return false;
}

double CameraModel::maxIncidenceAngle() const
{
  const double limit = (type_ == CameraModelType::Equidistant) ? M_PI : M_PI_2;
  // Bisection on the first angle where the mapping stops being monotonic.
  double lo = 0.0;
  double hi = limit * (1.0 - 1e-9);
  if (isMonotonicAt(hi))
  {
    return hi;
  }
  for (int it = 0; it < 60; ++it)
  {
    const double mid = 0.5 * (lo + hi);
    if (isMonotonicAt(mid))
    {
      lo = mid;
    }
    else
    {
      hi = mid;
    }
  }
  return lo;
}

bool CameraModel::project(
    const Eigen::Vector3d& P, Eigen::Vector2d& pixel, Eigen::Matrix<double, 2, 3>* J_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* J_params) const
{
  const double fx_ = params_[0];
  const double fy_ = params_[1];
  const double X   = P.x();
  const double Y   = P.y();
  const double Z   = P.z();

  if (type_ == CameraModelType::Pinhole || type_ == CameraModelType::RadTan)
  {
    // Require the point to be in front, at a non-grazing angle.
    if (Z <= 1e-9 * P.norm() || Z <= 0.0)
    {
      return false;
    }
    const double    iz = 1.0 / Z;
    const double    x  = X * iz;
    const double    y  = Y * iz;
    double          xd = x;
    double          yd = y;
    Eigen::Matrix2d Jd = Eigen::Matrix2d::Identity();
    if (type_ == CameraModelType::RadTan)
    {
      const double r2 = x * x + y * y;
      if (!positiveOn([&](double s) { return radTanRadialDerivative(params_, s); }, r2))
      {
        return false;
      }
      radTanDistort(params_, x, y, xd, yd, J_point != nullptr ? &Jd : nullptr);
    }
    pixel.x() = fx_ * xd + params_[2];
    pixel.y() = fy_ * yd + params_[3];

    if (J_point != nullptr)
    {
      Eigen::Matrix<double, 2, 3> Jn;  // d(x,y)/d(P)
      Jn << iz, 0.0, -x * iz, 0.0, iz, -y * iz;
      Eigen::Matrix2d F = Eigen::Matrix2d::Zero();
      F(0, 0)           = fx_;
      F(1, 1)           = fy_;
      *J_point          = F * Jd * Jn;
    }
    if (J_params != nullptr)
    {
      J_params->setZero(2, params_.size());
      (*J_params)(0, 0) = xd;
      (*J_params)(1, 1) = yd;
      (*J_params)(0, 2) = 1.0;
      (*J_params)(1, 3) = 1.0;
      if (type_ == CameraModelType::RadTan)
      {
        const double r2   = x * x + y * y;
        const double r4   = r2 * r2;
        const double r6   = r4 * r2;
        (*J_params)(0, 4) = fx_ * x * r2;
        (*J_params)(1, 4) = fy_ * y * r2;
        (*J_params)(0, 5) = fx_ * x * r4;
        (*J_params)(1, 5) = fy_ * y * r4;
        (*J_params)(0, 6) = fx_ * 2.0 * x * y;
        (*J_params)(1, 6) = fy_ * (r2 + 2.0 * y * y);
        (*J_params)(0, 7) = fx_ * (r2 + 2.0 * x * x);
        (*J_params)(1, 7) = fy_ * 2.0 * x * y;
        (*J_params)(0, 8) = fx_ * x * r6;
        (*J_params)(1, 8) = fy_ * y * r6;
      }
    }
    return true;
  }

  // Equidistant (Kannala-Brandt):
  const double r2  = X * X + Y * Y;
  const double r   = std::sqrt(r2);
  const double rho = std::sqrt(r2 + Z * Z);
  if (rho <= 0.0)
  {
    return false;
  }
  const double theta = std::atan2(r, Z);
  if (!positiveOn([&](double t) { return equidistantDerivative(params_, t); }, theta * theta))
  {
    return false;
  }
  const double theta_d = equidistantThetaD(params_, theta);
  const bool   small   = r <= kSmallRadius * rho;
  // s = theta_d / r, with its pinhole limit on the axis.
  const double s  = small ? 1.0 / Z : theta_d / r;
  const double mx = s * X;
  const double my = s * Y;
  pixel.x()       = fx_ * mx + params_[2];
  pixel.y()       = fy_ * my + params_[3];

  if (J_point != nullptr)
  {
    Eigen::Matrix<double, 2, 3> Jm;
    if (small)
    {
      Jm << 1.0 / Z, 0.0, -X / (Z * Z), 0.0, 1.0 / Z, -Y / (Z * Z);
    }
    else
    {
      const double g    = equidistantDerivative(params_, theta * theta);
      const double rho2 = rho * rho;
      const double a    = (g * Z / rho2 - s) / r2;  // common factor of the radial terms
      Jm(0, 0)          = s + X * X * a;
      Jm(0, 1)          = X * Y * a;
      Jm(0, 2)          = -g * X / rho2;
      Jm(1, 0)          = X * Y * a;
      Jm(1, 1)          = s + Y * Y * a;
      Jm(1, 2)          = -g * Y / rho2;
    }
    Eigen::Matrix2d F = Eigen::Matrix2d::Zero();
    F(0, 0)           = fx_;
    F(1, 1)           = fy_;
    *J_point          = F * Jm;
  }
  if (J_params != nullptr)
  {
    J_params->setZero(2, params_.size());
    (*J_params)(0, 0) = mx;
    (*J_params)(1, 1) = my;
    (*J_params)(0, 2) = 1.0;
    (*J_params)(1, 3) = 1.0;
    if (!small)
    {
      const double cx_ = X / r;
      const double cy_ = Y / r;
      double       tp  = theta * theta * theta;  // theta^(2i+1), i = 1..4
      for (int i = 0; i < 4; ++i)
      {
        (*J_params)(0, 4 + i) = fx_ * tp * cx_;
        (*J_params)(1, 4 + i) = fy_ * tp * cy_;
        tp *= theta * theta;
      }
    }
  }
  return true;
}

bool CameraModel::unproject(const Eigen::Vector2d& pixel, Eigen::Vector3d& bearing) const
{
  const double xd = (pixel.x() - params_[2]) / params_[0];
  const double yd = (pixel.y() - params_[3]) / params_[1];

  if (type_ == CameraModelType::Pinhole)
  {
    bearing = Eigen::Vector3d(xd, yd, 1.0).normalized();
    return true;
  }

  if (type_ == CameraModelType::RadTan)
  {
    // Newton iteration on distort(x, y) = (xd, yd).
    double x         = xd;
    double y         = yd;
    bool   converged = false;
    for (int it = 0; it < 40; ++it)
    {
      double          fx_ = 0;
      double          fy_ = 0;
      Eigen::Matrix2d J;
      radTanDistort(params_, x, y, fx_, fy_, &J);
      const Eigen::Vector2d err(fx_ - xd, fy_ - yd);
      if (err.norm() < 1e-12)
      {
        converged = true;
        break;
      }
      const double det = J.determinant();
      if (std::abs(det) < 1e-12)
      {
        return false;
      }
      const Eigen::Vector2d step = J.inverse() * err;
      x -= step.x();
      y -= step.y();
      if (!std::isfinite(x) || !std::isfinite(y))
      {
        return false;
      }
    }
    if (!converged)
    {
      double fx_ = 0;
      double fy_ = 0;
      radTanDistort(params_, x, y, fx_, fy_);
      converged = std::hypot(fx_ - xd, fy_ - yd) < 1e-9;
    }
    if (!converged)
    {
      return false;
    }
    const double r2 = x * x + y * y;
    if (!positiveOn([&](double s) { return radTanRadialDerivative(params_, s); }, r2))
    {
      return false;
    }
    bearing = Eigen::Vector3d(x, y, 1.0).normalized();
    return true;
  }

  // Equidistant: solve theta_d(theta) = |m| by Newton.
  const double td = std::hypot(xd, yd);
  if (td < 1e-15)
  {
    bearing = Eigen::Vector3d(0.0, 0.0, 1.0);
    return true;
  }
  double theta     = td;
  bool   converged = false;
  for (int it = 0; it < 40; ++it)
  {
    const double f  = equidistantThetaD(params_, theta) - td;
    const double df = equidistantDerivative(params_, theta * theta);
    if (std::abs(f) < 1e-13)
    {
      converged = true;
      break;
    }
    if (df <= 1e-12)
    {
      return false;
    }
    theta -= f / df;
    theta = std::clamp(theta, 0.0, M_PI);
  }
  if (!converged)
  {
    converged = std::abs(equidistantThetaD(params_, theta) - td) < 1e-10;
  }
  if (!converged || theta >= M_PI ||
      !positiveOn([&](double t) { return equidistantDerivative(params_, t); }, theta * theta))
  {
    return false;
  }
  const double st = std::sin(theta);
  bearing         = Eigen::Vector3d(st * xd / td, st * yd / td, std::cos(theta));
  return true;
}

// ---------------------------------------------------------------------------
// IntrinsicsMask / IntrinsicsParameterization
// ---------------------------------------------------------------------------

std::vector<bool> IntrinsicsMask::FirstN(int total, int n)
{
  std::vector<bool> v(static_cast<size_t>(std::max(total, 0)), false);
  for (int i = 0; i < std::min(n, total); ++i)
  {
    v[static_cast<size_t>(i)] = true;
  }
  return v;
}

namespace
{
// Index in the full parameter vector of the i-th distortion coefficient
// (in CameraModel::distortion() order).
int distortionIndex(CameraModelType type, int i)
{
  (void)type;
  // Both RadTan [k1 k2 p1 p2 k3] and Equidistant [k1 k2 k3 k4] store the
  // distortion right after the four pinhole parameters, in that order.
  return 4 + i;
}

std::string distortionName(CameraModelType type, int i)
{
  const auto names = CameraModel::ParamNames(type);
  return names.at(static_cast<size_t>(4 + i));
}
}  // namespace

IntrinsicsParameterization::IntrinsicsParameterization(
    const CameraModel& reference, const IntrinsicsMask& mask)
    : reference_(reference)
{
  const int N = CameraModel::NumParams(reference.type());

  // Columns of the Jacobian, built as a list of (full index, weight) sets.
  std::vector<std::vector<std::pair<int, double>>> cols;

  if (mask.estimate_focal)
  {
    if (mask.fixed_aspect_ratio)
    {
      shared_focal_ = true;
      aspect_       = reference.fx() != 0.0 ? reference.fy() / reference.fx() : 1.0;
      cols.push_back({{0, 1.0}, {1, aspect_}});
      indices_.push_back(0);
      names_.emplace_back("f");
    }
    else
    {
      cols.push_back({{0, 1.0}});
      cols.push_back({{1, 1.0}});
      indices_.push_back(0);
      indices_.push_back(1);
      names_.emplace_back("fx");
      names_.emplace_back("fy");
    }
  }
  if (mask.estimate_principal_point)
  {
    cols.push_back({{2, 1.0}});
    cols.push_back({{3, 1.0}});
    indices_.push_back(2);
    indices_.push_back(3);
    names_.emplace_back("cx");
    names_.emplace_back("cy");
  }
  const int n_dist = N - 4;
  for (int i = 0; i < n_dist && i < static_cast<int>(mask.estimate_distortion.size()); ++i)
  {
    if (mask.estimate_distortion[static_cast<size_t>(i)])
    {
      const int idx = distortionIndex(reference.type(), i);
      cols.push_back({{idx, 1.0}});
      indices_.push_back(idx);
      names_.push_back(distortionName(reference.type(), i));
    }
  }

  J_.setZero(N, static_cast<Eigen::Index>(cols.size()));
  for (size_t c = 0; c < cols.size(); ++c)
  {
    for (const auto& [row, w] : cols[c])
    {
      J_(row, static_cast<Eigen::Index>(c)) = w;
    }
  }
}

Eigen::VectorXd IntrinsicsParameterization::toFree(const CameraModel& cam) const
{
  Eigen::VectorXd v(dim());
  for (int i = 0; i < dim(); ++i)
  {
    v[i] = cam.params()[indices_[static_cast<size_t>(i)]];
  }
  return v;
}

void IntrinsicsParameterization::fromFree(
    const Eigen::VectorXd& free_params, CameraModel& cam) const
{
  ASSERT_EQUAL_(static_cast<int>(free_params.size()), dim());
  Eigen::VectorXd p = reference_.params();
  for (int i = 0; i < dim(); ++i)
  {
    p[indices_[static_cast<size_t>(i)]] = free_params[i];
  }
  if (shared_focal_)
  {
    p[1] = aspect_ * p[0];
  }
  if (cam.type() != reference_.type())
  {
    cam = reference_;
  }
  cam.setParams(p);
}
