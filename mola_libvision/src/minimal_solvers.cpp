/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Algorithms:
 *  - Five-point essential matrix: D. Nister, "An efficient solution to the
 *    five-point relative pose problem", IEEE TPAMI 26(6), 2004.
 *  - Seven/eight-point fundamental matrix, DLT homography: Hartley and
 *    Zisserman, "Multiple View Geometry", 2nd ed., chapters 4 and 11.
 *  - P3P: J.A. Grunert (1841), as reviewed by Haralick et al., IJCV 13(3),
 *    1994; the quartic is built here by polynomial elimination.
 *  - GRIC: P.H.S. Torr, "Geometric motion segmentation and model selection",
 *    Phil. Trans. R. Soc. A, 1998.
 * ------------------------------------------------------------------------- */
#include <mola_libvision/lie_utils.h>
#include <mola_libvision/minimal_solvers.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>

using namespace mola::vision;

namespace
{
// ---------------------------------------------------------------------------
// Univariate polynomials, coefficients in ascending order of degree.
// ---------------------------------------------------------------------------
using UPoly = std::vector<double>;

UPoly upolyMul(const UPoly& a, const UPoly& b)
{
  if (a.empty() || b.empty())
  {
    return {};
  }
  UPoly c(a.size() + b.size() - 1, 0.0);
  for (size_t i = 0; i < a.size(); ++i)
  {
    for (size_t j = 0; j < b.size(); ++j)
    {
      c[i + j] += a[i] * b[j];
    }
  }
  return c;
}

UPoly upolyAdd(const UPoly& a, const UPoly& b, double sb = 1.0)
{
  UPoly c(std::max(a.size(), b.size()), 0.0);
  for (size_t i = 0; i < a.size(); ++i)
  {
    c[i] += a[i];
  }
  for (size_t i = 0; i < b.size(); ++i)
  {
    c[i] += sb * b[i];
  }
  return c;
}

UPoly upolyScale(const UPoly& a, double s)
{
  UPoly c = a;
  for (auto& v : c)
  {
    v *= s;
  }
  return c;
}

/** Real roots of a polynomial via the eigenvalues of its companion matrix. */
std::vector<double> realRoots(UPoly p)
{
  // Drop negligible leading coefficients.
  double max_abs = 0.0;
  for (double v : p)
  {
    max_abs = std::max(max_abs, std::abs(v));
  }
  if (max_abs == 0.0)
  {
    return {};
  }
  while (p.size() > 1 && std::abs(p.back()) < 1e-14 * max_abs)
  {
    p.pop_back();
  }
  const int n = static_cast<int>(p.size()) - 1;
  if (n < 1)
  {
    return {};
  }
  if (n == 1)
  {
    return {-p[0] / p[1]};
  }
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i)
  {
    C(0, i) = -p[static_cast<size_t>(n - 1 - i)] / p[static_cast<size_t>(n)];
  }
  for (int i = 1; i < n; ++i)
  {
    C(i, i - 1) = 1.0;
  }
  Eigen::EigenSolver<Eigen::MatrixXd> es(C, false);
  std::vector<double>                 roots;
  const auto&                         ev = es.eigenvalues();
  for (int i = 0; i < n; ++i)
  {
    const double re = ev[i].real();
    const double im = ev[i].imag();
    if (std::abs(im) <= 1e-8 * std::max(1.0, std::abs(re)))
    {
      roots.push_back(re);
    }
  }
  return roots;
}

double upolyEval(const UPoly& p, double x)
{
  double v = 0.0;
  for (size_t i = p.size(); i-- > 0;)
  {
    v = v * x + p[i];
  }
  return v;
}

// ---------------------------------------------------------------------------
// Trivariate polynomials of total degree <= 3, in Nister's monomial order:
//  0 x^3, 1 y^3, 2 x^2y, 3 xy^2, 4 x^2z, 5 x^2, 6 y^2z, 7 y^2, 8 xyz, 9 xy,
// 10 xz^2, 11 xz, 12 x, 13 yz^2, 14 yz, 15 y, 16 z^3, 17 z^2, 18 z, 19 1.
// ---------------------------------------------------------------------------
constexpr std::array<std::array<int, 3>, 20> kMonomials = {{
    {3, 0, 0}, {0, 3, 0}, {2, 1, 0}, {1, 2, 0}, {2, 0, 1}, {2, 0, 0}, {0, 2, 1},
    {0, 2, 0}, {1, 1, 1}, {1, 1, 0}, {1, 0, 2}, {1, 0, 1}, {1, 0, 0}, {0, 1, 2},
    {0, 1, 1}, {0, 1, 0}, {0, 0, 3}, {0, 0, 2}, {0, 0, 1}, {0, 0, 0},
}};

int monomialIndex(int a, int b, int c)
{
  for (int i = 0; i < 20; ++i)
  {
    const auto& m = kMonomials[static_cast<size_t>(i)];
    if (m[0] == a && m[1] == b && m[2] == c)
    {
      return i;
    }
  }
  return -1;
}

using TPoly = std::array<double, 20>;

TPoly tpolyMul(const TPoly& p, const TPoly& q)
{
  TPoly r{};
  for (int i = 0; i < 20; ++i)
  {
    if (p[static_cast<size_t>(i)] == 0.0)
    {
      continue;
    }
    for (int j = 0; j < 20; ++j)
    {
      if (q[static_cast<size_t>(j)] == 0.0)
      {
        continue;
      }
      const auto& a   = kMonomials[static_cast<size_t>(i)];
      const auto& b   = kMonomials[static_cast<size_t>(j)];
      const int   idx = monomialIndex(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
      if (idx >= 0)
      {
        r[static_cast<size_t>(idx)] += p[static_cast<size_t>(i)] * q[static_cast<size_t>(j)];
      }
    }
  }
  return r;
}

TPoly tpolyAdd(const TPoly& p, const TPoly& q, double sq = 1.0)
{
  TPoly r{};
  for (size_t i = 0; i < 20; ++i)
  {
    r[i] = p[i] + sq * q[i];
  }
  return r;
}

TPoly tpolyScale(const TPoly& p, double s)
{
  TPoly r = p;
  for (auto& v : r)
  {
    v *= s;
  }
  return r;
}

/** Hartley normalization: centroid to origin, mean distance sqrt(2). */
Eigen::Matrix3d normalizationTransform(const std::vector<Eigen::Vector2d>& p)
{
  Eigen::Vector2d c = Eigen::Vector2d::Zero();
  for (const auto& v : p)
  {
    c += v;
  }
  c /= static_cast<double>(p.size());
  double d = 0.0;
  for (const auto& v : p)
  {
    d += (v - c).norm();
  }
  d /= static_cast<double>(p.size());
  const double    s = d > 1e-12 ? std::sqrt(2.0) / d : 1.0;
  Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
  T(0, 0)           = s;
  T(1, 1)           = s;
  T(0, 2)           = -s * c.x();
  T(1, 2)           = -s * c.y();
  return T;
}

Eigen::Vector3d hom(const Eigen::Vector2d& p) { return {p.x(), p.y(), 1.0}; }

Eigen::Matrix3d reshapeRowMajor(const Eigen::Matrix<double, 9, 1>& v)
{
  Eigen::Matrix3d M;
  M << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
  return M;
}

Eigen::Matrix<double, 1, 9> epipolarRow(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
  // Row of the constraint b^T M a = 0 with M flattened row-major.
  Eigen::Matrix<double, 1, 9> r;
  r << b.x() * a.x(), b.x() * a.y(), b.x() * a.z(), b.y() * a.x(), b.y() * a.y(), b.y() * a.z(),
      b.z() * a.x(), b.z() * a.y(), b.z() * a.z();
  return r;
}

Eigen::Matrix3d projectToEssential(const Eigen::Matrix3d& M)
{
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d             s(1.0, 1.0, 0.0);
  return svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();
}

// Depths (lambda1, lambda2) of the closest points of two rays: camera 1 at the
// origin with direction b1, camera 2 at -R^T t with direction R^T b2.
bool twoViewDepths(
    const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::Vector3d& b1,
    const Eigen::Vector3d& b2, double& l1, double& l2)
{
  // Solve l1 R u1 - l2 u2 = -t in the least-squares sense, on unit rays so
  // the conditioning test does not depend on the bearing norms (the depths
  // are then along unit rays, which keeps their signs).
  const Eigen::Vector3d       u1 = b1.normalized();
  const Eigen::Vector3d       u2 = b2.normalized();
  Eigen::Matrix<double, 3, 2> A;
  A.col(0)                  = R * u1;
  A.col(1)                  = -u2;
  const Eigen::Matrix2d AtA = A.transpose() * A;
  const double          det = AtA.determinant();
  if (std::abs(det) < 1e-14)
  {
    return false;
  }
  const Eigen::Vector2d l = AtA.inverse() * (A.transpose() * (-t));
  l1                      = l.x();
  l2                      = l.y();
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Five-point essential matrix
// ---------------------------------------------------------------------------
std::vector<Eigen::Matrix3d> mola::vision::solveEssentialFivePoint(
    const Eigen::Vector3d b1[5], const Eigen::Vector3d b2[5])
{
  Eigen::Matrix<double, 5, 9> A;
  for (int i = 0; i < 5; ++i)
  {
    A.row(i) = epipolarRow(b1[i], b2[i]);
  }
  // Null space (4-dim) from the full SVD.
  Eigen::JacobiSVD<Eigen::Matrix<double, 5, 9>> svd(A, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 9>             V = svd.matrixV();
  const Eigen::Matrix3d                         X = reshapeRowMajor(V.col(5));
  const Eigen::Matrix3d                         Y = reshapeRowMajor(V.col(6));
  const Eigen::Matrix3d                         Z = reshapeRowMajor(V.col(7));
  const Eigen::Matrix3d                         W = reshapeRowMajor(V.col(8));

  // E as a matrix of linear polynomials x X + y Y + z Z + W.
  std::array<std::array<TPoly, 3>, 3> E;
  const int                           ix = monomialIndex(1, 0, 0);
  const int                           iy = monomialIndex(0, 1, 0);
  const int                           iz = monomialIndex(0, 0, 1);
  const int                           i1 = monomialIndex(0, 0, 0);
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      TPoly p{};
      p[static_cast<size_t>(ix)]                        = X(r, c);
      p[static_cast<size_t>(iy)]                        = Y(r, c);
      p[static_cast<size_t>(iz)]                        = Z(r, c);
      p[static_cast<size_t>(i1)]                        = W(r, c);
      E[static_cast<size_t>(r)][static_cast<size_t>(c)] = p;
    }
  }
  auto e = [&](int r, int c) -> const TPoly&
  { return E[static_cast<size_t>(r)][static_cast<size_t>(c)]; };

  Eigen::Matrix<double, 10, 20> M;
  // det(E) = 0
  {
    const TPoly m0  = tpolyAdd(tpolyMul(e(1, 1), e(2, 2)), tpolyMul(e(1, 2), e(2, 1)), -1.0);
    const TPoly m1  = tpolyAdd(tpolyMul(e(1, 0), e(2, 2)), tpolyMul(e(1, 2), e(2, 0)), -1.0);
    const TPoly m2  = tpolyAdd(tpolyMul(e(1, 0), e(2, 1)), tpolyMul(e(1, 1), e(2, 0)), -1.0);
    TPoly       det = tpolyMul(e(0, 0), m0);
    det             = tpolyAdd(det, tpolyMul(e(0, 1), m1), -1.0);
    det             = tpolyAdd(det, tpolyMul(e(0, 2), m2));
    for (int k = 0; k < 20; ++k)
    {
      M(0, k) = det[static_cast<size_t>(k)];
    }
  }
  // 2 E E^T E - tr(E E^T) E = 0
  {
    std::array<std::array<TPoly, 3>, 3> EEt;
    for (int r = 0; r < 3; ++r)
    {
      for (int c = 0; c < 3; ++c)
      {
        TPoly acc{};
        for (int k = 0; k < 3; ++k)
        {
          acc = tpolyAdd(acc, tpolyMul(e(r, k), e(c, k)));
        }
        EEt[static_cast<size_t>(r)][static_cast<size_t>(c)] = acc;
      }
    }
    const TPoly tr  = tpolyAdd(tpolyAdd(EEt[0][0], EEt[1][1]), EEt[2][2]);
    int         row = 1;
    for (int r = 0; r < 3; ++r)
    {
      for (int c = 0; c < 3; ++c)
      {
        TPoly acc{};
        for (int k = 0; k < 3; ++k)
        {
          acc =
              tpolyAdd(acc, tpolyMul(EEt[static_cast<size_t>(r)][static_cast<size_t>(k)], e(k, c)));
        }
        acc = tpolyAdd(tpolyScale(acc, 2.0), tpolyMul(tr, e(r, c)), -1.0);
        for (int k = 0; k < 20; ++k)
        {
          M(row, k) = acc[static_cast<size_t>(k)];
        }
        ++row;
      }
    }
  }

  // Gauss-Jordan elimination of the first 10 columns, partial pivoting.
  for (int col = 0; col < 10; ++col)
  {
    int    piv  = col;
    double best = std::abs(M(col, col));
    for (int r = col + 1; r < 10; ++r)
    {
      if (std::abs(M(r, col)) > best)
      {
        best = std::abs(M(r, col));
        piv  = r;
      }
    }
    if (best < 1e-12)
    {
      return {};
    }
    if (piv != col)
    {
      M.row(piv).swap(M.row(col));
    }
    M.row(col) /= M(col, col);
    for (int r = 0; r < 10; ++r)
    {
      if (r != col && M(r, col) != 0.0)
      {
        M.row(r) -= M(r, col) * M.row(col);
      }
    }
  }

  // Rows 4..9 have leading monomials x^2z, x^2, y^2z, y^2, xyz, xy. Eliminate
  // the leading terms pairwise to get three equations linear in (x, y, 1)
  // with polynomial coefficients in z.
  auto rowPolys = [&](int ra, int rb)
  {
    const auto a = M.row(ra);
    const auto b = M.row(rb);
    // Basis columns: 10 xz^2, 11 xz, 12 x, 13 yz^2, 14 yz, 15 y,
    //                16 z^3, 17 z^2, 18 z, 19 1.
    std::array<UPoly, 3> out;
    out[0] = {a(12), a(11) - b(12), a(10) - b(11), -b(10)};
    out[1] = {a(15), a(14) - b(15), a(13) - b(14), -b(13)};
    out[2] = {a(19), a(18) - b(19), a(17) - b(18), a(16) - b(17), -b(16)};
    return out;
  };
  const auto K = rowPolys(4, 5);
  const auto L = rowPolys(6, 7);
  const auto N = rowPolys(8, 9);

  // det of [K; L; N] as a degree-10 polynomial in z.
  const UPoly c0  = upolyAdd(upolyMul(L[1], N[2]), upolyMul(L[2], N[1]), -1.0);
  const UPoly c1  = upolyAdd(upolyMul(L[0], N[2]), upolyMul(L[2], N[0]), -1.0);
  const UPoly c2  = upolyAdd(upolyMul(L[0], N[1]), upolyMul(L[1], N[0]), -1.0);
  UPoly       det = upolyMul(K[0], c0);
  det             = upolyAdd(det, upolyMul(K[1], c1), -1.0);
  det             = upolyAdd(det, upolyMul(K[2], c2));

  std::vector<Eigen::Matrix3d> solutions;
  for (const double z : realRoots(det))
  {
    Eigen::Matrix3d B;
    for (int c = 0; c < 3; ++c)
    {
      B(0, c) = upolyEval(K[static_cast<size_t>(c)], z);
      B(1, c) = upolyEval(L[static_cast<size_t>(c)], z);
      B(2, c) = upolyEval(N[static_cast<size_t>(c)], z);
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svdB(B, Eigen::ComputeFullV);
    const Eigen::Vector3d             v = svdB.matrixV().col(2);
    if (std::abs(v.z()) < 1e-12)
    {
      continue;
    }
    const double    x  = v.x() / v.z();
    const double    y  = v.y() / v.z();
    Eigen::Matrix3d Es = x * X + y * Y + z * Z + W;
    const double    n  = Es.norm();
    if (!(n > 0.0) || !std::isfinite(n))
    {
      continue;
    }
    solutions.push_back(Es / n);
  }
  return solutions;
}

// ---------------------------------------------------------------------------
// Fundamental matrix
// ---------------------------------------------------------------------------
std::vector<Eigen::Matrix3d> mola::vision::solveFundamentalSevenPoint(
    const Eigen::Vector2d p1[7], const Eigen::Vector2d p2[7])
{
  const std::vector<Eigen::Vector2d> v1(p1, p1 + 7);
  const std::vector<Eigen::Vector2d> v2(p2, p2 + 7);
  const Eigen::Matrix3d              T1 = normalizationTransform(v1);
  const Eigen::Matrix3d              T2 = normalizationTransform(v2);

  Eigen::Matrix<double, 7, 9> A;
  for (int i = 0; i < 7; ++i)
  {
    A.row(i) = epipolarRow(T1 * hom(p1[i]), T2 * hom(p2[i]));
  }
  Eigen::JacobiSVD<Eigen::Matrix<double, 7, 9>> svd(A, Eigen::ComputeFullV);
  const Eigen::Matrix3d                         F1 = reshapeRowMajor(svd.matrixV().col(7));
  const Eigen::Matrix3d                         F2 = reshapeRowMajor(svd.matrixV().col(8));

  // det(a F1 + (1 - a) F2) is a cubic in a: fit it from 4 samples.
  Eigen::Matrix4d Vm;
  Eigen::Vector4d d;
  const double    samples[4] = {-1.0, 0.0, 1.0, 2.0};
  for (int i = 0; i < 4; ++i)
  {
    const double a = samples[i];
    Vm.row(i) << 1.0, a, a * a, a * a * a;
    d[i] = (a * F1 + (1.0 - a) * F2).determinant();
  }
  const Eigen::Vector4d coef = Vm.colPivHouseholderQr().solve(d);

  std::vector<Eigen::Matrix3d> out;
  for (const double a : realRoots({coef[0], coef[1], coef[2], coef[3]}))
  {
    Eigen::Matrix3d F = T2.transpose() * (a * F1 + (1.0 - a) * F2) * T1;
    const double    n = F.norm();
    if (n > 0.0 && std::isfinite(n))
    {
      out.push_back(F / n);
    }
  }
  return out;
}

std::optional<Eigen::Matrix3d> mola::vision::solveFundamentalLinear(
    const std::vector<Eigen::Vector2d>& p1, const std::vector<Eigen::Vector2d>& p2)
{
  const size_t N = p1.size();
  if (N < 8 || p2.size() != N)
  {
    return std::nullopt;
  }
  const Eigen::Matrix3d T1 = normalizationTransform(p1);
  const Eigen::Matrix3d T2 = normalizationTransform(p2);
  Eigen::MatrixXd       A(static_cast<Eigen::Index>(N), 9);
  for (size_t i = 0; i < N; ++i)
  {
    A.row(static_cast<Eigen::Index>(i)) = epipolarRow(T1 * hom(p1[i]), T2 * hom(p2[i]));
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  Eigen::Matrix3d                   F = reshapeRowMajor(svd.matrixV().col(8));
  // Rank 2.
  Eigen::JacobiSVD<Eigen::Matrix3d> s2(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d                   sv = s2.singularValues();
  sv[2]                                = 0.0;
  F                                    = s2.matrixU() * sv.asDiagonal() * s2.matrixV().transpose();
  F                                    = T2.transpose() * F * T1;
  const double n                       = F.norm();
  if (!(n > 0.0) || !std::isfinite(n))
  {
    return std::nullopt;
  }
  return F / n;
}

std::optional<Eigen::Matrix3d> mola::vision::solveEssentialLinear(
    const std::vector<Eigen::Vector3d>& b1, const std::vector<Eigen::Vector3d>& b2)
{
  const size_t N = b1.size();
  if (N < 8 || b2.size() != N)
  {
    return std::nullopt;
  }
  Eigen::MatrixXd A(static_cast<Eigen::Index>(N), 9);
  for (size_t i = 0; i < N; ++i)
  {
    A.row(static_cast<Eigen::Index>(i)) = epipolarRow(b1[i].normalized(), b2[i].normalized());
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::Matrix3d             E = projectToEssential(reshapeRowMajor(svd.matrixV().col(8)));
  if (!E.allFinite())
  {
    return std::nullopt;
  }
  return E / E.norm();
}

// ---------------------------------------------------------------------------
// Homography and rotation
// ---------------------------------------------------------------------------
std::optional<Eigen::Matrix3d> mola::vision::solveHomographyDLT(
    const std::vector<Eigen::Vector2d>& p1, const std::vector<Eigen::Vector2d>& p2)
{
  const size_t N = p1.size();
  if (N < 4 || p2.size() != N)
  {
    return std::nullopt;
  }
  const Eigen::Matrix3d T1 = normalizationTransform(p1);
  const Eigen::Matrix3d T2 = normalizationTransform(p2);
  Eigen::MatrixXd       A(static_cast<Eigen::Index>(2 * N), 9);
  for (size_t i = 0; i < N; ++i)
  {
    const Eigen::Vector3d a = T1 * hom(p1[i]);
    const Eigen::Vector3d b = T2 * hom(p2[i]);
    const auto            r = static_cast<Eigen::Index>(2 * i);
    A.row(r) << 0, 0, 0, -b.z() * a.x(), -b.z() * a.y(), -b.z() * a.z(), b.y() * a.x(),
        b.y() * a.y(), b.y() * a.z();
    A.row(r + 1) << b.z() * a.x(), b.z() * a.y(), b.z() * a.z(), 0, 0, 0, -b.x() * a.x(),
        -b.x() * a.y(), -b.x() * a.z();
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::Matrix3d             Hn = reshapeRowMajor(svd.matrixV().col(8));
  Eigen::Matrix3d                   H  = T2.inverse() * Hn * T1;
  if (!H.allFinite() || std::abs(H(2, 2)) < 1e-15)
  {
    if (!H.allFinite() || H.norm() == 0.0)
    {
      return std::nullopt;
    }
    return H / H.norm();
  }
  return H / H(2, 2);
}

std::optional<Eigen::Matrix3d> mola::vision::solveRotationKabsch(
    const std::vector<Eigen::Vector3d>& b1, const std::vector<Eigen::Vector3d>& b2)
{
  const size_t N = b1.size();
  if (N < 2 || b2.size() != N)
  {
    return std::nullopt;
  }
  Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < N; ++i)
  {
    S += b2[i].normalized() * b1[i].normalized().transpose();
  }
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d                   D = Eigen::Matrix3d::Identity();
  D(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0 ? -1.0 : 1.0;
  const Eigen::Matrix3d R = svd.matrixU() * D * svd.matrixV().transpose();
  if (!R.allFinite())
  {
    return std::nullopt;
  }
  return R;
}

// ---------------------------------------------------------------------------
// Absolute pose
// ---------------------------------------------------------------------------
std::optional<AbsolutePose> mola::vision::alignPointSets(
    const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& q)
{
  const size_t N = p.size();
  if (N < 3 || q.size() != N)
  {
    return std::nullopt;
  }
  Eigen::Vector3d cp = Eigen::Vector3d::Zero();
  Eigen::Vector3d cq = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < N; ++i)
  {
    cp += p[i];
    cq += q[i];
  }
  cp /= static_cast<double>(N);
  cq /= static_cast<double>(N);
  Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < N; ++i)
  {
    S += (q[i] - cq) * (p[i] - cp).transpose();
  }
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d                   D = Eigen::Matrix3d::Identity();
  D(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0 ? -1.0 : 1.0;
  AbsolutePose out;
  out.R = svd.matrixU() * D * svd.matrixV().transpose();
  out.t = cq - out.R * cp;
  if (!out.R.allFinite() || !out.t.allFinite())
  {
    return std::nullopt;
  }
  return out;
}

namespace
{
std::vector<AbsolutePose> p3pChart(
    const Eigen::Vector3d bearings[3], const Eigen::Vector3d points[3], bool& degenerate)
{
  const Eigen::Vector3d j1 = bearings[0].normalized();
  const Eigen::Vector3d j2 = bearings[1].normalized();
  const Eigen::Vector3d j3 = bearings[2].normalized();

  const double a2 = (points[1] - points[2]).squaredNorm();
  const double b2 = (points[0] - points[2]).squaredNorm();
  const double c2 = (points[0] - points[1]).squaredNorm();
  if (a2 < 1e-18 || b2 < 1e-18 || c2 < 1e-18)
  {
    return {};
  }
  const double ca = j2.dot(j3);
  const double cb = j1.dot(j3);
  const double cg = j1.dot(j2);

  // s2 = u s1, s3 = v s1. With q(v) = 1 + v^2 - 2 v cb, the law of cosines
  // gives u = N(v) / D(v), and substituting it into the (1, 2) equation yields
  // a quartic in v.
  const UPoly q  = {1.0, -2.0 * cb, 1.0};
  const UPoly Np = upolyAdd(upolyScale(q, a2 - c2), UPoly{b2, 0.0, -b2});  // (a2-c2) q - b2 (v^2-1)
  const UPoly Dp = {2.0 * b2 * cg, -2.0 * b2 * ca};
  const UPoly D2 = upolyMul(Dp, Dp);
  UPoly       quartic = upolyAdd(D2, upolyMul(Np, Np));
  quartic             = upolyAdd(quartic, upolyScale(upolyMul(Np, Dp), -2.0 * cg));
  quartic             = upolyScale(quartic, b2);
  quartic             = upolyAdd(quartic, upolyScale(upolyMul(q, D2), c2), -1.0);

  std::vector<AbsolutePose> out;
  degenerate = false;
  // With cos(alpha) = cos(gamma) = 0 the denominator vanishes identically and
  // the quartic reduces to N(v)^2: take the roots of N directly.
  const bool d_zero = std::abs(Dp[0]) + std::abs(Dp[1]) < 1e-12 * b2;
  const auto roots  = d_zero ? realRoots(Np) : realRoots(quartic);
  for (const double v : roots)
  {
    if (v <= 0.0)
    {
      continue;
    }
    const double qv = upolyEval(q, v);
    if (qv <= 1e-15)
    {
      continue;
    }
    // u from the elimination, or, where it is undetermined, from the (1, 2)
    // equation (a quadratic in u) checked against the (2, 3) equation.
    std::vector<double> us;
    const double        den = upolyEval(Dp, v);
    if (std::abs(den) > 1e-10 * b2)
    {
      us.push_back(upolyEval(Np, v) / den);
    }
    else
    {
      degenerate        = true;
      const double disc = cg * cg - 1.0 + c2 * qv / b2;
      if (disc >= 0.0)
      {
        for (const double u : {cg + std::sqrt(disc), cg - std::sqrt(disc)})
        {
          const double r = b2 * (u * u + v * v - 2.0 * u * v * ca) - a2 * qv;
          if (std::abs(r) <= 1e-6 * std::max(a2 * qv, b2))
          {
            us.push_back(u);
          }
        }
      }
    }
    for (const double u : us)
    {
      if (u <= 0.0)
      {
        continue;
      }
      const double s1 = std::sqrt(b2 / qv);
      const double s2 = u * s1;
      const double s3 = v * s1;

      const std::vector<Eigen::Vector3d> P = {points[0], points[1], points[2]};
      const std::vector<Eigen::Vector3d> Q = {s1 * j1, s2 * j2, s3 * j3};
      if (const auto pose = alignPointSets(P, Q); pose)
      {
        out.push_back(*pose);
      }
    }
  }
  return out;
}

}  // namespace

std::vector<AbsolutePose> mola::vision::solveP3P(
    const Eigen::Vector3d bearings[3], const Eigen::Vector3d points[3])
{
  // The elimination divides by a term that vanishes for some configurations
  // (e.g. symmetric ones); relabeling the points changes that term, so the
  // other cyclic orders recover the lost solutions.
  std::vector<AbsolutePose> out;
  for (int shift = 0; shift < 3; ++shift)
  {
    const Eigen::Vector3d b[3] = {
        bearings[shift], bearings[(shift + 1) % 3], bearings[(shift + 2) % 3]};
    const Eigen::Vector3d X[3] = {points[shift], points[(shift + 1) % 3], points[(shift + 2) % 3]};
    bool                  degenerate = false;
    for (const auto& sol : p3pChart(b, X, degenerate))
    {
      const bool dup = std::any_of(
          out.begin(), out.end(),
          [&](const AbsolutePose& o)
          { return (o.R - sol.R).norm() + (o.t - sol.t).norm() < 1e-9; });
      if (!dup)
      {
        out.push_back(sol);
      }
    }
    if (!degenerate)
    {
      break;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Decomposition, triangulation, errors
// ---------------------------------------------------------------------------
std::optional<RelativePose> mola::vision::relativePoseFromEssential(
    const Eigen::Matrix3d& E, const std::vector<Eigen::Vector3d>& b1,
    const std::vector<Eigen::Vector3d>& b2, const std::vector<bool>* mask)
{
  if (b1.size() != b2.size() || (mask != nullptr && mask->size() != b1.size()))
  {
    return std::nullopt;
  }
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d                   U = svd.matrixU();
  Eigen::Matrix3d                   V = svd.matrixV();
  if (U.determinant() < 0)
  {
    U.col(2) *= -1.0;
  }
  if (V.determinant() < 0)
  {
    V.col(2) *= -1.0;
  }
  Eigen::Matrix3d Wm;
  Wm << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  const Eigen::Matrix3d Ra = U * Wm * V.transpose();
  const Eigen::Matrix3d Rb = U * Wm.transpose() * V.transpose();
  const Eigen::Vector3d tu = U.col(2);

  const std::array<std::pair<Eigen::Matrix3d, Eigen::Vector3d>, 4> cands = {
      {{Ra, tu}, {Ra, -tu}, {Rb, tu}, {Rb, -tu}}};

  RelativePose best;
  best.num_in_front = -1;
  for (const auto& [R, t] : cands)
  {
    int count = 0;
    for (size_t i = 0; i < b1.size(); ++i)
    {
      if (mask != nullptr && !(*mask)[i])
      {
        continue;
      }
      double l1 = 0;
      double l2 = 0;
      if (twoViewDepths(R, t, b1[i], b2[i], l1, l2) && l1 > 0 && l2 > 0)
      {
        ++count;
      }
    }
    if (count > best.num_in_front)
    {
      best.R            = R;
      best.t            = t;
      best.num_in_front = count;
    }
  }
  if (best.num_in_front <= 0)
  {
    return std::nullopt;
  }
  return best;
}

std::optional<Eigen::Vector3d> mola::vision::triangulateBearings(
    const std::vector<Eigen::Matrix3d>& R_cw, const std::vector<Eigen::Vector3d>& t_cw,
    const std::vector<Eigen::Vector3d>& bearings, double min_angle)
{
  const size_t N = bearings.size();
  if (N < 2 || R_cw.size() != N || t_cw.size() != N)
  {
    return std::nullopt;
  }
  std::vector<Eigen::Vector3d> C(N);
  std::vector<Eigen::Vector3d> d(N);
  Eigen::Matrix3d              A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d              b = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < N; ++i)
  {
    C[i]                    = -R_cw[i].transpose() * t_cw[i];
    d[i]                    = (R_cw[i].transpose() * bearings[i]).normalized();
    const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d[i] * d[i].transpose();
    A += P;
    b += P * C[i];
  }
  // The largest angle between rays bounds how well depth is determined.
  double max_angle = 0.0;
  for (size_t i = 0; i < N; ++i)
  {
    for (size_t j = i + 1; j < N; ++j)
    {
      max_angle = std::max(max_angle, angleBetween(d[i], d[j]));
    }
  }
  if (max_angle < std::max(min_angle, 1e-9))
  {
    return std::nullopt;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(A);
  if (es.eigenvalues()[0] < 1e-12 * std::max(1.0, es.eigenvalues()[2]))
  {
    return std::nullopt;
  }
  Eigen::Vector3d X = A.ldlt().solve(b);

  // Gauss-Newton refinement of the angular error: residual = component of the
  // predicted ray orthogonal to the observed bearing.
  for (int it = 0; it < 5; ++it)
  {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < N; ++i)
    {
      const Eigen::Vector3d p  = R_cw[i] * X + t_cw[i];
      const double          np = p.norm();
      if (np < 1e-12)
      {
        return std::nullopt;
      }
      const Eigen::Vector3d u  = p / np;
      const Eigen::Vector3d bi = bearings[i].normalized();
      // r = u - bi (small-angle residual); dr/dX = (I - u u^T)/|p| R.
      const Eigen::Vector3d r = u - bi;
      const Eigen::Matrix3d J = (Eigen::Matrix3d::Identity() - u * u.transpose()) / np * R_cw[i];
      H += J.transpose() * J;
      g += J.transpose() * r;
    }
    const Eigen::Vector3d dx = -H.ldlt().solve(g);
    if (!dx.allFinite())
    {
      break;
    }
    X += dx;
    if (dx.norm() < 1e-12 * std::max(1.0, X.norm()))
    {
      break;
    }
  }

  for (size_t i = 0; i < N; ++i)
  {
    const Eigen::Vector3d p = R_cw[i] * X + t_cw[i];
    if (p.dot(bearings[i]) <= 0.0)
    {
      return std::nullopt;
    }
  }
  return X;
}

double mola::vision::triangulationAngle(
    const Eigen::Vector3d& X, const std::vector<Eigen::Vector3d>& camera_centers)
{
  double best = 0.0;
  for (size_t i = 0; i < camera_centers.size(); ++i)
  {
    for (size_t j = i + 1; j < camera_centers.size(); ++j)
    {
      best = std::max(best, angleBetween(X - camera_centers[i], X - camera_centers[j]));
    }
  }
  return best;
}

double mola::vision::angleBetween(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
  return std::atan2(a.cross(b).norm(), a.dot(b));
}

double mola::vision::epipolarAngularError(
    const Eigen::Matrix3d& E, const Eigen::Vector3d& b1, const Eigen::Vector3d& b2)
{
  const Eigen::Vector3d u1 = b1.normalized();
  const Eigen::Vector3d u2 = b2.normalized();
  const Eigen::Vector3d n2 = E * u1;  // epipolar plane normal in camera 2
  const Eigen::Vector3d n1 = E.transpose() * u2;
  const double          l2 = n2.norm();
  const double          l1 = n1.norm();
  if (l1 < 1e-15 || l2 < 1e-15)
  {
    return 0.0;
  }
  const double e2 = std::abs(u2.dot(n2)) / l2;
  const double e1 = std::abs(u1.dot(n1)) / l1;
  return 0.5 * (std::asin(std::min(1.0, e1)) + std::asin(std::min(1.0, e2)));
}

double mola::vision::sampsonDistance(
    const Eigen::Matrix3d& F, const Eigen::Vector2d& p1, const Eigen::Vector2d& p2)
{
  const Eigen::Vector3d x1  = hom(p1);
  const Eigen::Vector3d x2  = hom(p2);
  const Eigen::Vector3d Fx1 = F * x1;
  const Eigen::Vector3d Ftx = F.transpose() * x2;
  const double          num = x2.dot(Fx1);
  const double den = Fx1.x() * Fx1.x() + Fx1.y() * Fx1.y() + Ftx.x() * Ftx.x() + Ftx.y() * Ftx.y();
  if (den < 1e-30)
  {
    return std::abs(num) > 0 ? 1e10 : 0.0;
  }
  return std::abs(num) / std::sqrt(den);
}

double mola::vision::homographyTransferError(
    const Eigen::Matrix3d& H, const Eigen::Vector2d& p1, const Eigen::Vector2d& p2)
{
  const Eigen::Vector3d q2 = H * hom(p1);
  const Eigen::Vector3d q1 = H.inverse() * hom(p2);
  if (std::abs(q2.z()) < 1e-15 || std::abs(q1.z()) < 1e-15)
  {
    return 1e10;
  }
  const double e2 = (q2.head<2>() / q2.z() - p2).norm();
  const double e1 = (q1.head<2>() / q1.z() - p1).norm();
  return 0.5 * (e1 + e2);
}

double mola::vision::gric(const std::vector<double>& residuals, double sigma, int d, int k, int r)
{
  const double n      = static_cast<double>(residuals.size());
  const double cap    = 2.0 * (r - d);
  double       sum    = 0.0;
  const double inv_s2 = 1.0 / (sigma * sigma);
  for (const double e : residuals)
  {
    sum += std::min(e * e * inv_s2, cap);
  }
  return sum + n * d * std::log(static_cast<double>(r)) + k * std::log(r * n);
}
