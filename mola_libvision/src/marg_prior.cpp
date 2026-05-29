/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Schur-complement marginalization prior. Inspired by Basalt (BSD-3-Clause,
 * (c) 2019 Usenko, Demmel) MargHelper.
 * ------------------------------------------------------------------------- */
#include <mola_libvision/marg_prior.h>

#include <Eigen/Dense>
#include <set>

using namespace mola::vision;

MargResult mola::vision::schurMarginalize(
    const Eigen::MatrixXd& H, const Eigen::VectorXd& b, const std::vector<int>& marg_indices)
{
  MargResult res;
  const int  N = static_cast<int>(H.rows());
  if (H.cols() != N || b.size() != N || marg_indices.empty())
  {
    return res;
  }

  std::set<int> marg(marg_indices.begin(), marg_indices.end());
  for (int idx : marg)
  {
    if (idx < 0 || idx >= N)
    {
      return res;  // invalid index
    }
  }

  // Partition indices into kept (ascending) and marginalized (ascending).
  std::vector<int> keep;
  std::vector<int> drop(marg.begin(), marg.end());
  keep.reserve(N - static_cast<int>(marg.size()));
  for (int i = 0; i < N; ++i)
  {
    if (marg.count(i) == 0)
    {
      keep.push_back(i);
    }
  }
  if (keep.empty())
  {
    return res;
  }

  const int nk = static_cast<int>(keep.size());
  const int nm = static_cast<int>(drop.size());

  // Gather the sub-blocks.
  Eigen::MatrixXd Hkk(nk, nk), Hkm(nk, nm), Hmm(nm, nm);
  Eigen::VectorXd bk(nk), bm(nm);
  for (int a = 0; a < nk; ++a)
  {
    bk(a) = b(keep[a]);
    for (int c = 0; c < nk; ++c)
    {
      Hkk(a, c) = H(keep[a], keep[c]);
    }
    for (int c = 0; c < nm; ++c)
    {
      Hkm(a, c) = H(keep[a], drop[c]);
    }
  }
  for (int a = 0; a < nm; ++a)
  {
    bm(a) = b(drop[a]);
    for (int c = 0; c < nm; ++c)
    {
      Hmm(a, c) = H(drop[a], drop[c]);
    }
  }

  // Invert the marginal block (symmetric PD expected). Use LDLT; fall back to
  // a small ridge if needed for numerical robustness.
  Eigen::LDLT<Eigen::MatrixXd> ldlt(Hmm);
  if (ldlt.info() != Eigen::Success)
  {
    ldlt.compute(Hmm + 1e-9 * Eigen::MatrixXd::Identity(nm, nm));
    if (ldlt.info() != Eigen::Success)
    {
      return res;
    }
  }
  const Eigen::MatrixXd HmmInv_Hmk = ldlt.solve(Hkm.transpose());  // Hmm^-1 * Hmk
  const Eigen::VectorXd HmmInv_bm  = ldlt.solve(bm);  // Hmm^-1 * bm

  res.H  = Hkk - Hkm * HmmInv_Hmk;
  res.b  = bk - Hkm * HmmInv_bm;
  res.ok = true;
  return res;
}

bool MargPrior::build(
    const Eigen::MatrixXd& H, const Eigen::VectorXd& b, const std::vector<int>& var_ids,
    const std::vector<int>& marg_ids)
{
  valid_      = false;
  const int N = static_cast<int>(H.rows());
  if (static_cast<int>(var_ids.size()) != N)
  {
    return false;
  }

  std::set<int>    marg_set(marg_ids.begin(), marg_ids.end());
  std::vector<int> marg_indices;
  kept_ids_.clear();
  for (int i = 0; i < N; ++i)
  {
    if (marg_set.count(var_ids[i]) != 0)
    {
      marg_indices.push_back(i);
    }
    else
    {
      kept_ids_.push_back(var_ids[i]);
    }
  }

  const MargResult r = schurMarginalize(H, b, marg_indices);
  if (!r.ok)
  {
    kept_ids_.clear();
    return false;
  }
  H_     = r.H;
  b_     = r.b;
  valid_ = true;
  return true;
}
