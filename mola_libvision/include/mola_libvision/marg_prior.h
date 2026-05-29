/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <Eigen/Core>
#include <vector>

namespace mola::vision
{
/** Result of a Schur-complement marginalization: the reduced linear system
 *  (information matrix + vector) over the *kept* variables. */
struct MargResult
{
  Eigen::MatrixXd H;  ///< reduced information matrix over kept variables
  Eigen::VectorXd b;  ///< reduced information vector over kept variables
  bool            ok = false;
};

/** Marginalize a subset of variables from a linearized system (H, b) via the
 *  Schur complement, yielding a Gaussian prior on the remaining variables.
 *
 *  Given the quadratic 0.5 xᵀ H x + bᵀ x partitioned into kept (k) and
 *  marginalized (m) variables:
 *      H = [ Hkk  Hkm ;  Hmk  Hmm ],   b = [ bk ; bm ],
 *  the reduced prior is
 *      H' = Hkk - Hkm Hmm⁻¹ Hmk,     b' = bk - Hkm Hmm⁻¹ bm.
 *  Minimizing the reduced quadratic over the kept variables yields the same
 *  values they take at the minimum of the full system (the defining property of
 *  marginalization). This is how the information from dropped keyframes is
 *  retained as a prior on the active window.
 *
 *  \param H            Symmetric (N x N) information matrix.
 *  \param b            (N) information vector (gradient at the linearization).
 *  \param marg_indices Variable indices (rows/cols of H) to marginalize out.
 *  \return Reduced (H', b') over the kept variables, ordered by ascending
 *          original index; ok=false if `marg_indices` is invalid or the
 *          marginal block Hmm is not invertible.
 *
 *  \note Inspired by Basalt (BSD-3-Clause, (c) 2019 Usenko, Demmel) MargHelper.
 */
[[nodiscard]] MargResult schurMarginalize(
    const Eigen::MatrixXd& H, const Eigen::VectorXd& b, const std::vector<int>& marg_indices);

/** Holds a marginalization prior (reduced H, b) together with the global ids of
 *  the variables it constrains, so it can be re-applied as a quadratic cost on
 *  subsequent optimizations of the active window. */
class MargPrior
{
 public:
  MargPrior() = default;

  /** Build from a full system, marginalizing the variables whose global ids are
   *  in `marg_ids`. `var_ids` lists the global id of each variable index in H
   *  (size N). The kept ids are stored in ascending H-index order. */
  bool build(
      const Eigen::MatrixXd& H, const Eigen::VectorXd& b, const std::vector<int>& var_ids,
      const std::vector<int>& marg_ids);

  [[nodiscard]] bool                   valid() const { return valid_; }
  [[nodiscard]] const Eigen::MatrixXd& H() const { return H_; }
  [[nodiscard]] const Eigen::VectorXd& b() const { return b_; }

  /** Global ids of the variables this prior constrains (ascending H-index). */
  [[nodiscard]] const std::vector<int>& keptIds() const { return kept_ids_; }

  /** Quadratic prior cost at increment dx (over the kept variables, same order
   *  as keptIds): 0.5 dxᵀ H dx + bᵀ dx. */
  [[nodiscard]] double cost(const Eigen::VectorXd& dx) const
  {
    return 0.5 * dx.dot(H_ * dx) + b_.dot(dx);
  }

 private:
  Eigen::MatrixXd  H_;
  Eigen::VectorXd  b_;
  std::vector<int> kept_ids_;
  bool             valid_ = false;
};

}  // namespace mola::vision
