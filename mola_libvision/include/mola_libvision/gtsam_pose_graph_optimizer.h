/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_libvision/pose_graph_optimizer.h>

#include <memory>

namespace mola::vision
{
/** Concrete SE(3) pose-graph optimizer backed by GTSAM
 *  (`gtsam::BetweenFactor<Pose3>` + Levenberg-Marquardt). Loop-closure edges
 *  can use a robust M-estimator (Huber) so a few wrong loop matches do not wreck
 *  the global solution; odometry edges stay least-squares. No GTSAM symbols are
 *  exposed in this header (pimpl), so consumers need not depend on GTSAM.
 *
 *  Edge information matrices use GTSAM's Pose3 tangent ordering
 *  [rx ry rz x y z] (rotation first). Gauge is anchored with a tight prior on
 *  each fixed node (at least one node must be fixed).
 *
 *  Use it to correct global drift after a verified loop closure: add keyframe
 *  poses as nodes, sequential relative poses as odometry edges, the loop as a
 *  loop edge (is_loop=true), fix the first node, and optimize().
 */
class GtsamPoseGraphOptimizer : public PoseGraphOptimizer
{
 public:
  GtsamPoseGraphOptimizer();
  ~GtsamPoseGraphOptimizer() override;

  void addNode(int id, const mrpt::poses::CPose3D& pose) override;
  void setNodeFixed(int id, bool fixed) override;
  void addEdge(const PoseGraphEdge& edge) override;
  void optimize(int max_iters = 10) override;

  [[nodiscard]] mrpt::poses::CPose3D nodePose(int id) const override;
  [[nodiscard]] std::size_t          numNodes() const override;
  void                               clear() override;

  /** Huber kernel threshold (in whitened residual units) applied to
   *  loop-closure edges. <= 0 disables robustness (pure least squares).
   *  Default 1.345 (95% efficiency for a unit-variance Gaussian). */
  void setLoopHuberThreshold(double k);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mola::vision
