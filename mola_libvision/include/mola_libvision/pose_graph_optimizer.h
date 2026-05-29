/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/poses/CPose3D.h>

#include <Eigen/Core>
#include <memory>

namespace mola::vision
{
/** A relative-pose constraint between two pose-graph nodes:
 *  node `to` expressed relative to node `from`, i.e. the measurement is
 *  T_from_to with pose(to) = pose(from) (+) relative_pose. */
struct PoseGraphEdge
{
  int                         from_id = -1;
  int                         to_id   = -1;
  mrpt::poses::CPose3D        relative_pose;
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
  bool                        is_loop     = false;  ///< loop-closure vs. odometry edge
};

/** Abstract SE(3) pose-graph optimizer interface.
 *
 *  Concrete back ends (e.g. GTSAM iSAM2, g2o, or MRPT's own graph-SLAM) optimize
 *  the node poses to best satisfy the relative-pose edges, anchoring the gauge
 *  via at least one fixed node. Used to correct drift after a verified loop
 *  closure.
 *
 *  This is the abstract base only; the concrete implementation is deferred to
 *  the loop-closure phase.
 */
class PoseGraphOptimizer
{
 public:
  using Ptr = std::shared_ptr<PoseGraphOptimizer>;

  PoseGraphOptimizer()                                     = default;
  PoseGraphOptimizer(const PoseGraphOptimizer&)            = default;
  PoseGraphOptimizer(PoseGraphOptimizer&&)                 = default;
  PoseGraphOptimizer& operator=(const PoseGraphOptimizer&) = default;
  PoseGraphOptimizer& operator=(PoseGraphOptimizer&&)      = default;
  virtual ~PoseGraphOptimizer()                            = default;

  /** Add (or update) a node with an initial pose estimate. */
  virtual void addNode(int id, const mrpt::poses::CPose3D& pose) = 0;

  /** Hold a node fixed (gauge anchor). At least one node must be fixed. */
  virtual void setNodeFixed(int id, bool fixed) = 0;

  /** Add a relative-pose constraint. */
  virtual void addEdge(const PoseGraphEdge& edge) = 0;

  /** Run the optimization for up to `max_iters` iterations. */
  virtual void optimize(int max_iters = 10) = 0;

  /** Current (optimized) pose of a node. */
  [[nodiscard]] virtual mrpt::poses::CPose3D nodePose(int id) const = 0;

  /** Number of nodes currently in the graph. */
  [[nodiscard]] virtual std::size_t numNodes() const = 0;

  /** Remove all nodes and edges. */
  virtual void clear() = 0;
};

}  // namespace mola::vision
