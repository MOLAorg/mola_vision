/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <mola_libvision/gtsam_pose_graph_optimizer.h>
#include <mrpt/poses/gtsam_wrappers.h>

#include <set>

using namespace mola::vision;

struct GtsamPoseGraphOptimizer::Impl
{
  gtsam::NonlinearFactorGraph graph;  ///< between-factors only (priors added at optimize)
  gtsam::Values               values;
  std::set<gtsam::Key>        keys;
  std::set<gtsam::Key>        fixed;
  double                      loop_huber = 1.345;
};

GtsamPoseGraphOptimizer::GtsamPoseGraphOptimizer() : impl_(std::make_unique<Impl>()) {}
GtsamPoseGraphOptimizer::~GtsamPoseGraphOptimizer() = default;

void GtsamPoseGraphOptimizer::setLoopHuberThreshold(double k) { impl_->loop_huber = k; }

void GtsamPoseGraphOptimizer::addNode(int id, const mrpt::poses::CPose3D& pose)
{
  const auto key = static_cast<gtsam::Key>(id);
  const auto p   = mrpt::gtsam_wrappers::toPose3(pose);
  if (impl_->values.exists(key))
  {
    impl_->values.update(key, p);
  }
  else
  {
    impl_->values.insert(key, p);
    impl_->keys.insert(key);
  }
}

void GtsamPoseGraphOptimizer::setNodeFixed(int id, bool fixed)
{
  const auto key = static_cast<gtsam::Key>(id);
  if (fixed)
  {
    impl_->fixed.insert(key);
  }
  else
  {
    impl_->fixed.erase(key);
  }
}

void GtsamPoseGraphOptimizer::addEdge(const PoseGraphEdge& edge)
{
  const auto from = static_cast<gtsam::Key>(edge.from_id);
  const auto to   = static_cast<gtsam::Key>(edge.to_id);

  gtsam::Matrix6 info;
  for (int r = 0; r < 6; ++r)
  {
    for (int c = 0; c < 6; ++c)
    {
      info(r, c) = edge.information(r, c);
    }
  }
  gtsam::SharedNoiseModel noise = gtsam::noiseModel::Gaussian::Information(info);

  // Robustify loop-closure edges with a Huber M-estimator so a few wrong loops
  // do not corrupt the global solution (odometry edges stay least-squares).
  if (edge.is_loop && impl_->loop_huber > 0.0)
  {
    noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(impl_->loop_huber), noise);
  }

  impl_->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      from, to, mrpt::gtsam_wrappers::toPose3(edge.relative_pose), noise);
}

void GtsamPoseGraphOptimizer::optimize(int max_iters)
{
  if (impl_->values.size() < 2 || impl_->graph.empty())
  {
    return;
  }

  // Build a working graph = between-factors + tight priors on the fixed nodes
  // (gauge). If nothing was fixed, anchor the lowest-id node.
  gtsam::NonlinearFactorGraph g       = impl_->graph;
  std::set<gtsam::Key>        anchors = impl_->fixed;
  if (anchors.empty() && !impl_->keys.empty())
  {
    anchors.insert(*impl_->keys.begin());
  }
  const auto priorNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);
  for (const auto key : anchors)
  {
    g.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        key, impl_->values.at<gtsam::Pose3>(key), priorNoise);
  }

  gtsam::LevenbergMarquardtParams params;
  params.maxIterations = max_iters;
  gtsam::LevenbergMarquardtOptimizer opt(g, impl_->values, params);
  impl_->values = opt.optimize();
}

mrpt::poses::CPose3D GtsamPoseGraphOptimizer::nodePose(int id) const
{
  const auto key = static_cast<gtsam::Key>(id);
  if (!impl_->values.exists(key))
  {
    return {};
  }
  return mrpt::poses::CPose3D(mrpt::gtsam_wrappers::toTPose3D(impl_->values.at<gtsam::Pose3>(key)));
}

std::size_t GtsamPoseGraphOptimizer::numNodes() const { return impl_->values.size(); }

void GtsamPoseGraphOptimizer::clear()
{
  impl_->graph.resize(0);
  impl_->values.clear();
  impl_->keys.clear();
  impl_->fixed.clear();
}
