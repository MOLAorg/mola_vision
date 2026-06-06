/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/gtsam_pose_graph_optimizer.h>

#include <cmath>
#include <vector>

using mrpt::poses::CPose3D;

namespace
{
// Ground-truth loop trajectory: nodes evenly spaced on a circle, each yawed
// tangent to the circle (so it forms a consistent SE(3) loop).
std::vector<CPose3D> groundTruthLoop(int n)
{
  std::vector<CPose3D> gt;
  for (int i = 0; i < n; ++i)
  {
    const double th = 2.0 * M_PI * i / n;
    gt.emplace_back(2.0 * std::cos(th), 2.0 * std::sin(th), 0.0, th + M_PI / 2, 0.0, 0.0);
  }
  return gt;
}

double posErr(const CPose3D& a, const CPose3D& b)
{
  return std::sqrt(
      (a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y()) +
      (a.z() - b.z()) * (a.z() - b.z()));
}
}  // namespace

TEST(PoseGraph, CorrectsDriftWithLoopClosure)
{
  const int  n  = 8;
  const auto gt = groundTruthLoop(n);

  mola::vision::GtsamPoseGraphOptimizer pgo;

  // Drifted initial estimate: integrate (slightly perturbed) odometry.
  std::vector<CPose3D> init(n);
  init[0] = gt[0];
  for (int i = 1; i < n; ++i)
  {
    const CPose3D odom = gt[i] - gt[i - 1];  // gt[i-1]^-1 o gt[i]
    // Inject a small constant rotational drift each step.
    const CPose3D noise(0.02, -0.01, 0.0, 0.01, 0.0, 0.0);
    init[i] = init[i - 1] + odom + noise;
  }

  for (int i = 0; i < n; ++i)
  {
    pgo.addNode(i, init[i]);
  }
  pgo.setNodeFixed(0, true);

  Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;

  // Odometry edges (exact relative poses).
  for (int i = 1; i < n; ++i)
  {
    mola::vision::PoseGraphEdge e;
    e.from_id       = i - 1;
    e.to_id         = i;
    e.relative_pose = gt[i] - gt[i - 1];
    e.information   = I;
    pgo.addEdge(e);
  }
  // Loop-closure edge n-1 -> 0 (exact).
  {
    mola::vision::PoseGraphEdge e;
    e.from_id       = n - 1;
    e.to_id         = 0;
    e.relative_pose = gt[0] - gt[n - 1];
    e.information   = I;
    e.is_loop       = true;
    pgo.addEdge(e);
  }

  // Drift before optimization is significant.
  EXPECT_GT(posErr(init[n - 1], gt[n - 1]), 0.1);

  pgo.optimize(50);

  // With consistent measurements + a fixed gauge node, the optimum is the GT.
  for (int i = 0; i < n; ++i)
  {
    EXPECT_LT(posErr(pgo.nodePose(i), gt[i]), 0.02) << "node " << i;
  }
}

TEST(PoseGraph, RobustToOutlierLoop)
{
  const int  n  = 8;
  const auto gt = groundTruthLoop(n);

  mola::vision::GtsamPoseGraphOptimizer pgo;
  pgo.setLoopHuberThreshold(1.0);

  for (int i = 0; i < n; ++i)
  {
    pgo.addNode(i, gt[i]);  // start at GT
  }
  pgo.setNodeFixed(0, true);

  const Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
  for (int i = 1; i < n; ++i)
  {
    mola::vision::PoseGraphEdge e;
    e.from_id       = i - 1;
    e.to_id         = i;
    e.relative_pose = gt[i] - gt[i - 1];
    e.information   = I;
    pgo.addEdge(e);
  }
  // A correct loop edge...
  {
    mola::vision::PoseGraphEdge e;
    e.from_id       = n - 1;
    e.to_id         = 0;
    e.relative_pose = gt[0] - gt[n - 1];
    e.information   = I;
    e.is_loop       = true;
    pgo.addEdge(e);
  }
  // ...and a GROSS OUTLIER loop edge (garbage relative pose).
  {
    mola::vision::PoseGraphEdge e;
    e.from_id       = 2;
    e.to_id         = 6;
    e.relative_pose = CPose3D(5.0, -4.0, 3.0, 1.0, 0.5, -0.7);
    e.information   = I;
    e.is_loop       = true;
    pgo.addEdge(e);
  }

  pgo.optimize(50);

  // The Huber kernel should down-weight the outlier loop; nodes stay near GT.
  double max_err = 0;
  for (int i = 0; i < n; ++i)
  {
    max_err = std::max(max_err, posErr(pgo.nodePose(i), gt[i]));
  }
  EXPECT_LT(max_err, 0.5) << "outlier loop corrupted the solution (max err " << max_err << ")";
}
