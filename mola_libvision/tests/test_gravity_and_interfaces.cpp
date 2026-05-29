/* -------------------------------------------------------------------------
 * mola_libvision unit tests: gravity estimator + Loop/PoseGraph interfaces
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/gravity_estimator.h>
#include <mola_libvision/loop_detector.h>
#include <mola_libvision/pose_graph_optimizer.h>

#include <Eigen/Geometry>
#include <map>
#include <random>

using namespace mola::vision;

// ---------------------------------------------------------------------------
// Static gravity estimation from noisy stationary accelerometer samples.
// ---------------------------------------------------------------------------
TEST(GravityEstimator, StaticFromAccel)
{
  // True gravity in the sensor frame (arbitrary tilt), magnitude 9.81.
  const Eigen::Vector3d g_true = 9.81 * Eigen::Vector3d(0.2, -0.3, -0.93).normalized();
  // A stationary accelerometer reads specific force f = -g.
  const Eigen::Vector3d f = -g_true;

  std::mt19937                     rng(7);
  std::normal_distribution<double> noise(0.0, 0.02);
  std::vector<Eigen::Vector3d>     samples;
  for (int i = 0; i < 500; ++i)
  {
    samples.emplace_back(f.x() + noise(rng), f.y() + noise(rng), f.z() + noise(rng));
  }

  auto est = estimateGravityStatic(samples, 9.81);
  ASSERT_TRUE(est.ok);
  EXPECT_NEAR(est.gravity.norm(), 9.81, 1e-9);

  const double cos_ang = est.gravity.normalized().dot(g_true.normalized());
  const double ang_deg = std::acos(std::min(1.0, cos_ang)) * 180.0 / M_PI;
  EXPECT_LT(ang_deg, 1.0) << "gravity direction within 1 deg";

  EXPECT_FALSE(estimateGravityStatic({}, 9.81).ok);
}

// ---------------------------------------------------------------------------
// Gravity-alignment rotation maps gravity to the target down axis.
// ---------------------------------------------------------------------------
TEST(GravityEstimator, AlignmentRotation)
{
  const Eigen::Vector3d g = 9.81 * Eigen::Vector3d(0.1, 0.2, -0.97).normalized();
  const Eigen::Vector3d down(0, 0, -1);
  const Eigen::Matrix3d R = gravityAlignmentRotation(g, down);

  // R is a valid rotation.
  EXPECT_NEAR((R.transpose() * R - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-9);
  EXPECT_NEAR(R.determinant(), 1.0, 1e-9);

  // R rotates the gravity direction onto -Z.
  const Eigen::Vector3d aligned = R * g.normalized();
  EXPECT_LT((aligned - down).norm(), 1e-9);

  // Already-aligned input -> identity.
  const Eigen::Matrix3d R0 = gravityAlignmentRotation(down, down);
  EXPECT_LT((R0 - Eigen::Matrix3d::Identity()).norm(), 1e-9);

  // Anti-parallel input -> flips to target.
  const Eigen::Matrix3d Rflip = gravityAlignmentRotation(-down, down);
  EXPECT_LT((Rflip * (-down) - down).norm(), 1e-9);
}

// ---------------------------------------------------------------------------
// Interface usability: trivial mock implementations compile and behave.
// ---------------------------------------------------------------------------
namespace
{
class MockLoopDetector : public LoopDetector
{
 public:
  void addKeyframe(const Keyframe::Ptr& kf) override { ids_.push_back(kf->id); }
  std::vector<LoopCandidate> detect(const Keyframe::Ptr& kf, int min_id_gap) override
  {
    std::vector<LoopCandidate> out;
    for (int id : ids_)
    {
      if (kf->id - id >= min_id_gap)
      {
        out.push_back({kf->id, id, 1.0f});
      }
    }
    return out;
  }
  void clear() override { ids_.clear(); }

 private:
  std::vector<int> ids_;
};

class MockPoseGraph : public PoseGraphOptimizer
{
 public:
  void addNode(int id, const mrpt::poses::CPose3D& pose) override { nodes_[id] = pose; }
  void setNodeFixed(int id, bool fixed) override { (void)id, (void)fixed; }
  void addEdge(const PoseGraphEdge& e) override
  {
    ++n_edges_;
    (void)e;
  }
  void                 optimize(int max_iters) override { (void)max_iters; }
  mrpt::poses::CPose3D nodePose(int id) const override { return nodes_.at(id); }
  std::size_t          numNodes() const override { return nodes_.size(); }
  void                 clear() override
  {
    nodes_.clear();
    n_edges_ = 0;
  }
  int edges() const { return n_edges_; }

 private:
  std::map<int, mrpt::poses::CPose3D> nodes_;
  int                                 n_edges_ = 0;
};
}  // namespace

TEST(Interfaces, LoopDetectorMock)
{
  MockLoopDetector det;
  for (int i = 0; i < 5; ++i)
  {
    auto kf = std::make_shared<Keyframe>();
    kf->id  = i;
    det.addKeyframe(kf);
  }
  auto query = std::make_shared<Keyframe>();
  query->id  = 100;
  auto cands = det.detect(query, /*min_id_gap=*/30);
  EXPECT_EQ(cands.size(), 5u);
  EXPECT_EQ(cands[0].query_kf_id, 100);

  // Polymorphic use through the base pointer.
  LoopDetector::Ptr base = std::make_shared<MockLoopDetector>();
  base->addKeyframe(query);
  base->clear();
}

TEST(Interfaces, PoseGraphMock)
{
  MockPoseGraph pg;
  pg.addNode(0, mrpt::poses::CPose3D());
  pg.addNode(1, mrpt::poses::CPose3D(1, 0, 0, 0, 0, 0));
  pg.setNodeFixed(0, true);
  PoseGraphEdge e;
  e.from_id = 0;
  e.to_id   = 1;
  pg.addEdge(e);
  pg.optimize(5);
  EXPECT_EQ(pg.numNodes(), 2u);
  EXPECT_EQ(pg.edges(), 1);
  EXPECT_NEAR(pg.nodePose(1).x(), 1.0, 1e-9);
}
