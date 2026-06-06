/* -------------------------------------------------------------------------
 * mola_visual_slam: monocular / stereo visual SLAM front-end for MOLA.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * End-to-end loop-closure demo on KITTI 00: run stereo VO, build a keyframe
 * pose-graph (VO odometry edges), propose loops from ground truth (revisits),
 * VERIFY each with the real images (ORB match + PnP), add the verified loop
 * edges, optimize with GTSAM, and compare ATE before/after. GT is used only to
 * PROPOSE candidate loop pairs; the constraint itself comes from the images.
 *
 * Skipped unless KITTI_STEREO_DIR and KITTI_POSES are set.
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/gtsam_pose_graph_optimizer.h>
#include <mola_libvision/loop_closure.h>
#include <mola_libvision/orb_descriptor.h>
#include <mola_libvision/stereo_matcher.h>
#include <mola_visual_slam/VisualSlam.h>
#include <mrpt/core/Clock.h>
#include <mrpt/img/CImage.h>
#include <mrpt/system/filesystem.h>

#include <Eigen/Geometry>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

using mrpt::poses::CPose3D;

namespace
{
bool readProjection(const std::string& path, const std::string& key, double out[12])
{
  std::ifstream f(path);
  std::string   line;
  while (std::getline(f, line))
  {
    if (line.rfind(key, 0) != 0) continue;
    std::istringstream ss(line.substr(key.size() + 1));
    for (int i = 0; i < 12; ++i)
      if (!(ss >> out[i])) return false;
    return true;
  }
  return false;
}

std::string frameName(int i)
{
  std::ostringstream ss;
  ss << std::setw(6) << std::setfill('0') << i << ".png";
  return ss.str();
}

std::vector<CPose3D> readGtPoses(const std::string& path)
{
  std::vector<CPose3D> out;
  std::ifstream        f(path);
  std::string          line;
  while (std::getline(f, line))
  {
    std::istringstream ss(line);
    double             m[12];
    bool               ok = true;
    for (double& v : m)
      if (!(ss >> v)) ok = false;
    if (!ok) break;
    mrpt::math::CMatrixDouble44 H = mrpt::math::CMatrixDouble44::Identity();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c) H(r, c) = m[r * 4 + c];
    out.emplace_back(H);
  }
  return out;
}

// SE3 (rigid) ATE of estimated keyframe positions vs GT positions.
double ate(const std::vector<Eigen::Vector3d>& A, const std::vector<Eigen::Vector3d>& B)
{
  const Eigen::Index                       m = static_cast<Eigen::Index>(A.size());
  Eigen::Matrix<double, 3, Eigen::Dynamic> Am(3, m), Bm(3, m);
  for (Eigen::Index k = 0; k < m; ++k)
  {
    Am.col(k) = A[static_cast<size_t>(k)];
    Bm.col(k) = B[static_cast<size_t>(k)];
  }
  const Eigen::Matrix4d S   = Eigen::umeyama(Am, Bm, false);
  double                sse = 0;
  for (Eigen::Index k = 0; k < m; ++k)
  {
    const Eigen::Vector4d a(Am(0, k), Am(1, k), Am(2, k), 1.0);
    sse += ((S * a).head<3>() - Bm.col(k)).squaredNorm();
  }
  return std::sqrt(sse / static_cast<double>(m));
}
}  // namespace

TEST(KittiLoop, StereoLoopClosureReducesDrift)
{
  const char* envDir = std::getenv("KITTI_STEREO_DIR");
  const char* envPos = std::getenv("KITTI_POSES");
  if (!envDir || !envPos)
  {
    GTEST_SKIP() << "Set KITTI_STEREO_DIR and KITTI_POSES.";
  }
  const std::string dir = envDir;
  double            P0[12];
  double            P1[12];
  ASSERT_TRUE(readProjection(dir + "/calib.txt", "P0", P0));
  ASSERT_TRUE(readProjection(dir + "/calib.txt", "P1", P1));
  const double fx = P0[0], fy = P0[5], cx = P0[2], cy = P0[6];
  const double baseline = -P1[3] / fx;

  mrpt::img::TCamera cam;
  bool               camReady = false;
  const auto         gt       = readGtPoses(envPos);

  size_t max_frames = 1700;  // KITTI 00's first same-heading loop is ~1569<->122
  if (const char* mf = std::getenv("KITTI_MAX_FRAMES"))
    max_frames = static_cast<size_t>(std::atol(mf));
  const int kf_stride = 10;

  mola::VisualSlam slam;
  slam.setMinLoggingLevel(mrpt::system::LVL_ERROR);

  struct KF
  {
    int               frame;
    CPose3D           vo;  // VO estimate (cam-in-world)
    mrpt::img::CImage left, right;
  };
  std::vector<KF> kfs;

  for (size_t i = 0; i < max_frames; ++i)
  {
    const std::string lp = dir + "/image_0/" + frameName(static_cast<int>(i));
    const std::string rp = dir + "/image_1/" + frameName(static_cast<int>(i));
    if (!mrpt::system::fileExists(lp)) break;
    mrpt::img::CImage left, right;
    ASSERT_TRUE(left.loadFromFile(lp));
    ASSERT_TRUE(right.loadFromFile(rp));
    if (!camReady)
    {
      cam.ncols = static_cast<uint32_t>(left.getWidth());
      cam.nrows = static_cast<uint32_t>(left.getHeight());
      cam.setIntrinsicParamsFromValues(fx, fy, cx, cy);
      camReady = true;
    }
    const auto pose = slam.processStereoFrame(left, right, cam, baseline, mrpt::Clock::now());
    if (slam.isInitialized() && (i % kf_stride == 0))
    {
      kfs.push_back({static_cast<int>(i), pose, left, right});
    }
  }
  ASSERT_GE(kfs.size(), 30u);
  std::cout << "[kitti-loop] " << kfs.size() << " keyframes over " << max_frames << " frames\n";

  // --- Pose graph: nodes = VO keyframe poses, odometry edges between them. ---
  mola::vision::GtsamPoseGraphOptimizer pgo;
  for (size_t k = 0; k < kfs.size(); ++k) pgo.addNode(static_cast<int>(k), kfs[k].vo);
  pgo.setNodeFixed(0, true);

  Eigen::Matrix<double, 6, 6> Iodom = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
  for (size_t k = 1; k < kfs.size(); ++k)
  {
    mola::vision::PoseGraphEdge e;
    e.from_id       = static_cast<int>(k - 1);
    e.to_id         = static_cast<int>(k);
    e.relative_pose = kfs[k].vo - kfs[k - 1].vo;  // "to" in "from" frame
    e.information   = Iodom;
    pgo.addEdge(e);
  }

  // --- Loop candidates proposed from GT, then VERIFIED with the images. ---
  mola::vision::GridDistributorParams gp;
  gp.max_corners  = 600;
  gp.min_distance = 12.f;
  mola::vision::GridFeatureDistributor dist(gp);

  auto describeQuery = [&](const KF& kf, std::vector<mola::vision::OrbDescriptor>& d,
                           std::vector<mrpt::math::TPoint2Df>& px)
  {
    const auto        g   = kf.left.grayscale();
    const auto        pts = dist.detect(g, {});
    std::vector<bool> v;
    const auto        dd = mola::vision::computeOrbDescriptors(g, pts, v);
    for (size_t i = 0; i < pts.size(); ++i)
      if (v[i])
      {
        d.push_back(dd[i]);
        px.push_back(pts[i]);
      }
  };
  auto describeCandidate = [&](const KF& kf, std::vector<mola::vision::OrbDescriptor>& d,
                               std::vector<mrpt::math::TPoint3Df>& xyz)
  {
    const auto        gl  = kf.left.grayscale();
    const auto        gr  = kf.right.grayscale();
    const auto        pts = dist.detect(gl, {});
    const auto        sm  = mola::vision::matchStereo(gl, gr, pts, fx, baseline);
    std::vector<bool> v;
    const auto        dd = mola::vision::computeOrbDescriptors(gl, pts, v);
    for (size_t i = 0; i < pts.size(); ++i)
    {
      if (!v[i] || !sm.valid[i]) continue;
      const float Z = sm.depth[i];
      d.push_back(dd[i]);
      xyz.push_back(
          {static_cast<float>((pts[i].x - cx) / fx) * Z,
           static_cast<float>((pts[i].y - cy) / fy) * Z, Z});
    }
  };

  // GT forward (camera +Z, world) for a same-heading filter.
  auto fwd = [&](int frame) -> Eigen::Vector3d
  {
    const auto R = gt[frame].getRotationMatrix();
    return Eigen::Vector3d(R(0, 2), R(1, 2), R(2, 2));
  };

  int n_loops = 0;
  for (size_t j = 0; j < kfs.size(); ++j)
  {
    // GT-proposed candidates: earlier keyframes (>=25 kf gap) within 5 m AND
    // roughly the same heading (a real visual loop detector only matches views
    // from a similar viewpoint, not the opposite-direction pass). Try the
    // nearest few and add every one that the IMAGES geometrically verify.
    std::vector<std::pair<double, int>> cands;
    for (size_t i = 0; i + 25 < j; ++i)
    {
      const double d = (gt[kfs[j].frame].translation() - gt[kfs[i].frame].translation()).norm();
      if (d < 5.0 && fwd(kfs[i].frame).dot(fwd(kfs[j].frame)) > 0.85)
      {
        cands.emplace_back(d, static_cast<int>(i));
      }
    }
    if (cands.empty()) continue;
    std::sort(cands.begin(), cands.end());

    std::vector<mola::vision::OrbDescriptor> qd;
    std::vector<mrpt::math::TPoint2Df>       qpx;
    describeQuery(kfs[j], qd, qpx);

    int tried = 0;
    for (const auto& [d, ci] : cands)
    {
      if (tried++ >= 3) break;  // a few nearest candidates per query
      std::vector<mola::vision::OrbDescriptor> cd;
      std::vector<mrpt::math::TPoint3Df>       cxyz;
      describeCandidate(kfs[static_cast<size_t>(ci)], cd, cxyz);

      mola::vision::LoopVerifyParams vp;
      vp.min_inliers = 20;
      vp.max_hamming = 70;  // tolerate the cross-view appearance change
      const auto vr  = mola::vision::verifyLoopPnP(qd, qpx, cd, cxyz, cam, vp);
      if (!vr.success) continue;

      mola::vision::PoseGraphEdge e;
      e.from_id       = ci;
      e.to_id         = static_cast<int>(j);
      e.relative_pose = -vr.relative_pose;  // "j" in "i" frame
      e.information   = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
      e.is_loop       = true;
      pgo.addEdge(e);
      ++n_loops;
    }
  }
  std::cout << "[kitti-loop] verified loop closures added: " << n_loops << "\n";
  ASSERT_GE(n_loops, 1) << "no loop closures verified";

  // --- ATE before / after optimization. ---
  std::vector<Eigen::Vector3d> voPos, gtPos;
  for (const auto& kf : kfs)
  {
    voPos.emplace_back(kf.vo.x(), kf.vo.y(), kf.vo.z());
    gtPos.emplace_back(gt[kf.frame].x(), gt[kf.frame].y(), gt[kf.frame].z());
  }
  const double ate_before = ate(voPos, gtPos);

  pgo.optimize(100);

  std::vector<Eigen::Vector3d> optPos;
  for (size_t k = 0; k < kfs.size(); ++k)
  {
    const auto p = pgo.nodePose(static_cast<int>(k));
    optPos.emplace_back(p.x(), p.y(), p.z());
  }
  const double ate_after = ate(optPos, gtPos);

  std::cout << "[kitti-loop] ATE before = " << ate_before << " m, after = " << ate_after << " m\n";

  // The full detect -> verify(images) -> SE(3) pose-graph optimize pipeline runs
  // on real KITTI 00 and reduces drift. The improvement is modest because an
  // SE(3) pose graph corrects rigid/rotational drift but NOT scale drift, which
  // dominates long monocular-like stereo VO here (a Sim(3) pose graph would be
  // needed to also absorb scale, as ORB-SLAM does).
  EXPECT_LT(ate_after, ate_before) << "loop closure did not reduce drift";
}
