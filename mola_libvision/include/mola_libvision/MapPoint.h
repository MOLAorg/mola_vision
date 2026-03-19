/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Adapted from: lightweight_vio (MIT License).
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/math/TPoint3D.h>

#include <Eigen/Core>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace mola::vision
{
// Forward declaration
struct Keyframe;

/** A 3D landmark visible from multiple keyframes.
 *
 *  Thread-safety: position, observations and flags are protected by a mutex.
 *  The ID is set at construction and never changes.
 */
class MapPoint
{
 public:
  using Ptr = std::shared_ptr<MapPoint>;

  explicit MapPoint(int id, const mrpt::math::TPoint3Df& pos = {});

  // -----------------------------------------------------------------------
  // Identity
  int id() const { return id_; }

  // -----------------------------------------------------------------------
  // 3D position
  mrpt::math::TPoint3Df position() const;
  void                  setPosition(const mrpt::math::TPoint3Df& p);

  Eigen::Vector3f positionEigen() const;
  void            setPositionEigen(const Eigen::Vector3f& p);

  // -----------------------------------------------------------------------
  // Observations  (keyframe + feature index within that keyframe)
  struct Observation
  {
    std::weak_ptr<Keyframe> keyframe;
    int                     feature_index = -1;
  };

  void addObservation(const std::shared_ptr<Keyframe>& kf, int feature_idx);
  void removeObservation(const std::shared_ptr<Keyframe>& kf);
  int  observationCount() const;

  std::vector<Observation> observations() const;

  // -----------------------------------------------------------------------
  // World-frame position covariance (3×3)
  Eigen::Matrix3f covariance() const;
  void            setCovariance(const Eigen::Matrix3f& cov);

  /** Project covariance to pixel space.
   *  cov_pixel = J_proj · R · cov_world · Rᵀ · J_projᵀ
   *  where J_proj is the 2×3 projection Jacobian at the given camera.
   *  Returns a 2×2 pixel-space covariance.
   */
  Eigen::Matrix2f projectCovariance(
      const Eigen::Matrix3f& R_cw,  ///< Camera←World rotation
      const Eigen::Matrix3f& K  ///< Camera intrinsic matrix
  ) const;

  // -----------------------------------------------------------------------
  // State flags
  bool isBad() const { return bad_.load(); }
  void markBad() { bad_.store(true); }

  bool isMultiViewTriangulated() const { return multi_view_triangulated_; }
  void setMultiViewTriangulated(bool v) { multi_view_triangulated_ = v; }

 private:
  const int id_;

  mutable std::mutex       mtx_;
  mrpt::math::TPoint3Df    position_{0, 0, 0};
  Eigen::Matrix3f          covariance_ = Eigen::Matrix3f::Identity() * 1e-4f;
  std::vector<Observation> observations_;

  std::atomic<bool> bad_{false};
  bool              multi_view_triangulated_ = false;
};

}  // namespace mola::vision
