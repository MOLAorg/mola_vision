/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Adapted from: lightweight_vio (MIT License).
 * ------------------------------------------------------------------------- */
#include <mola_libvision/MapPoint.h>

using namespace mola::vision;

MapPoint::MapPoint(int id, const mrpt::math::TPoint3Df& pos) : id_(id), position_(pos) {}

mrpt::math::TPoint3Df MapPoint::position() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return position_;
}

void MapPoint::setPosition(const mrpt::math::TPoint3Df& p)
{
  std::lock_guard<std::mutex> lk(mtx_);
  position_ = p;
}

Eigen::Vector3f MapPoint::positionEigen() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return Eigen::Vector3f(position_.x, position_.y, position_.z);
}

void MapPoint::setPositionEigen(const Eigen::Vector3f& p)
{
  std::lock_guard<std::mutex> lk(mtx_);
  position_ = {p.x(), p.y(), p.z()};
}

void MapPoint::addObservation(const std::shared_ptr<Keyframe>& kf, int feature_idx)
{
  std::lock_guard<std::mutex> lk(mtx_);
  observations_.push_back({kf, feature_idx});
}

void MapPoint::removeObservation(const std::shared_ptr<Keyframe>& kf)
{
  std::lock_guard<std::mutex> lk(mtx_);
  observations_.erase(
      std::remove_if(
          observations_.begin(), observations_.end(),
          [&](const Observation& o) { return o.keyframe.lock() == kf; }),
      observations_.end());
}

int MapPoint::observationCount() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return static_cast<int>(observations_.size());
}

std::vector<MapPoint::Observation> MapPoint::observations() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return observations_;
}

Eigen::Matrix3f MapPoint::covariance() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return covariance_;
}

void MapPoint::setCovariance(const Eigen::Matrix3f& cov)
{
  std::lock_guard<std::mutex> lk(mtx_);
  covariance_ = cov;
}

Eigen::Matrix2f MapPoint::projectCovariance(
    const Eigen::Matrix3f& R_cw, const Eigen::Matrix3f& K) const
{
  std::lock_guard<std::mutex> lk(mtx_);

  // Point in camera frame
  const Eigen::Vector3f Pw(position_.x, position_.y, position_.z);
  const Eigen::Vector3f Pc = R_cw * Pw;  // (translation assumed 0 here; caller handles it)

  const float z  = Pc.z();
  const float z2 = z * z;
  if (z2 < 1e-8f) return Eigen::Matrix2f::Identity() * 1e6f;

  // 2×3 projection Jacobian:  J_proj = (1/z) * [1 0 -x/z; 0 1 -y/z] * K
  const float                fx = K(0, 0), fy = K(1, 1);
  Eigen::Matrix<float, 2, 3> J;
  J << fx / z, 0, -fx * Pc.x() / z2, 0, fy / z, -fy * Pc.y() / z2;

  // Rotate covariance to camera frame, then project
  const Eigen::Matrix3f cov_cam = R_cw * covariance_ * R_cw.transpose();
  return J * cov_cam * J.transpose();
}
