/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/img/TCamera.h>

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

namespace mola::vision
{
/** Supported central camera models. */
enum class CameraModelType : uint8_t
{
  /** Distortion-free pinhole. Parameters: [fx, fy, cx, cy]. */
  Pinhole = 0,
  /** Pinhole with radial-tangential (Brown-Conrady, OpenCV/ROS "plumb_bob")
   *  distortion. Parameters: [fx, fy, cx, cy, k1, k2, p1, p2, k3]. */
  RadTan,
  /** Equidistant fisheye (Kannala-Brandt, OpenCV fisheye / ROS
   *  "equidistant"). Parameters: [fx, fy, cx, cy, k1, k2, k3, k4]. Valid for
   *  incidence angles beyond 90 degrees. */
  Equidistant
};

/** A central camera model with analytic Jacobians.
 *
 *  Projects a 3D point given in the CAMERA frame (x right, y down, z forward)
 *  to a pixel, optionally returning the Jacobians of the pixel with respect to
 *  the point (2x3) and to the intrinsic parameter vector (2xN). Unprojects a
 *  pixel to a unit bearing vector by Newton iteration on the distortion
 *  function, so it works for any incidence angle the model can represent.
 *
 *  Conventions follow OpenCV and ROS `camera_info`, so the coefficients can be
 *  exchanged with them and with mrpt::img::TCamera without conversion:
 *   - RadTan:  x' = X/Z, r^2 = x'^2 + y'^2,
 *              x_d = x'(1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x'y' + p2 (r^2 + 2x'^2),
 *              y_d = y'(1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2y'^2) + 2 p2 x'y'.
 *   - Equidistant: theta = atan2(sqrt(X^2+Y^2), Z),
 *              theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8),
 *              (x_d, y_d) = theta_d (X, Y) / sqrt(X^2 + Y^2).
 *   - Pixel:   u = fx x_d + cx, v = fy y_d + cy.
 */
class CameraModel
{
 public:
  CameraModel() = default;

  /** Builds a model from its parameter vector (see CameraModelType for the
   *  order). Throws if the vector size does not match the model. */
  CameraModel(CameraModelType type, uint32_t width, uint32_t height, const Eigen::VectorXd& params);

  static CameraModel Pinhole(
      uint32_t width, uint32_t height, double fx, double fy, double cx, double cy);
  static CameraModel RadTan(
      uint32_t width, uint32_t height, double fx, double fy, double cx, double cy, double k1,
      double k2, double p1, double p2, double k3 = 0.0);
  static CameraModel Equidistant(
      uint32_t width, uint32_t height, double fx, double fy, double cx, double cy, double k1,
      double k2, double k3, double k4);

  /** Converts from an MRPT camera. `none` maps to Pinhole, `plumb_bob` to
   *  RadTan (rational coefficients k4..k6 must be zero), `kannala_brandt` to
   *  Equidistant. */
  static CameraModel FromTCamera(const mrpt::img::TCamera& cam);

  /** Converts to an MRPT camera (same coefficients, no conversion). */
  [[nodiscard]] mrpt::img::TCamera asTCamera() const;

  /** Number of intrinsic parameters of a model. */
  static int NumParams(CameraModelType type);

  /** Names of the intrinsic parameters, in vector order. */
  static std::vector<std::string> ParamNames(CameraModelType type);

  /** Model name as used by ROS camera_info ("plumb_bob", "equidistant"); a
   *  distortion-free pinhole is reported as "plumb_bob" with zero
   *  coefficients. */
  static std::string RosDistortionModelName(CameraModelType type);

  /** Short human readable name ("pinhole", "radtan", "equidistant"). */
  static std::string TypeName(CameraModelType type);

  [[nodiscard]] CameraModelType type() const { return type_; }
  [[nodiscard]] uint32_t        width() const { return width_; }
  [[nodiscard]] uint32_t        height() const { return height_; }
  void                          setImageSize(uint32_t width, uint32_t height);

  [[nodiscard]] const Eigen::VectorXd& params() const { return params_; }
  /** Sets the whole parameter vector (size must match the model). */
  void setParams(const Eigen::VectorXd& params);

  [[nodiscard]] double fx() const { return params_[0]; }
  [[nodiscard]] double fy() const { return params_[1]; }
  [[nodiscard]] double cx() const { return params_[2]; }
  [[nodiscard]] double cy() const { return params_[3]; }

  /** Distortion coefficients only (params after the first four), in the
   *  order used by ROS/OpenCV: RadTan [k1, k2, p1, p2, k3], Equidistant
   *  [k1, k2, k3, k4], Pinhole empty. */
  [[nodiscard]] Eigen::VectorXd distortion() const;

  /** Projects a point in the camera frame to a pixel.
   *
   *  \param p         Point in the camera frame (any scale for a direction).
   *  \param pixel     Output pixel.
   *  \param J_point   Optional 2x3 Jacobian d(pixel)/d(p).
   *  \param J_params  Optional 2xN Jacobian d(pixel)/d(params).
   *  \return false when the point cannot be projected: behind a pinhole
   *          camera, or outside the region where the distortion mapping is
   *          monotonic (where the model folds back and is meaningless).
   *          The pixel may lie outside the image; that is not checked here.
   */
  bool project(
      const Eigen::Vector3d& p, Eigen::Vector2d& pixel,
      Eigen::Matrix<double, 2, 3>*              J_point  = nullptr,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* J_params = nullptr) const;

  /** Unprojects a pixel to a unit bearing vector in the camera frame.
   *  \return false if the undistortion did not converge or the pixel lies
   *          beyond the valid (monotonic) region of the model. */
  bool unproject(const Eigen::Vector2d& pixel, Eigen::Vector3d& bearing) const;

  /** Largest incidence angle (radians, from the optical axis) for which the
   *  radial distortion mapping stays monotonic, capped at the model limit
   *  (just under 90 deg for pinhole models, 180 deg for Equidistant). */
  [[nodiscard]] double maxIncidenceAngle() const;

  /** True if the radial mapping is monotonic at this incidence angle. */
  [[nodiscard]] bool isMonotonicAt(double incidence_angle) const;

 private:
  CameraModelType type_   = CameraModelType::Pinhole;
  uint32_t        width_  = 0;
  uint32_t        height_ = 0;
  Eigen::VectorXd params_ = Eigen::VectorXd::Zero(4);
};

/** Which intrinsic parameters are estimated, for calibration problems.
 *
 *  The free (optimized) parameter vector is laid out as: the focal length(s),
 *  then the principal point, then the free distortion coefficients, each group
 *  in model order. Fixed parameters keep the value of the reference model.
 */
struct IntrinsicsMask
{
  bool estimate_focal = true;
  /** When true, a single focal parameter is estimated and fy keeps the aspect
   *  ratio fy/fx of the reference model (usually 1, square pixels). */
  bool fixed_aspect_ratio       = false;
  bool estimate_principal_point = true;
  /** One flag per distortion coefficient, in model order (see
   *  CameraModel::distortion()). Shorter vectors leave the remaining
   *  coefficients fixed; an empty vector fixes all of them. */
  std::vector<bool> estimate_distortion;

  /** Convenience: estimate the first `n` distortion coefficients. */
  static std::vector<bool> FirstN(int total, int n);
};

/** Maps between a camera model's full parameter vector and the free
 *  parameter vector selected by an IntrinsicsMask. */
class IntrinsicsParameterization
{
 public:
  IntrinsicsParameterization() = default;
  IntrinsicsParameterization(const CameraModel& reference, const IntrinsicsMask& mask);

  /** Number of free parameters. */
  [[nodiscard]] int dim() const { return static_cast<int>(J_.cols()); }

  /** Free parameter vector extracted from a model. */
  [[nodiscard]] Eigen::VectorXd toFree(const CameraModel& cam) const;

  /** Writes the free parameters into `cam`, keeping fixed ones from the
   *  reference model given at construction. */
  void fromFree(const Eigen::VectorXd& free_params, CameraModel& cam) const;

  /** Constant Jacobian d(full params)/d(free params), N x dim(). */
  [[nodiscard]] const Eigen::MatrixXd& jacobian() const { return J_; }

  /** Name of each free parameter ("f" when a single focal is estimated). */
  [[nodiscard]] const std::vector<std::string>& freeNames() const { return names_; }

  /** Index into the full parameter vector of each free parameter (the fx
   *  index for a shared focal). */
  [[nodiscard]] const std::vector<int>& freeIndices() const { return indices_; }

  [[nodiscard]] const CameraModel& reference() const { return reference_; }

 private:
  CameraModel              reference_;
  double                   aspect_       = 1.0;
  bool                     shared_focal_ = false;
  std::vector<int>         indices_;
  std::vector<std::string> names_;
  Eigen::MatrixXd          J_;
};

}  // namespace mola::vision
