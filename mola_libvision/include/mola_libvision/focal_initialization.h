/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mola_libvision/camera_model.h>
#include <mola_libvision/robust_estimators.h>

#include <Eigen/Core>
#include <vector>

/** \file focal_initialization.h
 *  Initial estimates of a focal length shared by all views of one camera,
 *  for self-calibration from image correspondences alone. All of them assume
 *  square pixels and a known principal point (usually the image center);
 *  the result is meant to seed a bundle adjustment with free intrinsics. */
namespace mola::vision
{
/** Result of a focal length search. */
struct FocalEstimate
{
  bool   valid = false;
  double focal = 0.0;  ///< pixels
  double cost  = 0.0;  ///< cost at the optimum (method-specific units)
  /** Ratio between the cost at the optimum and the median cost of the sweep:
   *  close to 1 means a flat curve, i.e. the focal length is barely
   *  observable from these views. */
  double              flatness = 1.0;
  std::vector<double> focals;  ///< sampled focal lengths
  std::vector<double> costs;  ///< cost at each sample
};

/** Focal length from a set of fundamental matrices of view pairs sharing the
 *  intrinsics (Mendonca and Cipolla, CVPR 1999): for the right K, E = K^T F K
 *  has two equal singular values. Minimizes the median over pairs of
 *  (s1 - s2) / (s1 + s2) on a log-spaced grid in [f_min, f_max], refined by
 *  golden-section search. Needs approximately pinhole (low distortion) pixels. */
FocalEstimate estimateFocalFromFundamentals(
    const std::vector<Eigen::Matrix3d>& F, double cx, double cy, double f_min, double f_max,
    int num_samples = 80);

/** Focal length from infinite (rotation-only) homographies H ~ K R K^-1, for
 *  pairs with negligible parallax: for the right K, K^-1 H K is a scaled
 *  rotation. Same search as estimateFocalFromFundamentals(). */
FocalEstimate estimateFocalFromRotationHomographies(
    const std::vector<Eigen::Matrix3d>& H, double cx, double cy, double f_min, double f_max,
    int num_samples = 80);

/** Pixel correspondences between two views. */
struct ViewPairMatches
{
  std::vector<Eigen::Vector2d> p1;
  std::vector<Eigen::Vector2d> p2;
};

struct FocalSweepOptions
{
  /** Model whose focal is searched, with zero distortion: Pinhole/RadTan
   *  use a perspective mapping, Equidistant an equidistant one. */
  CameraModelType model       = CameraModelType::RadTan;
  uint32_t        width       = 0;
  uint32_t        height      = 0;
  double          f_min       = 0.0;  ///< 0: derived from a 175 deg (fisheye) or 120 deg FOV
  double          f_max       = 0.0;  ///< 0: derived from a 20 deg FOV
  int             num_samples = 30;
  /** Distortion coefficients assumed during the search (CameraModel order;
   *  empty means none). Radial-tangential coefficients are defined on the
   *  normalized plane, so they stay meaningful as the focal length changes,
   *  which allows a coarse 2D search over (focal, k1). Ignored for Pinhole. */
  Eigen::VectorXd distortion;
  /** Score pairs with a rotation-only model (two-point MSAC) instead of a
   *  general relative pose: for footage that pans about the camera center. */
  bool rotation_only = false;
  /** Inlier threshold, in pixels (converted to an angle for each focal). */
  double   threshold_px   = 2.0;
  int      max_iterations = 300;
  unsigned seed           = 7;
};

/** Model-aware focal search by two-view consensus: for each candidate focal
 *  (principal point at the image center, no distortion) the pixels are
 *  unprojected to bearings and a five-point MSAC relative pose is fitted to
 *  each pair; the cost is the MSAC score converted to squared pixels, summed
 *  over the pairs. Works for perspective and fisheye lenses, which the
 *  fundamental-matrix methods do not. A final fine search around the best
 *  sample uses a deterministic refit on that sample's inliers. Pixels that
 *  the candidate model cannot unproject count as outliers. */
FocalEstimate estimateFocalByTwoViewConsensus(
    const std::vector<ViewPairMatches>& pairs, const FocalSweepOptions& options);

}  // namespace mola::vision
