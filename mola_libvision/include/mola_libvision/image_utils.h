/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#pragma once

#include <mrpt/img/CImage.h>

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace mola::vision
{
// ---------------------------------------------------------------------------
// Zero-copy Eigen::Map views over CImage pixel data
// ---------------------------------------------------------------------------

/** Returns a zero-copy read-only Eigen::Map<RowMajor uint8> over a grayscale
 *  CImage.  No memory is allocated; the Map shares the CImage buffer.
 *
 *  Requirements: img must be CH_GRAY, PixelDepth::D8U, and already loaded.
 *  The Map remains valid as long as `img` is alive and not resized.
 *
 *  Row stride is handled via Eigen::OuterStride so images with padding are
 *  also supported (though MRPT's STB backend typically has no padding).
 */
using EigenGrayMapConst = Eigen::Map<
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Unaligned,
    Eigen::OuterStride<Eigen::Dynamic>>;

using EigenGrayMap = Eigen::Map<
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Unaligned,
    Eigen::OuterStride<Eigen::Dynamic>>;

inline EigenGrayMapConst asEigenMap(const mrpt::img::CImage& img)
{
  assert(img.channels() == mrpt::img::CH_GRAY);
  assert(img.getPixelDepth() == mrpt::img::PixelDepth::D8U);
  const int    rows   = img.getHeight();
  const int    cols   = img.getWidth();
  const size_t stride = img.getRowStride();  // bytes; for uint8_t == elements
  return EigenGrayMapConst(
      img.ptrLine<uint8_t>(0), rows, cols,
      Eigen::OuterStride<Eigen::Dynamic>(static_cast<int>(stride)));
}

inline EigenGrayMap asEigenMap(mrpt::img::CImage& img)
{
  assert(img.channels() == mrpt::img::CH_GRAY);
  assert(img.getPixelDepth() == mrpt::img::PixelDepth::D8U);
  const int    rows   = img.getHeight();
  const int    cols   = img.getWidth();
  const size_t stride = img.getRowStride();
  return EigenGrayMap(
      img.ptrLine<uint8_t>(0), rows, cols,
      Eigen::OuterStride<Eigen::Dynamic>(static_cast<int>(stride)));
}

// ---------------------------------------------------------------------------
// Image processing primitives
// ---------------------------------------------------------------------------

/** Compute Sobel gradient images Ix and Iy using 3x3 separable kernels.
 *
 *  Uses:  Ix = [-1 0 1] ⊗ [1 2 1]ᵀ
 *         Iy = [1 2 1]  ⊗ [-1 0 1]ᵀ
 *
 *  Input:  grayscale CImage (CH_GRAY, D8U).
 *  Output: Ix, Iy as Eigen::MatrixXf (same size as input).
 *          Border pixels are set to zero.
 *
 *  Complexity: O(W*H) with constant factor ≈ 8 muls + 6 adds per pixel
 *  (two separable passes of 3-tap filters).
 *
 *  \note MRPT 3.x has no equivalent public API: CImage offers Gaussian/median
 *        filtering and a single-point Shi-Tomasi score (KLT_response()), but no
 *        whole-image float gradient maps. This function provides the Ix/Iy
 *        needed to build the structure tensor in goodFeaturesToTrack().
 */
void sobelGradients(const mrpt::img::CImage& img, Eigen::MatrixXf& Ix, Eigen::MatrixXf& Iy);

/** Separable Gaussian blur on a grayscale CImage, parameterized by sigma.
 *
 *  Kernel size is automatically chosen as 2*ceil(2*sigma)+1.
 *  Output is a new CH_GRAY CImage of the same size.
 *  Border pixels are handled by clamping (replicate).
 *
 *  \note This overlaps `mrpt::img::CImage::filterGaussian(out, W, H, sigma)`,
 *        which takes an explicit window size. This `mola::vision` facade exists
 *        to (a) derive the kernel size from sigma automatically and (b)
 *        guarantee the replicate-border behavior the corner-detection pipeline
 *        depends on. Prefer `CImage::filterGaussian` directly for general use.
 */
void gaussianBlur(const mrpt::img::CImage& in, mrpt::img::CImage& out, float sigma);

}  // namespace mola::vision
