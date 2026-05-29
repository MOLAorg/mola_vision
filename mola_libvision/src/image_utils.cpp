/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/image_utils.h>

#include <cmath>
#include <vector>

using namespace mola::vision;

// ---------------------------------------------------------------------------
// sobelGradients
// ---------------------------------------------------------------------------
// Separable 3×3 Sobel:
//   Ix = [1 2 1]ᵀ ⊗ [-1 0 1]   (smooth vertically, diff horizontally)
//   Iy = [-1 0 1]ᵀ ⊗ [1 2 1]   (diff vertically, smooth horizontally)
//
// We do two passes:
//   Pass A (horizontal): rows of [-1 0 1] → temp1 (for Ix), [1 2 1] → temp2 (for Iy)
//   Pass B (vertical):   cols of [1 2 1]  applied to temp1 → Ix
//                        cols of [-1 0 1] applied to temp2 → Iy
// ---------------------------------------------------------------------------
void mola::vision::sobelGradients(
    const mrpt::img::CImage& img, Eigen::MatrixXf& Ix, Eigen::MatrixXf& Iy)
{
  const int rows = img.getHeight();
  const int cols = img.getWidth();

  Ix.setZero(rows, cols);
  Iy.setZero(rows, cols);

  // Temporary buffers: one row of floats for the horizontal pass
  Eigen::MatrixXf tempX(rows, cols), tempY(rows, cols);
  tempX.setZero();
  tempY.setZero();

  // --- Horizontal pass ---
  // tempX[r][c] = -img[r][c-1] + img[r][c+1]          (kernel [-1 0 1])
  // tempY[r][c] =  img[r][c-1] + 2*img[r][c] + img[r][c+1]  (kernel [1 2 1])
  for (int r = 0; r < rows; ++r)
  {
    const uint8_t* row_ptr = img.ptrLine<uint8_t>(r);
    for (int c = 1; c < cols - 1; ++c)
    {
      const float left  = static_cast<float>(row_ptr[c - 1]);
      const float mid   = static_cast<float>(row_ptr[c]);
      const float right = static_cast<float>(row_ptr[c + 1]);
      tempX(r, c)       = right - left;
      tempY(r, c)       = left + 2.f * mid + right;
    }
  }

  // --- Vertical pass ---
  // Ix[r][c] = tempX[r-1][c] + 2*tempX[r][c] + tempX[r+1][c]   (kernel [1 2 1]ᵀ)
  // Iy[r][c] = -tempY[r-1][c] + tempY[r+1][c]                   (kernel [-1 0 1]ᵀ)
  for (int r = 1; r < rows - 1; ++r)
  {
    for (int c = 1; c < cols - 1; ++c)
    {
      Ix(r, c) = tempX(r - 1, c) + 2.f * tempX(r, c) + tempX(r + 1, c);
      Iy(r, c) = tempY(r + 1, c) - tempY(r - 1, c);
    }
  }
  // Border pixels remain zero (consistent with OpenCV's BORDER_CONSTANT)
}

// ---------------------------------------------------------------------------
// gaussianBlur
// ---------------------------------------------------------------------------
void mola::vision::gaussianBlur(const mrpt::img::CImage& in, mrpt::img::CImage& out, float sigma)
{
  // Build 1D kernel
  const int half_k = static_cast<int>(std::ceil(2.0f * sigma));
  const int k_size = 2 * half_k + 1;

  std::vector<float> kernel(k_size);
  float              kernel_sum = 0.f;
  for (int i = 0; i < k_size; ++i)
  {
    const float x = static_cast<float>(i - half_k);
    kernel[i]     = std::exp(-0.5f * x * x / (sigma * sigma));
    kernel_sum += kernel[i];
  }
  for (auto& v : kernel) v /= kernel_sum;

  const int rows = in.getHeight();
  const int cols = in.getWidth();

  // Horizontal pass: in → temp (float)
  Eigen::MatrixXf temp(rows, cols);
  for (int r = 0; r < rows; ++r)
  {
    const uint8_t* row_ptr = in.ptrLine<uint8_t>(r);
    for (int c = 0; c < cols; ++c)
    {
      float acc = 0.f;
      for (int k = -half_k; k <= half_k; ++k)
      {
        const int cc = std::max(0, std::min(cols - 1, c + k));
        acc += kernel[k + half_k] * static_cast<float>(row_ptr[cc]);
      }
      temp(r, c) = acc;
    }
  }

  // Vertical pass: temp → out (uint8)
  out.resize(cols, rows, mrpt::img::CH_GRAY);
  for (int r = 0; r < rows; ++r)
  {
    uint8_t* out_ptr = out.ptrLine<uint8_t>(r);
    for (int c = 0; c < cols; ++c)
    {
      float acc = 0.f;
      for (int k = -half_k; k <= half_k; ++k)
      {
        const int rr = std::max(0, std::min(rows - 1, r + k));
        acc += kernel[k + half_k] * temp(rr, c);
      }
      out_ptr[c] = static_cast<uint8_t>(std::max(0.f, std::min(255.f, acc)));
    }
  }
}
