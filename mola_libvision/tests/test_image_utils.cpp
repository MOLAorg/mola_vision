/* -------------------------------------------------------------------------
 * mola_libvision unit tests: image_utils (Sobel, Gaussian blur, Eigen::Map)
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <mola_libvision/image_utils.h>

#include <cmath>

using namespace mola::vision;

// ---------------------------------------------------------------------------
// Helper: create a synthetic grayscale CImage from a lambda
template <typename F>
static mrpt::img::CImage makeImage(int W, int H, F pixel_fn)
{
  mrpt::img::CImage img(W, H, mrpt::img::CH_GRAY);
  for (int r = 0; r < H; ++r)
  {
    uint8_t* row = img.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c)
      row[c] = static_cast<uint8_t>(std::max(0, std::min(255, pixel_fn(c, r))));
  }
  return img;
}

// ---------------------------------------------------------------------------
// Eigen::Map zero-copy view
// ---------------------------------------------------------------------------
TEST(ImageUtils, EigenMapZeroCopy)
{
  const int W = 10, H = 8;
  auto      img = makeImage(W, H, [](int c, int r) { return c + r * 10; });

  auto map = asEigenMap(img);
  EXPECT_EQ(map.rows(), H);
  EXPECT_EQ(map.cols(), W);

  // Values should match pixel-by-pixel
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) EXPECT_EQ(static_cast<int>(map(r, c)), img.at<uint8_t>(c, r));

  // Zero-copy: modifying through map changes the image
  auto map_mutable  = asEigenMap(img);
  map_mutable(0, 0) = 99;
  EXPECT_EQ(static_cast<int>(img.at<uint8_t>(0, 0)), 99);
}

// ---------------------------------------------------------------------------
// Sobel gradients: step edge
// ---------------------------------------------------------------------------
TEST(ImageUtils, SobelStepEdge)
{
  // Vertical step edge at column 10: left=0, right=200
  const int W = 20, H = 20;
  auto      img = makeImage(W, H, [](int c, int /*r*/) { return c < 10 ? 0 : 200; });

  Eigen::MatrixXf Ix, Iy;
  sobelGradients(img, Ix, Iy);

  // At the edge (columns 9-11, middle rows), Ix should be strongly positive
  const int mid = H / 2;
  EXPECT_GT(Ix(mid, 10), 100.f) << "Expected strong positive Ix at vertical step edge";

  // Iy should be near zero everywhere for a vertical edge
  EXPECT_NEAR(Iy(mid, 5), 0.f, 2.f) << "Expected near-zero Iy in flat region";
  EXPECT_NEAR(Iy(mid, 15), 0.f, 2.f) << "Expected near-zero Iy in flat region";

  // Border pixels must be zero
  for (int c = 0; c < W; ++c)
  {
    EXPECT_FLOAT_EQ(Ix(0, c), 0.f);
    EXPECT_FLOAT_EQ(Ix(H - 1, c), 0.f);
  }
}

TEST(ImageUtils, SobelHorizontalEdge)
{
  // Horizontal step edge at row 10
  const int W = 20, H = 20;
  auto      img = makeImage(W, H, [](int /*c*/, int r) { return r < 10 ? 0 : 200; });

  Eigen::MatrixXf Ix, Iy;
  sobelGradients(img, Ix, Iy);

  const int mid = W / 2;
  EXPECT_GT(Iy(10, mid), 100.f) << "Expected strong positive Iy at horizontal step edge";
  EXPECT_NEAR(Ix(5, mid), 0.f, 2.f);
  EXPECT_NEAR(Ix(15, mid), 0.f, 2.f);
}

// ---------------------------------------------------------------------------
// Gaussian blur: delta function → Gaussian shape
// ---------------------------------------------------------------------------
TEST(ImageUtils, GaussianBlurDeltaFunction)
{
  const int W = 31, H = 31;
  const int cx = W / 2, cy = H / 2;
  auto      img = makeImage(W, H, [&](int c, int r) { return (c == cx && r == cy) ? 255 : 0; });

  mrpt::img::CImage blurred;
  const float       sigma = 2.0f;
  gaussianBlur(img, blurred, sigma);

  // Centre pixel should be the maximum
  uint8_t centre_val = blurred.at<uint8_t>(cx, cy);
  EXPECT_GT(static_cast<int>(centre_val), 0);

  // All other pixels should be <= centre (Gaussian is maximal at centre)
  for (int r = 0; r < H; ++r)
  {
    const uint8_t* row = blurred.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c) EXPECT_LE(row[c], centre_val);
  }

  // Energy should be approximately preserved (within 10% due to border clamping)
  int sum_in = 255, sum_out = 0;
  for (int r = 0; r < H; ++r)
  {
    const uint8_t* row = blurred.ptrLine<uint8_t>(r);
    for (int c = 0; c < W; ++c) sum_out += row[c];
  }
  EXPECT_NEAR(sum_out, sum_in, sum_in * 0.15f);
}

TEST(ImageUtils, GaussianBlurFlat)
{
  // Flat image → should remain flat
  const int W = 20, H = 20;
  auto      img = makeImage(W, H, [](int, int) { return 128; });

  mrpt::img::CImage blurred;
  gaussianBlur(img, blurred, 1.5f);

  // Interior pixels (avoid border clamping effects at corners)
  for (int r = 3; r < H - 3; ++r)
  {
    const uint8_t* row = blurred.ptrLine<uint8_t>(r);
    for (int c = 3; c < W - 3; ++c) EXPECT_EQ(row[c], 128);
  }
}
