# mola_vision: Visual SLAM for the MOLA framework

## Current Status (March 2026)

### What exists

**mola_basalt_vio** — A MOLA module wrapping the BSD-licensed [Basalt VIO](https://gitlab.com/VladyslavUsenko/basalt) library:
- Inherits `FrontEndBase` + `LocalizationSourceBase`.
- Supports monocular and stereo camera input with optional IMU.
- Multi-camera synchronization (configurable tolerance).
- Camera model conversion: MRPT `TCamera` → Basalt `GenericCamera` (pinhole, plumb-bob, kannala-brandt).
- Image pipeline: `CObservationImage` → OpenCV `cv::Mat` → Basalt `ImageData` (8/16-bit).
- Publishes `LocalizationUpdate` (pose + covariance).
- Threaded: worker pool for image processing, `tbb::concurrent_bounded_queue` for VIO output.
- IMU callback is **incomplete** (acceleration extraction started, integration TODO).
- Map publishing (`doPublishUpdatedMap`) is **stubbed**.
- MolaViz visualization parameters are parsed but **not wired** to actual rendering.
- Tested only with KITTI stereo (no IMU) via provided launch YAML.

**mola_libvision** — Embryonic reusable C++ library:
- Contains only an ORB descriptor bit-pattern lookup table (`orb_point_pairs.h`, from OpenCV BSD code).
- `orb_detector.cpp` is a stub (~10 lines, just includes the header).
- Unit tests directory exists but tests are **disabled** in CMake.

### External code available for reuse

**lightweight_vio** (`/home/jlblanco/code/lightweight_vio`, MIT license):
- Full stereo VO, stereo VIO, and RGBD VO pipeline (~18.5 K lines).
- Feature detection (goodFeaturesToTrack), KLT optical flow tracking, grid-based distribution.
- On-manifold IMU preintegration (Forster et al.) with bias Jacobians.
- Gravity estimation from visual constraints and gravity-aligned coordinate transform.
- Ceres-based optimization: PnP (single-frame), sliding-window BA, inertial optimizer.
- `MapPoint` with multi-view covariance and statistical uncertainty propagation.
- `Frame` supporting STEREO / RGBD / MONOCULAR types.
- RGBD depth uncertainty model (σ² = a·d² + b·d + c).
- Pangolin-based viewer with 3D map, trajectories, uncertainty ellipsoids, depth heatmaps.

**Basalt** (already vendored as submodule, BSD-3-Clause):
- Optical flow, VIO estimator, calibration, marginalization, factor graph.

**GTSAM** (BSD-2-Clause, external dependency):
- Factor graph / incremental smoothing (iSAM2), Lie group SE3/SO3, robust kernels.
- Candidate back-end for full SLAM (loop closure, global optimization).

---

## Project Goal

Build a family of MOLA modules for camera-based SLAM covering five configurations, all sharing a common reusable library (`mola_libvision`):

| # | Configuration | IMU | Priority |
|---|---------------|-----|----------|
| 5 | RGBD SLAM | No | **Highest** |
| 1 | Monocular VO/SLAM | No | High |
| 2 | Monocular Visual-Inertial SLAM | Yes | Medium |
| 3 | Stereo Visual SLAM | No | Medium |
| 4 | Stereo Visual-Inertial SLAM | Yes | Medium |

Each module:
- Is a MOLA `FrontEndBase` + `LocalizationSourceBase` (+ `MapSourceBase`).
- Consumes `CObservation{Image,RGBD,IMU}` via `onNewObservation`.
- Publishes poses and maps through standard MOLA interfaces.
- Integrates with MolaViz for 3D map view and 2D image+features overlay.
- Has unit / integration tests and example launch YAMLs.

---

## Architecture

```
mola_libvision          (reusable C++ library, deps: MRPT v3 + Eigen + TBB. NO OpenCV, NO Ceres)
  ├── image processing primitives (Sobel, Gaussian blur, CLAHE)
  ├── feature detection (goodFeaturesToTrack, FAST, ORB, grid distributor)
  ├── feature tracking (pyramidal LK optical flow, fundamental-matrix RANSAC filter)
  ├── geometric algorithms (triangulation, essential matrix decomposition, undistort batch)
  ├── stereo matching & depth from disparity
  ├── RGBD depth processing & uncertainty model
  ├── data structures (MapPoint, Keyframe, KeyframeDatabase, covisibility graph)
  ├── LM optimizer with Schur complement (Eigen LDLT, no Ceres)
  ├── PnP pose estimation (custom LM solver)
  ├── sliding-window bundle adjustment (QR landmark marginalization)
  ├── IMU preintegration (on-manifold, Forster et al.)
  ├── gravity estimation & alignment
  ├── keyframe selection policy
  └── uncertainty propagation (MapPoint covariance)

mola_visual_slam        (MOLA module: mono/stereo/RGBD SLAM, no IMU)
mola_visual_inertial    (MOLA module: mono/stereo VIO/VI-SLAM)
mola_basalt_vio         (existing Basalt wrapper, kept as alternative backend)
```

### Reusable components to put in `mola_libvision`

All library code uses MRPT types (`mrpt::img::CImage`, `mrpt::poses::CPose3D`, `mrpt::math::CMatrixDouble`) so it integrates naturally with the rest of MOLA. **No OpenCV dependency** — pure MRPT (v3) + Eigen + Ceres/GTSAM. MRPT v3 already provides image I/O (via STB), resize, color conversion, pixel access, KLT corner response, image pyramids, stereo rectification, undistortion, and camera models. What MRPT does **not** provide (and must be implemented in `mola_libvision`) are the core CV algorithms listed below.

| Component | Primary source | Notes |
|-----------|---------------|-------|
| `FeatureDetector` | lightweight_vio `FeatureTracker` | goodFeaturesToTrack, grid distribution, mask handling |
| `FeatureTracker` | lightweight_vio `FeatureTracker` | KLT optical flow, fundamental-matrix RANSAC filter |
| `StereoMatcher` | lightweight_vio `Frame` stereo code | Disparity, epipolar constraints, depth from stereo |
| `RGBDDepthProcessor` | lightweight_vio `Frame` RGBD code | Depth filtering, uncertainty model, dense cloud |
| `MapPoint` | lightweight_vio `MapPoint` | 3D point, observations, covariance, uncertainty propagation |
| `Keyframe` | lightweight_vio `Frame` | Image + features + pose + IMU data + map-point links |
| `KeyframeDatabase` | new | Keyframe storage, covisibility graph, place recognition hooks |
| `LevenbergMarquardt` | Basalt-inspired, new | Generic LM solver, Eigen LDLT, Huber kernel, adaptive λ |
| `PnPSolver` | lightweight_vio `PnPOptimizer` | Single-frame pose from 3D–2D via LM solver (no Ceres) |
| `SlidingWindowBA` | Basalt-inspired + lightweight_vio | QR landmark marginalization, Schur complement, TBB parallel |
| `MargPrior` | Basalt `MargHelper` | Hessian prior from marginalized keyframes |
| `IMUPreintegrator` | lightweight_vio `IMUHandler` | On-manifold ΔR/ΔV/ΔP, bias Jacobians |
| `GravityEstimator` | lightweight_vio `IMUHandler` | Gravity direction from visual+inertial constraints |
| `KeyframeSelector` | lightweight_vio `Estimator` | Grid-coverage + parallax + time policy |
| `LoopDetector` | new (DBoW2 / NetVLAD stub) | Visual place recognition interface |
| `PoseGraphOptimizer` | new (GTSAM iSAM2) | Global optimization with loop-closure factors |

---

## CV Primitives to Implement in `mola_libvision` (No OpenCV)

MRPT v3 (`mrpt::img::CImage`) already provides: image I/O (JPG/PNG/BMP via STB), `scaleImage()`, `grayscale()`, `colorImage()`, `getAsMatrix()`, pixel access (`at<>()`), `CImagePyramid`, `CUndistortMap`, `CStereoRectifyMap`, `TCamera` (pinhole + plumb-bob + Kannala-Brandt), `projectPoint_with_distortion()`, `undistort_point()`, and KLT corner response (`KLT_response()`).

The following algorithms must be implemented from scratch or ported from permissive-licensed sources. All operate on `mrpt::img::CImage` (internally backed by a raw buffer).

### Feature Detection

| Algorithm | Description | Source / Approach | Priority |
|-----------|------------|-------------------|----------|
| `goodFeaturesToTrack()` | Shi-Tomasi corner detector: compute min-eigenvalue response over image, non-maximum suppression, sort by score, return top-N | Port from lightweight_vio; eigenvalue computation via 2×2 structure tensor (Ix², Iy², IxIy from Sobel gradients). MRPT already has `KLT_response()` which computes the Shi-Tomasi score at a single point — extend to whole-image with NMS. | **P0** |
| `GridFeatureDistributor` | Divide image into cells, detect per-cell, enforce min-distance and max-per-cell | Port from lightweight_vio `FeatureTracker::extract_new_features()` grid logic | **P0** |
| `FAST` detector | Segment test on Bresenham circle of 16 pixels, with NMS | Classic algorithm, ~200 lines. Used by ORB and Basalt internally. | P1 |
| `ORBDetector` | Oriented FAST + rotated BRIEF descriptor | Already have bit-pattern LUT in `orb_point_pairs.h`; add oriented FAST keypoint + descriptor extraction | P2 |

### Feature Tracking

| Algorithm | Description | Source / Approach | Priority |
|-----------|------------|-------------------|----------|
| `calcOpticalFlowPyrLK()` | Pyramidal Lucas-Kanade optical flow tracker | Port from lightweight_vio. Core math: for each pyramid level, solve 2×2 linear system (Hessian of image gradient patch) for sub-pixel displacement. Use `CImagePyramid` for multi-scale. ~300 lines of core math + iteration logic. | **P0** |
| `FundamentalMatrixFilter` | RANSAC-based fundamental matrix estimation + inlier classification | Port from lightweight_vio `apply_fundamental_matrix_filter()`. 8-point algorithm + RANSAC. Eigen SVD for F estimation. | **P0** |

### Image Processing

| Algorithm | Description | Source / Approach | Priority |
|-----------|------------|-------------------|----------|
| `gaussianBlur()` | Separable Gaussian filter (σ-parameterized) | Eigen row/col convolution on `CImage::getAsMatrix()`. ~50 lines. Needed before corner detection. | **P0** |
| `sobelGradients()` | Sobel Ix, Iy gradient images (3×3 or 5×5 kernel) | Separable convolution with [−1,0,1] and [1,2,1] kernels. ~30 lines. Core dependency for Shi-Tomasi and Harris. | **P0** |
| `CLAHE` | Contrast-Limited Adaptive Histogram Equalization | Port from lightweight_vio players or implement from the original paper (Zuiderveld 1994). Tile-based histogram + bilinear interpolation of mappings. ~150 lines. | P1 |
| `histogramEqualize()` | Global histogram equalization | Cumulative histogram LUT. ~20 lines. | P1 |

### Geometric Algorithms

| Algorithm | Description | Source / Approach | Priority |
|-----------|------------|-------------------|----------|
| `undistortPoints()` | Batch undistort pixel→normalized coords (pinhole + fisheye) | MRPT already has `undistort_point()` for single points; wrap in batch API operating on feature vectors. | **P0** |
| `essentialMatrixFromF()` | E = K2ᵀ · F · K1 | Trivial 3×3 multiply. | P1 |
| `decomposeEssentialMatrix()` | Extract R, t from E via SVD | Standard decomposition + chirality check. ~50 lines. | P1 |
| `triangulatePoints()` | Linear triangulation (DLT) from two views | SVD of 4×n design matrix per point. Already partially in MRPT `mrpt::vision`. | **P0** |

### Stereo-Specific

| Algorithm | Description | Source / Approach | Priority |
|-----------|------------|-------------------|----------|
| `stereoMatchByKLT()` | Horizontal KLT search along epipolar line for rectified stereo | Port from lightweight_vio `Frame::compute_stereo_matches()`: 1D optical flow constrained to epipolar line. | P1 |
| `depthFromDisparity()` | depth = f·B / disparity, with validity mask | Trivial formula + outlier thresholds. | P1 |

### Data Structures (no OpenCV equivalents needed)

- `cv::Mat` → `mrpt::img::CImage` (for images) or `Eigen::MatrixXf` (for float matrices / gradient maps).
- `cv::Point2f` → `mrpt::math::TPoint2Df` or `Eigen::Vector2f`.
- `cv::Rect` → `mrpt::math::TBoundingBoxf` or simple struct.
- `cv::FileStorage` → MRPT YAML (`mrpt::containers::yaml`) — already used throughout MOLA.
- Drawing (circles, lines, text for debug viz) → `mrpt::img::CCanvas` methods on `CImage`, or directly via MolaViz 2D overlay.

---

## Detailed TO-DO List

### Phase 0: Foundation — `mola_libvision` library build-out

- [ ] **P0.1** Set up proper CMake for `mola_libvision`: find Eigen3, MRPT (v3), TBB, optional GTSAM. **No OpenCV, no Ceres**.
- [ ] **P0.2** Implement `sobelGradients()`: separable 3×3 Sobel filter on `CImage::getAsMatrix()` → Eigen matrices Ix, Iy. Unit test: gradient of synthetic step-edge, verify magnitude and direction. `CImage::getAsMatrix()` at present returns a copy, which is really inefficient. Investigate if it's possible to add a CImage helper to return an Eigen::Map view of the existing CImage data.
- [ ] **P0.3** Implement `gaussianBlur()`: separable Gaussian on `CImage` with configurable σ and kernel size. Unit test: blur a delta-function image, verify Gaussian shape.
- [ ] **P0.4** Implement `goodFeaturesToTrack()`: structure tensor from Sobel gradients → min-eigenvalue score map → NMS → sorted top-N. Operates on `mrpt::img::CImage`. Unit test: detect corners on synthetic checkerboard, verify locations.
- [ ] **P0.5** Implement `GridFeatureDistributor`: divide image into cells, call detector per cell, enforce min-distance. Port grid logic from lightweight_vio. Unit test: verify spatial distribution on real image.
- [ ] **P0.6** Implement `calcOpticalFlowPyrLK()`: pyramidal Lucas-Kanade using `CImagePyramid`. For each level: compute gradient patch, solve 2×2 system, propagate to next level. Port core math from lightweight_vio. Unit test: track features across synthetic translated image pair with known displacement.
- [ ] **P0.7** Implement `FundamentalMatrixFilter`: 8-point algorithm + RANSAC, Eigen SVD. Port from lightweight_vio `apply_fundamental_matrix_filter()`. Unit test: filter outliers from synthetic correspondences with known F.
- [ ] **P0.8** Implement batch `undistortPoints()`: wrap MRPT's `undistort_point()` for vector of features, supporting pinhole + Kannala-Brandt. Unit test: round-trip distort→undistort on known camera.
- [ ] **P0.9** Implement `triangulatePoints()`: linear DLT triangulation from two views via SVD. Unit test: triangulate synthetic correspondences, verify 3D error < ε.
- [ ] **P0.10** Port `MapPoint` class: 3D position, observation list, covariance, uncertainty propagation. Unit test: add/remove observations, verify covariance computation.
- [ ] **P0.11** Port `Keyframe` data structure: wraps `mrpt::img::CImage`, features, pose, depth, IMU data. Unit test: construction, feature extraction, serialization round-trip.
- [ ] **P0.12** Implement `LevenbergMarquardt` solver: generic templated LM optimizer using Eigen LDLT. Supports Huber robust kernel, diagonal damping (λ·diag(H)), adaptive λ update, convergence checks (relative cost decrease + step norm). Inspired by Basalt's approach but generic. ~200 lines. Unit test: solve Rosenbrock function, verify convergence.
- [ ] **P0.13** Implement `PnPSolver`: SE3 optimization from 3D–2D correspondences using the LM solver. Analytical Jacobians of reprojection error w.r.t. SE3 (Lie algebra parameterization via Sophus or MRPT). Huber loss + chi-square outlier rejection. Unit test: recover known pose from synthetic correspondences with outliers.
- [ ] **P0.14** Implement `SlidingWindowBA`: multi-keyframe bundle adjustment using LM solver with Schur complement. QR-based landmark marginalization (à la Basalt): linearize per-landmark block, Householder QR to eliminate landmark variables, solve reduced pose-only system with Eigen LDLT. TBB parallelization over landmark blocks. Optional IMU preintegration factors. Unit test: optimize a small 5-frame window with known ground truth.
- [ ] **P0.15** Implement `MargPrior`: Hessian-based prior from marginalized keyframes (encode information from removed frames as a quadratic cost on remaining variables). Inspired by Basalt's `MargHelper`. Unit test: marginalize one frame from a 3-frame problem, verify equivalent solution.
- [ ] **P0.16** Port `StereoMatcher`: stereo KLT along epipolar line + `depth = f·B/disparity`, validity mask. Port from lightweight_vio `compute_stereo_matches()`. Unit test: compute depth on rectified stereo pair, compare to ground truth.
- [ ] **P0.17** Port `RGBDDepthProcessor`: depth map filtering, quadratic uncertainty model (σ² = a·d² + b·d + c), depth-to-3D back-projection. Unit test: convert depth image to point cloud, verify against known geometry.
- [ ] **P0.18** Port `IMUPreintegrator`: on-manifold preintegration, bias Jacobians, bias update. Unit test: preintegrate synthetic constant-acceleration IMU, verify ΔR/ΔV/ΔP.
- [ ] **P0.19** Port `GravityEstimator`: gravity direction from visual+inertial constraints, coordinate transform. Unit test: estimate gravity from synthetic VIO data.
- [ ] **P0.20** Implement `KeyframeSelector`: configurable policy (grid coverage ratio, parallax, time threshold). Unit test: verify keyframe decisions on scripted sequences.
- [ ] **P0.21** Add `KeyframeDatabase` with covisibility graph: shared map-point counting, neighbor queries. Unit test: build graph from synthetic data, verify neighbors.
- [ ] **P0.22** Stub `LoopDetector` interface (virtual base class). Concrete DBoW2 or learned-descriptor implementation deferred.
- [ ] **P0.23** Stub `PoseGraphOptimizer` interface. Concrete GTSAM iSAM2 implementation deferred to Phase 5.
- [ ] **P0.24** (Deferred) Implement `CLAHE`: tile-based adaptive histogram equalization. ~150 lines. Needed for robust tracking under lighting changes.
- [ ] **P0.25** (Deferred) Implement `FAST` detector: segment test on 16-pixel Bresenham circle + NMS. ~200 lines. Needed for ORB.
- [ ] **P0.26** (Deferred) Implement `ORBDetector`: oriented FAST keypoints + rotated BRIEF descriptors using existing `orb_point_pairs.h` LUT. Needed for loop closure (Phase 5).
- [ ] **P0.27** (Deferred) Implement `decomposeEssentialMatrix()`: SVD decomposition of E → (R,t) with chirality check. Needed for monocular init (Phase 2).

### Phase 1: RGBD SLAM module — `mola_rgbd_slam` (highest priority)

- [ ] **P1.1** Create MOLA module skeleton: `FrontEndBase` + `LocalizationSourceBase` + `MapSourceBase`, `register.cpp`, CMakeLists.txt, package.xml.
- [ ] **P1.2** Implement `onNewObservation` for `CObservationRGBD` (or paired `CObservationImage` + depth): extract RGB + depth, create `Keyframe`.
- [ ] **P1.3** RGBD initialization: first frame → extract features → back-project to 3D using depth → create initial map points.
- [ ] **P1.4** Frame-to-frame tracking: KLT tracking → PnP pose estimation → outlier rejection.
- [ ] **P1.5** New map-point creation: untracked features with valid depth → triangulate/back-project → add to map.
- [ ] **P1.6** Keyframe management: decide keyframe, add to sliding window, run BA in background thread.
- [ ] **P1.7** Map publishing: publish `KeyframePointCloudMap` or sparse landmark map via `MapSourceBase`.
- [ ] **P1.8** MolaViz integration — 3D view: render map points (colored by depth or RGB), keyframe frustums, estimated trajectory.
- [ ] **P1.9** MolaViz integration — 2D view: show current image with tracked features (green=inlier, red=outlier), feature trails.
- [ ] **P1.10** Launch YAML for TUM RGB-D dataset.
- [ ] **P1.11** Launch YAML for live RealSense D435/D455.
- [ ] **P1.12** Unit tests:
  - Initialization from single RGBD frame.
  - Tracking across 10-frame synthetic sequence.
  - Sliding-window BA convergence.
  - Map point creation and culling.
- [ ] **P1.13** Integration test: run on TUM fr1/desk, compare ATE to ground truth (target: < 5 cm RMSE).

### Phase 2: Monocular Visual SLAM — `mola_visual_slam` (mono mode)

- [ ] **P2.1** Create MOLA module skeleton (or extend `mola_visual_slam` with mode parameter).
- [ ] **P2.2** Monocular initialization: two-view geometry (essential matrix via 5-point RANSAC), triangulation, scale from first baseline.
- [ ] **P2.3** Frame-to-frame tracking: feature tracking → PnP → outlier rejection (same as RGBD but without depth prior).
- [ ] **P2.4** New map-point creation via triangulation from multiple views (sufficient parallax check).
- [ ] **P2.5** Keyframe + sliding-window BA (visual-only factors).
- [ ] **P2.6** Scale drift mitigation: enforce consistent scale in BA window.
- [ ] **P2.7** MolaViz integration (reuse RGBD viz, adapt for mono — no dense cloud).
- [ ] **P2.8** Launch YAML for KITTI (monocular, image_0 only).
- [ ] **P2.9** Unit tests:
  - Two-view initialization geometry.
  - Triangulation accuracy.
  - Scale consistency over 100 frames.
- [ ] **P2.10** Integration test: KITTI seq 00 monocular, compare to ground truth.

### Phase 3: Stereo Visual SLAM — `mola_visual_slam` (stereo mode)

- [ ] **P3.1** Stereo initialization: extract features in left, stereo-match to right, triangulate for immediate depth.
- [ ] **P3.2** Integrate `StereoMatcher` from `mola_libvision` into the tracking pipeline.
- [ ] **P3.3** Map-point creation with stereo depth (no parallax wait needed).
- [ ] **P3.4** Sliding-window BA with stereo reprojection factors.
- [ ] **P3.5** Launch YAML for KITTI stereo, EuRoC stereo.
- [ ] **P3.6** Unit tests:
  - Stereo matching accuracy on rectified pairs.
  - Depth estimation vs ground truth.
- [ ] **P3.7** Integration test: KITTI seq 00 stereo, EuRoC MH_01.

### Phase 4: Visual-Inertial modules — `mola_visual_inertial` (mono + stereo)

- [ ] **P4.1** Create MOLA module skeleton with IMU support.
- [ ] **P4.2** IMU data handling: accumulate `CObservationIMU` between keyframes, preintegrate using `IMUPreintegrator`.
- [ ] **P4.3** VIO initialization: collect ≥5 keyframes → gravity + bias estimation via `GravityEstimator` → transform to gravity-aligned frame.
- [ ] **P4.4** Sliding-window BA with IMU preintegration factors (velocity, bias states per keyframe).
- [ ] **P4.5** Monocular VIO mode: scale observable from IMU → no scale drift.
- [ ] **P4.6** Stereo VIO mode: redundant scale from stereo + IMU for robustness.
- [ ] **P4.7** MolaViz: show gravity vector, IMU bias evolution plots.
- [ ] **P4.8** Launch YAML for EuRoC VIO (mono and stereo).
- [ ] **P4.9** Unit tests:
  - IMU preintegration numerical accuracy.
  - Gravity estimation from synthetic data.
  - VIO initialization convergence.
- [ ] **P4.10** Integration test: EuRoC MH_01 VIO, compare to ground truth (target: < 0.5% trajectory length).

### Phase 5: Loop closure and global optimization

- [ ] **P5.1** Implement `LoopDetector` with DBoW2 (or a learned descriptor approach).
- [ ] **P5.2** Implement `PoseGraphOptimizer` with GTSAM iSAM2.
- [ ] **P5.3** Loop-closure pipeline: detect candidate → geometric verification (PnP) → add loop factor → global optimization.
- [ ] **P5.4** Map correction: propagate pose-graph update to map points.
- [ ] **P5.5** Integrate into all five SLAM modules (loop closure is shared logic).
- [ ] **P5.6** Unit tests:
  - Loop detection precision/recall on known revisits.
  - Pose graph convergence with synthetic loop factors.
- [ ] **P5.7** Integration test: KITTI seq 00 (has loop), verify drift reduction.

### Phase 6: Polish and hardening

- [ ] **P6.1** Relocalization: re-localize after tracking failure using keyframe database.
- [ ] **P6.2** Map saving/loading: serialize keyframe database + map points to disk.
- [ ] **P6.3** Performance profiling: ensure real-time on target hardware (30 fps mono, 15 fps stereo/RGBD).
- [ ] **P6.4** ROS 2 launch files for live cameras (RealSense, ZED, USB webcam).
- [ ] **P6.5** Documentation: usage guide, parameter tuning, architecture diagram.
- [ ] **P6.6** CI: build + unit test pipeline in GitHub Actions.

---

## MolaViz Integration Plan

### 3D Map View (shared across all modules)

- Sparse map points (colored by: RGB from image / depth / observation count).
- Dense point cloud (RGBD module only, with depth-colored heatmap option).
- Keyframe frustum wireframes (current = green, past = gray).
- Estimated trajectory line.
- Ground-truth trajectory overlay (when available).
- Gravity vector arrow (VIO modules).
- Loop-closure edges (Phase 5).
- Follow-camera mode (auto-track current pose).

### 2D Image View (shared across all modules)

- Current frame with:
  - Tracked features (green circles = inliers, red = outliers).
  - Optical flow trails (lines from previous to current position).
  - Feature IDs (optional, for debugging).
  - Stereo matches (horizontal lines to right image, stereo mode).
  - Depth overlay (color-coded, RGBD mode).
- Side panel (optional): right stereo image or depth heatmap.

### Implementation approach

Use the existing MolaViz `VizInterface` API:
- `update_3d_object(name, CRenderizable)` for map points, frustums, trajectory.
- `subwindow_update_visualization(name, CObservation)` for 2D image panel.
- Visualization update in `spinOnce()` with decimation to avoid GUI bottleneck.
- All viz code in a shared helper (e.g., `mola_libvision/VisualSLAMViz.h`) so every module reuses the same rendering logic.

---

## Unit Test Strategy

| Layer | What to test | Framework |
|-------|-------------|-----------|
| `mola_libvision` | Each component in isolation with synthetic data | Google Test (`ament_cmake_gtest`) |
| MOLA modules | Module lifecycle: init → process N frames → check output | Google Test + MOLA test harness |
| Integration | Full pipeline on small real dataset clips (TUM, EuRoC, KITTI) | CTest + trajectory evaluation script |

**Synthetic test data**: generate programmatically (known camera, known 3D points, rendered 2D projections, synthetic IMU) so tests are deterministic and fast. Store small real-image snippets (5–10 frames) as test fixtures for tracker and matcher tests.

**CI expectations**: all unit tests pass; integration tests run nightly (datasets too large for per-commit CI).

---

## Notes on code reuse and attribution

- Code ported from **lightweight_vio** (MIT): add header comment `// Adapted from lightweight_vio (MIT License), original authors: Choi et al.`
- Code ported from **Basalt** (BSD-3-Clause): add header comment `// Adapted from Basalt (BSD-3-Clause), original authors: Usenko et al.`
- ORB bit-pattern LUT from **OpenCV** (BSD-3-Clause): already attributed in `orb_point_pairs.h`.
- GTSAM used as external dependency (BSD-2-Clause), no code copying needed.
- **No OpenCV dependency** — all CV primitives implemented in `mola_libvision` using MRPT v3 image types + Eigen math.
