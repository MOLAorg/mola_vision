# mola_vision: AI agent instructions

`mola_vision` is the successor to the removed `mrpt::vision` module: reusable
classic computer vision for the MOLA SLAM framework, built on **MRPT 3.x**
(`mrpt::img::CImage`, `mrpt::poses`, Eigen, TBB). **No OpenCV, no Ceres.**

Packages:
- `mola_libvision` — reusable CV library (namespace `mola::vision`).
- `mola_visual_tracking` — demo MOLA module exercising the library.
- `mola_rgbd_slam` — RGB-D visual SLAM front-end (`mola::RgbdSlam`):
  detect + LK-track features, depth back-projection, robust PnP tracking, and
  sliding-window BA over keyframes; publishes pose + sparse map.
- `mola_visual_slam` — monocular/stereo visual SLAM front-end
  (`mola::VisualSlam`). `mode=mono`: essential-matrix bootstrap, PnP tracking,
  cross-keyframe triangulation, windowed BA (up-to-scale). `mode=stereo`: pairs
  image_0/image_1, metric depth from `matchStereo`, PnP tracking seeded by a
  constant-velocity model, windowed BA with a stereo-disparity residual that
  anchors metric scale. Optional `right_camera_pose` rectifies a raw
  (non-pre-rectified) rig; `right_camera_pose.x` must be POSITIVE, i.e. the
  right camera in the LEFT camera's frame. `rectify_output_size` ("W H") resizes
  the rectified canvas, which is how the rectified FIELD OF VIEW is chosen:
  rectification targets a PINHOLE model at the source focal length and keeps the
  same angular resolution, so the canvas decides how much of a fisheye frustum
  survives. Measured on heap-1, whole-mission error falls monotonically as the
  canvas SHRINKS (1920x1440 / 1440x1080 / 1000x760 give 1.26 / 0.45 / 0.25 m),
  so the rectified periphery costs more in camera-model and warp error than its
  wide baseline is worth; about 1000x760 (~70 deg horizontal) is the knee.
  `clahe_clip_limit` applies contrast-limited adaptive histogram equalization
  before detection and tracking, for scenes whose usable texture spans only a
  few grey levels: detection thresholds are relative to each grid cell, but the
  LK gradient-energy gate is absolute. The per-axis `fuse_sigma_*` parameters
  refine the isotropic fused covariance (each falls back to it when 0). The CLI
  prints the stereo epipolar residual binned by image radius, which checks
  rectification quality without any ground truth: on a correct rectification it
  is flat.
  `imu_label` turns on gyro-aided prediction: the inter-frame rotation comes
  from a `CObservationIMU` stream instead of the constant-velocity model
  (translation still constant-velocity), which is what keeps tracking alive
  through fast turns. It is a PREDICTION only - no inertial residual enters PnP
  or BA - so a missing or late IMU degrades back to constant velocity. Needs the
  camera- and IMU-on-robot extrinsics (from the observations or
  `camera_pose_on_robot` / `imu_pose_on_robot`) to rotate the gyro into the
  camera frame. `currentPose()` is the physical left
  camera; `currentRobotPose()` is the vehicle body, and needs the camera-on-robot
  extrinsic (from the observations' `cameraPose`, or `camera_pose_on_robot`).
  `mola-visual-slam-cli` runs it offline on a ROS1 bag or KITTI and writes a TUM
  trajectory; `--input-rosbag1` may be repeated, so a separate IMU bag joins the
  images as one time-sorted stream, and `--imu-topic` / `--imu-sensor-pose` turn
  gyro aiding on. When a `mola::NavStateFilter` module is present in the same MOLA
  system, each localized frame's VEHICLE pose is also fused into it as its own
  odometry source (`state_estimator_frame_id`, default `visual_odom`), which is
  how visual odometry reaches `mola_lidar_odometry` -- through the estimator's
  motion prior, not through any direct coupling. See the
  `lidar_visual_odometry_from_{kitti,grandtour}.yaml` launch files, both with a
  `MOLA_WITH_VISUAL_ODOM` A/B switch. Tests: `test_stereo_synthetic` (analytically ray-traced textured
  room; ground-truth-exact, no dataset needed) plus KITTI/TUM integration tests
  that skip unless their env vars are set.

## Build & test (ROS-agnostic; build_type cmake)
This package needs `mola_common` + the MRPT-3.x `mola_kernel`/`mola_viz` +
MRPT 3.x on the prefix path. ROS jazzy bundles MRPT 2.x, and the MRPT-3.x colcon
install is isolated (its setup re-chains ROS ahead of itself), so MRPT 2.x wins
`find_package` unless you hoist the 3.x prefixes to the front:
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash      # mola_common + MRPT-3.x mola_kernel/mola_viz
source ~/code/mrpt/install/setup.bash    # MRPT 3.x (source last)
export CMAKE_PREFIX_PATH="$(ls -d ~/code/mrpt/install/*/ | tr '\n' ':')${CMAKE_PREFIX_PATH}"
cd ~/ros2_ws/src/mola_vision
colcon build  --packages-select mola_libvision --event-handlers console_direct+
colcon test   --packages-select mola_libvision
colcon test-result --all
```
Do NOT source `~/ros2_cadtech_ws` (old MRPT-2.x `mola_kernel`, whose
`VizInterface` uses `mrpt::opengl` instead of `mrpt::viz`). The repo root carries
a `COLCON_IGNORE` marker; build a single package with `--base-paths .`.

## Coding standard (MANDATORY before every commit)
1. **clang-format**: run `clang-format-14 -i` on every changed `.h`/`.cpp`
   (config: repo `.clang-format`).
2. **clang-tidy**: changed files must be clean against the repo `.clang-tidy`
   (`bugprone-*` minus `easily-swappable-parameters`, plus
   `readability-braces-around-statements`, member-init, etc.). Verify with
   `clang-tidy -p build/mola_libvision <file>`. In particular: always brace
   single-statement `if`/`for`; do index arithmetic for Eigen `.block`/`.segment`
   offsets in `Eigen::Index` (not `int`) to avoid implicit-widening warnings.
3. Also follow the root `common.md` style (no one-line ifs, one var per line,
   no em-dashes, American spelling, anonymous namespaces over `static`).

## Notes
- MRPT 3.x guarantees **C++17**, so `Eigen::aligned_allocator` is **not** needed
  for fixed-size Eigen types in STL containers (the default allocator honors
  over-alignment).
- Public APIs use MRPT types so the wider MRPT/MOLA audience can reuse them.
- Working plan with task checklist: `~/plans/mola_vision_plan.md`.
- `mola::VisualSlam` does all its work synchronously in `onNewObservation()`
  (`spinOnce()` is a no-op), so it is deterministic and can run as an extra
  front-end inside another app's loop. `mola_visual_slam/params/*.yaml` are
  standalone parameter files for exactly that: `mola-lidar-odometry-cli
  --module mola::VisualSlam --module-param-file <file>` runs LiDAR and visual
  odometry into one state estimator, in a batch tool that drops no scans. The
  `mola-cli-launchs/*.yaml` files carry the same keys inline instead.
