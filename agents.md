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
  image_0/image_1, metric depth from `matchStereo`, PnP tracking (true metric
  scale; scale-anchored stereo BA still pending). KITTI/TUM integration tests.

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
