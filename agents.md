# mola_vision: AI agent instructions

`mola_vision` is the successor to the removed `mrpt::vision` module: reusable
classic computer vision for the MOLA SLAM framework, built on **MRPT 3.x**
(`mrpt::img::CImage`, `mrpt::poses`, Eigen, TBB). **No OpenCV, no Ceres.**

Packages:
- `mola_libvision` — reusable CV library (namespace `mola::vision`).
- `mola_visual_tracking` — demo MOLA module exercising the library.

## Build & test (ROS-agnostic; build_type cmake)
This package needs `mola_common` + MRPT 3.x on the prefix path. MRPT 2.x is the
default in the ROS workspace, so source the MRPT 3 colcon workspace first:
```bash
source ~/code/mrpt/install/setup.bash
cd ~/ros2_ws/src/mola_vision
colcon build  --packages-select mola_libvision --event-handlers console_direct+
colcon test   --packages-select mola_libvision
colcon test-result --all
```

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
