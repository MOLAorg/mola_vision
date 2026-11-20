# mola_libvision

C++ data structures and algorithms for classic computer vision, the successor
to MRPT's removed `mrpt::vision` module. Built on MRPT 3.x + Eigen + TBB
(no OpenCV, no Ceres).

## Geometry with real (distorted) cameras

- `camera_model.h`: pinhole, radial-tangential (`plumb_bob`) and equidistant
  fisheye (Kannala-Brandt) models with analytic Jacobians with respect to the
  point and the intrinsics, unprojection to bearings (fisheye beyond 90 deg),
  conversion to/from `mrpt::img::TCamera`; `IntrinsicsParameterization`
  selects which intrinsics are free.
- `minimal_solvers.h`, `robust_estimators.h`: five-point essential, seven and
  eight-point fundamental, homography, rotation-only and P3P solvers on
  bearing vectors, multi-view triangulation, GRIC, and MSAC wrappers.
- `focal_initialization.h`: shared focal length from fundamental matrices,
  rotation homographies, or a model-aware two-view consensus sweep.
- `pnp_solver.h`: PnP and P3P-MSAC PnP on distorted pixels.
- `bundle_adjustment.h`: GTSAM bundle adjustment with intrinsics as shared
  variables, points at infinity, rotation-only motion, robust losses and
  marginal covariances of the intrinsics (the basis of camera
  self-calibration without a pattern).

## License

`mola_libvision` is released under the **GNU General Public License v3.0 or
later** (`GPL-3.0-or-later`); see the [LICENSE](LICENSE) file.

Some components are adapted from permissively-licensed projects and retain their
original copyright notices in the corresponding source files:
- ORB descriptor bit-pattern table from **OpenCV** (BSD-3-Clause).
- Bundle-adjustment / marginalization structure inspired by **Basalt**
  (BSD-3-Clause).
- Algorithms adapted from **lightweight_vio** (MIT).
