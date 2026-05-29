# mola_libvision

C++ data structures and algorithms for classic computer vision, the successor
to MRPT's removed `mrpt::vision` module. Built on MRPT 3.x + Eigen + TBB
(no OpenCV, no Ceres).

## License

`mola_libvision` is released under the **GNU General Public License v3.0 or
later** (`GPL-3.0-or-later`); see the [LICENSE](LICENSE) file.

Some components are adapted from permissively-licensed projects and retain their
original copyright notices in the corresponding source files:
- ORB descriptor bit-pattern table from **OpenCV** (BSD-3-Clause).
- Bundle-adjustment / marginalization structure inspired by **Basalt**
  (BSD-3-Clause).
- Algorithms adapted from **lightweight_vio** (MIT).
