/* -------------------------------------------------------------------------
 * mola_rgbd_slam: RGB-D visual SLAM front-end for the MOLA framework.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_rgbd_slam/RgbdSlam.h>
#include <mrpt/core/initializer.h>

MRPT_INITIALIZER(do_register_mola_rgbd_slam) { MOLA_REGISTER_MODULE(mola::RgbdSlam); }
