/* -------------------------------------------------------------------------
 * mola_visual_slam: monocular / stereo visual SLAM front-end for MOLA.
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_visual_slam/VisualSlam.h>
#include <mrpt/core/initializer.h>

MRPT_INITIALIZER(do_register_mola_visual_slam) { MOLA_REGISTER_MODULE(mola::VisualSlam); }
