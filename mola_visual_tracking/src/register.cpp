/* -------------------------------------------------------------------------
 * mola_visual_tracking: a MOLA demo front-end that detects and tracks image
 * features and shows them in a MolaViz subwindow (no 3D reconstruction).
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_visual_tracking/VisualTracking.h>
#include <mrpt/core/initializer.h>

MRPT_INITIALIZER(do_register_mola_visual_tracking) { MOLA_REGISTER_MODULE(mola::VisualTracking); }
