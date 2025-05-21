// -----------------------------------------------------------------------------
//   A Modular Optimization framework for Localization and mApping  (MOLA)
//
// Copyright (C) 2018-2025 Jose Luis Blanco, University of Almeria
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------

// This module:
#include <mola_basalt_vio/VisualInertialOdometry.h>

// MRPT:
#include <mrpt/io/CMemoryStream.h>
#include <mrpt/serialization/CArchive.h>

namespace mola
{

void VisualInertialOdometry::doPublishUpdatedLocalization(
    const mrpt::Clock::time_point& this_obs_tim)
{
  const ProfilerEntry tle(profiler_, "advertiseUpdatedLocalization");

  LocalizationUpdate lu;
  lu.method          = "vi_odometry";
  lu.reference_frame = params_.publish_reference_frame;
  lu.child_frame     = params_.publish_vehicle_frame;
  lu.timestamp       = this_obs_tim;
  lu.pose            = state_.last_vio_pose.mean.asTPose();
  lu.cov             = state_.last_vio_pose.cov;
  lu.quality         = 1.0;

  advertiseUpdatedLocalization(lu);
}

void VisualInertialOdometry::doPublishUpdatedMap(const mrpt::Clock::time_point& this_obs_tim)
{
  if (!state_.local_map_needs_publish)
  {
    return;
  }

  // TODO: Implement if needed, publish map updates.
}

void VisualInertialOdometry::onPublishDiagnostics()
{
  auto lckState = mrpt::lockHelper(state_mtx_);

  const auto curStamp = state_.last_obs_timestamp ? *state_.last_obs_timestamp : mrpt::Clock::now();

  mrpt::containers::yaml diagValues = mrpt::containers::yaml::Map();

  const double dtAvr = profiler_.getMeanTime("onImage");

  diagValues["average_process_time"] = dtAvr;
  diagValues["dropped_frames_ratio"] = getDropStats();
  diagValues["parameters"]           = getModuleParameters();

  DiagnosticsOutput diag;
  diag.timestamp = curStamp;
  diag.label     = "status";
  diag.value     = diagValues;

  module_publish_diagnostics(diag);
}

}  // namespace mola
