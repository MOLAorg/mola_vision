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
#include <mrpt/obs/CObservation2DRangeScan.h>
#include <mrpt/obs/CObservationComment.h>
#include <mrpt/obs/CObservationGPS.h>
#include <mrpt/poses/Lie/SO.h>

namespace mola
{

// here happens the main stuff:
void VisualInertialOdometry::processImage(const CObservation::Ptr& obs)
{
  using namespace std::string_literals;

  // Check if we need to process any pending async request:
  processPendingUserRequests();

  auto lckState = mrpt::lockHelper(state_mtx_);

  ASSERT_(obs);

  profiler_.leave("delay_onNewObs_to_process");

  // make sure data is loaded, if using an offline lazy-load dataset.
  obs->load();

  // Only process pointclouds that are sufficiently apart in time:
  const auto this_obs_tim = obs->timestamp;

  const ProfilerEntry tleg(profiler_, "onImage");

  // TODO!
  const bool icpIsGood     = true;
  state_.last_vio_was_good = icpIsGood;
  // state_.last_vio_pose     = xxx;

  // Update trajectory too:
  if (icpIsGood)
  {
    auto lck = mrpt::lockHelper(state_trajectory_mtx_);
    state_.estimated_trajectory.insert(this_obs_tim, state_.last_vio_pose.mean);
  }

  // In any case, publish the vehicle pose, no matter if it's a keyframe or not,
  // but if ICP quality was good enough:
  if (state_.last_vio_was_good)
  {
    doPublishUpdatedLocalization(this_obs_tim);
  }

  // Publish new local map:
  doPublishUpdatedMap(this_obs_tim);

  // Optional real-time GUI via MOLA VizInterface:
  if (visualizer_)
  {
    const ProfilerEntry tle(profiler_, "onImage.6.updateVisualization");
    // updateVisualization(observationRawForViz);
  }
}

}  // namespace mola
