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

// MOLA:
#include <mola_yaml/yaml_helpers.h>

namespace mola
{

// arguments: class_name, parent_class, class namespace
IMPLEMENTS_MRPT_OBJECT(VisualInertialOdometry, FrontEndBase, mola)

VisualInertialOdometry::VisualInertialOdometry() {}

VisualInertialOdometry::~VisualInertialOdometry()
{
  auto lckState = mrpt::lockHelper(state_mtx_);

  if (state_.opt_flow_ptr)
  {
    // Indicate the end of the sequence
    state_.opt_flow_ptr->input_queue.push(nullptr);
  }
  if (state_.vio)
  {
    state_.vio->drain_input_queues();
  }

  // todo: better synchronous ending:
  std::this_thread::sleep_for(std::chrono::seconds(1));
}

void VisualInertialOdometry::reset()
{
  ASSERTMSG_(!lastInitConfig_.empty(), "initialize() must be called first.");

  auto lck = mrpt::lockHelper(state_mtx_);

  state_ = MethodState();
  initialize(lastInitConfig_);
}

bool VisualInertialOdometry::isBusy() const
{
  bool b;
  is_busy_mtx_.lock();
  b = (state_.worker_tasks_camera != 0) || (state_.worker_tasks_others != 0);
  is_busy_mtx_.unlock();
  return b || worker_.pendingTasks();
}

bool VisualInertialOdometry::isActive() const
{
  auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);
  return state_.active;
}

void VisualInertialOdometry::setActive(const bool active)
{
  auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);
  state_.active      = active;
}

void VisualInertialOdometry::initialize_frontend(const Yaml& c)
{
  MRPT_TRY_START

  auto lckState = mrpt::lockHelper(state_mtx_);

  this->setLoggerName("VisualInertialOdometry");

  // make a copy of the initialization, for use in reset()
  lastInitConfig_ = c;

  // Load params:
  const auto cfg = c["params"];
  MRPT_LOG_DEBUG_STREAM("Loading these params:\n" << cfg);

  ENSURE_YAML_ENTRY_EXISTS(cfg, "camera_sensor_labels");
  if (cfg["camera_sensor_labels"].isSequence())
  {
    const auto lsl = cfg["camera_sensor_labels"].asSequenceRange();
    for (const auto& sl : lsl)
    {
      const auto s = sl.as<std::string>();
      MRPT_LOG_DEBUG_STREAM("Adding as input lidar sensor label: " << s);
      params_.camera_sensor_labels.emplace_back(s);
    }
  }
  else
  {
    ASSERT_(cfg["camera_sensor_labels"].isScalar());
    const auto s = cfg["camera_sensor_labels"].as<std::string>();
    MRPT_LOG_DEBUG_STREAM("Adding as input lidar sensor label: " << s);
    params_.camera_sensor_labels.emplace_back(s);
  }
  ASSERT_(!params_.camera_sensor_labels.empty());

  if (cfg.has("imu_sensor_label"))
  {
    params_.imu_sensor_label = cfg["imu_sensor_label"].as<std::string>();
  }

  YAML_LOAD_OPT(params_, use_imu, bool);

  YAML_LOAD_OPT(params_, start_active, bool);

  YAML_LOAD_OPT(params_, publish_reference_frame, std::string);
  YAML_LOAD_OPT(params_, publish_vehicle_frame, std::string);

  if (cfg.has("visualization"))
  {
    params_.visualization.initialize(cfg["visualization"]);
  }

  if (cfg.has("estimated_trajectory"))
  {
    params_.estimated_trajectory.initialize(cfg["estimated_trajectory"]);
  }

  // system-wide profiler:
  // profiler_.enable(params_.pipeline_profiler_enabled);

  // Basalt VIO initialization:
  // state_.vio_config.xxx

  // end of initialization:
  {
    auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);

    state_.initialized = true;
    state_.active      = params_.start_active;
  }

  // Make runtime params exposed:
  onExposeParameters();

  MRPT_TRY_END
}

void VisualInertialOdometry::addDropStats(bool frame_is_dropped)
{
  state_.drop_frames_stats_good[state_.drop_frames_stats_next_index]    = !frame_is_dropped;
  state_.drop_frames_stats_dropped[state_.drop_frames_stats_next_index] = frame_is_dropped;
  if (++state_.drop_frames_stats_next_index >= MethodState::DROP_STATS_WINDOW_LENGHT)
  {
    state_.drop_frames_stats_next_index = 0;
  }
}

double VisualInertialOdometry::getDropStats() const
{
  auto       lckStateFlags = mrpt::lockHelper(state_flags_mtx_);
  const auto good =
      std::count(state_.drop_frames_stats_good.begin(), state_.drop_frames_stats_good.end(), true);
  const auto bad = std::count(
      state_.drop_frames_stats_dropped.begin(), state_.drop_frames_stats_dropped.end(), true);
  const auto total = static_cast<double>(good + bad);
  return total != 0 ? static_cast<double>(bad) / total : .0;
}

void VisualInertialOdometry::enqueue_request(const std::function<void()>& userRequest)
{
  auto lck = mrpt::lockHelper(requests_mtx_);
  requests_.push_back(userRequest);
}

void VisualInertialOdometry::processPendingUserRequests()
{
  auto lckState = mrpt::lockHelper(state_mtx_);
  auto lck      = mrpt::lockHelper(requests_mtx_);

  for (const auto& r : requests_)
  {
    try
    {
      r();
    }
    catch (const std::exception& e)
    {
      MRPT_LOG_ERROR_STREAM("Error processing asynchronous enqueue_request(): " << e.what());
    }
  }
  requests_.clear();
}

}  // namespace mola
