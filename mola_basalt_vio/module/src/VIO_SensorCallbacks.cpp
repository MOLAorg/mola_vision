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

// Others:
#include <mola_kernel/interfaces/ExecutableBase.h>  // mola::ProfilerEntry
#include <mrpt/core/exceptions.h>  // MRPT_TRY_START
#include <mrpt/core/lock_helper.h>
#include <mrpt/obs/CObservationIMU.h>
#include <mrpt/obs/CObservationImage.h>

// Std:
#include <regex>

namespace mola
{

void VisualInertialOdometry::onNewObservation(const CObservation::Ptr& o)
{
  MRPT_TRY_START
  const ProfilerEntry tleg(profiler_, "onNewObservation");

  ASSERT_(o);

  {
    auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);

    if (!state_.initialized)
    {
      MRPT_LOG_THROTTLE_ERROR(
          2.0,
          "Discarding incoming observations: the system initialize() method has not been called "
          "yet!");
      return;
    }
    if (state_.fatal_error)
    {
      MRPT_LOG_THROTTLE_ERROR(
          2.0, "Discarding incoming observations: a fatal error ocurred above.");

      this->requestShutdown();  // request end of mola-cli app, if applicable
      return;
    }

    // SLAM enabled?
    if (!state_.active)
    {
      // and do not process the observation:
      return;
    }
  }

  // Is it an IMU obs?
  if (params_.imu_sensor_label &&
      std::regex_match(o->sensorLabel, params_.imu_sensor_label.value()))
  {
    {
      auto lck = mrpt::lockHelper(is_busy_mtx_);
      state_.worker_tasks_others++;
    }

    // Yes, it's an IMU obs:
    auto fut = worker_.enqueue(&VisualInertialOdometry::onIMU, this, o);
    (void)fut;
  }

  // Is it a camera obs?
  for (const auto& label : params_.camera_sensor_labels)
  {
    if (o->sensorLabel != label)
    {
      continue;
    }

    // Yes, it's a camera obs:
    auto obs = std::dynamic_pointer_cast<mrpt::obs::CObservationImage>(o);
    if (!obs)
    {
      THROW_EXCEPTION_FMT(
          "Expected CObservationImage input, got: %s", o->GetRuntimeClass()->className);
    }

    // Do we have all required images?
    state_.input_synchronizer.add(o);

    const auto input_obs_set_opt = state_.input_synchronizer.getObservationGroup();
    if (!input_obs_set_opt.has_value())
    {
      continue;
    }

    // Yes, we have all images (1 for monocular, 2 for stereo, etc.)

    const int queued = [this]()
    {
      auto lck = mrpt::lockHelper(is_busy_mtx_);
      return state_.worker_tasks_camera;
    }();

    profiler_.registerUserMeasure("onNewObservation.camera_queue_length", queued);
    if (queued > params_.max_camera_queue_before_drop)
    {
      MRPT_LOG_THROTTLE_WARN_FMT(
          1.0,
          "Dropping observation due to camera worker thread too busy (dropped frames: %.02f%%)",
          getDropStats() * 100.0);
      profiler_.registerUserMeasure("onNewObservation.drop_observation", 1);
      addDropStats(true);
      return;
    }
    addDropStats(false);
    profiler_.enter("delay_onNewObs_to_process");

    {
      auto lck = mrpt::lockHelper(is_busy_mtx_);
      state_.worker_tasks_camera++;
    }

    MRPT_LOG_DEBUG_STREAM("Synchronized observations: " << input_obs_set_opt->size());

    std::vector<std::shared_ptr<mrpt::obs::CObservationImage>> imageObs;
    for (const auto& [lb, this_obs] : *input_obs_set_opt)
    {
      imageObs.push_back(std::dynamic_pointer_cast<mrpt::obs::CObservationImage>(this_obs));
    }

    // Enqueue task:
    auto fut = worker_.enqueue(&VisualInertialOdometry::onImageSet, this, imageObs);

    (void)fut;

    break;  // do not keep processing the list
  }

  MRPT_TRY_END
}

void VisualInertialOdometry::onImageSet(
    const std::vector<std::shared_ptr<mrpt::obs::CObservationImage>>& obs)
{
  const bool abort_running = [this]()
  {
    auto lck = mrpt::lockHelper(is_busy_mtx_);
    return destructor_called_;
  }();

  // All methods that are enqueued into a thread pool should have its own
  // top-level try-catch:
  if (!abort_running)
  {
    try
    {
      processImageSet(obs);
    }
    catch (const std::exception& e)
    {
      MRPT_LOG_ERROR_STREAM("Exception:\n" << mrpt::exception_to_str(e));
      auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);
      state_.fatal_error = true;
    }
  }

  {
    auto lck = mrpt::lockHelper(is_busy_mtx_);
    state_.worker_tasks_camera--;
  }
}

void VisualInertialOdometry::onIMU(const CObservation::Ptr& o)
{
  // All methods that are enqueued into a thread pool should have its own
  // top-level try-catch:
  try
  {
    onIMUImpl(o);
  }
  catch (const std::exception& e)
  {
    MRPT_LOG_ERROR_STREAM("Exception:\n" << mrpt::exception_to_str(e));
    auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);
    state_.fatal_error = true;
  }

  {
    auto lck = mrpt::lockHelper(is_busy_mtx_);
    state_.worker_tasks_others--;
  }
}

void VisualInertialOdometry::onIMUImpl(const CObservation::Ptr& o)
{
  ASSERT_(o);

  const ProfilerEntry tleg(profiler_, "onIMU");

  auto imu = std::dynamic_pointer_cast<mrpt::obs::CObservationIMU>(o);
  ASSERTMSG_(
      imu, mrpt::format(
               "IMU observation with label '%s' does not have the expected "
               "type 'mrpt::obs::CObservationIMU', it is '%s' instead",
               o->sensorLabel.c_str(), o->GetRuntimeClass()->className));

  MRPT_LOG_DEBUG_STREAM(
      "onIMU called for timestamp=" << mrpt::system::dateTimeLocalToString(imu->timestamp));

  if (!imu->has(mrpt::obs::IMU_X_ACC) || !imu->has(mrpt::obs::IMU_Y_ACC) ||
      !imu->has(mrpt::obs::IMU_Z_ACC))
  {
    // No acceleration data:
    return;
  }

  const auto accel_sensor = mrpt::math::TTwist3D(  //
      imu->get(mrpt::obs::IMU_X_ACC),  //
      imu->get(mrpt::obs::IMU_Y_ACC),  //
      imu->get(mrpt::obs::IMU_Z_ACC),  //
      0, 0, 0);

  const auto accel_base_link = accel_sensor.rotated(imu->sensorPose.asTPose());

  MRPT_TODO("Continue");
}

}  // namespace mola
