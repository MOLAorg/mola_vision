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
#include <mrpt/obs/CObservationImage.h>

// OpenCV: (required for now... should be replaced)
#include <opencv2/core.hpp>

namespace
{

basalt::ImageData cv_to_basalt_image(const cv::Mat& img)
{
  basalt::ImageData res;

  res.img      = std::make_shared<basalt::ManagedImage<uint16_t>>(img.cols, img.rows);
  res.exposure = 0;  // default

  const auto full_size = static_cast<std::size_t>(img.cols) * static_cast<std::size_t>(img.rows);

  if (img.type() == CV_8UC1)
  {
    const uint8_t* data_in  = img.ptr();
    uint16_t*      data_out = res.img->ptr;

    for (size_t i = 0; i < full_size; i++)
    {
      int val     = data_in[i];
      val         = val << 8;
      data_out[i] = val;
    }
  }
  else if (img.type() == CV_8UC3)
  {
    const uint8_t* data_in  = img.ptr();
    uint16_t*      data_out = res.img->ptr;

    for (size_t i = 0; i < full_size; i++)
    {
      int val     = data_in[i * 3];
      val         = val << 8;
      data_out[i] = val;
    }
  }
  else if (img.type() == CV_16UC1)
  {
    std::memcpy(res.img->ptr, img.ptr(), full_size * sizeof(uint16_t));
  }
  else
  {
    THROW_EXCEPTION("Unsupported opencv image type");
  }

  return res;
}

basalt::GenericCamera<double> mrpt_to_basalt_camera_model(const mrpt::img::TCamera& cam)
{
  basalt::GenericCamera<double> gc;

  switch (cam.distortion)
  {
    case mrpt::img::DistortionModel::none:
    {
      auto ph = basalt::PinholeCamera();
      //  [fx, fy, cx, cy]
      ph.setFromInit({cam.fx(), cam.fy(), cam.cx(), cam.cy()});
      gc.variant = ph;
    }
    break;

    case mrpt::img::DistortionModel::plumb_bob:
    {
      // [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
      const double k4 = 0, k5 = 0, k6 = 0;

      auto ph    = basalt::PinholeRadtan8Camera<double>(  //
          {cam.fx(), cam.fy(), cam.cx(), cam.cy(),  //
              cam.k1(), cam.k2(), cam.p1(), cam.p2(), cam.k3(), k4, k5, k6});
      gc.variant = ph;
    }
    break;

    case mrpt::img::DistortionModel::kannala_brandt:
    {
      THROW_EXCEPTION("TO-DO");
    }
    break;

    default:
    {
      THROW_EXCEPTION("Unknown mrpt camera distortion model");
    }
  };
  return gc;
}

}  // namespace

namespace mola
{

// here happens the main stuff:
void VisualInertialOdometry::processImageSet(
    const std::vector<std::shared_ptr<mrpt::obs::CObservationImage>>& obs_list)
{
  using namespace std::string_literals;

  // Check if we need to process any pending async request:
  processPendingUserRequests();

  auto lckState = mrpt::lockHelper(state_mtx_);

  // make sure data is loaded, if using an offline lazy-load dataset.
  for (auto& obs : obs_list)
  {
    ASSERT_(obs);
    obs->load();
  }

  const auto this_obs_tim = (*obs_list.begin())->timestamp;

  const ProfilerEntry tleg(profiler_, "onImage");

  // On first call, initialize the calibration info:
  if (!state_.calibration)
  {
    bool use_double = false;

    auto& calibration = state_.calibration.emplace();

    // TODO: At present IMU is assumed to be at vehicle origin:
    // Camera to IMU transform:
    calibration.T_i_c.clear();
    calibration.intrinsics.clear();
    calibration.resolution.clear();
    for (const auto& cam_obs : obs_list)
    {
      const auto tf = Sophus::SE3d(
          cam_obs->cameraPose.getHomogeneousMatrixVal<mrpt::math::CMatrixDouble44>().asEigen());
      calibration.T_i_c.push_back(tf);
      calibration.intrinsics.push_back(mrpt_to_basalt_camera_model(cam_obs->cameraParams));
      calibration.resolution.push_back({cam_obs->cameraParams.ncols, cam_obs->cameraParams.nrows});
    }

    // Initialize basalt VIO pipeline:
    state_.opt_flow_ptr =
        basalt::OpticalFlowFactory::getOpticalFlow(state_.vio_config, calibration);

    state_.vio = basalt::VioEstimatorFactory::getVioEstimator(
        state_.vio_config, calibration, basalt::constants::g, params_.use_imu, use_double);

    state_.vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    // Connect the input queue of VIO to the output of the optical flow:
    state_.opt_flow_ptr->output_queue = &state_.vio->vision_data_queue;
    state_.vio->out_state_queue       = &out_state_queue_;
  }

  // Send the image to the optical flow:
  {
    auto img_data = std::make_shared<basalt::OpticalFlowInput>();

    img_data->t_ns = static_cast<int64_t>(mrpt::Clock::toDouble(this_obs_tim) * 1e9);

    for (const auto& cam_obs : obs_list)
    {
      img_data->img_data.push_back(cv_to_basalt_image(cam_obs->image.asCvMatRef()));
    }

    state_.opt_flow_ptr->input_queue.push(img_data);
  }

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

void VisualInertialOdometry::spinOnce()
{
  // Read all already processed trajectory data:
  basalt::PoseVelBiasState<double>::Ptr data;
  while (true)
  {
    out_state_queue_.pop(data);

    if (!data.get())
    {
      break;
    }

    auto lckState = mrpt::lockHelper(state_mtx_);

    int64_t t_ns = data->t_ns;

    std::cerr << "t_ns " << t_ns << std::endl;
    Sophus::SE3d    T_w_i   = data->T_w_i;
    Eigen::Vector3d vel_w_i = data->vel_w_i;
    Eigen::Vector3d bg      = data->bias_gyro;
    Eigen::Vector3d ba      = data->bias_accel;

    std::cerr << "T_w_i:\n" << T_w_i.matrix() << std::endl;

    // vio_t_ns.emplace_back(data->t_ns);
    // vio_t_w_i.emplace_back(T_w_i.translation());
    // vio_T_w_i.emplace_back(T_w_i);

#if 0
    if (show_gui)
    {
      std::vector<float> vals;
      vals.push_back((t_ns - start_t_ns) * 1e-9);

      for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
      for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
      for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
      for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

      vio_data_log.Log(vals);
    }
#endif
  }
}

}  // namespace mola
