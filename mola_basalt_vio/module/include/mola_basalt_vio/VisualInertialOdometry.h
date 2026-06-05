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
#pragma once

// MOLA interfaces & classes:
#include <mola_kernel/interfaces/FrontEndBase.h>
#include <mola_kernel/interfaces/LocalizationSourceBase.h>
#include <mola_kernel/utils/Synchronizer.h>

// MRPT
#include <mrpt/core/WorkerThreadsPool.h>
#include <mrpt/obs/obs_frwds.h>
#include <mrpt/viz/CSetOfLines.h>
#include <mrpt/viz/CSetOfObjects.h>
#include <mrpt/poses/CPose3DInterpolator.h>
#include <mrpt/poses/CPose3DPDFGaussian.h>

// Basalt:
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#include <basalt/io/dataset_io.h>
#include <basalt/io/marg_data_io.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/utils/system_utils.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>

#include <basalt/calibration/calibration.hpp>
#include <basalt/utils/format.hpp>
#include <basalt/utils/time_utils.hpp>
#include <sophus/se3.hpp>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// STD:
#include <cstdint>
#include <cstdlib>
#include <regex>

namespace mola
{

template <std::size_t N, typename T>
constexpr std::array<T, N> create_array(const T& value);

/** Visual-inertial odometry MOLA wrapper for the Basalt VIO system.
 *
 * Basalt was presented in:
 * - Visual-Inertial Mapping with Non-Linear Factor Recovery, V. Usenko, N. Demmel, D. Schubert, J.
 *   Stückler, D. Cremers, In IEEE Robotics and Automation Letters (RA-L)
 *   [DOI:10.1109/LRA.2019.2961227] [arXiv:1904.06504].
 *
 */
class VisualInertialOdometry : public mola::FrontEndBase, public mola::LocalizationSourceBase
{
  DEFINE_MRPT_OBJECT(VisualInertialOdometry, mola)

 public:
  VisualInertialOdometry();
  ~VisualInertialOdometry() override;

  // Rule of five: explicitly define or delete special member functions
  VisualInertialOdometry(const VisualInertialOdometry&)                = delete;
  VisualInertialOdometry& operator=(const VisualInertialOdometry&)     = delete;
  VisualInertialOdometry(VisualInertialOdometry&&) noexcept            = delete;
  VisualInertialOdometry& operator=(VisualInertialOdometry&&) noexcept = delete;

  /** @name Main API
   * @{ */

  // See docs in base class
  void initialize_frontend(const Yaml& cfg) override;
  void spinOnce() override;
  void onNewObservation(const CObservation::Ptr& o) override;

  void onNewObservationImpl(const CObservation::Ptr& o);

  /** Re-initializes the odometry system. It effectively calls initialize()
   *  once again with the same parameters that were used the first time.
   */
  void reset();

  enum class AlignKind : uint8_t
  {
    RegularOdometry = 0,
    NoMotionModel
  };

  struct Parameters
  {
    /** List of sensor labels for input observations to be used as camera image observations.
     *  Monocular systems will have only one entry, stereo
     *  two camera labels. More than 2 are also supported by Basalt.
     */
    std::vector<std::string> camera_sensor_labels;

    /** Sensor labels or regex to be matched to input observations
     *  to be used as raw IMU observations.
     */
    std::optional<std::regex> imu_sensor_label;

    struct Visualization
    {
      int    map_update_decimation       = 10;
      bool   show_trajectory             = true;
      bool   show_ground_grid            = true;
      float  ground_grid_spacing         = 5.0f;
      double current_pose_corner_size    = 1.5;  //! [m]
      float  local_map_point_size        = 3.0f;
      bool   gui_subwindow_starts_hidden = false;
      bool   show_console_messages       = true;
      bool   camera_follows_vehicle      = true;
      bool   camera_rotates_with_vehicle = false;

      /** If not empty, an optional 3D model (.DAE, etc) to load for
       * visualizing the robot/vehicle pose */
      struct ModelPart
      {
        std::string         file;
        mrpt::math::TPose3D tf;  /// Optional 3D model offset/rotation
        double              scale = 1.0;
      };
      std::vector<ModelPart> model;

      void initialize(const Yaml& c);
    };
    Visualization visualization;

    // === OUTPUT TRAJECTORY ====
    struct TrajectoryOutputOptions
    {
      bool save_to_file = false;

      /** If save_to_file==true, the final estimated trajectory will be
       * dumped to a file at destruction time */
      std::string output_file = "output.txt";

      void initialize(const Yaml& c);
    };

    TrajectoryOutputOptions estimated_trajectory;

    bool start_active = true;

    bool vio_debug_output = false;

    /** Use Visual Inertial Odometry (VIO) (true) or Visual Odometry (VO) (false) */
    bool use_imu = false;

    /** When publishing pose updates, the reference frame for both, estimated robot poses, and the
     * local map.*/
    std::string publish_reference_frame = "vi_odom";

    /** When publishing pose updates, the vehicle frame name.*/
    std::string publish_vehicle_frame = "base_link";
  };

  /** Algorithm parameters */
  Parameters params_;

  bool isBusy() const;

  bool isActive() const;
  void setActive(const bool active);

  /** Returns a copy of the estimated trajectory, with timestamps for each
   * lidar observation.
   * Multi-thread safe to call.
   */
  mrpt::poses::CPose3DInterpolator estimatedTrajectory() const;

  /** Enqueue a custom user request to be executed on the main LidarOdometry
   *  thread on the next iteration.
   *
   *  So, this method is safe to be called from any other thread.
   *
   */
  void enqueue_request(const std::function<void()>& userRequest);

  /** @} */

 protected:
  // See docs in base class.
  void onParameterUpdate(const mrpt::containers::yaml& names_values) override;
  void onExposeParameters();  // called after initialization

 private:
  /** All variables that hold the algorithm state */
  struct MethodState
  {
    MethodState() = default;

    // ------ these flags are protected by state_flags_mtx_  ---------
    bool initialized = false;
    bool fatal_error = false;
    bool active      = true;  //!< whether to process or ignore incoming sensors
    // ------ ^^^ end of these flags are protected ^^^^      ---------

    // All other fields are protected by state_mtx_

    // VIO variables
    std::optional<basalt::Calibration<double>> calibration;
    basalt::VioDatasetPtr                      vio_dataset;
    basalt::VioConfig                          vio_config;
    basalt::OpticalFlowBase::Ptr               opt_flow_ptr;
    basalt::VioEstimatorBase::Ptr              vio;

    mrpt::poses::CPose3DPDFGaussian last_vio_pose;  //!< in local map

    /// Used to synchronize stereo camera images
    mola::Synchronizer input_synchronizer;

    bool last_vio_was_good = true;

    std::optional<mrpt::Clock::time_point> last_obs_timestamp;
    std::optional<mrpt::Clock::time_point> last_vio_timestamp;

    /// Cache for multiple Image synchronization:
    // std::map<std::string /*label*/, mrpt::obs::CObservation::Ptr> sync_obs;

    mrpt::poses::CPose3DInterpolator estimated_trajectory;

    /// To update the map in the viz only if really needed
    bool local_map_needs_viz_update = true;
    bool local_map_needs_publish    = true;

    void mark_local_map_as_updated()
    {
      local_map_needs_viz_update = true;
      local_map_needs_publish    = true;
    }

    static constexpr std::size_t               DROP_STATS_WINDOW_LENGHT = 128;
    std::array<bool, DROP_STATS_WINDOW_LENGHT> drop_frames_stats_good =
        create_array<DROP_STATS_WINDOW_LENGHT>(true);
    std::array<bool, DROP_STATS_WINDOW_LENGHT> drop_frames_stats_dropped =
        create_array<DROP_STATS_WINDOW_LENGHT>(false);
    std::size_t drop_frames_stats_next_index = 0;

    // Visualization:
    mrpt::viz::CSetOfObjects::Ptr glVehicleFrame, glPathGrp;
    mrpt::viz::CSetOfLines::Ptr   glEstimatedPath;
    int                              mapUpdateCnt = std::numeric_limits<int>::max();

  };  // end of MethodState

  /** The worker thread pool with 1 thread for processing incoming IMU or Image observations*/
  mrpt::WorkerThreadsPool worker_{
      1 /*num threads*/, mrpt::WorkerThreadsPool::POLICY_FIFO, "worker_vio"};

  tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> out_state_queue_;

  MethodState        state_;
  const MethodState& state() const { return state_; }
  MethodState        stateCopy() const { return state_; }

  // Accessing this struct in gui_ requires adcquiring state_gui_mtx_
  struct StateUI
  {
    StateUI() = default;

    double timestampLastUpdateUI = 0;

    nanogui::Window* ui = nullptr;
    // nanogui::Label*    lbIcpQuality    = nullptr;
    // nanogui::CheckBox* cbActive        = nullptr;
    // nanogui::CheckBox* cbMapping       = nullptr;
  };

  // Accessing this struct in gui_ requires adcquiring state_gui_mtx_
  StateUI gui_;

  /// The configuration used in the last call to initialize()
  Yaml lastInitConfig_;

  bool                         destructor_called_ = false;
  mutable std::mutex           is_busy_mtx_;
  mutable std::mutex           state_flags_mtx_;
  mutable std::recursive_mutex state_mtx_;
  mutable std::mutex           state_trajectory_mtx_;
  mutable std::mutex           state_gui_mtx_;

  /// The list of pending tasks from enqueue_request():
  std::vector<std::function<void()>> requests_;
  std::mutex                         requests_mtx_;

  /// Must be called from a scope with state_flags_mtx_ already adcquired!
  void addDropStats(bool frame_is_dropped);

  /// Returns the ratio [0,1] of lidar frames dropped due to slow processing in the last few
  /// seconds.
  double getDropStats() const;

  // Process requests_(), at the spinOnce() rate.
  void processPendingUserRequests();

  void onImageSet(const std::vector<std::shared_ptr<mrpt::obs::CObservationImage>>& obs);
  void processImageSet(const std::vector<std::shared_ptr<mrpt::obs::CObservationImage>>& obs);

  void onIMU(const CObservation::Ptr& o);
  void onIMUImpl(const CObservation::Ptr& o);

  // void updateVisualization(const mp2p_icp::metric_map_t& currentObservation);

  void internalBuildGUI();

  void doPublishUpdatedLocalization(const mrpt::Clock::time_point& this_obs_tim);

  void doPublishUpdatedMap(const mrpt::Clock::time_point& this_obs_tim);

  void onPublishDiagnostics();
};

namespace detail
{
template <typename T, std::size_t... Is>
constexpr std::array<T, sizeof...(Is)> create_array(T value, std::index_sequence<Is...>)
{
  // cast Is to void to remove the warning: unused value
  return {{(static_cast<void>(Is), value)...}};
}
}  // namespace detail

template <std::size_t N, typename T>
constexpr std::array<T, N> create_array(const T& value)
{
  return detail::create_array(value, std::make_index_sequence<N>());
}

}  // namespace mola
