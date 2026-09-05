/*               _
 _ __ ___   ___ | | __ _
| '_ ` _ \ / _ \| |/ _` | Modular Optimization framework for
| | | | | | (_) | | (_| | Localization and mApping (MOLA)
|_| |_| |_|\___/|_|\__,_| https://github.com/MOLAorg/mola

 Copyright (C) 2018-2026 Jose Luis Blanco, University of Almeria,
                         and individual contributors.
 SPDX-License-Identifier: GPL-3.0
 See LICENSE for full license information.
*/

/**
 * @file   mola-visual-slam-cli.cpp
 * @brief  main() for the cli app running mola::VisualSlam for offline
 *         datasets - mirrors mola-lidar-odometry-cli's structure and
 *         conventions (same dataset-source flags, --output-tum-path).
 * @author Jose Luis Blanco Claraco
 * @date   Sep 5, 2026
 */

#include <mola_kernel/interfaces/OfflineDatasetSource.h>
#include <mola_kernel/pretty_print_exception.h>
#include <mola_visual_slam/VisualSlam.h>
#include <mola_yaml/yaml_helpers.h>
#include <mrpt/core/Clock.h>
#include <mrpt/core/exceptions.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/poses/CPose3DInterpolator.h>
#include <mrpt/system/COutputLogger.h>
#include <mrpt/system/os.h>
#include <mrpt/system/progress.h>
#include <mrpt/system/string_utils.h>

#if defined(HAVE_MOLA_INPUT_KITTI)
#include <mola_input_kitti_dataset/KittiOdometryDataset.h>
#endif

#if defined(HAVE_MOLA_INPUT_ROSBAG1)
#include <mola_input_rosbag1/Rosbag1Dataset.h>
#endif

#include <CLI/CLI.hpp>
#include <csignal>
#include <iostream>
#include <string>

namespace
{
template <typename T>
struct Opt
{
  T            value{};
  CLI::Option* opt = nullptr;

  bool     isSet() const { return opt && opt->count() > 0; }
  const T& getValue() const { return value; }
};

struct Cli
{
  CLI::App cmd{"mola-visual-slam-cli"};

  Opt<std::string> argYAML;
  Opt<std::string> arg_verbosity_level;
  Opt<std::string> arg_outPath;
  Opt<int>         arg_firstN;
  Opt<int>         arg_skipFirstN;
  Opt<std::string> arg_leftLabel;
  Opt<std::string> arg_rightLabel;
  Opt<std::string> arg_baseLinkName;
  Opt<bool>        arg_saveCameraFrame;
  Opt<double>      arg_progressBarPeriod;

#if defined(HAVE_MOLA_INPUT_ROSBAG1)
  Opt<std::vector<std::string>> argRosbag1;
  Opt<std::string>              arg_leftTopic;
  Opt<std::string>              arg_rightTopic;
  Opt<std::string>              arg_leftSensorPose;
  Opt<std::string>              arg_rightSensorPose;
  Opt<std::string>              arg_imuTopic;
  Opt<std::string>              arg_imuSensorPose;
#endif
  Opt<std::string> arg_imuLabel;

#if defined(HAVE_MOLA_INPUT_KITTI)
  Opt<std::string> argKittiSeq;
#endif

  Cli()
  {
    argYAML.opt = cmd.add_option(
        "-c,--config", argYAML.value,
        "Optional VisualSlam params YAML file (a 'params:' block, same "
        "content as a mola-cli-launchs/*.yaml 'visual_slam' module entry). "
        "Without it, VisualSlam's own defaults apply.");

    arg_verbosity_level.opt = cmd.add_option(
        "-v,--verbosity", arg_verbosity_level.value,
        "Verbosity level: ERROR|WARN|INFO|DEBUG {Default: INFO}");

    arg_outPath.opt =
        cmd.add_option(
               "--output-tum-path", arg_outPath.value,
               "Save the estimated path as a TXT file using the TUM file format {see evo docs}")
            ->option_text("output-trajectory.txt");

    arg_firstN.opt = cmd.add_option(
                            "--only-first-n", arg_firstN.value,
                            "Run for the first N stereo/mono pairs only {0=default, not used}")
                         ->check(CLI::NonNegativeNumber);

    arg_skipFirstN.opt = cmd.add_option(
                                "--skip-first-n", arg_skipFirstN.value,
                                "Skip the first N dataset entries {0=default, not used}")
                             ->check(CLI::NonNegativeNumber);

    arg_leftLabel.value = "image_0";
    arg_leftLabel.opt   = cmd.add_option(
                                 "--left-sensor-label", arg_leftLabel.value,
                                 "sensorLabel VisualSlam reads the left/monocular image from")
                            ->capture_default_str();

    arg_rightLabel.value = "image_1";
    arg_rightLabel.opt   = cmd.add_option(
                                  "--right-sensor-label", arg_rightLabel.value,
                                  "sensorLabel VisualSlam reads the right (stereo mode) image from")
                             ->capture_default_str();

    arg_baseLinkName.value = "base_link";
    arg_baseLinkName.opt =
        cmd.add_option(
               "--base-link-frame-id", arg_baseLinkName.value,
               "Only for rosbag input sources. The /tf frame_id used as reference frame "
               "to get sensor poses with respect to the vehicle from /tf data.")
            ->envname("MOLA_TF_BASE_LINK")
            ->capture_default_str();

    arg_saveCameraFrame.opt = cmd.add_flag(
        "--save-camera-frame", arg_saveCameraFrame.value,
        "Save the LEFT CAMERA trajectory instead of the vehicle-body one. By default the "
        "output is in the body frame whenever the camera-on-robot extrinsic is known (from "
        "/tf or --left-sensor-pose), which is what a dataset's ground truth uses.");

    arg_progressBarPeriod.value = -1.0;
    arg_progressBarPeriod.opt =
        cmd.add_option(
               "--progress-bar-period", arg_progressBarPeriod.value,
               "Minimum percentage step, in [0,100], between progress bar printouts. Use 0 "
               "to disable. {Default: print on (almost) every processed entry}")
            ->check(CLI::Range(0.0, 100.0));

    arg_imuLabel.value = "imu";
    arg_imuLabel.opt =
        cmd.add_option(
               "--imu-sensor-label", arg_imuLabel.value,
               "sensorLabel VisualSlam reads gyroscope data from, for the gyro-aided "
               "inter-frame rotation prediction. Only takes effect together with the "
               "module's own 'imu_label' parameter (or --imu-topic, which sets it).")
            ->capture_default_str();

#if defined(HAVE_MOLA_INPUT_ROSBAG1)
    argRosbag1.opt = cmd.add_option(
                            "--input-rosbag1", argRosbag1.value,
                            "INPUT DATASET: rosbag1. Input dataset in ROS 1 bag format {*.bag}. "
                            "May be given more than once: all bags are read as one time-sorted "
                            "stream, which is how a separate IMU bag joins the images.")
                         ->option_text("dataset.bag");

    arg_leftTopic.opt = cmd.add_option(
        "--left-topic", arg_leftTopic.value,
        "rosbag1 topic for the left/monocular image (required with --input-rosbag1)");

    arg_rightTopic.opt = cmd.add_option(
        "--right-topic", arg_rightTopic.value,
        "rosbag1 topic for the right image (stereo mode only)");

    arg_leftSensorPose.opt = cmd.add_option(
        "--left-sensor-pose", arg_leftSensorPose.value,
        "Overrides whatever /tf says about the left camera pose (e.g. when the bag "
        "has no per-camera /tf_static frame): 'x y z yaw_deg pitch_deg roll_deg'");

    arg_rightSensorPose.opt = cmd.add_option(
        "--right-sensor-pose", arg_rightSensorPose.value,
        "Same as --left-sensor-pose, for the right camera (stereo mode only)");

    arg_imuTopic.opt = cmd.add_option(
        "--imu-topic", arg_imuTopic.value,
        "rosbag1 topic with the gyroscope (sensor_msgs/Imu), for the gyro-aided rotation "
        "prediction. Enables it: also sets the module's 'imu_label'.");

    arg_imuSensorPose.opt = cmd.add_option(
        "--imu-sensor-pose", arg_imuSensorPose.value,
        "Overrides whatever /tf says about the IMU pose on the vehicle: "
        "'x y z yaw_deg pitch_deg roll_deg'. Must be in the same body frame as "
        "--left-sensor-pose, since the two chain into the IMU-to-camera rotation.");
#endif

#if defined(HAVE_MOLA_INPUT_KITTI)
    argKittiSeq.opt = cmd.add_option(
                             "--input-kitti-seq", argKittiSeq.value,
                             "INPUT DATASET: Use KITTI dataset sequence number 00|01|...")
                          ->option_text("00");
#endif
  }
};

#if defined(HAVE_MOLA_INPUT_KITTI)
std::shared_ptr<mola::OfflineDatasetSource> dataset_from_kitti(
    const std::string& kittiSeqNumber, const mrpt::system::VerbosityLevel logLevel)
{
  auto o = std::make_shared<mola::KittiOdometryDataset>();
  o->setMinLoggingLevel(logLevel);

  const auto cfg = mola::Yaml::FromText(mola::parse_yaml(mrpt::format(
      R""""(
    params:
      base_dir: ${KITTI_BASE_DIR}
      sequence: '%s'
      time_warp_scale: 1.0
      publish_lidar: false
      publish_image_0: true
      publish_image_1: true
      publish_ground_truth: true
)"""",
      kittiSeqNumber.c_str())));

  o->initialize(cfg);
  return o;
}
#endif

#if defined(HAVE_MOLA_INPUT_ROSBAG1)
std::shared_ptr<mola::OfflineDatasetSource> dataset_from_rosbag1(
    Cli& cli, const std::vector<std::string>& rosbag1files,
    const mrpt::system::VerbosityLevel logLevel)
{
  ASSERTMSG_(
      cli.arg_leftTopic.isSet(),
      "Using a rosbag1 as input requires --left-topic <TOPIC_NAME> (and "
      "--right-topic for stereo mode)");

  auto o = std::make_shared<mola::Rosbag1Dataset>();
  o->setMinLoggingLevel(logLevel);

  std::string sensorsYaml = "\n        - topic: '" + cli.arg_leftTopic.getValue() +
                            "'\n          type: CObservationImage\n          sensorLabel: '" +
                            cli.arg_leftLabel.getValue() + "'";
  if (cli.arg_leftSensorPose.isSet())
  {
    sensorsYaml += "\n          fixed_sensor_pose: \"" + cli.arg_leftSensorPose.getValue() +
                   "\"\n          use_fixed_sensor_pose: true";
  }
  if (cli.arg_rightTopic.isSet())
  {
    sensorsYaml += "\n        - topic: '" + cli.arg_rightTopic.getValue() +
                   "'\n          type: CObservationImage\n          sensorLabel: '" +
                   cli.arg_rightLabel.getValue() + "'";
    if (cli.arg_rightSensorPose.isSet())
    {
      sensorsYaml += "\n          fixed_sensor_pose: \"" + cli.arg_rightSensorPose.getValue() +
                     "\"\n          use_fixed_sensor_pose: true";
    }
  }
  if (cli.arg_imuTopic.isSet())
  {
    sensorsYaml += "\n        - topic: '" + cli.arg_imuTopic.getValue() +
                   "'\n          type: CObservationIMU\n          sensorLabel: '" +
                   cli.arg_imuLabel.getValue() + "'";
    if (cli.arg_imuSensorPose.isSet())
    {
      sensorsYaml += "\n          fixed_sensor_pose: \"" + cli.arg_imuSensorPose.getValue() +
                     "\"\n          use_fixed_sensor_pose: true";
    }
  }

  std::string bagsYaml;
  for (const auto& f : rosbag1files)
  {
    bagsYaml += "\n        - '" + f + "'";
  }

  const auto cfg = mola::Yaml::FromText(mola::parse_yaml(mrpt::format(
      R""""(
    params:
      rosbag_filename:%s
      base_link_frame_id: '%s'
      sensors:%s
)"""",
      bagsYaml.c_str(), cli.arg_baseLinkName.getValue().c_str(), sensorsYaml.c_str())));

  o->initialize(cfg);
  return o;
}
#endif

void mola_signal_handler(int s)
{
  std::cerr << "Caught signal " << s << ". Shutting down...\n";
  exit(0);  // NOLINT
}

void mola_install_signal_handler()
{
  struct sigaction sigIntHandler
  {
  };
  sigIntHandler.sa_handler = &mola_signal_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, nullptr);
}

int main_visual_slam(Cli& cli)
{
  auto vslam = mola::VisualSlam::Create();

  mrpt::system::VerbosityLevel logLevel = vslam->getMinLoggingLevel();
  if (cli.arg_verbosity_level.isSet())
  {
    using vl = mrpt::typemeta::TEnumType<mrpt::system::VerbosityLevel>;
    logLevel = vl::name2value(cli.arg_verbosity_level.getValue());
    vslam->setVerbosityLevel(logLevel);
  }

  // Initialize VisualSlam itself (mode, camera labels, rectification, etc.):
  mola::Yaml cfg;
  if (cli.argYAML.isSet())
  {
    cfg = mola::load_yaml_file(cli.argYAML.getValue());
  }
  else
  {
    cfg = mola::Yaml::FromText("params: {}");
  }
  // CLI-provided sensor labels always win, so a single --left-sensor-label /
  // --right-sensor-label pair drives both the dataset source's sensorLabel
  // and VisualSlam's own left_label/right_label without repeating them:
  cfg["params"]["left_label"]  = cli.arg_leftLabel.getValue();
  cfg["params"]["right_label"] = cli.arg_rightLabel.getValue();
#if defined(HAVE_MOLA_INPUT_ROSBAG1)
  // Asking for an IMU topic is the whole intent, so it also turns the module's
  // gyro aiding on rather than needing the same fact restated in the YAML.
  if (cli.arg_imuTopic.isSet())
  {
    cfg["params"]["imu_label"] = cli.arg_imuLabel.getValue();
    if (cli.arg_imuSensorPose.isSet())
    {
      cfg["params"]["imu_pose_on_robot"] = cli.arg_imuSensorPose.getValue();
    }
  }
#endif
  vslam->initialize(cfg);

  // Select dataset input:
  std::shared_ptr<mola::OfflineDatasetSource> dataset;
#if defined(HAVE_MOLA_INPUT_KITTI)
  if (cli.argKittiSeq.isSet())
  {
    dataset = dataset_from_kitti(cli.argKittiSeq.getValue(), logLevel);
  }
  else
#endif
#if defined(HAVE_MOLA_INPUT_ROSBAG1)
      if (cli.argRosbag1.isSet() && !cli.argRosbag1.getValue().empty())
  {
    dataset = dataset_from_rosbag1(cli, cli.argRosbag1.getValue(), logLevel);
  }
  else
#endif
  {
    THROW_EXCEPTION("At least one of the dataset input CLI flags must be defined. Use --help.");
  }
  ASSERT_(dataset);

  mrpt::poses::CPose3DInterpolator estimatedTrajectory;
  const bool                       saveCameraFrame = cli.arg_saveCameraFrame.isSet();

  const double tStart = mrpt::Clock::nowDouble();

  size_t firstDatasetEntry = cli.arg_skipFirstN.isSet() ? cli.arg_skipFirstN.getValue() : 0;
  size_t lastDatasetEntry  = dataset->datasetSize();
  if (cli.arg_firstN.isSet())
  {
    lastDatasetEntry = firstDatasetEntry + cli.arg_firstN.getValue();
  }
  mrpt::keep_min(lastDatasetEntry, dataset->datasetSize());

  const bool progressBarDisabled =
      cli.arg_progressBarPeriod.isSet() && cli.arg_progressBarPeriod.getValue() <= 0;
  const bool   progressBarThrottled = !progressBarDisabled && cli.arg_progressBarPeriod.isSet();
  const double progressBarPeriodFraction =
      cli.arg_progressBarPeriod.isSet() ? cli.arg_progressBarPeriod.getValue() / 100.0 : 0;
  double lastProgressBarPrintedFraction = -1.0;

  if (!progressBarDisabled && !progressBarThrottled)
  {
    std::cout << "\n";
  }

  for (size_t i = firstDatasetEntry; i < lastDatasetEntry; i++)
  {
    const auto sf = dataset->datasetGetObservations(i);
    ASSERT_(sf);

    // Feed everything the entry carries, in order: besides the images, the
    // module also consumes an IMU stream when one is configured.
    for (const auto& o : *sf)
    {
      if (o)
      {
        vslam->onNewObservation(o);
      }
    }

    // Only an image entry advances the trajectory; an IMU-only entry must not
    // duplicate the previous pose at a new timestamp.
    mrpt::obs::CObservation::Ptr obs = sf->getObservationByClass<mrpt::obs::CObservationImage>();
    if (!obs)
    {
      continue;
    }

    if (vslam->isInitialized())
    {
      // Body frame whenever the camera-on-robot extrinsic is known (from /tf
      // or --left-sensor-pose), so the saved trajectory is directly
      // comparable to a dataset's own body-frame ground truth; the camera's
      // own trajectory otherwise.
      estimatedTrajectory.insert(
          obs->timestamp, saveCameraFrame ? vslam->currentPose() : vslam->currentRobotPose());
    }

    const size_t N           = (dataset->datasetSize() - 1);
    const double pc          = N > 0 ? static_cast<double>(i) / static_cast<double>(N) : 1.0;
    const bool   isLastEntry = (i + 1 == lastDatasetEntry);

    bool doPrintProgress = false;
    if (progressBarDisabled)
    {
      doPrintProgress = false;
    }
    else if (progressBarThrottled)
    {
      if (lastProgressBarPrintedFraction < 0 ||
          pc - lastProgressBarPrintedFraction >= progressBarPeriodFraction || isLastEntry)
      {
        doPrintProgress                = true;
        lastProgressBarPrintedFraction = pc;
      }
    }
    else
    {
      doPrintProgress = true;
    }

    if (doPrintProgress)
    {
      const double tNow      = mrpt::Clock::nowDouble();
      const double ETA       = pc > 0 ? (tNow - tStart) * (1.0 / pc - 1) : .0;
      const double totalTime = ETA + (tNow - tStart);

      if (!progressBarThrottled)
      {
        std::cout << "\033[A\33[2KT\r";
      }
      std::cout << mrpt::system::progress(pc, 30)
                << mrpt::format(
                       " %6zu/%6zu (%.02f%%) ETA=%s/T=%s | init=%s landmarks=%zu keyframes=%zu\n",
                       i, N, 100 * pc, mrpt::system::formatTimeInterval(ETA).c_str(),
                       mrpt::system::formatTimeInterval(totalTime).c_str(),
                       vslam->isInitialized() ? "yes" : "no", vslam->numActiveLandmarks(),
                       vslam->numKeyframes());
      std::cout.flush();
    }
  }

  vslam->profiler().dumpAllStats();

  std::cout << "\nFrames with a gyro-measured rotation prediction: " << vslam->numGyroPredictions()
            << "\n";

  if (cli.arg_outPath.isSet())
  {
    const auto fil = cli.arg_outPath.getValue();
    std::cout << "\nSaving estimated path in TUM format to: " << fil << std::endl;
    estimatedTrajectory.saveToTextFile_TUM(fil);
  }

  return 0;
}
}  // namespace

int main(int argc, char** argv)
{
  try
  {
    Cli cli;
    try
    {
      cli.cmd.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
      return cli.cmd.exit(e);
    }

    mola_install_signal_handler();
    return main_visual_slam(cli);
  }
  catch (std::exception& e)
  {
    mola::pretty_print_exception(e, "Exit due to exception:");
    return 1;
  }
}
