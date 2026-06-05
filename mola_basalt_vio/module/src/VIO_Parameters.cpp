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

void VisualInertialOdometry::Parameters::Visualization::initialize(const Yaml& cfg)
{
  YAML_LOAD_OPT(map_update_decimation, int);
  YAML_LOAD_OPT(show_trajectory, bool);
  // YAML_LOAD_OPT(show_current_observation, bool);
  YAML_LOAD_OPT(show_ground_grid, bool);
  YAML_LOAD_OPT(ground_grid_spacing, float);
  YAML_LOAD_OPT(show_console_messages, bool);
  YAML_LOAD_OPT(current_pose_corner_size, double);
  YAML_LOAD_OPT(local_map_point_size, float);

  if (cfg.has("model"))
  {
    ASSERT_(cfg["model"].isSequence());
    const mrpt::containers::yaml modelsYml = cfg["model"];
    for (size_t _i = 0; _i < modelsYml.asSequence().size(); _i++)
    {
      const mrpt::containers::yaml e = modelsYml[static_cast<int>(_i)];
      ASSERT_(e.isMap());
      auto& m = model.emplace_back();
      ASSERT_(e.has("file"));
      m.file = e["file"].as<std::string>();

      if (m.file.empty())
      {
        model.erase(--model.end());
        continue;
      }

      if (e.has("tf.x"))
      {
        m.tf.x = e["tf.x"].as<float>();
      }
      if (e.has("tf.y"))
      {
        m.tf.y = e["tf.y"].as<float>();
      }
      if (e.has("tf.z"))
      {
        m.tf.z = e["tf.z"].as<float>();
      }
      if (e.has("tf.yaw"))
      {
        m.tf.yaw = mrpt::DEG2RAD(e["tf.yaw"].as<float>());
      }
      if (e.has("tf.pitch"))
      {
        m.tf.pitch = mrpt::DEG2RAD(e["tf.pitch"].as<float>());
      }
      if (e.has("tf.roll"))
      {
        m.tf.roll = mrpt::DEG2RAD(e["tf.roll"].as<float>());
      }
      if (e.has("scale"))
      {
        m.scale = e["scale"].as<float>();
      }
    }
  }

  YAML_LOAD_OPT(gui_subwindow_starts_hidden, bool);
  YAML_LOAD_OPT(camera_follows_vehicle, bool);
  YAML_LOAD_OPT(camera_rotates_with_vehicle, bool);
}

void VisualInertialOdometry::Parameters::TrajectoryOutputOptions::initialize(const Yaml& cfg)
{
  YAML_LOAD_OPT(save_to_file, bool);
  YAML_LOAD_OPT(output_file, std::string);
}

void VisualInertialOdometry::onParameterUpdate(const mrpt::containers::yaml& names_values)
{
  if (names_values.isNullNode() || names_values.empty())
  {
    return;
  }

  ASSERT_(names_values.isMap());

  auto lckState = mrpt::lockHelper(state_mtx_);

  // Load parameters:
  setActive(names_values.getOrDefault("active", isActive()));

  // Special triggering reset "variable":
  if (names_values.getOrDefault("reset_state", false))
  {
    this->enqueue_request(
        [this]()
        {
          MRPT_LOG_INFO("Received a reset() command via parameters update.");
          reset();
        });
  }

  // and reflect changes in the GUI, if used.
  this->enqueue_request(
      [this]()
      {
        auto lckGuiMtx = mrpt::lockHelper(state_gui_mtx_);
#if 0
        if (gui_.cbActive)
        {
          gui_.cbActive->setChecked(isActive());
          gui_.cbMapping->setChecked(params_.local_map_updates.enabled);
          gui_.cbSaveSimplemap->setChecked(params_.simplemap.generate);
        }
#endif
      });
}

void VisualInertialOdometry::onExposeParameters()
{
  mrpt::containers::yaml nv = mrpt::containers::yaml::Map();
  nv["active"]              = isActive();
  nv["reset_state"]         = false;

  this->exposeParameters(nv);
}

}  // namespace mola
