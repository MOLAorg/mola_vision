/* -------------------------------------------------------------------------
 * mola_visual_tracking: a MOLA demo front-end that detects and tracks image
 * features and shows them in a MolaViz subwindow (no 3D reconstruction).
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/optical_flow.h>
#include <mola_visual_tracking/VisualTracking.h>
#include <mrpt/containers/yaml.h>
#include <mrpt/img/TColor.h>
#include <mrpt/obs/CObservationImage.h>

#include <algorithm>
#include <sstream>
#include <string>

using namespace mola;

IMPLEMENTS_MRPT_OBJECT(VisualTracking, FrontEndBase, mola)

void VisualTracking::initialize_frontend(const Yaml& c)
{
  MRPT_START

  // Optional parameters (all have sensible defaults):
  if (c.has("params"))
  {
    const auto cfg = c["params"];
    if (cfg.has("sensor_label"))
    {
      sensor_label_ = cfg["sensor_label"].as<std::string>();
    }
    if (cfg.has("subwindow_title"))
    {
      subwindow_title_ = cfg["subwindow_title"].as<std::string>();
    }
    if (cfg.has("win_pos"))
    {
      win_pos_ = cfg["win_pos"].as<std::string>();
    }
    if (cfg.has("max_features"))
    {
      max_features_ = cfg["max_features"].as<int>();
    }
    if (cfg.has("min_distance"))
    {
      min_distance_ = cfg["min_distance"].as<float>();
    }
    if (cfg.has("redetect_below"))
    {
      redetect_below_ = cfg["redetect_below"].as<int>();
    }
    if (cfg.has("lk_win_size"))
    {
      lk_win_size_ = cfg["lk_win_size"].as<int>();
    }
    if (cfg.has("lk_max_levels"))
    {
      lk_max_levels_ = cfg["lk_max_levels"].as<int>();
    }
    if (cfg.has("trail_length"))
    {
      trail_length_ = cfg["trail_length"].as<int>();
    }
    if (cfg.has("apply_fmatrix"))
    {
      apply_fmatrix_ = cfg["apply_fmatrix"].as<bool>();
    }
  }

  MRPT_LOG_INFO_STREAM(
      "VisualTracking initialized. sensor_label='" << sensor_label_
                                                   << "' max_features=" << max_features_);
  MRPT_END
}

void VisualTracking::spinOnce()
{
  // All work happens in onNewObservation(); nothing periodic to do here.
}

void VisualTracking::onNewObservation(const CObservation::ConstPtr& o)
{
  MRPT_START
  if (!o)
  {
    return;
  }

  auto obs = std::dynamic_pointer_cast<const mrpt::obs::CObservationImage>(o);
  if (!obs)
  {
    return;  // not an image observation
  }
  if (!sensor_label_.empty() && obs->sensorLabel != sensor_label_)
  {
    return;  // not the sensor we are tracking
  }

  obs->load();
  if (obs->image.isEmpty())
  {
    return;
  }

  processImage(obs->image, obs->cameraParams);

  MRPT_END
}

void VisualTracking::processImage(
    const mrpt::img::CImage& image, [[maybe_unused]] const mrpt::img::TCamera& cam)
{
  using mrpt::math::TPoint2Df;

  // Work on a grayscale copy.
  const mrpt::img::CImage gray = image.grayscale();

  std::vector<bool> is_outlier;

  if (prev_gray_.isEmpty() || pts_.size() < static_cast<size_t>(redetect_below_))
  {
    // (Re)seed features, avoiding the currently tracked ones.
    mola::vision::GridDistributorParams gp;
    gp.max_corners  = max_features_;
    gp.min_distance = min_distance_;
    mola::vision::GridFeatureDistributor dist(gp);

    const auto fresh = dist.detect(gray, pts_);
    for (const auto& p : fresh)
    {
      if (pts_.size() >= static_cast<size_t>(max_features_))
      {
        break;
      }
      pts_.push_back(p);
      ids_.push_back(next_id_);
      trails_[next_id_].push_back(p);
      ++next_id_;
    }
    is_outlier.assign(pts_.size(), false);
  }

  if (!prev_gray_.isEmpty() && !pts_.empty())
  {
    // Track previous points into the current frame.
    std::vector<TPoint2Df>                 next_pts;
    std::vector<mola::vision::TrackStatus> status;
    mola::vision::LKParams                 lk;
    lk.win_size   = lk_win_size_;
    lk.max_levels = lk_max_levels_;
    mola::vision::calcOpticalFlowPyrLK(prev_gray_, gray, pts_, next_pts, status, lk);

    if (apply_fmatrix_)
    {
      mola::vision::fundamentalMatrixFilter(pts_, next_pts, status);
    }

    // Keep only successfully-tracked features; update trails.
    std::vector<TPoint2Df> kept_pts;
    std::vector<int>       kept_ids;
    is_outlier.clear();
    for (size_t i = 0; i < next_pts.size(); ++i)
    {
      if (status[i] == mola::vision::TrackStatus::LOST)
      {
        trails_.erase(ids_[i]);
        continue;
      }
      kept_pts.push_back(next_pts[i]);
      kept_ids.push_back(ids_[i]);
      is_outlier.push_back(status[i] == mola::vision::TrackStatus::OUTLIER);

      auto& tr = trails_[ids_[i]];
      tr.push_back(next_pts[i]);
      while (static_cast<int>(tr.size()) > trail_length_)
      {
        tr.pop_front();
      }
    }
    pts_ = std::move(kept_pts);
    ids_ = std::move(kept_ids);
  }

  if (is_outlier.size() != pts_.size())
  {
    is_outlier.assign(pts_.size(), false);
  }

  // Build an annotated RGB image and push it to the GUI.
  mrpt::img::CImage rgb = image.colorImage();
  publishViz(rgb, is_outlier);

  prev_gray_ = gray;
}

void VisualTracking::publishViz(
    const mrpt::img::CImage& rgb_in, const std::vector<bool>& is_outlier)
{
  using mrpt::img::TColor;

  // We must not draw on a const image; make a working copy.
  mrpt::img::CImage img = rgb_in.makeDeepCopy();

  const TColor colInlier(0, 220, 0);
  const TColor colOutlier(230, 40, 40);
  const TColor colTrail(255, 200, 0);

  for (size_t i = 0; i < pts_.size(); ++i)
  {
    const auto& p   = pts_[i];
    const bool  bad = is_outlier[i];

    // Motion trail.
    const auto it = trails_.find(ids_[i]);
    if (it != trails_.end() && it->second.size() >= 2)
    {
      const auto& tr = it->second;
      for (size_t k = 1; k < tr.size(); ++k)
      {
        img.line(
            {static_cast<int>(tr[k - 1].x), static_cast<int>(tr[k - 1].y)},
            {static_cast<int>(tr[k].x), static_cast<int>(tr[k].y)}, colTrail);
      }
    }

    img.drawCircle(
        {static_cast<int>(p.x), static_cast<int>(p.y)}, 3, bad ? colOutlier : colInlier, 1);
  }

  img.textOut({8, 8}, "tracked: " + std::to_string(pts_.size()), TColor(255, 255, 255));

  // Wrap into a CObservationImage so the built-in MolaViz image handler shows it.
  auto annotated         = mrpt::obs::CObservationImage::Create();
  annotated->sensorLabel = subwindow_title_;
  annotated->image       = std::move(img);

  if (!visualizer_)
  {
    return;  // no GUI running
  }

  if (!gui_created_)
  {
    mola::gui::WindowDescription desc;
    desc.title = subwindow_title_;
    if (!win_pos_.empty())
    {
      std::string cleaned = win_pos_;
      for (char& ch : cleaned)
      {
        if (ch == '[' || ch == ']' || ch == ',')
        {
          ch = ' ';
        }
      }
      int                x = 0, y = 0, w = 0, h = 0;
      std::istringstream ss(cleaned);
      if ((ss >> x) && (ss >> y) && (ss >> w) && (ss >> h) && w > 0 && h > 0)
      {
        desc.position = {x, y};
        desc.size     = {w, h};
      }
    }
    visualizer_->create_subwindow_from_description(desc).get();
    gui_created_ = true;
  }

  visualizer_->subwindow_update_visualization(annotated, subwindow_title_);
}
