/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/feature_detection.h>
#include <mola_libvision/orb_loop_detector.h>

#include <algorithm>

using namespace mola::vision;

std::vector<OrbDescriptor> OrbLoopDetector::describe(const Keyframe::Ptr& kf) const
{
  if (!kf || kf->image.isEmpty())
  {
    return {};
  }
  const mrpt::img::CImage gray = kf->image.grayscale();

  GridDistributorParams gp;
  gp.max_corners  = params_.max_features;
  gp.min_distance = params_.min_distance;
  GridFeatureDistributor dist(gp);
  const auto             pts = dist.detect(gray, {});

  std::vector<bool>          valid;
  auto                       desc = computeOrbDescriptors(gray, pts, valid);
  std::vector<OrbDescriptor> out;
  out.reserve(desc.size());
  for (size_t i = 0; i < desc.size(); ++i)
  {
    if (valid[i])
    {
      out.push_back(desc[i]);
    }
  }
  return out;
}

int OrbLoopDetector::countMatches(
    const std::vector<OrbDescriptor>& a, const std::vector<OrbDescriptor>& b) const
{
  if (a.empty() || b.size() < 2)
  {
    return 0;
  }
  int matches = 0;
  for (const auto& da : a)
  {
    int best   = 1 << 30;
    int second = 1 << 30;
    for (const auto& db : b)
    {
      const int d = hammingDistance(da, db);
      if (d < best)
      {
        second = best;
        best   = d;
      }
      else if (d < second)
      {
        second = d;
      }
    }
    if (best <= params_.max_hamming &&
        static_cast<float>(best) < params_.lowe_ratio * static_cast<float>(second))
    {
      ++matches;
    }
  }
  return matches;
}

void OrbLoopDetector::addKeyframe(const Keyframe::Ptr& kf)
{
  if (!kf)
  {
    return;
  }
  db_[kf->id] = describe(kf);
}

std::vector<LoopCandidate> OrbLoopDetector::detect(const Keyframe::Ptr& kf, int min_id_gap)
{
  std::vector<LoopCandidate> out;
  if (!kf)
  {
    return out;
  }
  const auto query = describe(kf);
  if (static_cast<int>(query.size()) < params_.min_matches)
  {
    return out;
  }

  for (const auto& [id, desc] : db_)
  {
    if (std::abs(id - kf->id) < min_id_gap)
    {
      continue;
    }
    const int m = countMatches(query, desc);
    if (m >= params_.min_matches)
    {
      LoopCandidate c;
      c.query_kf_id   = kf->id;
      c.matched_kf_id = id;
      c.score         = static_cast<float>(m);
      out.push_back(c);
    }
  }

  std::sort(
      out.begin(), out.end(),
      [](const LoopCandidate& x, const LoopCandidate& y) { return x.score > y.score; });
  if (out.size() > params_.max_candidates)
  {
    out.resize(params_.max_candidates);
  }
  return out;
}

void OrbLoopDetector::clear() { db_.clear(); }
