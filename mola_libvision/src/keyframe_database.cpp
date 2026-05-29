/* -------------------------------------------------------------------------
 * mola_libvision: reusable computer vision for MOLA SLAM
 * Copyright (C) 2026, Jose Luis Blanco-Claraco
 * SPDX-License-Identifier: GPL-3.0-or-later
 * ------------------------------------------------------------------------- */
#include <mola_libvision/keyframe_database.h>

#include <algorithm>
#include <set>

using namespace mola::vision;

void KeyframeDatabase::add(const Keyframe::Ptr& kf)
{
  if (!kf)
  {
    return;
  }
  keyframes_[kf->id] = kf;
}

void KeyframeDatabase::remove(int kf_id)
{
  keyframes_.erase(kf_id);
  covis_.erase(kf_id);
  for (auto& [other, edges] : covis_)
  {
    edges.erase(kf_id);
  }
}

Keyframe::Ptr KeyframeDatabase::get(int kf_id) const
{
  const auto it = keyframes_.find(kf_id);
  return (it == keyframes_.end()) ? nullptr : it->second;
}

std::vector<Keyframe::Ptr> KeyframeDatabase::keyframes() const
{
  std::vector<Keyframe::Ptr> out;
  out.reserve(keyframes_.size());
  for (const auto& [id, kf] : keyframes_)
  {
    out.push_back(kf);
  }
  return out;
}

void KeyframeDatabase::updateCovisibility(int min_shared)
{
  covis_.clear();

  // For each keyframe, the set of observed map-point ids.
  std::map<int, std::set<int>> observed;
  for (const auto& [id, kf] : keyframes_)
  {
    auto& s = observed[id];
    for (const auto& f : kf->features)
    {
      if (f.map_point && !f.map_point->isBad())
      {
        s.insert(f.map_point->id());
      }
    }
  }

  // Pairwise intersection counts (symmetric).
  for (auto it_a = observed.begin(); it_a != observed.end(); ++it_a)
  {
    auto it_b = it_a;
    for (++it_b; it_b != observed.end(); ++it_b)
    {
      // Count common map points.
      int         shared = 0;
      const auto& A      = it_a->second;
      const auto& B      = it_b->second;
      const auto& small  = (A.size() <= B.size()) ? A : B;
      const auto& big    = (A.size() <= B.size()) ? B : A;
      for (int mp : small)
      {
        if (big.count(mp) != 0)
        {
          ++shared;
        }
      }
      if (shared >= min_shared)
      {
        covis_[it_a->first][it_b->first] = shared;
        covis_[it_b->first][it_a->first] = shared;
      }
    }
  }
}

int KeyframeDatabase::connectionWeight(int kf_a, int kf_b) const
{
  const auto it = covis_.find(kf_a);
  if (it == covis_.end())
  {
    return 0;
  }
  const auto jt = it->second.find(kf_b);
  return (jt == it->second.end()) ? 0 : jt->second;
}

std::vector<std::pair<int, int>> KeyframeDatabase::covisibleKeyframes(
    int kf_id, int min_weight) const
{
  std::vector<std::pair<int, int>> out;
  const auto                       it = covis_.find(kf_id);
  if (it == covis_.end())
  {
    return out;
  }
  for (const auto& [neighbor, weight] : it->second)
  {
    if (weight >= min_weight)
    {
      out.emplace_back(neighbor, weight);
    }
  }
  std::sort(
      out.begin(), out.end(),
      [](const std::pair<int, int>& a, const std::pair<int, int>& b)
      { return a.second > b.second; });
  return out;
}
