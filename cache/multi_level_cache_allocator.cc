//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "cache/multi_level_cache_allocator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

namespace ROCKSDB_NAMESPACE {
namespace {

struct FractionalPart {
  size_t index;
  double fraction;
};

double ClampRatio(double ratio) {
  if (ratio < 0.0) {
    return 0.0;
  }
  if (ratio > 1.0) {
    return 1.0;
  }
  return ratio;
}

void ApplyCompactionAwareShiftByDataDelta(
    const MultiLevelCache::LevelMetricsSnapshot& snapshot,
    const std::vector<uint64_t>& prev_data_sizes,
    const std::vector<uint64_t>& prev_lookups,
    const std::vector<uint64_t>& prev_hits,
    double shift_ratio,
    double max_total_shift_ratio,
    bool enable_debug,
    uint64_t round_id,
    std::vector<size_t>* capacities) {
  if (capacities == nullptr || capacities->size() < 2 || shift_ratio <= 0.0) {
    return;
  }
  const size_t level_count = capacities->size();
  if (snapshot.data_sizes.size() != level_count ||
      snapshot.lookups.size() != level_count ||
      snapshot.hits.size() != level_count ||
      snapshot.capacities.size() != level_count ||
      prev_data_sizes.size() != level_count ||
      prev_lookups.size() != level_count ||
      prev_hits.size() != level_count) {
    return;
  }
  const double clamped_shift_ratio = ClampRatio(shift_ratio);
  if (clamped_shift_ratio <= 0.0) {
    return;
  }
  const double clamped_max_total_ratio = ClampRatio(max_total_shift_ratio);
  const size_t total_capacity =
      std::accumulate(capacities->begin(), capacities->end(), static_cast<size_t>(0));
  const size_t max_total_shift = static_cast<size_t>(
      std::floor(static_cast<double>(total_capacity) * clamped_max_total_ratio));
  if (max_total_shift == 0) {
    return;
  }

  size_t total_shifted = 0;
  for (size_t level = 0; level + 1 < level_count; ++level) {
    const uint64_t prev_data = prev_data_sizes[level];
    const uint64_t curr_data = snapshot.data_sizes[level];
    if (prev_data == 0 || curr_data >= prev_data) {
      continue;
    }
    const uint64_t moved_data = prev_data - curr_data;
    const double overlap_fraction =
        std::min(1.0, static_cast<double>(moved_data) / static_cast<double>(prev_data));

    const uint64_t curr_lookups = snapshot.lookups[level];
    const uint64_t curr_hits = snapshot.hits[level];
    const uint64_t delta_lookups =
        curr_lookups >= prev_lookups[level] ? curr_lookups - prev_lookups[level] : 0;
    const uint64_t delta_hits =
        curr_hits >= prev_hits[level] ? curr_hits - prev_hits[level] : 0;
    const double recent_hit_rate =
        delta_lookups > 0
            ? std::max(0.0, std::min(1.0, static_cast<double>(delta_hits) /
                                              static_cast<double>(delta_lookups)))
            : (curr_lookups > 0
                   ? std::max(0.0, std::min(1.0, static_cast<double>(curr_hits) /
                                                     static_cast<double>(curr_lookups)))
                   : 0.0);

    const double cached_hot_bytes_estimate =
        static_cast<double>(snapshot.capacities[level]) * recent_hit_rate *
        overlap_fraction;
    size_t shift_bytes = static_cast<size_t>(
        std::floor(clamped_shift_ratio * cached_hot_bytes_estimate));
    if (shift_bytes == 0) {
      continue;
    }
    shift_bytes = std::min(shift_bytes, (*capacities)[level]);
    if (shift_bytes == 0) {
      continue;
    }
    if (total_shifted >= max_total_shift) {
      break;
    }
    shift_bytes = std::min(shift_bytes, max_total_shift - total_shifted);
    if (shift_bytes == 0) {
      continue;
    }

    (*capacities)[level] -= shift_bytes;
    (*capacities)[level + 1] += shift_bytes;
    total_shifted += shift_bytes;

    if (enable_debug) {
      std::fprintf(stdout,
                   "[MLC][compaction_shift] round=%" PRIu64
                   " L%zu->L%zu moved_data=%" PRIu64
                   " overlap=%.4f hit=%.4f shift=%zu shifted_total=%zu cap_from=%zu cap_to=%zu\n",
                   round_id, level, level + 1, moved_data, overlap_fraction,
                   recent_hit_rate, shift_bytes, total_shifted,
                   (*capacities)[level], (*capacities)[level + 1]);
    }
  }
}

}  // namespace

MultiLevelCacheAllocator::MultiLevelCacheAllocator(
    std::shared_ptr<MultiLevelCache> cache, MetricsProvider provider,
    MultiLevelAllocationOptions options)
    : cache_(std::move(cache)),
      provider_(std::move(provider)),
      options_(std::move(options)) {
  if (options_.interval_ms == 0) {
    options_.interval_ms = 1000;
  }
  if (options_.solver_max_iterations <= 0) {
    options_.solver_max_iterations = 80;
  }
  if (options_.solver_epsilon <= 0) {
    options_.solver_epsilon = 1e-9;
  }
  options_.smoothing_ratio = ClampRatio(options_.smoothing_ratio);
  options_.compaction_shift_ratio = ClampRatio(options_.compaction_shift_ratio);
  options_.compaction_shift_max_total_ratio =
      ClampRatio(options_.compaction_shift_max_total_ratio);
  options_.total_deadband_ratio = ClampRatio(options_.total_deadband_ratio);
  options_.per_level_deadband_ratio =
      ClampRatio(options_.per_level_deadband_ratio);
  if (options_.max_interval_backoff == 0) {
    options_.max_interval_backoff = 1;
  }
}

MultiLevelCacheAllocator::~MultiLevelCacheAllocator() { Stop(); }

void MultiLevelCacheAllocator::Start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true,
                                        std::memory_order_acq_rel)) {
    return;
  }
  worker_ = std::thread(&MultiLevelCacheAllocator::BackgroundLoop, this);
}

void MultiLevelCacheAllocator::Stop() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false,
                                        std::memory_order_acq_rel)) {
    return;
  }
  if (worker_.joinable()) {
    worker_.join();
  }
}

Status MultiLevelCacheAllocator::RunOnce() {
  std::lock_guard<std::mutex> lock(mu_);
  return RunOnceLocked();
}

Status MultiLevelCacheAllocator::SolveCapacities(
    const std::vector<double>& lambda, const std::vector<double>& data,
    const std::vector<double>& alpha, size_t total_capacity,
    std::vector<size_t>* capacities, double epsilon, int max_iterations,
    const std::vector<double>& upper_bounds) {
  if (capacities == nullptr) {
    return Status::InvalidArgument("capacities output cannot be null");
  }
  const size_t levels = lambda.size();
  if (levels == 0 || data.size() != levels || alpha.size() != levels) {
    return Status::InvalidArgument(
        "lambda/data/alpha sizes must match and be non-zero");
  }
  if (max_iterations <= 0 || epsilon <= 0.0) {
    return Status::InvalidArgument("invalid solver configuration");
  }
  if (total_capacity == 0) {
    capacities->assign(levels, 0);
    return Status::OK();
  }

  // Optional per-level upper bound: a level cannot usefully cache more bytes
  // than it stores. When supplied, the water-filling caps each level and the
  // surplus flows to higher-marginal levels; the realizable budget is also
  // bounded by the total usable bytes.
  const bool use_caps = upper_bounds.size() == levels;
  size_t effective_budget = total_capacity;
  if (use_caps) {
    double sum_ub = 0.0;
    for (size_t i = 0; i < levels; ++i) {
      sum_ub += std::max(0.0, upper_bounds[i]);
    }
    if (sum_ub < static_cast<double>(total_capacity)) {
      effective_budget = static_cast<size_t>(std::floor(sum_ub));
    }
  }
  if (effective_budget == 0) {
    capacities->assign(levels, 0);
    return Status::OK();
  }

  std::vector<double> a(levels, 0.0);
  double a_max = 0.0;
  bool has_positive_term = false;
  for (size_t i = 0; i < levels; ++i) {
    if (lambda[i] > 0.0 && alpha[i] > 0.0 && data[i] > 0.0) {
      a[i] = lambda[i] * alpha[i] / data[i];
      a_max = std::max(a_max, a[i]);
      has_positive_term = true;
    }
  }

  // Diagnostics for the sparse-table Evict pathology: pin down whether capacity
  // spikes originate from the EqualSplit degenerate-signal fallback or from the
  // data/alpha prefactor exploding in the water-filling. Gated by MLC_ALLOC_DEBUG.
  static const bool kSolveDebug = getenv("MLC_ALLOC_DEBUG") != nullptr;
  static std::atomic<uint64_t> dbg_solve_calls{0};
  const uint64_t solve_call = dbg_solve_calls.fetch_add(1) + 1;

  if (!has_positive_term || a_max <= 0.0) {
    if (kSolveDebug) {
      fprintf(stderr,
              "[MLC_SOLVE] call=%llu NO_SIGNAL_FALLBACK levels=%zu "
              "budget=%lluMiB (no fundable term) -> zeros (hold via floor)\n",
              (unsigned long long)solve_call, levels,
              (unsigned long long)(total_capacity >> 20));
    }
    // Degenerate round: no level has a fundable (lambda>=floor, alpha>0, data>0)
    // term. Do NOT EqualSplit the whole budget across all levels -- that spikes
    // every level (including idle deep ones) to budget/levels and ratchets their
    // grow-only tables. Return zeros; the caller then lifts active levels to
    // their data-share floor and smooths from the previous allocation, i.e.
    // effectively holds the last stable allocation instead of thrashing.
    capacities->assign(levels, 0);
    return Status::OK();
  }

  auto sum_capacities_for_mu = [&](double mu,
                                   std::vector<double>* out) -> double {
    double sum = 0.0;
    for (size_t i = 0; i < levels; ++i) {
      double capacity = 0.0;
      if (a[i] > 0.0 && mu > 0.0 && mu < a[i]) {
        capacity = (data[i] / alpha[i]) * std::log(a[i] / mu);
        if (capacity < 0.0) {
          capacity = 0.0;
        }
      }
      if (use_caps) {
        const double ub = std::max(0.0, upper_bounds[i]);
        if (capacity > ub) {
          capacity = ub;
        }
      }
      if (out != nullptr) {
        (*out)[i] = capacity;
      }
      sum += capacity;
    }
    return sum;
  };

  double low = 0.0;
  double high = a_max;
  for (int iter = 0; iter < max_iterations; ++iter) {
    const double mid = 0.5 * (low + high);
    if (mid <= 0.0) {
      low = std::numeric_limits<double>::min();
      continue;
    }
    const double total = sum_capacities_for_mu(mid, nullptr);
    if (total > static_cast<double>(effective_budget)) {
      low = mid;
    } else {
      high = mid;
    }
    if ((high - low) <= epsilon * std::max(1.0, high)) {
      break;
    }
  }

  std::vector<double> continuous(levels, 0.0);
  sum_capacities_for_mu(high, &continuous);
  if (kSolveDebug && (solve_call % 50 == 0 || solve_call <= 5)) {
    std::string raw;
    for (size_t i = 0; i < levels; ++i) {
      raw += std::to_string(static_cast<uint64_t>(continuous[i]) >> 20);
      raw += (i + 1 < levels) ? "," : "";
    }
    std::string scale;
    for (size_t i = 0; i < levels; ++i) {
      scale += std::to_string(static_cast<uint64_t>(data[i]) >> 20);
      scale += (i + 1 < levels) ? "," : "";
    }
    fprintf(stderr,
            "[MLC_SOLVE] call=%llu raw_solve(MiB)=[%s] scale(MiB)=[%s]\n",
            (unsigned long long)solve_call, raw.c_str(), scale.c_str());
  }
  *capacities = QuantizeToBudget(continuous, effective_budget);
  if (use_caps) {
    // Guard against quantization rounding pushing a level past its cap.
    for (size_t i = 0; i < levels; ++i) {
      const size_t ub =
          static_cast<size_t>(std::floor(std::max(0.0, upper_bounds[i])));
      if ((*capacities)[i] > ub) {
        (*capacities)[i] = ub;
      }
    }
  }
  return Status::OK();
}

void MultiLevelCacheAllocator::EqualSplit(size_t total_capacity, size_t levels,
                                          std::vector<size_t>* capacities) {
  capacities->assign(levels, 0);
  if (levels == 0) {
    return;
  }
  const size_t base = total_capacity / levels;
  const size_t rem = total_capacity % levels;
  for (size_t i = 0; i < levels; ++i) {
    (*capacities)[i] = base + (i < rem ? 1 : 0);
  }
}

std::vector<size_t> MultiLevelCacheAllocator::QuantizeToBudget(
    const std::vector<double>& values, size_t budget) {
  std::vector<size_t> result(values.size(), 0);
  if (values.empty()) {
    return result;
  }

  std::vector<FractionalPart> fractions;
  fractions.reserve(values.size());
  size_t used = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    const double non_negative = std::max(0.0, values[i]);
    const double floor_value = std::floor(non_negative);
    const size_t base = static_cast<size_t>(
        std::min<double>(floor_value, static_cast<double>(SIZE_MAX)));
    result[i] = base;
    used += base;
    fractions.push_back({i, non_negative - floor_value});
  }

  if (used > budget) {
    const double scale = static_cast<double>(budget) / static_cast<double>(used);
    used = 0;
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] = static_cast<size_t>(std::floor(result[i] * scale));
      used += result[i];
    }
  }

  size_t remaining = budget - used;
  // Distribute any under-allocation surplus ONLY across levels the solver
  // actually funded (result[i] > 0), never equally across all levels. When the
  // access signal is weak (e.g. config B's foreground-only lambda), the solver
  // wants far less than the full budget; spreading the surplus equally hands
  // idle deep levels (L5/L6) a chunk of capacity they do not need, which grows
  // their grow-only AutoHCC hash tables. A later round then drains them, leaving
  // the table oversized and sparse -> every insert-time Evict sweeps it (CPU
  // thrash + lock contention = the config-B collapse). Keeping the surplus on
  // already-funded levels (re-capped afterward at their data size by the caller,
  // so any true excess is simply left unallocated) keeps idle levels at zero.
  if (!result.empty() && remaining > 0) {
    size_t funded = 0;
    for (size_t i = 0; i < result.size(); ++i) {
      if (result[i] > 0) {
        ++funded;
      }
    }
    if (funded > 0) {
      const size_t base_add = remaining / funded;
      if (base_add > 0) {
        for (size_t i = 0; i < result.size(); ++i) {
          if (result[i] > 0) {
            result[i] += base_add;
            remaining -= base_add;
          }
        }
      }
      std::sort(fractions.begin(), fractions.end(),
                [](const FractionalPart& lhs, const FractionalPart& rhs) {
                  return lhs.fraction > rhs.fraction;
                });
      for (size_t i = 0; i < fractions.size() && remaining > 0; ++i) {
        const size_t idx = fractions[i].index;
        if (result[idx] > 0) {
          ++result[idx];
          --remaining;
        }
      }
    }
    // funded == 0 (solver returned all zeros): leave result all zero. The caller
    // lifts active levels to their floor and smooths from the previous
    // allocation, i.e. holds the last stable state rather than EqualSplitting.
  }
  return result;
}

void MultiLevelCacheAllocator::SmoothCapacities(
    const std::vector<size_t>& previous, const std::vector<size_t>& target,
    double ratio, std::vector<size_t>* out) {
  const double clamped_ratio = ClampRatio(ratio);
  const size_t levels = target.size();
  out->assign(levels, 0);
  if (previous.size() != levels || clamped_ratio >= 1.0) {
    *out = target;
    return;
  }
  for (size_t i = 0; i < levels; ++i) {
    const double blended =
        (1.0 - clamped_ratio) * static_cast<double>(previous[i]) +
        clamped_ratio * static_cast<double>(target[i]);
    (*out)[i] = static_cast<size_t>(std::max(0.0, std::floor(blended)));
  }

  const size_t previous_sum =
      std::accumulate(previous.begin(), previous.end(), static_cast<size_t>(0));
  const size_t target_sum =
      std::accumulate(target.begin(), target.end(), static_cast<size_t>(0));
  const size_t budget = target_sum > 0 ? target_sum : previous_sum;
  size_t smoothed_sum =
      std::accumulate(out->begin(), out->end(), static_cast<size_t>(0));
  if (smoothed_sum < budget) {
    size_t remain = budget - smoothed_sum;
    if (levels > 0) {
      const size_t base_add = remain / levels;
      if (base_add > 0) {
        for (size_t i = 0; i < levels; ++i) {
          (*out)[i] += base_add;
        }
        remain -= base_add * levels;
      }
      for (size_t i = 0; i < levels && remain > 0; ++i, --remain) {
        ++(*out)[i];
      }
    }
  } else if (smoothed_sum > budget) {
    size_t over = smoothed_sum - budget;
    for (size_t i = 0; i < levels && over > 0; ++i) {
      const size_t dec = std::min((*out)[i], over);
      (*out)[i] -= dec;
      over -= dec;
    }
  }
}

void MultiLevelCacheAllocator::EnforceMinActiveLevelFloor(
    const std::vector<size_t>& in_capacities,
    const std::vector<uint64_t>& level_data_sizes, size_t total_budget,
    size_t min_active_level_capacity_bytes, std::vector<size_t>* out) {
  std::vector<size_t> adjusted = in_capacities;
  const size_t levels = adjusted.size();
  if (levels == 0 || min_active_level_capacity_bytes == 0 ||
      level_data_sizes.size() != levels) {
    *out = in_capacities;
    return;
  }

  std::vector<size_t> active;
  active.reserve(levels);
  for (size_t i = 0; i < levels; ++i) {
    if (level_data_sizes[i] > 0) {
      active.push_back(i);
    }
  }
  if (active.empty()) {
    return;
  }

  std::vector<size_t> required_floor(levels, 0);
  const uint64_t requested =
      static_cast<uint64_t>(active.size()) *
      static_cast<uint64_t>(min_active_level_capacity_bytes);
  if (requested <= total_budget) {
    for (size_t idx : active) {
      required_floor[idx] = min_active_level_capacity_bytes;
    }
  } else {
    const size_t base = total_budget / active.size();
    size_t rem = total_budget % active.size();
    for (size_t idx : active) {
      required_floor[idx] = base + (rem > 0 ? 1 : 0);
      if (rem > 0) {
        --rem;
      }
    }
  }

  uint64_t deficit = 0;
  for (size_t idx : active) {
    if (adjusted[idx] < required_floor[idx]) {
      deficit += static_cast<uint64_t>(required_floor[idx] - adjusted[idx]);
    }
  }
  if (deficit == 0) {
    *out = std::move(adjusted);
    return;
  }

  // Drain removable bytes from donors while preserving active floors.
  std::vector<size_t> donor_order(levels);
  std::iota(donor_order.begin(), donor_order.end(), 0);
  std::sort(donor_order.begin(), donor_order.end(),
            [&](size_t a, size_t b) { return adjusted[a] > adjusted[b]; });

  for (size_t idx : donor_order) {
    const size_t floor = required_floor[idx];
    if (adjusted[idx] <= floor) {
      continue;
    }
    const size_t removable = adjusted[idx] - floor;
    const size_t take =
        static_cast<size_t>(std::min<uint64_t>(removable, deficit));
    adjusted[idx] -= take;
    deficit -= take;
    if (deficit == 0) {
      break;
    }
  }

  if (deficit > 0) {
    // Cannot satisfy all floors under current budget distribution.
    *out = in_capacities;
    return;
  }

  for (size_t idx : active) {
    if (adjusted[idx] < required_floor[idx]) {
      const size_t add = required_floor[idx] - adjusted[idx];
      adjusted[idx] += add;
    }
  }
  *out = std::move(adjusted);
}

void MultiLevelCacheAllocator::EnforceDataShareFloor(
    const std::vector<size_t>& in_capacities,
    const std::vector<uint64_t>& level_data_sizes, size_t total_budget,
    double ratio, size_t floor_min_bytes, std::vector<size_t>* out,
    const std::vector<unsigned char>* relief_mask) {
  *out = in_capacities;
  const size_t levels = in_capacities.size();
  if (levels == 0 || level_data_sizes.size() != levels || ratio <= 0.0 ||
      total_budget == 0) {
    return;
  }

  std::vector<size_t> active;
  active.reserve(levels);
  uint64_t sum_active_data = 0;
  for (size_t i = 0; i < levels; ++i) {
    if (level_data_sizes[i] > 0) {
      active.push_back(i);
      sum_active_data += level_data_sizes[i];
    }
  }
  if (active.empty() || sum_active_data == 0) {
    return;
  }

  // Per-active-level floor: data-share-weighted slice of the reserved pool,
  // lifted to the absolute minimum so small upper levels stay functional.
  std::vector<size_t> adjusted = in_capacities;
  std::vector<size_t> required_floor(levels, 0);
  const double pool = static_cast<double>(total_budget) * ratio;
  for (size_t idx : active) {
    const double share =
        static_cast<double>(level_data_sizes[idx]) /
        static_cast<double>(sum_active_data);
    size_t fl = static_cast<size_t>(std::floor(pool * share));
    if (fl < floor_min_bytes) {
      fl = floor_min_bytes;
    }
    required_floor[idx] = fl;
  }

  // Only lift levels the mask permits (persistence gate). When no mask is given
  // every active level below its floor is lifted (original behaviour).
  uint64_t deficit = 0;
  for (size_t idx : active) {
    if (relief_mask != nullptr && (*relief_mask)[idx] == 0) {
      continue;
    }
    if (adjusted[idx] < required_floor[idx]) {
      deficit += static_cast<uint64_t>(required_floor[idx] - adjusted[idx]);
    }
  }
  if (deficit == 0) {
    *out = std::move(adjusted);
    return;
  }

  // Drain removable bytes from donors (largest first), never below their floor.
  std::vector<size_t> donor_order(levels);
  std::iota(donor_order.begin(), donor_order.end(), 0);
  std::sort(donor_order.begin(), donor_order.end(),
            [&](size_t a, size_t b) { return adjusted[a] > adjusted[b]; });
  for (size_t idx : donor_order) {
    const size_t floor = required_floor[idx];
    if (adjusted[idx] <= floor) {
      continue;
    }
    const size_t removable = adjusted[idx] - floor;
    const size_t take =
        static_cast<size_t>(std::min<uint64_t>(removable, deficit));
    adjusted[idx] -= take;
    deficit -= take;
    if (deficit == 0) {
      break;
    }
  }
  if (deficit > 0) {
    // Cannot satisfy all floors under the current budget distribution; leave
    // the input unchanged (best-effort, avoid a partial/invalid reshape).
    return;
  }
  for (size_t idx : active) {
    if (adjusted[idx] < required_floor[idx]) {
      adjusted[idx] += required_floor[idx] - adjusted[idx];
    }
  }
  *out = std::move(adjusted);
}

void MultiLevelCacheAllocator::BackgroundLoop() {
  // Priming: always run one round immediately so the initial equal-split is
  // replaced by a model-driven allocation without waiting for the first op
  // window / interval to elapse.
  {
    std::lock_guard<std::mutex> lock(mu_);
    Status s = RunOnceLocked();
    s.PermitUncheckedError();
  }
  if (cache_) {
    last_round_lookups_ = cache_->GetTotalLookups();
  }

  while (running_.load(std::memory_order_acquire)) {
    // Short poll so Stop() stays responsive and (when op-gating is enabled) the
    // op counter is checked frequently. When adjust_interval_ops == 0 we fall
    // back to the legacy fixed wall-clock cadence.
    const uint64_t poll_ms =
        options_.adjust_interval_ops > 0
            ? std::min<uint64_t>(50, std::max<uint64_t>(1, options_.interval_ms))
            : options_.interval_ms;
    uint64_t slept = 0;
    while (slept < poll_ms && running_.load(std::memory_order_acquire)) {
      const uint64_t chunk = std::min<uint64_t>(50, poll_ms - slept);
      std::this_thread::sleep_for(std::chrono::milliseconds(chunk));
      slept += chunk;
    }
    if (!running_.load(std::memory_order_acquire)) {
      break;
    }

    if (options_.adjust_interval_ops > 0) {
      // Op-count cadence: decouple the number of adjustment rounds from thread
      // count / throughput (wall-clock cadence made the number of rounds scale
      // with wall time, so hit ratio moved non-monotonically with thread
      // count). Gate on TOTAL lookups, NOT foreground-only: spacing rounds by
      // foreground ops makes each round's window wide enough that the model's
      // inter-round swing drops below the stability threshold and the (now
      // large, hundreds-of-MiB) reallocation is APPLIED every round. Each apply
      // runs PurgeSubCacheToCapacity under the shard locks, stalling the hot
      // path (measured: ~4.5x throughput collapse, 75 -> 17 KTPS at t8). Gating
      // on total lookups keeps rounds frequent so consecutive swings stay large
      // and the stability gate suppresses them, holding a stable allocation.
      const uint64_t total_lookups =
          cache_ ? cache_->GetTotalLookups() : 0;
      if (total_lookups - last_round_lookups_ < options_.adjust_interval_ops) {
        continue;
      }
      last_round_lookups_ = total_lookups;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      Status s = RunOnceLocked();
      s.PermitUncheckedError();
    }
  }
}

void MultiLevelCacheAllocator::ApplyAntiOscillation(
    std::vector<size_t>* proposed, size_t budget) {
  const size_t levels = proposed->size();
  if (last_capacities_.size() != levels) {
    return;
  }
  if (shrink_streak_.size() != levels) {
    shrink_streak_.assign(levels, 0);
  }
  // Per-level deadband: ignore small wiggles so noise does not move (and,
  // with purge-on-shrink, destroy) a level's working set.
  for (size_t i = 0; i < levels; ++i) {
    const size_t last = last_capacities_[i];
    size_t& prop = (*proposed)[i];
    const uint64_t delta = prop > last ? prop - last : last - prop;
    const uint64_t deadband = std::max<uint64_t>(
        options_.per_level_deadband_min_bytes,
        static_cast<uint64_t>(static_cast<double>(last) *
                              options_.per_level_deadband_ratio));
    if (delta < deadband) {
      prop = last;
    }
  }
  // Shrink hysteresis: a shrink must be proposed for shrink_confirm_rounds
  // consecutive rounds before taking effect, and then closes only half the
  // gap per round. Grows are applied immediately.
  if (options_.shrink_confirm_rounds > 1) {
    for (size_t i = 0; i < levels; ++i) {
      const size_t last = last_capacities_[i];
      size_t& prop = (*proposed)[i];
      if (prop < last) {
        ++shrink_streak_[i];
        if (shrink_streak_[i] < options_.shrink_confirm_rounds) {
          prop = last;
        } else {
          prop = last - (last - prop) / 2;
        }
      } else {
        shrink_streak_[i] = 0;
      }
    }
  }
  // Rebalance to the budget: deferred/halved shrinks may leave the sum above
  // it (the grows they funded no longer fit), spare slack goes back to the
  // largest level.
  uint64_t sum = std::accumulate(proposed->begin(), proposed->end(),
                                 static_cast<uint64_t>(0));
  if (sum > budget) {
    uint64_t over = sum - budget;
    // Trim the speculative side first: levels currently growing, largest
    // grow first, never below their previous capacity.
    std::vector<size_t> grow_order;
    for (size_t i = 0; i < levels; ++i) {
      if ((*proposed)[i] > last_capacities_[i]) {
        grow_order.push_back(i);
      }
    }
    std::sort(grow_order.begin(), grow_order.end(), [&](size_t a, size_t b) {
      return (*proposed)[a] - last_capacities_[a] >
             (*proposed)[b] - last_capacities_[b];
    });
    for (size_t i : grow_order) {
      if (over == 0) {
        break;
      }
      const uint64_t grow = (*proposed)[i] - last_capacities_[i];
      const uint64_t take = std::min<uint64_t>(grow, over);
      (*proposed)[i] -= static_cast<size_t>(take);
      over -= take;
    }
    // Defensive: trim anything left from the largest levels.
    for (size_t i = 0; i < levels && over > 0; ++i) {
      const uint64_t take = std::min<uint64_t>((*proposed)[i], over);
      (*proposed)[i] -= static_cast<size_t>(take);
      over -= take;
    }
  } else if (sum < budget) {
    size_t max_index = 0;
    for (size_t i = 1; i < levels; ++i) {
      if ((*proposed)[i] > (*proposed)[max_index]) {
        max_index = i;
      }
    }
    (*proposed)[max_index] += static_cast<size_t>(budget - sum);
  }
}

Status MultiLevelCacheAllocator::RunOnceLocked() {
  last_round_applied_ = false;
  static const bool kAllocDebug = getenv("MLC_ALLOC_DEBUG") != nullptr;
  static std::atomic<uint64_t> dbg_applied{0};
  static std::atomic<uint64_t> dbg_skipped{0};
  if (cache_ == nullptr) {
    return Status::InvalidArgument("cache cannot be null");
  }
  if (!provider_ && options_.mode == MultiLevelAllocatorMode::kModel) {
    return Status::InvalidArgument("metrics provider cannot be empty");
  }

  const auto snapshot = cache_->GetLevelMetricsSnapshot();
  const size_t level_count = snapshot.capacities.size();
  if (level_count == 0) {
    return Status::OK();
  }
  const size_t total_capacity = cache_->GetCapacity();
  const bool has_prev =
      prev_data_sizes_.size() == level_count &&
      prev_lookups_.size() == level_count &&
      prev_hits_.size() == level_count;

  std::vector<double> lambda;
  std::vector<double> data;
  std::vector<double> alpha;
  uint64_t l0_file_count = 0;
  std::vector<size_t> target_capacities;
  if (options_.mode == MultiLevelAllocatorMode::kBaselineEmulation) {
    target_capacities.assign(level_count, 0);
    target_capacities[0] = total_capacity;
  } else {
    if (!provider_(&lambda, &data, &alpha, &l0_file_count)) {
      prev_data_sizes_ = snapshot.data_sizes;
      prev_lookups_ = snapshot.lookups;
      prev_hits_ = snapshot.hits;
      return Status::OK();
    }

    std::vector<double> upper_bounds;
    if (options_.cap_at_data_size &&
        snapshot.data_sizes.size() == level_count) {
      upper_bounds.resize(level_count);
      for (size_t i = 0; i < level_count; ++i) {
        const double ds = static_cast<double>(snapshot.data_sizes[i]);
        upper_bounds[i] =
            ds > 0.0 ? ds * options_.data_cap_margin_ratio
                     : static_cast<double>(options_.empty_level_cap_bytes);
      }
    }
    Status solve_status =
        SolveCapacities(lambda, data, alpha, total_capacity, &target_capacities,
                        options_.solver_epsilon, options_.solver_max_iterations,
                        upper_bounds);
    if (!solve_status.ok()) {
      return solve_status;
    }
  }

  // Data-share floor + persistence gate. Compute each active level's floor and
  // track how many consecutive rounds the solver has proposed it below that
  // floor. The floor relief only fires for levels starved for >=
  // min_starvation_relief_rounds consecutive rounds: this distinguishes a
  // *transient* dip (benign read-only early convergence, which recovers on its
  // own within a few rounds and must NOT be perturbed) from *persistent*
  // starvation (the write-heavy doom loop, where the model-stability gate
  // suppresses the corrective swing every round and the level never recovers).
  // Relief is applied on BOTH the gate-skip path (minimal relief) and the apply
  // path (keeps the level at its floor every round so deep-level compaction
  // reads do not 100%-miss -> no L0 backlog -> no write stall).
  std::vector<unsigned char> relief_mask(level_count, 0);
  const bool floor_configured =
      options_.mode == MultiLevelAllocatorMode::kModel &&
      options_.min_active_level_capacity_ratio > 0.0 &&
      snapshot.data_sizes.size() == level_count && total_capacity > 0;
  // Compaction-pressure gate: only fire the floor relief when L0 is backing up
  // (the doom-loop signature). On a healthy read-only workload L0 stays at ~1
  // file, so relief never fires and the adaptive convergence is not perturbed.
  const bool l0_under_pressure =
      options_.floor_relief_l0_file_threshold == 0 ||
      l0_file_count >= options_.floor_relief_l0_file_threshold;
  const bool floor_enabled = floor_configured && l0_under_pressure;
  if (floor_configured) {
    if (starvation_rounds_.size() < level_count) {
      starvation_rounds_.assign(level_count, 0);
    }
    uint64_t sum_active_data = 0;
    for (size_t i = 0; i < level_count; ++i) {
      if (snapshot.data_sizes[i] > 0) sum_active_data += snapshot.data_sizes[i];
    }
    const double pool = static_cast<double>(total_capacity) *
                        options_.min_active_level_capacity_ratio;
    for (size_t i = 0; i < level_count; ++i) {
      if (snapshot.data_sizes[i] == 0 || sum_active_data == 0) {
        starvation_rounds_[i] = 0;
        continue;
      }
      const double share = static_cast<double>(snapshot.data_sizes[i]) /
                           static_cast<double>(sum_active_data);
      size_t fl = static_cast<size_t>(std::floor(pool * share));
      if (fl < options_.min_active_level_floor_bytes) {
        fl = options_.min_active_level_floor_bytes;
      }
      // Always track consecutive-below-floor rounds so the count resets cleanly
      // when a level recovers, even on rounds the L0 gate suppresses relief.
      if (target_capacities[i] < fl) {
        ++starvation_rounds_[i];
      } else {
        starvation_rounds_[i] = 0;
      }
      // Relief only fires when both the persistence count and the L0-pressure
      // gate are satisfied.
      if (floor_enabled &&
          static_cast<uint64_t>(starvation_rounds_[i]) >=
              options_.min_starvation_relief_rounds) {
        relief_mask[i] = 1;
      }
    }
  }

  // Model-stability gate. If the raw solved target swings too far from the
  // previous round's target, the model signal is untrustworthy (write-heavy or
  // low cache-to-data-ratio noise makes the water-filling solver emit wildly
  // different allocations each round). Acting on it churns
  // SetCapacity/PurgeToCapacity on the hot sub-caches every round with no
  // hit-ratio benefit and a large, compounding throughput loss. Skip the round
  // instead (hold the current allocation; the adaptive interval backs off). A
  // stable workload yields consecutive targets that agree, so it still adapts.
  // In incremental-step mode the bounded step size IS the stability mechanism;
  // the gate would fire on every high-swing round and prevent convergence.
  if (options_.mode == MultiLevelAllocatorMode::kModel &&
      options_.adjust_step_bytes == 0 &&   // legacy mode only
      options_.model_stability_threshold > 0.0 && total_capacity > 0 &&
      prev_target_capacities_.size() == target_capacities.size()) {
    uint64_t target_swing = 0;
    for (size_t i = 0; i < target_capacities.size(); ++i) {
      const size_t a = target_capacities[i];
      const size_t b = prev_target_capacities_[i];
      target_swing += static_cast<uint64_t>(a > b ? a - b : b - a);
    }
    const double swing_ratio =
        static_cast<double>(target_swing) / static_cast<double>(total_capacity);
    if (swing_ratio > options_.model_stability_threshold) {
      if (kAllocDebug) {
        const uint64_t s = dbg_skipped.fetch_add(1) + 1;
        const uint64_t ap = dbg_applied.load();
        if ((s + ap) % 50 == 0 || ap <= 5) {
          fprintf(stderr,
                  "[MLC_ALLOC] round=%llu SKIP(unstable) swing=%.2f "
                  "applied=%llu skipped=%llu\n",
                  (unsigned long long)round_id_, swing_ratio,
                  (unsigned long long)ap, (unsigned long long)s);
        }
      }
      // Even though the discretionary model decision is untrustworthy this
      // round, the data-share floor is a mandatory safety constraint for
      // PERSISTENTLY starved levels (prevents deep-level starvation ->
      // compaction stall -> write stall). Apply just the floor compliance to
      // the live allocation: lift any level that has been below its floor for
      // >= min_starvation_relief_rounds up to its floor, draining donors above
      // their floor. Transient dips (mask=0) are left alone so benign read-only
      // convergence is not perturbed. This is the minimal relief that breaks
      // the doom loop without acting on the noisy model signal.
      if (floor_enabled && !last_capacities_.empty() &&
          last_capacities_.size() == level_count) {
        std::vector<size_t> floor_compliant;
        EnforceDataShareFloor(last_capacities_, snapshot.data_sizes,
                              total_capacity,
                              options_.min_active_level_capacity_ratio,
                              options_.min_active_level_floor_bytes,
                              &floor_compliant, &relief_mask);
        if (floor_compliant != last_capacities_) {
          Status floor_status = cache_->AdjustCapacities(floor_compliant);
          if (floor_status.ok()) {
            last_capacities_ = std::move(floor_compliant);
            last_round_applied_ = true;
          } else {
            floor_status.PermitUncheckedError();
          }
        }
      }
      prev_target_capacities_ = target_capacities;
      prev_data_sizes_ = snapshot.data_sizes;
      prev_lookups_ = snapshot.lookups;
      prev_hits_ = snapshot.hits;
      ++round_id_;
      return Status::OK();
    }
  }
  prev_target_capacities_ = target_capacities;

  // -------------------------------------------------------------------------
  // Incremental marginal-step mode (adjust_step_bytes > 0, kModel only).
  //
  // Problem with the legacy apply-full-target path: the water-filling solver
  // emits correct directions but can swing the entire budget across levels in
  // one round. Each swing calls PurgeToCapacity under shard locks; eviction
  // cost grows with the resident set (~175ms/round at 8 GB full cache),
  // stalling the hot path and collapsing low-concurrency throughput.
  //
  // Solution: bounded single-pair transfer per round, guided by a per-byte
  // marginal score:
  //   Recipient = argmax(score) — highest marginal benefit per byte.
  //   Donor     = argmin(score) — lowest marginal benefit; can give bytes away.
  //
  // Primary scorer (use_ghost_marginal): per-level repeat-miss (ghost)
  // counts, a direct model-free measurement of capacity-convertible miss
  // traffic, normalized per byte (see ghost_normalize_by_data /
  // ghost_normalize_by_uncached in the header for the denominator debate).
  //
  // Fallback scorer (ghost vector unavailable): the exponential-MRC KKT
  // stationarity derivative
  //   score_i = λ_i · (α_i / D_i) · exp(-α_i · c_i / D_i)
  // which requires lambda = raw fg_lookups (use_reuse_lambda=false); the
  // reuse-lambda path drives lambda_L6 → ε and wrongly makes the largest
  // level a permanent donor.
  //
  // Properties:
  //   - Per-round purge is O(step), not O(GB swing) → ≤ ms.
  //   - Steady-state churn is bounded by the gates + hysteresis below;
  //     acceleration is a cold-start-only device (accel_cold_start_applies).
  //   - Floors / caps enforced by donor_avail / recv_room constraints.
  //   - Cold start: equal split so every level gets signal immediately.
  // -------------------------------------------------------------------------
  if (options_.mode == MultiLevelAllocatorMode::kModel &&
      options_.adjust_step_bytes > 0) {
    // Cold start: equal split.
    if (last_capacities_.empty() || last_capacities_.size() != level_count) {
      std::vector<size_t> init;
      EqualSplit(total_capacity, level_count, &init);
      Status s = cache_->AdjustCapacities(init);
      if (s.ok()) {
        last_capacities_ = std::move(init);
        last_round_applied_ = true;
      }
      prev_data_sizes_ = snapshot.data_sizes;
      prev_lookups_ = snapshot.lookups;
      prev_hits_ = snapshot.hits;
      ++round_id_;
      return s;
    }

    // --- Data-size smoothing ---
    // Every data-keyed mechanism below (floors, upper_bytes, structural
    // reclaim, score normalization, fully-cached score-0) reads the smoothed
    // series, so a flush burst pulsing L0's data through the thresholds
    // cannot cycle them (see data_ema_beta in the header). The raw series is
    // kept for the upper-bound computation below, which wants the OPPOSITE
    // guard (min instead of smooth): see sustained-data comment there.
    std::vector<double> raw_data = data;
    if (data.size() == level_count) {
      const double dbeta =
          (options_.data_ema_beta > 0.0 && options_.data_ema_beta < 1.0)
              ? options_.data_ema_beta
              : 1.0;
      if (data_ema_.size() != level_count) {
        data_ema_ = data;
      }
      for (size_t i = 0; i < level_count; ++i) {
        data_ema_[i] = (1.0 - dbeta) * data_ema_[i] + dbeta * data[i];
        data[i] = data_ema_[i];
      }
    }

    // --- Floors (data-share weighted + absolute minimum) ---
    //
    // Lower bound companion to upper_bytes below: every active level keeps a
    // minimum allocation (5% pool split by data share) so compaction and
    // point lookups make forward progress even when its marginal score never
    // wins a transfer. Empirically these floors ARE the steady-state
    // capacity of the deep levels (L5/L6 sit exactly at their floor in most
    // wlA runs), so their sizing directly shows up in the hit ratio.
    // Floor eligibility requires >= 1 MiB of (smoothed) data. The EMA decays
    // geometrically after a level empties, so without a threshold a level
    // that held transient compaction output (L1/L2) keeps a residual EMA of
    // a few bytes forever and parks its absolute floor (observed: empty
    // L1/L2 pinning 14 MiB each). 1 MiB is small enough that L0's EMA never
    // falls through it between flushes (which would re-introduce the pulse
    // cycle the smoothing exists to prevent).
    constexpr double kFloorActiveDataMin = static_cast<double>(1 << 20);
    std::vector<size_t> floor_bytes(level_count, 0);
    if (options_.min_active_level_capacity_ratio > 0.0 &&
        data.size() == level_count && total_capacity > 0) {
      double sum_active = 0.0;
      for (size_t i = 0; i < level_count; ++i) {
        if (data[i] >= kFloorActiveDataMin) sum_active += data[i];
      }
      if (sum_active > 0.0) {
        const double pool = static_cast<double>(total_capacity) *
                            options_.min_active_level_capacity_ratio;
        for (size_t i = 0; i < level_count; ++i) {
          if (data[i] < kFloorActiveDataMin) continue;
          const double share = data[i] / sum_active;
          floor_bytes[i] = std::max(
              options_.min_active_level_floor_bytes,
              static_cast<size_t>(std::floor(pool * share)));
        }
      }
    }
    if (options_.min_active_level_capacity_bytes > 0 &&
        data.size() == level_count) {
      for (size_t i = 0; i < level_count; ++i) {
        if (data[i] >= kFloorActiveDataMin) {
          floor_bytes[i] =
              std::max(floor_bytes[i],
                       options_.min_active_level_capacity_bytes);
        }
      }
    }
    // upper_bytes: cap each level at data_size * data_cap_margin_ratio to
    // prevent over-provisioned levels (capacity >> data) from being recipients.
    // A level that has already cached all of its data cannot benefit from more
    // capacity, and its alpha degenerates to 0 at near-100% hit rate, which
    // would otherwise trigger the kInfiniteScore sentinel and make it a
    // permanent recipient, starving larger levels like L6.  The margin ratio
    // (>1) gives headroom for block/cache accounting overhead so a fully-hot
    // level can still cache all of its blocks.  Same semantics as the legacy
    // water-filling upper bound above.
    //
    // upper_bytes serves two mechanisms with opposite transient needs, so
    // two bounds are computed:
    //
    //   - upper_bytes (EMA data): the structural-reclaim threshold and the
    //     donor-side guard. Smoothed so an L0 data trough between flushes
    //     does not read as excess and trigger a spurious drain (the pulse
    //     cycle data_ema_beta exists to prevent).
    //
    //   - recv_upper_bytes (min(raw, EMA) = SUSTAINED data): the recipient
    //     growth cap. A compaction passing through a normally-empty level
    //     (L1/L2) parks real data there for a handful of rounds; its
    //     short-distance repeat misses during that window are genuine, so
    //     the scorer correctly ranks it as a recipient, and under an
    //     EMA-only cap it wins real capacity that structural reclaim then
    //     claws back after the data moves on (observed: L1 briefly holding
    //     370 MiB of a 2 GiB budget -- a purge->refill round trip of
    //     warm-set loss both ways and a repeat-variance source). min(raw,
    //     EMA) caps a transient spike at the pre-spike EMA (raw high, EMA
    //     low) while a steady level (raw ~= EMA) is unaffected, so only
    //     data that PERSISTS for ~1/beta rounds can attract matching
    //     capacity. Levels with raw == 0 this round cannot grow at all.
    std::vector<size_t> upper_bytes(level_count, total_capacity);
    std::vector<size_t> recv_upper_bytes(level_count, total_capacity);
    if (options_.cap_at_data_size) {
      for (size_t i = 0; i < level_count; ++i) {
        if (data[i] > 0.0) {
          const double cap = data[i] * options_.data_cap_margin_ratio;
          if (cap < static_cast<double>(total_capacity)) {
            upper_bytes[i] = static_cast<size_t>(cap);
          }
        }
        const double sustained =
            (raw_data.size() == level_count && raw_data[i] < data[i])
                ? raw_data[i]
                : data[i];
        const double rcap = sustained * options_.data_cap_margin_ratio;
        if (rcap < static_cast<double>(total_capacity)) {
          recv_upper_bytes[i] = static_cast<size_t>(rcap);
        }
        // Growth must also respect the reclaim threshold, or a transfer
        // could immediately arm structural reclaim on its own recipient.
        recv_upper_bytes[i] = std::min(recv_upper_bytes[i], upper_bytes[i]);
      }
    }

    // --- Marginal scores: score[i] = lambda[i] * alpha[i] / data[i]  (= a_i) ---
    //
    // This is the water-filling marginal value used by the solver: the first
    // derivative of total miss count with respect to capacity at the current
    // allocation is  λ_i · (α_i / D_i) · miss_rate_i(c_i).  At the optimal
    // allocation all active levels equalize  a_i · miss_rate_i* = μ, so a_i
    // is the correct ordering signal for the greedy step: high a_i means the
    // level is under-provisioned relative to the optimum and should receive
    // capacity; low a_i means it is over-provisioned and can donate.
    //
    // Prerequisite: lambda must be raw access frequency (use_reuse_lambda=false).
    // If lambda_L6 → ε via the reuse-lambda path, a_L6 → 0 and L6 wrongly
    // becomes a permanent donor despite holding 99% of the data.  With raw
    // fg_lookups, lambda_L6 >> lambda_L5, correctly making a_L6 competitive.
    //
    // For levels with data > 0 but alpha == 0 AND capacity < data size
    // (genuinely starved: all accesses miss, no reliable alpha estimate yet),
    // use a high sentinel so they are always preferred as recipient.
    //
    // Do NOT use kInfiniteScore for over-provisioned levels (capacity >=
    // data_size): their alpha degenerates to 0 because miss_rate → 0, not
    // because they lack cache.  Giving them kInfiniteScore would make them
    // permanent recipients and drain large levels like L6.
    const double kInfiniteScore = std::numeric_limits<double>::infinity();
    std::vector<double> score(level_count, 0.0);
    bool ghost_scored = false;
    bool capture_scored = false;

    // Window fg stats: per-round deltas of the cumulative fg counters,
    // EMA-smoothed into (a) per-byte hit density -- the donor retention
    // cost, same hits/byte/window units as the capture score -- and
    // (b) window hit rate, which drives reuse-distance decompression in
    // the capture scorer. Computed before scoring so the current round's
    // scores see the current window.
    const bool window_stats_available =
        snapshot.fg_hits.size() == level_count &&
        snapshot.fg_lookups.size() == level_count;
    if (window_stats_available) {
      if (prev_fg_hits_.size() != level_count) {
        prev_fg_hits_ = snapshot.fg_hits;
        prev_fg_lookups_ = snapshot.fg_lookups;
      }
      if (hit_density_ema_.size() != level_count) {
        hit_density_ema_.assign(level_count, 0.0);
        hit_rate_ema_.assign(level_count, 0.0);
      }
      const double beta = (options_.ghost_score_ema_beta > 0.0 &&
                           options_.ghost_score_ema_beta < 1.0)
                              ? options_.ghost_score_ema_beta
                              : 1.0;
      for (size_t i = 0; i < level_count; ++i) {
        const uint64_t dh = snapshot.fg_hits[i] >= prev_fg_hits_[i]
                                ? snapshot.fg_hits[i] - prev_fg_hits_[i]
                                : 0;
        const uint64_t dl = snapshot.fg_lookups[i] >= prev_fg_lookups_[i]
                                ? snapshot.fg_lookups[i] - prev_fg_lookups_[i]
                                : 0;
        const double cap_bytes = static_cast<double>(
            std::max<size_t>(last_capacities_[i], size_t{1} << 20));
        hit_density_ema_[i] = (1.0 - beta) * hit_density_ema_[i] +
                              beta * static_cast<double>(dh) / cap_bytes;
        // Levels with no window traffic keep their previous rate estimate
        // (a 0/0 window says nothing about the hit curve).
        if (dl > 0) {
          hit_rate_ema_[i] =
              (1.0 - beta) * hit_rate_ema_[i] +
              beta * static_cast<double>(dh) / static_cast<double>(dl);
        }
      }
      prev_fg_hits_ = snapshot.fg_hits;
      prev_fg_lookups_ = snapshot.fg_lookups;
    }

    if (options_.use_ghost_marginal && options_.use_ghost_capture_rate) {
      // --- Capture-rate score: measured per-byte marginal utility ---
      // A repeat miss at reuse distance d (distinct missed blocks between
      // the two misses) becomes a hit once the level holds ~d more blocks,
      // so its value per byte of added capacity is 1/(d * block_bytes).
      // Summing over the drained distance histogram gives
      //   score_i = sum_b hist_i[b] / (mid_b * block_bytes)
      // -- the same quantity the static denominators (raw count / D /
      // uncached) each tried to guess with a fixed prior about within-level
      // reuse concentration, now measured directly. A level whose repeats
      // cluster at short distances (zipfian hot tail, e.g. L5's) scores by
      // that concentration instead of being flat-taxed by its 15 GiB
      // footprint; a level whose repeats sit at unreachable distances
      // scores ~0 instead of masquerading as capacity-hungry. The score is
      // also allocation-path independent (no feedback through c_i), which
      // is what makes different runs converge to the same attractor.
      const std::vector<uint64_t> hist =
          cache_->DrainGhostDistanceHistogram();
      constexpr size_t kB = MultiLevelCache::kGhostDistBuckets;
      if (hist.size() == level_count * kB) {
        const double block_bytes =
            std::max<double>(1.0, static_cast<double>(
                                      options_.ghost_dist_block_bytes));
        const double beta =
            (options_.ghost_score_ema_beta > 0.0 &&
             options_.ghost_score_ema_beta < 1.0)
                ? options_.ghost_score_ema_beta
                : 1.0;
        if (ghost_score_ema_.size() != level_count) {
          ghost_score_ema_.assign(level_count, 0.0);
        }
        for (size_t i = 0; i < level_count; ++i) {
          double val = 0.0;
          // Short-distance repeats on a level holding real capacity are
          // concurrent duplicate misses (threads re-missing a block before
          // the first miss's fill lands), which no amount of capacity
          // converts; only a near-defunded level's short-distance repeats
          // are genuine starvation signal. See ghost_inflight_dist_bytes.
          const bool skip_inflight =
              options_.ghost_inflight_dist_bytes > 0 &&
              last_capacities_[i] >= options_.ghost_inflight_min_cap_bytes;
          // Distances are measured in distinct MISSED blocks; the capacity
          // needed to capture a repeat scales with distinct ACCESSED
          // blocks, larger by ~1/(1-h). Without this, high-hit levels'
          // scores are inflated by the same factor (up to 4x at h=0.75).
          // See ghost_dist_decompress_max.
          double decompress = 1.0;
          if (options_.ghost_dist_decompress_max > 1.0 &&
              window_stats_available) {
            decompress = std::min(
                options_.ghost_dist_decompress_max,
                1.0 / std::max(0.05, 1.0 - hit_rate_ema_[i]));
          }
          for (size_t b = 0; b < kB; ++b) {
            const uint64_t cnt = hist[i * kB + b];
            if (cnt == 0) {
              continue;
            }
            // Geometric mid-distance of the bucket in blocks. Bucket 0
            // aggregates everything below one sampled-clock tick
            // (2^kGhostClockSampleShift blocks); buckets b >= shift span
            // [2^b, 2^(b+1)).
            const double mid =
                (b == 0)
                    ? static_cast<double>(
                          uint64_t{1}
                          << (MultiLevelCache::kGhostClockSampleShift - 1))
                    : std::sqrt(2.0) *
                          static_cast<double>(uint64_t{1} << b);
            // In-flight filtering uses the RAW measured distance: the
            // overlap window is itself measured in missed blocks, so it
            // must not be decompressed. Only the capacity-value weight is.
            if (skip_inflight &&
                mid * block_bytes <=
                    static_cast<double>(options_.ghost_inflight_dist_bytes)) {
              continue;
            }
            val += static_cast<double>(cnt) / (mid * decompress * block_bytes);
          }
          // Same EMA smoothing as the plain ghost path (per-window
          // histograms carry the same Poisson/compaction-burst noise).
          ghost_score_ema_[i] =
              (1.0 - beta) * ghost_score_ema_[i] + beta * val;
          score[i] = ghost_score_ema_[i];
        }
        ghost_scored = true;
        capture_scored = true;
        if (kAllocDebug) {
          std::string h;
          char tmp[64];
          for (size_t i = 0; i < level_count; ++i) {
            snprintf(tmp, sizeof(tmp), " L%zu:", i);
            h += tmp;
            bool any = false;
            for (size_t b = 0; b < kB; ++b) {
              const uint64_t cnt = hist[i * kB + b];
              if (cnt == 0) {
                continue;
              }
              snprintf(tmp, sizeof(tmp), "%s%zu=%llu", any ? "," : "", b,
                       (unsigned long long)cnt);
              h += tmp;
              any = true;
            }
            if (!any) {
              h += "-";
            }
          }
          fprintf(stderr, "[mlc-alloc] round=%llu GHOSTHIST%s\n",
                  (unsigned long long)round_id_, h.c_str());
        }
      }
    }
    if (options_.use_ghost_marginal && !ghost_scored) {
      // --- Ghost (repeat-miss) marginal score: direct measurement ---
      // score[i] = repeat misses on recently-missed keys this window = the
      // exact traffic a capacity increase would convert into hits. No MRC
      // shape assumption; immune to the alpha single-point-inversion
      // degeneracy (score collapsing to λ·m·(-ln m)/c, prior drag toward
      // α=1, and the fill-delay feedback loop). A genuinely starved level
      // produces high ghost hits naturally (its inserts are evicted before
      // re-access), so no starvation sentinel is needed; a fully-cached
      // level produces ~0 misses hence ~0 ghost hits, so it self-retires
      // as recipient (belt: upper_bytes cap still applies).
      const std::vector<uint64_t> ghost = cache_->DrainGhostHits();
      if (ghost.size() == level_count) {
        // EMA smoothing (steady-state suppression #1): per-window counts are
        // Poisson-noisy and compaction-bursty; smoothing keeps a transient
        // spike from flipping the transfer direction for a round.
        const double beta =
            (options_.ghost_score_ema_beta > 0.0 &&
             options_.ghost_score_ema_beta < 1.0)
                ? options_.ghost_score_ema_beta
                : 1.0;
        if (ghost_score_ema_.size() != level_count) {
          ghost_score_ema_.assign(level_count, 0.0);
        }
        for (size_t i = 0; i < level_count; ++i) {
          ghost_score_ema_[i] = (1.0 - beta) * ghost_score_ema_[i] +
                                beta * static_cast<double>(ghost[i]);
          score[i] = ghost_score_ema_[i];
          // Per-byte normalization: the raw count says how many misses more
          // capacity could convert, but the bytes needed per converted hit
          // scale with the level's footprint. Without this, comparable raw
          // counts on L3 (0.8 GiB data) and L5 (15 GiB data) read as equal
          // marginal value even though L5 needs ~20x the capacity per hit.
          //
          // The denominator is the UNCACHED footprint (data - capacity), not
          // the total data size: ghost hits are produced exclusively by the
          // uncached portion (cached blocks do not miss), so the marginal
          // utility of one more byte is ghost / uncached_bytes. Normalizing
          // by total data instead turns small-data levels into capacity
          // magnets: L0 (0.25-1 GiB of fast-churning flush output) had its
          // score inflated ~150x relative to L6 by the 1/D factor, ended a
          // 2 GiB run holding 1.1 GiB, and even drained L6 -- while its own
          // hit ratio barely responds to capacity. When capacity already
          // covers the data (uncached == 0), the residual ghost hits come
          // from file churn (new post-flush/compaction blocks always miss
          // once); no amount of extra capacity converts those, so the
          // marginal score is exactly 0. For deep levels (cap << data) this
          // is numerically identical to the old 1/D normalization.
          // Denominator: D by default (allocation-invariant, no feedback);
          // optionally the floored uncached footprint (see header comment on
          // ghost_normalize_by_uncached for the rich-get-richer caveat).
          // Fully-cached levels score 0 either way: their residual misses
          // are churn-compulsory and no capacity converts them.
          if (options_.ghost_normalize_by_data && data[i] > 0.0) {
            const double uncached =
                data[i] - static_cast<double>(last_capacities_[i]);
            if (uncached <= 0.0) {
              score[i] = 0.0;
            } else {
              const double denom =
                  options_.ghost_normalize_by_uncached
                      ? std::max(uncached,
                                 options_.ghost_uncached_floor_frac * data[i])
                      : data[i];
              score[i] = ghost_score_ema_[i] / denom;
            }
          }
        }
        ghost_scored = true;
      }
    }
    if (!ghost_scored) {
      // --- Fallback: exponential-model marginal score ---
      for (size_t i = 0; i < level_count; ++i) {
        if (lambda[i] > 0.0 && data[i] > 0.0) {
          if (alpha[i] <= 0.0) {
            // Only raise sentinel when the level is genuinely
            // capacity-starved; over-provisioned levels (alpha degenerated
            // via miss_rate → 0) stay at 0 so they cannot become permanent
            // recipients.
            if (last_capacities_[i] < static_cast<size_t>(data[i])) {
              score[i] = kInfiniteScore;
            }
            continue;
          }
          // Complete marginal benefit: the exact first derivative of total
          // miss count w.r.t. c_i (KKT stationarity):
          //   score_i = λ_i · (α_i / D_i) · exp(-α_i · c_i / D_i)
          //           = λ_i · (α_i / D_i) · miss_rate_i(c_i)
          // Prerequisite: lambda must use raw fg_lookups
          // (use_reuse_lambda=false); with reuse-lambda, lambda_L6 → ε and
          // L6 becomes a permanent donor regardless of provisioning.
          const double cap_d =
              static_cast<double>(last_capacities_[i]) / data[i];
          const double miss_rate = std::exp(-alpha[i] * cap_d);
          score[i] = lambda[i] * (alpha[i] / data[i]) * miss_rate;
        }
      }
    }

    // Donor retention cost (capture-rate mode): the per-byte hit density
    // computed above is the measurable shrink-side marginal the
    // growth-only capture score lacks -- a fully-fed level scores ~0 as a
    // recipient (correct) but its resident bytes may be the densest hit
    // earners in the cache, so it must NOT be the argmin-score donor. See
    // donor_retention_frac.
    const bool retention_available = capture_scored &&
                                     options_.donor_retention_frac > 0.0 &&
                                     window_stats_available;

    // Direction locks (steady-state suppression #2): a recent recipient may
    // not donate and a recent donor may not receive, killing ping-pong
    // transfers (observed: L3->L0 then L0->L5 within 50 rounds) regardless
    // of score noise.
    if (received_lock_round_.size() != level_count) {
      received_lock_round_.assign(level_count, 0);
      donated_lock_round_.assign(level_count, 0);
    }

    // --- Excess detection (structural reclaim) ---
    //
    // upper_bytes only limits a level's growth at transfer time; nothing
    // reclaims capacity once the level's data shrinks below it. L0 is the
    // canonical victim: its data is pulse-shaped (flush files arrive, then
    // compaction consumes them within seconds), and its ghost score stays
    // permanently high from compulsory misses on newly created files --
    // misses no amount of capacity can prevent, which the repeat-miss signal
    // cannot distinguish. Score-driven transfers therefore park a large
    // fraction of the budget on L0 and never take it back (observed: 768 MiB
    // for ~0 resident data at a 2 GiB budget, starving L4/L5 and costing
    // ~3pt fg hit ratio). Capacity beyond data*margin is unusable by
    // definition, so a level holding it becomes a mandatory donor,
    // overriding the score-based donor choice and the steady-state gates.
    size_t excess_donor = level_count;
    size_t excess_bytes = 0;
    for (size_t i = 0; i < level_count; ++i) {
      if (last_capacities_[i] > upper_bytes[i]) {
        const size_t ex = last_capacities_[i] - upper_bytes[i];
        if (ex > excess_bytes) {
          excess_bytes = ex;
          excess_donor = i;
        }
      }
    }
    // Usage-based excess (see usage_reclaim_margin in the header): capacity a
    // level has persistently failed to FILL is dead regardless of its on-disk
    // data size -- with cap > usage there is no eviction, so the slack earns
    // zero hits. The persistence window distinguishes "still filling a step
    // it just received" from "touched footprint reached, slack is dead".
    if (options_.usage_reclaim_rounds > 0 &&
        snapshot.usages.size() == level_count) {
      if (usage_excess_rounds_.size() != level_count) {
        usage_excess_rounds_.assign(level_count, 0);
      }
      for (size_t i = 0; i < level_count; ++i) {
        const double usable_d =
            static_cast<double>(snapshot.usages[i]) *
            std::max(1.0, options_.usage_reclaim_margin);
        size_t usable = usable_d < static_cast<double>(total_capacity)
                            ? static_cast<size_t>(usable_d)
                            : total_capacity;
        usable = std::max({usable, floor_bytes[i],
                           options_.usage_bootstrap_bytes});
        // A level still FILLING its capacity is not holding dead slack: its
        // usage grows every round until the touched footprint is reached.
        // Reset the persistence counter while growth continues, so the
        // effective window scales with the level's own fill rate (a 1%-
        // traffic level fills a 64 MiB step over ~27 rounds; a fixed
        // 12-round window would reclaim the step mid-fill and re-create the
        // grow/reclaim oscillation this mechanism is meant to end).
        const uint64_t prev_u =
            prev_usages_.size() == level_count ? prev_usages_[i] : 0;
        const bool still_filling =
            snapshot.usages[i] >
            prev_u + std::max<uint64_t>(
                         1 << 20, (last_capacities_[i] > snapshot.usages[i]
                                       ? last_capacities_[i] -
                                             snapshot.usages[i]
                                       : 0) /
                                      50);
        if (last_capacities_[i] > usable + (size_t{1} << 20) &&
            !still_filling) {
          ++usage_excess_rounds_[i];
          if (usage_excess_rounds_[i] >= options_.usage_reclaim_rounds) {
            const size_t ex = last_capacities_[i] - usable;
            if (ex > excess_bytes) {
              excess_bytes = ex;
              excess_donor = i;
            }
          }
        } else {
          usage_excess_rounds_[i] = 0;
        }
      }
      prev_usages_ = snapshot.usages;
    }
    // Ignore sub-MiB slack (accounting noise).
    const bool structural_reclaim =
        excess_donor != level_count && excess_bytes >= (size_t{1} << 20);

    // Probe transfer (anti-freeze annealing): after enough consecutive
    // gate-skipped rounds, run one small score-directed transfer that
    // bypasses the near-optimal and significance gates (but not the
    // no-signal check, direction locks, floors, or upper bounds). This
    // keeps the equilibrium annealing toward the optimum instead of
    // freezing wherever the convergence phase overshot.
    const bool probe_transfer =
        !structural_reclaim && options_.probe_after_skipped_rounds > 0 &&
        consecutive_gate_skips_ >= options_.probe_after_skipped_rounds;

    if (kAllocDebug) {
      // Full per-level decision state: score, capacity, growth ceiling,
      // floor, and any active direction locks. This is the line to read when
      // a level with a dominant score is mysteriously never selected.
      std::string st;
      char tmp[160];
      for (size_t i = 0; i < level_count; ++i) {
        snprintf(tmp, sizeof(tmp),
                 " L%zu[s=%.3g hd=%.3g c=%zu u=%zu ru=%zu fl=%zu d=%.0f "
                 "dl=%lld rl=%lld ux=%u]",
                 i, score[i],
                 hit_density_ema_.size() == level_count ? hit_density_ema_[i]
                                                        : 0.0,
                 last_capacities_[i] >> 20,
                 (snapshot.usages.size() == level_count ? snapshot.usages[i]
                                                        : 0) >>
                     20,
                 recv_upper_bytes[i] >> 20, floor_bytes[i] >> 20,
                 data[i] / (1 << 20),
                 donated_lock_round_[i] > round_id_
                     ? (long long)(donated_lock_round_[i] - round_id_)
                     : 0LL,
                 received_lock_round_[i] > round_id_
                     ? (long long)(received_lock_round_[i] - round_id_)
                     : 0LL,
                 usage_excess_rounds_.size() == level_count
                     ? usage_excess_rounds_[i]
                     : 0);
        st += tmp;
      }
      fprintf(stderr, "[mlc-alloc] round=%llu STATE%s\n",
              (unsigned long long)round_id_, st.c_str());
    }

    // Usage growth gate (see usage_grow_headroom in the header): a level
    // that has not filled the capacity it already holds cannot use more, so
    // it is not a recipient until its usage catches up. Growth thereby
    // tracks demonstrated demand, and lazily-growing levels lose nothing
    // (their misses are admitted either way while cap > usage). The
    // bootstrap exemption lets a fully-defunded level re-enter.
    auto usage_gated = [&](size_t i) {
      if (options_.usage_grow_headroom <= 0.0 ||
          snapshot.usages.size() != level_count) {
        return false;
      }
      if (last_capacities_[i] <= options_.usage_bootstrap_bytes) {
        return false;
      }
      return static_cast<double>(last_capacities_[i]) >
             static_cast<double>(snapshot.usages[i]) *
                 options_.usage_grow_headroom;
    };

    // --- Find recipient (highest score, has data, has room to grow) ---
    size_t recipient = level_count;
    double best_recv = -1.0;
    for (size_t i = 0; i < level_count; ++i) {
      if (structural_reclaim && i == excess_donor) continue;
      if (donated_lock_round_[i] > round_id_) continue;  // recent donor
      if (usage_gated(i)) continue;  // has not filled what it holds
      if (data[i] > 0.0 && last_capacities_[i] < recv_upper_bytes[i] &&
          score[i] > best_recv) {
        best_recv = score[i];
        recipient = i;
      }
    }
    if (structural_reclaim && recipient == level_count) {
      // No score-bearing recipient (e.g. zero-signal window): fall back to
      // the largest-data level with room, so the reclaim still proceeds.
      double best_data = 0.0;
      for (size_t i = 0; i < level_count; ++i) {
        if (i == excess_donor) continue;
        if (usage_gated(i)) continue;
        if (data[i] > best_data && last_capacities_[i] < recv_upper_bytes[i]) {
          best_data = data[i];
          recipient = i;
          best_recv = score[i];
        }
      }
    }

    // --- Find donor (lowest score, capacity above floor, not recipient) ---
    //
    // Two passes. Pass 1 considers only levels holding UNUSED capacity
    // (cap > usage): donating slack evicts nothing, so it is always the
    // cheapest source of bytes, and taking it first keeps a slowly-filling
    // level's capacity tracking its usage instead of digging into its
    // resident set (observed on wlC: score-driven donations drained a
    // filling L3 through its warm set to 0, its starved score then exploded
    // and the resulting refill/reversal cycle escalated the pair locks to
    // ~90 rounds, freezing the allocator with L3 dead at 0 capacity). Pass 2
    // (no slack anywhere) is the normal regime once the budget is fully
    // utilized: the argmin-score FULL level donates and evicts its coldest
    // blocks -- the intended marginal trade.
    size_t donor = level_count;
    double best_donor_score = kInfiniteScore;
    bool donor_found = false;
    bool donor_slack_only = false;
    if (structural_reclaim) {
      donor = excess_donor;
      best_donor_score = score[donor];
      donor_found = true;
    } else {
      const bool have_usage = snapshot.usages.size() == level_count;
      if (have_usage) {
        for (size_t i = 0; i < level_count; ++i) {
          if (i == recipient) continue;
          if (received_lock_round_[i] > round_id_) continue;
          const size_t protected_bytes =
              std::max(floor_bytes[i], static_cast<size_t>(snapshot.usages[i]));
          if (last_capacities_[i] <= protected_bytes + (size_t{1} << 20)) {
            continue;  // no meaningful slack
          }
          if (!donor_found || score[i] < best_donor_score) {
            best_donor_score = score[i];
            donor = i;
            donor_found = true;
            donor_slack_only = true;
          }
        }
      }
      if (!donor_found) {
        // Full-level donors evict resident data, so rank them by what
        // those bytes EARN (retention density), not by their capture
        // score: the capture score measures growth value only, and a
        // fully-fed level's ~0 score would otherwise mark the densest
        // hit earner in the cache as the cheapest donor (observed on wlD:
        // fully-cached L3 at 0.95 hit drained to zero this way).
        double best_donor_cost = 0.0;
        for (size_t i = 0; i < level_count; ++i) {
          if (i == recipient) continue;
          if (received_lock_round_[i] > round_id_) continue;  // recent recipient
          if (last_capacities_[i] <= floor_bytes[i]) continue;
          const double cost =
              retention_available ? hit_density_ema_[i] : score[i];
          if (!donor_found || cost < best_donor_cost) {
            best_donor_cost = cost;
            best_donor_score = score[i];
            donor = i;
            donor_found = true;
          }
        }
      }
    }

    auto do_skip_step = [&](const char* reason, bool gate_skip = false) {
      // Only skips caused by the steady-state gates advance the probe
      // counter; structural skips (no traffic, no pair, nothing to move)
      // are not situations a probe transfer could improve.
      if (gate_skip) {
        ++consecutive_gate_skips_;
      }
      if (kAllocDebug) {
        const uint64_t s = dbg_skipped.fetch_add(1) + 1;
        const uint64_t ap = dbg_applied.load();
        if ((s + ap) % 50 == 0 || ap <= 5) {
          fprintf(stderr,
                  "[MLC_ALLOC] round=%llu SKIP(%s) applied=%llu "
                  "skipped=%llu\n",
                  (unsigned long long)round_id_, reason,
                  (unsigned long long)ap, (unsigned long long)s);
        }
      }
      // Any skipped round breaks the same-direction streak: drop the
      // accelerated step back to the base step.
      current_step_bytes_ = 0;
      last_step_recipient_ = SIZE_MAX;
      prev_data_sizes_ = snapshot.data_sizes;
      prev_lookups_ = snapshot.lookups;
      prev_hits_ = snapshot.hits;
      ++round_id_;
    };

    if (recipient == level_count || !donor_found) {
      do_skip_step("no_pair");
      return Status::OK();
    }

    // The steady-state gates below compare marginal scores; a structural
    // reclaim is not a score-driven decision (the excess is unusable
    // regardless of any score), so it bypasses all of them.
    //
    // Require a real traffic signal before transferring: if best score is
    // zero (lambda=0 for all levels in the current window, e.g. the first
    // few rounds during wait_for_compact), defer until we have data.
    if (!structural_reclaim && best_recv <= 0.0) {
      do_skip_step("no_signal");
      return Status::OK();
    }

    // Gap check: skip if allocation is near-optimal (scores converged).
    if (!structural_reclaim && !probe_transfer &&
        options_.step_min_score_ratio > 0.0 && std::isfinite(best_recv)) {
      const double gap = (best_recv - best_donor_score) / best_recv;
      if (gap < options_.step_min_score_ratio) {
        do_skip_step("near_optimal", /*gate_skip=*/true);
        return Status::OK();
      }
    }

    // Significance gate for count-based ghost scores (steady-state
    // suppression #3): require both a multiplicative gap and a
    // Poisson-significance gap, so steady-state noise (all levels thrashing
    // comparably) stops transfers while convergence-phase gaps (10-100x)
    // pass trivially. Poisson noise lives on the raw counts; when the
    // selection scores are per-byte normalized (count/D), the noise must be
    // propagated through the same normalization -- σ(count/D) = sqrt(count)/D
    // -- rather than compared on raw counts, otherwise a legitimate transfer
    // between levels with similar counts but very different footprints
    // (e.g. L3 at 0.8 GiB vs L5 at 15 GiB) would read as insignificant.
    if (!structural_reclaim && !probe_transfer && ghost_scored &&
        options_.ghost_min_recv_donor_ratio > 0.0) {
      const bool ratio_ok =
          best_recv >
          options_.ghost_min_recv_donor_ratio * std::max(0.0, best_donor_score);
      // Scale factor applied to each level's count in score[]: 1/uncached
      // (or 1 when unnormalized), mirroring the normalization above.
      // Capture-rate scores are not simple scaled counts (each repeat
      // contributes 1/(d·block)), so the Poisson variance model below does
      // not apply; treat them as unscaled (the gate is off by default and
      // this path exists for ablation of the plain ghost scorer).
      auto score_scale = [&](size_t lvl) {
        if (capture_scored || !options_.ghost_normalize_by_data ||
            data[lvl] <= 0.0) {
          return 1.0;
        }
        const double uncached =
            data[lvl] - static_cast<double>(last_capacities_[lvl]);
        // A fully-cached level's score is pinned to 0 with no noise term.
        if (uncached <= 0.0) {
          return 0.0;
        }
        if (options_.ghost_normalize_by_uncached) {
          return 1.0 / std::max(uncached,
                                options_.ghost_uncached_floor_frac * data[lvl]);
        }
        return 1.0 / data[lvl];
      };
      const double recv_scale = score_scale(recipient);
      const double donor_scale = score_scale(donor);
      const double var =
          std::max(1.0, ghost_score_ema_[recipient]) * recv_scale * recv_scale +
          std::max(1.0, ghost_score_ema_[donor]) * donor_scale * donor_scale;
      const bool significant =
          (best_recv - best_donor_score) >
          options_.ghost_significance_k * std::sqrt(var);
      if (!ratio_ok || !significant) {
        do_skip_step("insignificant", /*gate_skip=*/true);
        return Status::OK();
      }
    }

    // Donor retention gate: a full-level donor's coldest bytes are evicted
    // by the transfer, so the recipient's measured per-byte gain must beat
    // a discounted fraction of the donor's measured per-byte earnings
    // (mean density discounted to marginal; see donor_retention_frac).
    // Applies to probe transfers too -- a probe exists to unfreeze stuck
    // gates, not to bleed a protected donor -- but not to structural
    // reclaims (dead capacity earns nothing) or slack donors (nothing is
    // evicted).
    if (!structural_reclaim && !donor_slack_only && retention_available &&
        best_recv <=
            options_.donor_retention_frac * hit_density_ema_[donor]) {
      do_skip_step("donor_protected", /*gate_skip=*/false);
      return Status::OK();
    }

    // --- Reversal hysteresis (steady-state suppression #5) ---
    // Detect whether this transfer undoes a recent transfer on the same
    // level pair. A reversal is applied at the base step only (an
    // accelerated reversal purges the warm set the previous transfer just
    // built), and escalates the pair's direction-lock length exponentially
    // with its streak, freezing a flip-flopping pair for progressively
    // longer. One-directional traffic keeps streak 0 and is never slowed.
    if (pair_last_round_.size() != level_count * level_count) {
      pair_last_round_.assign(level_count * level_count, 0);
      pair_last_dir_.assign(level_count * level_count, 0);
      pair_reversal_streak_.assign(level_count * level_count, 0);
    }
    const size_t pair_idx = std::min(donor, recipient) * level_count +
                            std::max(donor, recipient);
    const int transfer_dir = donor < recipient ? 1 : -1;
    bool is_reversal = false;
    uint32_t reversal_streak = 0;
    if (options_.reversal_window_rounds > 0 && pair_last_dir_[pair_idx] != 0 &&
        round_id_ - pair_last_round_[pair_idx] <=
            options_.reversal_window_rounds) {
      reversal_streak = pair_reversal_streak_[pair_idx];
      if (pair_last_dir_[pair_idx] != transfer_dir) {
        is_reversal = true;
        reversal_streak = std::min<uint32_t>(reversal_streak + 1, 16);
      }
      // Same direction within the window: streak carries over unchanged, so
      // the long locks stay armed against renewed flip-flopping.
    }

    // --- Compute and apply transfer (adaptive step) ---
    //
    // Rprop-style acceleration: consecutive applied rounds with the SAME
    // recipient mean the allocation is still far from the optimum and pushing
    // in one direction, so the step doubles (up to the effective max) to cut
    // the cold-start convergence tax. The moment the recipient changes -- or
    // any round is skipped -- the step resets to the effective base, so near
    // the optimum (where recipients alternate) transfers stay small.
    //
    // Both the base and max step are capped RELATIVE to the total budget
    // (total/32 and total/8). The absolute options (64 MiB / 512 MiB) were
    // tuned at 8 GiB; at a 1 GiB budget an uncapped 512 MiB step moves half
    // the cache in one round, and the resulting capacity swing leaves the
    // grow-only AutoHCC tables sparse relative to usage, degrading every
    // subsequent Evict sweep (the profiled 55%-of-cycles Evict hotspot).
    const size_t base_step = std::max<size_t>(
        1 << 20, std::min(options_.adjust_step_bytes, total_capacity / 32));
    const size_t max_step = std::max(
        base_step, std::min(options_.step_max_bytes, total_capacity / 8));
    // Acceleration is a cold-start device only: past the apply budget, or
    // after the first observed reversal (definitive overshoot evidence),
    // every transfer moves at the base step. See accel_cold_start_applies.
    if (is_reversal) {
      accel_disabled_ = true;
    }
    const bool accel_allowed =
        !accel_disabled_ &&
        applied_transfer_count_ < options_.accel_cold_start_applies;
    size_t step = base_step;
    if (options_.step_growth > 1.0 && accel_allowed &&
        recipient == last_step_recipient_ && current_step_bytes_ > 0) {
      const double grown =
          static_cast<double>(current_step_bytes_) * options_.step_growth;
      step = static_cast<size_t>(
          std::min(grown, static_cast<double>(max_step)));
    }

    // A probe bypasses the gates, so it must stay cheap: a fraction of the
    // base step, never accelerated. Worst case (probing at a genuine
    // optimum) this is ~1 MiB of purge per gate-skip window -- noise.
    if (probe_transfer) {
      step = std::max<size_t>(
          size_t{1} << 20,
          base_step / std::max<size_t>(1, options_.probe_step_divisor));
    }

    // A structural reclaim drains at the max step (the excess is dead
    // capacity; converge in few rounds) but never digs below the donor's
    // upper bound, and only moves the excess itself.
    size_t donor_avail = last_capacities_[donor] - floor_bytes[donor];
    if (structural_reclaim) {
      step = max_step;
      donor_avail = std::min(donor_avail, excess_bytes);
    } else if (donor_slack_only &&
               snapshot.usages.size() == level_count) {
      // A slack donor only gives away its unused capacity; its resident set
      // is untouched.
      const size_t protected_bytes = std::max(
          floor_bytes[donor], static_cast<size_t>(snapshot.usages[donor]));
      donor_avail = std::min(
          donor_avail, last_capacities_[donor] > protected_bytes
                           ? last_capacities_[donor] - protected_bytes
                           : 0);
    }
    const size_t recv_room =
        recv_upper_bytes[recipient] > last_capacities_[recipient]
            ? recv_upper_bytes[recipient] - last_capacities_[recipient]
            : 0;
    const size_t transfer = std::min({step, donor_avail, recv_room});

    if (transfer == 0) {
      do_skip_step("zero_transfer");
      return Status::OK();
    }

    std::vector<size_t> new_caps = last_capacities_;
    new_caps[donor] -= transfer;
    new_caps[recipient] += transfer;

    uint64_t dbg_t0 = 0;
    if (kAllocDebug) {
      dbg_t0 = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count());
    }
    Status s = cache_->AdjustCapacities(new_caps);
    if (kAllocDebug) {
      const uint64_t t1 = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count());
      const uint64_t ap = dbg_applied.fetch_add(1) + 1;
      const uint64_t sk = dbg_skipped.load();
      if ((ap + sk) % 50 == 0 || ap <= 5) {
        std::string caps_str;
        for (size_t i = 0; i < new_caps.size(); ++i) {
          caps_str += std::to_string(new_caps[i] >> 20);
          caps_str += i + 1 < new_caps.size() ? "," : "";
        }
        double gap_val = (std::isfinite(best_recv) && best_recv > 0.0)
                             ? (best_recv - best_donor_score) / best_recv
                             : 1.0;
        fprintf(stderr,
                "[MLC_ALLOC] round=%llu APPLY(%s) L%zu->L%zu "
                "transfer=%lluMiB step=%lluMiB rev=%u scorer=%s "
                "score_gap=%.3f apply_us=%llu applied=%llu skipped=%llu "
                "caps(MiB)=[%s]\n",
                (unsigned long long)round_id_,
                structural_reclaim ? "reclaim"
                                   : (probe_transfer ? "probe" : "step"),
                donor, recipient,
                (unsigned long long)(transfer >> 20),
                (unsigned long long)(step >> 20), reversal_streak,
                ghost_scored ? "ghost" : "model", gap_val,
                (unsigned long long)(t1 - dbg_t0),
                (unsigned long long)ap, (unsigned long long)sk,
                caps_str.c_str());
      }
    }
    prev_data_sizes_ = snapshot.data_sizes;
    prev_lookups_ = snapshot.lookups;
    prev_hits_ = snapshot.hits;
    ++round_id_;
    if (s.ok()) {
      last_capacities_ = std::move(new_caps);
      last_round_applied_ = true;
      ++applied_transfer_count_;
      // Any applied transfer restarts the probe countdown.
      consecutive_gate_skips_ = 0;
      // Feed the adaptive-step streak tracker. A structural reclaim moves at
      // max_step out-of-band, and a probe deliberately runs below the base
      // step; letting either seed the streak would distort the next
      // score-driven round's step, so reset instead.
      if (structural_reclaim || probe_transfer) {
        current_step_bytes_ = 0;
        last_step_recipient_ = SIZE_MAX;
      } else {
        current_step_bytes_ = step;
        last_step_recipient_ = recipient;
      }
      // Record the pair transfer for reversal detection (round_id_ was
      // already advanced above; use the pre-advance id the detection ran
      // against so window arithmetic stays consistent).
      pair_last_round_[pair_idx] = round_id_ - 1;
      pair_last_dir_[pair_idx] = transfer_dir;
      pair_reversal_streak_[pair_idx] = reversal_streak;
      // Arm the direction locks, escalated by the pair's reversal streak:
      // base 3 rounds, doubled per streak (3, 6, 12, ...), capped.
      if (options_.step_direction_lock_rounds > 0) {
        uint64_t lock_rounds = options_.step_direction_lock_rounds
                               << std::min<uint32_t>(reversal_streak, 10);
        if (options_.reversal_lock_max_rounds > 0) {
          lock_rounds = std::min(lock_rounds, options_.reversal_lock_max_rounds);
        }
        received_lock_round_[recipient] = round_id_ + lock_rounds;
        donated_lock_round_[donor] = round_id_ + lock_rounds;
      }
    } else {
      current_step_bytes_ = 0;
      last_step_recipient_ = SIZE_MAX;
    }
    return s;
  }  // end incremental marginal-step mode

  std::vector<size_t> capacities_to_apply = target_capacities;
  if (!last_capacities_.empty() &&
      last_capacities_.size() == capacities_to_apply.size()) {
    SmoothCapacities(last_capacities_, target_capacities, options_.smoothing_ratio,
                     &capacities_to_apply);
  }
  EnforceMinActiveLevelFloor(capacities_to_apply, snapshot.data_sizes,
                             total_capacity,
                             options_.min_active_level_capacity_bytes,
                             &capacities_to_apply);
  // Persistence-gated data-share floor on the apply path: lift only levels that
  // have been starved for >= min_starvation_relief_rounds consecutive rounds up
  // to their floor, draining donors above their floor. Keeps a doom-looped deep
  // level at its floor every apply round (so compaction keeps up); a transient
  // dip on a healthy read-only workload has mask=0 and is left unperturbed.
  if (floor_enabled) {
    EnforceDataShareFloor(capacities_to_apply, snapshot.data_sizes,
                          total_capacity,
                          options_.min_active_level_capacity_ratio,
                          options_.min_active_level_floor_bytes,
                          &capacities_to_apply, &relief_mask);
  }
  if (options_.mode == MultiLevelAllocatorMode::kModel && has_prev &&
      options_.compaction_shift_ratio > 0.0) {
    ApplyCompactionAwareShiftByDataDelta(
        snapshot, prev_data_sizes_, prev_lookups_, prev_hits_,
        options_.compaction_shift_ratio,
        options_.compaction_shift_max_total_ratio,
        options_.compaction_shift_debug, round_id_, &capacities_to_apply);
    EnforceMinActiveLevelFloor(capacities_to_apply, snapshot.data_sizes,
                               total_capacity,
                               options_.min_active_level_capacity_bytes,
                               &capacities_to_apply);
  }

  if (!last_capacities_.empty() &&
      last_capacities_.size() == capacities_to_apply.size()) {
    ApplyAntiOscillation(&capacities_to_apply, total_capacity);
    uint64_t total_change = 0;
    for (size_t i = 0; i < capacities_to_apply.size(); ++i) {
      const size_t lhs = capacities_to_apply[i];
      const size_t rhs = last_capacities_[i];
      total_change += static_cast<uint64_t>(lhs > rhs ? lhs - rhs : rhs - lhs);
    }
    // Percentage-based deadband: 1 MiB on an 8 GiB budget (0.012%) lets the
    // allocator chase noise; scale the threshold with the budget.
    const uint64_t effective_min_change = std::max<uint64_t>(
        options_.min_total_change_bytes,
        static_cast<uint64_t>(static_cast<double>(total_capacity) *
                              options_.total_deadband_ratio));
    if (total_change < effective_min_change) {
      if (kAllocDebug) {
        const uint64_t s = dbg_skipped.fetch_add(1) + 1;
        const uint64_t a = dbg_applied.load();
        if ((s + a) % 50 == 0) {
          fprintf(stderr,
                  "[MLC_ALLOC] round=%llu SKIP total_change=%llu applied=%llu "
                  "skipped=%llu\n",
                  (unsigned long long)round_id_,
                  (unsigned long long)total_change, (unsigned long long)a,
                  (unsigned long long)s);
        }
      }
      prev_data_sizes_ = snapshot.data_sizes;
      prev_lookups_ = snapshot.lookups;
      prev_hits_ = snapshot.hits;
      ++round_id_;
      return Status::OK();
    }
  }

  uint64_t dbg_t0 = 0;
  if (kAllocDebug) {
    dbg_t0 = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }
  Status adjust = cache_->AdjustCapacities(capacities_to_apply);
  if (kAllocDebug) {
    const uint64_t t1 = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    const uint64_t a = dbg_applied.fetch_add(1) + 1;
    const uint64_t s = dbg_skipped.load();
    if ((a + s) % 50 == 0 || a <= 5) {
      uint64_t tc = 0;
      for (size_t i = 0; i < capacities_to_apply.size(); ++i) {
        const size_t lhs = capacities_to_apply[i];
        const size_t rhs =
            i < last_capacities_.size() ? last_capacities_[i] : 0;
        tc += static_cast<uint64_t>(lhs > rhs ? lhs - rhs : rhs - lhs);
      }
      std::string caps;
      for (size_t i = 0; i < capacities_to_apply.size(); ++i) {
        caps += std::to_string(capacities_to_apply[i] >> 20);
        caps += i + 1 < capacities_to_apply.size() ? "," : "";
      }
      fprintf(stderr,
              "[MLC_ALLOC] round=%llu APPLY change=%lluMiB apply_us=%llu "
              "applied=%llu skipped=%llu caps(MiB)=[%s]\n",
              (unsigned long long)round_id_, (unsigned long long)(tc >> 20),
              (unsigned long long)(t1 - dbg_t0), (unsigned long long)a,
              (unsigned long long)s, caps.c_str());
    }
  }
  prev_data_sizes_ = snapshot.data_sizes;
  prev_lookups_ = snapshot.lookups;
  prev_hits_ = snapshot.hits;
  ++round_id_;
  if (adjust.ok()) {
    last_capacities_ = std::move(capacities_to_apply);
    last_round_applied_ = true;
  }
  return adjust;
}

}  // namespace ROCKSDB_NAMESPACE

