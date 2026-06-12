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
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <numeric>
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
    std::vector<size_t>* capacities, double epsilon, int max_iterations) {
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

  if (!has_positive_term || a_max <= 0.0) {
    EqualSplit(total_capacity, levels, capacities);
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
    if (total > static_cast<double>(total_capacity)) {
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
  *capacities = QuantizeToBudget(continuous, total_capacity);
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
  if (!result.empty() && remaining > 0) {
    const size_t base_add = remaining / result.size();
    if (base_add > 0) {
      for (size_t i = 0; i < result.size(); ++i) {
        result[i] += base_add;
      }
      remaining -= base_add * result.size();
    }
  }
  std::sort(fractions.begin(), fractions.end(),
            [](const FractionalPart& lhs, const FractionalPart& rhs) {
              return lhs.fraction > rhs.fraction;
            });
  for (size_t i = 0; i < remaining && i < fractions.size(); ++i) {
    ++result[fractions[i].index];
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

void MultiLevelCacheAllocator::BackgroundLoop() {
  uint64_t backoff = 1;
  while (running_.load(std::memory_order_acquire)) {
    bool applied = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      Status s = RunOnceLocked();
      s.PermitUncheckedError();
      applied = last_round_applied_;
    }
    // Adaptive interval: in steady state (rounds that change nothing) back
    // off exponentially so the allocator stops perturbing a converged
    // configuration; any applied change snaps back to the base interval.
    if (applied) {
      backoff = 1;
    } else {
      backoff = std::min<uint64_t>(backoff * 2, options_.max_interval_backoff);
    }
    const uint64_t sleep_total_ms = options_.interval_ms * backoff;
    // Sleep in small chunks so Stop() stays responsive under long backoffs.
    uint64_t slept = 0;
    while (slept < sleep_total_ms && running_.load(std::memory_order_acquire)) {
      const uint64_t chunk = std::min<uint64_t>(100, sleep_total_ms - slept);
      std::this_thread::sleep_for(std::chrono::milliseconds(chunk));
      slept += chunk;
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
  std::vector<size_t> target_capacities;
  if (options_.mode == MultiLevelAllocatorMode::kBaselineEmulation) {
    target_capacities.assign(level_count, 0);
    target_capacities[0] = total_capacity;
  } else {
    if (!provider_(&lambda, &data, &alpha)) {
      prev_data_sizes_ = snapshot.data_sizes;
      prev_lookups_ = snapshot.lookups;
      prev_hits_ = snapshot.hits;
      return Status::OK();
    }
    Status solve_status =
        SolveCapacities(lambda, data, alpha, total_capacity, &target_capacities,
                        options_.solver_epsilon, options_.solver_max_iterations);
    if (!solve_status.ok()) {
      return solve_status;
    }
  }

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
      prev_data_sizes_ = snapshot.data_sizes;
      prev_lookups_ = snapshot.lookups;
      prev_hits_ = snapshot.hits;
      ++round_id_;
      return Status::OK();
    }
  }

  Status adjust = cache_->AdjustCapacities(capacities_to_apply);
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

