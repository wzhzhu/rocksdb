//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "cache/multi_level_cache.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {

enum class MultiLevelAllocatorMode {
  kModel = 0,
  kBaselineEmulation = 1,
};

struct MultiLevelAllocationOptions {
  // Background solve interval in milliseconds.
  uint64_t interval_ms = 1000;
  // Exponential smoothing factor in [0, 1], where 1 means no smoothing.
  double smoothing_ratio = 0.5;
  // Skip applying capacities when total delta is below this threshold.
  size_t min_total_change_bytes = 1 << 20;  // 1 MiB
  // Binary search precision and iterations.
  double solver_epsilon = 1e-9;
  int solver_max_iterations = 80;
  // Allocation mode.
  MultiLevelAllocatorMode mode = MultiLevelAllocatorMode::kModel;
  // Minimum capacity for each active level (data_size > 0), applied before
  // AdjustCapacities.
  size_t min_active_level_capacity_bytes = 0;
  // Compaction-aware capacity transfer ratio in [0, 1].
  // 0 disables this heuristic.
  double compaction_shift_ratio = 0.0;
  // Per-round upper bound of total transfer as a fraction of total cache.
  // Applied only when compaction_shift_ratio > 0.
  double compaction_shift_max_total_ratio = 0.1;
  // Enable per-round debug logging for compaction-aware transfer.
  bool compaction_shift_debug = false;
};

// Periodically solves and applies multi-level cache capacities from
// (lambda_i, D_i, alpha_i) samples.
class MultiLevelCacheAllocator {
 public:
  // Fills lambda/data/alpha vectors and returns true when a valid sample
  // is available. Return false to skip this round.
  using MetricsProvider = std::function<bool(std::vector<double>* lambda,
                                             std::vector<double>* data,
                                             std::vector<double>* alpha)>;

  MultiLevelCacheAllocator(std::shared_ptr<MultiLevelCache> cache,
                           MetricsProvider provider,
                           MultiLevelAllocationOptions options);
  ~MultiLevelCacheAllocator();

  MultiLevelCacheAllocator(const MultiLevelCacheAllocator&) = delete;
  MultiLevelCacheAllocator& operator=(const MultiLevelCacheAllocator&) = delete;
  MultiLevelCacheAllocator(MultiLevelCacheAllocator&&) = delete;
  MultiLevelCacheAllocator& operator=(MultiLevelCacheAllocator&&) = delete;

  void Start();
  void Stop();
  bool IsRunning() const {
    return running_.load(std::memory_order_relaxed);
  }

  // Trigger one solve/apply cycle immediately.
  Status RunOnce();

  static Status SolveCapacities(const std::vector<double>& lambda,
                                const std::vector<double>& data,
                                const std::vector<double>& alpha,
                                size_t total_capacity,
                                std::vector<size_t>* capacities,
                                double epsilon = 1e-9,
                                int max_iterations = 80);

 private:
  static void EqualSplit(size_t total_capacity, size_t levels,
                         std::vector<size_t>* capacities);
  static std::vector<size_t> QuantizeToBudget(const std::vector<double>& values,
                                              size_t budget);
  static void SmoothCapacities(const std::vector<size_t>& previous,
                               const std::vector<size_t>& target, double ratio,
                               std::vector<size_t>* out);
  static void EnforceMinActiveLevelFloor(
      const std::vector<size_t>& in_capacities,
      const std::vector<uint64_t>& level_data_sizes, size_t total_budget,
      size_t min_active_level_capacity_bytes, std::vector<size_t>* out);

  void BackgroundLoop();
  Status RunOnceLocked();

  std::shared_ptr<MultiLevelCache> cache_;
  MetricsProvider provider_;
  MultiLevelAllocationOptions options_;

  std::atomic<bool> running_{false};
  std::thread worker_;
  mutable std::mutex mu_;
  std::vector<size_t> last_capacities_;
  std::vector<uint64_t> prev_data_sizes_;
  std::vector<uint64_t> prev_lookups_;
  std::vector<uint64_t> prev_hits_;
  uint64_t round_id_ = 0;
};

}  // namespace ROCKSDB_NAMESPACE

