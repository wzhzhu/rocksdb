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
  // Background solve interval in milliseconds. When adjust_interval_ops > 0 this
  // is only the poll granularity (how often the loop wakes to check the op
  // counter and Stop()); the round itself is gated on op count, not wall time.
  uint64_t interval_ms = 1000;
  // Op-count adjustment cadence: run a solve/apply round once this many cache
  // lookups have elapsed since the previous round, instead of on a wall-clock
  // interval. This decouples the number of adjustment rounds from thread count /
  // throughput (a 5s wall interval gives fewer rounds at high throughput, which
  // made the converged allocation -- and hit ratio -- vary non-monotonically
  // with thread count). 0 falls back to the legacy fixed interval_ms cadence.
  uint64_t adjust_interval_ops = 100000;
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
  // AdjustCapacities. When > 0 this acts as a flat per-active-level floor
  // (legacy/compat interface).
  size_t min_active_level_capacity_bytes = 0;
  // Data-share-weighted floor: reserve this fraction of the total budget as a
  // floor pool, distributed across active levels (data_size > 0) in proportion
  // to their data size so deep levels (e.g. L6 holding ~99% of data) get a
  // proportionally larger floor than small upper levels. Each active level also
  // receives at least `min_active_level_floor_bytes`. The floor is enforced
  // BEFORE the model-stability gate and is applied even on gate-skip, so a
  // starved level is always relieved (breaks the doom loop where the gate
  // suppresses the large corrective swing out of a starved state). 0 disables.
  double min_active_level_capacity_ratio = 0.05;
  // Per-active-level absolute floor minimum (keeps small upper-level sub-caches
  // functional even when their data share rounds toward zero).
  size_t min_active_level_floor_bytes = 1 << 16;  // 64 KiB
  // Persistence gate for the data-share floor: a level is only force-fed its
  // floor after it has been proposed below that floor for this many CONSECUTIVE
  // rounds. Default 1 = fire whenever a level is below its floor. (A value > 1
  // was found to worsen read-only perturbation; kept as a knob.)
  uint64_t min_starvation_relief_rounds = 1;
  // Compaction-pressure gate for the data-share floor: the floor relief only
  // fires when the L0 SST file count is >= this threshold, i.e. when compaction
  // is falling behind (the write-heavy doom-loop signature). On a healthy
  // read-only workload L0 stays at ~1 file, so the floor never fires and the
  // adaptive convergence is not perturbed. 0 = fire whenever a level is below
  // its floor regardless of L0 (aggressive; perturbs read-only). The threshold
  // is intentionally well below RocksDB's level0_slowdown_trigger so relief
  // starts before any actual write stall.
  uint64_t floor_relief_l0_file_threshold = 4;
  // Compaction-aware capacity transfer ratio in [0, 1].
  // 0 disables this heuristic.
  double compaction_shift_ratio = 0.0;
  // Per-round upper bound of total transfer as a fraction of total cache.
  // Applied only when compaction_shift_ratio > 0.
  double compaction_shift_max_total_ratio = 0.1;
  // Enable per-round debug logging for compaction-aware transfer.
  bool compaction_shift_debug = false;

  // --- Anti-oscillation (capacity jitter wastes warm blocks, and with
  // purge-on-shrink every spurious shrink is destructive) ---
  // Total-change deadband as a fraction of total capacity; the effective
  // skip threshold is max(min_total_change_bytes, total * this).
  double total_deadband_ratio = 0.005;
  // Per-level deadband: a proposal moving a level by less than
  // max(per_level_deadband_min_bytes, current * per_level_deadband_ratio)
  // is dropped (the level keeps its current capacity).
  double per_level_deadband_ratio = 0.05;
  size_t per_level_deadband_min_bytes = 64 << 10;  // 64 KiB
  // A level's shrink is only applied after this many consecutive rounds
  // proposing a shrink for it, and then only half the gap per round.
  // Grows apply immediately. <= 1 disables the hysteresis.
  uint32_t shrink_confirm_rounds = 3;
  // Deprecated/unused: the background loop now wakes at a fixed interval_ms
  // cadence (no adaptive backoff). Kept for ABI/option compatibility.
  uint32_t max_interval_backoff = 8;

  // --- Data-size cap (don't allocate a level more than it can cache) ---
  // The exponential MRC model can hand a hot-but-small level far more capacity
  // than its data occupies; the surplus sits idle while deeper, undersaturated
  // levels starve. When enabled, each level's capacity is upper-bounded by
  // data_size * data_cap_margin_ratio and the (capped) water-filling solver
  // redistributes the surplus to higher-marginal levels.
  bool cap_at_data_size = true;
  // Headroom over physical data size to absorb block/cache accounting overhead
  // so a fully-hot level still caches all of its blocks.
  double data_cap_margin_ratio = 1.10;
  // Capacity granted to levels with zero data (keeps the sub-cache functional
  // without squandering budget on empty levels).
  size_t empty_level_cap_bytes = 1 << 16;  // 64 KiB

  // --- Model-stability gate (don't act on an untrustworthy signal) ---
  // Under write-heavy or low-cache/data-ratio workloads the per-round model
  // inputs (per-level access rate lambda, hit-rate-derived alpha) are noisy, so
  // the water-filling solver emits wildly different target allocations from one
  // round to the next (e.g. the bulk bottom level swinging between ~5% and ~50%
  // of the budget). Applying those churns SetCapacity/PurgeToCapacity on the hot
  // sub-caches every round (no hit-ratio benefit, large throughput loss that
  // compounds as the run lengthens). When two consecutive raw solved targets
  // disagree by more than this fraction of the total capacity, the signal is
  // deemed unstable and the round is skipped (the previous allocation is held).
  // A stable workload produces consecutive targets that agree, so it still
  // adapts normally. Note the metric double-counts a transfer (moving X bytes
  // between levels contributes 2X to the swing), so this 0.20 fraction gates at
  // roughly a 10% single-round budget relocation. 0 disables the gate.
  // Only used in the legacy water-filling mode (adjust_step_bytes == 0).
  double model_stability_threshold = 0.20;

  // --- Incremental marginal-step mode ---
  // When > 0, RunOnceLocked replaces the global water-filling solver with a
  // bounded marginal-step algorithm:
  //   1. Compute a per-level marginal score: the windowed ghost-hit count
  //      (use_ghost_marginal, direct measurement) or the KKT marginal
  //      λ_i·(α_i/D_i)·miss_rate_i(c_i) from the exponential model (fallback).
  //   2. donor    = argmin(score) among levels with capacity > data-share floor.
  //   3. recipient = argmax(score) among levels with data > 0 and room to grow.
  //   4. Transfer min(adjust_step_bytes, donor_slack, recipient_room) bytes.
  //
  // Why this is better than global water-filling for large caches / low
  // concurrency:
  //   - The water-filling solver re-solves a globally optimal allocation each
  //     round and can swing the entire budget across levels (observed: 7 GB in
  //     one step). Each such swing evicts the excess under the shard locks, and
  //     the eviction cost grows with the resident set -- measured ~175 ms/round
  //     at 8 GB full cache -- stalling t4/t8 threads to ~2 KTPS while the
  //     working set is purged every round, so the hit ratio never rises either.
  //   - The incremental step bounds the per-round eviction to a few ms (64 MiB
  //     at a time), converges naturally to the same optimum over many rounds,
  //     and needs no stability gate, smoothing, or shrink hysteresis.
  //   - Near the optimum the score gap falls below step_min_score_ratio and
  //     transfers stop automatically (self-stabilising).
  //
  // Cold start: if last_capacities_ is empty, an equal split is applied once.
  // Set to 0 to revert to the legacy water-filling mode.
  size_t adjust_step_bytes = 64 << 20;  // 64 MiB per round

  // Minimum relative score gap (recipient_score - donor_score) / recipient_score
  // required to trigger a transfer. Prevents micro-transfers when the allocation
  // is already near-optimal. 0 = always transfer if a valid donor/recipient
  // pair exists.
  double step_min_score_ratio = 0.02;

  // Adaptive step acceleration (Rprop-style): while consecutive APPLIED rounds
  // pick the same recipient (the allocation is still far from the optimum and
  // keeps pushing in one direction), the effective step grows by step_growth
  // per round up to step_max_bytes; as soon as the recipient changes -- or a
  // round is skipped -- the step resets to adjust_step_bytes. Large steps
  // early (fast convergence out of the equal-split cold start), small steps
  // late (bounded churn near the optimum), with no schedule or target
  // knowledge required. step_max_bytes also bounds the per-round
  // PurgeToCapacity cost. Set step_growth <= 1.0 to disable acceleration.
  //
  // Both values are additionally capped relative to the total budget at run
  // time (base <= total/32, max <= total/8): the absolute defaults were tuned
  // at 8 GiB and would otherwise move up to half of a 1 GiB cache in a single
  // round, leaving the grow-only AutoHCC tables sparse and degrading Evict.
  size_t step_max_bytes = 512 << 20;  // 512 MiB cap for the accelerated step
  double step_growth = 2.0;           // per-round multiplier on same recipient

  // Ghost (repeat-miss) marginal scoring for the incremental mode. When true
  // and the cache has ghost tracking enabled, the per-level step score is the
  // windowed ghost-hit count -- a direct, model-free measurement of the miss
  // traffic that a capacity increase would convert into hits -- instead of the
  // exponential-model score λ·(α/D)·miss_rate. This removes the alpha
  // single-point-inversion pathologies (score degenerating to
  // λ·m·(-ln m)/c, prior drag, fill-delay feedback). Falls back to the model
  // score when the drained ghost vector is unavailable.
  bool use_ghost_marginal = false;

  // Per-byte normalization of the ghost score: score_i = ghost_i / D_i.
  // The raw ghost count measures HOW MANY misses more capacity could convert
  // into hits, but not the capacity price per hit: converting L3's repeat
  // misses (data 0.8 GiB) costs an order of magnitude fewer bytes per hit
  // than L5's (data 15 GiB), whose recently-missed keys are spread over a
  // far larger footprint. With raw counts the allocator parked ~1 GiB on L5
  // for a 0.037 level hit ratio while L3/L4 (which the model-based allocator
  // funded to 0.25/0.26) starved. Dividing by data size restores the per-byte
  // marginal ordering -- the same structural information the exponential
  // model carried in its α/D factor. The Poisson significance gate still
  // operates on the raw (EMA) counts of the selected pair, since the
  // sqrt-noise model only makes sense for counts.
  bool ghost_normalize_by_data = true;

  // --- Steady-state suppression (incremental mode) ---
  // Observed pathology on write-heavy steady state (wlA, 100M ops): ghost
  // hits never reach zero (compaction keeps shuffling data), per-window
  // counts are noisy, and the raw argmax/argmin flip every round -- so the
  // allocator applied a transfer on ~890 of 900 rounds, including ping-pong
  // pairs (L3->L0 followed by L0->L5), each paying a PurgeToCapacity that
  // grows to 9-39ms as the cache fills. Three complementary guards:
  //
  // 1. EMA smoothing of ghost scores: kills single-window noise as the
  //    direction signal. 1.0 = no smoothing (raw window counts).
  double ghost_score_ema_beta = 0.3;
  // 2. Direction lock: a level that received capacity in one of the last
  //    step_direction_lock_rounds APPLIED rounds cannot be a donor, and a
  //    recent donor cannot be a recipient. Directly prevents ping-pong
  //    transfers regardless of score noise. 0 disables.
  uint64_t step_direction_lock_rounds = 3;
  // 3. Significance gate (ghost scores only): a transfer requires BOTH
  //      recv_score > ghost_min_recv_donor_ratio * donor_score
  //    AND
  //      recv_score - donor_score > ghost_significance_k *
  //                                 sqrt(recv_score + donor_score)
  //    (Poisson-noise significance for count data). During convergence the
  //    gaps are 10-100x and pass trivially; in steady-state noise they fail
  //    and the round is skipped. Set ratio to 0 to disable both checks.
  double ghost_min_recv_donor_ratio = 2.0;
  double ghost_significance_k = 3.0;
};

// Periodically solves and applies multi-level cache capacities from
// (lambda_i, D_i, alpha_i) samples.
class MultiLevelCacheAllocator {
 public:
  // Fills lambda/data/alpha vectors and returns true when a valid sample
  // is available. Return false to skip this round. `l0_file_count` (when non-
  // null) is filled with the current L0 SST file count -- a direct compaction-
  // backlog signal used to gate the data-share floor relief so it only fires
  // under real compaction pressure (write-heavy doom loop) and never perturbs a
  // healthy read-only workload (where L0 stays at ~1 file).
  using MetricsProvider = std::function<bool(std::vector<double>* lambda,
                                             std::vector<double>* data,
                                             std::vector<double>* alpha,
                                             uint64_t* l0_file_count)>;

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
                                int max_iterations = 80,
                                const std::vector<double>& upper_bounds =
                                    std::vector<double>());

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
  // Data-share-weighted floor: floor_i = max(min_active_level_floor_bytes,
  // total * ratio * data_share_i), where data_share_i = data_size_i /
  // sum(active data_sizes). Lifts starved levels up to their floor, draining
  // the surplus from donors (largest first, never below their own floor). When
  // `relief_mask` is non-null, only levels with `(*relief_mask)[i] != 0` are
  // force-lifted; masked-off levels are left alone (neither lifted nor blocked
  // from donating), used by the persistence gate so transient dips don't fire.
  static void EnforceDataShareFloor(
      const std::vector<size_t>& in_capacities,
      const std::vector<uint64_t>& level_data_sizes, size_t total_budget,
      double ratio, size_t floor_min_bytes, std::vector<size_t>* out,
      const std::vector<unsigned char>* relief_mask = nullptr);

  void BackgroundLoop();
  Status RunOnceLocked();
  // Applies per-level deadband and shrink hysteresis against
  // last_capacities_, then rebalances the proposal back to `budget`.
  void ApplyAntiOscillation(std::vector<size_t>* proposed, size_t budget);

  std::shared_ptr<MultiLevelCache> cache_;
  MetricsProvider provider_;
  MultiLevelAllocationOptions options_;

  std::atomic<bool> running_{false};
  std::thread worker_;
  mutable std::mutex mu_;
  std::vector<size_t> last_capacities_;
  // Previous round's raw solved target capacities (pre-smoothing), used by the
  // model-stability gate to detect an untrustworthy (rapidly swinging) signal.
  std::vector<size_t> prev_target_capacities_;
  std::vector<uint64_t> prev_data_sizes_;
  std::vector<uint64_t> prev_lookups_;
  std::vector<uint64_t> prev_hits_;
  // Consecutive rounds each level has been proposed to shrink.
  std::vector<uint32_t> shrink_streak_;
  // Consecutive rounds each active level's solved target has fallen below its
  // data-share floor. Drives the persistence gate for the floor relief so it
  // only fires on persistent (doom-loop) starvation, not transient dips.
  std::vector<uint32_t> starvation_rounds_;
  // Whether the last RunOnceLocked applied a capacity change (drives the
  // adaptive interval backoff).
  bool last_round_applied_ = false;
  uint64_t round_id_ = 0;
  // Adaptive step state (incremental mode): the accelerated step used by the
  // last applied round, and its recipient. current_step_bytes_ == 0 means
  // "start from options_.adjust_step_bytes".
  size_t current_step_bytes_ = 0;
  size_t last_step_recipient_ = SIZE_MAX;
  // Steady-state suppression state (incremental mode). EMA-smoothed ghost
  // scores, and per-level direction locks: a level with
  // received_lock_round_[i] > round_id_ cannot donate, one with
  // donated_lock_round_[i] > round_id_ cannot receive.
  std::vector<double> ghost_score_ema_;
  std::vector<uint64_t> received_lock_round_;
  std::vector<uint64_t> donated_lock_round_;
  // Total cache lookups (summed across levels) at the last op-gated round.
  // Used by BackgroundLoop to decide when adjust_interval_ops have elapsed.
  uint64_t last_round_lookups_ = 0;
};

}  // namespace ROCKSDB_NAMESPACE

