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

  // Per-byte normalization of the ghost score:
  //   score_i = ghost_i / max(0, D_i - c_i)   (0 when fully cached).
  // The raw ghost count measures HOW MANY misses more capacity could convert
  // into hits, but not the capacity price per hit: converting L3's repeat
  // misses (data 0.8 GiB) costs an order of magnitude fewer bytes per hit
  // than L5's (data 15 GiB), whose recently-missed keys are spread over a
  // far larger footprint. With raw counts the allocator parked ~1 GiB on L5
  // for a 0.037 level hit ratio while L3/L4 (which the model-based allocator
  // funded to 0.25/0.26) starved. Normalizing restores the per-byte marginal
  // ordering -- the same structural information the exponential model
  // carried in its α/D factor.
  // The denominator is the UNCACHED footprint (D_i - c_i), because ghost
  // hits are produced only by the uncached portion; dividing by total D_i
  // instead made small-data levels capacity magnets (L0's ~150x 1/D
  // advantage let it hoard 1.1 GiB of a 2 GiB budget and drain L6, while
  // its residual misses were churn-compulsory and unconvertible).
  bool ghost_normalize_by_data = true;
  // Denominator choice for the normalization above.
  //   false (default): denom_i = D_i (total data size). Allocation-invariant,
  //     so scores have no feedback through the allocation itself.
  //   true: denom_i = max(D_i - c_i, ghost_uncached_floor_frac * D_i), i.e.
  //     the uncached footprint. Theoretically the marginal denominator under
  //     UNIFORM within-level access, but under zipfian it has a
  //     rich-get-richer positive feedback: as a level approaches full
  //     caching its denominator shrinks toward the floor, its score rises,
  //     it wins more capacity, and the loop repeats -- measured on wlA 100M
  //     at 2 GiB t64 it drove the allocation all-in on L4 (1948 MiB of a
  //     2 GiB budget; L3 starved at 0.8 MiB with 9.3M lookups) and cost
  //     ~2pt of fg hit ratio vs the D_i denominator. Kept as an option for
  //     A/B studies. Either way, fully-cached levels (c_i >= D_i) score 0.
  bool ghost_normalize_by_uncached = false;
  double ghost_uncached_floor_frac = 0.1;

  // Capture-rate scoring (segmented ghost). Replaces the static-denominator
  // normalizations above with a measured per-byte marginal utility: the
  // cache records each repeat miss's reuse distance (distinct missed blocks
  // between the two misses; see MultiLevelCache::DrainGhostDistanceHistogram)
  // and the score integrates the histogram as
  //   score_i = sum_b hist_i[b] / (mid_b * ghost_dist_block_bytes)
  // i.e. each repeat is weighted by the reciprocal of the capacity needed to
  // capture it. Raw counts assume every repeat is capturable by one step
  // (f=1); /D and /uncached assume uniform within-level reuse spread; the
  // histogram measures the actual concentration, so a level with a tight
  // zipfian hot tail (L5's 555 MiB / 6.2% level-hit opportunity that /D's
  // 20x footprint penalty killed) scores by its true short-distance mass,
  // and the score has no allocation feedback (distances are counted among
  // missed blocks, independent of current c_i). Requires the cache to have
  // segmented ghost tracking enabled; falls back to the plain ghost count
  // path (with the normalizations above) when the histogram is unavailable.
  bool use_ghost_capture_rate = false;
  // Bytes per block for converting histogram distances (in blocks) to a
  // per-byte score. Should match the workload's dominant block size
  // (uncompressed block-cache charge); only the RELATIVE score across
  // levels matters, so a uniform constant is sufficient.
  size_t ghost_dist_block_bytes = 4096;
  // Reuse-distance decompression. Histogram distances are counted in
  // distinct MISSED blocks, but the capacity needed to capture a repeat is
  // set by the distinct ACCESSED blocks in between -- of which a fraction h
  // (the level's hit rate) never reached the miss stream. The measured
  // distance therefore understates the true one by a factor ~(1-h), which
  // systematically INFLATES the capture scores of high-hit levels: at
  // h=0.75 a repeat's per-byte value reads 4x its true worth. On wlB
  // (zipfian spread over the whole keyspace, hot mass physically in L6)
  // this bias over-funded the mid levels: L4 held 833 MiB earning
  // 1.36 hits/MiB while L6 -- 60% of foreground traffic, earning
  // 6.3 hits/MiB -- was starved at 564 MiB, a large part of the -11pt fg
  // hit gap vs HCC at 2G t64. Each bucket's mid distance is multiplied by
  // min(this cap, 1/(1-h_window)) with h_window the level's EMA-smoothed
  // window fg hit rate. The cap bounds the correction where h -> 1
  // (a fully-fed level's residual misses give a noisy h estimate, and its
  // score is heading to 0 anyway). <= 1 disables decompression.
  // Default OFF: measured neutral on wlB 2G t64 (0.275 vs 0.277) and
  // wlD 2G t128 (0.699 vs 0.702) once the donor retention gate is active
  // -- the gate already blocks the trades this bias would mis-rank, so
  // the correction adds an unvalidated variable without measured benefit.
  // Kept as an option for ablation.
  double ghost_dist_decompress_max = 1.0;
  // In-flight duplicate-miss filter for the capture-rate score. Repeats at
  // short reuse distance cannot be capacity starvation for a level holding
  // real capacity: the first miss's fill would serve the re-access. They
  // are concurrent threads missing the same block before the first fill
  // lands -- an artifact that scales with thread count and I/O queueing
  // depth, not with capacity pressure. Measured on wlD t128 the artifact
  // was ~100% of the coldest levels' capture score (bucket 0 alone
  // contributed 5e-4 hits/byte to L6 vs 5e-8 from its genuine
  // far-distance buckets, and under deep queueing the overlap window
  // stretched into the 256 KiB - 2 MiB buckets), which made the coldest
  // levels read as the hungriest, drained the true hot level (L3: 60% of
  // traffic, capable of 0.95 hit) to zero, and collapsed fg hit ratio
  // 0.67 -> 0.45 at t128 while t32 (fewer overlapping misses) converged
  // fine. Buckets whose mid distance is at or below
  // ghost_inflight_dist_bytes are therefore discarded -- but only for
  // levels holding at least ghost_inflight_min_cap_bytes of capacity: a
  // genuinely defunded level's short-distance repeats ARE starvation
  // signal and must keep counting. Set dist to 0 to disable the filter.
  size_t ghost_inflight_dist_bytes = 2 << 20;
  size_t ghost_inflight_min_cap_bytes = 8 << 20;
  // Donor retention cost for capture-rate mode. The capture score measures
  // only the marginal value of GROWING a level; a fully-fed level's score
  // is correctly ~0 (its misses are all converted), but argmin-score donor
  // selection then reads exactly that level as the cheapest source of
  // bytes -- draining the highest-hit-density level in the cache to feed
  // marginal recipients (observed on wlD: fully-cached L3 at 0.95 hit and
  // ~324 hits/MB/window drained to zero to feed L4/L6, then frozen at zero
  // by the escalated reversal locks). The shrink cost of a FULL level is
  // what its resident bytes currently earn, which is directly measurable:
  //   density_i = window_fg_hits_i / capacity_i  (hits/byte/window)
  // -- the same units as the capture score, so recipient gain and donor
  // loss compare directly. Full-level donors are selected by argmin
  // density, and a transfer requires
  //   recipient_capture_score > donor_retention_frac * donor_density.
  // The fraction discounts mean density to marginal: clock eviction takes
  // the donor's coldest bytes, which earn far less than its average byte
  // (zipfian concentration), so requiring the recipient to beat the full
  // mean would over-protect donors and freeze legitimate rebalancing.
  // Slack donors (cap > usage) evict nothing and bypass the gate. 0
  // disables both the density selection and the gate.
  double donor_retention_frac = 0.2;

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
  //    (Poisson-noise significance for count data). Set ratio to 0 to
  //    disable both checks.
  //    DEFAULT OFF: k=3 was tuned against 2^16 ghost tables; at the current
  //    2^18 tables counts are ~4x larger, relative Poisson noise ~2x
  //    smaller, and the gate passed ~79% of steady-state rounds (measured,
  //    wlA 100M 2 GiB) -- de facto a no-op, while its historical failure
  //    mode (freezing convergence overshoot in place) motivated the probe
  //    mechanism below. With cold-start-only acceleration + the near-optimal
  //    ratio gate + direction locks + reversal hysteresis, steady-state
  //    churn is bounded at base-step size and mechanically cheap, so the
  //    lean stack omits this gate. Kept for ablation studies.
  double ghost_min_recv_donor_ratio = 0.0;
  double ghost_significance_k = 3.0;
  // 4. Probe transfers (anti-freeze annealing). If the gates freeze the
  //    allocation at a suboptimal point (historically: the significance gate
  //    freezing convergence overshoot), then after
  //    probe_after_skipped_rounds consecutive gate-skipped rounds allow ONE
  //    small transfer (base step / probe_step_divisor) from the score argmin
  //    to argmax, bypassing the near-optimal and significance gates but
  //    still respecting direction locks, floors, and upper bounds. With the
  //    significance gate now off by default the only remaining freezer is
  //    the near-optimal ratio gate, so the probe serves as cheap insurance
  //    (~base_step/4 purge per probe) rather than a primary mechanism.
  //    0 disables probing.
  uint64_t probe_after_skipped_rounds = 8;
  size_t probe_step_divisor = 4;
  // 5. Reversal hysteresis. The fixed 3-round direction lock stops
  //    round-to-round ping-pong but not the 50-100 round oscillation
  //    observed at 2 GiB (L3 capacity swinging 50<->924 MiB, L4->L0
  //    followed by reclaim L0->L4), where every accelerated 128-256 MiB
  //    reversal purges the warm set the previous transfer had just built.
  //    When a transfer's direction is the OPPOSITE of the last transfer on
  //    the same level pair within reversal_window_rounds, (a) the step is
  //    clamped to the base step (undoing prior work at an accelerated step
  //    maximizes warm-set loss), and (b) the pair's direction locks are
  //    escalated exponentially with its reversal streak
  //    (step_direction_lock_rounds << streak, capped at
  //    reversal_lock_max_rounds), so a pair that keeps flip-flopping gets
  //    frozen for progressively longer while one-directional (corrective)
  //    traffic is never slowed. 0 window disables.
  uint64_t reversal_window_rounds = 200;
  uint64_t reversal_lock_max_rounds = 96;
  // 6. Cold-start-only step acceleration. The Rprop-style step doubling was
  //    added to cut the cold-start convergence tax, but left enabled in
  //    steady state it is the primary overshoot generator: the EMA-smoothed
  //    ghost signal lags ~1/beta windows, so a 3-4 round streak moves
  //    64+128+256 MiB before the signal reflects any of it (measured: L4
  //    parked at 1948 MiB of a 2 GiB budget across repeats, deep levels
  //    pinned to their floors, ~2pt fg hit below the balanced allocation).
  //    Acceleration is therefore permitted only for the first
  //    accel_cold_start_applies applied transfers, and is disabled
  //    permanently the first time a reversal is detected (the definitive
  //    overshoot signal). Steady state always steps at the base step.
  //    0 disables acceleration entirely.
  uint64_t accel_cold_start_applies = 32;
  // 7. EMA smoothing of per-level data sizes (0 < beta <= 1; 1 = raw).
  //    L0's data pulses 0 <-> ~1.3 GiB with every flush/compaction cycle;
  //    the mechanisms keyed off data size (score-0 for fully-cached levels,
  //    floors, structural excess reclaim) all used the instantaneous value
  //    and took turns firing, producing a reclaim->refill loop on L0.
  //    Smoothing the data series they see removes the pulse without
  //    changing any of their semantics. The RECIPIENT growth cap is the one
  //    exception: it uses min(raw, EMA) -- the sustained data size -- so a
  //    compaction transiently parking data on a normally-empty level
  //    (L1/L2) cannot attract matching capacity that structural reclaim
  //    would have to claw back rounds later (see recv_upper_bytes in the
  //    allocator).
  double data_ema_beta = 0.3;
  // 8. Usage-aware growth gate and reclaim. The data-size bounds above cap a
  //    level at its ON-DISK footprint, but the binding constraint within a
  //    run is the level's TOUCHED footprint: a cold level (~1% of traffic)
  //    only ever inserts the distinct blocks it actually reads, so capacity
  //    beyond its resident usage never fills and contributes exactly zero
  //    hits (with cap > usage there is no eviction: every missed block is
  //    admitted and stays, so hit ratio is identical whether the slack is 1
  //    byte or 800 MiB). Measured on read-only wlC 100M at 2 GiB: L3 parked
  //    at 541 MiB capacity / 165 MiB usage and L4 at 800/210 -- ~1 GiB
  //    (half the budget) dead while fully-utilized L5/L6 starved, costing
  //    ~5pt fg hit ratio vs HCC. The capture-rate score cannot see this:
  //    it measures gain-from-growth on the uncached tail, which for a small
  //    level is a step function (huge below the touched footprint, ~0
  //    above), so score-driven transfers bang-bang around the boundary
  //    instead of resting at it.
  //
  //    (a) Growth gate: a level whose capacity already exceeds
  //        usage * usage_grow_headroom is not accepting more capacity until
  //        it fills what it has (exempt below usage_bootstrap_bytes, so a
  //        defunded level can bootstrap back). Capacity growth thereby
  //        tracks demonstrated demand at the level's own fill rate, and a
  //        lazily-growing level loses nothing (its misses are admitted
  //        either way).
  //    (b) Usage reclaim: capacity above
  //        max(floor, usage * usage_reclaim_margin, usage_bootstrap_bytes)
  //        sustained for usage_reclaim_rounds consecutive rounds is
  //        structural excess (mandatory donor, same machinery as the
  //        data-size reclaim). The persistence window lets a level that
  //        just received a step fill it before the slack is judged dead;
  //        the margin band between grow_headroom and reclaim_margin is the
  //        hysteresis that gives a converged level a stable resting zone.
  //    usage_grow_headroom <= 0 disables the gate;
  //    usage_reclaim_rounds == 0 disables the reclaim.
  double usage_grow_headroom = 1.25;
  double usage_reclaim_margin = 1.3;
  uint64_t usage_reclaim_rounds = 12;
  size_t usage_bootstrap_bytes = 32 << 20;
  // Realization credit cap for the capture-rate score. The capture score
  // promises "each byte granted converts this many repeat misses per
  // window", but the promise silently assumes the cached block SURVIVES
  // until the predicted re-access. Under write-heavy compaction churn it
  // often does not: the block's file is rewritten (new file number = new
  // cache key) before the repeat arrives, so the hit never lands. The
  // measured per-byte hit density of the level's resident bytes
  // (hit_density_ema_, same hits/byte/window units) IS the realized
  // earnings, and for a concave hit curve the marginal value of a new byte
  // cannot exceed the average value of the bytes already held -- so a FULL
  // level whose capture score exceeds its measured density is
  // over-promising, and its score is capped at
  //   score_credit_frac * hit_density_ema.
  // Observed on wlA (50% writes) 2G t64: L5's capture score stayed ~2x
  // above L0's (1.2e-6 vs 4.9e-7) while its resident bytes earned SIXTY
  // times less (0.6K vs 41K hits/MB) -- compaction rewrote L5's files
  // before the promised repeats arrived -- so 100M-op runs drifted to an
  // L5-heavy allocation (759 MiB at run end) costing ~3pt fg hit vs the
  // L0-heavy state short runs converge to. The cap only applies to levels
  // that have had a fair chance to prove their earnings: capacity of at
  // least score_credit_min_cap_bytes AND usage >= 90% of capacity (a
  // still-filling level's density lags its potential; an under-bootstrap
  // level has nothing to prove yet). <= 0 disables.
  double score_credit_frac = 1.0;
  size_t score_credit_min_cap_bytes = 64 << 20;
  // Density floor: the symmetric half of the credit cap, for the UNDER-
  // reporting side of the same churn bias. A high-churn level's repeats
  // never register in the ghost table at all -- the key dies (file
  // rewritten by flush/compaction) before the second miss arrives, so the
  // repeat appears as a first miss on a fresh key. On wlA, L0's capture
  // score sat at ~7e-7 while its resident bytes measurably earned 1.7e-5
  // hits/byte/window (24x more): the ghost simply cannot see through L0's
  // key rotation, and the allocator drained the cache's best earner to 127
  // MiB late in every long run. For an ELIGIBLE level (same guards as the
  // cap: full, i.e. usage >= 90% of capacity, and holding at least
  // score_credit_min_cap_bytes -- a level actively earning with every byte
  // it has), the score is floored at
  //   score_credit_floor_frac * hit_density_ema.
  // The fraction discounts average to marginal (same rationale as
  // donor_retention_frac); growth beyond genuine demand is bounded
  // upstream by the sustained-data recv cap and the usage growth gate, and
  // the floor is self-limiting: growing capacity dilutes density (hd =
  // window hits / capacity), so an over-fed level's floor sinks on its
  // own. Together with the cap, a proven level's effective score lives in
  // [floor_frac, frac] * hd -- realized earnings dominate for funded
  // levels while raw capture stays authoritative for starved ones (whose
  // capacity is too small to measure density on). <= 0 disables.
  //
  // Default 0.5 (raised from 0.25), validated on TTL-clean paired runs
  // (earlier evidence was polluted by RocksDB's 30-day ttl flattening
  // benchmark clones mid-run; see the ttl=0 fix in the YCSB harness).
  // wlA 2G t64: 0.5 gives +4.4% tps at -0.6pt fg (repeats tight and
  // non-overlapping) -- the stronger realized-density floor keeps the
  // high-churn upper levels funded, cutting compaction back-pressure
  // (stall 241-246s vs 250-252s); the win flows through the stall
  // channel, not raw hit ratio. wlB 4G t64: indistinguishable from 0.25
  // (fg 0.330-0.344 vs 0.336-0.349, tps +1.6% inside noise).
  double score_credit_floor_frac = 0.5;
  // --- Compaction-value term (cache-for-compaction) ---
  // Everything above scores levels by FOREGROUND reuse only; compaction's
  // streaming reads are deliberately excluded from the capture signal. But
  // on write-heavy workloads the throughput bottleneck is compaction speed
  // itself (pending-compaction-bytes back-pressure -> write stalls; the
  // no-stall ablation showed the entire MLC throughput advantage flows
  // through this channel), and every compaction read converted to a cache
  // hit accelerates it. Compaction reads have no popularity structure --
  // each live block is read exactly once when its file is merged -- so a
  // level's per-byte compaction value is LINEAR (marginal == average):
  //   comp_density_i = window compaction misses_i / data_bytes_i
  // (caching X% of the level converts ~X% of its compaction reads). The
  // term is added AFTER the realization credit band (the band's concavity
  // argument applies to foreground value only):
  //   score_i = band(fg_capture_i) + w * comp_value_weight * comp_density_i
  // and the same weighted compaction HIT density is folded into the donor
  // retention density so a level funded for compaction is not immediately
  // read as the cheapest donor by fg-only earnings.
  //
  // w is the measured stall intensity: the per-round delta of the DB's
  // cumulative write-stall micros divided by the round's wall time, EMA
  // smoothed, normalized by stall_weight_ref and clamped to [0, 1]. On
  // read-heavy workloads stall is ~0 -> w ~ 0 and behavior is unchanged
  // (fg hits are the only currency that buys anything there); under deep
  // back-pressure w -> 1 and capacity shifts toward converting compaction
  // reads -- exactly the regime where the ablation proved fg hits buy no
  // throughput. Requires the provider to supply cumulative stall micros;
  // if it does not (stall_micros == nullptr semantics: always 0), w stays
  // 0 and the term is inert. comp_value_weight <= 0 disables entirely.
  //
  // DEFAULT OFF after the wlA 2G t64 100M ablation: the term does not
  // convert. Compaction misses concentrate on base-level inputs whose
  // blocks are REWRITTEN by the very compactions that would re-read them
  // (the same churn-blindness physics as the fg credit band), so the
  // density promises capacity that never pays out -- fg hit slipped
  // 0.2097 -> 0.2038 and stall rose with no comp-hit gain. Kept as an
  // ablation lever for workloads with stable compaction inputs.
  double comp_value_weight = 0.0;
  // Stall intensity that maps to full weight w=1. 0.25 means: writers
  // spending an aggregate 25% of wall time in explicit delay/stop is
  // treated as fully stall-bound. wlA 2G t64 measures ~0.4 steady-state.
  double stall_weight_ref = 0.25;
  // --- Stall-adaptive lazy mode ---
  // MLC's fixed overheads (ghost recording on every fg miss, drain +
  // scoring + snapshot work every round) buy nothing once two conditions
  // hold simultaneously: no write back-pressure (a hit-ratio edge cannot
  // buy throughput without a stall channel to relieve -- the no-stall
  // ablation) and a converged allocation (rounds keep skipping, scores
  // only confirm the status quo). Low-thread cells sit in exactly that
  // regime and pay 1-6% throughput for it. When stall intensity stays
  // below lazy_stall_intensity_max AND the allocation has been drift-
  // stationary for lazy_after_stable_rounds calls (see
  // lazy_drift_tolerance_ratio), the allocator enters lazy mode: ghost
  // recording is hash-gate downsampled to 1/2^lazy_ghost_sample_shift of
  // fg misses and only every lazy_round_multiplier-th round runs the full
  // drain/score/transfer path (skipped rounds still read the provider and
  // stall ticker, so pressure is detected at the normal cadence). Stall
  // re-appearance or net capacity drift beyond the tolerance (workload
  // shift) exits immediately back to full speed. The signal stays LIVE in
  // lazy mode -- downsampling is
  // unbiased for the capture score (1/8 the repeats measured against a
  // clock running 1/8 as fast; the scorer rescales counts by 2^shift and
  // the recorder offsets distance buckets to keep true distinct-block
  // units), just noisier -- so a genuine workload shift still produces a
  // transfer, which itself un-lazies the allocator.
  bool lazy_mode_enabled = true;
  // Stall intensity (aggregate stall micros per wall micro, EMA) below
  // which the workload counts as unstalled. wlA 2G t64 measures ~0.4;
  // read-heavy workloads measure ~0.
  double lazy_stall_intensity_max = 0.01;
  // Convergence test: the allocation counts as stationary when the summed
  // per-level capacity DRIFT from a reference snapshot stays within
  // lazy_drift_tolerance_ratio * total for lazy_after_stable_rounds
  // consecutive calls. Net drift, not "no transfers": a converged
  // equilibrium still runs lock-throttled ping-pong and probe annealing
  // (observed on wlC t8: L3<->L5 single-step reversals every ~27 rounds,
  // so a no-apply test is only ever satisfied inside lock windows and
  // lazy mode thrashes), but those transfers cancel out while a genuine
  // workload shift moves capacity in one direction past the tolerance --
  // which resets the reference and (if engaged) exits lazy mode.
  uint64_t lazy_after_stable_rounds = 20;
  // Summed |capacity - reference| tolerance as a fraction of the total
  // budget. Default 1/16 (128 MiB at 2 GiB): a couple of base steps of
  // ping-pong amplitude fit inside; structural shifts do not.
  double lazy_drift_tolerance_ratio = 0.0625;
  // log2 of the ghost-recording downsample in lazy mode (3 = 1/8 of fg
  // misses feed the ghost tables).
  uint32_t lazy_ghost_sample_shift = 3;
  // Only every N-th round runs the full scoring/transfer path in lazy
  // mode; the rest return after the stall check.
  uint32_t lazy_round_multiplier = 4;

  // Physical-apply convergence gate (decouples the allocator's evolving
  // capacity MODEL from the PHYSICAL cache updates). Every round still
  // advances the internal model (last_capacities_) by its chosen step, but
  // the model is only flushed to the sub-caches (cache_->AdjustCapacities,
  // which locks every shard and purges/rebuilds under 128 foreground
  // threads) once the model has drifted, in summed per-level |delta|, at
  // least this fraction of the total budget away from the last physically
  // applied state.
  //
  // Why: at a converged equilibrium the incremental stepper keeps moving one
  // base step (adjust_step_bytes, default 64 MiB = 0.78% of an 8 GiB budget)
  // between a ping-ponging donor/recipient pair. Each single-round change
  // (0.78%) exceeds the per-round deadband (total_deadband_ratio, 0.5%), so
  // the old code physically re-applied EVERY round even though the moves net
  // to ~0 and buy no hit ratio (verified: auto_adjust=off matches default hit
  // ratio at 8 GiB). At 128 threads those zero-value applies avalanche into
  // kernel futex contention (native_queued_spin_lock_slowpath dominates the
  // profile) and cost ~33% throughput on wlC 8G t128. Gating on NET drift
  // from the applied state fully suppresses the equilibrium ping-pong (drift
  // oscillates below tolerance -> zero physical calls) while a genuine
  // workload shift still walks the model in one direction past the tolerance
  // within a couple of rounds and flushes. 0 disables the gate (apply every
  // round, legacy behaviour). Default 0.02 (= ~2.5 base steps at 8 GiB).
  double apply_convergence_drift_ratio = 0.02;
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
  // `stall_micros` (when non-null) is filled with the DB's CUMULATIVE write-
  // stall microseconds (rocksdb STALL_MICROS ticker); the allocator
  // differentiates it into a stall-intensity weight for the compaction-value
  // term. Providers that cannot supply it should leave it at 0 (term inert).
  using MetricsProvider = std::function<bool(std::vector<double>* lambda,
                                             std::vector<double>* data,
                                             std::vector<double>* alpha,
                                             uint64_t* l0_file_count,
                                             uint64_t* stall_micros)>;

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

  // Physical-apply convergence gate (see apply_convergence_drift_ratio).
  // Flushes new_caps to the sub-caches only when it has drifted at least the
  // configured fraction of total_capacity, in summed per-level |delta|, from
  // physical_applied_capacities_ (the last state actually pushed to the
  // cache) -- or when force is set (init / floor relief / disabled gate).
  // On a physical apply it refreshes physical_applied_capacities_. Returns
  // the AdjustCapacities status, or OK when the apply is gate-skipped. The
  // caller advances last_capacities_ (the model) regardless.
  Status MaybeApplyCapacities(const std::vector<size_t>& new_caps,
                              size_t total_capacity, bool force);

  std::shared_ptr<MultiLevelCache> cache_;
  MetricsProvider provider_;
  MultiLevelAllocationOptions options_;

  std::atomic<bool> running_{false};
  std::thread worker_;
  mutable std::mutex mu_;
  std::vector<size_t> last_capacities_;
  // Last capacities physically pushed to the sub-caches (see
  // MaybeApplyCapacities / apply_convergence_drift_ratio). Lags last_capacities_
  // (the model) by at most apply_convergence_drift_ratio * total while the
  // allocation is converged; re-synced on every physical apply.
  std::vector<size_t> physical_applied_capacities_;
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
  // Window fg stats (capture-rate mode): previous cumulative per-level fg
  // hits/lookups (for per-round deltas), the EMA-smoothed per-byte hit
  // density (donor retention cost, see donor_retention_frac), and the
  // EMA-smoothed window hit rate (distance decompression, see
  // ghost_dist_decompress_max).
  std::vector<uint64_t> prev_fg_hits_;
  std::vector<uint64_t> prev_fg_lookups_;
  std::vector<double> hit_density_ema_;
  std::vector<double> hit_rate_ema_;
  // Compaction-value term state (see comp_value_weight): previous cumulative
  // per-level TOTAL lookups/hits (comp = total - fg), previous cumulative DB
  // stall micros + wall clock for the stall-intensity weight, its EMA, and
  // the EMA-smoothed per-byte compaction miss/hit densities.
  std::vector<uint64_t> prev_total_hits_;
  std::vector<uint64_t> prev_total_lookups_;
  std::vector<double> comp_miss_density_ema_;
  std::vector<double> comp_hit_density_ema_;
  uint64_t prev_stall_micros_ = 0;
  uint64_t prev_stall_wall_micros_ = 0;
  bool stall_window_primed_ = false;
  double stall_weight_ema_ = 0.0;
  // Stall-adaptive lazy mode state (see lazy_mode_enabled): whether lazy is
  // engaged, the drift-reference capacity snapshot with the count of
  // consecutive calls the allocation stayed within tolerance of it (the
  // convergence test), calls since lazy engaged (for the 1-in-N full-round
  // gate), and the ghost downsample shift in effect for the window
  // accumulating since the last drain (the scorer's rescale factor; updated
  // at drain time so a mid-window mode flip rescales with the shift the
  // window was recorded under).
  bool lazy_active_ = false;
  std::vector<size_t> lazy_ref_capacities_;
  uint64_t lazy_stable_calls_ = 0;
  uint64_t lazy_call_counter_ = 0;
  uint32_t ghost_window_sample_shift_ = 0;
  uint32_t ghost_pending_sample_shift_ = 0;
  // Consecutive rounds skipped by the steady-state gates; when it reaches
  // probe_after_skipped_rounds, the next round runs as a probe transfer.
  uint64_t consecutive_gate_skips_ = 0;
  // Reversal-hysteresis state, indexed by unordered level pair
  // (min*level_count + max): round of the pair's last applied transfer, its
  // direction (+1 = lower->higher level index, -1 = reverse, 0 = none), and
  // the current reversal streak driving the escalated lock length.
  std::vector<uint64_t> pair_last_round_;
  std::vector<int> pair_last_dir_;
  std::vector<uint32_t> pair_reversal_streak_;
  // Cold-start acceleration budget: applied-transfer count and the
  // first-reversal kill switch (see accel_cold_start_applies).
  uint64_t applied_transfer_count_ = 0;
  bool accel_disabled_ = false;
  // EMA-smoothed per-level data sizes (see data_ema_beta).
  std::vector<double> data_ema_;
  // Consecutive rounds each level's capacity has exceeded its usage-based
  // usable bound (see usage_reclaim_rounds); reset whenever it does not.
  std::vector<uint32_t> usage_excess_rounds_;
  // Previous round's per-level usage, for the still-filling detection that
  // holds the usage reclaim off while a level is actively absorbing a step.
  std::vector<size_t> prev_usages_;
  // Total cache lookups (summed across levels) at the last op-gated round.
  // Used by BackgroundLoop to decide when adjust_interval_ops have elapsed.
  uint64_t last_round_lookups_ = 0;
};

}  // namespace ROCKSDB_NAMESPACE

