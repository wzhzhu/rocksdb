#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <array>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rocksdb/advanced_cache.h"
#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

// A cache wrapper that maintains one sub-cache per LSM-tree level.
// Routing prefers level metadata carried in an optional cache-key suffix.
class MultiLevelCache : public Cache {
 public:
  using SubCacheFactory = std::function<std::shared_ptr<Cache>(size_t)>;

  struct LevelMetricsSnapshot {
    std::vector<uint64_t> lookups;
    std::vector<uint64_t> hits;
    // Foreground-only (excludes compaction-induced) counters. `lookups`/`hits`
    // are totals (include compaction) and feed the allocator's per-level access
    // frequency (lambda); `fg_lookups`/`fg_hits` exclude compaction's streaming
    // one-shot reads and feed the hit-curve shape (alpha), so the model does not
    // let compaction misses dilute foreground cache value.
    std::vector<uint64_t> fg_lookups;
    std::vector<uint64_t> fg_hits;
    std::vector<size_t> capacities;
    std::vector<size_t> usages;
    std::vector<uint64_t> data_sizes;
    // Diagnostics for the AutoHCC sparse-table Evict pathology: the hash table
    // grows to fit peak occupancy but never shrinks, so after the allocator
    // shrinks a sub-cache its table stays large while occupancy is small ->
    // Evict sweeps a sparse oversized table. table_address_counts is the table
    // slot count; occupancy_counts is the live entry count.
    std::vector<size_t> table_address_counts;
    std::vector<size_t> occupancy_counts;
  };

  static const char* kClassName() { return "MultiLevelCache"; }

  MultiLevelCache(size_t num_levels, size_t total_capacity);
  MultiLevelCache(size_t num_levels, size_t total_capacity,
                  const LRUCacheOptions& lru_options,
                  bool initial_force_route_all_to_l0 = false);
  MultiLevelCache(size_t num_levels, size_t total_capacity,
                  const HyperClockCacheOptions& hcc_options,
                  bool initial_force_route_all_to_l0 = false);
  MultiLevelCache(size_t num_levels, size_t total_capacity,
                  SubCacheFactory sub_cache_factory,
                  bool initial_force_route_all_to_l0 = false);
  MultiLevelCache(std::vector<std::shared_ptr<Cache>> sub_caches,
                  std::shared_ptr<Cache> shared_cache, size_t total_capacity);
  ~MultiLevelCache() override;

  const char* Name() const override;
  std::string GetPrintableOptions() const override;

  Status Insert(const Slice& key, ObjectPtr obj, const CacheItemHelper* helper,
                size_t charge, Handle** handle = nullptr,
                Priority priority = Priority::LOW,
                const Slice& compressed = Slice(),
                CompressionType type = CompressionType::kNoCompression) override;

  Handle* CreateStandalone(const Slice& key, ObjectPtr obj,
                           const CacheItemHelper* helper, size_t charge,
                           bool allow_uncharged) override;

  Handle* Lookup(const Slice& key, const CacheItemHelper* helper = nullptr,
                 CreateContext* create_context = nullptr,
                 Priority priority = Priority::LOW,
                 Statistics* stats = nullptr) override;

  bool Ref(Handle* handle) override;

  using Cache::Release;
  bool Release(Handle* handle, bool erase_if_last_ref = false) override;

  ObjectPtr Value(Handle* handle) override;
  void Erase(const Slice& key) override;
  uint64_t NewId() override;

  void SetCapacity(size_t capacity) override;
  void SetStrictCapacityLimit(bool strict_capacity_limit) override;
  bool HasStrictCapacityLimit() const override;

  size_t GetCapacity() const override;
  size_t GetUsage() const override;
  size_t GetUsage(Handle* handle) const override;
  size_t GetPinnedUsage() const override;
  size_t GetCharge(Handle* handle) const override;
  const CacheItemHelper* GetCacheItemHelper(Handle* handle) const override;

  void ApplyToAllEntries(
      const std::function<void(const Slice& key, ObjectPtr obj, size_t charge,
                               const CacheItemHelper* helper)>& callback,
      const ApplyToAllEntriesOptions& opts) override;

  void ApplyToHandle(
      Cache* cache, Handle* handle,
      const std::function<void(const Slice& key, ObjectPtr obj, size_t charge,
                               const CacheItemHelper* helper)>& callback)
      override;

  void EraseUnRefEntries() override;

  size_t GetOccupancyCount() const override;
  size_t GetTableAddressCount() const override;

  // Dynamically updates each level cache capacity.
  Status AdjustCapacities(const std::vector<size_t>& new_capacities);

  // Returns overall and per-level cache hit stats.
  std::string PrintStats() const;
  void ResetStats();
  LevelMetricsSnapshot GetLevelMetricsSnapshot() const;
  // Total cache lookups summed across all levels. Cheap (16 stripes x levels
  // relaxed loads); used by the allocator's op-count adjustment cadence to gate
  // rounds without allocating a full metrics snapshot on every poll.
  uint64_t GetTotalLookups() const;
  // Foreground-only total lookups (compaction-induced lookups excluded). The
  // allocator's op-count cadence gates on THIS so the number of adjustment
  // rounds tracks foreground traffic, not compaction streaming -- consistent
  // with the compaction-excluded (config B) model, and so a write-heavy /
  // compaction-heavy workload does not fire far more rounds (and PurgeToCapacity
  // churn) than a read-only one at the same foreground op rate.
  uint64_t GetTotalForegroundLookups() const;
  std::vector<std::vector<uint64_t>> DrainLookupSamples();
  void SetLookupSampleRateLog2(uint32_t sample_rate_log2);

  // Foreground working-set tracking (Solution A: model-level fix for the
  // saturation-scale ill-conditioning). When enabled, every foreground Lookup
  // feeds a per-level HyperLogLog that estimates the count of DISTINCT blocks
  // the foreground touches, which the allocator uses as the MRC saturation
  // scale instead of the full on-disk data size. Off by default (zero hot-path
  // cost when disabled).
  // sample_shift>0 enables unbiased hash-gated sampling: only keys whose low
  // sample_shift hash bits are zero feed the sketch (1/2^shift of distinct
  // keys), and the drained estimate is scaled back up by 2^shift. Trades a
  // higher-variance distinct estimate for skipping the register write on the
  // (1-1/2^shift) unsampled foreground lookups.
  void SetWorkingSetTrackingEnabled(bool enabled, uint32_t sample_shift = 0);
  // Returns the per-level estimated distinct foreground block count observed
  // since the previous drain, and resets the sketches (windowed estimate).
  // Returns an empty vector when tracking is disabled.
  std::vector<double> DrainForegroundWorkingSetDistinct();

  // Ghost (repeat-miss) tracking: a direct, model-free measurement of each
  // level's marginal utility. Every foreground miss probes a per-level
  // fixed-size fingerprint table of recently-missed keys; a repeat miss on the
  // same key means the block was inserted and evicted (or bypassed) before its
  // re-access -- exactly the traffic that a small capacity increase would
  // convert into hits. The per-window ghost-hit count is therefore the
  // per-level marginal benefit signal for the incremental allocator, with no
  // MRC shape assumption (unlike the exponential-alpha inversion).
  // slots_log2 sizes the fingerprint table (1<<slots_log2 slots x 8B/level);
  // it bounds how far beyond current capacity the signal can "see".
  // segmented switches the slot layout from a plain 64-bit fingerprint to
  // (32-bit tag | 32-bit distinct-miss clock) and enables the reuse-distance
  // histogram below.
  void SetGhostTrackingEnabled(bool enabled, uint32_t slots_log2 = 16,
                               bool segmented = false);
  // Per-level ghost hits since the previous drain (read-and-reset windowed
  // counter). Empty when tracking is disabled.
  std::vector<uint64_t> DrainGhostHits();

  // Segmented ghost: reuse-distance histogram of repeat misses.
  //
  // Each ghost slot stores a 32-bit key tag plus the level's distinct-miss
  // clock at record time; on a repeat miss the clock delta is the number of
  // DISTINCT blocks that missed at this level in between -- a direct proxy
  // for how far beyond current capacity the block's stack distance lies. A
  // repeat miss at distance d would have been a hit with ~d more cached
  // blocks, so the histogram measures the CAPTURE RATE of a capacity step
  // directly, replacing the static per-byte denominators (raw / D /
  // uncached) that each encode a fixed prior about within-level reuse
  // concentration.
  //
  // Buckets are log2 of the distance in distinct missed blocks: bucket b
  // holds [2^b, 2^(b+1)). The miss clock is sampled 1-in-2^kGhostClockSampleShift
  // (see RecordGhostMiss), so bucket 0 aggregates all distances below one
  // clock tick (2^shift blocks) and buckets 1..shift-1 are never populated.
  // Returned flat as level * kGhostDistBuckets + bucket; read-and-reset.
  // Unlike DrainGhostHits this does NOT clear the fingerprint tables: the
  // timestamps make the reuse window uniform by construction (stale entries
  // just report long distances that fall in high buckets), so cross-window
  // visibility is a feature here. Empty when tracking is disabled.
  static constexpr size_t kGhostDistBuckets = 32;
  static constexpr uint32_t kGhostClockSampleShift = 6;
  std::vector<uint64_t> DrainGhostDistanceHistogram();

  // Installs the factory used by the sparse-table rebuild (swap a defunded
  // level's grow-only AutoHCC table for a fresh empty one). The HCC-options
  // constructor installs one automatically; callers of the per-level
  // (vector<Cache>) constructor MUST install one themselves or sparse-table
  // rebuild stays disabled and a defunded level's Evict sweeps degrade.
  void SetRebuildSubCacheFactory(
      std::function<std::shared_ptr<Cache>(size_t level, size_t new_capacity)>
          factory) {
    rebuild_sub_cache_ = std::move(factory);
  }

  // Replaces per-level data sizes used by allocator D_i metric.
  void UpdateLevelDataSizes(const std::vector<uint64_t>& level_data_sizes);
  // For A/B diagnostics: force all requests to route into L0.
  // When enabled, initial capacities are also switched to L0-only so the
  // setup is closer to single-cache baseline behavior.
  void SetForceRouteAllToL0(bool force_route_all_to_l0);
  // Capacity-gated insertion. When min_capacity_bytes > 0, an Insert routed to
  // a sub-cache whose current capacity is below this threshold bypasses the
  // hash-table insert and instead returns a standalone entry (CreateStandalone,
  // uncharged). A level the allocator has starved below this floor caches
  // nothing durably -- every insert is immediately evicted -- so a real insert
  // is pure churn (insert -> evict -> cfree over a grow-only sparse AutoHCC
  // table) with ~0 hit-ratio benefit. The standalone path never populates the
  // table, so the deep-level Evict sweep that dominates its CPU disappears while
  // the caller still gets a usable pinned block. 0 disables the gate (default).
  void SetInsertBypassCapacity(size_t min_capacity_bytes);
  void SetSharedPoolRatio(double shared_pool_ratio);
  void SetSharedPoolAdmissionThreshold(uint32_t admission_threshold);
  void SetSharedPoolDecayIntervalOps(uint32_t decay_interval_ops);
  void ConfigureDynamicSRHCC(bool enabled, uint32_t check_interval_ops = 4096,
                             uint32_t min_samples = 64,
                             double unique_ratio_enable_threshold = 0.50,
                             double unique_ratio_disable_threshold = 0.30,
                             uint32_t sample_rate_log2 = 7,
                             uint32_t poll_interval_ms = 200);

 private:
  enum class RouteCaller {
    kInsert,
    kLookup,
    kOther,
  };

  struct WrappedHandle : public Handle {
    Cache* owner_cache = nullptr;
    Cache::Handle* inner = nullptr;
  };

  // Maps a handle address to the sub-cache that issued it. HCC sub-caches
  // hand out pointers into their (lifetime-stable) slot arrays, so the owner
  // can be recovered by address and handles can be passed through unwrapped,
  // avoiding a heap-allocated WrappedHandle per hit. Handles that fall
  // outside all ranges (standalone entries, non-HCC sub-caches) still use a
  // WrappedHandle, distinguished by tagging the returned pointer's low bit.
  struct HandleOwnerRange {
    uintptr_t begin;
    uintptr_t end;
    Cache* owner;
  };

  struct SharedAdmissionShard {
    std::mutex mutex;
    std::unordered_map<uint64_t, uint32_t> miss_scores;
    uint32_t ops_since_decay = 0;
  };

  struct LevelSampleRing {
    std::unique_ptr<std::atomic<uint64_t>[]> seq;
    std::unique_ptr<std::atomic<uint64_t>[]> values;
    std::atomic<uint64_t> write_seq{0};
    std::atomic<uint64_t> consumed_seq{0};
    std::atomic<uint64_t> drained_seq{0};
  };

  // Per-level HyperLogLog for the foreground working-set (distinct block)
  // estimate. Registers hold the max observed rank and are updated with relaxed
  // atomic max, so concurrent Adds need no locking. Drain reads+resets each
  // register via exchange; the (rare) race between a concurrent Add and the
  // drain only perturbs an estimate, which is acceptable. 2^11 = 2048 registers
  // gives ~2.3% standard error at ~2KB/level.
  static constexpr uint32_t kWssRegisterBitsLog2 = 11;
  static constexpr size_t kWssRegisterCount = size_t{1} << kWssRegisterBitsLog2;
  struct ForegroundWorkingSetSketch {
    std::unique_ptr<std::atomic<uint8_t>[]> registers;
    ForegroundWorkingSetSketch()
        : registers(new std::atomic<uint8_t>[kWssRegisterCount]) {
      for (size_t i = 0; i < kWssRegisterCount; ++i) {
        registers[i].store(0, std::memory_order_relaxed);
      }
    }
  };

  Cache* SubCacheByLevel(size_t level_index);
  const Cache* SubCacheByLevel(size_t level_index) const;
  Cache* PrimarySubCache();
  const Cache* PrimarySubCache() const;
  Cache* SharedCache();
  const Cache* SharedCache() const;

  size_t RouteLevelByKey(const Slice& key, RouteCaller caller,
                         Slice* base_key) const;
  bool DecodeExtendedCacheRouting(const Slice& key, size_t* level,
                                  Slice* base_key) const;
  std::optional<uint64_t> GetCacheKeyPrefix(const Slice& key) const;
  void MaybeLogRouteMiss(RouteCaller caller, const Slice& key, bool has_prefix,
                         uint64_t key_prefix, const char* reason) const;
  static const char* RouteCallerToString(RouteCaller caller);
  static int64_t ParseDebugMissLimit();
  void MaybeRecordLookupSample(size_t level_index, const Slice& key);
  void RecordForegroundWorkingSet(size_t level_index, const Slice& base_key);
  void MaybeAdaptLevelMode(size_t level_index);
  bool MaybeSetLevelProbationInsert(size_t level_index, bool probation_insert);
  void StartDynamicSRHCCWorker();
  void StopDynamicSRHCCWorker();
  void DynamicSRHCCBackgroundLoop();
  void EvaluateDynamicSRHCCAllLevels();
  uint64_t HashCacheKey(const Slice& key) const;
  void RecordSharedPoolCandidate(uint64_t key_hash);
  bool IsSharedPoolAdmissionReady(uint64_t key_hash);
  void ClearSharedPoolAdmission(uint64_t key_hash);
  void MaybeDecaySharedAdmissionShard(SharedAdmissionShard* shard);
  void TrimSharedAdmissionShardIfNeeded(SharedAdmissionShard* shard);
  size_t GetSharedPoolCapacity(size_t total_capacity) const;

  void BuildHandleOwnerRanges();
  // Rebuilds and atomically publishes the handle-owner range snapshot from the
  // current sub-caches, the shared cache, and all not-yet-reclaimed retired
  // sub-caches. Must be called under rebuild_mutex_.
  void RebuildHandleOwnerRangesLocked();
  // After capacities are applied, swaps any sub-cache that has become sparse
  // (grown-large but now nearly empty, the AutoHCC grow-only pathology) for a
  // fresh empty one, and reclaims retired caches whose grace period elapsed and
  // whose handles have all been released. Runs on the allocator thread.
  void MaybeRebuildSparseSubCaches();
  void ReclaimRetiredCachesLocked();
  Cache* FindHandleOwner(const void* handle_addr) const;
  Handle* WrapOrPassHandle(Cache* owner_cache, Cache::Handle* inner);
  // Recovers (owner, inner) for any handle previously returned by this cache.
  void ResolveHandle(const Handle* handle, Cache** owner,
                     Cache::Handle** inner) const;

  Status ValidateCapacities(const std::vector<size_t>& capacities) const;
  void ApplyCapacities(const std::vector<size_t>& capacities);
  void InitializePerLevelState(size_t level_count);

  // Per-level lookup/hit counters, striped to avoid a single hot cache line:
  // one level typically receives ~90% of traffic (L6 after full compaction),
  // and a single atomic pair would be contended by all client threads.
  // Indexed [stripe * level_count + level]; readers sum across stripes.
  static constexpr size_t kCounterStripes = 16;
  struct alignas(64) StripedCounter {
    std::atomic<uint64_t> v{0};
  };

  void IncLookupCounter(size_t level_index);
  void IncHitCounter(size_t level_index);
  uint64_t SumLookupCounter(size_t level_index) const;
  uint64_t SumHitCounter(size_t level_index) const;
  void IncFgLookupCounter(size_t level_index);
  void IncFgHitCounter(size_t level_index);
  uint64_t SumFgLookupCounter(size_t level_index) const;
  uint64_t SumFgHitCounter(size_t level_index) const;
  void RecordGhostMiss(size_t level_index, const Slice& base_key);
  void IncGhostHitCounter(size_t level_index);

  // Owning vector of the current sub-cache per level. Elements are only
  // reassigned during a sparse-table rebuild, always under rebuild_mutex_.
  // Never dereference an element off the allocator/rebuild thread; readers on
  // other threads must route through current_ptr_ (below) instead.
  std::vector<std::shared_ptr<Cache>> sub_caches_;
  // Lock-free hot-path routing: current_ptr_[i] is the raw pointer of the cache
  // currently serving level i. Published with release ordering on rebuild and
  // read with acquire ordering by SubCacheByLevel and the low-frequency
  // iteration helpers. The pointee is kept alive by sub_caches_ (current) or
  // retired_sub_caches_ (retired, grace-period reclaimed), so a raw pointer read
  // here stays valid for the microsecond-scoped duration of any single call.
  std::unique_ptr<std::atomic<Cache*>[]> current_ptr_;
  std::shared_ptr<Cache> shared_cache_;
  // Builds a fresh empty sub-cache for a given level and target capacity, with
  // the mmap reservation sized for the whole budget (so it can grow again
  // later). Set by the HCC-backed constructor, or via
  // SetRebuildSubCacheFactory for the per-level (vector<Cache>) construction
  // path; null disables sparse-table rebuild (e.g. LRU sub-caches, which
  // shrink in place and never need it).
  std::function<std::shared_ptr<Cache>(size_t level, size_t new_capacity)>
      rebuild_sub_cache_;
  // Serializes sub_caches_ element reassignment and retired_sub_caches_ /
  // range-snapshot bookkeeping against the low-frequency iteration helpers.
  // Never held on the Lookup/Insert/Release hot path.
  mutable std::mutex rebuild_mutex_;
  // A sub-cache displaced by a rebuild, kept alive until all its outstanding
  // handles are released and a grace period has elapsed, then destroyed (which
  // munmaps its slot array, returning RSS to the OS).
  struct RetiredCache {
    std::shared_ptr<Cache> cache;
    uint64_t retire_round;
  };
  std::vector<RetiredCache> retired_sub_caches_;
  // Monotonic round counter driving retire/reclaim grace periods; advanced once
  // per ApplyCapacities under rebuild_mutex_.
  uint64_t rebuild_round_ = 0;
  // Sorted by begin. Published atomically; rebuilt when a sub-cache is swapped
  // or a retired cache is reclaimed (so freed address ranges are dropped).
  // Readers load with acquire ordering; the pointee is owned by live_ranges_.
  std::atomic<const std::vector<HandleOwnerRange>*> handle_owner_ranges_{nullptr};
  std::unique_ptr<const std::vector<HandleOwnerRange>> live_ranges_;
  // Superseded range snapshots awaiting grace-period reclaim (readers that
  // loaded the old pointer finish within microseconds).
  std::vector<std::pair<uint64_t, std::unique_ptr<const std::vector<HandleOwnerRange>>>>
      retired_ranges_;
  std::unique_ptr<StripedCounter[]> lookups_;
  std::unique_ptr<StripedCounter[]> hits_;
  std::unique_ptr<StripedCounter[]> fg_lookups_;
  std::unique_ptr<StripedCounter[]> fg_hits_;
  std::deque<std::atomic<uint64_t>> level_data_sizes_;
  // False until someone actually consumes lookup samples (dynamic SR-HCC
  // worker or an external driver via SetLookupSampleRateLog2). Lets the
  // lookup hot path skip the globally contended sample sequence atomic
  // when nobody is listening.
  std::atomic<bool> lookup_sampling_enabled_{false};
  std::atomic<uint32_t> lookup_sample_rate_log2_{10};
  // Large enough to absorb full-sampling bursts under high concurrency.
  static constexpr size_t kLookupSampleRingSize = 65536;
  std::vector<std::unique_ptr<LevelSampleRing>> lookup_sample_rings_;
  // Solution A working-set tracking. Off unless the allocator's saturation
  // scale is driven by the observed foreground footprint.
  std::atomic<bool> working_set_tracking_enabled_{false};
  std::atomic<uint32_t> working_set_sample_shift_{0};
  std::vector<std::unique_ptr<ForegroundWorkingSetSketch>> fg_working_set_;
  // Ghost (repeat-miss) tracking state. ghost_tables_[level] is a direct-
  // mapped table of 64-bit key-hash fingerprints of recently-missed keys
  // (0 = empty slot). Probed and updated with relaxed atomics on the
  // foreground miss path only; collisions simply overwrite (bounded window).
  // ghost_hits_ uses the same striping as the other hot-path counters.
  // In segmented mode a slot is (32-bit key tag << 32 | 32-bit distinct-miss
  // clock) instead of the full 64-bit fingerprint.
  std::atomic<bool> ghost_tracking_enabled_{false};
  std::atomic<bool> ghost_segmented_{false};
  std::atomic<uint32_t> ghost_slots_log2_{16};
  std::vector<std::unique_ptr<std::atomic<uint64_t>[]>> ghost_tables_;
  std::unique_ptr<StripedCounter[]> ghost_hits_;
  // Per-level distinct-miss clock (segmented mode): incremented once per
  // non-repeat foreground miss; the delta between two misses of the same key
  // is its reuse distance in distinct missed blocks. One cache line per
  // level (StripedCounter is alignas(64)), indexed by level directly.
  std::unique_ptr<StripedCounter[]> ghost_miss_clock_;
  // Reuse-distance histogram, [level * kGhostDistBuckets + log2bucket].
  // Repeat misses are ~two orders of magnitude rarer than lookups, so plain
  // (unstriped) relaxed counters are sufficient.
  std::unique_ptr<std::atomic<uint64_t>[]> ghost_dist_hist_;
  mutable std::atomic<uint64_t> insert_route_queries_{0};
  mutable std::atomic<uint64_t> insert_route_parse_failures_{0};
  mutable std::atomic<uint64_t> insert_route_prefix_hits_{0};
  mutable std::atomic<uint64_t> insert_route_prefix_misses_{0};
  mutable std::atomic<uint64_t> lookup_route_queries_{0};
  mutable std::atomic<uint64_t> lookup_route_parse_failures_{0};
  mutable std::atomic<uint64_t> lookup_route_prefix_hits_{0};
  mutable std::atomic<uint64_t> lookup_route_prefix_misses_{0};
  mutable std::atomic<uint64_t> route_normalize_fallbacks_{0};
  mutable std::atomic<int64_t> debug_miss_budget_{0};
  std::atomic<uint64_t> shared_pool_lookups_{0};
  std::atomic<uint64_t> shared_pool_hits_{0};
  std::atomic<uint64_t> shared_pool_admissions_{0};
  static constexpr size_t kSharedAdmissionShardCount = 64;
  std::array<SharedAdmissionShard, kSharedAdmissionShardCount>
      shared_admission_shards_;
  std::atomic<uint32_t> shared_pool_ratio_ppm_{0};
  std::atomic<uint32_t> shared_pool_admission_threshold_{2};
  std::atomic<uint32_t> shared_pool_decay_interval_ops_{2048};
  std::atomic<bool> force_route_all_to_l0_{false};
  // Capacity-gated insertion threshold (bytes). 0 = disabled. See
  // SetInsertBypassCapacity.
  std::atomic<size_t> insert_bypass_capacity_bytes_{0};
  std::atomic<bool> dynamic_srhcc_enabled_{false};
  std::atomic<uint32_t> dynamic_srhcc_check_interval_ops_{4096};
  std::atomic<uint32_t> dynamic_srhcc_min_samples_{64};
  std::atomic<uint32_t> dynamic_srhcc_sample_rate_log2_{7};
  std::atomic<uint32_t> dynamic_srhcc_poll_interval_ms_{200};
  std::atomic<uint32_t> dynamic_srhcc_unique_ratio_enable_ppm_{500000};
  std::atomic<uint32_t> dynamic_srhcc_unique_ratio_disable_ppm_{300000};
  std::deque<std::atomic<uint64_t>> adapt_last_lookups_;
  std::deque<std::atomic<uint64_t>> adapt_last_hits_;
  std::deque<std::atomic<bool>> level_probation_insert_enabled_;
  std::atomic<bool> dynamic_srhcc_running_{false};
  std::mutex dynamic_srhcc_mu_;
  std::condition_variable dynamic_srhcc_cv_;
  std::thread dynamic_srhcc_worker_;
  std::atomic<size_t> total_capacity_;
};

}  // namespace ROCKSDB_NAMESPACE
