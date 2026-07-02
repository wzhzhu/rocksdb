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

  // Replaces per-level data sizes used by allocator D_i metric.
  void UpdateLevelDataSizes(const std::vector<uint64_t>& level_data_sizes);
  // For A/B diagnostics: force all requests to route into L0.
  // When enabled, initial capacities are also switched to L0-only so the
  // setup is closer to single-cache baseline behavior.
  void SetForceRouteAllToL0(bool force_route_all_to_l0);
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

  std::vector<std::shared_ptr<Cache>> sub_caches_;
  std::shared_ptr<Cache> shared_cache_;
  // Sorted by begin; built once after sub-caches are created, immutable
  // afterwards (HCC slot arrays are never reallocated).
  std::vector<HandleOwnerRange> handle_owner_ranges_;
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
