#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rocksdb/advanced_cache.h"
#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

// A cache wrapper that maintains one sub-cache per LSM-tree level.
// Routing decodes level directly from cache key prefix.
class MultiLevelCache : public Cache {
 public:
  struct LevelMetricsSnapshot {
    std::vector<uint64_t> lookups;
    std::vector<uint64_t> hits;
    std::vector<size_t> capacities;
    std::vector<uint64_t> data_sizes;
  };

  static const char* kClassName() { return "MultiLevelCache"; }

  MultiLevelCache(size_t num_levels, size_t total_capacity);

  const char* Name() const override;

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

  // Replaces per-level data sizes used by allocator D_i metric.
  void UpdateLevelDataSizes(const std::vector<uint64_t>& level_data_sizes);

 private:
  enum class RouteCaller {
    kInsert,
    kLookup,
    kOther,
  };

  struct WrappedHandle : public Handle {
    size_t level_index = 0;
    Cache::Handle* inner = nullptr;
  };

  Cache* SubCacheByLevel(size_t level_index);
  const Cache* SubCacheByLevel(size_t level_index) const;
  Cache* PrimarySubCache();
  const Cache* PrimarySubCache() const;

  size_t RouteLevelByKey(const Slice& key, RouteCaller caller) const;
  std::optional<uint64_t> GetCacheKeyPrefix(const Slice& key) const;
  void MaybeLogRouteMiss(RouteCaller caller, const Slice& key, bool has_prefix,
                         uint64_t key_prefix, const char* reason) const;
  static const char* RouteCallerToString(RouteCaller caller);
  static int64_t ParseDebugMissLimit();

  WrappedHandle* NewWrappedHandle(size_t level_index, Cache::Handle* inner);
  static WrappedHandle* ToWrappedHandle(Handle* handle);
  static const WrappedHandle* ToWrappedHandle(const Handle* handle);

  Status ValidateCapacities(const std::vector<size_t>& capacities) const;
  void ApplyCapacities(const std::vector<size_t>& capacities);

  std::vector<std::shared_ptr<Cache>> sub_caches_;
  std::deque<std::atomic<uint64_t>> lookups_;
  std::deque<std::atomic<uint64_t>> hits_;
  std::deque<std::atomic<uint64_t>> level_data_sizes_;
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
  std::atomic<size_t> total_capacity_;
};

}  // namespace ROCKSDB_NAMESPACE
