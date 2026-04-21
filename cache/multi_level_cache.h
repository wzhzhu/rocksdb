#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "rocksdb/advanced_cache.h"
#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

// A cache wrapper that maintains one sub-cache per LSM-tree level.
// Routing is based on cache key prefix -> level mapping.
class MultiLevelCache : public Cache {
 public:
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

  // Thread-safe mapping update from SST file number to LSM-tree level.
  void UpdateFileLevelMapping(uint64_t file_number, int level);
  void RemoveFileLevelMapping(uint64_t file_number);
  // Thread-safe mapping update from cache key common prefix to LSM-tree level.
  void UpdateCacheKeyPrefixMapping(uint64_t cache_key_prefix, int level);
  void RemoveCacheKeyPrefixMapping(uint64_t cache_key_prefix);

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

  struct MappingShard {
    mutable std::shared_mutex mutex;
    std::unordered_map<uint64_t, int> map;
  };

  static constexpr size_t kMappingShardCount = 64;

  Cache* SubCacheByLevel(size_t level_index);
  const Cache* SubCacheByLevel(size_t level_index) const;
  Cache* PrimarySubCache();
  const Cache* PrimarySubCache() const;

  size_t RouteLevelByKey(const Slice& key, RouteCaller caller) const;
  std::optional<uint64_t> GetCacheKeyPrefix(const Slice& key) const;
  std::optional<int> FindLevelByCacheKeyPrefix(uint64_t cache_key_prefix) const;
  std::optional<int> FindLevelByFileNumber(uint64_t file_number) const;

  WrappedHandle* NewWrappedHandle(size_t level_index, Cache::Handle* inner);
  static WrappedHandle* ToWrappedHandle(Handle* handle);
  static const WrappedHandle* ToWrappedHandle(const Handle* handle);

  size_t NormalizeLevel(int level) const;
  uint64_t CountMappingEntries(
      const std::array<MappingShard, kMappingShardCount>& shards) const;
  Status ValidateCapacities(const std::vector<size_t>& capacities) const;
  void ApplyCapacities(const std::vector<size_t>& capacities);

  std::vector<std::shared_ptr<Cache>> sub_caches_;
  std::deque<std::atomic<uint64_t>> lookups_;
  std::deque<std::atomic<uint64_t>> hits_;
  mutable std::atomic<uint64_t> insert_route_queries_{0};
  mutable std::atomic<uint64_t> insert_route_parse_failures_{0};
  mutable std::atomic<uint64_t> insert_route_prefix_hits_{0};
  mutable std::atomic<uint64_t> insert_route_prefix_misses_{0};
  mutable std::atomic<uint64_t> lookup_route_queries_{0};
  mutable std::atomic<uint64_t> lookup_route_parse_failures_{0};
  mutable std::atomic<uint64_t> lookup_route_prefix_hits_{0};
  mutable std::atomic<uint64_t> lookup_route_prefix_misses_{0};
  mutable std::atomic<uint64_t> route_normalize_fallbacks_{0};
  std::atomic<size_t> total_capacity_;
  std::array<MappingShard, kMappingShardCount> file_number_to_level_;
  std::array<MappingShard, kMappingShardCount> cache_key_prefix_to_level_;
};

}  // namespace ROCKSDB_NAMESPACE
