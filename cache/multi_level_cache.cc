#include "cache/multi_level_cache.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <utility>

#include "util/coding.h"
namespace ROCKSDB_NAMESPACE {

namespace {

size_t SafeLevelCount(size_t num_levels) {
  return std::max<size_t>(num_levels, 1);
}

size_t ShardIndex(uint64_t file_number) {
  return file_number % 64;
}

}  // namespace

MultiLevelCache::MultiLevelCache(size_t num_levels, size_t total_capacity)
    : total_capacity_(total_capacity) {
  const size_t level_count = SafeLevelCount(num_levels);
  sub_caches_.reserve(level_count);

  const size_t per_level_capacity = total_capacity / level_count;
  const size_t remainder = total_capacity % level_count;
  for (size_t level = 0; level < level_count; ++level) {
    const size_t level_capacity =
        per_level_capacity + (level < remainder ? 1 : 0);
    sub_caches_.emplace_back(NewLRUCache(level_capacity));
  }

  lookups_.resize(level_count);
  hits_.resize(level_count);
  level_data_sizes_.resize(level_count);
  for (size_t level = 0; level < level_count; ++level) {
    lookups_[level].store(0, std::memory_order_relaxed);
    hits_[level].store(0, std::memory_order_relaxed);
    level_data_sizes_[level].store(0, std::memory_order_relaxed);
  }
}

const char* MultiLevelCache::Name() const { return "MultiLevelCache"; }

Status MultiLevelCache::Insert(const Slice& key, ObjectPtr obj,
                               const CacheItemHelper* helper, size_t charge,
                               Handle** handle, Priority priority,
                               const Slice& compressed, CompressionType type) {
  const size_t level_index = RouteLevelByKey(key, RouteCaller::kInsert);
  Cache::Handle* inner = nullptr;
  Cache::Handle** inner_handle = handle != nullptr ? &inner : nullptr;
  Status s = SubCacheByLevel(level_index)->Insert(key, obj, helper, charge,
                                                  inner_handle, priority,
                                                  compressed, type);
  if (!s.ok()) {
    return s;
  }
  if (handle != nullptr && inner != nullptr) {
    *handle = NewWrappedHandle(level_index, inner);
  }
  return s;
}

Cache::Handle* MultiLevelCache::CreateStandalone(
    const Slice& key, ObjectPtr obj, const CacheItemHelper* helper,
    size_t charge, bool allow_uncharged) {
  const size_t level_index = RouteLevelByKey(key, RouteCaller::kOther);
  Cache::Handle* inner = SubCacheByLevel(level_index)->CreateStandalone(
      key, obj, helper, charge, allow_uncharged);
  if (inner == nullptr) {
    return nullptr;
  }
  return NewWrappedHandle(level_index, inner);
}

Cache::Handle* MultiLevelCache::Lookup(const Slice& key,
                                       const CacheItemHelper* helper,
                                       CreateContext* create_context,
                                       Priority priority, Statistics* stats) {
  const size_t level_index = RouteLevelByKey(key, RouteCaller::kLookup);
  lookups_[level_index].fetch_add(1, std::memory_order_relaxed);
  Cache::Handle* inner = SubCacheByLevel(level_index)->Lookup(
      key, helper, create_context, priority, stats);
  if (inner == nullptr) {
    return nullptr;
  }
  hits_[level_index].fetch_add(1, std::memory_order_relaxed);
  return NewWrappedHandle(level_index, inner);
}

bool MultiLevelCache::Ref(Handle* handle) {
  WrappedHandle* wrapped = ToWrappedHandle(handle);
  return SubCacheByLevel(wrapped->level_index)->Ref(wrapped->inner);
}

bool MultiLevelCache::Release(Handle* handle, bool erase_if_last_ref) {
  WrappedHandle* wrapped = ToWrappedHandle(handle);
  const bool erased = SubCacheByLevel(wrapped->level_index)
                          ->Release(wrapped->inner, erase_if_last_ref);
  delete wrapped;
  return erased;
}

Cache::ObjectPtr MultiLevelCache::Value(Handle* handle) {
  WrappedHandle* wrapped = ToWrappedHandle(handle);
  return SubCacheByLevel(wrapped->level_index)->Value(wrapped->inner);
}

void MultiLevelCache::Erase(const Slice& key) {
  SubCacheByLevel(RouteLevelByKey(key, RouteCaller::kOther))->Erase(key);
}

uint64_t MultiLevelCache::NewId() { return PrimarySubCache()->NewId(); }

void MultiLevelCache::SetCapacity(size_t capacity) {
  total_capacity_.store(capacity, std::memory_order_relaxed);
  const size_t level_count = sub_caches_.size();
  const size_t per_level_capacity = capacity / level_count;
  const size_t remainder = capacity % level_count;
  std::vector<size_t> capacities;
  capacities.reserve(level_count);
  for (size_t level = 0; level < level_count; ++level) {
    const size_t level_capacity =
        per_level_capacity + (level < remainder ? 1 : 0);
    capacities.push_back(level_capacity);
  }
  ApplyCapacities(capacities);
}

void MultiLevelCache::SetStrictCapacityLimit(bool strict_capacity_limit) {
  for (const auto& sub_cache : sub_caches_) {
    sub_cache->SetStrictCapacityLimit(strict_capacity_limit);
  }
}

bool MultiLevelCache::HasStrictCapacityLimit() const {
  return PrimarySubCache()->HasStrictCapacityLimit();
}

size_t MultiLevelCache::GetCapacity() const {
  size_t total = 0;
  for (const auto& sub_cache : sub_caches_) {
    total += sub_cache->GetCapacity();
  }
  return total;
}

size_t MultiLevelCache::GetUsage() const {
  size_t total = 0;
  for (const auto& sub_cache : sub_caches_) {
    total += sub_cache->GetUsage();
  }
  return total;
}

size_t MultiLevelCache::GetUsage(Handle* handle) const {
  const WrappedHandle* wrapped = ToWrappedHandle(handle);
  return SubCacheByLevel(wrapped->level_index)->GetUsage(wrapped->inner);
}

size_t MultiLevelCache::GetPinnedUsage() const {
  size_t total = 0;
  for (const auto& sub_cache : sub_caches_) {
    total += sub_cache->GetPinnedUsage();
  }
  return total;
}

size_t MultiLevelCache::GetCharge(Handle* handle) const {
  const WrappedHandle* wrapped = ToWrappedHandle(handle);
  return SubCacheByLevel(wrapped->level_index)->GetCharge(wrapped->inner);
}

const Cache::CacheItemHelper* MultiLevelCache::GetCacheItemHelper(
    Handle* handle) const {
  const WrappedHandle* wrapped = ToWrappedHandle(handle);
  return SubCacheByLevel(wrapped->level_index)
      ->GetCacheItemHelper(wrapped->inner);
}

void MultiLevelCache::ApplyToAllEntries(
    const std::function<void(const Slice& key, ObjectPtr obj, size_t charge,
                             const CacheItemHelper* helper)>& callback,
    const ApplyToAllEntriesOptions& opts) {
  for (const auto& sub_cache : sub_caches_) {
    sub_cache->ApplyToAllEntries(callback, opts);
  }
}

void MultiLevelCache::ApplyToHandle(
    Cache* /*cache*/, Handle* handle,
    const std::function<void(const Slice& key, ObjectPtr obj, size_t charge,
                             const CacheItemHelper* helper)>& callback) {
  WrappedHandle* wrapped = ToWrappedHandle(handle);
  Cache* routed = SubCacheByLevel(wrapped->level_index);
  routed->ApplyToHandle(routed, wrapped->inner, callback);
}

void MultiLevelCache::EraseUnRefEntries() {
  for (const auto& sub_cache : sub_caches_) {
    sub_cache->EraseUnRefEntries();
  }
}

size_t MultiLevelCache::GetOccupancyCount() const {
  size_t total = 0;
  for (const auto& sub_cache : sub_caches_) {
    const size_t count = sub_cache->GetOccupancyCount();
    if (count == SIZE_MAX) {
      return SIZE_MAX;
    }
    total += count;
  }
  return total;
}

size_t MultiLevelCache::GetTableAddressCount() const {
  size_t total = 0;
  for (const auto& sub_cache : sub_caches_) {
    total += sub_cache->GetTableAddressCount();
  }
  return total;
}

Status MultiLevelCache::AdjustCapacities(
    const std::vector<size_t>& new_capacities) {
  Status validation = ValidateCapacities(new_capacities);
  if (!validation.ok()) {
    return validation;
  }
  ApplyCapacities(new_capacities);
  return Status::OK();
}

std::string MultiLevelCache::PrintStats() const {
  uint64_t total_lookups = 0;
  uint64_t total_hits = 0;
  for (size_t level = 0; level < sub_caches_.size(); ++level) {
    total_lookups += lookups_[level].load(std::memory_order_relaxed);
    total_hits += hits_[level].load(std::memory_order_relaxed);
  }

  const double total_hit_rate =
      total_lookups == 0
          ? 0.0
          : static_cast<double>(total_hits) / static_cast<double>(total_lookups);

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4);
  const uint64_t insert_queries =
      insert_route_queries_.load(std::memory_order_relaxed);
  const uint64_t insert_parse_failures =
      insert_route_parse_failures_.load(std::memory_order_relaxed);
  const uint64_t insert_prefix_hits =
      insert_route_prefix_hits_.load(std::memory_order_relaxed);
  const uint64_t insert_prefix_misses =
      insert_route_prefix_misses_.load(std::memory_order_relaxed);
  const uint64_t insert_known_prefix =
      insert_prefix_hits + insert_prefix_misses;
  const double insert_prefix_hit_rate =
      insert_known_prefix == 0
          ? 0.0
          : static_cast<double>(insert_prefix_hits) /
                static_cast<double>(insert_known_prefix);

  const uint64_t lookup_queries =
      lookup_route_queries_.load(std::memory_order_relaxed);
  const uint64_t lookup_parse_failures =
      lookup_route_parse_failures_.load(std::memory_order_relaxed);
  const uint64_t lookup_prefix_hits =
      lookup_route_prefix_hits_.load(std::memory_order_relaxed);
  const uint64_t lookup_prefix_misses =
      lookup_route_prefix_misses_.load(std::memory_order_relaxed);
  const uint64_t lookup_known_prefix =
      lookup_prefix_hits + lookup_prefix_misses;
  const double lookup_prefix_hit_rate =
      lookup_known_prefix == 0
          ? 0.0
          : static_cast<double>(lookup_prefix_hits) /
                static_cast<double>(lookup_known_prefix);

  const uint64_t route_normalize_fallbacks =
      route_normalize_fallbacks_.load(std::memory_order_relaxed);
  const uint64_t file_mapping_entries =
      CountMappingEntries(file_number_to_level_);
  const uint64_t prefix_mapping_entries =
      CountMappingEntries(cache_key_prefix_to_level_);
  oss << "route_insert: queries=" << insert_queries
      << ", parse_failures=" << insert_parse_failures
      << ", prefix_hits=" << insert_prefix_hits
      << ", prefix_misses=" << insert_prefix_misses
      << ", prefix_hit_rate=" << insert_prefix_hit_rate << "\n";
  oss << "route_lookup: queries=" << lookup_queries
      << ", parse_failures=" << lookup_parse_failures
      << ", prefix_hits=" << lookup_prefix_hits
      << ", prefix_misses=" << lookup_prefix_misses
      << ", prefix_hit_rate=" << lookup_prefix_hit_rate << "\n";
  oss << "route_normalize_fallbacks=" << route_normalize_fallbacks << "\n";
  oss << "mapping_entries: file_number=" << file_mapping_entries
      << ", cache_key_prefix=" << prefix_mapping_entries << "\n";
  oss << "total_hit_rate=" << total_hit_rate << " (" << total_hits << "/"
      << total_lookups << ")\n";
  for (size_t level = 0; level < sub_caches_.size(); ++level) {
    const uint64_t level_lookups =
        lookups_[level].load(std::memory_order_relaxed);
    const uint64_t level_hits = hits_[level].load(std::memory_order_relaxed);
    const uint64_t level_data_size =
        level_data_sizes_[level].load(std::memory_order_relaxed);
    const double level_hit_rate =
        level_lookups == 0 ? 0.0
                           : static_cast<double>(level_hits) /
                                 static_cast<double>(level_lookups);
    oss << "L" << level << ": capacity=" << sub_caches_[level]->GetCapacity()
        << ", lookups=" << level_lookups << ", hits=" << level_hits
        << ", hit_rate=" << level_hit_rate
        << ", data_size=" << level_data_size << "\n";
  }
  return oss.str();
}

void MultiLevelCache::ResetStats() {
  for (size_t level = 0; level < sub_caches_.size(); ++level) {
    lookups_[level].store(0, std::memory_order_relaxed);
    hits_[level].store(0, std::memory_order_relaxed);
  }
  insert_route_queries_.store(0, std::memory_order_relaxed);
  insert_route_parse_failures_.store(0, std::memory_order_relaxed);
  insert_route_prefix_hits_.store(0, std::memory_order_relaxed);
  insert_route_prefix_misses_.store(0, std::memory_order_relaxed);
  lookup_route_queries_.store(0, std::memory_order_relaxed);
  lookup_route_parse_failures_.store(0, std::memory_order_relaxed);
  lookup_route_prefix_hits_.store(0, std::memory_order_relaxed);
  lookup_route_prefix_misses_.store(0, std::memory_order_relaxed);
  route_normalize_fallbacks_.store(0, std::memory_order_relaxed);
}

MultiLevelCache::LevelMetricsSnapshot MultiLevelCache::GetLevelMetricsSnapshot()
    const {
  LevelMetricsSnapshot snapshot;
  snapshot.lookups.resize(sub_caches_.size());
  snapshot.hits.resize(sub_caches_.size());
  snapshot.capacities.resize(sub_caches_.size());
  snapshot.data_sizes.resize(sub_caches_.size());
  for (size_t level = 0; level < sub_caches_.size(); ++level) {
    snapshot.lookups[level] = lookups_[level].load(std::memory_order_relaxed);
    snapshot.hits[level] = hits_[level].load(std::memory_order_relaxed);
    snapshot.capacities[level] = sub_caches_[level]->GetCapacity();
    snapshot.data_sizes[level] =
        level_data_sizes_[level].load(std::memory_order_relaxed);
  }
  return snapshot;
}

void MultiLevelCache::UpdateFileLevelMapping(uint64_t file_number, int level) {
  const size_t shard_index = ShardIndex(file_number);
  MappingShard& shard = file_number_to_level_[shard_index];
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  shard.map[file_number] = level;
}

void MultiLevelCache::UpdateFileMetadata(uint64_t file_number, int level,
                                         uint64_t file_size) {
  const size_t normalized_level = NormalizeLevel(level);
  const size_t shard_index = ShardIndex(file_number);
  MappingShard& level_shard = file_number_to_level_[shard_index];
  SizeShard& size_shard = file_number_to_size_[shard_index];
  std::unique_lock<std::shared_mutex> level_lock(level_shard.mutex);
  std::unique_lock<std::shared_mutex> size_lock(size_shard.mutex);

  size_t old_level = normalized_level;
  uint64_t old_size = 0;
  auto level_it = level_shard.map.find(file_number);
  if (level_it != level_shard.map.end()) {
    old_level = NormalizeLevel(level_it->second);
  }
  auto size_it = size_shard.map.find(file_number);
  if (size_it != size_shard.map.end()) {
    old_size = size_it->second;
  }

  if (old_size > 0) {
    level_data_sizes_[old_level].fetch_sub(old_size, std::memory_order_relaxed);
  }
  level_data_sizes_[normalized_level].fetch_add(file_size,
                                                std::memory_order_relaxed);
  level_shard.map[file_number] = level;
  size_shard.map[file_number] = file_size;
}

void MultiLevelCache::RemoveFileLevelMapping(uint64_t file_number) {
  const size_t shard_index = ShardIndex(file_number);
  MappingShard& level_shard = file_number_to_level_[shard_index];
  SizeShard& size_shard = file_number_to_size_[shard_index];
  std::unique_lock<std::shared_mutex> level_lock(level_shard.mutex);
  std::unique_lock<std::shared_mutex> size_lock(size_shard.mutex);

  auto level_it = level_shard.map.find(file_number);
  auto size_it = size_shard.map.find(file_number);
  if (level_it != level_shard.map.end() && size_it != size_shard.map.end()) {
    const size_t old_level = NormalizeLevel(level_it->second);
    level_data_sizes_[old_level].fetch_sub(size_it->second,
                                           std::memory_order_relaxed);
  }
  if (level_it != level_shard.map.end()) {
    level_shard.map.erase(level_it);
  }
  if (size_it != size_shard.map.end()) {
    size_shard.map.erase(size_it);
  }
}

void MultiLevelCache::UpdateCacheKeyPrefixMapping(uint64_t cache_key_prefix,
                                                  int level) {
  const size_t shard_index = ShardIndex(cache_key_prefix);
  MappingShard& shard = cache_key_prefix_to_level_[shard_index];
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  shard.map[cache_key_prefix] = level;
}

void MultiLevelCache::RemoveCacheKeyPrefixMapping(uint64_t cache_key_prefix) {
  const size_t shard_index = ShardIndex(cache_key_prefix);
  MappingShard& shard = cache_key_prefix_to_level_[shard_index];
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  shard.map.erase(cache_key_prefix);
}

size_t MultiLevelCache::RouteLevelByKey(const Slice& key,
                                        RouteCaller caller) const {
  std::atomic<uint64_t>* route_queries = nullptr;
  std::atomic<uint64_t>* route_parse_failures = nullptr;
  std::atomic<uint64_t>* route_prefix_hits = nullptr;
  std::atomic<uint64_t>* route_prefix_misses = nullptr;
  switch (caller) {
    case RouteCaller::kInsert:
      route_queries = &insert_route_queries_;
      route_parse_failures = &insert_route_parse_failures_;
      route_prefix_hits = &insert_route_prefix_hits_;
      route_prefix_misses = &insert_route_prefix_misses_;
      break;
    case RouteCaller::kLookup:
      route_queries = &lookup_route_queries_;
      route_parse_failures = &lookup_route_parse_failures_;
      route_prefix_hits = &lookup_route_prefix_hits_;
      route_prefix_misses = &lookup_route_prefix_misses_;
      break;
    case RouteCaller::kOther:
      break;
  }
  if (route_queries != nullptr) {
    route_queries->fetch_add(1, std::memory_order_relaxed);
  }
  const std::optional<uint64_t> key_prefix = GetCacheKeyPrefix(key);
  if (!key_prefix.has_value()) {
    if (route_parse_failures != nullptr) {
      route_parse_failures->fetch_add(1, std::memory_order_relaxed);
    }
    return 0;
  }
  const std::optional<int> level = FindLevelByCacheKeyPrefix(*key_prefix);
  if (!level.has_value()) {
    if (route_prefix_misses != nullptr) {
      route_prefix_misses->fetch_add(1, std::memory_order_relaxed);
    }
    return 0;
  }
  if (route_prefix_hits != nullptr) {
    route_prefix_hits->fetch_add(1, std::memory_order_relaxed);
  }
  if (*level < 0 || static_cast<size_t>(*level) >= sub_caches_.size()) {
    route_normalize_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    return 0;
  }
  return static_cast<size_t>(*level);
}

std::optional<uint64_t> MultiLevelCache::GetCacheKeyPrefix(
    const Slice& key) const {
  // RocksDB block cache key layout is 16 bytes by default, with the first
  // 8 bytes as file-specific common prefix. Route by this prefix.
  if (key.size() < sizeof(uint64_t)) {
    return std::nullopt;
  }
  return DecodeFixed64(key.data());
}

std::optional<int> MultiLevelCache::FindLevelByCacheKeyPrefix(
    uint64_t cache_key_prefix) const {
  const size_t shard_index = ShardIndex(cache_key_prefix);
  const MappingShard& shard = cache_key_prefix_to_level_[shard_index];
  std::shared_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.map.find(cache_key_prefix);
  if (it == shard.map.end()) {
    return std::nullopt;
  }
  return it->second;
}

MultiLevelCache::WrappedHandle* MultiLevelCache::NewWrappedHandle(
    size_t level_index, Cache::Handle* inner) {
  assert(inner != nullptr);
  auto* wrapped = new WrappedHandle();
  wrapped->level_index = level_index;
  wrapped->inner = inner;
  return wrapped;
}

MultiLevelCache::WrappedHandle* MultiLevelCache::ToWrappedHandle(
    Handle* handle) {
  assert(handle != nullptr);
  auto* wrapped = static_cast<WrappedHandle*>(handle);
  assert(wrapped->inner != nullptr);
  return wrapped;
}

const MultiLevelCache::WrappedHandle* MultiLevelCache::ToWrappedHandle(
    const Handle* handle) {
  assert(handle != nullptr);
  const auto* wrapped = static_cast<const WrappedHandle*>(handle);
  assert(wrapped->inner != nullptr);
  return wrapped;
}

size_t MultiLevelCache::NormalizeLevel(int level) const {
  if (level < 0) {
    return 0;
  }
  const size_t level_index = static_cast<size_t>(level);
  if (level_index >= sub_caches_.size()) {
    return 0;
  }
  return level_index;
}

uint64_t MultiLevelCache::CountMappingEntries(
    const std::array<MappingShard, kMappingShardCount>& shards) const {
  uint64_t total = 0;
  for (const MappingShard& shard : shards) {
    std::shared_lock<std::shared_mutex> lock(shard.mutex);
    total += shard.map.size();
  }
  return total;
}

Status MultiLevelCache::ValidateCapacities(
    const std::vector<size_t>& capacities) const {
  if (capacities.size() != sub_caches_.size()) {
    return Status::InvalidArgument(
        "new_capacities size must equal number of levels");
  }
  uint64_t total = 0;
  const size_t capacity_limit = total_capacity_.load(std::memory_order_relaxed);
  for (size_t capacity : capacities) {
    total += capacity;
    if (total > capacity_limit) {
      return Status::InvalidArgument(
          "sum of new_capacities exceeds total_capacity");
    }
  }
  return Status::OK();
}

void MultiLevelCache::ApplyCapacities(const std::vector<size_t>& capacities) {
  assert(capacities.size() == sub_caches_.size());
  for (size_t level = 0; level < capacities.size(); ++level) {
    sub_caches_[level]->SetCapacity(capacities[level]);
  }
}

Cache* MultiLevelCache::SubCacheByLevel(size_t level_index) {
  assert(level_index < sub_caches_.size());
  return sub_caches_[level_index].get();
}

const Cache* MultiLevelCache::SubCacheByLevel(size_t level_index) const {
  assert(level_index < sub_caches_.size());
  return sub_caches_[level_index].get();
}

Cache* MultiLevelCache::PrimarySubCache() {
  return SubCacheByLevel(0);
}

const Cache* MultiLevelCache::PrimarySubCache() const {
  return SubCacheByLevel(0);
}

}  // namespace ROCKSDB_NAMESPACE
