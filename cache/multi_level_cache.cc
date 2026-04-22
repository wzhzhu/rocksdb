#include "cache/multi_level_cache.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <utility>

#include "cache/cache_key.h"
#include "util/coding.h"
namespace ROCKSDB_NAMESPACE {

namespace {

size_t SafeLevelCount(size_t num_levels) {
  return std::max<size_t>(num_levels, 1);
}

}  // namespace

MultiLevelCache::MultiLevelCache(size_t num_levels, size_t total_capacity)
    : debug_miss_budget_(ParseDebugMissLimit()), total_capacity_(total_capacity) {
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

void MultiLevelCache::UpdateLevelDataSizes(
    const std::vector<uint64_t>& level_data_sizes) {
  const size_t limit = std::min(level_data_sizes.size(), level_data_sizes_.size());
  for (size_t level = 0; level < limit; ++level) {
    level_data_sizes_[level].store(level_data_sizes[level],
                                   std::memory_order_relaxed);
  }
  for (size_t level = limit; level < level_data_sizes_.size(); ++level) {
    level_data_sizes_[level].store(0, std::memory_order_relaxed);
  }
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
    MaybeLogRouteMiss(caller, key, /*has_prefix=*/false, 0,
                      "prefix_parse_failure");
    return 0;
  }
  int encoded_level = 0;
  if (DecodeLevelFromEncodedCacheKeyCommonPrefix(*key_prefix, &encoded_level)) {
    if (route_prefix_hits != nullptr) {
      route_prefix_hits->fetch_add(1, std::memory_order_relaxed);
    }
    if (encoded_level < 0 ||
        static_cast<size_t>(encoded_level) >= sub_caches_.size()) {
      route_normalize_fallbacks_.fetch_add(1, std::memory_order_relaxed);
      return 0;
    }
    return static_cast<size_t>(encoded_level);
  }
  if (route_prefix_misses != nullptr) {
    route_prefix_misses->fetch_add(1, std::memory_order_relaxed);
  }
  MaybeLogRouteMiss(caller, key, /*has_prefix=*/true, *key_prefix,
                    "prefix_marker_miss");
  return 0;
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

void MultiLevelCache::MaybeLogRouteMiss(RouteCaller caller, const Slice& key,
                                        bool has_prefix, uint64_t key_prefix,
                                        const char* reason) const {
  int64_t remaining = debug_miss_budget_.load(std::memory_order_relaxed);
  while (remaining > 0) {
    if (debug_miss_budget_.compare_exchange_weak(
            remaining, remaining - 1, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      if (has_prefix) {
        std::fprintf(stderr,
                     "[MultiLevelCache][route_miss] caller=%s reason=%s "
                     "key_size=%zu prefix=0x%016" PRIx64 "\n",
                     RouteCallerToString(caller), reason, key.size(),
                     key_prefix);
      } else {
        std::fprintf(stderr,
                     "[MultiLevelCache][route_miss] caller=%s reason=%s "
                     "key_size=%zu\n",
                     RouteCallerToString(caller), reason, key.size());
      }
      return;
    }
  }
}

const char* MultiLevelCache::RouteCallerToString(RouteCaller caller) {
  switch (caller) {
    case RouteCaller::kInsert:
      return "insert";
    case RouteCaller::kLookup:
      return "lookup";
    case RouteCaller::kOther:
      return "other";
  }
  return "unknown";
}

int64_t MultiLevelCache::ParseDebugMissLimit() {
  const char* env = std::getenv("MLC_ROUTE_DEBUG_MISS_LIMIT");
  if (env == nullptr || env[0] == '\0') {
    return 0;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(env, &end, 10);
  if (end == env || (end != nullptr && *end != '\0') || parsed <= 0) {
    return 0;
  }
  return static_cast<int64_t>(parsed);
}

}  // namespace ROCKSDB_NAMESPACE
