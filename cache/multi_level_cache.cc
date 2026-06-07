#include "cache/multi_level_cache.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <utility>

#include "cache/cache_key.h"
#include "util/coding.h"
namespace ROCKSDB_NAMESPACE {

namespace {

size_t SafeLevelCount(size_t num_levels) {
  return std::max<size_t>(num_levels, 1);
}

template <typename CacheOptionsT>
std::shared_ptr<Cache> MakeSubCacheWithCapacity(const CacheOptionsT& base_options,
                                                size_t capacity) {
  CacheOptionsT options = base_options;
  options.capacity = capacity;
  return options.MakeSharedCache();
}

std::shared_ptr<Cache> MakeSubCacheWithCapacity(
    const HyperClockCacheOptions& base_options, size_t capacity) {
  HyperClockCacheOptions options = base_options;
  // Auto HCC may fail to initialize when constructed with zero capacity.
  // Use a tiny bootstrap capacity; runtime SetCapacity(0) is still allowed.
  options.capacity = std::max<size_t>(capacity, 1);
  return options.MakeSharedCache();
}

constexpr uint32_t kRatioScalePpm = 1000000U;
constexpr uint32_t kMaxSharedPoolRatioPpm = 900000U;
constexpr size_t kMaxAdmissionCandidatesPerShard = 4096;

}  // namespace

MultiLevelCache::MultiLevelCache(size_t num_levels, size_t total_capacity)
    : MultiLevelCache(num_levels, total_capacity,
                      LRUCacheOptions(total_capacity, -1,
                                      false /*strict_capacity_limit*/,
                                      0.0 /*high_pri_pool_ratio*/),
                      false /*initial_force_route_all_to_l0*/) {}

MultiLevelCache::MultiLevelCache(size_t num_levels, size_t total_capacity,
                                 const LRUCacheOptions& lru_options,
                                 bool initial_force_route_all_to_l0)
    : debug_miss_budget_(ParseDebugMissLimit()), total_capacity_(total_capacity) {
  const size_t level_count = SafeLevelCount(num_levels);
  sub_caches_.reserve(level_count);

  const size_t per_level_capacity =
      initial_force_route_all_to_l0 ? 0 : (total_capacity / level_count);
  const size_t remainder = initial_force_route_all_to_l0
                               ? 0
                               : (total_capacity % level_count);
  for (size_t level = 0; level < level_count; ++level) {
    size_t level_capacity = per_level_capacity + (level < remainder ? 1 : 0);
    if (initial_force_route_all_to_l0 && level == 0) {
      level_capacity = total_capacity;
    }
    sub_caches_.emplace_back(
        MakeSubCacheWithCapacity(lru_options, level_capacity));
  }
  shared_cache_ = MakeSubCacheWithCapacity(lru_options, 0);
  InitializePerLevelState(level_count);
}

MultiLevelCache::MultiLevelCache(size_t num_levels, size_t total_capacity,
                                 const HyperClockCacheOptions& hcc_options,
                                 bool initial_force_route_all_to_l0)
    : debug_miss_budget_(ParseDebugMissLimit()), total_capacity_(total_capacity) {
  const size_t level_count = SafeLevelCount(num_levels);
  sub_caches_.reserve(level_count);

  const size_t per_level_capacity =
      initial_force_route_all_to_l0 ? 0 : (total_capacity / level_count);
  const size_t remainder = initial_force_route_all_to_l0
                               ? 0
                               : (total_capacity % level_count);
  for (size_t level = 0; level < level_count; ++level) {
    size_t level_capacity = per_level_capacity + (level < remainder ? 1 : 0);
    if (initial_force_route_all_to_l0 && level == 0) {
      level_capacity = total_capacity;
    }
    sub_caches_.emplace_back(
        MakeSubCacheWithCapacity(hcc_options, level_capacity));
  }
  shared_cache_ = MakeSubCacheWithCapacity(hcc_options, 0);
  InitializePerLevelState(level_count);
}

MultiLevelCache::MultiLevelCache(std::vector<std::shared_ptr<Cache>> sub_caches,
                                 std::shared_ptr<Cache> shared_cache,
                                 size_t total_capacity)
    : sub_caches_(std::move(sub_caches)),
      shared_cache_(std::move(shared_cache)),
      debug_miss_budget_(ParseDebugMissLimit()),
      total_capacity_(total_capacity) {
  assert(!sub_caches_.empty());
  assert(shared_cache_ != nullptr);
  InitializePerLevelState(sub_caches_.size());
}

const char* MultiLevelCache::Name() const { return "MultiLevelCache"; }

Status MultiLevelCache::Insert(const Slice& key, ObjectPtr obj,
                               const CacheItemHelper* helper, size_t charge,
                               Handle** handle, Priority priority,
                               const Slice& compressed, CompressionType type) {
  const size_t level_index = RouteLevelByKey(key, RouteCaller::kInsert);
  Cache* target_cache = SubCacheByLevel(level_index);
  const uint64_t key_hash = HashCacheKey(key);
  bool route_to_shared = false;
  if (!force_route_all_to_l0_.load(std::memory_order_relaxed) &&
      shared_pool_ratio_ppm_.load(std::memory_order_relaxed) > 0 &&
      IsSharedPoolAdmissionReady(key_hash)) {
    target_cache = SharedCache();
    route_to_shared = true;
  }
  Cache::Handle* inner = nullptr;
  Cache::Handle** inner_handle = handle != nullptr ? &inner : nullptr;
  Status s = target_cache->Insert(key, obj, helper, charge, inner_handle,
                                  priority, compressed, type);
  if (!s.ok()) {
    return s;
  }
  if (route_to_shared) {
    ClearSharedPoolAdmission(key_hash);
    shared_pool_admissions_.fetch_add(1, std::memory_order_relaxed);
  }
  if (handle != nullptr && inner != nullptr) {
    *handle = NewWrappedHandle(target_cache, inner);
  }
  return s;
}

Cache::Handle* MultiLevelCache::CreateStandalone(
    const Slice& key, ObjectPtr obj, const CacheItemHelper* helper,
    size_t charge, bool allow_uncharged) {
  const size_t level_index = RouteLevelByKey(key, RouteCaller::kOther);
  Cache* target_cache = SubCacheByLevel(level_index);
  Cache::Handle* inner =
      target_cache->CreateStandalone(key, obj, helper, charge, allow_uncharged);
  if (inner == nullptr) {
    return nullptr;
  }
  return NewWrappedHandle(target_cache, inner);
}

Cache::Handle* MultiLevelCache::Lookup(const Slice& key,
                                       const CacheItemHelper* helper,
                                       CreateContext* create_context,
                                       Priority priority, Statistics* stats) {
  const size_t level_index = RouteLevelByKey(key, RouteCaller::kLookup);
  MaybeRecordLookupSample(level_index, key);
  lookups_[level_index].fetch_add(1, std::memory_order_relaxed);
  Cache* level_cache = SubCacheByLevel(level_index);
  Cache::Handle* inner =
      level_cache->Lookup(key, helper, create_context, priority, stats);
  Cache* owner_cache = level_cache;
  if (inner == nullptr) {
    const bool shared_enabled =
        !force_route_all_to_l0_.load(std::memory_order_relaxed) &&
        shared_pool_ratio_ppm_.load(std::memory_order_relaxed) > 0;
    if (!shared_enabled) {
      return nullptr;
    }
    shared_pool_lookups_.fetch_add(1, std::memory_order_relaxed);
    inner = SharedCache()->Lookup(key, helper, create_context, priority, stats);
    if (inner == nullptr) {
      RecordSharedPoolCandidate(HashCacheKey(key));
      return nullptr;
    }
    shared_pool_hits_.fetch_add(1, std::memory_order_relaxed);
    owner_cache = SharedCache();
  }
  hits_[level_index].fetch_add(1, std::memory_order_relaxed);
  return NewWrappedHandle(owner_cache, inner);
}

bool MultiLevelCache::Ref(Handle* handle) {
  WrappedHandle* wrapped = ToWrappedHandle(handle);
  return wrapped->owner_cache->Ref(wrapped->inner);
}

bool MultiLevelCache::Release(Handle* handle, bool erase_if_last_ref) {
  WrappedHandle* wrapped = ToWrappedHandle(handle);
  const bool erased =
      wrapped->owner_cache->Release(wrapped->inner, erase_if_last_ref);
  delete wrapped;
  return erased;
}

Cache::ObjectPtr MultiLevelCache::Value(Handle* handle) {
  WrappedHandle* wrapped = ToWrappedHandle(handle);
  return wrapped->owner_cache->Value(wrapped->inner);
}

void MultiLevelCache::Erase(const Slice& key) {
  SubCacheByLevel(RouteLevelByKey(key, RouteCaller::kOther))->Erase(key);
  if (!force_route_all_to_l0_.load(std::memory_order_relaxed) &&
      shared_pool_ratio_ppm_.load(std::memory_order_relaxed) > 0) {
    SharedCache()->Erase(key);
  }
}

uint64_t MultiLevelCache::NewId() { return PrimarySubCache()->NewId(); }

void MultiLevelCache::SetCapacity(size_t capacity) {
  total_capacity_.store(capacity, std::memory_order_relaxed);
  const size_t level_count = sub_caches_.size();
  const size_t shared_capacity = GetSharedPoolCapacity(capacity);
  const size_t level_budget = capacity - shared_capacity;
  const size_t per_level_capacity = level_budget / level_count;
  const size_t remainder = level_budget % level_count;
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
  if (shared_cache_ != nullptr) {
    shared_cache_->SetStrictCapacityLimit(strict_capacity_limit);
  }
}

bool MultiLevelCache::HasStrictCapacityLimit() const {
  if (shared_cache_ != nullptr && shared_cache_->HasStrictCapacityLimit()) {
    return true;
  }
  return PrimarySubCache()->HasStrictCapacityLimit();
}

size_t MultiLevelCache::GetCapacity() const {
  size_t total = 0;
  for (const auto& sub_cache : sub_caches_) {
    total += sub_cache->GetCapacity();
  }
  if (shared_cache_ != nullptr) {
    total += shared_cache_->GetCapacity();
  }
  return total;
}

size_t MultiLevelCache::GetUsage() const {
  size_t total = 0;
  for (const auto& sub_cache : sub_caches_) {
    total += sub_cache->GetUsage();
  }
  if (shared_cache_ != nullptr) {
    total += shared_cache_->GetUsage();
  }
  return total;
}

size_t MultiLevelCache::GetUsage(Handle* handle) const {
  const WrappedHandle* wrapped = ToWrappedHandle(handle);
  return wrapped->owner_cache->GetUsage(wrapped->inner);
}

size_t MultiLevelCache::GetPinnedUsage() const {
  size_t total = 0;
  for (const auto& sub_cache : sub_caches_) {
    total += sub_cache->GetPinnedUsage();
  }
  if (shared_cache_ != nullptr) {
    total += shared_cache_->GetPinnedUsage();
  }
  return total;
}

size_t MultiLevelCache::GetCharge(Handle* handle) const {
  const WrappedHandle* wrapped = ToWrappedHandle(handle);
  return wrapped->owner_cache->GetCharge(wrapped->inner);
}

const Cache::CacheItemHelper* MultiLevelCache::GetCacheItemHelper(
    Handle* handle) const {
  const WrappedHandle* wrapped = ToWrappedHandle(handle);
  return wrapped->owner_cache->GetCacheItemHelper(wrapped->inner);
}

void MultiLevelCache::ApplyToAllEntries(
    const std::function<void(const Slice& key, ObjectPtr obj, size_t charge,
                             const CacheItemHelper* helper)>& callback,
    const ApplyToAllEntriesOptions& opts) {
  for (const auto& sub_cache : sub_caches_) {
    sub_cache->ApplyToAllEntries(callback, opts);
  }
  if (shared_cache_ != nullptr) {
    shared_cache_->ApplyToAllEntries(callback, opts);
  }
}

void MultiLevelCache::ApplyToHandle(
    Cache* /*cache*/, Handle* handle,
    const std::function<void(const Slice& key, ObjectPtr obj, size_t charge,
                             const CacheItemHelper* helper)>& callback) {
  WrappedHandle* wrapped = ToWrappedHandle(handle);
  wrapped->owner_cache->ApplyToHandle(wrapped->owner_cache, wrapped->inner,
                                      callback);
}

void MultiLevelCache::EraseUnRefEntries() {
  for (const auto& sub_cache : sub_caches_) {
    sub_cache->EraseUnRefEntries();
  }
  if (shared_cache_ != nullptr) {
    shared_cache_->EraseUnRefEntries();
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
  if (shared_cache_ != nullptr) {
    const size_t count = shared_cache_->GetOccupancyCount();
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
  if (shared_cache_ != nullptr) {
    total += shared_cache_->GetTableAddressCount();
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
  const uint64_t shared_lookups =
      shared_pool_lookups_.load(std::memory_order_relaxed);
  const uint64_t shared_hits = shared_pool_hits_.load(std::memory_order_relaxed);
  const double shared_hit_rate =
      shared_lookups == 0
          ? 0.0
          : static_cast<double>(shared_hits) / static_cast<double>(shared_lookups);
  oss << "shared_pool: capacity="
      << (shared_cache_ != nullptr ? shared_cache_->GetCapacity() : 0)
      << ", lookups=" << shared_lookups << ", hits=" << shared_hits
      << ", hit_rate=" << shared_hit_rate
      << ", admissions="
      << shared_pool_admissions_.load(std::memory_order_relaxed) << "\n";
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
  shared_pool_lookups_.store(0, std::memory_order_relaxed);
  shared_pool_hits_.store(0, std::memory_order_relaxed);
  shared_pool_admissions_.store(0, std::memory_order_relaxed);
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

std::vector<std::vector<uint64_t>> MultiLevelCache::DrainLookupSamples() {
  std::vector<std::vector<uint64_t>> drained(sub_caches_.size());
  for (size_t level = 0; level < sub_caches_.size(); ++level) {
    std::lock_guard<std::mutex> lock(lookup_sample_mutexes_[level]);
    auto& src = lookup_samples_[level];
    drained[level].assign(src.begin(), src.end());
    src.clear();
  }
  return drained;
}

void MultiLevelCache::SetLookupSampleRateLog2(uint32_t sample_rate_log2) {
  // Guard overly sparse sampling and shift overflow.
  const uint32_t clamped = std::min<uint32_t>(sample_rate_log2, 20);
  lookup_sample_rate_log2_.store(clamped, std::memory_order_relaxed);
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

void MultiLevelCache::SetForceRouteAllToL0(bool force_route_all_to_l0) {
  force_route_all_to_l0_.store(force_route_all_to_l0, std::memory_order_relaxed);
  if (!force_route_all_to_l0 || sub_caches_.empty()) {
    return;
  }
  std::vector<size_t> l0_only_capacities(sub_caches_.size(), 0);
  l0_only_capacities[0] = total_capacity_.load(std::memory_order_relaxed);
  ApplyCapacities(l0_only_capacities);
}

void MultiLevelCache::SetSharedPoolRatio(double shared_pool_ratio) {
  const double clamped = std::max(0.0, std::min(0.9, shared_pool_ratio));
  const uint32_t ppm =
      static_cast<uint32_t>(clamped * static_cast<double>(kRatioScalePpm));
  shared_pool_ratio_ppm_.store(std::min(ppm, kMaxSharedPoolRatioPpm),
                               std::memory_order_relaxed);
  SetCapacity(total_capacity_.load(std::memory_order_relaxed));
}

void MultiLevelCache::SetSharedPoolAdmissionThreshold(
    uint32_t admission_threshold) {
  shared_pool_admission_threshold_.store(std::max<uint32_t>(1, admission_threshold),
                                         std::memory_order_relaxed);
}

void MultiLevelCache::SetSharedPoolDecayIntervalOps(uint32_t decay_interval_ops) {
  shared_pool_decay_interval_ops_.store(
      std::max<uint32_t>(1, decay_interval_ops), std::memory_order_relaxed);
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
  if (force_route_all_to_l0_.load(std::memory_order_relaxed)) {
    return 0;
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
    Cache* owner_cache, Cache::Handle* inner) {
  assert(inner != nullptr);
  assert(owner_cache != nullptr);
  auto* wrapped = new WrappedHandle();
  wrapped->owner_cache = owner_cache;
  wrapped->inner = inner;
  return wrapped;
}

MultiLevelCache::WrappedHandle* MultiLevelCache::ToWrappedHandle(
    Handle* handle) {
  assert(handle != nullptr);
  auto* wrapped = static_cast<WrappedHandle*>(handle);
  assert(wrapped->owner_cache != nullptr);
  assert(wrapped->inner != nullptr);
  return wrapped;
}

const MultiLevelCache::WrappedHandle* MultiLevelCache::ToWrappedHandle(
    const Handle* handle) {
  assert(handle != nullptr);
  const auto* wrapped = static_cast<const WrappedHandle*>(handle);
  assert(wrapped->owner_cache != nullptr);
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
  std::vector<size_t> level_capacities(capacities.size(), 0);
  if (force_route_all_to_l0_.load(std::memory_order_relaxed)) {
    level_capacities[0] = total_capacity_.load(std::memory_order_relaxed);
    for (size_t level = 0; level < level_capacities.size(); ++level) {
      sub_caches_[level]->SetCapacity(level_capacities[level]);
    }
    if (shared_cache_ != nullptr) {
      shared_cache_->SetCapacity(0);
    }
    return;
  }

  const size_t total_capacity = total_capacity_.load(std::memory_order_relaxed);
  const size_t shared_capacity = GetSharedPoolCapacity(total_capacity);
  const size_t level_budget = total_capacity - shared_capacity;
  uint64_t requested_total = 0;
  for (size_t value : capacities) {
    requested_total += value;
  }
  if (requested_total == 0) {
    const size_t per_level = level_budget / level_capacities.size();
    const size_t remainder = level_budget % level_capacities.size();
    for (size_t level = 0; level < level_capacities.size(); ++level) {
      level_capacities[level] = per_level + (level < remainder ? 1 : 0);
    }
  } else {
    size_t assigned = 0;
    for (size_t level = 0; level < level_capacities.size(); ++level) {
      const size_t scaled = static_cast<size_t>(
          (static_cast<unsigned __int128>(capacities[level]) * level_budget) /
          requested_total);
      level_capacities[level] = scaled;
      assigned += scaled;
    }
    for (size_t level = 0;
         assigned < level_budget && level < level_capacities.size();
         ++level, ++assigned) {
      level_capacities[level] += 1;
    }
  }

  for (size_t level = 0; level < level_capacities.size(); ++level) {
    sub_caches_[level]->SetCapacity(level_capacities[level]);
  }
  if (shared_cache_ != nullptr) {
    shared_cache_->SetCapacity(shared_capacity);
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

Cache* MultiLevelCache::SharedCache() {
  assert(shared_cache_ != nullptr);
  return shared_cache_.get();
}

const Cache* MultiLevelCache::SharedCache() const {
  assert(shared_cache_ != nullptr);
  return shared_cache_.get();
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

void MultiLevelCache::MaybeRecordLookupSample(size_t level_index,
                                              const Slice& key) {
  static constexpr size_t kMaxSamplesPerLevel = 8192;
  const uint32_t log2_rate =
      lookup_sample_rate_log2_.load(std::memory_order_relaxed);
  const uint64_t sample_mask =
      log2_rate == 0 ? 0 : ((1ULL << log2_rate) - 1ULL);
  if ((lookup_sample_seq_.fetch_add(1, std::memory_order_relaxed) &
       sample_mask) != 0) {
    return;
  }
  if (level_index >= lookup_samples_.size()) {
    return;
  }
  const uint64_t hash = static_cast<uint64_t>(
      std::hash<std::string_view>{}(std::string_view(key.data(), key.size())));
  std::lock_guard<std::mutex> lock(lookup_sample_mutexes_[level_index]);
  auto& samples = lookup_samples_[level_index];
  if (samples.size() >= kMaxSamplesPerLevel) {
    samples.pop_front();
  }
  samples.push_back(hash);
}

uint64_t MultiLevelCache::HashCacheKey(const Slice& key) const {
  return static_cast<uint64_t>(
      std::hash<std::string_view>{}(std::string_view(key.data(), key.size())));
}

void MultiLevelCache::RecordSharedPoolCandidate(uint64_t key_hash) {
  const size_t shard_idx = key_hash % kSharedAdmissionShardCount;
  SharedAdmissionShard& shard = shared_admission_shards_[shard_idx];
  std::lock_guard<std::mutex> lock(shard.mutex);
  MaybeDecaySharedAdmissionShard(&shard);
  uint32_t& count = shard.miss_scores[key_hash];
  if (count < 255U) {
    ++count;
  }
  TrimSharedAdmissionShardIfNeeded(&shard);
}

bool MultiLevelCache::IsSharedPoolAdmissionReady(uint64_t key_hash) {
  const size_t shard_idx = key_hash % kSharedAdmissionShardCount;
  SharedAdmissionShard& shard = shared_admission_shards_[shard_idx];
  std::lock_guard<std::mutex> lock(shard.mutex);
  MaybeDecaySharedAdmissionShard(&shard);
  auto it = shard.miss_scores.find(key_hash);
  if (it == shard.miss_scores.end()) {
    return false;
  }
  const uint32_t threshold =
      shared_pool_admission_threshold_.load(std::memory_order_relaxed);
  if (it->second < threshold) {
    return false;
  }
  return true;
}

void MultiLevelCache::ClearSharedPoolAdmission(uint64_t key_hash) {
  const size_t shard_idx = key_hash % kSharedAdmissionShardCount;
  SharedAdmissionShard& shard = shared_admission_shards_[shard_idx];
  std::lock_guard<std::mutex> lock(shard.mutex);
  shard.miss_scores.erase(key_hash);
}

void MultiLevelCache::MaybeDecaySharedAdmissionShard(
    SharedAdmissionShard* shard) {
  assert(shard != nullptr);
  const uint32_t decay_interval =
      shared_pool_decay_interval_ops_.load(std::memory_order_relaxed);
  ++shard->ops_since_decay;
  if (shard->ops_since_decay < decay_interval) {
    return;
  }
  shard->ops_since_decay = 0;
  for (auto it = shard->miss_scores.begin(); it != shard->miss_scores.end();) {
    it->second >>= 1;
    if (it->second == 0) {
      it = shard->miss_scores.erase(it);
      continue;
    }
    ++it;
  }
}

void MultiLevelCache::TrimSharedAdmissionShardIfNeeded(
    SharedAdmissionShard* shard) {
  assert(shard != nullptr);
  if (shard->miss_scores.size() <= kMaxAdmissionCandidatesPerShard) {
    return;
  }
  // Keep memory bounded by dropping low-score entries first.
  for (auto it = shard->miss_scores.begin();
       it != shard->miss_scores.end() &&
       shard->miss_scores.size() > kMaxAdmissionCandidatesPerShard;) {
    if (it->second <= 1) {
      it = shard->miss_scores.erase(it);
      continue;
    }
    ++it;
  }
  if (shard->miss_scores.size() <= kMaxAdmissionCandidatesPerShard) {
    return;
  }
  for (auto it = shard->miss_scores.begin();
       it != shard->miss_scores.end() &&
       shard->miss_scores.size() > kMaxAdmissionCandidatesPerShard;) {
    it = shard->miss_scores.erase(it);
  }
}

size_t MultiLevelCache::GetSharedPoolCapacity(size_t total_capacity) const {
  if (force_route_all_to_l0_.load(std::memory_order_relaxed)) {
    return 0;
  }
  const uint32_t ppm = shared_pool_ratio_ppm_.load(std::memory_order_relaxed);
  if (ppm == 0) {
    return 0;
  }
  return static_cast<size_t>(
      (static_cast<unsigned __int128>(total_capacity) * ppm) / kRatioScalePpm);
}

void MultiLevelCache::InitializePerLevelState(size_t level_count) {
  lookups_.resize(level_count);
  hits_.resize(level_count);
  level_data_sizes_.resize(level_count);
  lookup_sample_mutexes_.resize(level_count);
  lookup_samples_.resize(level_count);
  for (size_t level = 0; level < level_count; ++level) {
    lookups_[level].store(0, std::memory_order_relaxed);
    hits_[level].store(0, std::memory_order_relaxed);
    level_data_sizes_[level].store(0, std::memory_order_relaxed);
  }
}

}  // namespace ROCKSDB_NAMESPACE
