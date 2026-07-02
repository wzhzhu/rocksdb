#include "cache/multi_level_cache.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <utility>

#include "cache/cache_key.h"
#include "cache/clock_cache.h"
#include "cache/multi_level_cache_compaction.h"
#include "util/coding.h"
namespace ROCKSDB_NAMESPACE {

// Per-thread flag set by MLCLookupCompactionScope on the compaction read path.
// Read by MultiLevelCache::Lookup to skip per-level/hit counters for
// compaction-induced accesses so they never feed the allocator's model.
thread_local bool tls_mlc_lookup_is_compaction = false;

bool MLCLookupIsCompaction() { return tls_mlc_lookup_is_compaction; }

MLCLookupCompactionScope::MLCLookupCompactionScope(bool is_compaction)
    : prev_(tls_mlc_lookup_is_compaction) {
  tls_mlc_lookup_is_compaction = is_compaction;
}

MLCLookupCompactionScope::~MLCLookupCompactionScope() {
  tls_mlc_lookup_is_compaction = prev_;
}

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
    const HyperClockCacheOptions& base_options, size_t capacity,
    size_t mmap_capacity) {
  HyperClockCacheOptions options = base_options;
  // The Auto HCC slot array is an mmap whose size is fixed at construction
  // from `capacity` (CalcMaxUsableLength) and can never grow beyond it. MLC's
  // allocator later raises a hot level far above its equal-split start (e.g.
  // L6 to ~0.9x total), so reserve the mapping for the whole budget here and
  // set the smaller starting capacity afterward; otherwise Grow() runs past
  // the mapping and corrupts the heap. Auto HCC may also fail to initialize at
  // zero capacity, so floor the reservation at 1.
  options.capacity = std::max<size_t>(mmap_capacity, 1);
  auto cache = options.MakeSharedCache();
  cache->SetCapacity(capacity);
  return cache;
}

constexpr uint32_t kRatioScalePpm = 1000000U;

// Low-bit tag marking a MultiLevelCache handle as a heap-allocated
// WrappedHandle (rare path); untagged handles are passed through from the
// sub-caches and their owner is recovered by address range.
constexpr uintptr_t kWrappedHandleTagBit = uintptr_t{1};
constexpr uint32_t kMaxSharedPoolRatioPpm = 900000U;
constexpr size_t kMaxAdmissionCandidatesPerShard = 4096;

// Distributes client threads round-robin over counter stripes. The stripe is
// per-thread (not per-cache-instance), which is fine: the goal is only to
// spread concurrent increments across cache lines.
size_t CounterStripeIndex(size_t num_stripes) {
  static std::atomic<uint32_t> next_stripe{0};
  thread_local uint32_t stripe =
      next_stripe.fetch_add(1, std::memory_order_relaxed);
  return stripe % num_stripes;
}

// HCC's SetCapacity only stores the new value; eviction is realized lazily on
// the insert path. A sub-cache whose level drained (no more inserts routed to
// it) would hold stale blocks indefinitely, silently exceeding the total
// budget (observed: a 9.7KB-capacity L0 sub-cache squatting on 934MB). After
// shrinking, synchronously purge down to the new capacity. Name()-based
// dispatch keeps this working in no-RTTI builds.
void PurgeSubCacheToCapacity(Cache* cache) {
  if (cache == nullptr) {
    return;
  }
  const char* name = cache->Name();
  if (std::strcmp(name, "FixedHyperClockCache") == 0) {
    static_cast<clock_cache::FixedHyperClockCache*>(cache)->PurgeToCapacity();
  } else if (std::strcmp(name, "AutoHyperClockCache") == 0) {
    static_cast<clock_cache::AutoHyperClockCache*>(cache)->PurgeToCapacity();
  }
}

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
    : MultiLevelCache(
          num_levels, total_capacity,
          [&lru_options](size_t level_capacity) {
            LRUCacheOptions options = lru_options;
            options.capacity = level_capacity;
            return options.MakeSharedCache();
          },
          initial_force_route_all_to_l0) {}

MultiLevelCache::MultiLevelCache(size_t num_levels, size_t total_capacity,
                                 SubCacheFactory sub_cache_factory,
                                 bool initial_force_route_all_to_l0)
    : debug_miss_budget_(ParseDebugMissLimit()), total_capacity_(total_capacity) {
  const size_t level_count = SafeLevelCount(num_levels);
  assert(sub_cache_factory != nullptr);
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
    auto sub_cache = sub_cache_factory(level_capacity);
    assert(sub_cache != nullptr);
    sub_caches_.emplace_back(std::move(sub_cache));
  }
  shared_cache_ = sub_cache_factory(0);
  assert(shared_cache_ != nullptr);
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
        MakeSubCacheWithCapacity(hcc_options, level_capacity, total_capacity));
  }
  shared_cache_ = MakeSubCacheWithCapacity(hcc_options, 0, total_capacity);
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

MultiLevelCache::~MultiLevelCache() { StopDynamicSRHCCWorker(); }

const char* MultiLevelCache::Name() const { return "MultiLevelCache"; }

std::string MultiLevelCache::GetPrintableOptions() const {
  std::ostringstream oss;
  oss << "multilevel_cache.levels=" << sub_caches_.size() << "\n";
  oss << "multilevel_cache.total_capacity="
      << total_capacity_.load(std::memory_order_relaxed) << "\n";
  for (size_t level = 0; level < sub_caches_.size(); ++level) {
    const auto& sub = sub_caches_[level];
    oss << "multilevel_cache.level[" << level
        << "].name=" << sub->Name() << "\n";
    const std::string sub_opts = sub->GetPrintableOptions();
    if (!sub_opts.empty()) {
      oss << "multilevel_cache.level[" << level << "].stats_begin\n";
      oss << sub_opts << "\n";
      oss << "multilevel_cache.level[" << level << "].stats_end\n";
    }
  }
  if (shared_cache_ != nullptr) {
    oss << "multilevel_cache.shared.name=" << shared_cache_->Name() << "\n";
    const std::string shared_opts = shared_cache_->GetPrintableOptions();
    if (!shared_opts.empty()) {
      oss << "multilevel_cache.shared.stats_begin\n";
      oss << shared_opts << "\n";
      oss << "multilevel_cache.shared.stats_end\n";
    }
  }
  return oss.str();
}

Status MultiLevelCache::Insert(const Slice& key, ObjectPtr obj,
                               const CacheItemHelper* helper, size_t charge,
                               Handle** handle, Priority priority,
                               const Slice& compressed, CompressionType type) {
  Slice base_key = key;
  const size_t level_index =
      RouteLevelByKey(key, RouteCaller::kInsert, &base_key);
  Cache* target_cache = SubCacheByLevel(level_index);
  const uint64_t key_hash = HashCacheKey(base_key);
  bool route_to_shared = false;
  if (!force_route_all_to_l0_.load(std::memory_order_relaxed) &&
      shared_pool_ratio_ppm_.load(std::memory_order_relaxed) > 0 &&
      IsSharedPoolAdmissionReady(key_hash)) {
    target_cache = SharedCache();
    route_to_shared = true;
  }
  Cache::Handle* inner = nullptr;
  Cache::Handle** inner_handle = handle != nullptr ? &inner : nullptr;
  Status s = target_cache->Insert(base_key, obj, helper, charge, inner_handle,
                                  priority, compressed, type);
  if (!s.ok()) {
    return s;
  }
  if (route_to_shared) {
    ClearSharedPoolAdmission(key_hash);
    shared_pool_admissions_.fetch_add(1, std::memory_order_relaxed);
  }
  if (handle != nullptr && inner != nullptr) {
    *handle = WrapOrPassHandle(target_cache, inner);
  }
  return s;
}

Cache::Handle* MultiLevelCache::CreateStandalone(
    const Slice& key, ObjectPtr obj, const CacheItemHelper* helper,
    size_t charge, bool allow_uncharged) {
  Slice base_key = key;
  const size_t level_index =
      RouteLevelByKey(key, RouteCaller::kOther, &base_key);
  Cache* target_cache = SubCacheByLevel(level_index);
  Cache::Handle* inner =
      target_cache->CreateStandalone(base_key, obj, helper, charge,
                                     allow_uncharged);
  if (inner == nullptr) {
    return nullptr;
  }
  return WrapOrPassHandle(target_cache, inner);
}

Cache::Handle* MultiLevelCache::Lookup(const Slice& key,
                                       const CacheItemHelper* helper,
                                       CreateContext* create_context,
                                       Priority priority, Statistics* stats) {
  // Compaction-induced lookups feed the per-level access frequency (lambda,
  // via lookups_/hits_) because compaction activity is a proxy for write-path
  // pressure on levels like L0 -- excluding it defunds the write buffer and
  // stalls. But compaction reads are streaming/one-shot, so their hits/misses
  // must NOT feed the hit-curve shape (alpha): that is captured by the
  // foreground-only fg_lookups_/fg_hits_ counters. The is_compaction flag is
  // set on the compaction read path via MLCLookupCompactionScope.
  const bool is_compaction = MLCLookupIsCompaction();
  Slice base_key = key;
  const size_t level_index =
      RouteLevelByKey(key, RouteCaller::kLookup, &base_key);
  MaybeRecordLookupSample(level_index, base_key);
  if (!is_compaction &&
      working_set_tracking_enabled_.load(std::memory_order_relaxed)) {
    RecordForegroundWorkingSet(level_index, base_key);
  }
  IncLookupCounter(level_index);
  if (!is_compaction) {
    IncFgLookupCounter(level_index);
  }
  Cache* level_cache = SubCacheByLevel(level_index);
  Cache::Handle* inner =
      level_cache->Lookup(base_key, helper, create_context, priority, stats);
  Cache* owner_cache = level_cache;
  if (inner == nullptr) {
    const bool shared_enabled =
        !force_route_all_to_l0_.load(std::memory_order_relaxed) &&
        shared_pool_ratio_ppm_.load(std::memory_order_relaxed) > 0;
    if (!shared_enabled) {
      return nullptr;
    }
    shared_pool_lookups_.fetch_add(1, std::memory_order_relaxed);
    inner =
        SharedCache()->Lookup(base_key, helper, create_context, priority, stats);
    if (inner == nullptr) {
      RecordSharedPoolCandidate(HashCacheKey(base_key));
      return nullptr;
    }
    shared_pool_hits_.fetch_add(1, std::memory_order_relaxed);
    owner_cache = SharedCache();
  }
  IncHitCounter(level_index);
  if (!is_compaction) {
    IncFgHitCounter(level_index);
  }
  return WrapOrPassHandle(owner_cache, inner);
}

bool MultiLevelCache::Ref(Handle* handle) {
  Cache* owner = nullptr;
  Cache::Handle* inner = nullptr;
  ResolveHandle(handle, &owner, &inner);
  return owner->Ref(inner);
}

bool MultiLevelCache::Release(Handle* handle, bool erase_if_last_ref) {
  const uintptr_t bits = reinterpret_cast<uintptr_t>(handle);
  if (bits & kWrappedHandleTagBit) {
    auto* wrapped =
        reinterpret_cast<WrappedHandle*>(bits & ~kWrappedHandleTagBit);
    const bool erased =
        wrapped->owner_cache->Release(wrapped->inner, erase_if_last_ref);
    delete wrapped;
    return erased;
  }
  Cache* owner = nullptr;
  Cache::Handle* inner = nullptr;
  ResolveHandle(handle, &owner, &inner);
  return owner->Release(inner, erase_if_last_ref);
}

Cache::ObjectPtr MultiLevelCache::Value(Handle* handle) {
  Cache* owner = nullptr;
  Cache::Handle* inner = nullptr;
  ResolveHandle(handle, &owner, &inner);
  return owner->Value(inner);
}

void MultiLevelCache::Erase(const Slice& key) {
  Slice base_key = key;
  SubCacheByLevel(RouteLevelByKey(key, RouteCaller::kOther, &base_key))
      ->Erase(base_key);
  if (!force_route_all_to_l0_.load(std::memory_order_relaxed) &&
      shared_pool_ratio_ppm_.load(std::memory_order_relaxed) > 0) {
    SharedCache()->Erase(base_key);
  }
}

void MultiLevelCache::IncLookupCounter(size_t level_index) {
  lookups_[CounterStripeIndex(kCounterStripes) * sub_caches_.size() +
           level_index]
      .v.fetch_add(1, std::memory_order_relaxed);
}

void MultiLevelCache::IncHitCounter(size_t level_index) {
  hits_[CounterStripeIndex(kCounterStripes) * sub_caches_.size() + level_index]
      .v.fetch_add(1, std::memory_order_relaxed);
}

uint64_t MultiLevelCache::SumLookupCounter(size_t level_index) const {
  uint64_t sum = 0;
  for (size_t stripe = 0; stripe < kCounterStripes; ++stripe) {
    sum += lookups_[stripe * sub_caches_.size() + level_index].v.load(
        std::memory_order_relaxed);
  }
  return sum;
}

uint64_t MultiLevelCache::SumHitCounter(size_t level_index) const {
  uint64_t sum = 0;
  for (size_t stripe = 0; stripe < kCounterStripes; ++stripe) {
    sum += hits_[stripe * sub_caches_.size() + level_index].v.load(
        std::memory_order_relaxed);
  }
  return sum;
}

void MultiLevelCache::IncFgLookupCounter(size_t level_index) {
  fg_lookups_[CounterStripeIndex(kCounterStripes) * sub_caches_.size() +
              level_index]
      .v.fetch_add(1, std::memory_order_relaxed);
}

void MultiLevelCache::IncFgHitCounter(size_t level_index) {
  fg_hits_[CounterStripeIndex(kCounterStripes) * sub_caches_.size() +
           level_index]
      .v.fetch_add(1, std::memory_order_relaxed);
}

uint64_t MultiLevelCache::SumFgLookupCounter(size_t level_index) const {
  uint64_t sum = 0;
  for (size_t stripe = 0; stripe < kCounterStripes; ++stripe) {
    sum += fg_lookups_[stripe * sub_caches_.size() + level_index].v.load(
        std::memory_order_relaxed);
  }
  return sum;
}

uint64_t MultiLevelCache::SumFgHitCounter(size_t level_index) const {
  uint64_t sum = 0;
  for (size_t stripe = 0; stripe < kCounterStripes; ++stripe) {
    sum += fg_hits_[stripe * sub_caches_.size() + level_index].v.load(
        std::memory_order_relaxed);
  }
  return sum;
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
  Cache* owner = nullptr;
  Cache::Handle* inner = nullptr;
  ResolveHandle(handle, &owner, &inner);
  return owner->GetUsage(inner);
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
  Cache* owner = nullptr;
  Cache::Handle* inner = nullptr;
  ResolveHandle(handle, &owner, &inner);
  return owner->GetCharge(inner);
}

const Cache::CacheItemHelper* MultiLevelCache::GetCacheItemHelper(
    Handle* handle) const {
  Cache* owner = nullptr;
  Cache::Handle* inner = nullptr;
  ResolveHandle(handle, &owner, &inner);
  return owner->GetCacheItemHelper(inner);
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
  Cache* owner = nullptr;
  Cache::Handle* inner = nullptr;
  ResolveHandle(handle, &owner, &inner);
  owner->ApplyToHandle(owner, inner, callback);
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
    total_lookups += SumLookupCounter(level);
    total_hits += SumHitCounter(level);
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
    const uint64_t level_lookups = SumLookupCounter(level);
    const uint64_t level_hits = SumHitCounter(level);
    const uint64_t level_data_size =
        level_data_sizes_[level].load(std::memory_order_relaxed);
    const bool probation_insert =
        level_probation_insert_enabled_[level].load(std::memory_order_relaxed);
    const double level_hit_rate =
        level_lookups == 0 ? 0.0
                           : static_cast<double>(level_hits) /
                                 static_cast<double>(level_lookups);
    oss << "L" << level << ": capacity=" << sub_caches_[level]->GetCapacity()
        << ", usage=" << sub_caches_[level]->GetUsage()
        << ", lookups=" << level_lookups << ", hits=" << level_hits
        << ", hit_rate=" << level_hit_rate
        << ", data_size=" << level_data_size
        << ", probation_insert=" << (probation_insert ? 1 : 0) << "\n";
  }
  return oss.str();
}

void MultiLevelCache::ResetStats() {
  const size_t counter_count = kCounterStripes * sub_caches_.size();
  for (size_t i = 0; i < counter_count; ++i) {
    lookups_[i].v.store(0, std::memory_order_relaxed);
    hits_[i].v.store(0, std::memory_order_relaxed);
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
  snapshot.fg_lookups.resize(sub_caches_.size());
  snapshot.fg_hits.resize(sub_caches_.size());
  snapshot.capacities.resize(sub_caches_.size());
  snapshot.usages.resize(sub_caches_.size());
  snapshot.data_sizes.resize(sub_caches_.size());
  snapshot.table_address_counts.resize(sub_caches_.size());
  snapshot.occupancy_counts.resize(sub_caches_.size());
  for (size_t level = 0; level < sub_caches_.size(); ++level) {
    snapshot.lookups[level] = SumLookupCounter(level);
    snapshot.hits[level] = SumHitCounter(level);
    snapshot.fg_lookups[level] = SumFgLookupCounter(level);
    snapshot.fg_hits[level] = SumFgHitCounter(level);
    snapshot.capacities[level] = sub_caches_[level]->GetCapacity();
    snapshot.usages[level] = sub_caches_[level]->GetUsage();
    snapshot.table_address_counts[level] =
        sub_caches_[level]->GetTableAddressCount();
    snapshot.occupancy_counts[level] = sub_caches_[level]->GetOccupancyCount();
    snapshot.data_sizes[level] =
        level_data_sizes_[level].load(std::memory_order_relaxed);
  }
  return snapshot;
}

std::vector<std::vector<uint64_t>> MultiLevelCache::DrainLookupSamples() {
  std::vector<std::vector<uint64_t>> drained(sub_caches_.size());
  for (size_t level = 0; level < sub_caches_.size(); ++level) {
    LevelSampleRing* ring = lookup_sample_rings_[level].get();
    if (ring == nullptr) {
      continue;
    }
    const uint64_t end = ring->write_seq.load(std::memory_order_acquire);
    uint64_t start = ring->drained_seq.load(std::memory_order_relaxed);
    if (end <= start) {
      continue;
    }
    if (end - start > kLookupSampleRingSize) {
      start = end - kLookupSampleRingSize;
    }
    drained[level].reserve(static_cast<size_t>(end - start));
    for (uint64_t seq = start; seq < end; ++seq) {
      const size_t idx = static_cast<size_t>(seq % kLookupSampleRingSize);
      if (ring->seq[idx].load(std::memory_order_acquire) == seq + 1) {
        drained[level].push_back(
            ring->values[idx].load(std::memory_order_relaxed));
      }
    }
    ring->drained_seq.store(end, std::memory_order_release);
  }
  return drained;
}

void MultiLevelCache::SetLookupSampleRateLog2(uint32_t sample_rate_log2) {
  // Guard overly sparse sampling and shift overflow.
  const uint32_t clamped = std::min<uint32_t>(sample_rate_log2, 20);
  lookup_sample_rate_log2_.store(clamped, std::memory_order_relaxed);
  // Caller expressed interest in samples (external driver e.g. db_bench);
  // ConfigureDynamicSRHCC overrides this right after for the dynamic path.
  lookup_sampling_enabled_.store(true, std::memory_order_relaxed);
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

void MultiLevelCache::ConfigureDynamicSRHCC(bool enabled,
                                            uint32_t check_interval_ops,
                                            uint32_t min_samples,
                                            double unique_ratio_enable_threshold,
                                            double unique_ratio_disable_threshold,
                                            uint32_t sample_rate_log2,
                                            uint32_t poll_interval_ms) {
  dynamic_srhcc_check_interval_ops_.store(std::max<uint32_t>(64, check_interval_ops),
                                          std::memory_order_relaxed);
  dynamic_srhcc_min_samples_.store(std::max<uint32_t>(8, min_samples),
                                   std::memory_order_relaxed);
  dynamic_srhcc_sample_rate_log2_.store(std::min<uint32_t>(20, sample_rate_log2),
                                        std::memory_order_relaxed);
  dynamic_srhcc_poll_interval_ms_.store(std::max<uint32_t>(10, poll_interval_ms),
                                        std::memory_order_relaxed);
  const double clamped_enable =
      std::max(0.0, std::min(1.0, unique_ratio_enable_threshold));
  const double clamped_disable_raw =
      std::max(0.0, std::min(1.0, unique_ratio_disable_threshold));
  const double clamped_disable =
      std::min(clamped_enable, clamped_disable_raw);
  dynamic_srhcc_unique_ratio_enable_ppm_.store(
      static_cast<uint32_t>(clamped_enable * kRatioScalePpm),
      std::memory_order_relaxed);
  dynamic_srhcc_unique_ratio_disable_ppm_.store(
      static_cast<uint32_t>(clamped_disable * kRatioScalePpm),
      std::memory_order_relaxed);
  SetLookupSampleRateLog2(dynamic_srhcc_sample_rate_log2_.load(std::memory_order_relaxed));
  lookup_sampling_enabled_.store(enabled, std::memory_order_relaxed);
  dynamic_srhcc_enabled_.store(enabled, std::memory_order_relaxed);
  if (enabled) {
    StartDynamicSRHCCWorker();
  } else {
    StopDynamicSRHCCWorker();
  }
}

size_t MultiLevelCache::RouteLevelByKey(const Slice& key, RouteCaller caller,
                                        Slice* base_key) const {
  if (base_key != nullptr) {
    *base_key = key;
  }
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
  size_t extended_level = 0;
  Slice extended_base_key;
  if (DecodeExtendedCacheRouting(key, &extended_level, &extended_base_key)) {
    if (base_key != nullptr) {
      *base_key = extended_base_key;
    }
    if (route_prefix_hits != nullptr) {
      route_prefix_hits->fetch_add(1, std::memory_order_relaxed);
    }
    return extended_level;
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
  if (route_prefix_misses != nullptr) {
    route_prefix_misses->fetch_add(1, std::memory_order_relaxed);
  }
  MaybeLogRouteMiss(caller, key, /*has_prefix=*/true, *key_prefix,
                    "prefix_marker_miss");
  return 0;
}

bool MultiLevelCache::DecodeExtendedCacheRouting(const Slice& key, size_t* level,
                                                 Slice* base_key) const {
  if (key.size() != kExtendedCacheKeySize) {
    return false;
  }
  int decoded_level = 0;
  const uint8_t level_tag = static_cast<uint8_t>(key.data()[kCacheKeySize]);
  if (!DecodeLevelFromCacheKeyLevelTag(level_tag, &decoded_level)) {
    return false;
  }
  if (decoded_level < 0 ||
      static_cast<size_t>(decoded_level) >= sub_caches_.size()) {
    route_normalize_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (level != nullptr) {
    *level = static_cast<size_t>(decoded_level);
  }
  if (base_key != nullptr) {
    *base_key = Slice(key.data(), kCacheKeySize);
  }
  return true;
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

void MultiLevelCache::BuildHandleOwnerRanges() {
  handle_owner_ranges_.clear();
  auto append_ranges = [this](Cache* cache) {
    if (cache == nullptr) {
      return;
    }
    std::vector<std::pair<const void*, const void*>> ranges;
    // Name()-based dispatch keeps this working in no-RTTI builds.
    const char* name = cache->Name();
    if (std::strcmp(name, "FixedHyperClockCache") == 0) {
      static_cast<clock_cache::FixedHyperClockCache*>(cache)
          ->AppendHandleAddressRanges(&ranges);
    } else if (std::strcmp(name, "AutoHyperClockCache") == 0) {
      static_cast<clock_cache::AutoHyperClockCache*>(cache)
          ->AppendHandleAddressRanges(&ranges);
    } else {
      // No stable handle array (e.g. LRU sub-caches): handles from this
      // cache always go through the WrappedHandle path.
      return;
    }
    for (const auto& range : ranges) {
      handle_owner_ranges_.push_back(
          {reinterpret_cast<uintptr_t>(range.first),
           reinterpret_cast<uintptr_t>(range.second), cache});
    }
  };
  for (const auto& sub_cache : sub_caches_) {
    append_ranges(sub_cache.get());
  }
  append_ranges(shared_cache_.get());
  std::sort(handle_owner_ranges_.begin(), handle_owner_ranges_.end(),
            [](const HandleOwnerRange& a, const HandleOwnerRange& b) {
              return a.begin < b.begin;
            });
}

Cache* MultiLevelCache::FindHandleOwner(const void* handle_addr) const {
  const uintptr_t addr = reinterpret_cast<uintptr_t>(handle_addr);
  // Binary search over at most (levels + 1) * shards ranges.
  size_t lo = 0;
  size_t hi = handle_owner_ranges_.size();
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (handle_owner_ranges_[mid].begin <= addr) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo == 0) {
    return nullptr;
  }
  const HandleOwnerRange& range = handle_owner_ranges_[lo - 1];
  return addr < range.end ? range.owner : nullptr;
}

Cache::Handle* MultiLevelCache::WrapOrPassHandle(Cache* owner_cache,
                                                 Cache::Handle* inner) {
  assert(inner != nullptr);
  assert(owner_cache != nullptr);
  if (FindHandleOwner(inner) == owner_cache) {
    // Common path (slot-array handle from an HCC sub-cache): the owner is
    // recoverable by address, so pass the handle through with no allocation.
    return inner;
  }
  // Standalone (heap-allocated) handle or non-HCC sub-cache: fall back to a
  // wrapper, marked by tagging the pointer's low bit (handles are at least
  // 8-byte aligned).
  auto* wrapped = new WrappedHandle();
  wrapped->owner_cache = owner_cache;
  wrapped->inner = inner;
  return reinterpret_cast<Handle*>(reinterpret_cast<uintptr_t>(wrapped) |
                                   kWrappedHandleTagBit);
}

void MultiLevelCache::ResolveHandle(const Handle* handle, Cache** owner,
                                    Cache::Handle** inner) const {
  assert(handle != nullptr);
  const uintptr_t bits = reinterpret_cast<uintptr_t>(handle);
  if (bits & kWrappedHandleTagBit) {
    const auto* wrapped =
        reinterpret_cast<const WrappedHandle*>(bits & ~kWrappedHandleTagBit);
    assert(wrapped->owner_cache != nullptr);
    assert(wrapped->inner != nullptr);
    *owner = wrapped->owner_cache;
    *inner = wrapped->inner;
    return;
  }
  *inner = const_cast<Cache::Handle*>(handle);
  *owner = FindHandleOwner(handle);
  // An untagged handle is only ever issued when its owner is resolvable.
  assert(*owner != nullptr);
  if (*owner == nullptr) {
    // Defensive (release builds): HCC handle accessors do not depend on the
    // particular instance.
    *owner = const_cast<MultiLevelCache*>(this)->PrimarySubCache();
  }
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
    for (const auto& sub_cache : sub_caches_) {
      PurgeSubCacheToCapacity(sub_cache.get());
    }
    PurgeSubCacheToCapacity(shared_cache_.get());
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
  // Apply all new capacities first, then reclaim, so that purges run against
  // the final targets (a level being grown is never purged spuriously).
  for (const auto& sub_cache : sub_caches_) {
    PurgeSubCacheToCapacity(sub_cache.get());
  }
  PurgeSubCacheToCapacity(shared_cache_.get());
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
  // Nobody consumes samples in non-dynamic configurations; skip all
  // bookkeeping (this runs on every Lookup).
  if (!lookup_sampling_enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  const uint32_t log2_rate =
      lookup_sample_rate_log2_.load(std::memory_order_relaxed);
  const uint64_t sample_mask =
      log2_rate == 0 ? 0 : ((1ULL << log2_rate) - 1ULL);
  // The sample rate only needs to hold statistically, so a thread-local
  // counter suffices; no reason to serialize all threads on one global
  // sequence atomic.
  thread_local uint64_t tls_sample_seq = 0;
  if ((tls_sample_seq++ & sample_mask) != 0) {
    return;
  }
  if (level_index >= lookup_sample_rings_.size()) {
    return;
  }
  LevelSampleRing* ring = lookup_sample_rings_[level_index].get();
  if (ring == nullptr) {
    return;
  }
  const uint64_t hash = static_cast<uint64_t>(
      std::hash<std::string_view>{}(std::string_view(key.data(), key.size())));
  const uint64_t seq =
      ring->write_seq.fetch_add(1, std::memory_order_relaxed);
  const size_t idx = static_cast<size_t>(seq % kLookupSampleRingSize);
  ring->values[idx].store(hash, std::memory_order_relaxed);
  ring->seq[idx].store(seq + 1, std::memory_order_release);
}

void MultiLevelCache::SetWorkingSetTrackingEnabled(bool enabled,
                                                   uint32_t sample_shift) {
  working_set_sample_shift_.store(sample_shift, std::memory_order_relaxed);
  working_set_tracking_enabled_.store(enabled, std::memory_order_relaxed);
}

void MultiLevelCache::RecordForegroundWorkingSet(size_t level_index,
                                                 const Slice& base_key) {
  if (level_index >= fg_working_set_.size()) {
    return;
  }
  ForegroundWorkingSetSketch* sketch = fg_working_set_[level_index].get();
  if (sketch == nullptr) {
    return;
  }
  const uint64_t hash = HashCacheKey(base_key);
  // Unbiased hash-gated sampling: keep only keys whose low `shift` bits are
  // zero. Uses low bits for the gate and high bits for the register index/rank,
  // so the two are independent. Drain scales the estimate back up by 2^shift.
  const uint32_t shift = working_set_sample_shift_.load(std::memory_order_relaxed);
  if (shift > 0 && (hash & ((uint64_t{1} << shift) - 1)) != 0) {
    return;
  }
  // Top kWssRegisterBitsLog2 bits select the register; the remaining bits give
  // the rank (1 + count of leading zeros). Shifting the index bits out and OR-
  // ing a sentinel at the boundary bounds the rank and keeps clz well-defined.
  const uint32_t idx =
      static_cast<uint32_t>(hash >> (64 - kWssRegisterBitsLog2));
  const uint64_t w = (hash << kWssRegisterBitsLog2) |
                     (uint64_t{1} << (kWssRegisterBitsLog2 - 1));
  const uint8_t rho =
      static_cast<uint8_t>(__builtin_clzll(w) + 1);
  std::atomic<uint8_t>& reg = sketch->registers[idx];
  uint8_t cur = reg.load(std::memory_order_relaxed);
  while (rho > cur) {
    if (reg.compare_exchange_weak(cur, rho, std::memory_order_relaxed)) {
      break;
    }
  }
}

std::vector<double> MultiLevelCache::DrainForegroundWorkingSetDistinct() {
  std::vector<double> result;
  if (!working_set_tracking_enabled_.load(std::memory_order_relaxed)) {
    return result;
  }
  const size_t levels = fg_working_set_.size();
  result.assign(levels, 0.0);
  const double m = static_cast<double>(kWssRegisterCount);
  // Standard HLL bias constant for m registers.
  const double alpha_m = 0.7213 / (1.0 + 1.079 / m);
  // Scale sampled cardinality back to the full stream (1/2^shift sampled).
  const double sample_scale = static_cast<double>(
      uint64_t{1} << working_set_sample_shift_.load(std::memory_order_relaxed));
  for (size_t level = 0; level < levels; ++level) {
    ForegroundWorkingSetSketch* sketch = fg_working_set_[level].get();
    if (sketch == nullptr) {
      continue;
    }
    double harmonic = 0.0;
    size_t zeros = 0;
    for (size_t i = 0; i < kWssRegisterCount; ++i) {
      // Read-and-reset so each drain yields a fresh window.
      const uint8_t r =
          sketch->registers[i].exchange(0, std::memory_order_relaxed);
      harmonic += std::ldexp(1.0, -static_cast<int>(r));
      if (r == 0) {
        ++zeros;
      }
    }
    double estimate = harmonic > 0.0 ? alpha_m * m * m / harmonic : 0.0;
    // Small-range correction: linear counting when many registers are empty.
    if (estimate <= 2.5 * m && zeros > 0) {
      estimate = m * std::log(m / static_cast<double>(zeros));
    }
    result[level] = estimate * sample_scale;
  }
  return result;
}

void MultiLevelCache::MaybeAdaptLevelMode(size_t level_index) {
  if (!dynamic_srhcc_enabled_.load(std::memory_order_relaxed) ||
      level_index >= sub_caches_.size()) {
    return;
  }
  const uint64_t curr_lookups = SumLookupCounter(level_index);
  const uint64_t prev_lookups =
      adapt_last_lookups_[level_index].load(std::memory_order_relaxed);
  const uint32_t interval =
      dynamic_srhcc_check_interval_ops_.load(std::memory_order_relaxed);
  if (curr_lookups <= prev_lookups || (curr_lookups - prev_lookups) < interval) {
    return;
  }

  const uint64_t curr_hits = SumHitCounter(level_index);

  std::vector<uint64_t> samples;
  LevelSampleRing* ring = lookup_sample_rings_[level_index].get();
  if (ring == nullptr) {
    return;
  }
  const uint64_t end = ring->write_seq.load(std::memory_order_acquire);
  uint64_t start = ring->consumed_seq.load(std::memory_order_relaxed);
  if (end <= start) {
    return;
  }
  if (end - start > kLookupSampleRingSize) {
    start = end - kLookupSampleRingSize;
  }
  samples.reserve(static_cast<size_t>(end - start));
  for (uint64_t seq = start; seq < end; ++seq) {
    const size_t idx = static_cast<size_t>(seq % kLookupSampleRingSize);
    if (ring->seq[idx].load(std::memory_order_acquire) == seq + 1) {
      samples.push_back(ring->values[idx].load(std::memory_order_relaxed));
    }
  }
  if (samples.size() <
      dynamic_srhcc_min_samples_.load(std::memory_order_relaxed)) {
    // Keep accumulating sample evidence and keep current lookup window.
    return;
  }

  adapt_last_lookups_[level_index].store(curr_lookups,
                                         std::memory_order_relaxed);
  adapt_last_hits_[level_index].store(curr_hits, std::memory_order_relaxed);
  ring->consumed_seq.store(end, std::memory_order_release);

  std::sort(samples.begin(), samples.end());
  const size_t unique_count = static_cast<size_t>(
      std::distance(samples.begin(), std::unique(samples.begin(), samples.end())));
  const double unique_ratio =
      static_cast<double>(unique_count) / static_cast<double>(samples.size());
  const double unique_enable_threshold =
      static_cast<double>(
          dynamic_srhcc_unique_ratio_enable_ppm_.load(
              std::memory_order_relaxed)) /
      kRatioScalePpm;
  const double unique_disable_threshold =
      static_cast<double>(
          dynamic_srhcc_unique_ratio_disable_ppm_.load(
              std::memory_order_relaxed)) /
      kRatioScalePpm;

  const bool current_mode =
      level_probation_insert_enabled_[level_index].load(std::memory_order_relaxed);
  bool target_mode = current_mode;
  if (!current_mode) {
    if (unique_ratio >= unique_enable_threshold) {
      target_mode = true;
    }
  } else if (unique_ratio <= unique_disable_threshold) {
    target_mode = false;
  }

  if (target_mode != current_mode &&
      MaybeSetLevelProbationInsert(level_index, target_mode)) {
    level_probation_insert_enabled_[level_index].store(target_mode,
                                                       std::memory_order_relaxed);
  }
}

bool MultiLevelCache::MaybeSetLevelProbationInsert(size_t level_index,
                                                   bool probation_insert) {
  Cache* cache = SubCacheByLevel(level_index);
  if (cache == nullptr) {
    return false;
  }
  // Name()-based dispatch instead of dynamic_cast so this also works in
  // builds with RTTI disabled (release default).
  const char* name = cache->Name();
  if (std::strcmp(name, "FixedHyperClockCache") == 0) {
    static_cast<clock_cache::FixedHyperClockCache*>(cache)->SetProbationInsert(
        probation_insert);
    return true;
  }
  if (std::strcmp(name, "AutoHyperClockCache") == 0) {
    static_cast<clock_cache::AutoHyperClockCache*>(cache)->SetProbationInsert(
        probation_insert);
    return true;
  }
  return false;
}

void MultiLevelCache::StartDynamicSRHCCWorker() {
  std::lock_guard<std::mutex> lock(dynamic_srhcc_mu_);
  if (dynamic_srhcc_running_.load(std::memory_order_acquire)) {
    return;
  }
  dynamic_srhcc_running_.store(true, std::memory_order_release);
  dynamic_srhcc_worker_ =
      std::thread(&MultiLevelCache::DynamicSRHCCBackgroundLoop, this);
}

void MultiLevelCache::StopDynamicSRHCCWorker() {
  {
    std::lock_guard<std::mutex> lock(dynamic_srhcc_mu_);
    if (!dynamic_srhcc_running_.load(std::memory_order_acquire)) {
      return;
    }
    dynamic_srhcc_running_.store(false, std::memory_order_release);
  }
  dynamic_srhcc_cv_.notify_all();
  if (dynamic_srhcc_worker_.joinable()) {
    dynamic_srhcc_worker_.join();
  }
}

void MultiLevelCache::DynamicSRHCCBackgroundLoop() {
  while (dynamic_srhcc_running_.load(std::memory_order_acquire)) {
    EvaluateDynamicSRHCCAllLevels();
    std::unique_lock<std::mutex> lock(dynamic_srhcc_mu_);
    dynamic_srhcc_cv_.wait_for(
        lock,
        std::chrono::milliseconds(
            dynamic_srhcc_poll_interval_ms_.load(std::memory_order_relaxed)),
        [this]() { return !dynamic_srhcc_running_.load(std::memory_order_acquire); });
  }
}

void MultiLevelCache::EvaluateDynamicSRHCCAllLevels() {
  if (!dynamic_srhcc_enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  for (size_t level = 0; level < sub_caches_.size(); ++level) {
    MaybeAdaptLevelMode(level);
  }
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
  // Value-initialized: all stripes start at zero.
  lookups_ = std::make_unique<StripedCounter[]>(kCounterStripes * level_count);
  hits_ = std::make_unique<StripedCounter[]>(kCounterStripes * level_count);
  fg_lookups_ = std::make_unique<StripedCounter[]>(kCounterStripes * level_count);
  fg_hits_ = std::make_unique<StripedCounter[]>(kCounterStripes * level_count);
  level_data_sizes_.resize(level_count);
  lookup_sample_rings_.resize(level_count);
  fg_working_set_.resize(level_count);
  adapt_last_lookups_.resize(level_count);
  adapt_last_hits_.resize(level_count);
  level_probation_insert_enabled_.resize(level_count);
  for (size_t level = 0; level < level_count; ++level) {
    level_data_sizes_[level].store(0, std::memory_order_relaxed);
    fg_working_set_[level] = std::make_unique<ForegroundWorkingSetSketch>();
    lookup_sample_rings_[level] = std::make_unique<LevelSampleRing>();
    lookup_sample_rings_[level]->seq =
        std::make_unique<std::atomic<uint64_t>[]>(kLookupSampleRingSize);
    lookup_sample_rings_[level]->values =
        std::make_unique<std::atomic<uint64_t>[]>(kLookupSampleRingSize);
    lookup_sample_rings_[level]->write_seq.store(0, std::memory_order_relaxed);
    lookup_sample_rings_[level]->consumed_seq.store(0,
                                                    std::memory_order_relaxed);
    lookup_sample_rings_[level]->drained_seq.store(0,
                                                   std::memory_order_relaxed);
    for (size_t i = 0; i < kLookupSampleRingSize; ++i) {
      lookup_sample_rings_[level]->seq[i].store(0, std::memory_order_relaxed);
      lookup_sample_rings_[level]->values[i].store(0, std::memory_order_relaxed);
    }
    adapt_last_lookups_[level].store(0, std::memory_order_relaxed);
    adapt_last_hits_[level].store(0, std::memory_order_relaxed);
    level_probation_insert_enabled_[level].store(false,
                                                 std::memory_order_relaxed);
    // Best effort initialization for HCC/SR-HCC sub-caches.
    const bool initialized = MaybeSetLevelProbationInsert(level, false);
    if (initialized) {
      level_probation_insert_enabled_[level].store(false,
                                                   std::memory_order_relaxed);
    }
  }
  BuildHandleOwnerRanges();
}

}  // namespace ROCKSDB_NAMESPACE
