#include "cache/sharded_wrapper_cache.h"

#include <cassert>
#include <sstream>

#include "util/hash.h"

namespace ROCKSDB_NAMESPACE {

ShardedWrapperCache::ShardedWrapperCache(
    std::shared_ptr<Cache> backing, std::vector<std::shared_ptr<Cache>> shards,
    std::vector<WrapperCacheShard*> policies, std::string stats_prefix,
    uint32_t hash_seed,
    std::function<size_t(size_t)> backing_capacity_of_logical)
    : CacheWrapper(std::move(backing)),
      shards_(std::move(shards)),
      policies_(std::move(policies)),
      stats_prefix_(std::move(stats_prefix)),
      shard_mask_(static_cast<uint32_t>(shards_.size()) - 1),
      hash_seed_(hash_seed),
      backing_capacity_of_logical_(std::move(backing_capacity_of_logical)) {
  assert(!shards_.empty());
  assert((shards_.size() & (shards_.size() - 1)) == 0);
  assert(policies_.size() == shards_.size());
  // Single dispatcher for the shared backing cache: route the eviction
  // notification to the wrapper shard owning the key. Individual shards must
  // not install their own callback (they would overwrite each other).
  target_->SetEvictionCallback(
      [this](const Slice& key, Handle* /*h*/, bool /*was_hit*/) {
        policies_[ShardIndex(key)]->HandleBackingEviction(key);
        return false;
      });
}

size_t ShardedWrapperCache::ShardIndex(const Slice& key) const {
  return Lower32of64(GetSliceNPHash64(key, hash_seed_)) & shard_mask_;
}

Status ShardedWrapperCache::Insert(const Slice& key, ObjectPtr value,
                                   const CacheItemHelper* helper, size_t charge,
                                   Handle** handle, Priority priority,
                                   const Slice& compressed_value,
                                   CompressionType type) {
  return shards_[ShardIndex(key)]->Insert(key, value, helper, charge, handle,
                                          priority, compressed_value, type);
}

Cache::Handle* ShardedWrapperCache::Lookup(const Slice& key,
                                           const CacheItemHelper* helper,
                                           CreateContext* create_context,
                                           Priority priority,
                                           Statistics* stats) {
  return shards_[ShardIndex(key)]->Lookup(key, helper, create_context, priority,
                                          stats);
}

void ShardedWrapperCache::Erase(const Slice& key) {
  shards_[ShardIndex(key)]->Erase(key);
}

void ShardedWrapperCache::SetCapacity(size_t capacity) {
  const size_t num_shards = shards_.size();
  const size_t per_shard = (capacity + num_shards - 1) / num_shards;
  for (const auto& shard : shards_) {
    shard->SetCapacity(per_shard);
  }
  // Shards run in non-backing-managing mode, so the router owns the single
  // physical capacity update for the shared backing cache.
  target_->SetCapacity(backing_capacity_of_logical_(capacity));
}

void ShardedWrapperCache::ResetWrapperCounters() {
  for (auto* policy : policies_) {
    policy->ResetWrapperCounters();
  }
}

size_t ShardedWrapperCache::GetCapacity() const {
  size_t total = 0;
  for (const auto& shard : shards_) {
    total += shard->GetCapacity();
  }
  return total;
}

std::string ShardedWrapperCache::GetPrintableOptions() const {
  uint64_t total_lookups = 0;
  uint64_t total_hits = 0;
  for (const auto* policy : policies_) {
    uint64_t lookups = 0;
    uint64_t hits = 0;
    policy->GetWrapperCounters(&lookups, &hits);
    total_lookups += lookups;
    total_hits += hits;
  }
  const double hit_ratio =
      total_lookups > 0
          ? static_cast<double>(total_hits) / static_cast<double>(total_lookups)
          : 0.0;
  std::ostringstream oss;
  oss << stats_prefix_ << ".wrapper_shards=" << shards_.size() << "\n";
  oss << stats_prefix_ << ".capacity=" << GetCapacity() << "\n";
  oss << stats_prefix_ << ".wrapper_lookups=" << total_lookups << "\n";
  oss << stats_prefix_ << ".wrapper_hits=" << total_hits << "\n";
  oss << stats_prefix_ << ".wrapper_hit_ratio=" << hit_ratio;
  return oss.str();
}

}  // namespace ROCKSDB_NAMESPACE
