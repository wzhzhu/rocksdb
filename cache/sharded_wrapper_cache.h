#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "cache/wrapper_cache_shard.h"
#include "rocksdb/advanced_cache.h"
#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

// Routes Lookup/Insert/Erase to one of 2^k wrapper-policy shards (ARCCache or
// CacheusCache instances) by key hash, while all shards share one (natively
// sharded) backing cache. Handles are owned by the backing cache, so all
// handle-centric operations (Release/Ref/Value/...) are forwarded straight to
// it via the CacheWrapper base with target_ == backing.
class ShardedWrapperCache : public CacheWrapper {
 public:
  static const char* kClassName() { return "ShardedWrapperCache"; }

  // `shards` must share `backing` as their target and be constructed in
  // non-backing-managing mode (no eviction callback installed, no capacity
  // propagation). `policies` are the same objects as `shards` viewed through
  // their WrapperCacheShard interface (passed explicitly because RocksDB may
  // build without RTTI). `shards.size()` must be a power of two.
  // `backing_capacity_of_logical` maps a logical wrapper capacity to the
  // over-provisioned physical backing capacity.
  ShardedWrapperCache(
      std::shared_ptr<Cache> backing,
      std::vector<std::shared_ptr<Cache>> shards,
      std::vector<WrapperCacheShard*> policies, std::string stats_prefix,
      uint32_t hash_seed,
      std::function<size_t(size_t)> backing_capacity_of_logical);

  const char* Name() const override { return kClassName(); }
  std::string GetPrintableOptions() const override;

  Status Insert(const Slice& key, ObjectPtr value,
                const CacheItemHelper* helper, size_t charge,
                Handle** handle = nullptr, Priority priority = Priority::LOW,
                const Slice& compressed_value = Slice(),
                CompressionType type = CompressionType::kNoCompression) override;

  Handle* Lookup(const Slice& key, const CacheItemHelper* helper,
                 CreateContext* create_context,
                 Priority priority = Priority::LOW,
                 Statistics* stats = nullptr) override;

  void Erase(const Slice& key) override;
  void SetCapacity(size_t capacity) override;
  size_t GetCapacity() const override;

 private:
  size_t ShardIndex(const Slice& key) const;

  std::vector<std::shared_ptr<Cache>> shards_;
  std::vector<WrapperCacheShard*> policies_;
  std::string stats_prefix_;
  uint32_t shard_mask_;
  uint32_t hash_seed_;
  std::function<size_t(size_t)> backing_capacity_of_logical_;
};

}  // namespace ROCKSDB_NAMESPACE
