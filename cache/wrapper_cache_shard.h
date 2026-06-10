#pragma once

#include <cstdint>

#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

// Interface implemented by wrapper-policy caches (ARC, Cacheus) so that a
// sharding router can aggregate wrapper-level statistics and dispatch
// backing-cache eviction notifications to the shard owning the key.
class WrapperCacheShard {
 public:
  virtual ~WrapperCacheShard() = default;

  // Returns wrapper-policy lookup/hit counters (policy-level, not backing).
  virtual void GetWrapperCounters(uint64_t* lookups, uint64_t* hits) const = 0;

  // Notification that the shared backing cache evicted `key`. Must be safe to
  // call from the backing cache's eviction callback.
  virtual void HandleBackingEviction(const Slice& key) = 0;
};

}  // namespace ROCKSDB_NAMESPACE
