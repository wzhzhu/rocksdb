#pragma once

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

// Thread-local signal used to mark block-cache lookups issued by compaction.
// Compaction reads pollute the MLC allocator's per-level hit/miss model with
// streaming, low-reuse accesses that do not reflect foreground query value.
// RocksDB already avoids inserting compaction-read blocks into the cache
// (fill_cache=false for compaction); this flag extends the same principle to
// the MLC *counters* so the estimator/solver only see foreground accesses.
//
// Set the flag with MLCLookupCompactionScope around a cache lookup on the
// compaction path; MultiLevelCache::Lookup reads it to skip counter bumps.
bool MLCLookupIsCompaction();

// RAII guard: saves the previous per-thread flag, sets it to `is_compaction`
// for the scope, and restores it on destruction. Nested scopes take the
// innermost value while active and restore the enclosing value on exit. In
// practice reads do not cross the foreground/compaction boundary when nested
// (a compaction read only nests further compaction reads, and vice versa), so
// the effective tag is stable for a given logical read.
class MLCLookupCompactionScope {
 public:
  explicit MLCLookupCompactionScope(bool is_compaction);
  ~MLCLookupCompactionScope();
  MLCLookupCompactionScope(const MLCLookupCompactionScope&) = delete;
  MLCLookupCompactionScope& operator=(const MLCLookupCompactionScope&) = delete;

 private:
  bool prev_;
};

}  // namespace ROCKSDB_NAMESPACE
