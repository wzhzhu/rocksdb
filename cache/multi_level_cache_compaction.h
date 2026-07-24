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

// Thread-local user key of the in-flight point read (Get path), exposed to
// MultiLevelCache::RecordGhostMiss so the ghost fingerprint can be the USER
// KEY hash instead of the block cache key hash. Block cache keys die on every
// compaction rewrite (new file number => new key), so block-keyed ghost
// entries recorded before a rewrite can never match the re-miss after it --
// on high-churn levels most genuine repeats never register and the allocator
// under-values exactly the levels compaction touches most. User keys are
// churn-immune: the same logical record fingerprints identically no matter
// which file its block currently lives in.
// Returns false (and leaves *key_data/*key_size untouched) when no point-read
// user key is in scope (iterators, scans, compaction reads).
bool MLCLookupUserKey(const char** key_data, size_t* key_size);

// RAII guard installing the user key for the scope of a block retrieval.
// Pass data=nullptr to explicitly clear (e.g. non-point-read paths). Restores
// the enclosing value on destruction, like MLCLookupCompactionScope.
class MLCLookupUserKeyScope {
 public:
  MLCLookupUserKeyScope(const char* key_data, size_t key_size);
  ~MLCLookupUserKeyScope();
  MLCLookupUserKeyScope(const MLCLookupUserKeyScope&) = delete;
  MLCLookupUserKeyScope& operator=(const MLCLookupUserKeyScope&) = delete;

 private:
  const char* prev_data_;
  size_t prev_size_;
};

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
