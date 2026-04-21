// Copyright (c) 2011-present, Facebook, Inc.
// This source code is licensed under both the GPLv2 and Apache 2.0 License.

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cache/multi_level_cache.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"

namespace {

void PrintUsage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " [db_path] [cache_mb] [num_levels] [num_keys]\n";
}

bool ParseUint64(const std::string& s, uint64_t* out) {
  if (out == nullptr) {
    return false;
  }
  try {
    *out = std::stoull(s);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string db_path = "/tmp/rocksdb_multilevel_runner";
  uint64_t cache_mb = 512;
  uint64_t num_levels = 7;
  uint64_t num_keys = 200000;

  if (argc > 1) {
    db_path = argv[1];
  }
  if (argc > 2 && !ParseUint64(argv[2], &cache_mb)) {
    PrintUsage(argv[0]);
    return 1;
  }
  if (argc > 3 && !ParseUint64(argv[3], &num_levels)) {
    PrintUsage(argv[0]);
    return 1;
  }
  if (argc > 4 && !ParseUint64(argv[4], &num_keys)) {
    PrintUsage(argv[0]);
    return 1;
  }
  if (num_levels == 0) {
    num_levels = 1;
  }

  const size_t total_cache_bytes = static_cast<size_t>(cache_mb) * 1024 * 1024;
  auto ml_cache = std::make_shared<ROCKSDB_NAMESPACE::MultiLevelCache>(
      static_cast<size_t>(num_levels), total_cache_bytes);

  ROCKSDB_NAMESPACE::BlockBasedTableOptions table_opts;
  table_opts.block_cache = ml_cache;

  ROCKSDB_NAMESPACE::Options opts;
  opts.create_if_missing = true;
  opts.num_levels = static_cast<int>(num_levels);
  opts.table_factory.reset(
      ROCKSDB_NAMESPACE::NewBlockBasedTableFactory(table_opts));

  std::unique_ptr<ROCKSDB_NAMESPACE::DB> db;
  ROCKSDB_NAMESPACE::Status s =
      ROCKSDB_NAMESPACE::DB::Open(opts, db_path, &db);
  if (!s.ok()) {
    std::cerr << "DB::Open failed: " << s.ToString() << "\n";
    return 1;
  }

  // Fill with deterministic key space.
  for (uint64_t i = 0; i < num_keys; ++i) {
    s = db->Put(ROCKSDB_NAMESPACE::WriteOptions(), "k" + std::to_string(i),
                "v" + std::to_string(i));
    if (!s.ok()) {
      std::cerr << "Put failed at " << i << ": " << s.ToString() << "\n";
      return 1;
    }
  }

  ROCKSDB_NAMESPACE::FlushOptions flush_opts;
  flush_opts.wait = true;
  s = db->Flush(flush_opts);
  if (!s.ok()) {
    std::cerr << "Flush failed: " << s.ToString() << "\n";
    return 1;
  }

  ROCKSDB_NAMESPACE::CompactRangeOptions compact_opts;
  s = db->CompactRange(compact_opts, nullptr, nullptr);
  if (!s.ok()) {
    std::cerr << "CompactRange failed: " << s.ToString() << "\n";
    return 1;
  }

  ml_cache->ResetStats();

  std::string value;
  for (uint64_t i = 0; i < num_keys; ++i) {
    s = db->Get(ROCKSDB_NAMESPACE::ReadOptions(), "k" + std::to_string(i),
                &value);
    if (!s.ok() && !s.IsNotFound()) {
      std::cerr << "Get failed at " << i << ": " << s.ToString() << "\n";
      return 1;
    }
  }

  std::cout << "=== MultiLevelCache Stats (before adjust) ===\n";
  std::cout << ml_cache->PrintStats();

  // Example of runtime capacity reconfiguration: make the deepest level larger.
  std::vector<size_t> new_caps(static_cast<size_t>(num_levels), 0);
  size_t consumed = 0;
  for (size_t level = 0; level < new_caps.size(); ++level) {
    size_t cap = total_cache_bytes / (new_caps.size() * 2);
    if (level + 1 == new_caps.size()) {
      cap = total_cache_bytes - consumed;
    }
    new_caps[level] = cap;
    consumed += cap;
  }
  s = ml_cache->AdjustCapacities(new_caps);
  if (!s.ok()) {
    std::cerr << "AdjustCapacities failed: " << s.ToString() << "\n";
    return 1;
  }

  std::cout << "=== MultiLevelCache Stats (after adjust) ===\n";
  std::cout << ml_cache->PrintStats();
  std::cout << "Runner finished successfully.\n";

  return 0;
}
