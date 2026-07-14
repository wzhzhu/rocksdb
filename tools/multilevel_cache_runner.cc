// Copyright (c) 2011-present, Facebook, Inc.
// This source code is licensed under both the GPLv2 and Apache 2.0 License.

#include <cstdint>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cache/multi_level_cache_allocator.h"
#include "cache/multi_level_cache.h"
#include "cache/cache_key.h"
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
  if (num_levels > static_cast<uint64_t>(ROCKSDB_NAMESPACE::kMaxEncodedCacheKeyLevel + 1)) {
    std::cerr << "num_levels=" << num_levels
              << " exceeds encoded route limit "
              << (ROCKSDB_NAMESPACE::kMaxEncodedCacheKeyLevel + 1) << "\n";
    return 1;
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

  // Online demo: periodically solve capacities from real cache stats.
  auto snapshot = ml_cache->GetLevelMetricsSnapshot();
  std::vector<uint64_t> prev_lookups = snapshot.lookups;
  ROCKSDB_NAMESPACE::MultiLevelCacheAllocator::MetricsProvider provider =
      [ml_cache, prev_lookups](std::vector<double>* lambda,
                               std::vector<double>* data,
                               std::vector<double>* alpha,
                               uint64_t* /*l0_file_count*/,
                               uint64_t* /*stall_micros*/) mutable {
        if (lambda == nullptr || data == nullptr || alpha == nullptr) {
          return false;
        }
        const auto stats = ml_cache->GetLevelMetricsSnapshot();
        const size_t level_count = stats.lookups.size();
        if (level_count == 0 || prev_lookups.size() != level_count) {
          prev_lookups = stats.lookups;
          return false;
        }

        lambda->assign(level_count, 1.0);
        data->assign(level_count, 1.0);
        alpha->assign(level_count, 1.0);
        uint64_t total_observed_data = 0;
        size_t observed_levels = 0;
        for (size_t level = 0; level < level_count; ++level) {
          if (stats.data_sizes[level] > 0) {
            total_observed_data += stats.data_sizes[level];
            ++observed_levels;
          }
        }
        const double default_data = observed_levels > 0
                                        ? static_cast<double>(total_observed_data) /
                                              static_cast<double>(observed_levels)
                                        : 1.0;
        constexpr double kLambdaEpsilon = 1e-6;
        for (size_t level = 0; level < level_count; ++level) {
          const uint64_t curr = stats.lookups[level];
          const uint64_t prev = prev_lookups[level];
          const uint64_t delta = curr >= prev ? curr - prev : 0;
          (*lambda)[level] =
              delta > 0 ? static_cast<double>(delta) : kLambdaEpsilon;
          (*data)[level] = stats.data_sizes[level] > 0
                               ? static_cast<double>(stats.data_sizes[level])
                               : default_data;
          (*alpha)[level] = 1.0;
        }
        prev_lookups = stats.lookups;
        return true;
      };

  ROCKSDB_NAMESPACE::MultiLevelAllocationOptions alloc_opts;
  alloc_opts.interval_ms = 200;
  alloc_opts.smoothing_ratio = 1.0;
  alloc_opts.min_total_change_bytes = 0;
  ROCKSDB_NAMESPACE::MultiLevelCacheAllocator allocator(ml_cache, provider,
                                                        alloc_opts);
  allocator.Start();

  // Run another round of reads while allocator periodically updates capacities.
  for (uint64_t i = 0; i < num_keys; ++i) {
    s = db->Get(ROCKSDB_NAMESPACE::ReadOptions(), "k" + std::to_string(i),
                &value);
    if (!s.ok() && !s.IsNotFound()) {
      std::cerr << "Get failed at " << i << " in adaptive pass: "
                << s.ToString() << "\n";
      allocator.Stop();
      return 1;
    }
    if ((i % 1024) == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  allocator.Stop();

  auto after_alloc = ml_cache->GetLevelMetricsSnapshot();
  std::cout << "Allocator capacities:";
  for (size_t cap : after_alloc.capacities) {
    std::cout << " " << cap;
  }
  std::cout << "\n";

  std::cout << "=== MultiLevelCache Stats (after adjust) ===\n";
  std::cout << ml_cache->PrintStats();
  std::cout << "Runner finished successfully.\n";

  return 0;
}
