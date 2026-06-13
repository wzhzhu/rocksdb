//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "cache/cacheus_cache.h"

#include <atomic>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <list>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rocksdb/cache.h"
#include "rocksdb/utilities/options_type.h"
#include "port/stack_trace.h"
#include "test_util/testharness.h"
#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {
namespace {

std::string EncodeKey(uint32_t v) {
  std::string key;
  PutFixed32(&key, v);
  return key;
}

Cache::ObjectPtr EncodeValue(uintptr_t v) {
  return reinterpret_cast<Cache::ObjectPtr>(v);
}

int DecodeValue(Cache::ObjectPtr v) {
  return static_cast<int>(reinterpret_cast<uintptr_t>(v));
}

int DecodeTrackedKey(const std::string& key) {
  if (key.size() != sizeof(uint32_t)) {
    return -1;
  }
  return static_cast<int>(DecodeFixed32(key.data()));
}

const Cache::CacheItemHelper kHelper{
    CacheEntryRole::kMisc,
    [](Cache::ObjectPtr /*value*/, MemoryAllocator* /*alloc*/) {}};

// Exact LRU over uint32 keys, capacity in entries. Returns true on hit.
class RefLRU {
 public:
  explicit RefLRU(size_t cap) : cap_(cap) {}
  bool Access(uint32_t k) {
    auto it = pos_.find(k);
    if (it != pos_.end()) {
      lru_.splice(lru_.begin(), lru_, it->second);
      return true;
    }
    if (lru_.size() >= cap_) {
      pos_.erase(lru_.back());
      lru_.pop_back();
    }
    lru_.push_front(k);
    pos_[k] = lru_.begin();
    return false;
  }

 private:
  size_t cap_;
  std::list<uint32_t> lru_;
  std::unordered_map<uint32_t, std::list<uint32_t>::iterator> pos_;
};

// Zipfian sampler over [0, n) with precomputed CDF. Fixed seed -> fixed
// sequence, so two instances with the same seed yield identical traces.
class Zipf {
 public:
  Zipf(uint32_t n, double alpha, uint64_t seed) : rng_(seed), u_(0.0, 1.0) {
    cdf_.resize(n);
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
      sum += 1.0 / std::pow(static_cast<double>(i) + 1.0, alpha);
      cdf_[i] = sum;
    }
    norm_ = sum;
  }
  uint32_t Next() {
    const double r = u_(rng_) * norm_;
    return static_cast<uint32_t>(
        std::lower_bound(cdf_.begin(), cdf_.end(), r) - cdf_.begin());
  }

 private:
  std::vector<double> cdf_;
  double norm_ = 0.0;
  std::mt19937_64 rng_;
  std::uniform_real_distribution<double> u_;
};

double ParsePrintableField(const std::string& s, const std::string& key) {
  const std::string needle = key + "=";
  size_t pos = s.find(needle);
  if (pos == std::string::npos) {
    return -1.0;
  }
  pos += needle.size();
  return std::strtod(s.c_str() + pos, nullptr);
}

}  // namespace

TEST(CacheusCacheTest, BasicInsertLookup) {
  auto cache = NewCacheusCache(1 << 20, 0);
  ASSERT_NE(cache, nullptr);
  ASSERT_STREQ(cache->Name(), "CacheusCache");

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_OK(cache->Insert(EncodeKey(i), EncodeValue(i + 100), &kHelper, 1));
  }
  for (uint32_t i = 0; i < 8; ++i) {
    auto* h = cache->Lookup(EncodeKey(i));
    ASSERT_NE(h, nullptr);
    ASSERT_EQ(DecodeValue(cache->Value(h)), static_cast<int>(i + 100));
    cache->Release(h);
  }
}

TEST(CacheusCacheTest, CreateFromString) {
  std::shared_ptr<Cache> cache;
  ConfigOptions config_options;
  ASSERT_OK(Cache::CreateFromString(
      config_options,
      "cacheus://capacity=4096;num_shard_bits=2;initial_weight=0.8;"
      "learning_rate=0.2;history_size=64;period_len=128;"
      "entry_charge_equivalent=true",
      &cache));
  ASSERT_NE(cache, nullptr);
  ASSERT_STREQ(cache->Name(), "CacheusCache");

  std::shared_ptr<Cache> default_cache;
  ASSERT_OK(Cache::CreateFromString(config_options, "cacheus://", &default_cache));
  ASSERT_NE(default_cache, nullptr);
  ASSERT_STREQ(default_cache->Name(), "CacheusCache");
}

TEST(CacheusCacheTest, CapacityBounded) {
  auto cache = NewCacheusCache(1 << 20, 0);
  ASSERT_NE(cache, nullptr);
  auto* typed = dynamic_cast<CacheusCache*>(cache.get());
  ASSERT_NE(typed, nullptr);

  for (uint32_t i = 0; i < 6; ++i) {
    ASSERT_OK(cache->Insert(EncodeKey(i), EncodeValue(i), &kHelper, 4096));
  }
  auto snap = typed->TEST_GetSnapshot();
  ASSERT_LE(snap.logical_usage, cache->GetCapacity());
}

TEST(CacheusCacheTest, TrajectoryAlignmentWithPythonReference) {
  struct ExpectedStep {
    double w_lru;
    double w_lfu;
    size_t s_limit;
    size_t q_limit;
    size_t s_len;
    size_t q_len;
    size_t lru_hist_len;
    size_t lfu_hist_len;
    size_t dem_count = 0;
    size_t nor_count = 0;
    uint64_t period_hits = 0;
    uint64_t evict_lru_count = 0;
    uint64_t evict_lfu_count = 0;
    uint64_t evict_tie_count = 0;
    int last_evicted_policy = -2;
    int last_evicted_key = -1;
  };
  struct Scenario {
    std::string name;
    size_t cache_size;
    double initial_weight;
    double learning_rate;
    size_t history_size;
    uint64_t period_len;
    std::vector<uint32_t> reqs;
    std::vector<ExpectedStep> expected;
  };
  const std::vector<Scenario> scenarios = {
#include "cache/cacheus_cache_test_data.inc"
  };

  for (const auto& scenario : scenarios) {
    LRUCacheOptions backing_opts;
    backing_opts.capacity = scenario.cache_size;
    backing_opts.num_shard_bits = 0;
    CacheusTuningOptions tuning;
    tuning.initial_weight = scenario.initial_weight;
    tuning.learning_rate = scenario.learning_rate;
    tuning.history_size = scenario.history_size;
    tuning.period_len = scenario.period_len;
    tuning.rng_seed = 123;

    auto cache = NewCacheusCache(backing_opts, tuning);
    ASSERT_NE(cache, nullptr);
    auto* typed = dynamic_cast<CacheusCache*>(cache.get());
    ASSERT_NE(typed, nullptr);
    ASSERT_EQ(scenario.reqs.size(), scenario.expected.size());

    for (size_t i = 0; i < scenario.reqs.size(); ++i) {
      const std::string key = EncodeKey(scenario.reqs[i]);
      typed->TEST_RequestStep(key, 1);
      CacheusCache::DebugSnapshot snap = typed->TEST_GetSnapshot();
      EXPECT_NEAR(snap.w_lru, scenario.expected[i].w_lru, 1e-5)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_NEAR(snap.w_lfu, scenario.expected[i].w_lfu, 1e-5)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.s_limit, scenario.expected[i].s_limit)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.q_limit, scenario.expected[i].q_limit)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.s_len, scenario.expected[i].s_len)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.q_len, scenario.expected[i].q_len)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.lru_hist_len, scenario.expected[i].lru_hist_len)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.lfu_hist_len, scenario.expected[i].lfu_hist_len)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.dem_count, scenario.expected[i].dem_count)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.nor_count, scenario.expected[i].nor_count)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.period_hits, scenario.expected[i].period_hits)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.evict_lru_count, scenario.expected[i].evict_lru_count)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.evict_lfu_count, scenario.expected[i].evict_lfu_count)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.evict_tie_count, scenario.expected[i].evict_tie_count)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(snap.last_evicted_policy, scenario.expected[i].last_evicted_policy)
          << "scenario=" << scenario.name << " step=" << i;
      EXPECT_EQ(DecodeTrackedKey(snap.last_evicted_key),
                scenario.expected[i].last_evicted_key)
          << "scenario=" << scenario.name << " step=" << i;
    }
  }
}

TEST(CacheusCacheTest, TinyCapacityChurnStability) {
  auto cache = NewCacheusCache(1, 0);
  ASSERT_NE(cache, nullptr);
  auto* typed = dynamic_cast<CacheusCache*>(cache.get());
  ASSERT_NE(typed, nullptr);

  for (uint32_t i = 0; i < 500; ++i) {
    typed->TEST_RequestStep(EncodeKey(i % 7), 1);
  }
  auto snap = typed->TEST_GetSnapshot();
  ASSERT_LE(cache->GetUsage(), cache->GetCapacity());
  ASSERT_GE(snap.evict_lru_count + snap.evict_lfu_count + snap.evict_tie_count, 1U);
}

TEST(CacheusCacheTest, FrequentSetCapacityRegression) {
  auto cache = NewCacheusCache(1 << 16, 0);
  ASSERT_NE(cache, nullptr);
  auto* typed = dynamic_cast<CacheusCache*>(cache.get());
  ASSERT_NE(typed, nullptr);

  for (uint32_t i = 0; i < 200; ++i) {
    const size_t cap = (i % 2 == 0) ? (1 << 16) : (1 << 12);
    cache->SetCapacity(cap);
    ASSERT_OK(cache->Insert(EncodeKey(i), EncodeValue(i), &kHelper, 256));
    auto snap = typed->TEST_GetSnapshot();
    ASSERT_LE(snap.logical_usage, cache->GetCapacity());
  }
}

TEST(CacheusCacheTest, LargeChargeEntriesRegression) {
  auto cache = NewCacheusCache(1 << 20, 0);
  ASSERT_NE(cache, nullptr);
  auto* typed = dynamic_cast<CacheusCache*>(cache.get());
  ASSERT_NE(typed, nullptr);

  for (uint32_t i = 0; i < 16; ++i) {
    ASSERT_OK(cache->Insert(EncodeKey(i), EncodeValue(i), &kHelper, 128 * 1024));
    auto snap = typed->TEST_GetSnapshot();
    ASSERT_LE(snap.logical_usage, cache->GetCapacity());
  }
}

TEST(CacheusCacheTest, ConcurrentAccessSmoke) {
  auto cache = NewCacheusCache(1 << 20, 0);
  ASSERT_NE(cache, nullptr);
  std::atomic<bool> failed{false};

  auto worker = [&](uint32_t tid) {
    for (uint32_t i = 0; i < 2000; ++i) {
      const uint32_t k = (tid * 997 + i) % 256;
      const std::string key = EncodeKey(k);
      auto* h = cache->Lookup(key);
      if (h != nullptr) {
        cache->Release(h);
      } else {
        if (!cache->Insert(key, EncodeValue(k), &kHelper, 64).ok()) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }
      }
    }
  };

  std::vector<std::thread> threads;
  for (uint32_t t = 0; t < 4; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& th : threads) {
    th.join();
  }

  ASSERT_FALSE(failed.load(std::memory_order_relaxed));
  auto* typed = dynamic_cast<CacheusCache*>(cache.get());
  ASSERT_NE(typed, nullptr);
  auto snap = typed->TEST_GetSnapshot();
  ASSERT_LE(snap.logical_usage, cache->GetCapacity());
}

// Diagnostic micro-benchmark (disabled by default). Run with:
//   ./cacheus_cache_test --gtest_filter='*ZipfianHitRatioDiagnostic*'
//       --gtest_also_run_disabled_tests
// Compares pure Cacheus policy hit ratio vs exact LRU on a zipfian trace
// across cache/keyspace ratios, and dumps internal utilization/S-Q/weights.
TEST(CacheusCacheTest, DISABLED_ZipfianHitRatioDiagnostic) {
  const uint32_t N = 50000;
  const double alpha = 0.99;
  const size_t warmup = static_cast<size_t>(N) * 8;
  const size_t measure = static_cast<size_t>(N) * 16;
  const uint64_t trace_seed = 20260613;

  printf(
      "\n%-6s %9s %9s %7s %9s %9s %7s %7s %7s %7s %9s %9s\n", "ratio",
      "cacheus", "lru", "delta", "usage", "cap", "s_len", "q_len", "s_lim",
      "q_lim", "evictLRU", "evictLFU");

  for (double ratio : {0.01, 0.02, 0.05, 0.10, 0.20, 0.40}) {
    const size_t cap = std::max<size_t>(1, static_cast<size_t>(N * ratio));

    // --- Cacheus (pure policy via TEST_RequestStep) ---
    auto cache = NewCacheusCache(cap, 0);
    auto* typed = dynamic_cast<CacheusCache*>(cache.get());
    ASSERT_NE(typed, nullptr);
    Zipf zc(N, alpha, trace_seed);
    for (size_t i = 0; i < warmup; ++i) {
      typed->TEST_RequestStep(EncodeKey(zc.Next()), 1);
    }
    const std::string before = cache->GetPrintableOptions();
    const double h0 = ParsePrintableField(before, "cacheus.total_hits");
    const double m0 = ParsePrintableField(before, "cacheus.total_misses");
    for (size_t i = 0; i < measure; ++i) {
      typed->TEST_RequestStep(EncodeKey(zc.Next()), 1);
    }
    const std::string after = cache->GetPrintableOptions();
    const double h1 = ParsePrintableField(after, "cacheus.total_hits");
    const double m1 = ParsePrintableField(after, "cacheus.total_misses");
    const double cacheus_hr = (h1 - h0) / std::max(1.0, (h1 - h0) + (m1 - m0));
    auto snap = typed->TEST_GetSnapshot();

    // --- Exact LRU on identical trace ---
    RefLRU ref(cap);
    Zipf zl(N, alpha, trace_seed);
    for (size_t i = 0; i < warmup; ++i) {
      ref.Access(zl.Next());
    }
    size_t lru_hits = 0;
    for (size_t i = 0; i < measure; ++i) {
      if (ref.Access(zl.Next())) {
        ++lru_hits;
      }
    }
    const double lru_hr = static_cast<double>(lru_hits) / measure;

    printf(
        "%-6.2f %9.4f %9.4f %+7.4f %9zu %9zu %7zu %7zu %7zu %7zu %9llu %9llu\n",
        ratio, cacheus_hr, lru_hr, cacheus_hr - lru_hr, snap.logical_usage, cap,
        snap.s_len, snap.q_len, snap.s_limit, snap.q_limit,
        (unsigned long long)snap.evict_lru_count,
        (unsigned long long)snap.evict_lfu_count);
  }
}

namespace {
bool g_variable_block_size = false;

// Real Lookup/Insert path, configurable sharding. charge=1 with metadata not
// charged so capacity == entry count. Returns measured hit ratio.
double RunRealPath(bool cacheus, int shard_bits, uint32_t N, double alpha,
                   size_t cap_entries, size_t warmup, size_t measure,
                   uint64_t seed, size_t block_bytes = 16384) {
  // Use a realistic per-entry charge so HCC's ~445B/entry metadata overhead is
  // negligible (<3%); charge=1 makes overhead dominate and is unrepresentative.
  const size_t cap = cap_entries * block_bytes;
  LRUCacheOptions opts;
  opts.capacity = cap;
  opts.num_shard_bits = shard_bits;
  opts.metadata_charge_policy = kDontChargeCacheMetadata;
  std::shared_ptr<Cache> cache =
      cacheus ? NewCacheusCache(opts) : NewLRUCache(opts);
  Zipf z(N, alpha, seed);
  // Per-key block size: 0 => uniform block_bytes; otherwise deterministic
  // 4KB..32KB derived from the key so cacheus/LRU see identical traces.
  auto charge_of = [&](uint32_t k) -> size_t {
    if (g_variable_block_size) {
      uint64_t h = k * 0x9E3779B97F4A7C15ULL;
      return 4096 + static_cast<size_t>((h >> 33) % 28673);
    }
    return block_bytes;
  };
  auto step = [&](bool count, size_t* hits) {
    const uint32_t k = z.Next();
    std::string key;
    PutFixed64(&key, k);
    PutFixed64(&key, 0);  // pad to 16-byte cache key for HCC backing
    auto* h = cache->Lookup(key, &kHelper, nullptr);
    if (h != nullptr) {
      if (count) {
        ++(*hits);
      }
      cache->Release(h);
    } else {
      cache->Insert(key, EncodeValue(k), &kHelper, charge_of(k));
    }
  };
  size_t hits = 0, dummy = 0;
  for (size_t i = 0; i < warmup; ++i) {
    step(false, &dummy);
  }
  for (size_t i = 0; i < measure; ++i) {
    step(true, &hits);
  }
  return static_cast<double>(hits) / static_cast<double>(measure);
}
}  // namespace

// Decisive isolation: does a plain HCC at the cacheus backing size hold all
// keys, and does an eviction-callback-returning-false change that?
TEST(CacheusCacheTest, DISABLED_BackingIsolation) {
  const uint32_t N = 50000;
  const double alpha = 0.99;
  const size_t backing_cap = 16 * 20000 + (1 << 20);  // ComputeBackingCapacity
  const size_t warmup = static_cast<size_t>(N) * 8;
  const size_t measure = static_cast<size_t>(N) * 16;
  const uint64_t seed = 20260613;

  auto run = [&](std::shared_ptr<Cache> cache, const char* label) {
    Zipf z(N, alpha, seed);
    auto step = [&](bool count, size_t* hits) {
      const uint32_t k = z.Next();
      std::string key;
      PutFixed64(&key, k);
      PutFixed64(&key, 0);
      auto* h = cache->Lookup(key, &kHelper, nullptr);
      if (h != nullptr) {
        if (count) ++(*hits);
        cache->Release(h);
      } else {
        cache->Insert(key, EncodeValue(k), &kHelper, 1);
      }
    };
    size_t dummy = 0, hits = 0;
    for (size_t i = 0; i < warmup; ++i) step(false, &dummy);
    for (size_t i = 0; i < measure; ++i) step(true, &hits);
    printf("%-28s hit=%.4f usage=%zu cap=%zu\n", label,
           static_cast<double>(hits) / measure, cache->GetUsage(),
           cache->GetCapacity());
  };

  {
    HyperClockCacheOptions o(backing_cap, 0, 0, false, nullptr,
                             kDontChargeCacheMetadata);
    run(o.MakeSharedCache(), "plain_hcc");
  }
  {
    HyperClockCacheOptions o(backing_cap, 0, 0, false, nullptr,
                             kDontChargeCacheMetadata);
    auto c = o.MakeSharedCache();
    c->SetEvictionCallback(
        [](const Slice&, Cache::Handle*, bool) { return false; });
    run(c, "hcc_with_evict_cb_false");
  }
}

// Low-coverage regime: large keyspace, few ops per key (mimics YCSB wlC where
// 8GB cache fills with mid-tail blocks revisited <1x within the run). This is
// the regime where Cacheus's 2-miss Q->S admission cost is expected to bite.
TEST(CacheusCacheTest, DISABLED_LowCoverageRegime) {
  const uint32_t N = 200000;
  const double alpha = 0.99;
  const size_t block = 8192;
  g_variable_block_size = false;

  for (double cov : {0.5, 1.0, 2.0}) {
    const size_t warmup = static_cast<size_t>(N * cov);
    const size_t measure = static_cast<size_t>(N * cov);
    printf("\n[coverage=%.1fx warmup=%zu measure=%zu]\n%-6s %10s %10s %10s\n",
           cov, warmup, measure, "ratio", "cacheus_s0", "lru_s0", "delta");
    for (double ratio : {0.05, 0.10, 0.20, 0.40, 0.60}) {
      const size_t cap = std::max<size_t>(64, static_cast<size_t>(N * ratio));
      const double c0 =
          RunRealPath(true, 0, N, alpha, cap, warmup, measure, 20260613, block);
      const double l0 =
          RunRealPath(false, 0, N, alpha, cap, warmup, measure, 20260613,
                      block);
      printf("%-6.2f %10.4f %10.4f %+10.4f\n", ratio, c0, l0, c0 - l0);
    }
  }
}

// Single-instance real-path deep dump: compares measured backing hit ratio to
// the shadow policy's internal state to locate where hits are lost.
TEST(CacheusCacheTest, DISABLED_RealPathDeepDump) {
  const uint32_t N = 50000;
  const double alpha = 0.99;
  const size_t cap = 20000;  // 40%
  const size_t warmup = static_cast<size_t>(N) * 8;
  const size_t measure = static_cast<size_t>(N) * 16;
  const uint64_t seed = 20260613;

  LRUCacheOptions opts;
  opts.capacity = cap;
  opts.num_shard_bits = 0;
  opts.metadata_charge_policy = kDontChargeCacheMetadata;
  auto cache = NewCacheusCache(opts);
  auto* typed = dynamic_cast<CacheusCache*>(cache.get());
  ASSERT_NE(typed, nullptr);

  Zipf z(N, alpha, seed);
  auto step = [&](bool count, size_t* hits) {
    const uint32_t k = z.Next();
    std::string key;
    PutFixed64(&key, k);
    PutFixed64(&key, 0);
    auto* h = cache->Lookup(key, &kHelper, nullptr);
    if (h != nullptr) {
      if (count) ++(*hits);
      cache->Release(h);
    } else {
      cache->Insert(key, EncodeValue(k), &kHelper, 1);
    }
  };
  size_t dummy = 0, hits = 0;
  for (size_t i = 0; i < warmup; ++i) step(false, &dummy);
  for (size_t i = 0; i < measure; ++i) step(true, &hits);

  printf("\nmeasured backing hit ratio = %.4f\n",
         static_cast<double>(hits) / measure);
  printf("--- shadow printable ---\n%s\n",
         cache->GetPrintableOptions().c_str());
  printf("GetUsage(backing-reported)=%zu GetCapacity=%zu\n", cache->GetUsage(),
         cache->GetCapacity());
}

// Isolates the effect of wrapper sharding on Cacheus vs LRU hit ratio.
TEST(CacheusCacheTest, DISABLED_ShardingHitRatioDiagnostic) {
  const uint32_t N = 50000;
  const double alpha = 0.99;
  const size_t warmup = static_cast<size_t>(N) * 8;
  const size_t measure = static_cast<size_t>(N) * 16;
  const uint64_t seed = 20260613;

  for (bool var : {false, true}) {
    g_variable_block_size = var;
    printf("\n[block_size=%s]\n%-6s %10s %10s %10s %10s\n",
           var ? "variable_4-32KB" : "uniform_16KB", "ratio", "cacheus_s0",
           "cacheus_s6", "lru_s0", "lru_s6");
    for (double ratio : {0.02, 0.05, 0.10, 0.20, 0.40}) {
      const size_t cap = std::max<size_t>(64, static_cast<size_t>(N * ratio));
      const double c0 =
          RunRealPath(true, 0, N, alpha, cap, warmup, measure, seed);
      const double c6 =
          RunRealPath(true, 6, N, alpha, cap, warmup, measure, seed);
      const double l0 =
          RunRealPath(false, 0, N, alpha, cap, warmup, measure, seed);
      const double l6 =
          RunRealPath(false, 6, N, alpha, cap, warmup, measure, seed);
      printf("%-6.2f %10.4f %10.4f %10.4f %10.4f\n", ratio, c0, c6, l0, l6);
    }
  }
  g_variable_block_size = false;
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
