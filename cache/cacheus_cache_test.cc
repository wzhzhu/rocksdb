//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "cache/cacheus_cache.h"

#include <atomic>
#include <array>
#include <string>
#include <thread>
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

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
