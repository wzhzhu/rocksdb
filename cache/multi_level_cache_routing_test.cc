//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "cache/multi_level_cache.h"

#include <memory>
#include <string>
#include <vector>

#include "cache/cache_key.h"
#include "rocksdb/cache.h"
#include "table/block_based/block_cache.h"
#include "test_util/testharness.h"

namespace ROCKSDB_NAMESPACE {

class MultiLevelCacheRoutingTest : public testing::Test {
 public:
  MultiLevelCacheRoutingTest() {
    sub_caches_.push_back(NewLRUCache(/*capacity=*/1 << 20, /*num_shard_bits=*/0));
    sub_caches_.push_back(NewLRUCache(/*capacity=*/1 << 20, /*num_shard_bits=*/0));
    shared_cache_ = NewLRUCache(/*capacity=*/1 << 20, /*num_shard_bits=*/0);
    cache_ = std::make_unique<MultiLevelCache>(sub_caches_, shared_cache_,
                                               /*total_capacity=*/2 << 20);
  }

 protected:
  static std::string MakeBaseKey() {
    OffsetableCacheKey base("dbid", "session", 123);
    return base.WithOffset(4096 >> 2).AsSlice().ToString();
  }

  std::vector<std::shared_ptr<Cache>> sub_caches_;
  std::shared_ptr<Cache> shared_cache_;
  std::unique_ptr<MultiLevelCache> cache_;
};

TEST_F(MultiLevelCacheRoutingTest, ExtendedKeyRoutesToTargetLevel) {
  const std::string base_key = MakeBaseKey();
  std::string extended_key = base_key;
  extended_key.push_back(static_cast<char>(EncodeCacheKeyLevelTag(1)));

  ASSERT_OK(cache_->Insert(extended_key, nullptr, &kNoopCacheItemHelper,
                           /*charge=*/1));

  Cache::Handle* level1_hit = cache_->Lookup(extended_key);
  ASSERT_NE(level1_hit, nullptr);
  cache_->Release(level1_hit);

  Cache::Handle* base_miss = cache_->Lookup(base_key);
  ASSERT_EQ(base_miss, nullptr);

  auto metrics = cache_->GetLevelMetricsSnapshot();
  ASSERT_EQ(metrics.lookups.size(), 2U);
  ASSERT_EQ(metrics.hits.size(), 2U);
  EXPECT_EQ(metrics.lookups[1], 1U);
  EXPECT_EQ(metrics.hits[1], 1U);
  EXPECT_EQ(metrics.lookups[0], 1U);
  EXPECT_EQ(metrics.hits[0], 0U);
}

TEST_F(MultiLevelCacheRoutingTest, StoresBaseIdentityKeyOnly) {
  const std::string base_key = MakeBaseKey();
  std::string extended_key = base_key;
  extended_key.push_back(static_cast<char>(EncodeCacheKeyLevelTag(1)));

  ASSERT_OK(cache_->Insert(extended_key, nullptr, &kNoopCacheItemHelper,
                           /*charge=*/1));

  std::vector<std::string> observed_keys;
  cache_->ApplyToAllEntries(
      [&observed_keys](const Slice& key, Cache::ObjectPtr /*obj*/,
                       size_t /*charge*/,
                       const Cache::CacheItemHelper* /*helper*/) {
        observed_keys.emplace_back(key.ToString());
      },
      Cache::ApplyToAllEntriesOptions());

  ASSERT_EQ(observed_keys.size(), 1U);
  EXPECT_EQ(observed_keys[0].size(), static_cast<size_t>(kCacheKeySize));
  EXPECT_EQ(observed_keys[0], base_key);
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
