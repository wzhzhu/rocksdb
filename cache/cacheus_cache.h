#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "rocksdb/advanced_cache.h"
#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

struct CacheusTuningOptions {
  double initial_weight = 0.5;
  double learning_rate = -1.0;
  size_t history_size = 0;
  uint64_t period_len = 0;
  uint64_t rng_seed = 123;
  bool entry_charge_equivalent = false;
};

// Cacheus cache skeleton.
// Current implementation wraps an internal cache instance so integration
// points can be validated before replacing with full Cacheus logic.
class CacheusCache : public CacheWrapper {
 public:
  struct DebugSnapshot {
    double w_lru = 0.0;
    double w_lfu = 0.0;
    size_t s_limit = 0;
    size_t q_limit = 0;
    size_t logical_usage = 0;
    size_t s_len = 0;
    size_t q_len = 0;
    size_t lru_hist_len = 0;
    size_t lfu_hist_len = 0;
    size_t dem_count = 0;
    size_t nor_count = 0;
    uint64_t period_hits = 0;
    uint64_t evict_lru_count = 0;
    uint64_t evict_lfu_count = 0;
    uint64_t evict_tie_count = 0;
    std::string last_evicted_key;
    int last_evicted_policy = -2;
  };

  static const char* kClassName() { return "CacheusCache"; }

  CacheusCache(std::shared_ptr<Cache> target, size_t capacity,
               const CacheusTuningOptions& tuning_options);

  const char* Name() const override { return kClassName(); }
  std::string GetPrintableOptions() const override;

  Status Insert(const Slice& key, ObjectPtr value, const CacheItemHelper* helper,
                size_t charge, Handle** handle = nullptr,
                Priority priority = Priority::LOW,
                const Slice& compressed_value = Slice(),
                CompressionType type = CompressionType::kNoCompression) override;

  Handle* Lookup(const Slice& key, const CacheItemHelper* helper,
                 CreateContext* create_context,
                 Priority priority = Priority::LOW,
                 Statistics* stats = nullptr) override;

  void Erase(const Slice& key) override;
  void SetCapacity(size_t capacity) override;
  size_t GetCapacity() const override;
  DebugSnapshot TEST_GetSnapshot() const;
  void TEST_RequestStep(const Slice& key, size_t charge = 1);

 private:
  struct EntryMeta {
    uint64_t freq = 1;
    uint64_t time = 0;
    size_t charge = 0;
    bool in_s = false;
    bool is_demoted = false;
    bool is_new = true;
  };

  struct HistMeta {
    uint64_t freq = 1;
    bool is_new = true;
  };

  struct LfuNode {
    uint64_t freq = 0;
    uint64_t time = 0;
    std::string key;
  };

  struct PendingInsertMeta {
    enum class Source {
      kUnknown = 0,
      kMiss,
      kLruHist,
      kLfuHist,
    };

    bool valid = false;
    Source source = Source::kUnknown;
    uint64_t freq = 1;
    bool is_new = false;
    bool force_to_s = false;
  };

  struct LfuNodeComp {
    bool operator()(const LfuNode& a, const LfuNode& b) const {
      if (a.freq != b.freq) {
        return a.freq < b.freq;
      }
      if (a.time != b.time) {
        // Keep MRU-first tie-break to match python prototype behavior.
        return a.time > b.time;
      }
      return a.key < b.key;
    }
  };

  using ListIt = std::list<std::string>::iterator;
  using HistListIt = std::list<std::pair<std::string, HistMeta>>::iterator;
  using LfuSetIt = std::set<LfuNode, LfuNodeComp>::iterator;

  void EnsureCapacityLocked(size_t incoming_charge,
                            std::vector<std::string>* evicted_keys);
  void RecordInsertLocked(const std::string& key, size_t charge, bool is_new,
                          uint64_t freq = 1, bool force_to_s = false,
                          bool apply_limit_stack = false);
  void ResetStepTraceLocked();
  void TouchOnHitLocked(const std::string& key);
  void RemoveEntryLocked(const std::string& key, int policy);
  std::string ChooseVictimKeyLocked(int* policy);
  void PromoteToSLocked(const std::string& key);
  void LimitSLocked();
  void AddHistoryLocked(const std::string& key, const EntryMeta& meta, int policy);
  void AddToLruHistoryLocked(const std::string& key, const HistMeta& hist);
  void AddToLfuHistoryLocked(const std::string& key, const HistMeta& hist);
  void AdjustWeightsLocked(double reward_lru, double reward_lfu);
  void AdjustSizeLocked(bool hit_in_q);
  void UpdateLearningRateLocked();
  void RecomputePartitionLimitsLocked();
  void RemoveFromQueuesLocked(const std::string& key, const EntryMeta& meta);
  void UpdateLfuLocked(const std::string& key, const EntryMeta& meta);
  PendingInsertMeta ConsumePendingInsertMetaLocked(const std::string& key);
  double RandomUnitLocked();
  std::string SliceToKey(const Slice& key) const;

  mutable std::mutex mu_;
  size_t capacity_;
  size_t usage_ = 0;
  size_t s_usage_ = 0;
  size_t q_usage_ = 0;
  size_t s_limit_ = 0;
  size_t q_limit_ = 0;
  size_t history_limit_ = 1024;
  size_t configured_history_size_ = 0;

  uint64_t time_ = 0;
  double weight_lru_ = 0.5;
  double weight_lfu_ = 0.5;
  double learning_rate_ = 0.45;
  double learning_rate_prev_ = 0.0;
  double learning_rate_curr_ = 0.45;
  double learning_rate_reset_ = 0.45;
  uint64_t period_len_ = 1;
  uint64_t configured_period_len_ = 0;
  uint64_t period_hits_ = 0;
  double hitrate_prev_ = 0.0;
  int hitrate_zero_count_ = 0;
  int hitrate_nega_count_ = 0;
  size_t dem_count_ = 0;
  size_t nor_count_ = 0;
  uint64_t evict_lru_count_ = 0;
  uint64_t evict_lfu_count_ = 0;
  uint64_t evict_tie_count_ = 0;
  uint64_t total_hit_count_ = 0;
  uint64_t total_miss_count_ = 0;
  uint64_t lru_hist_hit_count_ = 0;
  uint64_t lfu_hist_hit_count_ = 0;
  std::string last_evicted_key_;
  int last_evicted_policy_ = -2;

  std::list<std::string> s_queue_;
  std::list<std::string> q_queue_;
  std::unordered_map<std::string, ListIt> s_pos_;
  std::unordered_map<std::string, ListIt> q_pos_;

  std::unordered_map<std::string, EntryMeta> entries_;
  std::set<LfuNode, LfuNodeComp> lfu_set_;
  std::unordered_map<std::string, LfuSetIt> lfu_pos_;

  std::list<std::pair<std::string, HistMeta>> lru_hist_;
  std::list<std::pair<std::string, HistMeta>> lfu_hist_;
  std::unordered_map<std::string, HistListIt> lru_hist_pos_;
  std::unordered_map<std::string, HistListIt> lfu_hist_pos_;
  std::unordered_map<std::string, PendingInsertMeta> pending_insert_meta_;

  uint32_t mt_state_[624];
  int mt_index_ = 625;
  bool entry_charge_equivalent_ = false;
};

std::shared_ptr<Cache> NewCacheusCache(const LRUCacheOptions& options);
std::shared_ptr<Cache> NewCacheusCache(const LRUCacheOptions& options,
                                       const CacheusTuningOptions& tuning_options);
std::shared_ptr<Cache> NewCacheusCache(size_t capacity,
                                       int num_shard_bits = -1);

}  // namespace ROCKSDB_NAMESPACE
