#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rocksdb/advanced_cache.h"
#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

struct ARCTuningOptions {
  uint64_t pending_max_age_ops = 65536;
};

// ARC cache implementation wrapping an internal cache instance. ARC state is
// maintained at key granularity while the underlying cache stores objects.
class ARCCache : public CacheWrapper {
 public:
  static const char* kClassName() { return "ARCCache"; }

  ARCCache(std::shared_ptr<Cache> target, size_t capacity,
           const ARCTuningOptions& tuning_options);

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

 private:
  enum class ListType { kT1 = 0, kB1, kT2, kB2 };

  struct EntryMeta {
    ListType list_type;
    size_t charge;
    std::list<std::string>::iterator iter;
  };

  struct PendingState {
    bool hit_b1 = false;
    bool hit_b2 = false;
    uint64_t observed_at_op = 0;
  };

  struct TombstoneMeta {
    uint64_t created_at_op = 0;
    uint64_t generation = 0;
    uint64_t erase_ticket = 0;
  };

  struct KeySyncState {
    uint64_t generation = 0;
    uint64_t erase_issued = 0;
    uint64_t erase_acked = 0;
    uint64_t last_touched_at_op = 0;
  };

  std::string SliceToKey(const Slice& key) const;
  // Evicts one resident entry (T1 -> B1 or T2 -> B2). Returns false when both
  // resident lists are empty (no progress possible).
  bool ReplaceLocked(bool in_b2, std::vector<std::string>* evicted_keys);
  // Evicts resident entries only while the incoming charge does not fit.
  void MakeRoomLocked(size_t incoming_charge, bool in_b2,
                      std::vector<std::string>* evicted_keys);
  void MoveLocked(const std::string& key, ListType dst, size_t new_charge);
  void RemoveLocked(const std::string& key);
  void InsertToListFrontLocked(const std::string& key, ListType dst, size_t charge);
  void TrimGhostLocked(std::list<std::string>* list, ListType list_type,
                       size_t* usage, size_t target_limit);
  void AdjustTargetLocked(bool hit_b1, size_t charge);
  void EnsureResidentLimitLocked(std::vector<std::string>* evicted_keys);
  void AdvanceGenerationLocked(const std::string& key);
  void MaybeCleanupKeySyncLocked(const std::string& key);
  void MarkTombstoneLocked(const std::string& key);
  bool IsTombstonedLocked(const std::string& key);
  void OnBackingEraseAckLocked(const std::string& key);
  void MaybePruneTombstonesLocked();
  void MaybePrunePendingStateLocked();
  bool IsResident(ListType type) const;

  mutable std::mutex mu_;
  size_t capacity_;
  size_t usage_t1_ = 0;
  size_t usage_t2_ = 0;
  size_t usage_b1_ = 0;
  size_t usage_b2_ = 0;
  size_t target_t1_ = 0;

  std::list<std::string> t1_;
  std::list<std::string> b1_;
  std::list<std::string> t2_;
  std::list<std::string> b2_;
  std::unordered_map<std::string, EntryMeta> entries_;
  std::unordered_map<std::string, PendingState> pending_state_;
  std::unordered_map<std::string, TombstoneMeta> pending_erased_keys_;
  std::unordered_map<std::string, KeySyncState> key_sync_states_;
  uint64_t desync_backing_miss_reconciled_count_ = 0;
  uint64_t tombstone_lookup_dropped_count_ = 0;
  uint64_t wrapper_lookup_count_ = 0;
  uint64_t wrapper_hit_count_ = 0;
  uint64_t request_counter_ = 0;
  uint64_t pending_max_age_ops_ = 65536;
  std::string tombstone_prune_resume_key_;
  bool tombstone_prune_resume_valid_ = false;
  std::string pending_state_prune_resume_key_;
  bool pending_state_prune_resume_valid_ = false;
  static constexpr uint64_t kTombstonePruneIntervalOps = 8192;
  static constexpr size_t kTombstonePruneScanBudget = 64;
};

std::shared_ptr<Cache> NewARCCache(const LRUCacheOptions& options);
std::shared_ptr<Cache> NewARCCache(const LRUCacheOptions& options,
                                   const ARCTuningOptions& tuning_options);
std::shared_ptr<Cache> NewARCCache(size_t capacity, int num_shard_bits = -1);

}  // namespace ROCKSDB_NAMESPACE
