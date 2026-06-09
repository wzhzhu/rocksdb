#include "cache/arc_cache.h"

#include <algorithm>
#include <limits>
#include <sstream>

#include "cache/clock_cache.h"
namespace ROCKSDB_NAMESPACE {
namespace {

size_t ComputeBackingCapacity(size_t logical_capacity) {
  if (logical_capacity == 0) {
    return 0;
  }
  static constexpr size_t kScale = 16;
  static constexpr size_t kMinHeadroom = 1ULL << 20;
  const size_t scaled_limit = std::numeric_limits<size_t>::max() / kScale;
  size_t scaled = logical_capacity > scaled_limit
                      ? std::numeric_limits<size_t>::max()
                      : logical_capacity * kScale;
  if (scaled <= std::numeric_limits<size_t>::max() - kMinHeadroom) {
    scaled += kMinHeadroom;
  } else {
    scaled = std::numeric_limits<size_t>::max();
  }
  return std::max(logical_capacity, scaled);
}

std::shared_ptr<Cache> BuildBackingCache(const LRUCacheOptions& options) {
  HyperClockCacheOptions backing_opts(
      ComputeBackingCapacity(options.capacity),
      /*estimated_entry_charge=*/0, options.num_shard_bits,
      /*strict_capacity_limit=*/false, options.memory_allocator,
      options.metadata_charge_policy);
  backing_opts.hash_seed = options.hash_seed;
  backing_opts.secondary_cache = options.secondary_cache;
  std::shared_ptr<Cache> backing = backing_opts.MakeSharedCache();
  if (backing == nullptr) {
    LRUCacheOptions fallback = options;
    fallback.capacity = ComputeBackingCapacity(options.capacity);
    fallback.strict_capacity_limit = false;
    backing = fallback.MakeSharedCache();
  }
  return backing;
}

}  // namespace

ARCCache::ARCCache(std::shared_ptr<Cache> target, size_t capacity,
                   const ARCTuningOptions& tuning_options)
    : CacheWrapper(std::move(target)), capacity_(capacity), target_t1_(0) {
  pending_max_age_ops_ = std::max<uint64_t>(1, tuning_options.pending_max_age_ops);
  target_->SetEvictionCallback([this](const Slice& key, Handle* /*h*/,
                                      bool /*was_hit*/) {
    std::lock_guard<std::mutex> lock(mu_);
    const std::string key_str = SliceToKey(key);
    OnBackingEraseAckLocked(key_str);
    auto it = entries_.find(key_str);
    if (it != entries_.end() && IsResident(it->second.list_type)) {
      RemoveLocked(it->first);
    }
    return false;
  });
}

std::string ARCCache::SliceToKey(const Slice& key) const {
  return std::string(key.data(), key.size());
}

void ARCCache::AdvanceGenerationLocked(const std::string& key) {
  KeySyncState& state = key_sync_states_[key];
  ++state.generation;
  state.last_touched_at_op = request_counter_;
  pending_erased_keys_.erase(key);
}

void ARCCache::MaybeCleanupKeySyncLocked(const std::string& key) {
  if (pending_erased_keys_.find(key) != pending_erased_keys_.end()) {
    return;
  }
  if (entries_.find(key) != entries_.end()) {
    return;
  }
  if (pending_state_.find(key) != pending_state_.end()) {
    return;
  }
  auto sync_it = key_sync_states_.find(key);
  if (sync_it == key_sync_states_.end()) {
    return;
  }
  const KeySyncState& sync = sync_it->second;
  const bool no_pending_erase = sync.erase_acked >= sync.erase_issued;
  const bool stale =
      request_counter_ > sync.last_touched_at_op &&
      request_counter_ - sync.last_touched_at_op > pending_max_age_ops_;
  if (no_pending_erase || stale) {
    key_sync_states_.erase(sync_it);
  }
}

void ARCCache::MarkTombstoneLocked(const std::string& key) {
  KeySyncState& state = key_sync_states_[key];
  ++state.erase_issued;
  state.last_touched_at_op = request_counter_;
  pending_erased_keys_[key] =
      TombstoneMeta{request_counter_, state.generation, state.erase_issued};
}

bool ARCCache::IsTombstonedLocked(const std::string& key) {
  auto it = pending_erased_keys_.find(key);
  if (it == pending_erased_keys_.end()) {
    return false;
  }
  if (request_counter_ > it->second.created_at_op &&
      request_counter_ - it->second.created_at_op > pending_max_age_ops_) {
    pending_erased_keys_.erase(it);
    MaybeCleanupKeySyncLocked(key);
    return false;
  }
  return true;
}

void ARCCache::OnBackingEraseAckLocked(const std::string& key) {
  auto sync_it = key_sync_states_.find(key);
  if (sync_it == key_sync_states_.end()) {
    return;
  }
  KeySyncState& state = sync_it->second;
  ++state.erase_acked;
  state.last_touched_at_op = request_counter_;
  auto tomb_it = pending_erased_keys_.find(key);
  if (tomb_it == pending_erased_keys_.end()) {
    MaybeCleanupKeySyncLocked(key);
    return;
  }
  const TombstoneMeta& tomb = tomb_it->second;
  if (tomb.generation == state.generation &&
      state.erase_acked >= tomb.erase_ticket) {
    pending_erased_keys_.erase(tomb_it);
    MaybeCleanupKeySyncLocked(key);
  }
}

void ARCCache::MaybePruneTombstonesLocked() {
  if ((request_counter_ % kTombstonePruneIntervalOps) != 0) {
    return;
  }
  if (pending_erased_keys_.empty()) {
    tombstone_prune_resume_valid_ = false;
    tombstone_prune_resume_key_.clear();
    return;
  }
  size_t scan_budget =
      std::min(kTombstonePruneScanBudget, pending_erased_keys_.size());
  auto it = pending_erased_keys_.begin();
  if (tombstone_prune_resume_valid_) {
    auto resume_it = pending_erased_keys_.find(tombstone_prune_resume_key_);
    if (resume_it != pending_erased_keys_.end()) {
      it = std::next(resume_it);
      if (it == pending_erased_keys_.end()) {
        it = pending_erased_keys_.begin();
      }
    }
  }

  std::vector<std::string> expired_keys;
  expired_keys.reserve(scan_budget);
  std::string last_scanned_key;
  for (size_t scanned = 0; scanned < scan_budget; ++scanned) {
    if (it == pending_erased_keys_.end()) {
      it = pending_erased_keys_.begin();
      if (it == pending_erased_keys_.end()) {
        break;
      }
    }
    last_scanned_key = it->first;
    if (request_counter_ > it->second.created_at_op &&
        request_counter_ - it->second.created_at_op > pending_max_age_ops_) {
      expired_keys.push_back(it->first);
    }
    ++it;
  }

  if (!last_scanned_key.empty()) {
    tombstone_prune_resume_key_ = last_scanned_key;
    tombstone_prune_resume_valid_ = true;
  } else {
    tombstone_prune_resume_valid_ = false;
    tombstone_prune_resume_key_.clear();
  }

  for (const auto& key : expired_keys) {
    pending_erased_keys_.erase(key);
    MaybeCleanupKeySyncLocked(key);
  }
}

bool ARCCache::IsResident(ListType type) const {
  return type == ListType::kT1 || type == ListType::kT2;
}

void ARCCache::InsertToListFrontLocked(const std::string& key, ListType dst,
                                       size_t charge) {
  std::list<std::string>* dst_list = nullptr;
  size_t* dst_usage = nullptr;
  switch (dst) {
    case ListType::kT1:
      dst_list = &t1_;
      dst_usage = &usage_t1_;
      break;
    case ListType::kB1:
      dst_list = &b1_;
      dst_usage = &usage_b1_;
      break;
    case ListType::kT2:
      dst_list = &t2_;
      dst_usage = &usage_t2_;
      break;
    case ListType::kB2:
      dst_list = &b2_;
      dst_usage = &usage_b2_;
      break;
  }
  dst_list->push_front(key);
  entries_[key] = EntryMeta{dst, charge, dst_list->begin()};
  *dst_usage += charge;
}

void ARCCache::MoveLocked(const std::string& key, ListType dst, size_t new_charge) {
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return;
  }
  EntryMeta& meta = it->second;
  if (meta.list_type == dst && meta.charge == new_charge) {
    std::list<std::string>* list =
        (dst == ListType::kT1 ? &t1_
                              : (dst == ListType::kB1 ? &b1_
                                                      : (dst == ListType::kT2 ? &t2_
                                                                              : &b2_)));
    list->splice(list->begin(), *list, meta.iter);
    meta.iter = list->begin();
    return;
  }

  std::list<std::string>* src_list = nullptr;
  size_t* src_usage = nullptr;
  switch (meta.list_type) {
    case ListType::kT1:
      src_list = &t1_;
      src_usage = &usage_t1_;
      break;
    case ListType::kB1:
      src_list = &b1_;
      src_usage = &usage_b1_;
      break;
    case ListType::kT2:
      src_list = &t2_;
      src_usage = &usage_t2_;
      break;
    case ListType::kB2:
      src_list = &b2_;
      src_usage = &usage_b2_;
      break;
  }
  *src_usage = (*src_usage >= meta.charge) ? (*src_usage - meta.charge) : 0;
  src_list->erase(meta.iter);
  entries_.erase(it);
  InsertToListFrontLocked(key, dst, new_charge);
}

void ARCCache::RemoveLocked(const std::string& key) {
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return;
  }
  EntryMeta meta = it->second;
  switch (meta.list_type) {
    case ListType::kT1:
      t1_.erase(meta.iter);
      usage_t1_ = (usage_t1_ >= meta.charge) ? (usage_t1_ - meta.charge) : 0;
      break;
    case ListType::kB1:
      b1_.erase(meta.iter);
      usage_b1_ = (usage_b1_ >= meta.charge) ? (usage_b1_ - meta.charge) : 0;
      break;
    case ListType::kT2:
      t2_.erase(meta.iter);
      usage_t2_ = (usage_t2_ >= meta.charge) ? (usage_t2_ - meta.charge) : 0;
      break;
    case ListType::kB2:
      b2_.erase(meta.iter);
      usage_b2_ = (usage_b2_ >= meta.charge) ? (usage_b2_ - meta.charge) : 0;
      break;
  }
  entries_.erase(it);
}

void ARCCache::TrimGhostLocked(std::list<std::string>* list, ListType list_type,
                               size_t* usage, size_t target_limit) {
  while (*usage > target_limit && !list->empty()) {
    const std::string victim = list->back();
    auto it = entries_.find(victim);
    if (it == entries_.end() || it->second.list_type != list_type) {
      list->pop_back();
      continue;
    }
    const size_t victim_charge = it->second.charge;
    list->pop_back();
    *usage = (*usage >= victim_charge) ? (*usage - victim_charge) : 0;
    entries_.erase(it);
  }
}

void ARCCache::AdjustTargetLocked(bool hit_b1) {
  if (capacity_ == 0) {
    target_t1_ = 0;
    return;
  }
  if (hit_b1) {
    const size_t delta =
        std::max<size_t>(1, usage_b1_ == 0 ? 1 : (usage_b2_ / usage_b1_));
    target_t1_ = std::min(capacity_, target_t1_ + delta);
  } else {
    const size_t delta =
        std::max<size_t>(1, usage_b2_ == 0 ? 1 : (usage_b1_ / usage_b2_));
    target_t1_ = (target_t1_ > delta) ? (target_t1_ - delta) : 0;
  }
}

void ARCCache::ReplaceLocked(bool in_b2, std::vector<std::string>* evicted_keys) {
  if (!t1_.empty() &&
      (usage_t1_ > target_t1_ || (in_b2 && usage_t1_ == target_t1_))) {
    const std::string victim = t1_.back();
    auto it = entries_.find(victim);
    if (it == entries_.end()) {
      t1_.pop_back();
      return;
    }
    const size_t charge = it->second.charge;
    MoveLocked(victim, ListType::kB1, charge);
    evicted_keys->push_back(victim);
  } else if (!t2_.empty()) {
    const std::string victim = t2_.back();
    auto it = entries_.find(victim);
    if (it == entries_.end()) {
      t2_.pop_back();
      return;
    }
    const size_t charge = it->second.charge;
    MoveLocked(victim, ListType::kB2, charge);
    evicted_keys->push_back(victim);
  }
}

void ARCCache::EnsureResidentLimitLocked(std::vector<std::string>* evicted_keys) {
  while (usage_t1_ + usage_t2_ > capacity_) {
    ReplaceLocked(false, evicted_keys);
    if (t1_.empty() && t2_.empty()) {
      break;
    }
  }
  TrimGhostLocked(&b1_, ListType::kB1, &usage_b1_, capacity_);
  TrimGhostLocked(&b2_, ListType::kB2, &usage_b2_, capacity_);
}

Status ARCCache::Insert(const Slice& key, ObjectPtr value,
                        const CacheItemHelper* helper, size_t charge,
                        Handle** handle, Priority priority,
                        const Slice& compressed_value, CompressionType type) {
  const std::string key_str = SliceToKey(key);
  const Status s = target_->Insert(key, value, helper, charge, handle, priority,
                                   compressed_value, type);
  if (!s.ok()) {
    return s;
  }

  std::vector<std::string> evicted_keys;
  {
    std::lock_guard<std::mutex> lock(mu_);
    ++request_counter_;
    MaybePruneTombstonesLocked();
    AdvanceGenerationLocked(key_str);
    if (capacity_ == 0) {
      pending_state_.erase(key_str);
      MarkTombstoneLocked(key_str);
      evicted_keys.push_back(key_str);
    } else if (charge > capacity_) {
      pending_state_.erase(key_str);
      RemoveLocked(key_str);
      MarkTombstoneLocked(key_str);
      evicted_keys.push_back(key_str);
    } else {
      PendingState pending;
      auto p_it = pending_state_.find(key_str);
      if (p_it != pending_state_.end()) {
        pending = p_it->second;
        pending_state_.erase(p_it);
        if (request_counter_ > pending.observed_at_op &&
            request_counter_ - pending.observed_at_op > pending_max_age_ops_) {
          pending.hit_b1 = false;
          pending.hit_b2 = false;
        }
      }

      auto it = entries_.find(key_str);
      if (it != entries_.end()) {
        if (it->second.list_type == ListType::kB1) {
          AdjustTargetLocked(true);
          ReplaceLocked(false, &evicted_keys);
          MoveLocked(key_str, ListType::kT2, charge);
        } else if (it->second.list_type == ListType::kB2) {
          AdjustTargetLocked(false);
          ReplaceLocked(true, &evicted_keys);
          MoveLocked(key_str, ListType::kT2, charge);
        } else {
          MoveLocked(key_str, ListType::kT2, charge);
        }
      } else {
        if (pending.hit_b1) {
          AdjustTargetLocked(true);
          ReplaceLocked(false, &evicted_keys);
          InsertToListFrontLocked(key_str, ListType::kT2, charge);
        } else if (pending.hit_b2) {
          AdjustTargetLocked(false);
          ReplaceLocked(true, &evicted_keys);
          InsertToListFrontLocked(key_str, ListType::kT2, charge);
        } else {
          if (usage_t1_ + usage_b1_ >= capacity_) {
            if (usage_t1_ < capacity_) {
              TrimGhostLocked(&b1_, ListType::kB1, &usage_b1_, capacity_ - 1);
              ReplaceLocked(false, &evicted_keys);
            } else if (!t1_.empty()) {
              const std::string victim = t1_.back();
              RemoveLocked(victim);
              MarkTombstoneLocked(victim);
              evicted_keys.push_back(victim);
            }
          } else if (usage_t1_ + usage_b1_ < capacity_) {
            const size_t total = usage_t1_ + usage_t2_ + usage_b1_ + usage_b2_;
            if (total >= capacity_) {
              if (total >= 2 * capacity_) {
                TrimGhostLocked(&b2_, ListType::kB2, &usage_b2_, capacity_ - 1);
              }
              ReplaceLocked(false, &evicted_keys);
            }
          }
          InsertToListFrontLocked(key_str, ListType::kT1, charge);
        }
      }
      EnsureResidentLimitLocked(&evicted_keys);
      for (const std::string& evicted : evicted_keys) {
        MarkTombstoneLocked(evicted);
      }
    }
  }

  for (const std::string& evicted_key : evicted_keys) {
    target_->Erase(Slice(evicted_key));
  }
  return s;
}

Cache::Handle* ARCCache::Lookup(const Slice& key, const CacheItemHelper* helper,
                                CreateContext* create_context, Priority priority,
                                Statistics* stats) {
  const std::string key_str = SliceToKey(key);
  Handle* handle = target_->Lookup(key, helper, create_context, priority, stats);
  bool tombstoned_hit = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    ++request_counter_;
    ++wrapper_lookup_count_;
    MaybePruneTombstonesLocked();
    if (handle != nullptr) {
      tombstoned_hit = IsTombstonedLocked(key_str);
      if (!tombstoned_hit) {
        auto it = entries_.find(key_str);
        if (it == entries_.end()) {
          InsertToListFrontLocked(key_str, ListType::kT2,
                                  target_->GetCharge(handle));
        } else if (it->second.list_type == ListType::kT1 ||
                   it->second.list_type == ListType::kT2) {
          MoveLocked(key_str, ListType::kT2, it->second.charge);
        }
        pending_state_.erase(key_str);
        ++wrapper_hit_count_;
        return handle;
      }
      ++tombstone_lookup_dropped_count_;
    }
    PendingState pending;
    auto it = entries_.find(key_str);
    if (it != entries_.end()) {
      if (IsResident(it->second.list_type)) {
        RemoveLocked(key_str);
        ++desync_backing_miss_reconciled_count_;
      } else {
        pending.hit_b1 = (it->second.list_type == ListType::kB1);
        pending.hit_b2 = (it->second.list_type == ListType::kB2);
      }
    }
    pending.observed_at_op = request_counter_;
    pending_state_[key_str] = pending;
  }
  if (tombstoned_hit) {
    target_->Release(handle);
    target_->Erase(key);
  }
  return nullptr;
}

void ARCCache::Erase(const Slice& key) {
  const std::string key_str = SliceToKey(key);
  {
    std::lock_guard<std::mutex> lock(mu_);
    ++request_counter_;
    MaybePruneTombstonesLocked();
    MarkTombstoneLocked(key_str);
    RemoveLocked(key_str);
    pending_state_.erase(key_str);
  }
  target_->Erase(key);
}

void ARCCache::SetCapacity(size_t capacity) {
  std::vector<std::string> evicted_keys;
  {
    std::lock_guard<std::mutex> lock(mu_);
    ++request_counter_;
    MaybePruneTombstonesLocked();
    capacity_ = capacity;
    target_t1_ = std::min(target_t1_, capacity_);
    EnsureResidentLimitLocked(&evicted_keys);
    for (const std::string& evicted_key : evicted_keys) {
      MarkTombstoneLocked(evicted_key);
    }
    TrimGhostLocked(&b1_, ListType::kB1, &usage_b1_, capacity_);
    TrimGhostLocked(&b2_, ListType::kB2, &usage_b2_, capacity_);
  }
  for (const std::string& evicted_key : evicted_keys) {
    target_->Erase(Slice(evicted_key));
  }
  target_->SetCapacity(ComputeBackingCapacity(capacity));
}

size_t ARCCache::GetCapacity() const {
  std::lock_guard<std::mutex> lock(mu_);
  return capacity_;
}

std::string ARCCache::GetPrintableOptions() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::ostringstream oss;
  const double wrapper_hit_ratio =
      wrapper_lookup_count_ > 0
          ? static_cast<double>(wrapper_hit_count_) /
                static_cast<double>(wrapper_lookup_count_)
          : 0.0;
  oss << "arc.capacity=" << capacity_ << "\n";
  oss << "arc.target_t1=" << target_t1_ << "\n";
  oss << "arc.t1_usage=" << usage_t1_ << "\n";
  oss << "arc.t2_usage=" << usage_t2_ << "\n";
  oss << "arc.b1_usage=" << usage_b1_ << "\n";
  oss << "arc.b2_usage=" << usage_b2_ << "\n";
  oss << "arc.pending_max_age_ops=" << pending_max_age_ops_ << "\n";
  oss << "arc.resident_usage=" << (usage_t1_ + usage_t2_) << "\n";
  oss << "arc.desync_backing_miss_reconciled="
      << desync_backing_miss_reconciled_count_ << "\n";
  oss << "arc.tombstone_lookup_dropped=" << tombstone_lookup_dropped_count_
      << "\n";
  oss << "arc.tombstone_size=" << pending_erased_keys_.size() << "\n";
  oss << "arc.wrapper_lookups=" << wrapper_lookup_count_ << "\n";
  oss << "arc.wrapper_hits=" << wrapper_hit_count_ << "\n";
  oss << "arc.wrapper_hit_ratio=" << wrapper_hit_ratio;
  return oss.str();
}

std::shared_ptr<Cache> NewARCCache(const LRUCacheOptions& options) {
  return NewARCCache(options, ARCTuningOptions{});
}

std::shared_ptr<Cache> NewARCCache(const LRUCacheOptions& options,
                                   const ARCTuningOptions& tuning_options) {
  return std::make_shared<ARCCache>(BuildBackingCache(options), options.capacity,
                                    tuning_options);
}

std::shared_ptr<Cache> NewARCCache(size_t capacity, int num_shard_bits) {
  LRUCacheOptions options;
  options.capacity = capacity;
  options.num_shard_bits = num_shard_bits;
  return NewARCCache(options);
}

}  // namespace ROCKSDB_NAMESPACE
