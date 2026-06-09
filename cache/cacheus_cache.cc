#include "cache/cacheus_cache.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include "cache/clock_cache.h"
#include "cache/lru_cache.h"

namespace ROCKSDB_NAMESPACE {

namespace {

constexpr int kMtN = 624;
constexpr int kMtM = 397;
constexpr uint32_t kMtMatrixA = 0x9908b0dfU;
constexpr uint32_t kMtUpperMask = 0x80000000U;
constexpr uint32_t kMtLowerMask = 0x7fffffffU;

void NumpyInitGenrand(uint32_t seed, uint32_t state[kMtN], int* index) {
  state[0] = seed;
  for (int i = 1; i < kMtN; ++i) {
    state[i] =
        (1812433253U * (state[i - 1] ^ (state[i - 1] >> 30)) + static_cast<uint32_t>(i));
  }
  *index = kMtN;
}

void NumpyInitByArray(const uint32_t* init_key, int key_length,
                      uint32_t state[kMtN], int* index) {
  NumpyInitGenrand(19650218U, state, index);
  int i = 1;
  int j = 0;
  int k = (kMtN > key_length ? kMtN : key_length);
  for (; k > 0; --k) {
    state[i] = (state[i] ^
                ((state[i - 1] ^ (state[i - 1] >> 30)) * 1664525U)) +
               init_key[j] + static_cast<uint32_t>(j);
    ++i;
    ++j;
    if (i >= kMtN) {
      state[0] = state[kMtN - 1];
      i = 1;
    }
    if (j >= key_length) {
      j = 0;
    }
  }
  for (k = kMtN - 1; k > 0; --k) {
    state[i] = (state[i] ^
                ((state[i - 1] ^ (state[i - 1] >> 30)) * 1566083941U)) -
               static_cast<uint32_t>(i);
    ++i;
    if (i >= kMtN) {
      state[0] = state[kMtN - 1];
      i = 1;
    }
  }
  state[0] = 0x80000000U;
  *index = kMtN;
}

uint32_t NumpyGenrandUInt32(uint32_t state[kMtN], int* index) {
  uint32_t y;
  static const uint32_t mag01[2] = {0x0U, kMtMatrixA};
  if (*index >= kMtN) {
    int kk = 0;
    for (; kk < kMtN - kMtM; ++kk) {
      y = (state[kk] & kMtUpperMask) | (state[kk + 1] & kMtLowerMask);
      state[kk] = state[kk + kMtM] ^ (y >> 1) ^ mag01[y & 0x1U];
    }
    for (; kk < kMtN - 1; ++kk) {
      y = (state[kk] & kMtUpperMask) | (state[kk + 1] & kMtLowerMask);
      state[kk] = state[kk + (kMtM - kMtN)] ^ (y >> 1) ^ mag01[y & 0x1U];
    }
    y = (state[kMtN - 1] & kMtUpperMask) | (state[0] & kMtLowerMask);
    state[kMtN - 1] = state[kMtM - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
    *index = 0;
  }
  y = state[*index];
  ++(*index);

  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680U;
  y ^= (y << 15) & 0xefc60000U;
  y ^= (y >> 18);
  return y;
}

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

CacheusCache::CacheusCache(std::shared_ptr<Cache> target, size_t capacity,
                           const CacheusTuningOptions& tuning_options)
    : CacheWrapper(std::move(target)),
      capacity_(capacity) {
  entry_charge_equivalent_ = tuning_options.entry_charge_equivalent;
  const uint32_t seed = static_cast<uint32_t>(tuning_options.rng_seed);
  NumpyInitByArray(&seed, 1, mt_state_, &mt_index_);
  RecomputePartitionLimitsLocked();
  configured_history_size_ = tuning_options.history_size;
  history_limit_ = tuning_options.history_size > 0
                       ? tuning_options.history_size
                       : std::max<size_t>(128, capacity_ / 4096);
  configured_period_len_ = tuning_options.period_len;
  period_len_ = tuning_options.period_len > 0
                    ? tuning_options.period_len
                    : std::max<uint64_t>(1, capacity_ / 4096);
  weight_lru_ =
      std::min(0.99, std::max(0.01, tuning_options.initial_weight));
  weight_lfu_ = 1.0 - weight_lru_;
  if (tuning_options.learning_rate > 0) {
    learning_rate_ = std::min(1.0, std::max(0.001, tuning_options.learning_rate));
  } else {
    const double base_period = static_cast<double>(std::max<uint64_t>(1, period_len_));
    learning_rate_ = std::sqrt((2.0 * std::log(2.0)) / base_period);
    learning_rate_ = std::min(1.0, std::max(0.001, learning_rate_));
  }
  learning_rate_reset_ = std::min(1.0, std::max(0.001, learning_rate_));
  learning_rate_curr_ = learning_rate_;
  pending_max_age_ops_ = std::max<uint64_t>(1, tuning_options.pending_max_age_ops);

  target_->SetEvictionCallback([this](const Slice& key, Handle* /*h*/,
                                      bool /*was_hit*/) {
    std::lock_guard<std::mutex> lock(mu_);
    const std::string key_str = SliceToKey(key);
    OnBackingEraseAckLocked(key_str);
    auto it = entries_.find(key_str);
    if (it == entries_.end()) {
      return false;
    }
    EntryMeta meta = it->second;
    RemoveFromQueuesLocked(key_str, meta);
    auto lfu_it = lfu_pos_.find(key_str);
    if (lfu_it != lfu_pos_.end()) {
      lfu_set_.erase(lfu_it->second);
      lfu_pos_.erase(lfu_it);
    }
    usage_ = (usage_ >= meta.charge) ? (usage_ - meta.charge) : 0;
    entries_.erase(it);
    return false;
  });
}

std::string CacheusCache::SliceToKey(const Slice& key) const {
  return std::string(key.data(), key.size());
}

void CacheusCache::AdvanceGenerationLocked(const std::string& key) {
  KeySyncState& state = key_sync_states_[key];
  ++state.generation;
  state.last_touched_at_op = request_counter_;
  pending_erased_keys_.erase(key);
  MaybeCleanupKeySyncLocked(key);
}

void CacheusCache::MaybeCleanupKeySyncLocked(const std::string& key) {
  if (pending_erased_keys_.find(key) != pending_erased_keys_.end()) {
    return;
  }
  if (entries_.find(key) != entries_.end()) {
    return;
  }
  if (pending_insert_meta_.find(key) != pending_insert_meta_.end()) {
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

void CacheusCache::MarkTombstoneLocked(const std::string& key) {
  KeySyncState& state = key_sync_states_[key];
  ++state.erase_issued;
  state.last_touched_at_op = request_counter_;
  pending_erased_keys_[key] =
      TombstoneMeta{request_counter_, state.generation, state.erase_issued};
}

bool CacheusCache::IsTombstonedLocked(const std::string& key) {
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

void CacheusCache::OnBackingEraseAckLocked(const std::string& key) {
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

void CacheusCache::MaybePruneTombstonesLocked() {
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

void CacheusCache::RecomputePartitionLimitsLocked() {
  if (capacity_ == 0) {
    s_limit_ = 0;
    q_limit_ = 0;
    return;
  }
  q_limit_ = std::max<size_t>(1, (capacity_ + 99) / 100);
  q_limit_ = std::min(q_limit_, capacity_);
  s_limit_ = capacity_ - q_limit_;
}

void CacheusCache::AdjustWeightsLocked(double reward_lru, double reward_lfu) {
  weight_lru_ *= std::exp(learning_rate_ * reward_lru);
  weight_lfu_ *= std::exp(learning_rate_ * reward_lfu);
  const double sum = weight_lru_ + weight_lfu_;
  if (sum <= 0.0) {
    weight_lru_ = 0.5;
    weight_lfu_ = 0.5;
    return;
  }
  weight_lru_ /= sum;
  weight_lfu_ /= sum;
  if (weight_lru_ >= 0.99) {
    weight_lru_ = 0.99;
    weight_lfu_ = 0.01;
  } else if (weight_lfu_ >= 0.99) {
    weight_lru_ = 0.01;
    weight_lfu_ = 0.99;
  }
}

void CacheusCache::AdjustSizeLocked(bool hit_in_q) {
  if (capacity_ <= 1) {
    s_limit_ = 0;
    q_limit_ = capacity_;
    return;
  }
  if (hit_in_q) {
    const size_t dem = std::max<size_t>(1, dem_count_);
    const size_t delta =
        std::max<size_t>(1, static_cast<size_t>((nor_count_ / dem) + 1));
    s_limit_ = std::min(capacity_ - 1, s_limit_ + delta);
    q_limit_ = capacity_ - s_limit_;
  } else {
    const size_t nor = std::max<size_t>(1, nor_count_);
    const size_t delta =
        std::max<size_t>(1, static_cast<size_t>((dem_count_ / nor) + 1));
    q_limit_ = std::min(capacity_ - 1, q_limit_ + delta);
    s_limit_ = capacity_ - q_limit_;
  }
}

void CacheusCache::UpdateLearningRateLocked() {
  if (period_len_ == 0 || (time_ % period_len_) != 0) {
    return;
  }
  const auto round3 = [](double v) { return std::round(v * 1000.0) / 1000.0; };
  const double hitrate_curr = round3(
      static_cast<double>(period_hits_) / static_cast<double>(period_len_));
  const double hitrate_diff = round3(hitrate_curr - hitrate_prev_);
  const double delta_lr = round3(learning_rate_curr_) - round3(learning_rate_prev_);
  const double prod = delta_lr * hitrate_diff;
  int delta = 0;
  if (prod > 0) {
    delta = 1;
  } else if (prod < 0) {
    delta = -1;
  }

  if (delta > 0) {
    learning_rate_ =
        std::min(1.0, learning_rate_ + std::abs(learning_rate_ * delta_lr));
    hitrate_nega_count_ = 0;
    hitrate_zero_count_ = 0;
  } else if (delta < 0) {
    learning_rate_ =
        std::max(0.001, learning_rate_ - std::abs(learning_rate_ * delta_lr));
    hitrate_nega_count_ = 0;
    hitrate_zero_count_ = 0;
  } else if (hitrate_diff <= 0.0) {
    if (hitrate_curr <= 0.0 && hitrate_diff == 0.0) {
      ++hitrate_zero_count_;
    }
    if (hitrate_diff < 0.0) {
      ++hitrate_nega_count_;
      ++hitrate_zero_count_;
    }
    if (hitrate_zero_count_ >= 10 || hitrate_nega_count_ >= 10) {
      learning_rate_ = learning_rate_reset_;
      hitrate_zero_count_ = 0;
      hitrate_nega_count_ = 0;
    } else if (hitrate_diff < 0.0) {
      if (learning_rate_ >= 1.0) {
        learning_rate_ = 0.9;
      } else if (learning_rate_ <= 0.001) {
        learning_rate_ = 0.005;
      } else if (RandomUnitLocked() < 0.5) {
        learning_rate_ = std::min(1.0, learning_rate_ * 1.25);
      } else {
        learning_rate_ = std::max(0.001, learning_rate_ * 0.75);
      }
    }
  }

  learning_rate_prev_ = learning_rate_curr_;
  learning_rate_curr_ = learning_rate_;
  hitrate_prev_ = hitrate_curr;
  period_hits_ = 0;
}

void CacheusCache::AddToLruHistoryLocked(const std::string& key,
                                         const HistMeta& hist) {
  auto it = lru_hist_pos_.find(key);
  if (it != lru_hist_pos_.end()) {
    lru_hist_.erase(it->second);
    lru_hist_pos_.erase(it);
  }
  lru_hist_.emplace_back(key, hist);
  auto inserted = std::prev(lru_hist_.end());
  lru_hist_pos_[key] = inserted;
  while (lru_hist_.size() > history_limit_) {
    auto old = lru_hist_.begin();
    if (old->second.is_new && nor_count_ > 0) {
      --nor_count_;
    }
    lru_hist_pos_.erase(old->first);
    lru_hist_.erase(old);
  }
}

void CacheusCache::AddToLfuHistoryLocked(const std::string& key,
                                         const HistMeta& hist) {
  auto it = lfu_hist_pos_.find(key);
  if (it != lfu_hist_pos_.end()) {
    lfu_hist_.erase(it->second);
    lfu_hist_pos_.erase(it);
  }
  lfu_hist_.emplace_back(key, hist);
  auto inserted = std::prev(lfu_hist_.end());
  lfu_hist_pos_[key] = inserted;
  while (lfu_hist_.size() > history_limit_) {
    auto old = lfu_hist_.begin();
    lfu_hist_pos_.erase(old->first);
    lfu_hist_.erase(old);
  }
}

void CacheusCache::AddHistoryLocked(const std::string& key, const EntryMeta& meta,
                                    int policy) {
  if (policy == 0) {
    if (meta.is_new) {
      ++nor_count_;
    }
    AddToLruHistoryLocked(key, HistMeta{meta.freq, meta.is_new});
  } else if (policy == 1) {
    AddToLfuHistoryLocked(key, HistMeta{meta.freq, meta.is_new});
  }
}

void CacheusCache::RemoveFromQueuesLocked(const std::string& key,
                                          const EntryMeta& meta) {
  if (meta.in_s) {
    auto it = s_pos_.find(key);
    if (it != s_pos_.end()) {
      s_queue_.erase(it->second);
      s_pos_.erase(it);
    }
    s_usage_ = (s_usage_ >= meta.charge) ? (s_usage_ - meta.charge) : 0;
  } else {
    auto it = q_pos_.find(key);
    if (it != q_pos_.end()) {
      q_queue_.erase(it->second);
      q_pos_.erase(it);
    }
    q_usage_ = (q_usage_ >= meta.charge) ? (q_usage_ - meta.charge) : 0;
  }
}

void CacheusCache::UpdateLfuLocked(const std::string& key, const EntryMeta& meta) {
  auto it = lfu_pos_.find(key);
  if (it != lfu_pos_.end()) {
    lfu_set_.erase(it->second);
    lfu_pos_.erase(it);
  }
  auto inserted = lfu_set_.emplace(LfuNode{meta.freq, meta.time, key}).first;
  lfu_pos_[key] = inserted;
}

std::string CacheusCache::ChooseVictimKeyLocked(int* policy) {
  if (entries_.empty()) {
    *policy = -1;
    return "";
  }
  std::string lru_key;
  if (!q_queue_.empty()) {
    lru_key = q_queue_.front();
  } else if (!s_queue_.empty()) {
    lru_key = s_queue_.front();
  }

  if (lfu_set_.empty()) {
    *policy = 0;
    return lru_key;
  }
  const std::string lfu_key = lfu_set_.begin()->key;
  const bool choose_lru = RandomUnitLocked() < weight_lru_;

  if (lru_key == lfu_key) {
    *policy = -1;
    return lru_key;
  }
  if (choose_lru) {
    *policy = 0;
    return lru_key;
  }
  *policy = 1;
  return lfu_key;
}

void CacheusCache::RemoveEntryLocked(const std::string& key, int policy) {
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return;
  }
  EntryMeta meta = it->second;
  if (meta.is_demoted && dem_count_ > 0) {
    --dem_count_;
    meta.is_demoted = false;
  }
  RemoveFromQueuesLocked(key, meta);
  auto lfu_it = lfu_pos_.find(key);
  if (lfu_it != lfu_pos_.end()) {
    lfu_set_.erase(lfu_it->second);
    lfu_pos_.erase(lfu_it);
  }
  usage_ = (usage_ >= meta.charge) ? (usage_ - meta.charge) : 0;
  entries_.erase(it);
  last_evicted_key_ = key;
  last_evicted_policy_ = policy;
  if (policy == 0) {
    ++evict_lru_count_;
  } else if (policy == 1) {
    ++evict_lfu_count_;
  } else if (policy == -1) {
    ++evict_tie_count_;
  }
  AddHistoryLocked(key, meta, policy);
}

void CacheusCache::EnsureCapacityLocked(size_t incoming_charge,
                                        std::vector<std::string>* evicted_keys) {
  while (capacity_ > 0 && usage_ + incoming_charge > capacity_ &&
         !entries_.empty()) {
    int policy = -1;
    std::string victim = ChooseVictimKeyLocked(&policy);
    if (victim.empty()) {
      break;
    }
    RemoveEntryLocked(victim, policy);
    evicted_keys->push_back(victim);
  }
}

void CacheusCache::LimitSLocked() {
  while (s_usage_ >= s_limit_ && !s_queue_.empty()) {
    const std::string demoted_key = s_queue_.front();
    s_queue_.pop_front();
    s_pos_.erase(demoted_key);
    auto it = entries_.find(demoted_key);
    if (it == entries_.end()) {
      continue;
    }
    EntryMeta& meta = it->second;
    meta.in_s = false;
    meta.is_demoted = true;
    ++dem_count_;
    s_usage_ = (s_usage_ >= meta.charge) ? (s_usage_ - meta.charge) : 0;
    q_usage_ += meta.charge;
    q_queue_.push_back(demoted_key);
    q_pos_[demoted_key] = std::prev(q_queue_.end());
  }
}

void CacheusCache::PromoteToSLocked(const std::string& key) {
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return;
  }
  EntryMeta& meta = it->second;
  if (meta.in_s) {
    auto pos = s_pos_.find(key);
    if (pos != s_pos_.end()) {
      s_queue_.erase(pos->second);
      s_queue_.push_back(key);
      s_pos_[key] = std::prev(s_queue_.end());
    }
    return;
  }

  auto q_it = q_pos_.find(key);
  if (q_it != q_pos_.end()) {
    q_queue_.erase(q_it->second);
    q_pos_.erase(q_it);
  }
  q_usage_ = (q_usage_ >= meta.charge) ? (q_usage_ - meta.charge) : 0;
  if (s_usage_ >= s_limit_ && !s_queue_.empty()) {
    const std::string demoted_key = s_queue_.front();
    s_queue_.pop_front();
    s_pos_.erase(demoted_key);
    auto dem_it = entries_.find(demoted_key);
    if (dem_it != entries_.end()) {
      EntryMeta& dem_meta = dem_it->second;
      dem_meta.in_s = false;
      dem_meta.is_demoted = true;
      ++dem_count_;
      s_usage_ = (s_usage_ >= dem_meta.charge) ? (s_usage_ - dem_meta.charge) : 0;
      q_usage_ += dem_meta.charge;
      q_queue_.push_back(demoted_key);
      q_pos_[demoted_key] = std::prev(q_queue_.end());
    }
  }
  meta.in_s = true;
  if (meta.is_demoted && dem_count_ > 0) {
    --dem_count_;
  }
  meta.is_demoted = false;
  s_queue_.push_back(key);
  s_pos_[key] = std::prev(s_queue_.end());
  s_usage_ += meta.charge;
}

void CacheusCache::RecordInsertLocked(const std::string& key, size_t charge,
                                      bool is_new, uint64_t freq,
                                      bool force_to_s, bool apply_limit_stack) {
  EntryMeta meta;
  meta.freq = std::max<uint64_t>(1, freq);
  meta.time = time_;
  meta.charge = charge;
  meta.in_s = false;
  meta.is_demoted = false;
  meta.is_new = is_new;

  if (force_to_s || (s_usage_ < s_limit_ && q_queue_.empty())) {
    meta.in_s = true;
    s_queue_.push_back(key);
    s_pos_[key] = std::prev(s_queue_.end());
    s_usage_ += charge;
  } else {
    q_queue_.push_back(key);
    q_pos_[key] = std::prev(q_queue_.end());
    q_usage_ += charge;
  }

  entries_[key] = meta;
  usage_ += charge;
  UpdateLfuLocked(key, meta);
  if (apply_limit_stack) {
    LimitSLocked();
  }
}

CacheusCache::PendingInsertMeta CacheusCache::ConsumePendingInsertMetaLocked(
    const std::string& key) {
  PendingInsertMeta result;
  auto it = pending_insert_meta_.find(key);
  if (it != pending_insert_meta_.end()) {
    result = it->second;
    result.valid = true;
    pending_insert_meta_.erase(it);
    if (request_counter_ > result.observed_at_op &&
        request_counter_ - result.observed_at_op > pending_max_age_ops_) {
      result.valid = false;
      result.source = PendingInsertMeta::Source::kUnknown;
      result.freq = 1;
      result.is_new = false;
      result.force_to_s = false;
    }
  }
  return result;
}

double CacheusCache::RandomUnitLocked() {
  const uint32_t a = NumpyGenrandUInt32(mt_state_, &mt_index_) >> 5;
  const uint32_t b = NumpyGenrandUInt32(mt_state_, &mt_index_) >> 6;
  const double numerator =
      static_cast<double>(a) * 67108864.0 + static_cast<double>(b);
  return numerator * (1.0 / 9007199254740992.0);
}

void CacheusCache::ResetStepTraceLocked() {
  last_evicted_key_.clear();
  last_evicted_policy_ = -2;
}

void CacheusCache::TouchOnHitLocked(const std::string& key) {
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return;
  }
  EntryMeta& meta = it->second;
  meta.time = time_;
  meta.freq += 1;

  if (meta.in_s) {
    auto pos = s_pos_.find(key);
    if (pos != s_pos_.end()) {
      s_queue_.erase(pos->second);
      s_queue_.push_back(key);
      s_pos_[key] = std::prev(s_queue_.end());
    }
  } else {
    auto pos = q_pos_.find(key);
    if (pos != q_pos_.end()) {
      q_queue_.erase(pos->second);
      q_queue_.push_back(key);
      q_pos_[key] = std::prev(q_queue_.end());
    }
    if (meta.is_demoted && capacity_ > 0) {
      AdjustSizeLocked(true);
    }
    PromoteToSLocked(key);
  }
  UpdateLfuLocked(key, meta);
}

Status CacheusCache::Insert(const Slice& key, ObjectPtr value,
                            const CacheItemHelper* helper, size_t charge,
                            Handle** handle, Priority priority,
                            const Slice& compressed_value,
                            CompressionType type) {
  const std::string key_str = SliceToKey(key);
  const Status insert_status =
      target_->Insert(key, value, helper, charge, handle, priority,
                      compressed_value, type);
  if (!insert_status.ok()) {
    return insert_status;
  }

  std::vector<std::string> evicted_keys;
  {
    std::lock_guard<std::mutex> lock(mu_);
    ++request_counter_;
    MaybePruneTombstonesLocked();
    AdvanceGenerationLocked(key_str);
    ResetStepTraceLocked();
    ++time_;
    UpdateLearningRateLocked();
    const bool was_full_before_insert = (capacity_ > 0 && usage_ >= capacity_);
    PendingInsertMeta pending = ConsumePendingInsertMetaLocked(key_str);
    if (!pending.valid) {
      pending.valid = true;
      pending.source = PendingInsertMeta::Source::kMiss;
      pending.freq = 1;
      pending.is_new = was_full_before_insert;
      pending.force_to_s = false;
      auto lru_it = lru_hist_pos_.find(key_str);
      if (lru_it != lru_hist_pos_.end()) {
        ++lru_hist_hit_count_;
        pending.source = PendingInsertMeta::Source::kLruHist;
        pending.freq = std::max<uint64_t>(1, lru_it->second->second.freq + 1);
        pending.is_new = false;
        pending.force_to_s = true;
        const bool was_new = lru_it->second->second.is_new;
        lru_hist_.erase(lru_it->second);
        lru_hist_pos_.erase(lru_it);
        if (was_new && nor_count_ > 0) {
          --nor_count_;
        }
        AdjustWeightsLocked(-1.0, 0.0);
        if (was_new && capacity_ > 0) {
          AdjustSizeLocked(false);
        }
      } else {
        auto lfu_it = lfu_hist_pos_.find(key_str);
        if (lfu_it != lfu_hist_pos_.end()) {
          ++lfu_hist_hit_count_;
          pending.source = PendingInsertMeta::Source::kLfuHist;
          pending.freq = std::max<uint64_t>(1, lfu_it->second->second.freq + 1);
          pending.is_new = false;
          pending.force_to_s = true;
          lfu_hist_.erase(lfu_it->second);
          lfu_hist_pos_.erase(lfu_it);
          AdjustWeightsLocked(0.0, -1.0);
        }
      }
    } else if (pending.source == PendingInsertMeta::Source::kLruHist) {
      auto lru_it = lru_hist_pos_.find(key_str);
      if (lru_it != lru_hist_pos_.end()) {
        const bool was_new = lru_it->second->second.is_new;
        lru_hist_.erase(lru_it->second);
        lru_hist_pos_.erase(lru_it);
        if (was_new && nor_count_ > 0) {
          --nor_count_;
        }
        AdjustWeightsLocked(-1.0, 0.0);
        if (was_new && capacity_ > 0) {
          AdjustSizeLocked(false);
        }
      }
    } else if (pending.source == PendingInsertMeta::Source::kLfuHist) {
      auto lfu_it = lfu_hist_pos_.find(key_str);
      if (lfu_it != lfu_hist_pos_.end()) {
        lfu_hist_.erase(lfu_it->second);
        lfu_hist_pos_.erase(lfu_it);
        AdjustWeightsLocked(0.0, -1.0);
      }
    }
    bool is_new = pending.valid ? pending.is_new : was_full_before_insert;
    uint64_t seeded_freq = pending.valid ? pending.freq : 1;
    bool force_to_s = pending.valid ? pending.force_to_s : false;
    auto existing = entries_.find(key_str);
    if (existing != entries_.end()) {
      RemoveEntryLocked(key_str, -1);
    }
    const size_t logical_charge = entry_charge_equivalent_ ? 1 : charge;
    EnsureCapacityLocked(logical_charge, &evicted_keys);
    RecordInsertLocked(key_str, logical_charge, is_new, seeded_freq, force_to_s,
                       force_to_s || was_full_before_insert);
    for (const std::string& evicted : evicted_keys) {
      MarkTombstoneLocked(evicted);
    }
  }

  for (const std::string& evicted : evicted_keys) {
    target_->Erase(Slice(evicted));
  }
  return insert_status;
}

Cache::Handle* CacheusCache::Lookup(const Slice& key,
                                    const CacheItemHelper* helper,
                                    CreateContext* create_context,
                                    Priority priority, Statistics* stats) {
  const std::string key_str = SliceToKey(key);
  Handle* handle = target_->Lookup(key, helper, create_context, priority, stats);
  bool tombstoned_hit = false;
  bool request_counted = false;
  if (handle != nullptr) {
    std::lock_guard<std::mutex> lock(mu_);
    ++request_counter_;
    request_counted = true;
    MaybePruneTombstonesLocked();
    tombstoned_hit = IsTombstonedLocked(key_str);
    if (tombstoned_hit) {
      ++tombstone_lookup_dropped_count_;
    }
  }
  if (handle == nullptr || tombstoned_hit) {
    if (tombstoned_hit) {
      target_->Release(handle);
      target_->Erase(key);
      handle = nullptr;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (!request_counted) {
      ++request_counter_;
    }
    MaybePruneTombstonesLocked();
    ++total_miss_count_;
    auto live_it = entries_.find(key_str);
    if (live_it != entries_.end()) {
      EntryMeta meta = live_it->second;
      if (meta.is_demoted && dem_count_ > 0) {
        --dem_count_;
      }
      RemoveFromQueuesLocked(key_str, meta);
      auto lfu_it = lfu_pos_.find(key_str);
      if (lfu_it != lfu_pos_.end()) {
        lfu_set_.erase(lfu_it->second);
        lfu_pos_.erase(lfu_it);
      }
      usage_ = (usage_ >= meta.charge) ? (usage_ - meta.charge) : 0;
      entries_.erase(live_it);
      ++desync_backing_miss_reconciled_count_;
    }
    PendingInsertMeta pending;
    pending.valid = true;
    pending.source = PendingInsertMeta::Source::kMiss;
    pending.freq = 1;
    pending.is_new = (capacity_ > 0 && usage_ >= capacity_);
    pending.force_to_s = false;
    pending.observed_at_op = request_counter_;
    auto lru_it = lru_hist_pos_.find(key_str);
    if (lru_it != lru_hist_pos_.end()) {
      ++lru_hist_hit_count_;
      pending.freq = std::max<uint64_t>(1, lru_it->second->second.freq + 1);
      pending.is_new = false;
      pending.force_to_s = true;
      pending.source = PendingInsertMeta::Source::kLruHist;
      pending_insert_meta_[key_str] = pending;
      return nullptr;
    }

    auto lfu_it = lfu_hist_pos_.find(key_str);
    if (lfu_it != lfu_hist_pos_.end()) {
      ++lfu_hist_hit_count_;
      pending.freq = std::max<uint64_t>(1, lfu_it->second->second.freq + 1);
      pending.is_new = false;
      pending.force_to_s = true;
      pending.source = PendingInsertMeta::Source::kLfuHist;
      pending_insert_meta_[key_str] = pending;
      return nullptr;
    }

    pending_insert_meta_[key_str] = pending;
    return nullptr;
  }

  std::vector<std::string> evicted_keys;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!request_counted) {
      ++request_counter_;
    }
    MaybePruneTombstonesLocked();
    ++total_hit_count_;
    ResetStepTraceLocked();
    ++time_;
    UpdateLearningRateLocked();
    auto it = entries_.find(key_str);
    if (it == entries_.end()) {
      const size_t charge = target_->GetCharge(handle);
      const size_t logical_charge = entry_charge_equivalent_ ? 1 : charge;
      EnsureCapacityLocked(logical_charge, &evicted_keys);
      for (const std::string& evicted : evicted_keys) {
        MarkTombstoneLocked(evicted);
      }
      RecordInsertLocked(key_str, logical_charge, false, 1, false, false);
    }
    TouchOnHitLocked(key_str);
    ++period_hits_;
  }
  for (const std::string& evicted : evicted_keys) {
    target_->Erase(Slice(evicted));
  }
  return handle;
}

void CacheusCache::Erase(const Slice& key) {
  const std::string key_str = SliceToKey(key);
  {
    std::lock_guard<std::mutex> lock(mu_);
    ++request_counter_;
    MaybePruneTombstonesLocked();
    MarkTombstoneLocked(key_str);
    RemoveEntryLocked(key_str, -1);
  }
  target_->Erase(key);
}

void CacheusCache::SetCapacity(size_t capacity) {
  std::vector<std::string> evicted_keys;
  {
    std::lock_guard<std::mutex> lock(mu_);
    ++request_counter_;
    MaybePruneTombstonesLocked();
    capacity_ = capacity;
    RecomputePartitionLimitsLocked();
    history_limit_ = configured_history_size_ > 0
                         ? configured_history_size_
                         : std::max<size_t>(128, capacity_ / 4096);
    period_len_ = configured_period_len_ > 0
                      ? configured_period_len_
                      : std::max<uint64_t>(1, capacity_ / 4096);
    EnsureCapacityLocked(0, &evicted_keys);
    for (const std::string& evicted : evicted_keys) {
      MarkTombstoneLocked(evicted);
    }
  }
  for (const std::string& evicted : evicted_keys) {
    target_->Erase(Slice(evicted));
  }
  // Backing cache is intentionally over-provisioned to avoid introducing
  // a second independent eviction policy below Cacheus.
  target_->SetCapacity(ComputeBackingCapacity(capacity));
}

size_t CacheusCache::GetCapacity() const {
  std::lock_guard<std::mutex> lock(mu_);
  return capacity_;
}

std::string CacheusCache::GetPrintableOptions() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::ostringstream oss;
  const uint64_t wrapper_lookups = total_hit_count_ + total_miss_count_;
  const double wrapper_hit_ratio =
      wrapper_lookups > 0
          ? static_cast<double>(total_hit_count_) /
                static_cast<double>(wrapper_lookups)
          : 0.0;
  oss << std::fixed << std::setprecision(6);
  oss << "cacheus.w_lru=" << weight_lru_ << "\n";
  oss << "cacheus.w_lfu=" << weight_lfu_ << "\n";
  oss << "cacheus.learning_rate=" << learning_rate_ << "\n";
  oss << "cacheus.capacity_semantics="
      << (entry_charge_equivalent_ ? "entries_equivalent" : "bytes") << "\n";
  oss << "cacheus.logical_usage=" << usage_ << "\n";
  oss << "cacheus.capacity=" << capacity_ << "\n";
  oss << "cacheus.s_len=" << s_queue_.size() << "\n";
  oss << "cacheus.q_len=" << q_queue_.size() << "\n";
  oss << "cacheus.s_limit=" << s_limit_ << "\n";
  oss << "cacheus.q_limit=" << q_limit_ << "\n";
  oss << "cacheus.lru_hist_len=" << lru_hist_.size() << "\n";
  oss << "cacheus.lfu_hist_len=" << lfu_hist_.size() << "\n";
  oss << "cacheus.dem_count=" << dem_count_ << "\n";
  oss << "cacheus.nor_count=" << nor_count_ << "\n";
  oss << "cacheus.period_hits=" << period_hits_ << "\n";
  oss << "cacheus.total_hits=" << total_hit_count_ << "\n";
  oss << "cacheus.total_misses=" << total_miss_count_ << "\n";
  oss << "cacheus.pending_max_age_ops=" << pending_max_age_ops_ << "\n";
  oss << "cacheus.desync_backing_miss_reconciled="
      << desync_backing_miss_reconciled_count_ << "\n";
  oss << "cacheus.tombstone_lookup_dropped=" << tombstone_lookup_dropped_count_
      << "\n";
  oss << "cacheus.tombstone_size=" << pending_erased_keys_.size() << "\n";
  oss << "cacheus.lru_hist_hits=" << lru_hist_hit_count_ << "\n";
  oss << "cacheus.lfu_hist_hits=" << lfu_hist_hit_count_ << "\n";
  oss << "cacheus.evict_lru_count=" << evict_lru_count_ << "\n";
  oss << "cacheus.evict_lfu_count=" << evict_lfu_count_ << "\n";
  oss << "cacheus.evict_tie_count=" << evict_tie_count_ << "\n";
  oss << "cacheus.wrapper_lookups=" << wrapper_lookups << "\n";
  oss << "cacheus.wrapper_hits=" << total_hit_count_ << "\n";
  oss << "cacheus.wrapper_hit_ratio=" << wrapper_hit_ratio;
  return oss.str();
}

CacheusCache::DebugSnapshot CacheusCache::TEST_GetSnapshot() const {
  std::lock_guard<std::mutex> lock(mu_);
  DebugSnapshot snapshot;
  snapshot.w_lru = weight_lru_;
  snapshot.w_lfu = weight_lfu_;
  snapshot.s_limit = s_limit_;
  snapshot.q_limit = q_limit_;
  snapshot.logical_usage = usage_;
  snapshot.s_len = s_queue_.size();
  snapshot.q_len = q_queue_.size();
  snapshot.lru_hist_len = lru_hist_.size();
  snapshot.lfu_hist_len = lfu_hist_.size();
  snapshot.dem_count = dem_count_;
  snapshot.nor_count = nor_count_;
  snapshot.period_hits = period_hits_;
  snapshot.evict_lru_count = evict_lru_count_;
  snapshot.evict_lfu_count = evict_lfu_count_;
  snapshot.evict_tie_count = evict_tie_count_;
  snapshot.last_evicted_key = last_evicted_key_;
  snapshot.last_evicted_policy = last_evicted_policy_;
  return snapshot;
}

void CacheusCache::TEST_RequestStep(const Slice& key, size_t charge) {
  const std::string key_str = SliceToKey(key);
  std::vector<std::string> evicted_keys;
  const size_t logical_charge = entry_charge_equivalent_ ? 1 : charge;
  {
    std::lock_guard<std::mutex> lock(mu_);
    ResetStepTraceLocked();
    ++time_;
    UpdateLearningRateLocked();

    auto it = entries_.find(key_str);
    if (it != entries_.end()) {
      ++total_hit_count_;
      TouchOnHitLocked(key_str);
      ++period_hits_;
      return;
    }
    ++total_miss_count_;

    auto lru_it = lru_hist_pos_.find(key_str);
    if (lru_it != lru_hist_pos_.end()) {
      ++lru_hist_hit_count_;
      const uint64_t freq = std::max<uint64_t>(1, lru_it->second->second.freq + 1);
      const bool was_new = lru_it->second->second.is_new;
      lru_hist_.erase(lru_it->second);
      lru_hist_pos_.erase(lru_it);
      if (was_new && nor_count_ > 0) {
        --nor_count_;
      }
      AdjustWeightsLocked(-1.0, 0.0);
      if (was_new && capacity_ > 0) {
        AdjustSizeLocked(false);
      }
      EnsureCapacityLocked(logical_charge, &evicted_keys);
      RecordInsertLocked(key_str, logical_charge, false, freq, true, true);
      return;
    }

    auto lfu_it = lfu_hist_pos_.find(key_str);
    if (lfu_it != lfu_hist_pos_.end()) {
      ++lfu_hist_hit_count_;
      const uint64_t freq = std::max<uint64_t>(1, lfu_it->second->second.freq + 1);
      lfu_hist_.erase(lfu_it->second);
      lfu_hist_pos_.erase(lfu_it);
      AdjustWeightsLocked(0.0, -1.0);
      EnsureCapacityLocked(logical_charge, &evicted_keys);
      RecordInsertLocked(key_str, logical_charge, false, freq, true, true);
      return;
    }

    // Miss path aligned with Python miss():
    // 1) Fill S first when Q is empty and S has room.
    if (s_usage_ < s_limit_ && q_queue_.empty()) {
      RecordInsertLocked(key_str, logical_charge, false, 1, true, false);
      return;
    }
    // 2) Fill Q if cache not full and Q has room.
    if (usage_ < capacity_ && q_queue_.size() < q_limit_) {
      RecordInsertLocked(key_str, logical_charge, false, 1, false, false);
      return;
    }
    // 3) Otherwise evict if full, insert to Q as new, and run limitStack.
    if (usage_ >= capacity_) {
      EnsureCapacityLocked(logical_charge, &evicted_keys);
    }
    RecordInsertLocked(key_str, logical_charge, true, 1, false, true);
  }
}

std::shared_ptr<Cache> NewCacheusCache(const LRUCacheOptions& options) {
  return NewCacheusCache(options, CacheusTuningOptions{});
}

std::shared_ptr<Cache> NewCacheusCache(
    const LRUCacheOptions& options, const CacheusTuningOptions& tuning_options) {
  return std::make_shared<CacheusCache>(
      BuildBackingCache(options), options.capacity, tuning_options);
}

std::shared_ptr<Cache> NewCacheusCache(size_t capacity, int num_shard_bits) {
  LRUCacheOptions options;
  options.capacity = capacity;
  options.num_shard_bits = num_shard_bits;
  return NewCacheusCache(options);
}

}  // namespace ROCKSDB_NAMESPACE
