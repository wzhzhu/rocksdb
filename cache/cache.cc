//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "rocksdb/cache.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <vector>

#include "cache/arc_cache.h"
#include "cache/cacheus_cache.h"
#include "cache/lru_cache.h"
#include "rocksdb/secondary_cache.h"
#include "rocksdb/utilities/customizable_util.h"
#include "rocksdb/utilities/options_type.h"
#include "util/string_util.h"

namespace ROCKSDB_NAMESPACE {
const Cache::CacheItemHelper kNoopCacheItemHelper{};

static std::unordered_map<std::string, OptionTypeInfo>
    lru_cache_options_type_info = {
        {"capacity",
         {offsetof(struct LRUCacheOptions, capacity), OptionType::kSizeT,
          OptionVerificationType::kNormal, OptionTypeFlags::kMutable}},
        {"num_shard_bits",
         {offsetof(struct LRUCacheOptions, num_shard_bits), OptionType::kInt,
          OptionVerificationType::kNormal, OptionTypeFlags::kMutable}},
        {"strict_capacity_limit",
         {offsetof(struct LRUCacheOptions, strict_capacity_limit),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kMutable}},
        {"high_pri_pool_ratio",
         {offsetof(struct LRUCacheOptions, high_pri_pool_ratio),
          OptionType::kDouble, OptionVerificationType::kNormal,
          OptionTypeFlags::kMutable}},
        {"low_pri_pool_ratio",
         {offsetof(struct LRUCacheOptions, low_pri_pool_ratio),
          OptionType::kDouble, OptionVerificationType::kNormal,
          OptionTypeFlags::kMutable}},
};

static std::unordered_map<std::string, OptionTypeInfo>
    comp_sec_cache_options_type_info = {
        {"capacity",
         {offsetof(struct CompressedSecondaryCacheOptions, capacity),
          OptionType::kSizeT, OptionVerificationType::kNormal,
          OptionTypeFlags::kMutable}},
        {"num_shard_bits",
         {offsetof(struct CompressedSecondaryCacheOptions, num_shard_bits),
          OptionType::kInt, OptionVerificationType::kNormal,
          OptionTypeFlags::kMutable}},
        {"compression_type",
         {offsetof(struct CompressedSecondaryCacheOptions, compression_type),
          OptionType::kCompressionType, OptionVerificationType::kNormal,
          OptionTypeFlags::kMutable}},
        {"enable_custom_split_merge",
         {offsetof(struct CompressedSecondaryCacheOptions,
                   enable_custom_split_merge),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kMutable}},
};

namespace {
struct CacheusCreateOptions {
  double initial_weight = 0.5;
  double learning_rate = -1.0;
  size_t history_size = 0;
  uint64_t period_len = 0;
  uint64_t rng_seed = 123;
  bool entry_charge_equivalent = false;
  uint64_t pending_max_age_ops = 65536;
};

struct ArcCreateOptions {
  uint64_t pending_max_age_ops = 65536;
};

std::string TrimCopy(std::string s) {
  size_t start = 0;
  while (start < s.size() &&
         std::isspace(static_cast<unsigned char>(s[start])) != 0) {
    ++start;
  }
  size_t end = s.size();
  while (end > start &&
         std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
    --end;
  }
  return s.substr(start, end - start);
}

Status ParseDoubleStrict(const std::string& input, double* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("null output for double parse");
  }
  char* end = nullptr;
  const double parsed = std::strtod(input.c_str(), &end);
  if (end == input.c_str() || (end != nullptr && *end != '\0')) {
    return Status::InvalidArgument("invalid double value: " + input);
  }
  *out = parsed;
  return Status::OK();
}

Status ParseUint64Strict(const std::string& input, uint64_t* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("null output for uint64 parse");
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(input.c_str(), &end, 10);
  if (end == input.c_str() || (end != nullptr && *end != '\0')) {
    return Status::InvalidArgument("invalid uint64 value: " + input);
  }
  *out = static_cast<uint64_t>(parsed);
  return Status::OK();
}

Status ParseBoolStrict(const std::string& input, bool* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("null output for bool parse");
  }
  std::string lowered = input;
  for (char& c : lowered) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (lowered == "1" || lowered == "true" || lowered == "yes" ||
      lowered == "on") {
    *out = true;
    return Status::OK();
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" ||
      lowered == "off") {
    *out = false;
    return Status::OK();
  }
  return Status::InvalidArgument("invalid bool value: " + input);
}

Status ParseCacheusOptionsFromArgs(std::string* args,
                                   CacheusCreateOptions* cacheus_options) {
  if (args == nullptr || cacheus_options == nullptr) {
    return Status::InvalidArgument("null arguments for cacheus parsing");
  }
  std::vector<std::string> passthrough;
  size_t start = 0;
  while (start <= args->size()) {
    size_t end = args->find(';', start);
    if (end == std::string::npos) {
      end = args->size();
    }
    std::string token = TrimCopy(args->substr(start, end - start));
    start = end + 1;
    if (token.empty()) {
      if (end == args->size()) {
        break;
      }
      continue;
    }
    size_t eq = token.find('=');
    if (eq == std::string::npos) {
      passthrough.emplace_back(std::move(token));
      if (end == args->size()) {
        break;
      }
      continue;
    }
    const std::string key = TrimCopy(token.substr(0, eq));
    const std::string value = TrimCopy(token.substr(eq + 1));
    if (key == "initial_weight") {
      double parsed = 0.0;
      Status s = ParseDoubleStrict(value, &parsed);
      if (!s.ok()) {
        return s;
      }
      cacheus_options->initial_weight = parsed;
    } else if (key == "learning_rate") {
      double parsed = 0.0;
      Status s = ParseDoubleStrict(value, &parsed);
      if (!s.ok()) {
        return s;
      }
      cacheus_options->learning_rate = parsed;
    } else if (key == "history_size") {
      uint64_t parsed = 0;
      Status s = ParseUint64Strict(value, &parsed);
      if (!s.ok()) {
        return s;
      }
      cacheus_options->history_size = static_cast<size_t>(parsed);
    } else if (key == "period_len") {
      uint64_t parsed = 0;
      Status s = ParseUint64Strict(value, &parsed);
      if (!s.ok()) {
        return s;
      }
      cacheus_options->period_len = parsed;
    } else if (key == "rng_seed") {
      uint64_t parsed = 0;
      Status s = ParseUint64Strict(value, &parsed);
      if (!s.ok()) {
        return s;
      }
      cacheus_options->rng_seed = parsed;
    } else if (key == "entry_charge_equivalent") {
      bool parsed = false;
      Status s = ParseBoolStrict(value, &parsed);
      if (!s.ok()) {
        return s;
      }
      cacheus_options->entry_charge_equivalent = parsed;
    } else if (key == "pending_max_age_ops") {
      uint64_t parsed = 0;
      Status s = ParseUint64Strict(value, &parsed);
      if (!s.ok()) {
        return s;
      }
      cacheus_options->pending_max_age_ops = std::max<uint64_t>(1, parsed);
    } else {
      passthrough.emplace_back(std::move(token));
    }
    if (end == args->size()) {
      break;
    }
  }

  std::string filtered;
  for (size_t i = 0; i < passthrough.size(); ++i) {
    if (i > 0) {
      filtered.append(";");
    }
    filtered.append(passthrough[i]);
  }
  *args = std::move(filtered);
  return Status::OK();
}

Status ParseArcOptionsFromArgs(std::string* args, ArcCreateOptions* arc_options) {
  if (args == nullptr || arc_options == nullptr) {
    return Status::InvalidArgument("null arguments for arc parsing");
  }
  std::vector<std::string> passthrough;
  size_t start = 0;
  while (start <= args->size()) {
    size_t end = args->find(';', start);
    if (end == std::string::npos) {
      end = args->size();
    }
    std::string token = TrimCopy(args->substr(start, end - start));
    start = end + 1;
    if (token.empty()) {
      if (end == args->size()) {
        break;
      }
      continue;
    }
    size_t eq = token.find('=');
    if (eq == std::string::npos) {
      passthrough.emplace_back(std::move(token));
      if (end == args->size()) {
        break;
      }
      continue;
    }
    const std::string key = TrimCopy(token.substr(0, eq));
    const std::string value = TrimCopy(token.substr(eq + 1));
    if (key == "pending_max_age_ops") {
      uint64_t parsed = 0;
      Status s = ParseUint64Strict(value, &parsed);
      if (!s.ok()) {
        return s;
      }
      arc_options->pending_max_age_ops = std::max<uint64_t>(1, parsed);
    } else {
      passthrough.emplace_back(std::move(token));
    }
    if (end == args->size()) {
      break;
    }
  }

  std::string filtered;
  for (size_t i = 0; i < passthrough.size(); ++i) {
    if (i > 0) {
      filtered.append(";");
    }
    filtered.append(passthrough[i]);
  }
  *args = std::move(filtered);
  return Status::OK();
}

static void NoopDelete(Cache::ObjectPtr /*obj*/,
                       MemoryAllocator* /*allocator*/) {
  assert(false);
}

static size_t SliceSize(Cache::ObjectPtr obj) {
  return static_cast<Slice*>(obj)->size();
}

static Status SliceSaveTo(Cache::ObjectPtr from_obj, size_t from_offset,
                          size_t length, char* out) {
  const Slice& slice = *static_cast<Slice*>(from_obj);
  std::memcpy(out, slice.data() + from_offset, length);
  return Status::OK();
}

static Status NoopCreate(const Slice& /*data*/, CompressionType /*type*/,
                         CacheTier /*source*/, Cache::CreateContext* /*ctx*/,
                         MemoryAllocator* /*allocator*/,
                         Cache::ObjectPtr* /*out_obj*/,
                         size_t* /*out_charge*/) {
  assert(false);
  return Status::NotSupported();
}

static Cache::CacheItemHelper kBasicCacheItemHelper(CacheEntryRole::kMisc,
                                                    &NoopDelete);
}  // namespace

const Cache::CacheItemHelper kSliceCacheItemHelper{
    CacheEntryRole::kMisc, &NoopDelete, &SliceSize,
    &SliceSaveTo,          &NoopCreate, &kBasicCacheItemHelper,
};

Status SecondaryCache::CreateFromString(
    const ConfigOptions& config_options, const std::string& value,
    std::shared_ptr<SecondaryCache>* result) {
  if (value.find("compressed_secondary_cache://") == 0) {
    std::string args = value;
    args.erase(0, std::strlen("compressed_secondary_cache://"));
    Status status;
    std::shared_ptr<SecondaryCache> sec_cache;

    CompressedSecondaryCacheOptions sec_cache_opts;
    status = OptionTypeInfo::ParseStruct(config_options, "",
                                         &comp_sec_cache_options_type_info, "",
                                         args, &sec_cache_opts);
    if (status.ok()) {
      sec_cache = NewCompressedSecondaryCache(sec_cache_opts);
    }

    if (status.ok()) {
      result->swap(sec_cache);
    }
    return status;
  } else {
    return LoadSharedObject<SecondaryCache>(config_options, value, result);
  }
}

Status Cache::CreateFromString(const ConfigOptions& config_options,
                               const std::string& value,
                               std::shared_ptr<Cache>* result) {
  Status status;
  std::shared_ptr<Cache> cache;
  if (StartsWith(value, "null")) {
    cache = nullptr;
  } else if (StartsWith(value, "cacheus://")) {
    std::string args = value;
    args.erase(0, std::strlen("cacheus://"));
    CacheusCreateOptions cacheus_create_opts;
    status = ParseCacheusOptionsFromArgs(&args, &cacheus_create_opts);
    LRUCacheOptions cache_opts;
    if (status.ok()) {
      if (args.empty()) {
        cache_opts = LRUCacheOptions(8ULL << 20, -1, false, 0.5);
      } else {
        status = OptionTypeInfo::ParseStruct(config_options, "",
                                             &lru_cache_options_type_info, "",
                                             args, &cache_opts);
      }
    }
    if (status.ok()) {
      CacheusTuningOptions tuning_options;
      tuning_options.initial_weight = cacheus_create_opts.initial_weight;
      tuning_options.learning_rate = cacheus_create_opts.learning_rate;
      tuning_options.history_size = cacheus_create_opts.history_size;
      tuning_options.period_len = cacheus_create_opts.period_len;
      tuning_options.rng_seed = cacheus_create_opts.rng_seed;
      tuning_options.entry_charge_equivalent =
          cacheus_create_opts.entry_charge_equivalent;
      tuning_options.pending_max_age_ops =
          cacheus_create_opts.pending_max_age_ops;
      cache = NewCacheusCache(cache_opts, tuning_options);
    }
    if (status.ok()) {
      result->swap(cache);
    }
  } else if (StartsWith(value, "arc://")) {
    std::string args = value;
    args.erase(0, std::strlen("arc://"));
    ArcCreateOptions arc_create_opts;
    status = ParseArcOptionsFromArgs(&args, &arc_create_opts);
    LRUCacheOptions cache_opts;
    if (status.ok() && args.empty()) {
      cache_opts = LRUCacheOptions(8ULL << 20, -1, false, 0.5);
    } else if (status.ok()) {
      status = OptionTypeInfo::ParseStruct(config_options, "",
                                           &lru_cache_options_type_info, "",
                                           args, &cache_opts);
    }
    if (status.ok()) {
      ARCTuningOptions tuning_options;
      tuning_options.pending_max_age_ops = arc_create_opts.pending_max_age_ops;
      cache = NewARCCache(cache_opts, tuning_options);
      result->swap(cache);
    }
  } else if (value.find("://") == std::string::npos) {
    if (value.find('=') == std::string::npos) {
      cache = NewLRUCache(ParseSizeT(value));
    } else {
      LRUCacheOptions cache_opts;
      status = OptionTypeInfo::ParseStruct(config_options, "",
                                           &lru_cache_options_type_info, "",
                                           value, &cache_opts);
      if (status.ok()) {
        cache = NewLRUCache(cache_opts);
      }
    }
    if (status.ok()) {
      result->swap(cache);
    }
  } else {
    status = LoadSharedObject<Cache>(config_options, value, result);
  }
  return status;
}

bool Cache::AsyncLookupHandle::IsReady() {
  return pending_handle == nullptr || pending_handle->IsReady();
}

bool Cache::AsyncLookupHandle::IsPending() { return pending_handle != nullptr; }

Cache::Handle* Cache::AsyncLookupHandle::Result() {
  assert(!IsPending());
  return result_handle;
}

void Cache::StartAsyncLookup(AsyncLookupHandle& async_handle) {
  async_handle.found_dummy_entry = false;  // in case re-used
  assert(!async_handle.IsPending());
  async_handle.result_handle =
      Lookup(async_handle.key, async_handle.helper, async_handle.create_context,
             async_handle.priority, async_handle.stats);
}

Cache::Handle* Cache::Wait(AsyncLookupHandle& async_handle) {
  WaitAll(&async_handle, 1);
  return async_handle.Result();
}

void Cache::WaitAll(AsyncLookupHandle* async_handles, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (async_handles[i].IsPending()) {
      // If a pending handle gets here, it should be marked at "to be handled
      // by a caller" by that caller erasing the pending_cache on it.
      assert(async_handles[i].pending_cache == nullptr);
    }
  }
}

void Cache::SetEvictionCallback(EvictionCallback&& fn) {
  // Overwriting non-empty with non-empty could indicate a bug
  assert(!eviction_callback_ || !fn);
  eviction_callback_ = std::move(fn);
}

}  // namespace ROCKSDB_NAMESPACE
