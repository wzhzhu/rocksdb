#pragma once

#include <cstdint>
#include <memory>

#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

// A scan-resistant HyperClockCache variant that enables probation-style
// insertion while preserving the original 64-bit SlotMeta layout.
std::shared_ptr<Cache> NewSRHyperClockCache(
    const HyperClockCacheOptions& hcc_options);

}  // namespace ROCKSDB_NAMESPACE

