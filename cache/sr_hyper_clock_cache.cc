#include "cache/sr_hyper_clock_cache.h"

namespace ROCKSDB_NAMESPACE {

std::shared_ptr<Cache> NewSRHyperClockCache(
    const HyperClockCacheOptions& hcc_options) {
  HyperClockCacheOptions opts = hcc_options;
  opts.probation_insert = true;
  return opts.MakeSharedCache();
}

}  // namespace ROCKSDB_NAMESPACE

