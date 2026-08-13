# MultiLevelCache (Step B) One-Page Summary

> ⚠️ **Early architecture snapshot (2026-04-22); partially outdated.** The
> mapping-free routing design still holds, but sub-caches now default to
> HyperClockCache (not LRU) and capacity is set by a `robust_hit_rate` model
> allocator (not equal-split). Deepest LSM levels may be merged into one bottom
> slot. Current source of truth: `cache/multi_level_cache*.{h,cc}` and
> `YCSB-C-master/scripts/KNOWN_ISSUES.md`.

## 1. Problem and Goal

RocksDB block cache is traditionally a single shared cache. Under multi-level LSM workloads, a static single-pool policy may under-serve hot levels.  
This work implements `MultiLevelCache`, a level-partitioned cache with runtime capacity reallocation, and finalizes routing into a **mapping-free design**.

## 2. Final Architecture (Step B)

`MultiLevelCache` maintains one sub-cache per LSM level:

```
                    +-------------------------+
read/write block -> |      MultiLevelCache    |
cache key (16B)     |  (inherits rocksdb::Cache)
                    +-----------+-------------+
                                |
               decode level from key prefix (8B)
                                |
    +-------------+-------------+-------------+-------------+
    |             |             |             |             |
  sub_cache[L0] sub_cache[L1] sub_cache[L2] ...        sub_cache[Ln]
     (LRU)         (LRU)         (LRU)                    (LRU)
```

Key properties:

- Routing is **direct key decode** (`cache key prefix -> level`), no external lookup map.
- Each sub-cache is independently resizable (`AdjustCapacities()`).
- Per-level statistics are exported for online allocation and evaluation.

## 3. Routing Path

At SST open time, `BlockBasedTable::SetupBaseCacheKey(...)` encodes level into the 8-byte common prefix (`OffsetableCacheKey::WithLevel(level)`).

During `Insert` / `Lookup`:

1. Parse common prefix from cache key (`DecodeFixed64`).
2. Decode level bits from prefix marker/payload.
3. Route to `sub_caches_[level]`.
4. If decode fails (legacy/non-encoded key), fallback to `L0`.

Step B removes:

- `file_number_to_level_`
- `cache_key_prefix_to_level_`
- all mapping refresh/cleanup paths in `VersionSet` and `BlockBasedTable::Open`.

## 4. Runtime Metrics and Control

`MultiLevelCache` provides:

- Per-level: `lookups_i`, `hits_i`, `capacity_i`, `data_size_i`
- Route quality:
  - `route_insert: queries/parse_failures/prefix_hits/prefix_misses/prefix_hit_rate`
  - `route_lookup: queries/parse_failures/prefix_hits/prefix_misses/prefix_hit_rate`
  - `route_normalize_fallbacks`

Dynamic resizing:

- `AdjustCapacities(vector<size_t>)` with budget validation.
- Thread-safe stats and counters with `memory_order_relaxed`.

## 5. Allocator Model (Integrated)

`MultiLevelCacheAllocator` periodically collects `(lambda_i, D_i, alpha_i)` and solves:

- Objective: maximize weighted sum of level hit rates
- Constraint: `sum_i c_i <= C_total`
- Solver: KKT + binary search on Lagrange multiplier (`mu`)
- Engineering constraints:
  - quantized capacities with exact budget conservation
  - smoothing ratio
  - minimum total change threshold

`D_i` source in Step B:

- `VersionSet` computes per-level SST total size
- pushes vector via `UpdateLevelDataSizes(...)`

## 6. Validation Criteria (Routing Correctness)

Recommended pass criteria:

- `route_lookup.queries > 0`
- `route_lookup.prefix_hit_rate` close to `1.0` for encoded-key workloads
- `route_normalize_fallbacks == 0`
- multi-level workloads show non-zero activity in multiple levels

One-command validation script:

```bash
cd /home/gpu/wzhzhu/rocksdb
NUM=2000000 READS=1000000 THREADS=16 CACHE_SIZE=268435456 \
MIN_PREFIX_HIT_RATE=0.95 MIN_LOOKUP_QUERIES=1000 \
tools/verify_mlc_routing.sh ./build/db_bench
```

## 7. Main Contribution Summary

- Transition from mapping-based routing to **mapping-free direct routing**.
- Maintains allocator observability (`D_i`) without coupling to route state.
- Preserves high-throughput cache path while enabling adaptive per-level budgets.
