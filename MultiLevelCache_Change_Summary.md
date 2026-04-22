# MultiLevelCache 修改说明（Step B 最终版）

本文档描述当前主线实现：**cache key 显式携带 level，`MultiLevelCache` 纯直解路由，不再依赖路由映射表**。

## 1. 核心架构（当前状态）

- `MultiLevelCache` 继承 `rocksdb::Cache`，内部维护按层子缓存 `sub_caches_`
- 构造时按层数创建 LRU 子缓存，默认容量均分（余数前置分配）
- 所有 `Insert/Lookup/CreateStandalone` 先路由再访问对应子缓存
- 通过 `WrappedHandle` 保证 `Ref/Release/Value/GetCharge/GetUsage` 始终回到正确层

## 2. 路由机制（Step A + Step B 收敛）

- 在 `cache/cache_key.h/.cc` 中增加 level 编解码能力，并提供 `OffsetableCacheKey::WithLevel(int level)`
- 在 `BlockBasedTable::SetupBaseCacheKey(...)` 中将 level 注入 common prefix
- `MultiLevelCache::RouteLevelByKey(...)` 只做两步：
  1. 从 key 前 8 字节取 common prefix（`DecodeFixed64`）
  2. 直接 `DecodeLevelFromEncodedCacheKeyCommonPrefix(...)` 得到 level
- 若 key 未携带可识别 level，则回退 `L0`

> Step B 后已删除 `file_number_to_level_` / `cache_key_prefix_to_level_` 及其 fallback 路径。

## 3. `VersionSet` 与状态同步（Step B 调整）

- `VersionSet::AppendVersion()` 中调用 `MaybeRefreshLevelCacheState(...)`
- 该函数不再维护任何路由映射，只负责聚合每层 SST 总大小
- 通过 `MultiLevelCache::UpdateLevelDataSizes(...)` 更新 `level_data_sizes_`
- 目的：为 allocator 提供 `D_i` 指标，且与路由链路解耦

## 4. 动态容量调整（Phase 3）

- 接口：`Status AdjustCapacities(const std::vector<size_t>& new_capacities)`
- 校验：
  - `new_capacities.size() == num_levels`
  - `sum(new_capacities) <= total_capacity_`
- 通过逐层 `SetCapacity()` 生效
- `total_capacity_` 使用原子变量保存，读校验开销低

## 5. 统计能力（Phase 4）

- 按层统计：
  - `lookups_`
  - `hits_`
  - `level_data_sizes_`
- 路由统计（按调用类型拆分）：
  - `route_insert: queries / parse_failures / prefix_hits / prefix_misses / prefix_hit_rate`
  - `route_lookup: queries / parse_failures / prefix_hits / prefix_misses / prefix_hit_rate`
  - `route_normalize_fallbacks`
- `PrintStats()` 输出总命中率与逐层命中率
- `ResetStats()` 支持实验前清零
- 统计原子操作统一使用 `std::memory_order_relaxed`

> Step B 后 `PrintStats()` 不再输出 `mapping_entries`，因为映射表已移除。

## 6. 并发与性能设计

- 前台路由路径不再涉及 map 查找和分片锁，仅做 key 解析与位级解码
- 容量调整依赖底层 `ShardedCache/LRUCache` 已有同步机制，不引入额外全局大锁
- 统计路径使用 relaxed 原子语义，降低热点开销

## 7. Allocator（数学模型模块）

`cache/multi_level_cache_allocator.h/.cc` 提供在线求解与周期下发：

- `MultiLevelAllocationOptions`：`interval_ms` / `smoothing_ratio` / `min_total_change_bytes` / `solver_epsilon` / `solver_max_iterations`
- `MultiLevelCacheAllocator`：`Start()/Stop()/RunOnce()` + `MetricsProvider`
- `SolveCapacities(...)`：基于 KKT + 拉格朗日乘子 `mu` 二分
- 工程策略：预算量化、平滑、最小变更阈值防抖

### 已修复的关键问题

- 修复“大预算仅分配最多 N 字节剩余量”：
  - 先按 `remaining / N` 批量分发
  - 再按 `remaining % N` 分发余数
- 修复同类问题在平滑路径中的残留

## 8. Runner / db_bench 接入

- `tools/db_bench_tool.cc` 支持 `--use_multi_level_cache=true` 并可自动接入 allocator
- `tools/multilevel_cache_runner.cc` 提供在线求解演示
- `tools/verify_mlc_routing.sh` 用于一键验证路由正确性（fill + read + 自动判定）

## 9. 当前版本行为总结

- 默认多层 LRU 子缓存均分容量
- 路由优先且仅依赖 key 编码 level（纯直解）
- 非编码 key 回退 L0
- 支持外部算法按周期调节容量
- 支持实验前后统计清零与导出

## 10. 路由验证结论（Step B 回归）

在 `db_bench` 场景下已验证：

- `route_lookup` 查询量充足
- `prefix_hit_rate` 接近 1.0
- `route_normalize_fallbacks=0`
- 多层场景可观测到非单层访问

结论：**Step B 后路由链路已稳定收敛为“key-level 直解路由”，可继续用于 allocator 策略实验。**

## 11. 当前遗留风险（待后续解决）

本节记录当前版本仍存在的已知风险与建议处理方向，作为后续迭代 TODO。

### 风险 R1：level 编码对 common prefix 有损（高优先级）

- 现状：
  - 当前编码布局为 `[marker | level | payload]`，其中 payload 只保留了原始 64-bit 前缀的低 53 bit。
  - 这会丢失 11 bit 原始信息，导致 key 空间压缩。
- 影响：
  - 理论上提高 cache key 前缀碰撞概率。
  - 在大规模/长时间/高并发场景中，存在误路由或错误复用风险。
- 当前缓解：
  - 已通过路由统计脚本和回归测试确认常规负载下行为稳定。
- 后续建议：
  - 设计并落地“无损 level 携带方案”，避免截断原始前缀位信息。

### 风险 R2：当前编码仅支持最多 8 层（中高优先级）

- 现状：
  - 当前 `level` 编码位宽为 3 bit，仅支持 `0..7`。
- 影响：
  - 无法直接支持 `num_levels > 8` 的实验/部署配置。
- 当前缓解（已完成）：
  - 在 `db_bench`、`multilevel_cache_runner`、`verify_mlc_routing.sh` 增加显式参数校验。
  - 超过 8 层时直接失败退出，避免“静默回绕”。
- 后续建议：
  - 若需要 >8 层，需同步升级编码方案并回归验证。

### 风险 R3：非主路径调用点可能不携带 level（低优先级兼容项）

- 现状：
  - 主路径（`BlockBasedTable::Open`）已传入 level，当前实验链路已按层直路由。
  - `SetupBaseCacheKey(...)` 仍允许 `level=-1`（默认值）调用。
  - 这类 key 在 MultiLevelCache 中会被识别为“非编码 key”，回退到 L0。
- 影响：
  - 在个别工具链/测试/旁路路径中，可能出现路由退化（非按层分流）。
- 当前缓解：
  - 主路径已解决，主实验链路正常。
- 后续建议：
  - 梳理所有调用 `SetupBaseCacheKey(...)` 的路径，明确哪些必须传 level。
  - 对不具备 level 信息的路径补充注释/文档约束或显式告警。

## 12. 后续处理优先级建议

1. **P0**：落地无损编码方案（解决 R1，兼顾 R2 扩展性）。
2. **P1**：扩展层数支持（>8）并补充单测/压力回归。
3. **P2**：按需清理非主路径调用点（兼容性完善），降低“隐式回落 L0”概率。

## Release Notes（简版）

- **架构变更**
  - 完成 Step B：`MultiLevelCache` 路由改为纯 `key-level` 直解（从 cache key 前缀解码 level）。
  - 移除 map-based 路由依赖与维护链路（不再依赖 file/prefix 映射表做路由）。
  - `VersionSet` 仅保留按层 `D_i`（`level_data_sizes_`）同步，用于 allocator 指标，不参与路由判定。

- **稳定性与防错**
  - 增加 encoded-level 上限防护（当前编码支持 8 层）：
    - `db_bench` / `multilevel_cache_runner` / 验证脚本都会拒绝 `num_levels > 8`，避免静默回绕。
  - 增加可选 miss 调试能力：
    - 通过 `MLC_ROUTE_DEBUG_MISS_LIMIT` 打印前 N 条 route miss（caller/reason/key_size/prefix）。

- **验证与工具**
  - 新增 `tools/verify_mlc_routing.sh`：一键做路由正确性检查（fill + read + 统计判定）。
  - 新增 `tools/verify_mlc_ci.sh`：CI 风格一键检查（编译、路由、guard、allocator 冒烟）。
  - 新增 `tools/compare_cache_hit_rate.sh`：MLC vs baseline 命中率对比。
  - 新增论文一页摘要文档：`MultiLevelCache_Paper_OnePager.md`。

- **当前已知风险（已记录）**
  - 编码方案当前对原始前缀有损（空间压缩），理论上仍有碰撞风险（后续可做无损编码重构）。
  - 非主路径少量 key 不带 level marker，可能出现极小量 `prefix_miss`（主要表现为统计噪声，对主路径影响很小）。
