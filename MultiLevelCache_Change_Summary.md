# MultiLevelCache 本次代码修改详细说明

- 提交哈希：`c8eacb097`
- 提交信息：`Add MultiLevelCache with level-aware routing and runtime controls`
- 影响文件：
  - `cache/multi_level_cache.h`
  - `cache/multi_level_cache.cc`
  - `db/version_set.cc`

## 1. 新增 `MultiLevelCache` 主体实现

在 `cache/multi_level_cache.h/.cc` 新增了 `MultiLevelCache`，继承自 `rocksdb::Cache`，并实现多层子缓存封装。

主要内容：

- 内部维护按层子缓存容器：`sub_caches_`
- 构造时按层数创建子缓存，默认容量均分（余数分配到前几层）
- 覆盖 `Cache` 核心接口，包括 `Insert`、`Lookup`、`Release`、`Erase`、`SetCapacity`、`GetUsage` 等
- 引入 `WrappedHandle` 包装底层 handle，保证跨层路由后 `Ref/Release/Value/GetCharge/GetUsage` 等操作始终落到正确子缓存

## 2. 实现按 SST 层级路由（并修复键空间冲突）

在 `MultiLevelCache` 中实现了基于缓存键前缀的路由逻辑，并修复了“`file_number` 与 `cache_key_prefix` 共用同一映射表”导致的键空间冲突。

主要内容：

- 新增并发映射（分片 `unordered_map + shared_mutex`，分片数固定 64）：
  - `file_number_to_level_`
  - `cache_key_prefix_to_level_`
- 新增/更新接口：
  - `UpdateFileLevelMapping(uint64_t file_number, int level)`
  - `UpdateCacheKeyPrefixMapping(uint64_t cache_key_prefix, int level)`
- 路由流程（`Insert`/`Lookup`）：
  1. 从 `Cache key` 中提取 8-byte common prefix（`DecodeFixed64`）
  2. 查询 `cache_key_prefix -> level` 映射
  3. 命中则路由到 `sub_caches_[level]`
  4. 未命中或解析失败回退到 `sub_caches_[0]`

## 3. 接入 `VersionSet` 自动刷新映射（含生命周期清理）

在 `db/version_set.cc` 中加入了版本切换时对层级映射的同步逻辑，并补上映射生命周期管理（避免“只增不删”）。

主要内容：

- 新增/更新辅助函数：
  - `MaybeRefreshLevelCacheMapping(Cache*, const VersionStorageInfo&, const VersionStorageInfo* old_storage_info, ...)`
- 通过 `CheckedCast<MultiLevelCache>()` 进行类型门控，仅在使用 `MultiLevelCache` 时生效
- 在 `VersionSet::AppendVersion()` 中调用同步函数
- 使用当前 `VersionStorageInfo` 全量刷新活跃文件映射
- 对 `old_storage_info` 与新版本做差量比较，删除不再存活文件的映射：
  - `RemoveFileLevelMapping(file_number)`
  - `RemoveCacheKeyPrefixMapping(cache_key_prefix)`

## 4. 实现动态容量调整（Phase 3）

在 `MultiLevelCache` 中加入按层容量动态调整能力。

主要内容：

- 新增接口：`Status AdjustCapacities(const std::vector<size_t>& new_capacities)`
- 新增校验逻辑：
  - `new_capacities.size()` 必须等于层数
  - `sum(new_capacities)` 不得超过 `total_capacity_`
- 新增内部方法：
  - `ValidateCapacities(...)`
  - `ApplyCapacities(...)`
- 通过逐层调用底层 `SetCapacity()` 完成更新
- `total_capacity_` 使用原子变量存储，保证并发下读取校验成本低

## 5. 实现统计能力（Phase 4）

为论文 evaluation 增加按层命中统计。

主要内容：

- 新增按层统计原子计数：
  - `lookups_`
  - `hits_`
- 在 `Lookup` 路径增加统计：
  - 路由后先 `lookups_[level]++`
  - 命中（返回非空 handle）后 `hits_[level]++`
- 新增 `PrintStats() const`：
  - 输出总命中率
  - 输出各层 `capacity / lookups / hits / hit_rate`
- 新增 `ResetStats()`：
  - 可在每轮实验前清零统计计数
- 统计相关原子读写统一使用 `std::memory_order_relaxed`

后续增强（为实验可观测性）：

- 路由统计按调用类型拆分：
  - `route_insert: queries / parse_failures / prefix_hits / prefix_misses / prefix_hit_rate`
  - `route_lookup: queries / parse_failures / prefix_hits / prefix_misses / prefix_hit_rate`
- 新增 `route_normalize_fallbacks`：
  - 当映射命中但 level 非法（负数或越界）时，回落 L0 并计数
- 新增映射表规模观测：
  - `mapping_entries: file_number=..., cache_key_prefix=...`
  - 便于直接观察映射生命周期清理效果

## 6. 并发与性能设计说明

- 容量调整路径未引入额外全局重锁，依赖底层 cache（`ShardedCache/LRUCache`）已有同步机制
- 路由映射采用分片锁，降低全局锁竞争
- 统计路径使用 `relaxed` 原子语义，减少前台读写开销
- 设计目标是优先保证：
  - 前台查询路径轻量
  - 映射和统计在高并发下可观测且可维护

## 7. 当前版本行为总结

- 默认子缓存容量初始化为均分
- 路由优先按 `cache_key_prefix -> level`；映射未命中回退到 L0 子缓存
- 支持外部算法周期性下发容量向量进行动态调整
- 支持实验前清零统计与实验后导出：
  - 按层命中率
  - Insert/Lookup 路由命中质量
  - 映射表实时条目数

## 8. Runner 与文档一致性修正

- 修正 `tools/multilevel_cache_runner.cc` 注释与行为不一致问题：
  - 由“make lower levels larger”改为“make the deepest level larger”
  - 与当前容量分配逻辑（最后一层吃剩余容量）一致

## 9. 新增凸优化求解器与后台分配模块

新增 `cache/multi_level_cache_allocator.h/.cc`，提供基于数学模型的在线容量求解与周期下发能力。

主要内容：

- 新增 `MultiLevelAllocationOptions`：
  - `interval_ms`
  - `smoothing_ratio`
  - `min_total_change_bytes`
  - `solver_epsilon`
  - `solver_max_iterations`
- 新增 `MultiLevelCacheAllocator`：
  - `Start()/Stop()/RunOnce()`
  - `MetricsProvider` 回调接口（输入 `lambda_i/D_i/alpha_i`）
  - 静态求解接口 `SolveCapacities(...)`
- 求解算法：
  - 基于 KKT 条件和 `mu` 的二分查找
  - 先求连续解，再量化到 `size_t` 容量并保证预算守恒
- 工程增强：
  - 支持容量平滑与最小变更阈值（防抖）

## 10. 量化预算分配缺陷修复

修复了“预算很大时最多只补 N 字节剩余量”的问题。

问题根因：

- 旧逻辑在分配 `remaining = budget - used` 时，仅对前 `N` 个层做一次 `+1`，导致 `remaining >> N` 场景下大量预算未被分配。

修复方式：

- 先按 `remaining / N` 批量平均分配给所有层；
- 再将 `remaining % N` 按小数部分排序补齐。
- 同样修复了平滑路径中的同类剩余分配问题。

## 11. 接入在线自适应 demo（runner）

`tools/multilevel_cache_runner.cc` 从“固定向量调整”升级为“在线周期求解 + 自动下发”示例。

主要内容：

- 接入 `MultiLevelCacheAllocator`
- 在第二轮读请求中开启后台线程周期求解
- 运行结束打印 `Allocator capacities`

## 12. D_i（层数据量）采集链路增强

为让 `D_i` 更贴近真实值，新增了文件级元数据到层级数据量的聚合。

主要内容：

- `MultiLevelCache` 新增：
  - `UpdateFileMetadata(file_number, level, file_size)`
  - `GetLevelMetricsSnapshot()`（输出 `lookups/hits/capacities/data_sizes`）
- 内部维护：
  - `file_number -> level`
  - `file_number -> file_size`
  - 聚合得到 `level_data_sizes_`
- 在 `BlockBasedTable::Open` 与 `VersionSet` 刷新路径中都调用 `UpdateFileMetadata(...)`，提升观测及时性和覆盖率。

## 13. 代码审查后修复项

根据新增代码审查结果，已完成以下修复：

- 修复 `D_i` 双写冲突：
  - 删除按层 `store` 覆盖路径（`UpdateLevelDataSize`）
  - 统一使用文件级增量聚合路径（`UpdateFileMetadata`）
- 删除未使用的 `FindLevelByFileNumber` 路径，减少冗余
- 优化 runner 指标构造：
  - 冷层 `lambda` 从强制 `+1` 改为 `epsilon`
  - `D_i` 优先使用真实 `data_size`，缺失时使用已观测层均值兜底
  - demo 中 `alpha` 统一为 `1.0`，减少人为偏置
