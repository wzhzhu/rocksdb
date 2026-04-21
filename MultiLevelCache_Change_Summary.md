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

## 2. 实现按 SST 层级路由

在 `MultiLevelCache` 中实现了基于文件层级的路由逻辑。

主要内容：

- 新增并发映射：`file_to_level_`
  - 结构为分片 `unordered_map + shared_mutex`
  - 分片数量固定（64）
- 新增接口：`UpdateFileLevelMapping(uint64_t file_number, int level)`
- 路由流程（`Insert`/`Lookup`）：
  1. 从 `Cache key` 中提取 `file_number`（当前实现为首 8 字节 `DecodeFixed64`）
  2. 查询 `file_number -> level` 映射
  3. 命中则路由到 `sub_caches_[level]`
  4. 未命中或解析失败回退到 `sub_caches_[0]`

## 3. 接入 `VersionSet` 自动刷新映射

在 `db/version_set.cc` 中加入了版本切换时对层级映射的同步逻辑。

主要内容：

- 新增辅助函数：`MaybeRefreshLevelCacheMapping(Cache*, const VersionStorageInfo&)`
- 通过 `CheckedCast<MultiLevelCache>()` 进行类型门控，仅在使用 `MultiLevelCache` 时生效
- 在 `VersionSet::AppendVersion()` 中调用同步函数
- 使用当前 `VersionStorageInfo` 全量遍历各层 SST 文件，刷新 `file_number -> level`

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

## 6. 并发与性能设计说明

- 容量调整路径未引入额外全局重锁，依赖底层 cache（`ShardedCache/LRUCache`）已有同步机制
- 路由映射采用分片锁，降低全局锁竞争
- 统计路径使用 `relaxed` 原子语义，减少前台读写开销
- 设计目标是优先保证：
  - 前台查询路径轻量
  - 映射和统计在高并发下可观测且可维护

## 7. 当前版本行为总结

- 默认子缓存容量初始化为均分
- 路由优先按层映射；映射未命中回退到 L0 子缓存
- 支持外部算法周期性下发容量向量进行动态调整
- 支持实验前清零统计与实验后按层命中率导出
