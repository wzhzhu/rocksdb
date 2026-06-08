# Cacheus 确定性模式配置模板

本文给出 `cacheus` 在 `db_bench` 中的可复现实验模板，目标是：

- 相同参数 + 相同 workload 下，得到稳定可对比结果
- 能周期打印 `Cacheus` 运行时状态，便于调参与回归分析

## 1) 单缓存（非 MultiLevelCache）模板

使用 `--cache_uri` 显式指定 `cacheus://` 参数，推荐固定以下字段：

- `rng_seed`：固定随机种子（建议实验内保持不变）
- `period_len`：学习率窗口长度
- `history_size`：ghost history 大小
- `initial_weight` / `learning_rate`：初始策略权重与初始学习率

示例：

```bash
./db_bench \
  --benchmarks=readrandom \
  --db=/tmp/db_cacheus_det \
  --use_existing_db=true \
  --threads=8 \
  --duration=60 \
  --stats_interval=100000 \
  --stats_per_interval=1 \
  --cache_printable_stats_per_interval=true \
  --cache_uri="cacheus://capacity=1073741824;num_shard_bits=6;initial_weight=0.5;learning_rate=0.45;history_size=131072;period_len=262144;rng_seed=123"
```

## 2) MultiLevelCache + Cacheus 子缓存模板

当启用多层缓存时，推荐使用：

- `--use_multi_level_cache=true`
- `--multi_level_cache_sub_cache_uri="cacheus://..."`

并固定：

- `--num_levels`
- `--multi_level_cache_force_route_all_to_l0`（是否强制全部路由到 L0）
- `cacheus://` 内各参数（尤其 `rng_seed`）

示例：

```bash
./db_bench \
  --benchmarks=readrandom \
  --db=/tmp/db_mlc_cacheus_det \
  --use_existing_db=true \
  --threads=8 \
  --duration=60 \
  --num_levels=7 \
  --use_multi_level_cache=true \
  --multi_level_cache_force_route_all_to_l0=false \
  --multi_level_cache_sub_cache_uri="cacheus://initial_weight=0.5;learning_rate=0.45;history_size=65536;period_len=131072;rng_seed=123" \
  --stats_interval=100000 \
  --stats_per_interval=1 \
  --cache_printable_stats_per_interval=true
```

## 3) 参数建议（回归基线）

- `rng_seed`：固定为单一值（例如 `123`）
- `period_len`：与请求规模同量级（避免过小导致频繁抖动）
- `history_size`：至少覆盖一个中等重访周期
- `initial_weight`：`0.5`（中性起点）
- `learning_rate`：先用 `0.45` 对齐 Python 参考，再按实验目标调参

## 4) 运行时观测说明

开启 `--stats_per_interval=1` 且 `--cache_printable_stats_per_interval=true` 后，
`db_bench` 会周期打印 `Cache::GetPrintableOptions()`。

当使用 `cacheus` 时，输出包含：

- `cacheus.w_lru`, `cacheus.w_lfu`, `cacheus.learning_rate`
- `cacheus.s_len/q_len`, `cacheus.s_limit/q_limit`
- `cacheus.dem_count/nor_count`
- `cacheus.lru_hist_hits/lfu_hist_hits`
- `cacheus.evict_lru_count/evict_lfu_count/evict_tie_count`
- `cacheus.total_hits/total_misses`

当使用 `MultiLevelCache` 时，会额外打印每层子缓存名称及其 `printable stats`（若子缓存支持）。

## 5) 复现实验注意事项

- 固定二进制版本与编译参数
- 固定 benchmark 参数（尤其 `threads`、`duration`、key/value 分布）
- 固定预热策略（是否先 fill，再 read）
- 避免混入后台不相关负载（减少系统抖动）

