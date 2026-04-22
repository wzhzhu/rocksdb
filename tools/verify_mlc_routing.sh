#!/usr/bin/env bash
set -euo pipefail

# Validate MultiLevelCache routing correctness (Step A transitional version):
# - key carries encoded level in common prefix
# - MultiLevelCache routes by decode first, mapping table as fallback
#
# Usage:
#   tools/verify_mlc_routing.sh [db_bench_path]
#
# Example:
#   tools/verify_mlc_routing.sh ./build/db_bench
#
# Environment overrides:
#   DB_PATH=/tmp/mlc_route_db
#   RESULT_DIR=/tmp/mlc_route_verify
#   NUM=2000000
#   READS=1000000
#   THREADS=16
#   VALUE_SIZE=128
#   CACHE_SIZE=268435456
#   NUM_LEVELS=7
#   COMPRESSION_TYPE=none
#   AUTO_ADJUST=false
#   MIN_PREFIX_HIT_RATE=0.00
#   MAX_NORMALIZE_FALLBACKS=0
#   REQUIRE_MULTI_LAYER=0
#   MIN_LOOKUP_QUERIES=1000
#   SEED=42

DB_BENCH_BIN="${1:-./build/db_bench}"
DB_PATH="${DB_PATH:-/tmp/mlc_route_db}"
RESULT_DIR="${RESULT_DIR:-/tmp/mlc_route_verify}"

NUM="${NUM:-2000000}"
READS="${READS:-1000000}"
THREADS="${THREADS:-16}"
VALUE_SIZE="${VALUE_SIZE:-128}"
CACHE_SIZE="${CACHE_SIZE:-268435456}"
NUM_LEVELS="${NUM_LEVELS:-7}"
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"
AUTO_ADJUST="${AUTO_ADJUST:-false}"
SEED="${SEED:-42}"

MIN_PREFIX_HIT_RATE="${MIN_PREFIX_HIT_RATE:-0.00}"
MAX_NORMALIZE_FALLBACKS="${MAX_NORMALIZE_FALLBACKS:-0}"
REQUIRE_MULTI_LAYER="${REQUIRE_MULTI_LAYER:-0}"
MIN_LOOKUP_QUERIES="${MIN_LOOKUP_QUERIES:-1000}"
MAX_ENCODED_LEVELS=8

FILL_LOG="${RESULT_DIR}/fillrandom.log"
READ_LOG="${RESULT_DIR}/readrandom.log"

if [[ ! -x "${DB_BENCH_BIN}" ]]; then
  echo "ERROR: db_bench not executable: ${DB_BENCH_BIN}" >&2
  echo "Tip: pass an absolute path, e.g. /home/gpu/wzhzhu/rocksdb/build/db_bench" >&2
  exit 1
fi

if [[ "${NUM_LEVELS}" -gt "${MAX_ENCODED_LEVELS}" ]]; then
  echo "ERROR: NUM_LEVELS=${NUM_LEVELS} exceeds encoded routing limit ${MAX_ENCODED_LEVELS}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"
rm -rf "${DB_PATH}"

echo "[1/3] fillrandom to build LSM levels..."
"${DB_BENCH_BIN}" \
  --benchmarks=fillrandom \
  --db="${DB_PATH}" \
  --num="${NUM}" \
  --threads="${THREADS}" \
  --value_size="${VALUE_SIZE}" \
  --cache_size="${CACHE_SIZE}" \
  --num_levels="${NUM_LEVELS}" \
  --use_existing_db=false \
  --use_multi_level_cache=true \
  --multi_level_cache_auto_adjust="${AUTO_ADJUST}" \
  --compression_type="${COMPRESSION_TYPE}" \
  --statistics=1 \
  --stats_interval_seconds=0 \
  --seed="${SEED}" \
  > "${FILL_LOG}" 2>&1

echo "[2/3] readrandom to trigger routing..."
"${DB_BENCH_BIN}" \
  --benchmarks=readrandom \
  --db="${DB_PATH}" \
  --num="${NUM}" \
  --reads="${READS}" \
  --threads="${THREADS}" \
  --value_size="${VALUE_SIZE}" \
  --cache_size="${CACHE_SIZE}" \
  --num_levels="${NUM_LEVELS}" \
  --use_existing_db=true \
  --use_multi_level_cache=true \
  --multi_level_cache_auto_adjust="${AUTO_ADJUST}" \
  --compression_type="${COMPRESSION_TYPE}" \
  --statistics=1 \
  --stats_interval_seconds=0 \
  --seed="${SEED}" \
  > "${READ_LOG}" 2>&1

echo "[3/3] parse routing statistics..."

route_lookup_line="$(awk '/route_lookup:/{line=$0} END{print line}' "${READ_LOG}")"
route_insert_line="$(awk '/route_insert:/{line=$0} END{print line}' "${READ_LOG}")"
route_normalize_line="$(awk '/route_normalize_fallbacks=/{line=$0} END{print line}' "${READ_LOG}")"

if [[ -z "${route_lookup_line}" || -z "${route_normalize_line}" ]]; then
  echo "FAIL: cannot find route stats in ${READ_LOG}" >&2
  echo "Hint: check if --use_multi_level_cache=true actually took effect." >&2
  exit 2
fi

lookup_queries="$(echo "${route_lookup_line}" | sed -E 's/.*queries=([0-9]+).*/\1/')"
lookup_prefix_hits="$(echo "${route_lookup_line}" | sed -E 's/.*prefix_hits=([0-9]+).*/\1/')"
lookup_prefix_misses="$(echo "${route_lookup_line}" | sed -E 's/.*prefix_misses=([0-9]+).*/\1/')"
lookup_prefix_hit_rate="$(echo "${route_lookup_line}" | sed -E 's/.*prefix_hit_rate=([0-9.]+).*/\1/')"
normalize_fallbacks="$(echo "${route_normalize_line}" | sed -E 's/.*=([0-9]+).*/\1/')"

non_zero_layers="$(awk '
  /^L[0-9]+:/ {
    if (match($0, /lookups=[0-9]+/)) {
      s = substr($0, RSTART, RLENGTH);
      split(s, a, "=");
      if ((a[2] + 0) > 0) c++;
    }
  }
  END { print c + 0; }
' "${READ_LOG}")"

echo "----- Routing Check Summary -----"
echo "route_insert: ${route_insert_line}"
echo "route_lookup: ${route_lookup_line}"
echo "route_normalize_fallbacks: ${normalize_fallbacks}"
echo "layers_with_nonzero_lookups: ${non_zero_layers}"
echo "logs: ${RESULT_DIR}"

pass=1

if [[ "${lookup_queries}" -le 0 ]]; then
  echo "FAIL: route_lookup.queries must be > 0" >&2
  pass=0
fi

if [[ "${lookup_queries}" -lt "${MIN_LOOKUP_QUERIES}" ]]; then
  echo "WARN: lookup queries (${lookup_queries}) < MIN_LOOKUP_QUERIES (${MIN_LOOKUP_QUERIES})." >&2
  echo "      This often means the workload did not generate enough block-cache traffic for strict routing validation." >&2
fi

if ! awk -v x="${lookup_prefix_hit_rate}" -v y="${MIN_PREFIX_HIT_RATE}" \
  'BEGIN{exit !(x >= y)}'; then
  echo "FAIL: prefix_hit_rate=${lookup_prefix_hit_rate} < ${MIN_PREFIX_HIT_RATE}" >&2
  pass=0
fi

if [[ "${lookup_prefix_hits}" -eq 0 && "${lookup_prefix_misses}" -gt 0 ]]; then
  echo "WARN: prefix_hits=0 with prefix_misses=${lookup_prefix_misses}." >&2
  echo "      For strict Step-A verification, rerun with a larger workload or set MIN_PREFIX_HIT_RATE=0.95." >&2
fi

if ! awk -v x="${normalize_fallbacks}" -v y="${MAX_NORMALIZE_FALLBACKS}" \
  'BEGIN{exit !(x <= y)}'; then
  echo "FAIL: route_normalize_fallbacks=${normalize_fallbacks} > ${MAX_NORMALIZE_FALLBACKS}" >&2
  pass=0
fi

if [[ "${REQUIRE_MULTI_LAYER}" == "1" && "${non_zero_layers}" -lt 2 ]]; then
  echo "FAIL: require at least 2 active levels, got ${non_zero_layers}" >&2
  pass=0
fi

if [[ "${pass}" -eq 1 ]]; then
  echo "PASS: MultiLevelCache routing checks passed."
  echo "      lookup queries=${lookup_queries}, prefix_hits=${lookup_prefix_hits}, prefix_misses=${lookup_prefix_misses}, prefix_hit_rate=${lookup_prefix_hit_rate}"
  exit 0
fi

exit 2
