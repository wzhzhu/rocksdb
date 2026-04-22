#!/usr/bin/env bash
set -euo pipefail

# CI-style end-to-end verification for current MultiLevelCache (Step B).
#
# Includes:
# 1) Build targets (db_bench, multilevel_cache_runner)
# 2) Routing correctness via verify_mlc_routing.sh
# 3) Guard checks for num_levels > encoded limit
# 4) Allocator auto-adjust smoke run
#
# Usage:
#   tools/verify_mlc_ci.sh [rocksdb_root]
#
# Example:
#   tools/verify_mlc_ci.sh /home/gpu/wzhzhu/rocksdb
#
# Environment overrides:
#   BUILD_JOBS=4
#   NUM=1000000
#   READS=500000
#   THREADS=8
#   CACHE_SIZE=268435456
#   NUM_LEVELS=7
#   VALUE_SIZE=128
#   MIN_PREFIX_HIT_RATE=0.95
#   MIN_LOOKUP_QUERIES=1000
#   RESULT_ROOT=/tmp/mlc_ci_verify

ROOT="${1:-/home/gpu/wzhzhu/rocksdb}"
BUILD_JOBS="${BUILD_JOBS:-4}"
NUM="${NUM:-1000000}"
READS="${READS:-500000}"
THREADS="${THREADS:-8}"
CACHE_SIZE="${CACHE_SIZE:-268435456}"
NUM_LEVELS="${NUM_LEVELS:-7}"
VALUE_SIZE="${VALUE_SIZE:-128}"
MIN_PREFIX_HIT_RATE="${MIN_PREFIX_HIT_RATE:-0.95}"
MIN_LOOKUP_QUERIES="${MIN_LOOKUP_QUERIES:-1000}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/mlc_ci_verify}"

ROUTING_LOG_DIR="${RESULT_ROOT}/routing"
ALLOC_LOG="${RESULT_ROOT}/allocator_smoke.log"
DB_PATH_ROUTING="${RESULT_ROOT}/db_routing"
DB_PATH_ALLOC="${RESULT_ROOT}/db_allocator"

pass=true

die() {
  echo "FAIL: $*" >&2
  exit 1
}

run_step() {
  local title="$1"
  shift
  echo
  echo "==> ${title}"
  "$@"
}

if [[ ! -d "${ROOT}" ]]; then
  die "RocksDB root not found: ${ROOT}"
fi

if [[ ! -x "${ROOT}/tools/verify_mlc_routing.sh" ]]; then
  die "Missing executable: ${ROOT}/tools/verify_mlc_routing.sh"
fi

mkdir -p "${RESULT_ROOT}"

run_step "Build db_bench and runner" \
  cmake --build "${ROOT}/build" -j"${BUILD_JOBS}" --target db_bench multilevel_cache_runner

run_step "Routing correctness" \
  env \
    NUM="${NUM}" \
    READS="${READS}" \
    THREADS="${THREADS}" \
    CACHE_SIZE="${CACHE_SIZE}" \
    NUM_LEVELS="${NUM_LEVELS}" \
    MIN_PREFIX_HIT_RATE="${MIN_PREFIX_HIT_RATE}" \
    MIN_LOOKUP_QUERIES="${MIN_LOOKUP_QUERIES}" \
    RESULT_DIR="${ROUTING_LOG_DIR}" \
    DB_PATH="${DB_PATH_ROUTING}" \
    "${ROOT}/tools/verify_mlc_routing.sh" "${ROOT}/build/db_bench"

echo
echo "==> Guard check: db_bench must reject num_levels > 8"
if "${ROOT}/build/db_bench" \
  --benchmarks=fillrandom \
  --db="${RESULT_ROOT}/db_guard_dbbench" \
  --num=1000 \
  --threads=1 \
  --cache_size=67108864 \
  --use_multi_level_cache=true \
  --num_levels=9 \
  --compression_type=none >/tmp/mlc_ci_guard_dbbench.out 2>&1; then
  echo "FAIL: db_bench unexpectedly accepted --num_levels=9" >&2
  pass=false
else
  if ! rg -n "exceeds encoded route limit" /tmp/mlc_ci_guard_dbbench.out >/dev/null 2>&1; then
    echo "FAIL: db_bench rejected num_levels=9 but missing expected error text" >&2
    pass=false
  else
    echo "PASS: db_bench guard works"
  fi
fi

echo
echo "==> Guard check: runner must reject num_levels > 8"
if "${ROOT}/build/multilevel_cache_runner" "${RESULT_ROOT}/db_guard_runner" 32 9 1000 \
  >/tmp/mlc_ci_guard_runner.out 2>&1; then
  echo "FAIL: runner unexpectedly accepted num_levels=9" >&2
  pass=false
else
  if ! rg -n "exceeds encoded route limit" /tmp/mlc_ci_guard_runner.out >/dev/null 2>&1; then
    echo "FAIL: runner rejected num_levels=9 but missing expected error text" >&2
    pass=false
  else
    echo "PASS: runner guard works"
  fi
fi

echo
echo "==> Allocator auto-adjust smoke"
if ! "${ROOT}/build/db_bench" \
  --benchmarks=fillrandom,readrandom \
  --db="${DB_PATH_ALLOC}" \
  --num=200000 \
  --reads=100000 \
  --threads=4 \
  --value_size="${VALUE_SIZE}" \
  --cache_size="${CACHE_SIZE}" \
  --num_levels="${NUM_LEVELS}" \
  --use_multi_level_cache=true \
  --multi_level_cache_auto_adjust=true \
  --compression_type=none \
  --statistics=1 \
  --stats_interval_seconds=0 \
  >"${ALLOC_LOG}" 2>&1; then
  echo "FAIL: allocator smoke run failed" >&2
  pass=false
else
  if ! rg -n "MultiLevelCache Stats \\(db_bench final\\)" "${ALLOC_LOG}" >/dev/null 2>&1; then
    echo "FAIL: allocator smoke missing final MLC stats output" >&2
    pass=false
  else
    echo "PASS: allocator smoke completed"
  fi
fi

echo
echo "==> Summary"
echo "routing logs: ${ROUTING_LOG_DIR}"
echo "allocator log: ${ALLOC_LOG}"

if [[ "${pass}" == "true" ]]; then
  echo "PASS: verify_mlc_ci completed successfully."
  exit 0
fi

echo "FAIL: verify_mlc_ci detected one or more issues." >&2
exit 2
