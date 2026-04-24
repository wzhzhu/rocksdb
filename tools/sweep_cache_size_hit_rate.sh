#!/usr/bin/env bash
set -euo pipefail

# Sweep cache sizes and compare data block hit rates.
#
# Workflow:
# 1) Prepare DB once per (cache_size, run) using fillrandom.
# 2) Reuse the same DB for multiple read cases:
#    - baseline_lru
#    - mlc_model
#    - mlc_force_l0 (optional)
#
# Usage:
#   tools/sweep_cache_size_hit_rate.sh [db_bench_path]
#
# Environment overrides:
#   RUNS=3
#   NUM=2000000
#   READS=1000000
#   THREADS=16
#   VALUE_SIZE=128
#   NUM_LEVELS=7
#   COMPRESSION_TYPE=none
#   SEED=42
#   READ_RANDOM_EXP_RANGE=0
#   FILL_STABILIZE_SECONDS=60
#   CACHE_SIZE_MB_LIST=32,64,128,256
#   INCLUDE_FORCE_L0=1
#   RESULT_DIR=/tmp/mlc_cache_sweep
#   DB_ROOT=/tmp/mlc_cache_sweep_db

DB_BENCH_BIN="${1:-./build/db_bench}"
RUNS="${RUNS:-3}"
NUM="${NUM:-2000000}"
READS="${READS:-1000000}"
THREADS="${THREADS:-16}"
VALUE_SIZE="${VALUE_SIZE:-128}"
NUM_LEVELS="${NUM_LEVELS:-7}"
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"
SEED="${SEED:-42}"
READ_RANDOM_EXP_RANGE="${READ_RANDOM_EXP_RANGE:-0}"
FILL_STABILIZE_SECONDS="${FILL_STABILIZE_SECONDS:-60}"
CACHE_SIZE_MB_LIST="${CACHE_SIZE_MB_LIST:-32,64,128,256}"
INCLUDE_FORCE_L0="${INCLUDE_FORCE_L0:-1}"
RESULT_DIR="${RESULT_DIR:-/tmp/mlc_cache_sweep}"
DB_ROOT="${DB_ROOT:-/tmp/mlc_cache_sweep_db}"

mkdir -p "${RESULT_DIR}" "${DB_ROOT}"

if [[ ! -x "${DB_BENCH_BIN}" ]]; then
  echo "ERROR: db_bench not executable: ${DB_BENCH_BIN}" >&2
  exit 1
fi

extract_data_hit_rate() {
  local log_file="$1"
  awk '
    /rocksdb\.block\.cache\.data\.hit COUNT/ {h=$NF}
    /rocksdb\.block\.cache\.data\.miss COUNT/ {m=$NF}
    END {
      t=h+m;
      if (t == 0) print "0.0";
      else printf "%.6f\n", h/t;
    }
  ' "${log_file}"
}

prepare_db_once() {
  local cache_mb="$1"
  local run_id="$2"
  local db_path="$3"
  local fill_log="${RESULT_DIR}/dataset_c${cache_mb}_r${run_id}_fill.log"

  rm -rf "${db_path}"
  echo "[prepare] cache=${cache_mb}MB run=${run_id}/${RUNS}" >&2
  "${DB_BENCH_BIN}" \
    --benchmarks=fillrandom \
    --use_existing_db=false \
    --db="${db_path}" \
    --num="${NUM}" \
    --threads="${THREADS}" \
    --value_size="${VALUE_SIZE}" \
    --num_levels="${NUM_LEVELS}" \
    --compression_type="${COMPRESSION_TYPE}" \
    --statistics=1 \
    --stats_interval_seconds=0 \
    --seed="${SEED}" \
    > "${fill_log}" 2>&1
}

run_read_case() {
  local mode="$1"
  local cache_mb="$2"
  local run_id="$3"
  local db_path="$4"
  local cache_bytes=$((cache_mb * 1024 * 1024))
  local read_log="${RESULT_DIR}/${mode}_c${cache_mb}_r${run_id}_read.log"

  local -a args=(
    "--benchmarks=readrandom"
    "--use_existing_db=true"
    "--db=${db_path}"
    "--num=${NUM}"
    "--reads=${READS}"
    "--threads=${THREADS}"
    "--value_size=${VALUE_SIZE}"
    "--cache_size=${cache_bytes}"
    "--num_levels=${NUM_LEVELS}"
    "--compression_type=${COMPRESSION_TYPE}"
    "--statistics=1"
    "--stats_interval_seconds=0"
    "--seed=${SEED}"
    "--read_random_exp_range=${READ_RANDOM_EXP_RANGE}"
  )

  case "${mode}" in
    baseline_lru)
      args+=("--use_multi_level_cache=false" "--cache_type=lru_cache")
      ;;
    mlc_model)
      args+=(
        "--use_multi_level_cache=true"
        "--multi_level_cache_auto_adjust=true"
        "--multi_level_cache_allocator_mode=model"
      )
      ;;
    mlc_force_l0)
      args+=(
        "--use_multi_level_cache=true"
        "--multi_level_cache_auto_adjust=true"
        "--multi_level_cache_allocator_mode=baseline_emulation"
        "--multi_level_cache_force_route_all_to_l0=true"
        "--multi_level_cache_adjust_min_active_level_capacity_bytes=0"
        "--multi_level_cache_adjust_smoothing_ratio=1.0"
        "--multi_level_cache_adjust_min_change_bytes=0"
      )
      ;;
    *)
      echo "ERROR: unknown mode ${mode}" >&2
      return 1
      ;;
  esac

  echo "[${mode}] cache=${cache_mb}MB run=${run_id}/${RUNS}" >&2
  "${DB_BENCH_BIN}" "${args[@]}" > "${read_log}" 2>&1
  extract_data_hit_rate "${read_log}"
}

csv_file="${RESULT_DIR}/summary.csv"
echo "cache_mb,mode,run,data_hit_rate" > "${csv_file}"

IFS=',' read -r -a cache_list <<< "${CACHE_SIZE_MB_LIST}"
for cache_mb in "${cache_list[@]}"; do
  for ((run_id=1; run_id<=RUNS; ++run_id)); do
    db_path="${DB_ROOT}/dataset_c${cache_mb}_r${run_id}"
    prepare_db_once "${cache_mb}" "${run_id}" "${db_path}"
    if [[ "${FILL_STABILIZE_SECONDS}" -gt 0 ]]; then
      echo "[stabilize] sleep ${FILL_STABILIZE_SECONDS}s after fill (cache=${cache_mb}MB run=${run_id}/${RUNS})" >&2
      sleep "${FILL_STABILIZE_SECONDS}"
    fi

    rate="$(run_read_case baseline_lru "${cache_mb}" "${run_id}" "${db_path}")"
    echo "${cache_mb},baseline_lru,${run_id},${rate}" >> "${csv_file}"

    rate="$(run_read_case mlc_model "${cache_mb}" "${run_id}" "${db_path}")"
    echo "${cache_mb},mlc_model,${run_id},${rate}" >> "${csv_file}"

    if [[ "${INCLUDE_FORCE_L0}" == "1" ]]; then
      rate="$(run_read_case mlc_force_l0 "${cache_mb}" "${run_id}" "${db_path}")"
      echo "${cache_mb},mlc_force_l0,${run_id},${rate}" >> "${csv_file}"
    fi
  done
done

echo
echo "===== Cache Sweep Mean (data block hit rate) ====="
awk -F',' '
  NR == 1 { next }
  {
    key = $1 "," $2;
    sum[key] += $4;
    cnt[key] += 1;
  }
  END {
    print "cache_mb,mode,mean_hit_rate,runs";
    for (k in sum) {
      split(k, a, ",");
      printf "%s,%s,%.6f,%d\n", a[1], a[2], sum[k] / cnt[k], cnt[k];
    }
  }
' "${csv_file}" | sort -t',' -k1,1n -k2,2

echo
echo "Detailed CSV: ${csv_file}"
echo "Logs dir: ${RESULT_DIR}"
