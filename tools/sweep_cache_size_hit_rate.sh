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
#   NUM=2000000  # compatibility fallback
#   KEY_SPACE_NUM=2000000
#   FILL_WRITES_PER_THREAD=2000000
#   READS=1000000
#   THREADS=16
#   WRITE_THREADS=16
#   READ_THREADS=16
#   VALUE_SIZE=128
#   NUM_LEVELS=7
#   COMPRESSION_TYPE=none
#   SEED=42
#   READ_RANDOM_EXP_RANGE=0
#   FILL_STABILIZE_SECONDS=60
#   WAIT_FOR_COMPACTION_AFTER_FILL=1
#   DISABLE_AUTO_COMPACTIONS_ON_READ=1
#   MLC_MODEL_SMOOTHING_RATIO=1.0
#   MLC_MODEL_MIN_CHANGE_BYTES=0
#   MLC_MODEL_MIN_ACTIVE_LEVEL_CAPACITY_BYTES=0
#   MLC_MODEL_ALPHA_ESTIMATOR=shadow_cache   # single estimator (compat mode)
#   MLC_MODEL_ALPHA_ESTIMATOR_LIST=constant_one,robust_hit_rate,shadow_cache
#   MLC_MODEL_ALPHA_SHADOW_SCALE=1.5
#   MLC_MODEL_ALPHA_SHADOW_SAMPLE_RATE_LOG2=8
#   MLC_MODEL_ALPHA_SHADOW_WINDOW_ROUNDS=5
#   MLC_MODEL_ALPHA_SHADOW_MIN_CAPACITY_BYTES=33554432
#   MLC_MODEL_BIAS_DEBUG=0
#   MLC_MODEL_BIAS_DEBUG_EVERY_ROUNDS=10
#   MLC_MODEL_SHARED_POOL_RATIO=0.0
#   MLC_MODEL_SHARED_POOL_ADMISSION_THRESHOLD=2
#   MLC_MODEL_SHARED_POOL_DECAY_INTERVAL_OPS=2048
#   CACHE_SIZE_MB_LIST=32,64,128,256
#   INCLUDE_FORCE_L0=1
#   RESULT_DIR=/tmp/mlc_cache_sweep
#   DB_ROOT=/tmp/mlc_cache_sweep_db

DB_BENCH_BIN="${1:-./build/db_bench}"
RUNS="${RUNS:-3}"
NUM="${NUM:-2000000}"
KEY_SPACE_NUM="${KEY_SPACE_NUM:-${NUM}}"
FILL_WRITES_PER_THREAD="${FILL_WRITES_PER_THREAD:-${NUM}}"
READS="${READS:-1000000}"
THREADS="${THREADS:-16}"
WRITE_THREADS="${WRITE_THREADS:-${THREADS}}"
READ_THREADS="${READ_THREADS:-${THREADS}}"
VALUE_SIZE="${VALUE_SIZE:-128}"
NUM_LEVELS="${NUM_LEVELS:-7}"
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"
SEED="${SEED:-42}"
READ_RANDOM_EXP_RANGE="${READ_RANDOM_EXP_RANGE:-0}"
FILL_STABILIZE_SECONDS="${FILL_STABILIZE_SECONDS:-60}"
WAIT_FOR_COMPACTION_AFTER_FILL="${WAIT_FOR_COMPACTION_AFTER_FILL:-1}"
DISABLE_AUTO_COMPACTIONS_ON_READ="${DISABLE_AUTO_COMPACTIONS_ON_READ:-1}"
MLC_MODEL_SMOOTHING_RATIO="${MLC_MODEL_SMOOTHING_RATIO:-1.0}"
MLC_MODEL_MIN_CHANGE_BYTES="${MLC_MODEL_MIN_CHANGE_BYTES:-0}"
MLC_MODEL_MIN_ACTIVE_LEVEL_CAPACITY_BYTES="${MLC_MODEL_MIN_ACTIVE_LEVEL_CAPACITY_BYTES:-0}"
MLC_MODEL_ALPHA_ESTIMATOR="${MLC_MODEL_ALPHA_ESTIMATOR:-shadow_cache}"
MLC_MODEL_ALPHA_ESTIMATOR_LIST="${MLC_MODEL_ALPHA_ESTIMATOR_LIST:-${MLC_MODEL_ALPHA_ESTIMATOR}}"
MLC_MODEL_ALPHA_SHADOW_SCALE="${MLC_MODEL_ALPHA_SHADOW_SCALE:-1.5}"
MLC_MODEL_ALPHA_SHADOW_SAMPLE_RATE_LOG2="${MLC_MODEL_ALPHA_SHADOW_SAMPLE_RATE_LOG2:-8}"
MLC_MODEL_ALPHA_SHADOW_WINDOW_ROUNDS="${MLC_MODEL_ALPHA_SHADOW_WINDOW_ROUNDS:-5}"
MLC_MODEL_ALPHA_SHADOW_MIN_CAPACITY_BYTES="${MLC_MODEL_ALPHA_SHADOW_MIN_CAPACITY_BYTES:-33554432}"
MLC_MODEL_BIAS_DEBUG="${MLC_MODEL_BIAS_DEBUG:-0}"
MLC_MODEL_BIAS_DEBUG_EVERY_ROUNDS="${MLC_MODEL_BIAS_DEBUG_EVERY_ROUNDS:-10}"
MLC_MODEL_SHARED_POOL_RATIO="${MLC_MODEL_SHARED_POOL_RATIO:-0.0}"
MLC_MODEL_SHARED_POOL_ADMISSION_THRESHOLD="${MLC_MODEL_SHARED_POOL_ADMISSION_THRESHOLD:-2}"
MLC_MODEL_SHARED_POOL_DECAY_INTERVAL_OPS="${MLC_MODEL_SHARED_POOL_DECAY_INTERVAL_OPS:-2048}"
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
  local hit_line
  local miss_line
  hit_line="$(rg "rocksdb\\.block\\.cache\\.data\\.hit COUNT" "${log_file}" -n -m 1 || true)"
  miss_line="$(rg "rocksdb\\.block\\.cache\\.data\\.miss COUNT" "${log_file}" -n -m 1 || true)"
  if [[ -z "${hit_line}" || -z "${miss_line}" ]]; then
    echo "ERROR: failed to parse data block hit/miss counters from ${log_file}" >&2
    echo "------ log tail ------" >&2
    tail -n 40 "${log_file}" >&2
    return 1
  fi
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
    --num="${KEY_SPACE_NUM}" \
    --writes="${FILL_WRITES_PER_THREAD}" \
    --threads="${WRITE_THREADS}" \
    --value_size="${VALUE_SIZE}" \
    --num_levels="${NUM_LEVELS}" \
    --compression_type="${COMPRESSION_TYPE}" \
    --statistics=1 \
    --stats_interval_seconds=0 \
    --seed="${SEED}" \
    > "${fill_log}" 2>&1
}

wait_for_compaction_after_fill() {
  local cache_mb="$1"
  local run_id="$2"
  local db_path="$3"
  local wait_log="${RESULT_DIR}/dataset_c${cache_mb}_r${run_id}_waitforcompaction.log"

  if [[ "${WAIT_FOR_COMPACTION_AFTER_FILL}" != "1" ]]; then
    return
  fi
  echo "[waitforcompaction] cache=${cache_mb}MB run=${run_id}/${RUNS}" >&2
  "${DB_BENCH_BIN}" \
    --benchmarks=waitforcompaction \
    --use_existing_db=true \
    --db="${db_path}" \
    --num="${KEY_SPACE_NUM}" \
    --threads=1 \
    --num_levels="${NUM_LEVELS}" \
    --compression_type="${COMPRESSION_TYPE}" \
    --statistics=1 \
    --stats_interval_seconds=0 \
    --seed="${SEED}" \
    > "${wait_log}" 2>&1
}

run_read_case() {
  local mode="$1"
  local cache_mb="$2"
  local run_id="$3"
  local db_path="$4"
  local cache_bytes=$((cache_mb * 1024 * 1024))
  local mode_kind="${mode}"
  local mode_label="${mode}"
  local alpha_estimator="${MLC_MODEL_ALPHA_ESTIMATOR}"
  if [[ "${mode}" == mlc_model:* ]]; then
    mode_kind="mlc_model"
    alpha_estimator="${mode#mlc_model:}"
    mode_label="mlc_model_${alpha_estimator}"
  fi
  local read_log="${RESULT_DIR}/${mode_label}_c${cache_mb}_r${run_id}_read.log"

  local -a args=(
    "--benchmarks=readrandom"
    "--use_existing_db=true"
    "--db=${db_path}"
    "--num=${KEY_SPACE_NUM}"
    "--reads=${READS}"
    "--threads=${READ_THREADS}"
    "--value_size=${VALUE_SIZE}"
    "--cache_size=${cache_bytes}"
    "--num_levels=${NUM_LEVELS}"
    "--compression_type=${COMPRESSION_TYPE}"
    "--statistics=1"
    "--stats_interval_seconds=0"
    "--seed=${SEED}"
    "--read_random_exp_range=${READ_RANDOM_EXP_RANGE}"
  )
  if [[ "${DISABLE_AUTO_COMPACTIONS_ON_READ}" == "1" ]]; then
    args+=("--disable_auto_compactions=true")
  fi

  case "${mode_kind}" in
    baseline_lru)
      args+=("--use_multi_level_cache=false" "--cache_type=lru_cache")
      ;;
    mlc_model)
      args+=(
        "--use_multi_level_cache=true"
        "--multi_level_cache_auto_adjust=true"
        "--multi_level_cache_allocator_mode=model"
        "--multi_level_cache_adjust_smoothing_ratio=${MLC_MODEL_SMOOTHING_RATIO}"
        "--multi_level_cache_adjust_min_change_bytes=${MLC_MODEL_MIN_CHANGE_BYTES}"
        "--multi_level_cache_adjust_min_active_level_capacity_bytes=${MLC_MODEL_MIN_ACTIVE_LEVEL_CAPACITY_BYTES}"
        "--multi_level_cache_alpha_estimator=${alpha_estimator}"
        "--multi_level_cache_alpha_shadow_scale=${MLC_MODEL_ALPHA_SHADOW_SCALE}"
        "--multi_level_cache_alpha_shadow_sample_rate_log2=${MLC_MODEL_ALPHA_SHADOW_SAMPLE_RATE_LOG2}"
        "--multi_level_cache_alpha_shadow_window_rounds=${MLC_MODEL_ALPHA_SHADOW_WINDOW_ROUNDS}"
        "--multi_level_cache_alpha_shadow_min_capacity_bytes=${MLC_MODEL_ALPHA_SHADOW_MIN_CAPACITY_BYTES}"
        "--multi_level_cache_model_bias_debug=${MLC_MODEL_BIAS_DEBUG}"
        "--multi_level_cache_model_bias_debug_every_rounds=${MLC_MODEL_BIAS_DEBUG_EVERY_ROUNDS}"
        "--multi_level_cache_shared_pool_ratio=${MLC_MODEL_SHARED_POOL_RATIO}"
        "--multi_level_cache_shared_pool_admission_threshold=${MLC_MODEL_SHARED_POOL_ADMISSION_THRESHOLD}"
        "--multi_level_cache_shared_pool_decay_interval_ops=${MLC_MODEL_SHARED_POOL_DECAY_INTERVAL_OPS}"
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
        "--multi_level_cache_shared_pool_ratio=0"
      )
      ;;
    *)
      echo "ERROR: unknown mode ${mode}" >&2
      return 1
      ;;
  esac

  echo "[${mode_label}] cache=${cache_mb}MB run=${run_id}/${RUNS}" >&2
  "${DB_BENCH_BIN}" "${args[@]}" > "${read_log}" 2>&1
  if rg -q "^ERROR: unknown command line flag" "${read_log}"; then
    echo "ERROR: db_bench rejected flags in ${read_log}" >&2
    tail -n 20 "${read_log}" >&2
    return 1
  fi
  local rate
  rate="$(extract_data_hit_rate "${read_log}")"
  printf "%s,%s\n" "${mode_label}" "${rate}"
}

csv_file="${RESULT_DIR}/summary.csv"
echo "cache_mb,mode,run,data_hit_rate" > "${csv_file}"

read_modes=(baseline_lru)
if [[ "${INCLUDE_FORCE_L0}" == "1" ]]; then
  read_modes+=(mlc_force_l0)
fi
IFS=',' read -r -a model_alpha_list <<< "${MLC_MODEL_ALPHA_ESTIMATOR_LIST}"
for estimator in "${model_alpha_list[@]}"; do
  if [[ -n "${estimator}" ]]; then
    read_modes+=("mlc_model:${estimator}")
  fi
done

IFS=',' read -r -a cache_list <<< "${CACHE_SIZE_MB_LIST}"
for cache_mb in "${cache_list[@]}"; do
  for ((run_id=1; run_id<=RUNS; ++run_id)); do
    db_path="${DB_ROOT}/dataset_c${cache_mb}_r${run_id}"
    prepare_db_once "${cache_mb}" "${run_id}" "${db_path}"
    wait_for_compaction_after_fill "${cache_mb}" "${run_id}" "${db_path}"
    if [[ "${FILL_STABILIZE_SECONDS}" -gt 0 ]]; then
      echo "[stabilize] sleep ${FILL_STABILIZE_SECONDS}s after fill (cache=${cache_mb}MB run=${run_id}/${RUNS})" >&2
      sleep "${FILL_STABILIZE_SECONDS}"
    fi

    for mode in "${read_modes[@]}"; do
      result="$(run_read_case "${mode}" "${cache_mb}" "${run_id}" "${db_path}")"
      mode_label="${result%%,*}"
      rate="${result#*,}"
      echo "${cache_mb},${mode_label},${run_id},${rate}" >> "${csv_file}"
    done
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
