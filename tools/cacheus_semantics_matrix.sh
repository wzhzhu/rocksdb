#!/usr/bin/env bash
set -euo pipefail

# Compare Cacheus capacity semantics on a small matrix.
#
# Usage:
#   tools/cacheus_semantics_matrix.sh [db_bench_path]
#
# Env overrides:
#   RESULT_DIR=/tmp/cacheus_semantics_matrix
#   DB_ROOT=/tmp/cacheus_semantics_db
#   RUNS=3
#   CACHE_SIZE_MB_LIST=64,128
#   VALUE_SIZE_LIST=64,1024
#   READ_RANDOM_EXP_RANGE_LIST=0,1.2
#   NUM=1000000
#   READS=1000000
#   THREADS=16
#   SEED=42
#   CACHEUS_INITIAL_WEIGHT=0.5
#   CACHEUS_LEARNING_RATE=0.45
#   CACHEUS_HISTORY_SIZE=65536
#   CACHEUS_PERIOD_LEN=131072
#   CACHEUS_RNG_SEED=123

DB_BENCH_BIN="${1:-./build/db_bench}"
RESULT_DIR="${RESULT_DIR:-/tmp/cacheus_semantics_matrix}"
DB_ROOT="${DB_ROOT:-/tmp/cacheus_semantics_db}"
RUNS="${RUNS:-3}"
CACHE_SIZE_MB_LIST="${CACHE_SIZE_MB_LIST:-64,128}"
VALUE_SIZE_LIST="${VALUE_SIZE_LIST:-64,1024}"
READ_RANDOM_EXP_RANGE_LIST="${READ_RANDOM_EXP_RANGE_LIST:-0,1.2}"
NUM="${NUM:-1000000}"
READS="${READS:-1000000}"
THREADS="${THREADS:-16}"
SEED="${SEED:-42}"

CACHEUS_INITIAL_WEIGHT="${CACHEUS_INITIAL_WEIGHT:-0.5}"
CACHEUS_LEARNING_RATE="${CACHEUS_LEARNING_RATE:-0.45}"
CACHEUS_HISTORY_SIZE="${CACHEUS_HISTORY_SIZE:-65536}"
CACHEUS_PERIOD_LEN="${CACHEUS_PERIOD_LEN:-131072}"
CACHEUS_RNG_SEED="${CACHEUS_RNG_SEED:-123}"

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

run_case() {
  local semantics="$1"      # bytes | entries_equivalent
  local cache_mb="$2"
  local value_size="$3"
  local exp_range="$4"
  local run_id="$5"

  local cache_bytes=$((cache_mb * 1024 * 1024))
  local db_path="${DB_ROOT}/db_c${cache_mb}_v${value_size}_e${exp_range}_r${run_id}"
  local fill_log="${RESULT_DIR}/fill_c${cache_mb}_v${value_size}_e${exp_range}_r${run_id}.log"
  local read_log="${RESULT_DIR}/read_${semantics}_c${cache_mb}_v${value_size}_e${exp_range}_r${run_id}.log"
  local semantic_flag=false
  if [[ "${semantics}" == "entries_equivalent" ]]; then
    semantic_flag=true
  fi
  local cache_uri="cacheus://capacity=${cache_bytes};num_shard_bits=6;initial_weight=${CACHEUS_INITIAL_WEIGHT};learning_rate=${CACHEUS_LEARNING_RATE};history_size=${CACHEUS_HISTORY_SIZE};period_len=${CACHEUS_PERIOD_LEN};rng_seed=${CACHEUS_RNG_SEED};entry_charge_equivalent=${semantic_flag}"

  rm -rf "${db_path}"
  "${DB_BENCH_BIN}" \
    --benchmarks=fillrandom \
    --db="${db_path}" \
    --use_existing_db=false \
    --num="${NUM}" \
    --threads="${THREADS}" \
    --value_size="${value_size}" \
    --statistics=1 \
    --seed="${SEED}" \
    > "${fill_log}" 2>&1

  "${DB_BENCH_BIN}" \
    --benchmarks=readrandom \
    --db="${db_path}" \
    --use_existing_db=true \
    --num="${NUM}" \
    --reads="${READS}" \
    --threads="${THREADS}" \
    --value_size="${value_size}" \
    --statistics=1 \
    --stats_interval=100000 \
    --stats_per_interval=1 \
    --cache_printable_stats_per_interval=true \
    --cache_uri="${cache_uri}" \
    --read_random_exp_range="${exp_range}" \
    --seed="${SEED}" \
    > "${read_log}" 2>&1

  local hit_rate
  hit_rate="$(extract_data_hit_rate "${read_log}")"
  echo "${cache_mb},${value_size},${exp_range},${semantics},${run_id},${hit_rate}"
}

csv_file="${RESULT_DIR}/summary.csv"
echo "cache_mb,value_size,read_random_exp_range,semantics,run,data_hit_rate" > "${csv_file}"

IFS=',' read -r -a cache_list <<< "${CACHE_SIZE_MB_LIST}"
IFS=',' read -r -a value_list <<< "${VALUE_SIZE_LIST}"
IFS=',' read -r -a exp_list <<< "${READ_RANDOM_EXP_RANGE_LIST}"

for cache_mb in "${cache_list[@]}"; do
  for value_size in "${value_list[@]}"; do
    for exp_range in "${exp_list[@]}"; do
      for ((run_id=1; run_id<=RUNS; ++run_id)); do
        run_case "bytes" "${cache_mb}" "${value_size}" "${exp_range}" "${run_id}" >> "${csv_file}"
        run_case "entries_equivalent" "${cache_mb}" "${value_size}" "${exp_range}" "${run_id}" >> "${csv_file}"
      done
    done
  done
done

echo "===== Mean Data Hit Rate ====="
awk -F',' '
  NR==1 {next}
  {
    key=$1","$2","$3","$4;
    sum[key]+=$6;
    cnt[key]+=1;
  }
  END {
    print "cache_mb,value_size,read_random_exp_range,semantics,mean_hit_rate,runs";
    for (k in sum) {
      split(k, a, ",");
      printf "%s,%s,%s,%s,%.6f,%d\n", a[1], a[2], a[3], a[4], sum[k]/cnt[k], cnt[k];
    }
  }
' "${csv_file}" | sort -t',' -k1,1n -k2,2n -k3,3 -k4,4

echo
echo "Detailed CSV: ${csv_file}"
echo "Logs dir: ${RESULT_DIR}"
