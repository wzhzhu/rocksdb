#!/usr/bin/env bash
set -euo pipefail

# A/B compare script:
# A: MultiLevelCache (--use_multi_level_cache=true)
# B: Baseline RocksDB block cache (--use_multi_level_cache=false)
#
# Usage:
#   tools/compare_cache_hit_rate.sh [db_bench_path]
#
# Environment overrides:
#   RUNS=3
#   NUM=5000000
#   READS=5000000
#   THREADS=16
#   CACHE_SIZE=1073741824
#   NUM_LEVELS=7
#   BENCHMARKS=fillrandom,readrandom
#   COMPRESSION_TYPE=none
#   RESULT_DIR=/tmp/mlc_ab_results
#   DB_ROOT=/tmp/mlc_ab_db
#   SEED=42

DB_BENCH_BIN="${1:-./db_bench}"
RUNS="${RUNS:-3}"
NUM="${NUM:-5000000}"
READS="${READS:-5000000}"
THREADS="${THREADS:-16}"
CACHE_SIZE="${CACHE_SIZE:-1073741824}"
NUM_LEVELS="${NUM_LEVELS:-7}"
BENCHMARKS="${BENCHMARKS:-fillrandom,readrandom}"
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"
RESULT_DIR="${RESULT_DIR:-/tmp/mlc_ab_results}"
DB_ROOT="${DB_ROOT:-/tmp/mlc_ab_db}"
SEED="${SEED:-42}"

mkdir -p "${RESULT_DIR}"
mkdir -p "${DB_ROOT}"

if [[ ! -x "${DB_BENCH_BIN}" ]]; then
  echo "ERROR: db_bench not executable: ${DB_BENCH_BIN}" >&2
  echo "Tip: pass absolute path, e.g. /path/to/build/db_bench" >&2
  exit 1
fi

extract_mlc_hit_rate() {
  local log_file="$1"
  # Prefer the explicit MultiLevelCache total hit rate line.
  local line
  line="$(awk '/total_hit_rate=/{val=$0} END{print val}' "${log_file}")"
  if [[ -z "${line}" ]]; then
    return 1
  fi
  echo "${line}" | sed -E 's/.*total_hit_rate=([0-9.]+).*/\1/'
}

extract_baseline_hit_rate() {
  local log_file="$1"

  # Match common db_bench statistics styles:
  # - "rocksdb.block.cache.hit COUNT : 12345"
  # - "BLOCK_CACHE_HIT COUNT : 12345"
  # - "BLOCK_CACHE_HIT : 12345"
  local hit miss
  hit="$(awk '
    /rocksdb\.block\.cache\.hit|BLOCK_CACHE_HIT/ {
      n=split($0, a, /[^0-9]+/);
      for (i=n; i>=1; --i) if (a[i] != "") { v=a[i]; break; }
    }
    END { if (v != "") print v; }
  ' "${log_file}")"
  miss="$(awk '
    /rocksdb\.block\.cache\.miss|BLOCK_CACHE_MISS/ {
      n=split($0, a, /[^0-9]+/);
      for (i=n; i>=1; --i) if (a[i] != "") { v=a[i]; break; }
    }
    END { if (v != "") print v; }
  ' "${log_file}")"

  if [[ -z "${hit}" || -z "${miss}" ]]; then
    return 1
  fi
  awk -v h="${hit}" -v m="${miss}" 'BEGIN {
    t = h + m;
    if (t <= 0) {
      print "0.0";
    } else {
      printf "%.6f\n", h / t;
    }
  }'
}

summarize_rates() {
  local label="$1"
  shift
  local rates=("$@")

  if [[ "${#rates[@]}" -eq 0 ]]; then
    echo "${label}: no samples"
    return
  fi

  local joined
  joined="$(printf "%s\n" "${rates[@]}")"
  awk -v label="${label}" '
    {
      x[NR]=$1;
      sum+=$1;
    }
    END {
      n=NR;
      mean=sum/n;
      var=0;
      for (i=1; i<=n; ++i) {
        d=x[i]-mean;
        var+=d*d;
      }
      std=(n>1)?sqrt(var/(n-1)):0;
      printf "%s: runs=%d, mean=%.6f, std=%.6f\n", label, n, mean, std;
    }
  ' <<< "${joined}"
}

run_one() {
  local mode="$1"       # mlc or baseline
  local run_id="$2"
  local use_mlc="$3"    # true/false

  local db_path="${DB_ROOT}/${mode}_run_${run_id}"
  local log_file="${RESULT_DIR}/${mode}_run_${run_id}.log"
  rm -rf "${db_path}"

  echo "[${mode}] run ${run_id}/${RUNS} ..." >&2
  "${DB_BENCH_BIN}" \
    --db="${db_path}" \
    --benchmarks="${BENCHMARKS}" \
    --num="${NUM}" \
    --reads="${READS}" \
    --threads="${THREADS}" \
    --cache_size="${CACHE_SIZE}" \
    --num_levels="${NUM_LEVELS}" \
    --use_multi_level_cache="${use_mlc}" \
    --compression_type="${COMPRESSION_TYPE}" \
    --statistics=1 \
    --stats_interval_seconds=0 \
    --seed="${SEED}" \
    > "${log_file}" 2>&1

  local rate=""
  if [[ "${mode}" == "mlc" ]]; then
    rate="$(extract_mlc_hit_rate "${log_file}")" || {
      echo "ERROR: failed to parse MultiLevelCache hit rate from ${log_file}" >&2
      return 1
    }
  else
    rate="$(extract_baseline_hit_rate "${log_file}")" || {
      echo "ERROR: failed to parse baseline hit/miss from ${log_file}" >&2
      return 1
    }
  fi

  echo "[${mode}] run ${run_id} hit_rate=${rate}" >&2
  printf "%s\n" "${rate}"
}

mlc_rates=()
base_rates=()

for ((i=1; i<=RUNS; ++i)); do
  rate="$(run_one "mlc" "${i}" "true")"
  mlc_rates+=("${rate}")
done

for ((i=1; i<=RUNS; ++i)); do
  rate="$(run_one "baseline" "${i}" "false")"
  base_rates+=("${rate}")
done

echo
echo "===== A/B Summary ====="
summarize_rates "MultiLevelCache" "${mlc_rates[@]}"
summarize_rates "Baseline" "${base_rates[@]}"

mlc_mean="$(printf "%s\n" "${mlc_rates[@]}" | awk '{s+=$1} END{printf "%.6f", s/NR}')"
base_mean="$(printf "%s\n" "${base_rates[@]}" | awk '{s+=$1} END{printf "%.6f", s/NR}')"
delta="$(awk -v a="${mlc_mean}" -v b="${base_mean}" 'BEGIN{printf "%.6f", a-b}')"
delta_pct="$(awk -v a="${mlc_mean}" -v b="${base_mean}" 'BEGIN{
  if (b == 0) printf "inf"; else printf "%.2f", (a-b)/b*100;
}')"
echo "Delta (MLC - Baseline): ${delta} (${delta_pct}%)"
echo "Logs: ${RESULT_DIR}"

