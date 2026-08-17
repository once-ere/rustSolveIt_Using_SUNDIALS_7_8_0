#!/usr/bin/env bash
# pow_differential.sh [domain|random|all]
#
# Measure the deterministic `pow` in crates/sundials_core/src/sundials_math.rs
# against the NATIVE host `pow`, on the host it is meant to reproduce
# (Linux, glibc, Intel/AMD x86-64).
#
# Builds tools/pow_oracle.c with the system compiler, generates the
# reference bit-stream for one or both corpora, then runs the two
# differential unit tests with the oracle paths in the environment.
#
#   tools/pow_differential.sh domain   # 5.9M inputs, SUNDIALS operating domain (the gate)
#   tools/pow_differential.sh random   # 20M unrestricted finite pairs (bound, not a gate)
#   tools/pow_differential.sh all
#
# Everything is written to logs/; read logs/pow_differential.log afterwards.
set -u
export PATH="$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.."
WS_ROOT="$(pwd)"
LOGS="$WS_ROOT/logs"
OUT="${POW_ORACLE_DIR:-${TMPDIR:-/tmp}}"
mkdir -p "$LOGS"

WHAT="${1:-all}"
CC_BIN="${CC:-cc}"
LOG="$LOGS/pow_differential.log"

{
  echo "== host =="
  uname -srm
  (ldd --version 2>/dev/null || true) | head -1
  "$CC_BIN" --version | head -1
  echo -n "cpu fma: "
  grep -qm1 ' fma ' /proc/cpuinfo && echo yes || echo no
  echo
} > "$LOG"

echo "building oracle ..." | tee -a "$LOG"
"$CC_BIN" -O2 -o "$OUT/pow_oracle" "$WS_ROOT/tools/pow_oracle.c" -lm 2>&1 | tee -a "$LOG" || exit 1

env_args=()
if [ "$WHAT" = domain ] || [ "$WHAT" = all ]; then
  echo "generating domain corpus (5,900,000) ..." | tee -a "$LOG"
  "$OUT/pow_oracle" domain > "$OUT/pow_domain.bin" || exit 1
  env_args+=("SUNDIALS_POW_ORACLE_DOMAIN=$OUT/pow_domain.bin")
fi
if [ "$WHAT" = random ] || [ "$WHAT" = all ]; then
  echo "generating random corpus (20,000,000) ..." | tee -a "$LOG"
  "$OUT/pow_oracle" random > "$OUT/pow_random.bin" || exit 1
  env_args+=("SUNDIALS_POW_ORACLE_RANDOM=$OUT/pow_random.bin")
fi

echo "running differential ..." | tee -a "$LOG"
env "${env_args[@]}" cargo test --release -p sundials_core --lib \
    pow_glibc_vs_native_oracle -- --nocapture --test-threads=2 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "exit: $status" | tee -a "$LOG"
exit "$status"
