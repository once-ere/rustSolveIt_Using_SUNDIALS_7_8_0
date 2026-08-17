#!/usr/bin/env bash
# libm_differential.sh — measure the pure-Rust elementary functions in
# `crates/sundials_core/src/sundials_libm.rs` against the *host* C library.
#
#   tools/libm_differential.sh [count]        # default 1000000 per corpus
#
# What it does, in order:
#   1. builds tools/libm_oracle.c with the host cc (libquadmath if present),
#   2. runs it once per (function, corpus) pair, writing logs/oracle/*.bin,
#   3. runs the Rust differential tests with SUNDIALS_LIBM_ORACLE_DIR set.
#
# Output lands in logs/libm_differential.log. Every number in LIBM.md comes
# from that file; nothing in this pipeline is hand-entered.
set -u

cd "$(dirname "$0")/.."
ROOT=$(pwd)
COUNT=${1:-1000000}
ORACLE_DIR=$ROOT/logs/oracle
LOG=$ROOT/logs/libm_differential.log

mkdir -p "$ORACLE_DIR" "$ROOT/logs"
: >"$LOG"

say() { printf '%s\n' "$*" | tee -a "$LOG"; }

say "== libm differential =="
say "host:      $(uname -srm)"
say "libc:      $(ldd --version 2>/dev/null | head -1)"
say "compiler:  $(${CC:-cc} --version 2>/dev/null | head -1)"
say "rustc:     $(rustc --version)"
say "count:     $COUNT per (function, corpus)"
say ""

CC=${CC:-cc}
QUAD_FLAGS="-lquadmath"
if $CC -O2 -o "$ROOT/logs/libm_oracle" "$ROOT/tools/libm_oracle.c" -lm $QUAD_FLAGS 2>"$ROOT/logs/libm_oracle_build.log"; then
  say "oracle: built with libquadmath (accuracy in ulp will be reported)"
elif $CC -O2 -DNO_QUADMATH -o "$ROOT/logs/libm_oracle" "$ROOT/tools/libm_oracle.c" -lm 2>>"$ROOT/logs/libm_oracle_build.log"; then
  say "oracle: built WITHOUT libquadmath -- agreement only, no ulp figures"
  say "        (install the gcc quadmath runtime to get accuracy numbers)"
else
  say "ERROR: could not build tools/libm_oracle.c; see logs/libm_oracle_build.log"
  exit 1
fi
say ""

FUNCS="exp log expm1 log1p sin cos atan asin acos sinh cosh acosh"
for f in $FUNCS; do
  for corpus in domain wide; do
    "$ROOT/logs/libm_oracle" "$f" "$corpus" "$COUNT" "$ORACLE_DIR/${f}_${corpus}.bin" ||
      { say "ERROR: oracle failed for $f/$corpus"; exit 1; }
  done
done
say "oracle corpora written to logs/oracle ($(ls "$ORACLE_DIR" | wc -l) files)"
say ""

SUNDIALS_LIBM_ORACLE_DIR=$ORACLE_DIR \
  cargo test --release -p sundials_core --lib libm -- --nocapture 2>&1 | tee -a "$LOG"

say ""
say "== end =="
