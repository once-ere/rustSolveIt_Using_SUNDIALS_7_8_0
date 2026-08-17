#!/usr/bin/env bash
# compare_lapack_substituted.sh — close the last two rows of
# tools/compare_pristine_c.sh.
#
# cvRoberts_dnsL and cvsRoberts_dnsL are the only in-scope examples whose
# C original uses a backend this port deliberately does not have: the
# LAPACK dense linear solver. The port builds them on the NATIVE dense
# solver instead (a documented scope substitution, prompt §1), so they
# cannot be compared against a stock pristine C build — that build does
# not contain them at all when LAPACK is off.
#
# The honest comparison is against C making the SAME substitution. This
# script copies the two upstream sources into a scratch directory, swaps
# exactly the two LAPACK tokens for their native equivalents — the same
# two the Rust port swapped, and nothing else — compiles them against the
# pristine C library from tools/pristine_c_build.sh, and diffs the result
# against the Rust example.
#
#   RS_vs_Csub == same  =>  the port is faithful, and the divergence from
#                           the shipped .out is entirely attributable to
#                           the LAPACK -> native substitution.
#
# The upstream tree is never written to; the copies live under logs/.
set -u
cd "$(dirname "$0")/.."
WS_ROOT="$PWD"
UP="$WS_ROOT/.."
SRC="$UP"
[ -f "$SRC/CMakeLists.txt" ] || SRC="${SUNDIALS_C_SRC:-/mnt/c/Users/nsh/Developer/sundials-7.8.0}"
CB="${1:-$HOME/sdl/cbuild}"
LOGS="$WS_ROOT/logs"
WORK="$LOGS/lsub"
mkdir -p "$WORK"

noise_filter() { grep -v -E 'Total run time|CPU time|cpu time|wall clock' || true; }

LIBS=$(find "$CB" -name 'libsundials_*.a' | tr '\n' ' ')
[ -n "$LIBS" ] || { echo "no pristine C libraries under $CB — run tools/pristine_c_build.sh first"; exit 1; }

printf '%-22s %-12s %-12s %s\n' VARIANT RS_vs_Csub Csub_vs_REF RS_vs_REF
printf '%s\n' '-----------------------------------------------------------------'

for pair in "cvode/serial/cvRoberts_dnsL" "cvodes/serial/cvsRoberts_dnsL"; do
  solver="${pair%%/*}"; rest="${pair#*/}"; exdir="${rest%%/*}"; name="${rest##*/}"
  csrc="$SRC/examples/$solver/$exdir/$name.c"
  ref="$SRC/examples/$solver/$exdir/$name.out"
  rbin="$WS_ROOT/target/release/examples/$name"

  [ -f "$csrc" ] || { printf '%-22s NO-SOURCE\n' "$name"; continue; }
  [ -x "$rbin" ] || { printf '%-22s NO-RUST-BINARY\n' "$name"; continue; }

  # The substitution: exactly the header and the constructor. Nothing else.
  sed -e 's#sunlinsol/sunlinsol_lapackdense\.h#sunlinsol/sunlinsol_dense.h#' \
      -e 's#SUNLinSol_LapackDense#SUNLinSol_Dense#g' \
      "$csrc" > "$WORK/$name.c"

  if ! gcc -O2 -o "$WORK/$name" "$WORK/$name.c" \
        -I"$SRC/include" -I"$CB/include" \
        -Wl,--start-group $LIBS -Wl,--end-group -lm > "$WORK/$name.build.log" 2>&1; then
    printf '%-22s BUILD-FAILED (see logs/lsub/%s.build.log)\n' "$name" "$name"
    continue
  fi

  ( cd "$WORK" && "./$name" > "$WORK/csub-$name.out" 2>&1 )
  ( cd "$WORK" && "$rbin"   > "$WORK/rs-$name.out"   2>&1 )

  rc=$(diff <(noise_filter < "$WORK/rs-$name.out") <(noise_filter < "$WORK/csub-$name.out") \
        >/dev/null 2>&1 && echo same || echo DIFF)
  cr=$(diff <(noise_filter < "$WORK/csub-$name.out") <(noise_filter < "$ref") \
        >/dev/null 2>&1 && echo same || echo DIFF)
  rr=$(diff <(noise_filter < "$WORK/rs-$name.out") <(noise_filter < "$ref") \
        >/dev/null 2>&1 && echo same || echo DIFF)
  [ "$rc" = same ] || diff <(noise_filter < "$WORK/rs-$name.out") \
                           <(noise_filter < "$WORK/csub-$name.out") \
                           > "$LOGS/cdiff-$name-lapacksub.txt" 2>&1

  printf '%-22s %-12s %-12s %s\n' "$name" "$rc" "$cr" "$rr"
done
