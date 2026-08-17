#!/usr/bin/env bash
# verify_examples.sh [crate|all|list]
#
# Batch example-verification harness (prompt §6).
#
# PLATFORM SCOPE — Linux on Intel/AMD x86-64 with glibc.
#   Requirements: POSIX bash, cargo, and the read-only upstream SUNDIALS
#   7.8.0 C tree as this workspace's PARENT directory (reference .out files
#   are read from ../examples/<solver>/<serial dir>/). Without that tree the
#   script has nothing to diff against and every variant reports NO-REF. On
#   Windows it needs Git Bash / MSYS2 / WSL; it will not run under cmd.exe
#   or PowerShell. tools/wsl_sync_build.sh gate wires up the WSL case.
#
#   It executes on any POSIX host meeting those requirements, but its
#   VERDICTS are only meaningful on glibc/x86-64. The port takes sin, cos,
#   asin, acos, atan, sinh, cosh, acosh, exp and ln from the host libm (only
#   `pow` was made host-independent; sqrt/mul_add/ceil/round are IEEE-exact
#   and portable), and on glibc that host libm is the one that generated the
#   upstream references — which is why the gate reaches 153 IDENTICAL here
#   against 127 on macOS/arm64. On a different libm a different set of
#   variants diverges; those must be re-classified from scratch on that
#   host — never by tuning an example, and never by widening noise_filter()
#   to swallow last-ulp drift. Current result: 153 IDENTICAL / 26
#   reference-side / 20 excluded, 0 port defects. tools/classify_diffs.sh is
#   the second pass over the 26. See README.md § "Platform scope".
#
#   list        print every (crate, example, args, outfile, status) variant
#               tuple for all crates, tab-separated — used to (re)generate
#               VERIFICATION.md and to cross-check .out coverage.
#   <crate>     build that crate's examples in release and verify every
#               variant against the upstream reference outputs.
#   all         verify every crate ported so far (cumulative gate).
#
# Reference outputs live in the upstream C tree, which contains this
# workspace: ../examples/<solver>/<dir>/<name>[_<args>].out
#
# Machine-noise filter: lines matching the regexes in noise_filter() are
# removed from BOTH sides before diffing (documented per §6; extend only
# with genuinely machine-dependent lines, symmetrically).

set -u
cd "$(dirname "$0")/.."
WS_ROOT="$(pwd)"
# Reference outputs and CMake tuples: prefer the tree vendored in this
# repository, and fall back to the historical layout where the upstream
# SUNDIALS checkout is this workspace's parent directory.
if [ -d "$WS_ROOT/examples/cvode/serial" ]; then
  UPSTREAM="$WS_ROOT"
else
  UPSTREAM="$WS_ROOT/.."
fi
LOGS="$WS_ROOT/logs"
mkdir -p "$LOGS"

exdir_of() {
  case "$1" in
    cvode_rs)  echo "examples/cvode/serial" ;;
    cvodes_rs) echo "examples/cvodes/serial" ;;
    kinsol_rs) echo "examples/kinsol/serial" ;;
    ida_rs)    echo "examples/ida/serial" ;;
    idas_rs)   echo "examples/idas/serial" ;;
    arkode_rs) echo "examples/arkode/C_serial" ;;
    *) echo "unknown crate: $1" >&2; exit 2 ;;
  esac
}

ALL_CRATES="cvode_rs cvodes_rs kinsol_rs ida_rs idas_rs arkode_rs"

# Exactly the 20 KLU/SuperLU-dependent examples excluded by prompt §1.
excluded_reason() {
  case "$1" in
    cvRoberts_klu|cvRoberts_block_klu|cvsRoberts_klu|cvsRoberts_FSA_klu|\
    cvsRoberts_ASAi_klu|kinFerTron_klu|idaHeat2D_klu|idaRoberts_klu|\
    idasRoberts_klu|idasRoberts_FSA_klu|idasRoberts_ASAi_klu)
      echo "excluded(klu)" ;;
    cvRoberts_sps|cvsRoberts_sps|cvsRoberts_FSA_sps|cvsRoberts_ASAi_sps|\
    kinRoboKin_slu|idaRoberts_sps|idasRoberts_sps|idasRoberts_FSA_sps|\
    idasRoberts_ASAi_sps)
      echo "excluded(superlu)" ;;
    *) echo "" ;;
  esac
}

# Parse one CMakeLists.txt: emit "name<TAB>args<TAB>outfile" per tuple.
# Rules (prompt §6): plain text-level scan of non-comment lines; tuples are
# quoted strings with `\;` separators; 3 fields = name/args/label, 2 fields
# = name/label (kinsol KLU/SLU lists); arkode names carry a .c suffix to
# strip; outfile = name.out if args empty else name_<args, spaces->_>.out.
parse_cmake() {
  local cml="$1"
  grep -v '^[[:space:]]*#' "$cml" \
    | grep -o '"[^"]*\\;[^"]*"' \
    | sed -e 's/^"//' -e 's/"$//' \
    | while IFS= read -r tuple; do
        local name args rest
        name="${tuple%%\\;*}"
        rest="${tuple#*\\;}"
        case "$rest" in
          *\\\;*) args="${rest%%\\;*}" ;;   # 3-field tuple
          *)      args="" ;;                # 2-field tuple (label only)
        esac
        name="${name%.c}"
        local outfile
        if [ -z "$args" ]; then
          outfile="${name}.out"
        else
          outfile="${name}_$(printf '%s' "$args" | tr ' ' '_').out"
        fi
        printf '%s|%s|%s\n' "$name" "$args" "$outfile"
      done
}

# Cross-check: every .out in the dir must be claimed by some variant.
crosscheck_outs() {
  local exdir="$1"
  local claimed missing=0
  claimed="$(parse_cmake "$UPSTREAM/$exdir/CMakeLists.txt" | cut -d'|' -f3 | sort -u)"
  for f in "$UPSTREAM/$exdir"/*.out; do
    local base; base="$(basename "$f")"
    if ! printf '%s\n' "$claimed" | grep -qxF "$base"; then
      echo "ORPHAN .out not claimed by any CMake tuple: $exdir/$base" >&2
      missing=1
    fi
  done
  return $missing
}

# Remove machine-dependent noise lines symmetrically before diffing.
noise_filter() {
  grep -v -E 'Total run time|CPU time|cpu time|wall clock' || true
}

list_all() {
  local rc=0
  for crate in $ALL_CRATES; do
    local exdir; exdir="$(exdir_of "$crate")"
    crosscheck_outs "$exdir" || rc=1
    parse_cmake "$UPSTREAM/$exdir/CMakeLists.txt" \
      | while IFS='|' read -r name args outfile; do
          local status; status="$(excluded_reason "$name")"
          [ -z "$status" ] && status="port"
          printf '%s|%s|%s|%s|%s\n' "$crate" "$name" "$args" "$outfile" "$status"
        done
  done
  return $rc
}

verify_crate() {
  local crate="$1"
  local exdir; exdir="$(exdir_of "$crate")"
  local cml="$UPSTREAM/$exdir/CMakeLists.txt"

  crosscheck_outs "$exdir" || echo "WARNING: orphan .out files (see above)" >&2

  echo "== Building $crate examples (release) =="
  if ! cargo build --release --examples -p "$crate" \
       >"$LOGS/build-$crate.log" 2>&1; then
    echo "$crate BUILD FAILED — see logs/build-$crate.log" | tee -a "$LOGS/summary.txt"
    return 1
  fi

  parse_cmake "$cml" | while IFS='|' read -r name args outfile; do
    local status; status="$(excluded_reason "$name")"
    if [ -n "$status" ]; then
      printf '%-40s [%s]  %s\n' "$name" "$args" "EXCLUDED ${status}" >>"$LOGS/summary.txt"
      continue
    fi
    local ref="$UPSTREAM/$exdir/$outfile"
    if [ ! -f "$ref" ]; then
      printf '%-40s [%s]  %s\n' "$name" "$args" "NO-REF($outfile)" >>"$LOGS/summary.txt"
      continue
    fi
    local bin="$WS_ROOT/target/release/examples/$name"
    if [ ! -x "$bin" ]; then
      printf '%-40s [%s]  %s\n' "$name" "$args" "NO-BINARY" >>"$LOGS/summary.txt"
      continue
    fi
    # shellcheck disable=SC2086  # args must word-split exactly like a shell command line
    ( cd "$LOGS" && "$bin" $args >"$LOGS/$outfile" 2>&1 )
    local code=$?
    if [ $code -ne 0 ]; then
      printf '%-40s [%s]  FAIL(%d)\n' "$name" "$args" "$code" >>"$LOGS/summary.txt"
      continue
    fi
    local nd
    nd="$(diff <(noise_filter <"$ref") <(noise_filter <"$LOGS/$outfile") | grep -c '^[<>]')"
    if [ "$nd" -eq 0 ]; then
      printf '%-40s [%s]  IDENTICAL\n' "$name" "$args" >>"$LOGS/summary.txt"
    else
      printf '%-40s [%s]  DIFF(%d lines)\n' "$name" "$args" "$nd" >>"$LOGS/summary.txt"
      diff <(noise_filter <"$ref") <(noise_filter <"$LOGS/$outfile") \
        >"$LOGS/diff-$outfile.txt" 2>&1
    fi
  done
}

case "${1:-all}" in
  list) list_all ;;
  all)
    : >"$LOGS/summary.txt"
    for crate in $ALL_CRATES; do
      if grep -q '\[\[example\]\]' "$WS_ROOT/crates/$crate/Cargo.toml" 2>/dev/null; then
        echo "### $crate ###" >>"$LOGS/summary.txt"
        verify_crate "$crate"
      fi
    done
    echo "Done. Read logs/summary.txt"
    ;;
  *)
    : >"$LOGS/summary.txt"
    verify_crate "$1"
    echo "Done. Read logs/summary.txt"
    ;;
esac
