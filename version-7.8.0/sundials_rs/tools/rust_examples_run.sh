#!/usr/bin/env bash
# rust_examples_run.sh — execute every translated Rust example, once per
# (example, argv) variant, and record exactly what happened.
#
#   tools/rust_examples_run.sh
#
# This is the deliberate mirror image of tools/c_examples_run.sh: the same
# CMakeLists tuples decide the variant set, the same directory layout and
# the same file names are produced, so `differences/` can pair the two
# sides up by path alone.
#
# Output, all under rust-results/:
#   raw/<dir>/<variant>.stdout   captured stdout (byte for byte)
#   raw/<dir>/<variant>.stderr   captured stderr
#   raw/<dir>/<variant>.meta     argv, exit status, wall time, sha256
#   index.tsv                    one row per variant, tab separated
set -u

cd "$(dirname "$0")/.."
WS=$PWD
OUT=$WS/rust-results
RAW=$OUT/raw
IDX=$OUT/index.tsv
BINDIR=$WS/target/release/examples
SCRATCH=$WS/build/rust-run
TIMEOUT=${SUNDIALS_EXAMPLE_TIMEOUT:-600}

# crate -> the upstream example directory it was translated from
exdir_of() {
  case "$1" in
    cvode_rs)  echo "cvode/serial" ;;
    cvodes_rs) echo "cvodes/serial" ;;
    kinsol_rs) echo "kinsol/serial" ;;
    ida_rs)    echo "ida/serial" ;;
    idas_rs)   echo "idas/serial" ;;
    arkode_rs) echo "arkode/C_serial" ;;
    *) return 1 ;;
  esac
}
CRATES="cvode_rs cvodes_rs kinsol_rs ida_rs idas_rs arkode_rs"
# Parse one CMakeLists.txt into "name|args|tasks" rows.
#
# Upstream declares its examples as quoted, backslash-semicolon separated
# tuples, but the *arity differs by directory* and the header row that is
# supposed to document it is sometimes stale:
#
#   examples/cvode/serial     "cvRoberts_dns\;\;develop"                name args type
#   examples/cvode/parallel   "cvAdvDiff_diag_p\;2\;4\;exclude-single"  name nodes tasks type
#   examples/arkode/C_parallel"ark_diurnal_kry_p\;\;1\;4\;...\;default" name args nodes tasks type ...
#
# Taking field 2 as argv unconditionally would run `cvAdvDiff_diag_p 2`.
# So the schema is recovered from the data instead: the first pair of
# *consecutive all-integer* fields is (nodes, tasks); everything between
# the name and that pair is argv. Tuples whose first field is literally
# "name" are the header row and are skipped.
parse_cmake() {
  local cml=$1
  [ -f "$cml" ] || return 0
  grep -v '^[[:space:]]*#' "$cml" \
    | grep -o '"[^"]*\;[^"]*"' \
    | sed -e 's/^"//' -e 's/"$//' \
    | awk -F'\\\;' '
        $1 == "name" { next }                      # schema header, not an example
        {
          name = $1; sub(/\.(c|cpp|f90)$/, "", name)
          nodes_at = 0
          for (i = 2; i < NF; i++) {
            if ($i ~ /^[0-9]+$/ && $(i+1) ~ /^[0-9]+$/) { nodes_at = i; break }
          }
          args = ""; tasks = ""
          if (nodes_at > 0) {
            for (i = 2; i < nodes_at; i++) args = (args == "" ? $i : args " " $i)
            tasks = $(nodes_at + 1)
          } else if (NF >= 3) {
            args = $2
          }
          printf "%s|%s|%s\n", name, args, tasks
        }'
}

variant_id() {
  if [ -z "$2" ]; then printf '%s' "$1"
  else printf '%s__%s' "$1" "$(printf '%s' "$2" | tr ' /' '__')"; fi
}

echo "building release examples..."
cargo build --release --workspace --examples 2>&1 | tail -3

rm -rf "$RAW" "$SCRATCH"
mkdir -p "$RAW" "$SCRATCH" "$OUT"
printf 'dir\texample\targv\tvariant\tstatus\texit\tseconds\tstdout_bytes\tstdout_sha256\n' >"$IDX"

total=0; ok=0; failed=0; missing=0
for crate in $CRATES; do
  dir=$(exdir_of "$crate")
  mkdir -p "$RAW/$dir"
  while IFS='|' read -r name args; do
    [ -z "$name" ] && continue
    bin=$BINDIR/$name
    vid=$(variant_id "$name" "$args")
    total=$((total + 1))

    if [ ! -x "$bin" ]; then
      # not translated (KLU / SuperLU backends have no pure-Rust counterpart)
      : >"$RAW/$dir/$vid.stdout"
      : >"$RAW/$dir/$vid.stderr"
      {
        echo "example:  $name"
        echo "crate:    $crate"
        echo "argv:     $args"
        echo "status:   NOT_PORTED (no release binary at $bin)"
      } >"$RAW/$dir/$vid.meta"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dir" "$name" "$args" "$vid" "NOT_PORTED" "-" "-" "0" "-" >>"$IDX"
      missing=$((missing + 1))
      continue
    fi

    run=$SCRATCH/$dir/$vid
    mkdir -p "$run"
    start=$(date +%s.%N)
    # shellcheck disable=SC2086  # argv must word-split exactly as CMake declares it
    ( cd "$run" && timeout "$TIMEOUT" "$bin" $args ) \
      >"$RAW/$dir/$vid.stdout" 2>"$RAW/$dir/$vid.stderr"
    rc=$?
    end=$(date +%s.%N)
    secs=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')
    case $rc in
      0)   status=OK ;;
      124) status=TIMEOUT ;;
      *)   status=NONZERO_EXIT ;;
    esac
    [ "$status" = OK ] && ok=$((ok + 1)) || failed=$((failed + 1))

    bytes=$(stat -c%s "$RAW/$dir/$vid.stdout")
    sha=$(sha256sum "$RAW/$dir/$vid.stdout" | cut -c1-16)
    {
      echo "example:  $name"
      echo "crate:    $crate"
      echo "source:   crates/$crate/examples/$name.rs"
      echo "binary:   $bin"
      echo "argv:     $args"
      echo "cwd:      $run"
      echo "exit:     $rc ($status)"
      echo "seconds:  $secs"
      echo "stdout:   $bytes bytes, sha256 $(sha256sum "$RAW/$dir/$vid.stdout" | cut -d' ' -f1)"
      echo "stderr:   $(stat -c%s "$RAW/$dir/$vid.stderr") bytes"
    } >"$RAW/$dir/$vid.meta"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$dir" "$name" "$args" "$vid" "$status" "$rc" "$secs" "$bytes" "$sha" >>"$IDX"
  done < <(parse_cmake "$WS/examples/$dir/CMakeLists.txt")
done

echo
echo "ran $total variants: $ok OK, $failed not-OK, $missing not ported"
echo "index: $IDX"
