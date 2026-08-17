#!/usr/bin/env bash
# c_examples_run.sh — execute every C example binary produced by
# tools/c_build.sh, once per (example, argv) variant declared in the
# upstream CMakeLists.txt files, and record exactly what happened.
#
#   tools/c_examples_run.sh [build-dir]     # default: build/c
#
# Output, all under c-results/:
#   raw/<dir>/<variant>.stdout   captured stdout            (byte for byte)
#   raw/<dir>/<variant>.stderr   captured stderr
#   raw/<dir>/<variant>.meta     argv, exit status, wall time, sha256
#   index.tsv                    one row per variant, tab separated
#
# Nothing here interprets or edits program output. The .stdout files are
# what the binaries printed, unmodified, so any claim made later in
# c-results/*.md can be checked against them with plain `cat` and `diff`.
set -u

cd "$(dirname "$0")/.."
WS=$PWD
BUILD=${1:-$WS/build/c}
OUT=$WS/c-results
RAW=$OUT/raw
IDX=$OUT/index.tsv
SCRATCH=$WS/build/c-run
TIMEOUT=${SUNDIALS_EXAMPLE_TIMEOUT:-600}

[ -d "$BUILD/examples" ] || { echo "no build at $BUILD — run tools/c_build.sh first"; exit 1; }

rm -rf "$RAW" "$SCRATCH"
mkdir -p "$RAW" "$SCRATCH" "$OUT"
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

variant_id() { # <name> <args>
  if [ -z "$2" ]; then printf '%s' "$1"
  else printf '%s__%s' "$1" "$(printf '%s' "$2" | tr ' /' '__')"; fi
}

printf 'dir\texample\targv\tvariant\tstatus\texit\tseconds\tstdout_bytes\tstdout_sha256\n' >"$IDX"

total=0; ok=0; failed=0
mapfile -t BINS < <(find "$BUILD/examples" -type f -perm -u+x ! -name '*.*' | sort)
echo "found ${#BINS[@]} example binaries"

for bin in "${BINS[@]}"; do
  rel=${bin#"$BUILD"/examples/}
  dir=$(dirname "$rel")
  name=$(basename "$rel")
  cml=$WS/examples/$dir/CMakeLists.txt

  # every (argv, mpi-task-count) variant this example is declared with
  mapfile -t VARIANTS < <(parse_cmake "$cml" | awk -F'|' -v n="$name" '$1==n {print $2 "|" $3}')
  [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=("|")

  mkdir -p "$RAW/$dir"
  for spec in "${VARIANTS[@]}"; do
    args=${spec%%|*}
    tasks=${spec##*|}
    vid=$(variant_id "$name" "$args")
    run=$SCRATCH/$dir/$vid
    mkdir -p "$run"
    total=$((total + 1))

    # Examples that CMake declares with an MPI task count are launched
    # under mpirun with exactly that count, which is how upstream runs
    # them. Everything else runs directly.
    LAUNCH=()
    if [ -n "$tasks" ] && [ "$tasks" != "0" ] && command -v mpirun >/dev/null 2>&1; then
      LAUNCH=(mpirun --oversubscribe -np "$tasks")
    fi

    start=$(date +%s.%N)
    # shellcheck disable=SC2086  # argv must word-split exactly as CMake declares it
    ( cd "$run" && timeout "$TIMEOUT" "${LAUNCH[@]}" "$bin" $args ) \
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
      src="examples/$dir/$name.c"
      for ext in c cpp f90 F90 cu; do
        if [ -f "$WS/examples/$dir/$name.$ext" ] || [ -f "$WS/upstream-c/examples/$dir/$name.$ext" ]; then
          src="examples/$dir/$name.$ext"; break
        fi
      done
      # C++ and Fortran examples were previously all recorded as .c, which
      # named a file that does not exist for 101 of the 337 runs.
      echo "source:   $src"
      echo "binary:   $bin"
      echo "argv:     $args"
      echo "launcher: ${LAUNCH[*]:-<direct>}"
      echo "cwd:      $run"
      echo "exit:     $rc ($status)"
      echo "seconds:  $secs"
      echo "stdout:   $bytes bytes, sha256 $(sha256sum "$RAW/$dir/$vid.stdout" | cut -d' ' -f1)"
      echo "stderr:   $(stat -c%s "$RAW/$dir/$vid.stderr") bytes"
    } >"$RAW/$dir/$vid.meta"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$dir" "$name" "$args" "$vid" "$status" "$rc" "$secs" "$bytes" "$sha" >>"$IDX"
  done
done

echo
echo "ran $total variants: $ok OK, $failed not-OK"
echo "index: $IDX"
