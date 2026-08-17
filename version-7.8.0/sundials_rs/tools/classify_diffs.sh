#!/usr/bin/env bash
# classify_diffs.sh — second pass over the non-IDENTICAL variants.
#
# tools/verify_examples.sh writes each run's captured stdout to
# logs/<expected .out name>. This script re-diffs every one of those
# against the upstream reference under three progressively looser
# comparisons, so a divergence can be placed in one of the classes
# VERIFICATION.md uses without opening 26 diffs by hand:
#
#   exact      byte-identical (what the gate requires)
#   squeeze    `tr -s " "` on both sides — catches SUN_TABLE_WIDTH 28->29
#              column drift, i.e. every value identical, spacing stale
#   ws         `diff -w` — also ignores trailing whitespace
#
# A variant that is `squeeze`-clean has no numeric difference at all.
set -u
cd "$(dirname "$0")/.."
LOGS="$PWD/logs"
UP="$PWD/.."

printf '%-58s %-9s %-9s %s\n' VARIANT EXACT SQUEEZE WS
for f in "$LOGS"/*.out; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  # -H: the examples/ argument may itself be a symlink into the C tree.
  ref=$(find -H "$UP/examples/" -name "$b" -print -quit 2>/dev/null)
  [ -n "$ref" ] || { printf '%-58s %s\n' "$b" NO-REF; continue; }
  e=$(diff -q "$f" "$ref" >/dev/null 2>&1 && echo same || echo DIFF)
  [ "$e" = same ] && continue
  s=$(diff -q <(tr -s ' ' < "$f") <(tr -s ' ' < "$ref") >/dev/null 2>&1 && echo same || echo DIFF)
  w=$(diff -qw "$f" "$ref" >/dev/null 2>&1 && echo same || echo DIFF)
  printf '%-58s %-9s %-9s %s\n' "$b" "$e" "$s" "$w"
done
