#!/usr/bin/env bash
# vendor_evidence.sh — copy a measurement run from the working repository
# that produced it into this one, at the repository root.
#
#     tools/vendor_evidence.sh <source-repo>
#     tools/vendor_evidence.sh ../SUNDIALS_7_8_Rust_port_for_Linux_on_ubuntu
#
# The C-vs-Rust pipeline (c_build.sh, c_examples_run.sh, rust_examples_run.sh,
# compare_results.py, make_reports.py) writes c-results/, rust-results/ and
# differences/ at the root of whatever repository it runs in, and they land at
# the root here too. That keeps every relative link inside them correct with no
# rewriting -- ../requirements.md, ../LIBM.md and ../tools/... all resolve at
# this depth -- and it is why the copy is a straight rsync.
#
# The host these numbers belong to is recorded in the provenance table of each
# directory's README.md, not in a path. (evidence/linux-x86_64-glibc239/ still
# uses a host slug; that is the older .out-reference gate, a different
# measurement kept separate on purpose.)
#
# Nothing is transformed: .stdout, .stderr, .meta and the .tsv indexes are
# copied byte for byte, and the script fails if any link dangles afterwards.
set -euo pipefail

SRC=${1:?usage: vendor_evidence.sh <source-repo>}
# CDPATH= is not optional: with CDPATH set in the invoking shell -- and it is
# set in at least one shell this was run from -- `cd` echoes the directory it
# landed in, so $(cd ... && pwd) captures that echo *and* pwd, giving a
# two-line $ROOT. Every path built from it then contains a newline, rsync
# happily creates the resulting directory, and the real destination is never
# written. `--` guards a path that begins with a dash.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DEST="$ROOT"

case $ROOT in
  *[$'\n\t']* | "") echo "refusing to run: \$ROOT is not a single clean path" >&2; exit 1 ;;
esac

for d in c-results rust-results differences; do
  [ -d "$SRC/$d" ] || { echo "no $d in $SRC — run the pipeline there first" >&2; exit 1; }
done

mkdir -p "$DEST"
for d in c-results rust-results differences; do
  rsync -a --delete "$SRC/$d/" "$DEST/$d/"
done

# requirements.md records which optional C libraries this run could reach, and
# c-results/README.md links to it as ../requirements.md. Vendoring it beside
# the results keeps that link correct without rewriting it.
[ -f "$SRC/requirements.md" ] && cp "$SRC/requirements.md" "$DEST/requirements.md"

# The logs the documents cite by line number. requirements.md §3 attributes
# each backend failure to a line of c_build.log, and ATTRIBUTION.md's ulp
# table is read off libm_differential.log -- citations nobody can check if
# the logs stay behind in the working repository, where /logs is gitignored.
mkdir -p "$DEST/logs"
for l in c_build.log libm_differential.log; do
  [ -f "$SRC/logs/$l" ] && cp "$SRC/logs/$l" "$DEST/logs/$l"
done

# No link rewriting is needed: the result directories sit at the same depth
# here as in the source repository, so ../requirements.md, ../LIBM.md and
# ../tools/... resolve unchanged. That is the whole reason for putting them at
# the root rather than under a host slug.

# Every link must resolve from where the file now lives; a vendored evidence
# tree with dangling references is worse than none, because it looks checked.
python3 - "$DEST" <<'PY'
import pathlib
import re
import sys

dest = pathlib.Path(sys.argv[1])
root = dest
# Only the vendored trees: dest is the repository root now, so rglob over all
# of it would drag in README.md, crates/ and evidence/ as well.
targets = sorted(
    p
    for d in ("c-results", "rust-results", "differences")
    for p in (dest / d).rglob("*.md")
)
targets.append(dest / "requirements.md")
ok = bad = 0
for md in targets:
    if not md.exists():
        continue
    for _text, target in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", md.read_text()):
        if target.startswith(("http://", "https://", "#")):
            continue
        t = re.sub(r":\d+$", "", target.split("#")[0])
        if (md.parent / t).exists():
            ok += 1
        else:
            bad += 1
            print(f"  BROKEN  {md.relative_to(root)}  ->  {target}")
print(f"{ok} links resolve, {bad} broken")
sys.exit(1 if bad else 0)
PY

echo "vendored $(find "$DEST"/c-results "$DEST"/rust-results "$DEST"/differences -type f | wc -l) files into the repository root"
