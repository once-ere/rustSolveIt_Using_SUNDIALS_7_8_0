#!/usr/bin/env bash
#
# Certify that this repository carries no licence-encumbered source.
#
# Run it against a FRESH PLAIN CLONE, which is the only thing that tells
# you what a user actually receives:
#
#     git clone https://github.com/once-ere/rustSimulate.git /tmp/cert
#     /tmp/cert/scripts/certify_clean.sh
#
# WHY THIS EXISTS
#
# On 2026-07-26, 749 files of the SolveIt 2002 C/C++ sources — including
# Numerical Recipes and GSL-derived code — were pushed to the then-public
# repository, because `.gitignore` said `./obsolete_or_historic` and a
# leading `./` matches nothing. Three separate ad-hoc checks failed to
# catch it: one was scoped to the wrong file set, one used `ls ... ||`
# so it could not fail loudly, and one set a flag inside `$(...)` where
# the assignment died with the subshell.
#
# So this script is written to fail LOUDLY and to be the single place
# that knows what "clean" means. Every check asserts on a count, prints
# what it found when it fails, and the exit status is real.
#
# See CLEANROOM_PROVENANCE.md §7.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

fail=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=1; }

# Paths that must never be tracked. These are local-only reference trees.
FORBIDDEN_PATHS='^(obsolete_or_historic|SolveIt|SolveIt_2026_MFC)(/|$)'

# The SAME set, matched against `git rev-list --objects` output, whose
# lines are "<sha> <path>" — so the path is NOT at the start of the line.
# Anchoring this one with `^` is precisely the bug that made check 2
# vacuous the first time. Keep the two patterns separate and obviously
# different so the distinction cannot be lost in an edit.
FORBIDDEN_OBJECTS='[[:space:]](obsolete_or_historic|SolveIt|SolveIt_2026_MFC)(/|$)'

# Signatures of encumbered CODE. Deliberately matched only against
# SOURCE files: the documentation legitimately *discusses* Numerical
# Recipes by name (CLEANROOM_PROVENANCE.md, THIRD_PARTY.md), and a check
# that cannot tell "mentions" from "contains" is a check that cries wolf.
CODE_SIGNATURES='bessj0|bessj1|nrutil|gsl_linalg_solve|gsl_vector_|NR_END|FREE_ARG'

# A LIVE claim about the current tree's test count, as opposed to a
# dated historical record. Three stages running (2B, 2C, 2D) turned on
# documentation that was accurate when written and false when read, so
# the rule is mechanised rather than written down again as an action.
#
# The distinction matters and is the same one CODE_SIGNATURES makes:
# EXPORT_PROVENANCE.md legitimately RECORDS "104 passed" as the state at
# export time, and TUNNELING_RESULTS.md records 304 for its round. Those
# are history and must not be rewritten. Only the two canonical phrasings
# below assert something about the tree *now*, and only those are checked.
LIVE_COUNT='(expect ([0-9]+) passed|([0-9]+) passed workspace-wide)'

# Claims that were true once and are now false. Each is a real sentence
# that shipped, with the stage that retired it; the gate exists so they
# cannot quietly return.
RETIRED_CLAIMS='remains unfixed|complex Airy is not implemented|no complex arithmetic anywhere|10\.20 is not the right tool|MAX_EVENTS_PER_OUTPUT'
SOURCE_EXT='\.(rs|c|h|cpp|hpp|cc|f|f90|py)$'

echo "Certifying $(pwd)"
echo "HEAD: $(git log --oneline -1 2>/dev/null || echo '<not a git repo>')"
echo

# ---- 0. SELF-TEST: prove each detector can actually fire ---------------
#
# The first version of this script shipped a VACUOUS gate. Check 2 piped
# `git rev-list --objects`, whose lines are "<sha> <path>", into a
# pattern anchored with `^` — so it matched against the SHA and never
# fired. 676 forbidden objects were present and it reported PASS.
#
# A check that cannot fail is worse than no check, because it manufactures
# confidence. So every detector below is first run against a synthetic
# line that MUST match. If a detector does not fire on known-bad input,
# this script aborts rather than printing reassuring PASS lines.
selftest() { # <description> <text that must match> <pattern>
  if ! printf '%s\n' "$2" | grep -qE "$3"; then
    printf '  \033[31mABORT\033[0m  self-test failed: %s\n' "$1"
    printf '         the detector below cannot fire, so its PASS would be meaningless\n'
    exit 2
  fi
}
selftest "tracked-path detector"  "obsolete_or_historic/SolveIt/QM/QMEvolve.h" "$FORBIDDEN_PATHS"
selftest "history-object detector" \
         "9da888ae98f3ac8b3e997384da2654a298f5dcd3 obsolete_or_historic/SolveIt/QM/QMEvolve.h" \
         "$FORBIDDEN_OBJECTS"
selftest "code-signature detector" "float bessj0(float x)" "$CODE_SIGNATURES"
selftest "live-count detector"    "cargo test --workspace   # expect 104 passed" "$LIVE_COUNT"
selftest "live-count detector 2"  "**256 passed workspace-wide**; zero failures" "$LIVE_COUNT"
selftest "retired-claim detector" "This is a real defect and remains unfixed." "$RETIRED_CLAIMS"
# ...and the stripping must not swallow an UNquoted assertion:
selftest "retired-claim survives stripping" \
         "$(printf 'a defect that remains unfixed' | sed 's/"[^"]*"//g')" "$RETIRED_CLAIMS"

# ---- 1. no forbidden path tracked at the tip --------------------------
n=$(git ls-files | grep -cE "$FORBIDDEN_PATHS")
if [ "$n" -eq 0 ]; then
  pass "no reference-tree files tracked"
else
  bad "$n reference-tree files are TRACKED:"
  git ls-files | grep -E "$FORBIDDEN_PATHS" | head -10 | sed 's/^/        /'
fi

# ---- 2. no forbidden path anywhere in history -------------------------
# Removing files at the tip is not enough: old commits stay fetchable.
n=$(git rev-list --objects --all | grep -cE "$FORBIDDEN_OBJECTS" || true)
if [ "$n" -eq 0 ]; then
  pass "no reference-tree objects reachable in history"
else
  bad "$n reference-tree objects are still REACHABLE IN HISTORY"
  git rev-list --objects --all | grep -E "$FORBIDDEN_OBJECTS" \
    | head -5 | sed 's/^/        /'
  echo "        Reachable from these refs:"
  git for-each-ref --format='          %(refname:short)' | sed 's/$//' | head -10
  echo "        (removing files at the tip is NOT enough — the objects stay"
  echo "         fetchable by SHA. Rewrite history, or drop the ref holding them.)"
fi

# ---- 3. no encumbered code signatures in source files -----------------
hits=$(git ls-files | grep -E "$SOURCE_EXT" \
       | xargs -r grep -lniE "$CODE_SIGNATURES" 2>/dev/null || true)
if [ -z "$hits" ]; then
  pass "no encumbered-code signatures in tracked source"
else
  bad "encumbered-code signatures found in tracked SOURCE files:"
  echo "$hits" | sed 's/^/        /'
fi

# ---- 4. the ignore rules must actually match --------------------------
# The whole incident was an ignore pattern that looked right and matched
# nothing. Never assume — interrogate git.
for p in obsolete_or_historic SolveIt SolveIt_2026_MFC; do
  if git check-ignore -q "$p/probe.h"; then
    pass "gitignore matches $p/"
  else
    bad "gitignore does NOT match $p/ — a pattern like './$p' silently ignores nothing"
  fi
done

# ---- 4b. Stages_output must never be published ------------------------
# An internal record of every stage response. It is deliberately not part
# of the repository, and one mechanism is not evidence: the ignore rule is
# interrogated, AND the index is checked for tracked files. Either failing
# is a defect.
if git check-ignore -q "Stages_output/probe.md"; then
  pass "gitignore matches Stages_output/"
else
  bad "gitignore does NOT match Stages_output/ — the transcripts could be published"
fi
tracked_stages=$(git ls-files Stages_output/ | head -20)
if [ -z "$tracked_stages" ]; then
  pass "no Stages_output files tracked"
else
  bad "Stages_output files are TRACKED and would be pushed:"
  echo "$tracked_stages" | sed 's/^/        /'
fi

# ---- 5. the build and test gates --------------------------------------
if cargo build --workspace --release >/tmp/cert_build.$$ 2>&1; then
  w=$(grep -c '^warning' /tmp/cert_build.$$ || true)
  if [ "$w" -eq 0 ]; then pass "release build, 0 warnings"
  else bad "release build emitted $w warnings"; fi
else
  bad "release build FAILED"; tail -20 /tmp/cert_build.$$ | sed 's/^/        /'
fi
rm -f /tmp/cert_build.$$

# clippy across ALL targets, not just the library: examples, tests and
# benches are shipped code too, and lints only surface in the crate the
# run reaches before the first failure — so a partial pass here is
# genuinely uninformative.
if cargo clippy --workspace --all-targets >/tmp/cert_clippy.$$ 2>&1; then
  pass "clippy --workspace --all-targets, 0 errors"
else
  bad "clippy reported problems:"
  grep -E '^(error|warning)' /tmp/cert_clippy.$$ | head -10 | sed 's/^/        /'
fi
rm -f /tmp/cert_clippy.$$

if cargo test --workspace >/tmp/cert_test.$$ 2>&1; then
  p=$(grep -E 'test result' /tmp/cert_test.$$ | awk '{s+=$4} END {print s}')
  pass "all tests pass ($p assertions across the workspace)"

  # ---- 6. documentation claims about THIS tree ------------------------
  #
  # Stages 2B, 2C and 2D each found prose that was accurate when written
  # and false by the time it was read — including EXPORT_PROVENANCE.md,
  # the document a reader uses to AUDIT the release, still saying a
  # repaired defect "remains unfixed" and to expect 104 tests against
  # 556. The lesson was written down as an action three times; this is
  # it mechanised.
  stale=""
  while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    file=${hit%%:*}
    n=$(printf '%s\n' "$hit" | grep -oE "$LIVE_COUNT" | grep -oE '[0-9]+' | head -1)
    [ -z "$n" ] && continue
    if [ "$n" != "$p" ]; then
      stale="$stale\n        $file claims $n tests; the tree has $p"
    fi
  done <<EOF
$(git ls-files '*.md' | while read -r f; do grep -nE "$LIVE_COUNT" "$f" 2>/dev/null | sed "s|^|$f:|"; done)
EOF
  if [ -z "$stale" ]; then
    pass "documented test counts match the tree ($p)"
  else
    bad "documentation states a test count this tree does not have:"
    printf "$stale\n"
  fi

  # ---- 7. the stage notebooks must still execute ----------------------
  #
  # StageNbks/ is user-facing teaching material: eleven standalone posim
  # notebooks, each claiming to reproduce a stage. They are the only
  # shipped content whose correctness depends on the LANGUAGE not moving
  # under them, and nothing else in this script exercises them.
  #
  # The need is not hypothetical. When they were written, running them
  # found two real errors that reading them had not: two notebooks used a
  # `version` command that does not exist, and two more asserted that
  # `bessel_y_nu` would refuse at points where the language actually
  # routes to a different, working implementation. Both would have
  # shipped as confident, wrong instructions.
  #
  # Cost: about 14 seconds for all eleven.
  nb_bad=""
  if [ -x target/release/posim ]; then
    for nb in StageNbks/*.posim; do
      [ -e "$nb" ] || continue
      if target/release/posim --notebook "$nb" 2>&1 | grep -q 'Err\['; then
        nb_bad="$nb_bad $nb"
      fi
    done
    if [ -z "$nb_bad" ]; then
      pass "every StageNbks notebook executes without error"
    else
      bad "stage notebooks failed to execute cleanly:"
      for nb in $nb_bad; do
        printf '        %s\n' "$nb"
        target/release/posim --notebook "$nb" 2>&1 | grep 'Err\[' | head -2 \
          | sed 's/^/          /'
      done
    fi
  else
    bad "target/release/posim missing — cannot check the stage notebooks"
  fi

  # ---- 8. no stage notebook may refer to another stage notebook -------
  #
  # The folder's whole premise is that each notebook is stand-alone: a
  # reader who opens one file must never be sent to a second one. That is
  # a rule prose cannot keep on its own, and it was already broken once —
  # a fix for a display bug ended eight notebooks with "see Stage_2C.posim
  # for a working window", which is exactly the cross-reference the folder
  # forbids. It shipped, and a reader caught it, not a gate.
  #
  # A notebook naming ITSELF is normal and expected: every one prints its
  # own path in its run instructions. So self-references are stripped
  # before matching, and only mentions of a DIFFERENT notebook fail.
  #
  # README.md is deliberately not checked: it is an index, so naming every
  # notebook is its job. What it must not do is send the reader into one
  # to learn how to run it, and that is a judgement no grep can make.
  xref=""
  for nb in StageNbks/*.posim; do
    [ -e "$nb" ] || continue
    self=$(basename "$nb" .posim)
    hits=$(grep -oE 'Stage_(24|2[A-K])' "$nb" | grep -v "^$self$" | sort -u | tr '\n' ' ')
    if [ -n "$hits" ]; then
      xref="$xref$nb -> $hits
"
    fi
  done
  if [ -z "$xref" ]; then
    pass "no stage notebook refers to another stage notebook"
  else
    bad "a stage notebook sends the reader to another notebook:"
    printf '%s' "$xref" | sed 's/^/        /'
  fi

  # Claims retired by a later stage must not come back. These are real
  # sentences that shipped and became false; the pattern is the record.
  #
  # QUOTING one is not asserting it. Both PROJECT_STATUS.md and
  # airy_uniform.rs quote their retired sentence *while explaining that
  # it was retired*, which is exactly the behaviour this project wants —
  # so quoted spans are stripped before matching, the same "mentions vs
  # contains" distinction CODE_SIGNATURES makes by only reading source.
  # Without it the gate would punish the honest write-up.
  revived=$(git ls-files '*.md' '*.rs' | grep -v '^sundials_rs/' | grep -v '^vendor/' \
            | while read -r f; do
                if sed 's/"[^"]*"//g' "$f" | grep -qE "$RETIRED_CLAIMS"; then printf '%s\n' "$f"; fi
              done)
  if [ -z "$revived" ]; then
    pass "no retired claim has reappeared"
  else
    bad "a claim retired by an earlier stage is back:"
    printf '%s\n' "$revived" | sed 's/^/        /'
    printf '%s\n' "$revived" | while read -r f; do
      sed 's/"[^"]*"//g' "$f" | grep -nE "$RETIRED_CLAIMS" | sed "s|^|          $f:|"
    done | head -6
  fi
else
  bad "tests FAILED"
  grep -A4 'panicked' /tmp/cert_test.$$ | head -20 | sed 's/^/        /'
fi
rm -f /tmp/cert_test.$$

echo
if [ "$fail" -eq 0 ]; then
  printf '\033[32mCLEAN\033[0m — nothing encumbered is tracked or reachable.\n'
else
  printf '\033[31mDEFECTS REMAIN\033[0m — see the FAIL lines above.\n'
fi
exit "$fail"
