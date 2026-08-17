#!/usr/bin/env bash
#
# Mutation probe: does this test suite actually notice when the code is
# wrong?
#
# The dominant failure mode in this project's history is a check that
# CANNOT FAIL. `certify_clean.sh` shipped a gate anchored `^` against
# "<sha> <path>" lines and reported PASS with 676 forbidden objects
# present. A Wronskian residual scaled by its own largest term reported
# 8.2e-24 and was measuring |H1/H2| rather than any error. A refusal test
# fell through to success when nothing was refused. Every one of those
# looked like coverage.
#
# A passing suite is evidence only if a broken program would fail it. So
# this script breaks the program, deliberately, in the places where the
# real defects have historically lived — branch choices, guard
# thresholds, safety factors, sampling offsets — and requires the suite
# to catch each one.
#
# A mutation that SURVIVES is the output. It names a line whose value the
# tests do not constrain, and there are three reasons that happens:
#
#   1. a missing test           — a real gap, fix it
#   2. an EQUIVALENT mutant     — the change cannot alter behaviour
#                                 inside the domain the code serves
#   3. a constant that turns out not to matter
#
# Only (1) is a defect, and telling them apart is the analysis this
# script exists to prompt. Survivors from the first full run are worked
# through in SPECIAL_FUNCTIONS_PROVENANCE.md; `zeta-anchor` is a
# textbook (2), and finding out WHY was worth more than the green tick
# would have been.
#
#   ./scripts/mutation_probe.sh            # every mutation
#   ./scripts/mutation_probe.sh debye      # only those whose id matches
#
# Exit status is the number of survivors, so it composes with CI.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

FILTER="${1:-}"
red=$'\033[31m'; green=$'\033[32m'; yellow=$'\033[33m'; off=$'\033[0m'

# id @@ crate @@ file @@ search @@ replace @@ what it breaks
#
# The delimiter is `@@`, not `|`: Rust closure syntax is full of `|i|`,
# and the first version of this table split a search string in half at
# `.map(|i|`, applied a mutation nobody chose, and reported it CAUGHT.
# A harness that mis-parses its own input manufactures exactly the false
# confidence it exists to remove.
#
# Each search string must be UNIQUE in its file — the harness checks
# that and aborts otherwise, because a mutation that silently applies
# somewhere else is testing nothing in particular.
MUTATIONS=$(cat <<'TABLE'
debye-branch@@special_functions@@special_functions/src/debye.rs@@if (exponent(t_principal) - target).abs() <= (exponent(flipped) - target).abs() {@@if false {@@the Debye branch discriminator (Stage 24: returned H1 as H2, wrong by 2e23)
debye-sector@@special_functions@@special_functions/src/debye.rs@@&& sector <= 1.2)@@&& sector <= 3.0)@@the Debye sector guard (Stage 24: 5.0e12 optimism beyond it)
debye-ratio@@special_functions@@special_functions/src/debye.rs@@(ratio >= 8.0 && sector@@(ratio >= 2.0 && sector@@the Debye |z|/|nu| validity threshold
asym-safety@@special_functions@@special_functions/src/bessel_cnu_large.rs@@const SAFETY: f64 = 150.0;@@const SAFETY: f64 = 1.0;@@the truncation safety factor (Stage 18: measured 110x optimistic)
asym-floor@@special_functions@@special_functions/src/bessel_cnu_large.rs@@const FLOOR: f64 = 5e-14;@@const FLOOR: f64 = 1e-300;@@the estimate floor (Stage 16: a zero estimate wins every comparison)
zeta-anchor@@special_functions@@special_functions/src/airy_uniform.rs@@let target = 1.5 * w.arg();@@let target = 1.0 * w.arg();@@the Stage 2D zeta branch anchor
airy-sector@@special_functions@@special_functions/src/airy_uniform.rs@@if !near && x.arg().abs() > 0.8 {@@if !near && x.arg().abs() > 9.9 {@@the Stage 2D Airy sector guard
tm-midpoint@@quantum@@quantum/src/transfer.rs@@(i as f64 + 0.5) * d)))@@(i as f64) * d)))@@midpoint sampling, which makes the transfer matrix 2nd order
tm-guard@@quantum@@quantum/src/transfer.rs@@if e <= v_left.re {@@if false {@@the below-asymptote refusal (fixed in 2B; used to report R = 0)
nash-phase@@quantum@@quantum/src/nash.rs@@coeff.push(power * jm);@@coeff.push(C::real(jm));@@the i^M phase in the Jacobi-Anger stencil coefficients
nash-order@@quantum@@quantum/src/nash.rs@@None => order_for(lambda, f64::EPSILON)?,@@None => 2,@@the automatic Jacobi-Anger truncation order
cap-cells@@quantum@@quantum/src/absorber.rs@@const CELLS_PER_LENGTH: f64 = 200.0;@@const CELLS_PER_LENGTH: f64 = 8.0;@@the absorber's cell resolution
gamma-shift@@special_functions@@special_functions/src/gamma_complex.rs@@const SHIFT_TO: f64 = 14.0;@@const SHIFT_TO: f64 = 2.0;@@the Stirling argument shift (small |w| is where the series is worst)
airy-series@@special_functions@@special_functions/src/airy_complex.rs@@const SERIES_LIMIT: f64 = 6.0;@@const SERIES_LIMIT: f64 = 60.0;@@where complex Airy switches from series to asymptotics
airy-c1@@special_functions@@special_functions/src/airy_complex.rs@@const C1: f64 = 0.355_028_053_887_817_2;@@const C1: f64 = 0.355_028_053_887_8;@@Ai(0), truncated: 4 digits dropped from a defining constant
restitution@@physical_object@@physical_object/src/collide.rs@@rel_vel_n.abs() < system.restitution_threshold@@rel_vel_n.abs() < 1.0e30@@the restitution threshold (every impact becomes plastic)
zeno-window@@physical_object@@physical_object/src/collide.rs@@pub const ZENO_GAP_RELATIVE: f64 = 1e-9;@@pub const ZENO_GAP_RELATIVE: f64 = 1e9;@@the Stage 2C burst window (everything becomes one burst again)
zeno-plastic@@physical_object@@physical_object/src/collide.rs@@pub const MAX_EVENTS_IN_BURST: usize = 64;@@pub const MAX_EVENTS_IN_BURST: usize = 0;@@the Zeno guard's EFFECT: every impact becomes inelastic
zeno-never@@physical_object@@physical_object/src/collide.rs@@pub const MAX_EVENTS_IN_BURST: usize = 64;@@pub const MAX_EVENTS_IN_BURST: usize = usize::MAX;@@the Zeno guard firing at all (a settling ball must still terminate)
TABLE
)

restore() { git checkout -- "$1" 2>/dev/null; }
trap 'git checkout -- special_functions/src quantum/src physical_object/src 2>/dev/null' EXIT

if ! git diff --quiet -- special_functions/src quantum/src physical_object/src; then
  echo "${red}refusing to run:${off} the source tree has uncommitted changes."
  echo "This harness edits sources in place and restores them with git checkout,"
  echo "which would discard your work. Commit or stash first."
  exit 2
fi

total=0; caught=0; survived=0; survivors=""
while IFS= read -r line; do
  [ -z "$line" ] && continue
  IFS='@@' read -r _ <<< "" 2>/dev/null || true
  id=${line%%@@*};       rest=${line#*@@}
  crate=${rest%%@@*};    rest=${rest#*@@}
  file=${rest%%@@*};     rest=${rest#*@@}
  search=${rest%%@@*};   rest=${rest#*@@}
  replace=${rest%%@@*};  what=${rest#*@@}
  [ -z "${id:-}" ] && continue
  case "$id" in \#*) continue;; esac
  if [ -n "$FILTER" ] && [[ "$id" != *"$FILTER"* ]]; then continue; fi
  total=$((total + 1))

  n=$(grep -Fo -- "$search" "$file" 2>/dev/null | wc -l)
  if [ "$n" -ne 1 ]; then
    printf '  %sABORT%s  %-14s the anchor appears %s times in %s (need exactly 1)\n' \
           "$red" "$off" "$id" "$n" "$file"
    printf '         a mutation that lands somewhere unintended tests nothing in particular\n'
    exit 2
  fi

  python3 - "$file" "$search" "$replace" <<'PY'
import sys
path, search, replace = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding='utf-8').read()
open(path, 'w', encoding='utf-8').write(s.replace(search, replace, 1))
PY

  if cargo test -q -p "$crate" --release >/dev/null 2>&1; then
    printf '  %sSURVIVED%s %-14s %s\n' "$red" "$off" "$id" "$what"
    survived=$((survived + 1))
    survivors="$survivors\n    $id ($file) — $what"
  else
    printf '  %scaught%s   %-14s %s\n' "$green" "$off" "$id" "$what"
    caught=$((caught + 1))
  fi
  restore "$file"
done <<< "$MUTATIONS"

echo
echo "$caught/$total mutations caught."
if [ "$survived" -gt 0 ]; then
  printf '%s%s survived%s — each names a line the suite does not constrain:%b\n' \
         "$yellow" "$survived" "$off" "$survivors"
  echo
  echo "A survivor is not automatically a bug. It is a line whose value nothing"
  echo "asserts, which is either a missing test or a constant that turns out not"
  echo "to matter. Deciding which is the point of running this."
fi
exit "$survived"
