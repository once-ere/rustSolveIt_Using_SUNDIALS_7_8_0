#!/usr/bin/env bash
# compare_pristine_c.sh — three-way root-cause for the divergent variants.
#
# A variant that differs from its shipped reference .out is a PORT DEFECT
# only if the Rust output also differs from what the pristine upstream C
# produces on this same host. If Rust == C and both differ from the .out,
# the reference is stale and the port is correct. Nothing else settles it,
# and it has to be measured natively — the macOS sibling made this
# comparison against Apple-clang binaries, which proves nothing here.
#
# For each variant this runs, with identical argv:
#     C   = pristine upstream binary from tools/pristine_c_build.sh
#     RS  = this workspace's release example
#     REF = the shipped .out in the upstream examples/ tree
# and reports three diffs, all through verify_examples.sh's noise filter:
#     RS_vs_C    the verdict. `same` => not a port defect.
#     C_vs_REF   `diff` => the shipped reference is stale on this host.
#     RS_vs_REF  restates the gate result, for context.
#
#   tools/compare_pristine_c.sh [build-dir]
#
# Writes logs/pristine-c-comparison.txt and, for every non-`same` RS_vs_C,
# logs/cdiff-<outfile>.txt.
set -u
cd "$(dirname "$0")/.."
WS_ROOT="$PWD"
UP="$WS_ROOT/.."
CB="${1:-$HOME/sdl/cbuild}"
LOGS="$WS_ROOT/logs"
OUT="$LOGS/pristine-c-comparison.txt"
mkdir -p "$LOGS/cwork"

noise_filter() { grep -v -E 'Total run time|CPU time|cpu time|wall clock' || true; }

# The 26 variants the Linux gate reports as DIFF: solver dir | example |
# argv | reference .out. Group 2 (content differences) first, then
# Group 1 (whitespace-only), so the interesting rows are at the top.
VARIANTS='
cvode|serial|cvPendulum_dns||cvPendulum_dns.out
cvodes|serial|cvsPendulum_dns||cvsPendulum_dns.out
cvodes|serial|cvsKrylovDemo_ls||cvsKrylovDemo_ls.out
cvodes|serial|cvsKrylovDemo_ls|1|cvsKrylovDemo_ls_1.out
cvodes|serial|cvsKrylovDemo_ls|2|cvsKrylovDemo_ls_2.out
cvodes|serial|cvsKrylovDemo_ls|0 1|cvsKrylovDemo_ls_0_1.out
idas|serial|idasAkzoNob_ASAi_dns||idasAkzoNob_ASAi_dns.out
arkode|C_serial|ark_conserved_exp_entropy_ark|1 1|ark_conserved_exp_entropy_ark_1_1.out
arkode|C_serial|ark_dissipated_exp_entropy|1 1|ark_dissipated_exp_entropy_1_1.out
cvode|serial|cvRoberts_dnsL||cvRoberts_dnsL.out
cvodes|serial|cvsRoberts_dnsL||cvsRoberts_dnsL.out
kinsol|serial|kinRoboKin_dns||kinRoboKin_dns.out
cvode|serial|cvRoberts_dns_negsol||cvRoberts_dns_negsol.out
arkode|C_serial|ark_analytic_partitioned|forcing|ark_analytic_partitioned_forcing.out
arkode|C_serial|ark_analytic_partitioned|splitting|ark_analytic_partitioned_splitting.out
arkode|C_serial|ark_analytic_partitioned|splitting ARKODE_SPLITTING_BEST_2_2_2|ark_analytic_partitioned_splitting_ARKODE_SPLITTING_BEST_2_2_2.out
arkode|C_serial|ark_analytic_partitioned|splitting ARKODE_SPLITTING_RUTH_3_3_2|ark_analytic_partitioned_splitting_ARKODE_SPLITTING_RUTH_3_3_2.out
arkode|C_serial|ark_analytic_partitioned|splitting ARKODE_SPLITTING_YOSHIDA_8_6_2|ark_analytic_partitioned_splitting_ARKODE_SPLITTING_YOSHIDA_8_6_2.out
arkode|C_serial|ark_damped_harmonic_symplectic||ark_damped_harmonic_symplectic.out
arkode|C_serial|ark_harmonic_symplectic||ark_harmonic_symplectic.out
arkode|C_serial|ark_reaction_diffusion_mri||ark_reaction_diffusion_mri.out
arkode|C_serial|ark_kepler||ark_kepler.out
arkode|C_serial|ark_kepler|--stepper ERK --step-mode fixed --count-orbits|ark_kepler_--stepper_ERK_--step-mode_fixed_--count-orbits.out
arkode|C_serial|ark_kepler|--stepper SPRK --step-mode fixed --count-orbits --use-compensated-sums|ark_kepler_--stepper_SPRK_--step-mode_fixed_--count-orbits_--use-compensated-sums.out
arkode|C_serial|ark_kepler|--stepper SPRK --step-mode fixed --method ARKODE_SPRK_EULER_1_1 --tf 50 --check-order --nout 1|ark_kepler_--stepper_SPRK_--step-mode_fixed_--method_ARKODE_SPRK_EULER_1_1_--tf_50_--check-order_--nout_1.out
arkode|C_serial|ark_kepler|--stepper SPRK --step-mode fixed --method ARKODE_SPRK_RUTH_3_3 --tf 50 --check-order --nout 1|ark_kepler_--stepper_SPRK_--step-mode_fixed_--method_ARKODE_SPRK_RUTH_3_3_--tf_50_--check-order_--nout_1.out
'

{
  printf '%-46s %-34s %-10s %-10s %s\n' VARIANT ARGS RS_vs_C C_vs_REF RS_vs_REF
  printf '%s\n' '-------------------------------------------------------------------------------------------------------------'
} > "$OUT"

while IFS='|' read -r solver exdir name args outfile; do
  [ -n "${name:-}" ] || continue
  cbin="$CB/examples/$solver/$exdir/$name"
  rbin="$WS_ROOT/target/release/examples/$name"
  ref="$UP/examples/$solver/$exdir/$outfile"
  short="${args:0:33}"

  if [ ! -x "$cbin" ]; then
    printf '%-46s %-34s %s\n' "$name" "$short" "NO-C-BINARY (backend disabled in the C build)" >> "$OUT"
    continue
  fi
  if [ ! -x "$rbin" ]; then
    printf '%-46s %-34s %s\n' "$name" "$short" "NO-RUST-BINARY" >> "$OUT"
    continue
  fi

  # shellcheck disable=SC2086  # argv must word-split exactly as on a command line
  ( cd "$LOGS/cwork" && "$cbin" $args > "$LOGS/cwork/c-$outfile" 2>&1 )
  # shellcheck disable=SC2086
  ( cd "$LOGS/cwork" && "$rbin" $args > "$LOGS/cwork/rs-$outfile" 2>&1 )

  rc=$(diff <(noise_filter < "$LOGS/cwork/rs-$outfile") \
            <(noise_filter < "$LOGS/cwork/c-$outfile")  >/dev/null 2>&1 && echo same || echo DIFF)
  cr=$(diff <(noise_filter < "$LOGS/cwork/c-$outfile")  \
            <(noise_filter < "$ref")                    >/dev/null 2>&1 && echo same || echo DIFF)
  rr=$(diff <(noise_filter < "$LOGS/cwork/rs-$outfile") \
            <(noise_filter < "$ref")                    >/dev/null 2>&1 && echo same || echo DIFF)

  [ "$rc" = same ] || diff <(noise_filter < "$LOGS/cwork/rs-$outfile") \
                           <(noise_filter < "$LOGS/cwork/c-$outfile") \
                           > "$LOGS/cdiff-$outfile.txt" 2>&1

  printf '%-46s %-34s %-10s %-10s %s\n' "$name" "$short" "$rc" "$cr" "$rr" >> "$OUT"
done <<< "$VARIANTS"

cat "$OUT"
