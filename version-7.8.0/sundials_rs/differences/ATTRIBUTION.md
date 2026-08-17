# ATTRIBUTION — why each divergence happens, proved by experiment

`README.md` in this directory reports *how many* variants differ.
This file answers the only question that matters about them: **is the Rust
translation wrong, or is the difference the deliberate pure-Rust libm?**

The answer is not argued. It is measured, by an experiment anyone can
re-run in about a minute.

## The experiment

`crates/sundials_core` carries a cargo feature, `host-libm`, that changes
exactly one thing: the `SunMath` trait's thirteen methods call the host C
library (`f64::sin`, `f64::exp`, …) instead of
`crates/sundials_core/src/sundials_libm.rs`. Every other line of the port
— all 144 translated modules, all 119 translated examples — is byte for
byte the same code in both builds.

So if a variant differs from the C binary in the default build but matches
it in the `host-libm` build, the difference is caused by the libm
substitution and by nothing else.

If it differs in **both**, the switch has told you nothing, and there are
two possibilities. For anything outside the eleven `*_klu` examples that is
a port defect, to be fixed. For a `*_klu` example it is expected: the switch
does not touch the sparse linear solver, so those variants are outside this
experiment's reach altogether and are attributed separately below.

```bash
tools/ab_host_libm.sh
```

## The result

| build | comparable variants | byte-identical to the pristine C |
|---|---:|---:|
| default | 190 | **175** |
| `--features host-libm` | 190 | **183** |

The seven the switch does not explain are `cvRoberts_block_klu`,
`cvRoberts_klu`, `cvsRoberts_ASAi_klu`, `cvsRoberts_FSA_klu`,
`cvsRoberts_klu`, `idasRoberts_ASAi_klu` and `idasRoberts_FSA_klu` — that
is, **exactly the seven `*_klu` examples that still differ**, a second and
separate substitution with its own cause.

**No variant is left unaccounted for.** That is the measurement behind the
claim of **0 port defects** — but the two halves are not equally strong, and
it would be dishonest to present them as if they were. The libm half is a
controlled comparison with one variable. The sparse-LU half has no control
build and rests on direct verification of the replacement solver instead.
One experiment and one argument, not two experiments.

## Two substitutions, two causes

This port replaces two pieces of third-party numerics, for the same
licensing reason, and each shows up in a different set of variants:

| substitution | why | isolated by |
|---|---|---|
| host libm → [`sundials_libm`](../crates/sundials_core/src/sundials_libm.rs) | glibc's `sin`/`cos`/`atan`/`asin`/`acos` are LGPL | `--features host-libm` |
| SuiteSparse KLU → [`sundials_sparse_lu`](../crates/sundials_core/src/sundials_sparse_lu.rs) | KLU is LGPL, and FFI is forbidden | only affects the 11 `*_klu` examples |

There is no `host-klu` control build, because there is nothing to switch
to: KLU cannot be linked at all under this port's rules. What stands in for
it is direct verification of the replacement — the sparse LU is checked
against dense Gaussian elimination on 300 random systems (worst relative
residual 7.3e-16), and `idaHeat2D_klu`'s hand-packed Jacobian is checked
entry by entry against an independently constructed reference and against
finite differences of the residual.

Four of the eleven `*_klu` examples match the C **byte for byte** anyway:
`idaHeat2D_klu`, `idaRoberts_klu`, `idasRoberts_klu` and `kinFerTron_klu`.

### The pivoting rule was not a free choice

The sparse LU originally used pure partial pivoting — largest magnitude
wins. `idaHeat2D_klu` exposed why that is wrong here. Its boundary
equations are literally `e_i`: a unit diagonal and nothing else. Pure
partial pivoting discards that `1` in favour of a neighbouring `-1/dx^2`,
which mixes the boundary and interior unknowns and lets round-off into
components the problem pins exactly. The solution then grew without bound
where the C decays to zero — a qualitative divergence, not a last-bit one.
(The pre-fix run was observed during development at `umax` on the order of
1e+04 while the C ends near zero; that capture was not committed, so treat
the magnitude as anecdote and the sign of the effect — divergence rather
than decay — as the claim. The post-fix state is committed and checkable:
`idaHeat2D_klu` is byte-identical.)

Switching to KLU's documented default, threshold partial pivoting with a
diagonal preference at `tol = 0.001`, fixed it, and made two *further*
variants byte-identical — `idaRoberts_klu` and `idasRoberts_klu`. The lesson is worth recording: for these matrices
the pivoting rule is output-critical, and matching KLU's was the faithful
choice rather than the merely defensible one.

Raw data: [`ab-host-libm.tsv`](ab-host-libm.tsv), one row per variant, with
the default-build class and the host-libm-build class side by side.

## The variants the libm accounts for

These differ in the default build and match under `--features host-libm`:

| variant | elementary functions on its hot path |
|---|---|
| `cvodes/serial/cvsDiurnal_FSA_kry__-sensi_sim_t` | `sin` in the diurnal source term |
| `idas/serial/idasSlCrank_dns` | `sin`, `cos` in the crank geometry |
| `idas/serial/idasSlCrank_FSA_dns` | same |
| `arkode/C_serial/ark_analytic_lsrk` | `sinh`, `cosh`, `acosh`, `log` (LSRK stage-count formula) |
| `arkode/C_serial/ark_analytic_lsrk_varjac` | same |
| `arkode/C_serial/ark_analytic_lsrk_domeigest` | same |
| `arkode/C_serial/ark_analytic_lsrk_domeigest__arkid.dom_eig_est_init_preprocess_iters_1_sundomeigestimator.max_iters_1` | same |
| `arkode/C_serial/ark_kpr_mri__10_4_0.001_-100_100_0.5_1` | `sin`, `cos` in the reference solution |

The ARKODE **LSRK** entries are not a coincidence and were predictable.
`sinh`, `cosh`, `acosh` and `ln` are reached from exactly one module in the
whole library: the wrappers `SUNRsinh`, `SUNRcosh`, `SUNRacosh` and
`SUNRlog` are *defined* at
[`crates/arkode_rs/src/arkode_lsrkstep.rs:83-98`](../crates/arkode_rs/src/arkode_lsrkstep.rs:83)
and *called* from two sites in that same file,
[`:1158`](../crates/arkode_rs/src/arkode_lsrkstep.rs:1158) and
[`:3255`](../crates/arkode_rs/src/arkode_lsrkstep.rs:3255). The first feeds
the formula that chooses the *number of stages*: a last-bit difference there
changes an integer, which changes the method, which changes everything
downstream.

```bash
grep -rn 'sun_sinh()\|sun_cosh()\|sun_acosh()\|sun_ln()' ../crates/ --include=*.rs \
  | grep -v sundials_libm.rs | grep -v /examples/
```

The sibling Linux repository saw **three of these four** LSRK variants break
on Arch's glibc 2.44 for the same reason — `ark_analytic_lsrk_domeigest`
(both argv variants) and `ark_analytic_lsrk_varjac`, but not
`ark_analytic_lsrk`. glibc changed `sinh`, `cosh` and `acosh` between 2.41
and 2.44.

The rest is ordinary adaptive-integrator chaos: a one-ulp difference in
`sin` inside a right-hand side moves an error estimate, which moves a
step-size decision, which moves the whole trajectory. Note what the
magnitudes actually are — `idasSlCrank_dns` differs in the 13th
significant digit of one printed number, and `ark_kpr_mri` in the third
digit of one error estimate on one of 74 lines.

## Which side is closer to the true answer?

The pure-Rust one, measurably. `tools/libm_differential.sh` measures every
function against a 113-bit `__float128` reference (see
[`../LIBM.md`](../LIBM.md)):

Every figure below is read off [`../logs/libm_differential.log`](../logs/libm_differential.log),
which records 1,000,000 samples per (function, corpus):

| function | pure-Rust max error | host glibc 2.43 max error |
|---|---|---|
| `sin`, `cos`, `atan`, `asin`, `acos` | 0.5000 ulp | 0.5042 – 0.5186 ulp |
| `expm1`, `log1p` | 0.5000 ulp | 0.7783 – 0.8414 ulp |
| `sinh`, `cosh` | 0.5000 ulp | 0.9883 – 1.7848 ulp |
| `exp`, `log` | 0.5003 – 0.5071 ulp | identical, bit for bit — same source |
| `acosh` | 0.5000 ulp | 0.5000 ulp — independent implementations that agree because both are correctly rounded |

0.5000 ulp is correct rounding — the smallest error a binary64 result can
have. So on these eight variants the C column is not a target the Rust
column failed to hit; it is the less accurate of the two. `exp` and `log`
are the ARM optimized-routines kernels glibc itself ships, so they are not
correctly rounded and are not expected to be: they match the host exactly.

## What would count as a defect

A row of `ab-host-libm.tsv` whose `host_libm_class` is `DIFFERS` **and
whose example is not one of the eleven `*_klu`**. The `*_klu` rows differ
under both builds by construction, because the `host-libm` switch does not
touch the sparse LU — there is no KLU to switch back to. They are covered
instead by the direct verification described above.

On that criterion there are currently no defects. `tools/ab_host_libm.sh`
prints every still-divergent variant under the heading "variants that remain
divergent even with the host libm (**real port defects**)" — that heading
predates the sparse LU and is now too strong: the list it prints is exactly
the seven `*_klu` variants, none of which is a defect. Read it as "variants
this experiment cannot attribute". A new **non-`klu`** entry there must be
fixed before the change lands; see `../CLAUDE.md` § "Classifying a
divergence", which carries the same carve-out.
