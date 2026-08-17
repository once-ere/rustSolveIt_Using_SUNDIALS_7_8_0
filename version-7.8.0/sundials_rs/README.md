# SUNDIALS_7_8_Rust_port_for_Linux

A line-by-line translation of [SUNDIALS](https://github.com/LLNL/sundials)
7.8.0 into safe Rust. **No `unsafe`, no FFI, no external crates, no build
warnings** — and **no host C library on the numerical path**: every
elementary function, and the sparse direct solver, is implemented here in
pure Rust.

Acceptance is byte-identical printed output against the upstream C examples,
**established on Linux running on Intel/AMD x86-64 with glibc.**

## Platform scope: Linux on x86-64, glibc

> The `.out` reference gate was measured on **Ubuntu 24.04 x86-64, glibc
> 2.39, gcc 13.3.0, rustc 1.93.1**, on a CPU with FMA, and **re-run natively
> on Debian 12 (glibc 2.36), Fedora 41 (glibc 2.40) and Arch (glibc 2.44)** —
> see [Distribution coverage](#distribution-coverage--measured-not-argued),
> which reports the one distribution where three variants differed under the
> host libm, and why that no longer happens. The
> C-versus-Rust measurement in
> [`differences/`](differences/) (with [`c-results/`](c-results/) and
> [`rust-results/`](rust-results/)) is a
> second, later run on **Ubuntu 26.04, glibc 2.43, gcc 15.2.0, rustc 1.96.1**.
> The reference gate has since been re-run under the pure-Rust libm on **four
> glibc versions, on musl, and on aarch64 under emulation**, all giving the
> same result; **arm64 hardware and Windows remain untested.**
>
> **What is still platform-bound, and what no longer is.** This used to be a
> glibc-shaped port: the library evaluated `sin`, `cos`, `exp`, `ln` and the
> rest through `f64` methods that Rust `std` documents as having *unspecified
> precision*, which forward to whatever libm the binary links. On glibc those
> calls landed on the very implementation the upstream `.out` files came from,
> which is why this was the *good* platform — and why the sibling
> [macOS/Apple-Silicon port](https://github.com/once-ere/SUNDIALS_7_8_Rust_port_for_AppleSilicon_macos)
> had to document 52 variants away as Apple-libm divergence while **26 of
> those are byte-identical here.**
>
> That dependence is gone. All thirteen functions now resolve to
> [`sundials_libm.rs`](crates/sundials_core/src/sundials_libm.rs), so **the
> Rust output is a function of this repository's own source**, not of the
> host's patch level. What remains platform-bound is the *comparison*: the
> shipped `.out` files, and the C binaries built here to check against. The
> port runs the same everywhere; the references do not.

## Headline facts

* 7 crates: `sundials_core` plus `cvode_rs`, `cvodes_rs`, `kinsol_rs`,
  `ida_rs`, `idas_rs`, `arkode_rs`. Solver crates depend on the core, never on
  each other.
* 144 modules, one per upstream C file, and 119 examples, keeping the exact C
  function names, constants and return-flag conventions (`CV_SUCCESS = 0`;
  negative fatal, positive recoverable).
* Serial only. No MPI, GPU, SuperLU, LAPACK, Fortran or XBraid backends —
  **KLU is no longer on that list**, see below.
* `cargo build --workspace` → **zero warnings**. `cargo test --workspace
  --lib` → **39 passed**.
* Thirteen elementary functions in pure Rust. Ten are **correctly rounded
  (0.5000 ulp)** against a 113-bit `__float128` reference, where the host
  glibc 2.43 reaches 0.5042 – 1.7848 ulp; `exp`, `log` and `acosh` are
  bit-identical to it. See [`LIBM.md`](LIBM.md).
* Deterministic `pow` vs the **native glibc `pow`**: **0 mismatches over
  5,900,000 domain inputs and 0 over 20,000,000 unrestricted finite inputs.**
* Two gates, both over the same 199 `(example, argv)` variants — see
  [Two gates](#two-gates-and-why-both):
  * **vs the shipped `.out` references:** **145 identical, 34
    reference-side, 20 excluded, 0 port defects** as the tree stands. It was
    153 / 26 / 20 before the pure-Rust libm; the eight that moved are the
    eight the libm accounts for, and the score is now the same on every
    glibc tested rather than varying by host.
  * **vs the upstream C rebuilt on the same machine:** 175 of 190 comparable
    identical; the 15 that differ decompose exactly into 8 libm + 7 sparse LU
    with **0 unaccounted for**.

## Quick start

```bash
cargo build --workspace
cargo run -p cvode_rs --example cvRoberts_dns
tools/pow_differential.sh all
```

```rust
use cvode_rs::prelude::*;
```

The upstream example tree is **vendored** under `examples/`, so the reference
gate needs nothing else (it still falls back to a parent-directory tree if one
is there):

```bash
tools/verify_examples.sh all      # vs the shipped .out references
python3 tools/cross_gate.py       # both gates, cross-tabulated
```

## Relationship to the macOS port

**This repository reuses the macOS port's crate tree wholesale.** The solver
translation was not re-derived here, and `ARCHITECTURE.md` and `PROGRESS.md`
are inherited unchanged because they describe the translation, which is
platform-independent. What is new is the target-platform work — a native
x86-64 `pow` oracle and differential, a re-run of the whole verification gate
under glibc, documentation scoped to Linux — and then the two substitutions
that removed the host C library from the numerical path entirely, which grew
the tree to 144 modules and 119 examples.

## The `pow` question

`crates/sundials_core/src/sundials_math.rs` contains `pow_glibc`, a pure-Rust
port of the ARM optimized-routines `pow` (MIT, © 2018 Arm Limited, taken via
musl). The "ARM" in that provenance is a red herring for a Linux/x86-64
target: **that algorithm is what glibc ≥ 2.28 ships** as
`sysdeps/ieee754/dbl-64/e_pow.c`, and on x86-64 glibc ifunc-dispatches to
`__ieee754_pow_fma` — the same source rebuilt with `-mfma -mavx2
-ffp-contract=fast`. Reproducing *that* build, contraction site for
contraction site, is exactly what the Rust routine does.

What was missing was evidence gathered on x86-64. The macOS project measured
against oracle binaries built on arm64 and said so
([`POW_FMA_EXACTNESS.md`](POW_FMA_EXACTNESS.md) §5: "No differential run was
made on a native x86-64 host"). This repository supplies it:

| artefact | role |
|---|---|
| [`tools/pow_oracle.c`](tools/pow_oracle.c) | built with the host `cc`, calls the host `pow`, emits the reference bit-stream |
| [`tools/pow_differential.sh`](tools/pow_differential.sh) | driver; writes `logs/pow_differential.log` |
| `pow_glibc_vs_native_oracle_{domain,random}` | the Rust side, in `sundials_math.rs` |

Both sides regenerate the corpus from the same splitmix64 recurrence rather
than exchanging inputs, so they cannot disagree about what they evaluated.
The result on glibc 2.39 / x86-64 is **0 mismatches on both corpora** — the
two residual 1-ulp disagreements the macOS project could not eliminate are
absent against a native oracle, which is precisely the doubt that document
raised. No new `pow` source was written, because writing one would have
replaced a routine already bit-exact against the target with an unmeasured
rewrite.

`pow` was the *first* libm substitution and for a long time the only one.
It is now one of thirteen — see [The pure-Rust libm](#the-pure-rust-libm) —
and it remains the one with the strongest claim, because it is measured
bit-for-bit against the host rather than against a 113-bit reference.

Not every host-C-library dependence was a libm one:
`ark_analytic_lsrk_domeigest`, `ark_brusselator_lsrk_domeigest` and
`ark_brusselator_lsrk_externaldomeigest` reproduce the BSD/glibc `rand()`
TYPE_3 additive-feedback generator in Rust, sequence for sequence, because
those examples feed pseudo-random vectors into a dominant-eigenvalue estimator
and the draws are output-observable. See [`NOTICE`](NOTICE).

## The pure-Rust libm

[`crates/sundials_core/src/sundials_libm.rs`](crates/sundials_core/src/sundials_libm.rs)
implements `exp`, `log`, `pow`, `expm1`, `log1p`, `sin`, `cos`, `atan`,
`asin`, `acos`, `sinh`, `cosh` and `acosh`. `exp`/`log`/`pow` are the ARM
optimized-routines kernels taken via musl (MIT) — the same source glibc ≥ 2.28
ships. The other ten are written here, on a double-double core and an exact
Payne–Hanek argument reduction over the bits of 2/π; their constant tables are
generated by `tools/gen_libm_constants.py` from Python's stdlib `decimal`, so
they are reproducible rather than pasted.

Call sites are spelled `x.sun_sin()`, `x.sun_exp()`, … through a `SunMath`
trait. That rename is what makes the host unreachable: Rust resolves inherent
methods before trait ones, so `x.sin()` could not have been redirected. The
only `f64` methods left anywhere in `crates/` are `sqrt`, `mul_add`, `abs`,
`ceil`, `round` and `copysign`, all IEEE-754 exact.

`tools/libm_differential.sh` measures each function against a 113-bit
`__float128` reference over 1,000,000 samples per corpus:

| function | pure Rust | host glibc 2.43 |
|---|---:|---:|
| `exp`, `log` | 0.5003 – 0.5071 ulp | identical, bit for bit (same source) |
| `acosh` | 0.5000 ulp | 0.5000 ulp (independent, both correctly rounded) |
| `sin`, `cos`, `atan`, `asin`, `acos` | **0.5000 ulp** | 0.5042 – 0.5186 ulp |
| `expm1`, `log1p` | **0.5000 ulp** | 0.7783 – 0.8414 ulp |
| `sinh`, `cosh` | **0.5000 ulp** | 0.9883 – 1.7848 ulp |

0.5000 ulp is correct rounding. Where the two disagree, the pure-Rust answer
is the right one. **These functions are deliberately not bit-identical to
glibc and cannot be**: glibc implements them with the LGPL IBM Accurate
Mathematical Library. The alternative was an fdlibm clone at ~0.55 ulp, which
would have disagreed with glibc *more often*.

A `host-libm` cargo feature (default off) switches the thirteen methods back
to the host. It exists for `tools/ab_host_libm.sh`, and it is what keeps
"0 port defects" a measurement rather than a claim. Never enable it for a
production build.

## The sparse LU that replaced KLU

SuiteSparse KLU is LGPL-2.1+, so it could not be translated into this
BSD-3-Clause tree and could not be called either, since the port forbids FFI.
That is why the eleven `*_klu` examples were unportable.
[`sundials_sparse_lu.rs`](crates/sundials_core/src/sundials_sparse_lu.rs)
replaces it: a left-looking sparse LU (Gilbert & Peierls, SIAM J. Sci. Stat.
Comput. 9(5), 1988) under a faithful translation of SUNDIALS' own BSD-3
`sunlinsol_klu.c`. Nothing is derived from KLU, CSparse or any SuiteSparse
source. **All 11 `*_klu` examples are ported; 4 are byte-identical to the C.**

It pivots the way KLU documents its default — threshold partial pivoting with
a diagonal preference at `tol = 0.001` — and that was not a stylistic choice.
`idaHeat2D_klu`'s boundary equations are literally `e_i`, a unit diagonal and
nothing else; largest-magnitude pivoting discards that `1` for a neighbouring
`-1/dx²`, mixes the boundary and interior unknowns, and lets round-off into
components the problem pins exactly. The run diverged where the C decays to
zero. Matching KLU's rule fixed it and made two further variants
byte-identical.

Unlike the libm, this substitution has **no control build** — there is no KLU
to switch back to. It is verified directly instead: against dense Gaussian
elimination on 300 random sparse systems (worst relative residual 7.3e-16),
and, for `idaHeat2D_klu`, its hand-packed CSC Jacobian checked entry by entry
against an independent reference and against finite differences of the
residual (`cargo test -p ida_rs --example idaHeat2D_klu`).

## Verification results

Both columns are **host-libm** measurements — the configuration in which the
macOS comparison was made. The current tree scores 145 / 34 / 20; see
[`evidence/purerust-libm-gate/`](evidence/purerust-libm-gate/).

| | macOS / arm64 (inherited) | **Linux / x86-64 (here)** |
|---|---:|---:|
| IDENTICAL | 127 | **153** |
| divergent, reference-side | 52 | **26** |
| excluded (KLU/SuperLU) | 20 | 20 |
| port defects | 0 | **0** |

**"0 port defects" is measured, not asserted.** A divergence from a shipped
`.out` is a port defect only if the Rust output also differs from what the
pristine upstream C produces on the same machine — so the upstream C library
and its serial examples were built here with cmake + gcc 13.3.0
([`tools/pristine_c_build.sh`](tools/pristine_c_build.sh)) and every
divergent variant run three ways
([`tools/compare_pristine_c.sh`](tools/compare_pristine_c.sh)):

| comparison | result across all 26 |
|---|---|
| **Rust vs pristine C** | **`same` — 26 / 26** |
| pristine C vs shipped `.out` | `DIFF` — 26 / 26 |
| Rust vs shipped `.out` | `DIFF` — 26 / 26 (the gate result) |

The C and the Rust agree with each other and disagree with the shipped
reference, every time: the references are stale, the translation is not
wrong anywhere. The two LAPACK examples are absent from a pristine
`ENABLE_LAPACK=OFF` build, so
[`tools/compare_lapack_substituted.sh`](tools/compare_lapack_substituted.sh)
compiles them with exactly the two tokens the port also substitutes; both
also come out `same`.

Secondarily, [`tools/classify_diffs.sh`](tools/classify_diffs.sh) shows
**15 of the 26 are whitespace-only** — `tr -s ' '` makes the diff empty, so
every printed *value* is byte-identical and only column spacing differs
(references predating the `SUN_TABLE_WIDTH` 28 → 29 change). The other 11
have real content differences, all reference-side and each root-caused in
[`VERIFICATION.md`](VERIFICATION.md): two LAPACK→native dense variants
(`cv[s]Roberts_dnsL`), two upstream `.out` anomalies (`cv[s]Pendulum_dns`),
five trailing-whitespace-stripped references (`cvsKrylovDemo_ls` ×4,
`idasAkzoNob_ASAi_dns`), and two references missing a final blank line the
source prints unconditionally.

## Two gates, and why both

The section above is one gate. There is a second, and they are easy to
confuse because both cover the same 199 `(example, argv)` variants and both
report a count of byte-identical outputs — 145 and 175. **175 is not a
correction of 145.** They compare Rust against different things:

| | **vs the shipped `.out`** | **vs C rebuilt here** |
|---|---|---|
| reference | the files inside SUNDIALS 7.8.0 | the upstream C compiled from source, minutes apart |
| machine | Ubuntu 24.04, glibc 2.39, host libm | Ubuntu 26.04, glibc 2.43, pure-Rust libm |
| KLU examples | 20 excluded with SuperLU | 11 ported and compared, 9 SuperLU still out |
| identical | **145** of 199 (153 before the pure-Rust libm) | **175** of 199 |
| where | [`VERIFICATION.md`](VERIFICATION.md), [`evidence/linux-x86_64-glibc239/`](evidence/linux-x86_64-glibc239/) | [`differences/`](differences/), [`c-results/`](c-results/), [`rust-results/`](rust-results/) |

The first asks whether the port reproduces the *published* reference —
external, unfakeable, and it charges the port for a decade of libm drift. The
second asks whether the translation agrees with the C it was translated from,
machine held fixed — it cannot be blamed for reference drift, but its
reference is one this project built. Neither supersedes the other.

```bash
python3 tools/cross_gate.py
```

asserts the variant sets are equal and prints the cross-tabulation. Three
things fall out that neither gate shows alone:

1. **All 26** variants that differ from the shipped `.out` are byte-identical
   to pristine C rebuilt here — "0 port defects" reproduced on a second
   distribution, glibc, compiler and rustc, against a reference *built* rather
   than downloaded.
2. The 15 divergences in the second gate **decompose exactly**: 8 to the libm
   (the `host-libm` control build restores every one), 7 to the sparse LU
   (all `*_klu`, no control build possible), **0 unaccounted for**.
3. The 8 the control build names are *precisely* the 8 that flipped from
   IDENTICAL under the first gate. Two experiments sharing nothing but the
   source tree single out the same set — which is what turns the libm
   attribution from an explanation into a measurement.

The raw captures for the second gate — every `.stdout`, `.stderr`, `.meta`
and SHA-256, plus a unified diff for each divergent variant — live at the
repository root in [`c-results/`](c-results/), [`rust-results/`](rust-results/)
and [`differences/`](differences/), with
[`requirements.md`](requirements.md) recording which optional C backends the
machine could reach. The pipeline has been re-run from source four times and
every capture **in the compared set** came back byte-identical each time.

Outside the compared set, three kinds of capture do move, and knowing which is
the difference between reading a `git diff` and being alarmed by one:

| what moves | why | in the compared set? |
|---|---|---|
| up to 6 `*_omp` `.stdout` | OpenMP reduction order changes the numbers themselves | no |
| `kin_diagon_kry_f2003` `.stdout` | four MPI ranks interleave 47 identical lines differently | no |
| 63 MPI `.stderr` | hwloc 2.13.0 mis-reads this CPU's hybrid core layout and every MPI example inherits the complaint | no |

The last of those appeared between two otherwise identical runs and moved 63
files at once, which looks far worse than it is.
[`c-results/README.md`](c-results/README.md) derives the count from the
captures and gives the check: none is in the compared set,
[`tools/compare_results.py`](tools/compare_results.py) opens only `.stdout`,
and all 337 runs still exit 0.

## Distribution coverage — measured, not argued

Nothing in the Rust tree is distribution-specific: `std` only, no
`cfg(target_os)`, no `cfg(target_arch)`, no build script, no system library
beyond what `std` itself links. The only distribution-visible dependency is
the libm behind `f64`'s transcendental methods.

The tempting argument is "Debian, Arch and Fedora all ship glibc, so the
claim carries to all of them." **That argument is wrong, and measuring it
is what showed so.** glibc's libm is not frozen across releases.
[`tools/glibc_sweep.sh`](tools/glibc_sweep.sh) fingerprints every function
the port reaches — an FNV-1a hash over 1,000,000 deterministic inputs each,
via [`tools/libm_probe.c`](tools/libm_probe.c) — in each distribution's
container:

| distro | libc | functions disagreeing with the reference host (glibc 2.39) |
|---|---|---|
| Debian 12 | glibc 2.36 | `atan` |
| **Ubuntu 24.04** | **glibc 2.39** | — (reference host) |
| Fedora 41 | glibc 2.40 | none |
| Debian 13 | glibc 2.41 | none |
| Arch (rolling) | glibc 2.44 | `sinh`, `cosh`, `acosh` |
| Alpine 3.20 | musl | everything except `sqrt` — including `pow` |

`pow` is bit-identical across every glibc version tested, so the
deterministic `pow` result carries to all of them. `sqrt` matches
everywhere, as IEEE-754 requires.

Then [`tools/gate_in_container.sh`](tools/gate_in_container.sh) ran the
**full 199-variant gate natively inside three of those containers** to find
out whether the libm differences are output-observable:

All four rows below are **host-libm** measurements, kept as the historical
baseline. They are the reason the pure-Rust libm exists: the score moves with
the host. Under the pure-Rust libm all four give 145 / 34 / 20 instead — see
the note after the table.

| distro | libc | rustc | gate (host libm) | vs. the reference host |
|---|---|---|---|---|
| Ubuntu 24.04 | 2.39 | 1.93.1 | **153 / 26 / 20** | reference |
| Debian 12 | 2.36 | 1.97.1 | **153 / 26 / 20** | identical variant set |
| Fedora 41 | 2.40 | 1.97.1 | **153 / 26 / 20** | identical variant set |
| Arch | 2.44 | 1.97.1 | **150 / 29 / 20** | +3 variants diverge |

(`IDENTICAL / DIFF / EXCLUDED`; 0 build failures and 0 run failures
everywhere. The containers also used a *newer* rustc than the host, so the
result is toolchain-stable as well as distribution-stable.)

**Conclusion.** **Verified coverage under the host libm: glibc 2.36 through 2.41** — but the
four distributions were not equally verified, and this used to be blurred.
The gate was actually *run* on Debian 12, Ubuntu 24.04, Fedora 41 and Arch.
**Debian 13 was only fingerprinted** by `tools/glibc_sweep.sh`, which found
its 2.41 libm matching 2.39; there is no `gate-debian-13.txt` in
`evidence/linux-x86_64-glibc239/`. A fingerprint match is a prediction that
nothing is output-observable, not a measurement that nothing is — which is
the entire reason `gate_in_container.sh` exists. Under the pure-Rust libm
Debian 13 *has* now been gate-run, so glibc 2.41 is properly covered for the
current build.

Debian 12's `atan` difference is real but not output-observable: nothing in
the 199 variants evaluates `atan` where 2.36 and 2.39 disagree.

On **Arch (glibc 2.44)** exactly three more variants diverge —
`ark_analytic_lsrk_domeigest` (both argv variants) and
`ark_analytic_lsrk_varjac`. This was predicted before the gate was run and
then confirmed by it: `sinh`, `cosh` and `acosh` are reached from exactly one
module in the library — the wrappers are defined at
[`arkode_lsrkstep.rs:83-98`](crates/arkode_rs/src/arkode_lsrkstep.rs:83) and
used from two sites in that same file — and glibc 2.44 changed all three.

**This whole section describes host-libm behaviour, and the gate has since
been re-run without it — on four hosts.** Under the pure-Rust libm, glibc
**2.36 (Debian 12), 2.40 (Fedora 41), 2.41 (Debian 13), 2.43 (Ubuntu 26.04)
and 2.44 (Arch)** plus **musl 1.2.5 (Alpine)** and **aarch64 under
emulation** all score **145 / 34 / 20**, and not merely the same tally: the
same 34 variants, name for name, with byte-identical DIFF lists — seven hosts,
two libcs, two CPU architectures. Arch is no longer an outlier because there
is no outlier — the score no longer depends on the host. Two rustc versions
are covered as well.

The cost is that 153 became 145 on all four: the eight variants that flipped
are exactly the eight attributed to the libm, the three former Arch ones among
them. Host-independence was bought with eight reference matches, not with
reference agreement. See
[`evidence/purerust-libm-gate/`](evidence/purerust-libm-gate/). Everything else — including the other
three LSRK variants — is unaffected. This is a libm-version effect, not a
port defect; running the port on Arch is fine, but three reference outputs
will not reproduce byte-for-byte there.

**musl is no longer out of scope, and the sweep row above is exactly why the
old exclusion is void.** That row says Alpine's libm disagrees with glibc on
everything except `sqrt` — which mattered enormously when the port called the
host for `sin`, `cos`, `exp`, `log` and the rest, and matters not at all now
that it calls none of them. Measured: **Alpine 3.20.10 / musl 1.2.5 scores
145 / 34 / 20, with a DIFF list byte-identical to all four glibc hosts.** The
port is libc-independent, not merely glibc-version-independent.

Two limits, so this is not read as more than it is. Only the reference gate
ran on Alpine — there is no C toolchain build there, so the C-versus-Rust
comparison and the libm differential were not repeated on musl. And it is
**arm64 is measured now, under emulation.** `tools/gate_in_container.sh
--platform linux/arm64 debian:13` gives **145 / 34 / 20** with a DIFF list
byte-identical to the same image on x86-64 — same 34 variants, same order,
same digest. The log header records `aarch64 [EMULATED]`, because it ran under
QEMU user-mode on this x86-64 host rather than on arm64 silicon.

How far to trust that: QEMU implements `sqrt` and fused multiply-add to the
IEEE-754 specification, which pins them exactly, and those plus integer
arithmetic are what the pure-Rust libm is built on — so a faithful emulator
and real hardware should agree, and the emulator agrees with x86-64. It cannot
exclude a genuine aarch64 codegen difference reproduced identically by QEMU.
Strong corroboration of architecture-independence; not a substitute for a run
on an arm64 machine.

Compilation for aarch64 is separately clean and needs no emulation:
`cargo check --target aarch64-unknown-linux-{gnu,musl} --workspace
--all-targets` gives 0 errors and 0 warnings across 7 crates and 119 example
targets.

## Documentation

| file | contents |
|---|---|
| [`current_status.md`](current_status.md) | **start here** — what is done, what is measured, what remains, how to resume |
| [`sundials.md`](sundials.md) | public guide — crate map, worked example, C-to-Rust API conventions |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | handle model, locked porting patterns, numbered deviation classes (inherited) |
| [`VERIFICATION.md`](VERIFICATION.md) | per-variant matrix; Linux results at the top, inherited macOS evidence below |
| [`PROGRESS.md`](PROGRESS.md) | per-file port status (inherited) |
| [`LIBM.md`](LIBM.md) | the thirteen pure-Rust elementary functions: algorithms, provenance, measured accuracy |
| [`POW_FMA_EXACTNESS.md`](POW_FMA_EXACTNESS.md) | how far the deterministic `pow` is bit-exact, and on which host that was measured |
| [`c-results/`](c-results/) | every upstream C example built and run on the gate-B host, with raw captures and SHA-256 |
| [`rust-results/`](rust-results/) | every ported Rust example, same layout |
| [`differences/`](differences/) | the comparison, variant by variant, plus [`ATTRIBUTION.md`](differences/ATTRIBUTION.md) — the experiment behind "0 port defects" |
| [`requirements.md`](requirements.md) | which optional C backends the gate-B machine had, which it could not use, and why |
| [`CLAUDE.md`](CLAUDE.md) | workspace rules for future work in this repo |

## Licence

Derivative work of SUNDIALS, **BSD-3-Clause**, Copyright © 2002–2026 Lawrence
Livermore National Security, Southern Methodist University, University of
Maryland Baltimore County and the SUNDIALS contributors.

`exp`, `log` and `pow` — in `crates/sundials_core/src/sundials_libm.rs`, and
the deterministic `pow` in `sundials_math.rs` — are pure-Rust ports of the ARM
optimized-routines kernels taken via musl's `src/math/`, **MIT**, Copyright ©
2018 Arm Limited; that is the algorithm glibc ≥ 2.28 ships on this platform.
The double-double core and the ten routines built on it are original to this
project.

`sundials_sparse_lu.rs` is an independent implementation of the Gilbert–
Peierls algorithm from the published literature, wired up through a faithful
translation of SUNDIALS' own BSD-3-Clause `sunlinsol_klu.c`. **No SuiteSparse
code — KLU, BTF, AMD or CSparse — was read, copied or derived from.**

Not an LLNL product; not endorsed by the SUNDIALS project. See `sundials.md`
§8 and `NOTICE`.
