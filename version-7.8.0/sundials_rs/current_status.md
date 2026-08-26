# current_status.md — SUNDIALS_7_8_Rust_port_for_Linux

**Status: the port is complete and green on Linux / glibc / x86-64, and the
host C library is no longer on its numerical path at all.**
Last updated 2026-08-12. Read this file first when resuming.

| gate | result |
|---|---|
| `cargo build --workspace` (Linux x86-64) | **0 errors, 0 warnings**, with and without `--features host-libm` |
| `cargo test --workspace --lib` | **39 passed, 0 failed** |
| deterministic `pow` vs **native glibc `pow`**, SUNDIALS domain corpus | **5,900,000 inputs, 0 mismatches** |
| deterministic `pow` vs **native glibc `pow`**, unrestricted corpus | **20,000,000 inputs, 0 mismatches** |
| pure-Rust libm vs a 113-bit `__float128` reference | **ten routines correctly rounded (0.5000 ulp)**; host glibc 2.43 reaches 0.5042 – 1.7848 |
| **gate A** — `tools/verify_examples.sh all`, vs the shipped `.out` references | **145 IDENTICAL / 34 reference-side / 20 excluded**, identical on seven hosts spanning glibc 2.36–2.44, musl, and aarch64 under emulation (§4). Was 153 / 26 / 20 under the host libm |
| port defects among those 26 | **0 — proven twice**, on two hosts and two glibc versions (§3) |
| **gate B** — vs the upstream C rebuilt on the same machine | **175 of 190 comparable identical**; the 15 that differ are 8 libm + 7 sparse LU, **0 unaccounted for** (§3a) |

Two hosts are involved, and every figure above belongs to one of them:

* **gate A**, the `pow` differential and the distribution sweep — Ubuntu
  24.04 x86-64, glibc 2.39, gcc 13.3.0, rustc 1.93.1, **a WSL2 guest on
  Windows 11**. Artefacts: [`evidence/linux-x86_64-glibc239/`](evidence/linux-x86_64-glibc239/).
* **gate B**, the libm differential and everything after it — Ubuntu 26.04
  x86-64, glibc 2.43, gcc 15.2.0, rustc 1.96.1, 24 cores, **bare metal**
  (`systemd-detect-virt` → `none`). Artefacts:
  [`c-results/`](c-results/), [`rust-results/`](rust-results/),
  [`differences/`](differences/) and [`requirements.md`](requirements.md), at
  the repository root.

**Why the two are filed differently**, since it looks inconsistent and the
reason is not guessable. Gate A's artefacts are a handful of summary logs, so
a host slug (`evidence/linux-x86_64-glibc239/`) is a cheap way to say which
machine they describe. Gate B's are 1,664 files of captured process output
whose own documents link outward — `../requirements.md`, `../LIBM.md`,
`../tools/compare_results.py`, `../crates/...` — and those links are written
for a directory sitting at a repository root, which is where the pipeline
writes them. Nesting them two levels deeper broke all of it and forced a
rewriting step in `tools/vendor_evidence.sh`. At root depth nothing needs
rewriting, the vendored copy is byte-identical to what the pipeline produces,
and the host is recorded where it belongs anyway: the provenance table at the
top of each directory's `README.md`.

> ### Published
>
> `main` is on
> `https://github.com/once-ere/SUNDIALS_7_8_Rust_port_for_Linux.git`.
> Work lands **directly on `main`** — no feature branch, no PR to review —
> confirmed as the preference on 2026-08-12. The one exception so far was
> PR #1 (the libm and sparse LU, ~350k lines), merged as `299a697`; a change
> that large is worth a diff on GitHub, and the branch was deleted after.
>
> The credential note that used to live here applied to the **Windows/WSL2
> machine**, where `credential.helper` was set to the retired `manager-core`
> and pushes needed a prompt. It does not apply to the bare-metal Ubuntu
> 26.04 host, which pushes unattended over HTTPS without trouble. If you are
> back on the Windows box and a push hangs, that is the cause; fix with
> `git config --global credential.helper manager`.

---

## 1. What this project is

A pure-Rust port of SUNDIALS 7.8.0 (LLNL) scoped to **Linux on Intel/AMD
x86-64** — Ubuntu 24.04/26.04 and, by the argument in §4, the Debian, Arch
and Fedora families. No `unsafe`, no FFI, no external crates, `std` only.
Seven crates, 144 modules, one per upstream C file, and 119 examples,
keeping the exact C names, constants and return-flag conventions.

It **reuses the entire crate tree** of the sibling port
`SUNDIALS_7_8_Rust_port_for_AppleSilicon_macos` (github.com/once-ere/…),
which is where the solver modules were originally translated from the C
sources. Not one line of solver code was re-derived. What is new here is
the *target-platform* work — the `pow` question below, a native x86-64
oracle, a re-run of the whole verification gate under glibc, documentation
scoped to Linux — and then the two substitutions that removed the host C
library from the numerical path: a pure-Rust libm (§2a) and a pure-Rust
sparse LU (§2b), which together grew the tree from 141 modules and 108
examples to 144 and 119.

## 2. The `pow` question — resolved

The task set this as the gating item: the deterministic `pow` inherited
from the macOS port is a translation of the **ARM optimized-routines /
musl** `pow`, so does a Linux/x86-64 port need a different one?

**Answer: no — the algorithm is already the right one, but its evidence
was not.** ARM's optimized-routines `pow` *is* what glibc >= 2.28 ships as
`sysdeps/ieee754/dbl-64/e_pow.c`; on x86-64 glibc ifunc-dispatches to
`__ieee754_pow_fma`, the same source rebuilt with `-mfma -mavx2
-ffp-contract=fast`. So the correct x86-64 target is that FMA-contracted
build, and the Rust routine already reproduces its contraction map. What
was missing was any measurement made **on x86-64**: the macOS project
measured against oracle binaries built on arm64 and said so explicitly
(`POW_FMA_EXACTNESS.md` §5, "No differential run was made on a native
x86-64 host").

That measurement now exists and is part of this repository:

* `tools/pow_oracle.c` — builds with the host `cc`, calls the host `pow`,
  emits the reference bit-stream. Must be built and run on the target.
* `tools/pow_differential.sh` — driver; writes `logs/pow_differential.log`.
  Note that `/logs` is gitignored **except** `c_build.log` and
  `libm_differential.log`, which are committed because `requirements.md` §3
  attributes each backend failure to a line of the first and §2a's ulp table
  is read off the second. Every other log, this one included, is scratch and
  regenerated by re-running its tool.
* `pow_glibc_vs_native_oracle_{domain,random}` in
  `crates/sundials_core/src/sundials_math.rs` — the Rust side. Both sides
  regenerate the corpus from the same splitmix64 recurrence, so they
  cannot disagree about which inputs they evaluated. With no oracle file
  in the environment the tests report "not run" and pass, so `cargo test`
  stays green on hosts where the oracle would be meaningless.

Result on Ubuntu 24.04 / glibc 2.39 / x86-64: **0 mismatches over
5,900,000 domain inputs and 0 over 20,000,000 unrestricted finite inputs.**
The two residual 1-ulp disagreements the macOS project could not eliminate
do not exist against a native glibc oracle — they were artefacts of the
arm64-built oracle, which is exactly the doubt that document raised.

No new `pow` source was written: writing one would have replaced a routine
already bit-exact against the target with an unmeasured rewrite.

## 2a. The pure-Rust libm — the host libm is gone

`pow` was the first substitution and for a long time the only one. It is now
one of thirteen. `crates/sundials_core/src/sundials_libm.rs` implements
`exp`, `log`, `pow`, `expm1`, `log1p`, `sin`, `cos`, `atan`, `asin`, `acos`,
`sinh`, `cosh` and `acosh`. `exp`/`log`/`pow` are the ARM optimized-routines
kernels via musl (MIT), the same source glibc >= 2.28 ships; the other ten
are written here on a double-double core with an exact Payne-Hanek reduction,
their constant tables generated by `tools/gen_libm_constants.py`.

Call sites read `x.sun_sin()`, `x.sun_exp()`, … through a `SunMath` trait.
The rename is what makes the host unreachable: Rust resolves inherent methods
before trait ones, so `x.sin()` could not have been redirected. The only
`f64` methods left in `crates/` are `sqrt`, `mul_add`, `abs`, `ceil`, `round`
and `copysign` — all IEEE-754 exact.

`tools/libm_differential.sh` measures each against a 113-bit `__float128`
reference, 1,000,000 samples per corpus:

| function | pure Rust | host glibc 2.43 |
|---|---:|---:|
| `exp`, `log` | 0.5003 – 0.5071 ulp | identical, bit for bit (same source) |
| `acosh` | 0.5000 ulp | 0.5000 ulp (independent, both correctly rounded) |
| `sin`, `cos`, `atan`, `asin`, `acos` | **0.5000 ulp** | 0.5042 – 0.5186 ulp |
| `expm1`, `log1p` | **0.5000 ulp** | 0.7783 – 0.8414 ulp |
| `sinh`, `cosh` | **0.5000 ulp** | 0.9883 – 1.7848 ulp |

Bit-identity with glibc is impossible here, not merely unachieved: glibc uses
the LGPL IBM Accurate Mathematical Library for these. The alternative was an
fdlibm clone at ~0.55 ulp, which would have disagreed with glibc *more* often.
Correct rounding is both more accurate and the closer match. Details in
[`LIBM.md`](LIBM.md).

A `host-libm` cargo feature (default off) switches the thirteen methods back
to the host. It exists for `tools/ab_host_libm.sh` and is what keeps "0 port
defects" a measurement. **Never enable it for a production build.**

## 2b. The pure-Rust sparse LU — KLU is gone too

SuiteSparse KLU is LGPL-2.1+: not translatable into this BSD-3 tree, not
callable without FFI. That is why the eleven `*_klu` examples were unportable.
`crates/sundials_core/src/sundials_sparse_lu.rs` replaces it — a left-looking
sparse LU (Gilbert & Peierls 1988) under a faithful translation of SUNDIALS'
own BSD-3 `sunlinsol_klu.c`. Nothing derives from KLU, CSparse or any
SuiteSparse source. **All 11 `*_klu` examples are ported; 4 are byte-identical
to the C.**

It uses KLU's documented default pivoting — threshold partial with a diagonal
preference at `tol = 0.001` — and that was forced, not chosen. `idaHeat2D_klu`'s
boundary equations are literally `e_i`; largest-magnitude pivoting discards
that `1` for a neighbouring `-1/dx^2`, mixes boundary and interior unknowns,
and the run diverged where the C decays to zero. Matching KLU's rule fixed it
and made two further variants byte-identical.

**This substitution has no control build** — there is no KLU to switch back
to — so it cannot be attributed the way the libm is. It is verified directly:
against dense Gaussian elimination on 300 random sparse systems (worst
relative residual 7.3e-16), and for `idaHeat2D_klu` its hand-packed CSC
Jacobian checked entry by entry against an independent reference and against
finite differences of the residual (`cargo test -p ida_rs --example
idaHeat2D_klu`).

## 3. Verification gate, Linux vs macOS

Both columns are **host-libm** measurements. The current tree scores
145 / 34 / 20 — see §4 and [`evidence/purerust-libm-gate/`](evidence/purerust-libm-gate/).

| | macOS / arm64 (inherited) | **Linux / x86-64 (this repo)** |
|---|---:|---:|
| IDENTICAL | 127 | **153** |
| divergent, reference-side | 52 | **26** |
| excluded (KLU/SuperLU) | 20 | 20 |
| port defects | 0 | **0** |

26 variants that diverged on macOS are byte-identical here. That was the
predicted effect and the original evidence for the platform claim: at the
time, the port took `sin`, `cos`, `asin`, `acos`, `atan`, `sinh`, `cosh`,
`acosh`, `exp` and `ln` from the host through `f64`'s unspecified-precision
methods, so on a glibc host they landed on the very libm that generated the
upstream `.out` files, and the mismatch macOS had to document away was not
there.

**That is now history rather than mechanism.** Since §2a the port does not
call the host for any of them. The gate A numbers above were measured under
the old arrangement and have not been re-run since; gate B (§3a) is the
current measurement, and it is the one that reflects the tree as it stands.

The 26 that remain are **all reference-side, and that is now proven on
this host rather than inherited.** A divergence from a shipped `.out` is a
port defect only if the Rust output also differs from what the pristine
upstream C produces on the same machine. So the upstream C library and its
serial examples were built here with cmake + gcc 13.3.0
(`tools/pristine_c_build.sh`, 112 example binaries) and every divergent
variant was run three ways — Rust, pristine C, shipped reference — by
`tools/compare_pristine_c.sh`:

| comparison | result across all 26 |
|---|---|
| **Rust vs pristine C** | **`same` — 26 / 26** |
| pristine C vs shipped `.out` | `DIFF` — 26 / 26 |
| Rust vs shipped `.out` | `DIFF` — 26 / 26 (the gate result) |

The C and the Rust agree with each other and disagree with the shipped
reference, in every case. **The references are stale; the port is not
wrong anywhere.**

The two LAPACK examples needed one extra step, because a pristine build
with `ENABLE_LAPACK=OFF` does not contain them at all.
`tools/compare_lapack_substituted.sh` compiles `cv[s]Roberts_dnsL.c` with
exactly the two tokens the port also substitutes
(`sunlinsol_lapackdense.h` -> `sunlinsol_dense.h`, `SUNLinSol_LapackDense`
-> `SUNLinSol_Dense`) against the pristine C library, and both come out
`same` against the Rust. Their divergence from the reference is therefore
entirely the documented LAPACK -> native substitution, not a translation
error.

Secondary classification, from `tools/classify_diffs.sh`: **15 of the 26
are whitespace-only** — `tr -s ' '` makes the diff empty, so every printed
*value* is byte-identical and only column spacing differs
(`SUN_TABLE_WIDTH` 28 -> 29 in references that predate the change). The
other 11 have real content differences, all reference-side: two
LAPACK->native variants (`cv[s]Roberts_dnsL`), two upstream `.out`
anomalies (`cv[s]Pendulum_dns`), five trailing-whitespace-stripped
references (`cvsKrylovDemo_ls` x4, `idasAkzoNob_ASAi_dns`), and two
references missing a final blank line the source prints unconditionally
(`ark_conserved_exp_entropy_ark 1 1`, `ark_dissipated_exp_entropy 1 1`).

## 3a. Gate B — Rust vs C rebuilt on the same machine

Gate A compares against a reference file shipped years ago. Gate B compares
against the upstream C compiled from source on the same machine, minutes
apart, by the same toolchain — a different question, not a better answer, and
neither supersedes the other. Both cover the same 199 variants;
`python3 tools/cross_gate.py` asserts that before comparing and prints the
cross-tabulation.

| | gate A: vs shipped `.out` | gate B: vs C rebuilt here |
|---|---|---|
| host | Ubuntu 24.04, glibc 2.39, host libm | Ubuntu 26.04, glibc 2.43, pure-Rust libm |
| KLU examples | 20 excluded with SuperLU | 11 ported and compared, 9 SuperLU still out |
| identical | 145 of 199 (153 under the host libm) | **175 of 199** |

**175 is not a correction of 145.** Three results come out of putting them
side by side, none visible from either alone:

1. **All 26** of gate A's divergences are byte-identical to pristine C rebuilt
   here — "0 port defects" reproduced on a second distribution, glibc,
   compiler and rustc, against a reference *built* rather than downloaded.
2. Gate B's 15 divergences **decompose exactly**: 8 to the libm (the
   `host-libm` control build restores every one), 7 to the sparse LU (all
   `*_klu`, no control build possible), **0 unaccounted for**.
3. The 8 the control build names are *precisely* the 8 that flipped from
   IDENTICAL under gate A. Two experiments sharing nothing but the source
   tree pick the same set.

The pipeline was re-run from source afterwards: of the 190 C captures in the
compared set and all 199 Rust captures, **not one byte changed**. Six captures
outside the compared set moved — five OpenMP examples, whose numbers genuinely
differ run to run, and one MPI example whose lines merely come back in a
different order with every number identical.

Everything is at the repository root — [`c-results/`](c-results/),
[`rust-results/`](rust-results/), [`differences/`](differences/) and
[`requirements.md`](requirements.md) — raw captures included, produced by the
pipeline in the sibling working repository
[`SUNDIALS_7_8_Rust_port_for_Linux_on_ubuntu`](https://github.com/once-ere/SUNDIALS_7_8_Rust_port_for_Linux_on_ubuntu)
and vendored here by `tools/vendor_evidence.sh`.

## 4. Distribution coverage — measured, and one claim retracted

Nothing in the Rust tree is distribution-specific: `std` only, no
`cfg(target_os)`, no `cfg(target_arch)`, no build script, no system
library beyond what `std` itself links. The only distribution-visible
dependency is the libm behind `f64`'s transcendental methods.

**An earlier draft of this file argued that the claim therefore carries to
"Debian, Arch and Fedora on glibc >= 2.28". That was wrong and has been
retracted.** glibc's libm is not frozen across releases, and measuring it
is what showed so. `tools/glibc_sweep.sh` builds `tools/libm_probe.c` in
each distribution's container and hashes 1,000,000 results per function:

| distro | libc | functions disagreeing with the reference host (glibc 2.39) |
|---|---|---|
| Debian 12 | glibc 2.36 | `atan` |
| **Ubuntu 24.04** | **glibc 2.39** | — (reference host) |
| Fedora 41 | glibc 2.40 | none |
| Debian 13 | glibc 2.41 | none |
| Arch (rolling) | glibc 2.44 | `sinh`, `cosh`, `acosh` |
| Alpine 3.20 | musl | everything except `sqrt` — including `pow` |

`pow` is bit-identical on every glibc tested, so §2's result carries to all
of them. `sqrt` matches everywhere, as IEEE-754 requires.

`tools/gate_in_container.sh` then ran the **full 199-variant gate natively
inside three of those containers** — still under the host libm — to find out
whether those libm differences are output-observable:

All four rows are **host-libm** measurements, kept as the historical
baseline — they are the reason the pure-Rust libm exists, since the score
moves with the host. Under the pure-Rust libm all four give 145 / 34 / 20 —
as does Alpine/musl, which this table lists as hopeless.

| distro | libc | rustc | gate (host libm) | vs. reference host |
|---|---|---|---|---|
| Ubuntu 24.04 | 2.39 | 1.93.1 | **153 / 26 / 20** | reference |
| Debian 12 | 2.36 | 1.97.1 | **153 / 26 / 20** | identical variant set |
| Fedora 41 | 2.40 | 1.97.1 | **153 / 26 / 20** | identical variant set |
| Arch | 2.44 | 1.97.1 | **150 / 29 / 20** | +3 variants diverge |

0 build failures and 0 run failures everywhere; the containers used a
*newer* rustc than the host, so the result is toolchain-stable too.

**Verified coverage under the host libm: glibc 2.36 through 2.41** — but the
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

**Arch (glibc 2.44): three variants diverged** —
`ark_analytic_lsrk_domeigest` (both argv variants) and
`ark_analytic_lsrk_varjac`. Predicted from the fingerprint table before the
gate was run, then confirmed by it: `sinh`, `cosh` and `acosh` are reached
from exactly one module, the wrappers at `arkode_lsrkstep.rs:83-98` (used
from `:1158` and `:3255`), and glibc 2.44 changed all three. A libm-version
effect, not a port defect.

**§2a removed the cause — and the measurement, now made on seven hosts across
two libcs and two CPU architectures, is more interesting than the
prediction.**
`tools/gate_in_container.sh debian:12 debian:13 fedora:41 archlinux:latest alpine:3.20`
was run under the pure-Rust libm, alongside `tools/verify_examples.sh all` on
this host:

| host | libc | rustc | gate |
|---|---|---|---|
| Debian 12 | glibc 2.36 | 1.97.1 | 145 / 34 / 20 |
| Fedora 41 | glibc 2.40 | 1.97.1 | 145 / 34 / 20 |
| Debian 13 | glibc 2.41 | 1.97.1 | 145 / 34 / 20 |
| Ubuntu 26.04 (this host) | glibc 2.43 | 1.96.1 | 145 / 34 / 20 |
| Arch | glibc 2.44 | 1.97.1 | 145 / 34 / 20 |
| Alpine 3.20.10 | **musl 1.2.5** | 1.97.1 | 145 / 34 / 20 |
| Debian 13 on **aarch64** *(emulated)* | glibc 2.41 | 1.97.1 | 145 / 34 / 20 |

Not merely the same tally: the *same 34 variants*, name for name. The seven
DIFF lists are byte-identical files, every one hashing to
`6581e4918e5ab2c71ee6354f383a0f34` — `evidence/purerust-libm-gate/README.md`
gives the recipe. Two rustc versions, two libcs and two CPU architectures are
covered, so the result is toolchain-stable, libc-stable and
architecture-stable, not just distribution-stable. The host dependence this
section documents is gone — Arch is not an outlier any more because nothing
is.

The aarch64 row is Debian 13 a second time, which is the point of choosing it:
same image, same glibc 2.41, same rustc 1.97.1 as the x86-64 row above it, so
the comparison isolates the CPU architecture and nothing else. The two DIFF
lists are byte-identical.

It did not restore the three variants, though, and it would be dishonest to
imply otherwise. 153 became 145 on **all seven hosts**, not just Arch. The
eight that flipped from IDENTICAL to DIFF are exactly the eight that
`differences/ab-host-libm.tsv` attributes to the libm — zero other class
changes — and the three Arch ones are inside that eight. The port stopped
tracking the host's libm and started disagreeing with eight stale references
instead, on every host equally. Since those references were generated by
glibc routines that §2a measures as less accurate, this is the port being
right where the reference is old; it is still eight fewer matches for anyone
whose criterion is "reproduces the shipped `.out`".

Evidence: [`evidence/purerust-libm-gate/`](evidence/purerust-libm-gate/).
The table above describes **host-libm** behaviour and is kept as the
historical baseline.

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

## 5. Open items

Nothing blocks the port; these would strengthen the evidence.

1. ~~**Bare-metal re-run.**~~ **Largely done.** Gate B, the libm
   differential and the `pow` differential were all run on a **bare-metal**
   Ubuntu 26.04 host (`systemd-detect-virt` -> `none`, 24 cores), and the
   substantive part of the port-defect proof came out the same there: all 26
   of gate A's divergences are byte-identical to pristine C rebuilt on that
   machine (§3a). Gate A has since been run bare-metal as well:
   `tools/verify_examples.sh all` on the Ubuntu 26.04 host gives
   **145 / 34 / 20**, matching Arch variant for variant (§4). The
   153 / 26 / 20 in the tables above is the historical host-libm result from
   WSL2 and is labelled as such throughout. **This item is closed.**
2. ~~**Native pristine-C rebuild for the content divergences.**~~ **Done.**
   Upstream SUNDIALS 7.8.0 was built here with cmake + gcc 13.3.0 and all
   26 divergent variants compared three ways; Rust == pristine C in every
   case (§3). `tools/pristine_c_build.sh`,
   `tools/compare_pristine_c.sh`, `tools/compare_lapack_substituted.sh`.
3. ~~**glibc version sweep.**~~ **Done, and it changed the answer** — see
   §4. libm fingerprints across five distributions plus the full gate
   re-run natively on three of them. Verified coverage is glibc 2.36–2.41;
   Arch's 2.44 moves three LSRK variants. `tools/glibc_sweep.sh`,
   `tools/gate_in_container.sh`.
4. ~~**Arch / glibc 2.44, if byte-identity there is ever wanted.**~~
   **Done, measured, and the answer is not the one this item assumed.** It proposed porting
   `SUNRsinh`/`SUNRcosh`/`SUNRacosh` "the way `pow` was ported", and judged
   it not worth a second hand-maintained routine set. §2a did exactly that
   for **all thirteen** functions instead, which removes the host dependence
   everywhere rather than in one stepper. The cost the item worried about is
   real and was accepted: `sundials_libm.rs` is now a maintained component,
   with `tools/libm_differential.sh` as its regression test.

   That measurement has now been made on seven hosts, and it refutes the
   item's premise. Debian 12, Debian 13, Fedora 41, this host, Arch,
   Alpine/musl and Debian 13 on emulated aarch64 all give
   **145 / 34 / 20 with the same 34 variants**,
   so byte-identity *across hosts* is achieved — but the three variants do
   **not** reproduce against the shipped `.out`. They cannot: the references
   came from glibc's routines and the port no longer computes what glibc
   computes. What the substitution bought was host-independence, not
   reference agreement, and it cost eight reference matches on every host.
   See [`evidence/purerust-libm-gate/`](evidence/purerust-libm-gate/).
5. ~~**musl**, if it is ever wanted: out of scope, for the measured reason
   in §4.~~ **Done.** Alpine 3.20.10 / musl 1.2.5 gives 145 / 34 / 20 with a
   DIFF list byte-identical to the glibc hosts (§4). Gate A only: there is no
   C build on Alpine, so gate B and the libm differential were not repeated
   there.

## 6. How to reproduce, from a clean checkout

On a Linux x86-64 host with rustup and a C compiler:

```bash
git clone https://github.com/once-ere/SUNDIALS_7_8_Rust_port_for_Linux.git
cd SUNDIALS_7_8_Rust_port_for_Linux
cargo build --workspace          # 0 warnings
cargo test  --workspace --lib    # 39 passed
tools/pow_differential.sh all    # 0 mismatches / 25.9M inputs
tools/libm_differential.sh 1000000   # the ulp table in §2a
```

The upstream example tree is **vendored** under `examples/`, so the reference
gate needs nothing else (it still falls back to a parent-directory tree if one
is present):

```bash
tools/verify_examples.sh all     # gate A; then read logs/summary.txt
tools/classify_diffs.sh          # second pass over the non-IDENTICAL ones
python3 tools/cross_gate.py      # gate A and gate B, cross-tabulated
```

Gate B is *not* reproduced from this repository — it needs a C toolchain and
the full pipeline, which live in the sibling working repo
[`SUNDIALS_7_8_Rust_port_for_Linux_on_ubuntu`](https://github.com/once-ere/SUNDIALS_7_8_Rust_port_for_Linux_on_ubuntu).
Run the pipeline there, then bring the results across:

```bash
tools/vendor_evidence.sh ../SUNDIALS_7_8_Rust_port_for_Linux_on_ubuntu
```

That is an `rsync` of `c-results/`, `rust-results/` and `differences/` plus
`requirements.md` and the two cited logs, followed by a link check that fails
the script if anything dangles. It transforms nothing — both repositories put
these directories at their root, so the copy is byte-for-byte. Each
subdirectory's `README.md` carries the commands that produced it.

To reproduce the port-defect proof, which needs a native C build of the
upstream tree (out of source; the tree stays read-only):

```bash
tools/pristine_c_build.sh            # cmake + gcc, ~112 example binaries
tools/compare_pristine_c.sh          # Rust vs pristine C vs reference
tools/compare_lapack_substituted.sh  # the two *L examples
```

From Windows, `tools/wsl_sync_build.sh {build|test|rel|gate|pow}` mirrors
the working copy into a WSL Ubuntu sandbox (`~/sdl/port`, with
`~/sdl/examples` symlinked at the C tree) and runs the step there. It also
strips CRLF from `tools/*.sh`, which a Windows checkout can introduce;
`.gitattributes` pins those files to LF to prevent it.

## 7. Provenance

* Upstream: SUNDIALS 7.8.0, LLNL, BSD-3-Clause. Read-only reference at
  `C:\Users\youruser\Developer\sundials-7.8.0` on the Windows/WSL2 machine and
  `/home/youruser/Developer/sundials-7.8.0` on the Ubuntu 26.04 one. The tree is
  also vendored under `examples/`.
* Crate tree: inherited wholesale from
  `SUNDIALS_7_8_Rust_port_for_AppleSilicon_macos`, BSD-3-Clause, same
  author lineage. `ARCHITECTURE.md`, `PROGRESS.md` and the body of
  `VERIFICATION.md` come from there unchanged and remain accurate — they
  describe the translation, which is platform-independent.
* `exp`, `log`, `pow` in `sundials_libm.rs`, and the deterministic `pow` in
  `sundials_math.rs`: ARM optimized-routines via musl `src/math/`, MIT,
  (c) 2018 Arm Limited — the algorithm glibc >= 2.28 ships. See `NOTICE`,
  `LIBM.md` and `POW_FMA_EXACTNESS.md`.
* The double-double core and the ten routines on it are original here;
  `sundials_sparse_lu.rs` is an independent implementation of Gilbert-Peierls
  from the literature. **No SuiteSparse code — KLU, BTF, AMD or CSparse — was
  read, copied or derived from.**
* New in this repository: `tools/pow_oracle.c`, `tools/pow_differential.sh`,
  `tools/classify_diffs.sh`, `tools/wsl_sync_build.sh`,
  `tools/gen_libm_constants.py`, `tools/libm_differential.sh`,
  `tools/ab_host_libm.sh`, `tools/cross_gate.py`,
  `tools/vendor_evidence.sh`, the differential unit tests, `.gitattributes`,
  this file, `LIBM.md`, and the Linux scoping in `README.md`, `CLAUDE.md`,
  `POW_FMA_EXACTNESS.md` and `VERIFICATION.md`.
