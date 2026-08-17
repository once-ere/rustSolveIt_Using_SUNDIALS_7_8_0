# differences — C output versus Rust output, variant by variant

Every serial example with a pure-Rust translation — 190 variants — was
executed twice on this machine: once as the upstream C binary
(`c-results/`) and once as its translation (`rust-results/`). This
directory is the comparison of the two stdout streams. A further 9 variants ran on **neither** side and are listed as `NOT_PORTED`;
they are not a comparison that failed, they are a comparison that does
not exist. Nothing here is asserted — every classification is computed by
[`../tools/compare_results.py`](../tools/compare_results.py) from the
captured bytes.

## Provenance

| item | value |
|---|---|
| generated | 2026-08-14 14:07:36 UTC |
| operating system | Ubuntu 26.04 LTS |
| kernel / platform | Linux-7.0.0-29-generic-x86_64-with-glibc2.43 |
| architecture | x86_64 |
| C library | ldd (Ubuntu GLIBC 2.43-2ubuntu2.3) 2.43 |
| C compiler | cc (Ubuntu 15.2.0-16ubuntu1) 15.2.0 |
| C++ compiler | c++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0 |
| Fortran compiler | GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0 |
| CMake | cmake version 4.2.3 |
| rustc | rustc 1.96.1 (31fca3adb 2026-06-26) |
| cargo | cargo 1.96.1 (356927216 2026-06-26) |
| CPU cores | 24 |

## How to reproduce all of it

```bash
tools/c_build.sh && tools/c_examples_run.sh      # the C side
tools/rust_examples_run.sh                       # the Rust side
python3 tools/compare_results.py                 # the comparison
tools/ab_host_libm.sh                            # the host-libm control build
python3 tools/make_reports.py                    # these documents
```

## Headline result

**Of 190 comparable variants, 175 are byte-for-byte identical (92.1%).**

**With the elementary functions delegated back to the host C library (`--features host-libm`), 183 of 190 are identical.** The switch changes nothing else in the port, so the 8 variants it restores are caused by the pure-Rust libm and by nothing else — measured, not asserted.

The 7 that differ under **both** builds are exactly the `*_klu` examples. That is not a second finding, it is the same one seen twice: `host-libm` does not touch the sparse linear solver, and there is no KLU to switch back to, so those variants cannot be attributed this way. They are covered instead by direct verification of the replacement solver.

See [ATTRIBUTION.md](ATTRIBUTION.md).

| class | variants | meaning |
|---|---:|---|
| IDENTICAL | 175 | the two stdout streams are equal byte for byte |
| WHITESPACE | 0 | every printed character matches; only column padding differs |
| NUMERIC | 15 | same text, same field count, at least one number differs |
| STRUCTURAL | 0 | different lines, words or field counts |
| NOT_PORTED | 9 | SuperLU_MT example; absent on both sides, so there is no output to compare |
| NO_C_RUN | 0 | the C example could not be built on this machine |

## How to read a difference

For every non-identical variant there is a unified diff, and for every
`NUMERIC` one there is also a `.numbers` file naming the single worst
field:

```bash
cat differences/diffs/<dir>/<variant>.diff
cat differences/diffs/<dir>/<variant>.numbers
```

`worst rel` below is the largest relative difference between any pair of
printed numbers, and `worst ulp` is the same pair measured in
representable double steps. One ulp is the smallest difference two
doubles can have — the granularity of the format itself, not an error
in either program.

**Do not read the whole `worst ulp` column as last-bit noise.** These
are the worst pair in each variant, and they range from 5575 up to 9.46e+18
across the table below. A ulp distance only means "almost equal" when
it is small; the large values are two numbers that genuinely parted
company — for the largest, the pair has opposite signs, which makes the
ulp count meaningless and the relative difference the number to read.

## Attribution

[**ATTRIBUTION.md**](ATTRIBUTION.md) — the controlled experiment that
decides, for every divergent variant, whether the translation is wrong
or the libm substitution accounts for it. Raw data in
[`ab-host-libm.tsv`](ab-host-libm.tsv).

## Per-solver tables

* [ARKODE](by-solver/arkode_C_serial.md) — 73 identical of 78
* [CVODE](by-solver/cvode_serial.md) — 21 identical of 24
* [CVODES](by-solver/cvodes_serial.md) — 32 identical of 39
* [IDA](by-solver/ida_serial.md) — 13 identical of 14
* [IDAS](by-solver/idas_serial.md) — 15 identical of 22
* [KINSOL](by-solver/kinsol_serial.md) — 21 identical of 22

