# Is the deterministic `pow` 100% correct?

**On this repository's target — Linux, glibc, Intel/AMD x86-64 — yes, as far
as measurement reaches: 0 mismatches against the native glibc `pow` over
5,900,000 inputs in the domain SUNDIALS evaluates and 0 over 20,000,000
unrestricted finite inputs.**

§§1–7 below are the inherited macOS/arm64 investigation, kept because it is
where the FMA-contraction map was derived and where the reasoning lives.
**§0 is the result that supersedes its "what is still not proven" section for
this platform.** Read §0 first.

---

## 0. Native x86-64 measurement (this repository)

The macOS project closed §5 with: *"No differential run was made on a native
x86-64 host."* That run now exists and is part of this repository.

| | |
|---|---|
| host | Ubuntu 24.04 x86-64, **glibc 2.39**, gcc 13.3.0, CPU with FMA |
| oracle | `tools/pow_oracle.c`, built with the host `cc -O2`, calling the host `pow` |
| harness | `tools/pow_differential.sh` → `logs/pow_differential.log` |
| Rust side | `pow_glibc_vs_native_oracle_{domain,random}` in `crates/sundials_core/src/sundials_math.rs` |

| corpus | inputs | mismatches |
|---|---:|---:|
| SUNDIALS operating domain (`x` ∈ (0,100], `y` = ±1/k, k = 1..13, plus `y` uniform in [-1,1]) | 5,900,000 | **0** |
| unrestricted finite bit patterns | 20,000,000 | **0** |

Two things this settles:

* **The "ARM optimized-routines" provenance is not a portability problem
  here — it is the point.** That algorithm *is* what glibc ≥ 2.28 ships as
  `sysdeps/ieee754/dbl-64/e_pow.c`, and on x86-64 glibc ifunc-dispatches to
  `__ieee754_pow_fma`, the same source rebuilt with `-mfma -mavx2
  -ffp-contract=fast`. §2 derived the contraction map for exactly that
  build. Measuring it against the real thing confirms the derivation.
* **The two residual 1-ulp disagreements of §5 do not exist against a native
  glibc oracle.** They were artefacts of the arm64-built oracle — which is
  precisely the doubt §5 raised about its own provenance. `x =
  0.60506369761398, y = -35.20830257243234` and `x = 1.700984188552496,
  y = 44.56990535118902` both agree bit-for-bit here.

Method notes, so the run can be repeated rather than trusted:

* The two sides never exchange inputs. Both regenerate the corpus from the
  same splitmix64 recurrence (`pow_corpus` in Rust, `domain_pair`/
  `random_pair` in C), so they cannot disagree about what they evaluated.
  If those two generators ever drift apart the test silently compares
  different inputs — keep them in lockstep.
* The oracle is a separate binary calling libm; it shares no code with the
  Rust routine.
* Both-NaN counts as agreement (NaN payloads are not architecturally
  specified). No mismatch in either corpus was a NaN case.
* With no oracle path in the environment the two tests report "not run" and
  pass, so `cargo test` stays green on hosts where the oracle would be
  meaningless.

**Still not claimed:** exhaustiveness (the input space is 2^128 pairs; 25.9M
measured inputs is strong evidence, not proof), other glibc versions (2.39
only — see `current_status.md` §5), musl hosts, and anything about the
*other* libm functions. On glibc the rest of the libm surface needs no
substitution, because the host libm is the one the references came from;
that is a property of the platform, not of this routine.

---

> **Scope of §§1–7 (inherited).** Every measurement in the sections below was
> run on the macOS/Apple-Silicon (arm64) development machine of the sibling
> port, with oracle binaries compiled there. §3's oracle matrix lists an
> x86-64 axis while §5's provenance note says "on arm64", and the document
> does not record which the numbers came from — so treat only the arm64
> oracles as established there, and read §0 above for this platform.
>
> These sections also cover **`pow` and nothing else**.

## 1. Why "correct" is the wrong word

`crates/sundials_core/src/sundials_math.rs` contains `pow_glibc`, a Rust port
of the ARM optimized-routines `pow` (via musl, MIT) — the same algorithm
glibc >= 2.28 ships as `sysdeps/ieee754/dbl-64/e_pow.c`.

It is **not** correctly rounded, and must not be. Its documented worst-case
error is ~0.54 ulp. The goal is not mathematical correctness but *bit-exact
agreement with the specific libm build that generated this project's
reference outputs*. This was tested directly in an earlier phase: forcing
correct rounding (80-digit decimal, iterated to fixpoint) did **not**
reproduce the reference output. We are deliberately reproducing another
implementation's rounding behaviour, errors included.

Why it matters at all: `SUNRpowerR` is called from the step-size controller
of every solver. A 1-ulp difference in one `eta` changes the next step size,
which changes every subsequent step. There is no error budget to absorb it —
the acceptance criterion is byte-identical printed output.

## 2. The defect

The algorithm is written to compile two ways. Its own source says so:

> `/* Without fma the worst case error is 0.25/N ulp larger. */`

glibc on x86-64 dispatches to `__ieee754_pow_fma`, built from
`sysdeps/x86_64/fpu/multiarch/e_pow-fma.c` with `-mfma -mavx2
-ffp-contract=fast`. That flag contracts **every** eligible `a*b + c` in the
translation unit into a single fused multiply-add. Fused and unfused round
differently whenever the exact result lies near a rounding boundary.

Fused multiply-adds enter that build two ways:

1. **Explicit** — the C calls `__builtin_fma` under `#if __FP_FAST_FMA`.
   The port had all three of these right from the beginning.
2. **Implicit** — the compiler contracts ordinary expressions.
   **The port originally had none of these.**

Disassembling a `-ffp-contract=fast` build of the same C shows **21 FMA
instructions**. The port had **5**.

## 3. Method — measure, never infer

Which multiply folds into which add inside a nested expression is a compiler
scheduling decision. Guessing it is exactly how the first attempt went wrong,
so every site here was settled empirically:

* Reference C (`pow.c`, `pow_data.c`, `exp_data.c`, plus glibc's `e_pow.c`
  and `e_pow_fma.c`) compiled locally into oracle binaries under
  gcc x clang, arm64 x x86-64, `-ffp-contract=fast` x `off`.
* A standalone Rust harness reading `(x_bits, y_bits)` and emitting result
  bits, driven over the same corpora as the oracles.
* Differential comparison, then per-site bisection.

The gcc and clang `fast` oracles agree with each other, which is what makes
the oracle trustworthy.

## 4. Results

Corpus: 20,000,000 random finite `(x, y)` bit-pattern pairs.

| configuration | diff lines vs `gcc -ffp-contract=fast` |
|---|---:|
| uncontracted reference build (`-ffp-contract=off`) | 12,350 |
| port as shipped (5 fused sites) | 228 |
| **port with the two verified sites fused** | **4** |
| naive "fuse everything" (21 sites) | 494 |

Two things worth reading carefully:

* **Fusing everything made it worse** (494 > 228). The compiler does *not*
  contract every eligible site. This is the empirical proof that the map has
  to be measured.
* **`lo4` was a trap.** Fusing `lo4 = t2 - hi + ar2` fixed one failing input
  and broke 93 others (4 -> 188 diff lines). Rejected.

### The two sites that are genuinely contracted

1. `pow_exp_inline` — the exponential polynomial
   `tmp = tail + r + r2*(C2 + r*C3) + r2*r2*(C4 + r*C5)`
2. `pow_exp_specialcase`, `k > 0` branch — `0x1p1009 * (scale + scale*tmp)`,
   reached whenever the result is near overflow. This was the larger of the
   two by frequency and had been missed entirely.

### The domain that actually matters

SUNDIALS calls `SUNRpowerR(bias*dsm, ±1/order)`, so `x` in `(0, ~100]` and
`|y| <= 1`. Corpus: 5,900,000 inputs — `x` uniform in `(0, 100]` against
`y = ±1/k` for `k = 1..13` (every integrator order), plus `y` uniform in
`[-1, 1]`.

| configuration | mismatches |
|---|---:|
| port as shipped | 4 |
| **port after this fix** | **0** |

**The shipped implementation was wrong inside the operating domain.** That is
the finding that justifies the change.

## 5. What is still not proven

* **The residual 2 inputs.** In the 20M random corpus, two still differ by
  1 ulp: `x = 0.60506369761398, y = -35.20830257243234` and
  `x = 1.700984188552496, y = 44.56990535118902`. Both have `|y|` far above
  anything SUNDIALS evaluates. No single further contraction fixes them, and
  every candidate tried made the overall picture worse. They are recorded
  here rather than hidden.
* **Empirical, not exhaustive.** The input space is 2^128 pairs. Zero
  mismatches over 5.9M domain inputs is strong evidence, not proof.
* **Oracle provenance.** The oracles were built with gcc/clang on arm64 (this
  project's development platform, macOS on Apple Silicon); the references were
  generated by GCC-built glibc on x86-64. Contraction choices could in
  principle differ. The mitigating evidence is that the gcc and clang `fast`
  builds agree with each other, and that the frozen test triples in
  `sundials_math.rs` came from a run that reproduced upstream reference output
  byte-for-byte.
* ~~**No differential run was made on a native x86-64 host.**~~
  **Superseded by §0** — that run has now been made, on Ubuntu 24.04 /
  glibc 2.39 / x86-64, with 0 mismatches over both corpora. The original
  text follows for the record. §3's oracle
  matrix names an x86-64 axis but §5's provenance note above says the oracles
  were built on arm64, and nothing records which of the two the numbers came
  from — so the safe reading is that the 20M and 5.9M corpora were measured
  against arm64-built oracles only. The `pow_glibc_bits` unit test does pass
  on Windows 11 x86-64 (checked 2026-08-09), but that test samples a handful
  of frozen triples — it is the weak instrument §7 warns about, not a re-run
  of the differential. A port targeting another platform must rebuild the
  oracle natively on that platform and repeat §3 there.

## 6. Effect on the verification gate: none

`tools/verify_examples.sh all` before and after the fix:

```
127 IDENTICAL / 52 documented / 20 excluded  (199 variants)
```

Byte-for-byte identical summaries — **no variant changed status**.

This is worth stating plainly rather than burying. The defect was **real but
latent**: none of the 199 example variants happened to evaluate `pow` at one
of the affected inputs. The example gate is a *sampling* instrument for this
class of bug, not a proof, and it would not have caught this. It took a
20-million-input differential test against a compiled oracle.

It also resolves a doubt raised earlier. When the first FMA fix landed, the
six LSRK variants classified `ref-libm` were called into question, because
that classification had been derived pre-fix and cited "a single 1-ULP `pow`
disagreement". Completing the contraction map changes none of their output,
so **`pow` contraction does not explain the LSRK divergence** and the
`ref-libm` classification stands on its own evidence.

## 7. Regression protection

The pre-existing `pow_glibc_bits` unit test passed against the *broken*
implementation — it only sampled easy inputs, which is precisely how this
survived. Any future change to `pow_glibc` should be re-validated with the
differential harness against a freshly built oracle, not against the example
gate, which is blind to it.

## 8. Verdict

| question | answer |
|---|---|
| Correctly rounded? | No — and deliberately so; correct rounding fails the acceptance test. |
| Bit-exact with the reference libm across all doubles? | **On this repository's target, yes as far as measured** — 0 mismatches over 20M unrestricted inputs against the native glibc `pow` (§0). Against the arm64-built oracles of §§1–7, 2 inputs in 20M differed by 1 ulp. |
| Bit-exact across the domain SUNDIALS evaluates? | **Yes** — 0 mismatches over 5.9M inputs on both hosts. |
| Does any of this change the 199-variant gate? | No. The §2 defect was latent; §0 changed no variant either. |
| On which platform was this measured? | §0: Linux / glibc 2.39 / x86-64, native. §§1–7: macOS on Apple Silicon (arm64). |
| Does an exact `pow` make the port's output portable? | **No** — but on glibc it does not need to. `pow` is the only libm function taken off the host; `sin`, `cos`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `acosh`, `exp` and `ln` still come from the host, which on this platform *is* the libm the references came from. That is why the gate improves from 127 to 153 IDENTICAL here, and why the claim is scoped to glibc rather than to "any OS". |
