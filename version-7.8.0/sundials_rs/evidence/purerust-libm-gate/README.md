# Gate A re-run under the pure-Rust libm

`VERIFICATION.md` reports **153 / 26 / 20** against the shipped `.out`
references. That was measured with the **host** libm, before
`sundials_libm.rs` existed, and it had not been re-run since. These two files
are that re-run.

| build | host | glibc | rustc | IDENTICAL / DIFF / EXCLUDED |
|---|---|---|---:|---|
| host libm (historical) | Ubuntu 24.04 | 2.39 | 1.93.1 | 153 / 26 / 20 |
| host libm (historical) | Debian 12 | 2.36 | 1.97.1 | 153 / 26 / 20 |
| host libm (historical) | Fedora 41 | 2.40 | 1.97.1 | 153 / 26 / 20 |
| host libm (historical) | Arch | 2.44 | 1.97.1 | **150 / 29 / 20** |
| **pure-Rust libm** | **Ubuntu 26.04** | **2.43** | **1.96.1** | **145 / 34 / 20** |
| **pure-Rust libm** | **Debian 12** | **2.36** | **1.97.1** | **145 / 34 / 20** |
| **pure-Rust libm** | **Fedora 41** | **2.40** | **1.97.1** | **145 / 34 / 20** |
| **pure-Rust libm** | **Debian 13** | **2.41** | **1.97.1** | **145 / 34 / 20** |
| **pure-Rust libm** | **Arch** | **2.44** | **1.97.1** | **145 / 34 / 20** |
| **pure-Rust libm** | **Alpine 3.20.10** | **musl 1.2.5** | **1.97.1** | **145 / 34 / 20** |
| **pure-Rust libm** | **Debian 13 on aarch64** *(emulated)* | **2.41** | **1.97.1** | **145 / 34 / 20** |

Two results, and the second is the one that matters.

**1. The host dependence is gone.** Under the host libm the score depended on
which glibc you linked: 153 on 2.36 through 2.41, but 150 on Arch's 2.44,
because that release changed `sinh`, `cosh` and `acosh`. Under the pure-Rust
libm, **seven hosts spanning glibc 2.36, 2.40, 2.41, 2.43, 2.44, musl 1.2.5
and a second CPU architecture all report 145 / 34 / 20 — and the same 34
variants, name for name.** The DIFF
lists are byte-identical files. Arch is no longer an outlier; there is no
outlier, because the score no longer depends on the host at all. Two rustc
versions are covered too (1.96.1 on the host, 1.97.1 in every container), so
the result is toolchain-stable as well.

**musl is the surprising one.** It was previously out of scope for a measured
reason: `tools/glibc_sweep.sh` showed Alpine disagreeing with glibc on `sin`,
`cos`, `exp`, `log`, `asin`, `acos`, `atan`, the hyperbolics and `pow` —
everything except `sqrt`. That reason is void now, because the port does not
call any of them. On musl it produces the same 145 and the same 34 variants
as on glibc, which is a stronger statement than distribution-independence:
the port is **libc-independent**, at least for what the example gate can see.

**The aarch64 row, and how far to trust it.** It is Debian 13 again, same
glibc and rustc, run under QEMU user-mode emulation on this x86-64 host — the
`--- aarch64 [EMULATED] / ... ---` header in `gate-debian-13-arm64.txt` says
so, and the filename is architecture-tagged so it cannot be confused with the
x86-64 run of the same image. Its DIFF list is byte-identical to that x86-64
run: same 34 variants, same order, same digest.

That is real evidence and it is not hardware. QEMU is not an aarch64 CPU. It
implements `sqrt` and fused multiply-add to the IEEE-754 specification, which
pins them exactly, and those plus integer arithmetic are what the pure-Rust
libm is built on — so a faithful emulator and real silicon should agree here,
and they do agree with x86-64. What it cannot exclude is a genuine aarch64
code-generation difference that QEMU reproduces identically. Treat it as
strong corroboration of architecture-independence, not as a substitute for a
run on an arm64 machine.

Compiling for aarch64 is separately clean, and needs no emulation at all:
`cargo check --target aarch64-unknown-linux-gnu --workspace --all-targets`
and the `-musl` equivalent both give **0 errors and 0 warnings** across all 7
crates and all 119 example targets.

**A note on Debian 13, because it corrects the historical record.** The
host-libm documentation listed it under "verified coverage: glibc 2.36
through 2.41", but the gate was never run there — only
`tools/glibc_sweep.sh`, which fingerprints the libm and found 2.41 matching
2.39. That is weaker evidence than a gate run, and the distinction is the
entire reason `gate_in_container.sh` exists: a fingerprint difference may or
may not be output-observable, so a fingerprint *match* is a prediction, not a
result. There is no `gate-debian-13.txt` in
[`../linux-x86_64-glibc239/`](../linux-x86_64-glibc239/). Under the pure-Rust
libm it has now actually been run, so glibc 2.41 is gate-verified for the
current build even though it never was for the old one.

Two limits on the above, so it is not read as more than it is. Only gate A
ran on Alpine and on aarch64 — there is no C toolchain build there, so the C-versus-Rust comparison
and the libm differential were not repeated on musl. And this is x86-64
throughout; nothing here says anything about arm64.

Each log's header line now records the architecture it ran on
(`--- x86_64 / ldd ... / rustc ... ---`), so an emulated or cross-architecture
run can never be mistaken for a native one. All five above say `x86_64`.

Checking that claim yourself:

```bash
# the three container runs
for f in gate-*.txt; do
  printf '%-28s ' "$f"
  sed -n '/variants reported DIFF here/,$p' "$f" | tail -n +2 |
    grep -v '^[[:space:]]*$' | LC_ALL=C sort | md5sum
done
# the host run, whose file is a full gate summary rather than a container log
printf '%-28s ' '(host)'
grep 'DIFF(' summary-ubuntu-2604-glibc243.txt | awk '{print $1, $2}' |
  LC_ALL=C sort | md5sum
```

All four print `6581e4918e5ab2c71ee6354f383a0f34`. The host is extracted
differently because its file is the raw `logs/summary.txt`, not a container
log — `gate-*.txt` alone would only cover three of the four.

**2. It cost eight reference matches, on every host.** 153 became 145 on all
of them, not just on Arch. The eight that flipped from IDENTICAL to DIFF
are *exactly* the eight that `differences/ab-host-libm.tsv` attributes to the
libm, with zero other class changes:

```
ark_analytic_lsrk
ark_analytic_lsrk_domeigest                    (both argv variants)
ark_analytic_lsrk_varjac
ark_kpr_mri            [10 4 0.001 -100 100 0.5 1]
cvsDiurnal_FSA_kry     [-sensi sim t]
idasSlCrank_dns
idasSlCrank_FSA_dns
```

That is the trade, stated plainly: the port stopped tracking the host's libm
and started disagreeing with eight stale references instead. The references
were generated by glibc's routines, which are less accurate than these — see
`differences/ATTRIBUTION.md` for the ulp measurements — so this is the port
being right where the reference is old, not the port regressing. It is still
a cost, and pretending otherwise would be dishonest: anyone whose acceptance
criterion is "matches the shipped `.out`" is worse off by eight variants.

Reproduce:

```bash
tools/verify_examples.sh all                                  # this host
tools/gate_in_container.sh debian:12 debian:13 fedora:41 archlinux:latest alpine:3.20
```

`gate_in_container.sh` takes docker or podman, and copies the workspace into
each container rather than mounting it writable, so nothing on the host is
touched.
