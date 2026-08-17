# Evidence — Linux / x86-64 / glibc 2.39

Raw artefacts from the verification run behind every claim in
[`../../current_status.md`](../../current_status.md) and Part A of
[`../../VERIFICATION.md`](../../VERIFICATION.md). Committed because
`logs/` is gitignored and a result nobody can inspect is not a result.

| file | produced by | contents |
|---|---|---|
| `host.txt` | — | kernel, glibc, gcc, rustc and CPU of the measuring host |
| `pow_differential.log` | `tools/pow_differential.sh all` | deterministic `pow` vs the native glibc `pow`: 0 mismatches over 5,900,000 domain inputs and 0 over 20,000,000 unrestricted finite inputs |
| `summary.txt` | `tools/verify_examples.sh all` | one line per reference variant: 153 IDENTICAL / 26 DIFF / 20 EXCLUDED |
| `classify_diffs.txt` | `tools/classify_diffs.sh` | second pass over the non-IDENTICAL variants — `EXACT` / `SQUEEZE` (`tr -s ' '`) / `WS` (`diff -w`). A `SQUEEZE same` row means every printed value is byte-identical and only column spacing differs |
| `pristine_c_comparison.txt` | `tools/compare_pristine_c.sh` + `tools/compare_lapack_substituted.sh` | the port-defect proof: each divergent variant run as Rust, as pristine upstream C (cmake + gcc 13.3.0), and against the shipped reference. **`RS_vs_C` is `same` on all 26 rows** — the port matches the C byte-for-byte and both differ from the stale reference |
| `glibc-sweep.txt` | `tools/glibc_sweep.sh` | per-function libm fingerprints across Debian 12 / Ubuntu 24.04 / Fedora 41 / Debian 13 / Arch, plus Alpine (musl) as the negative control. This is what turned the distribution-coverage claim from an argument into a measurement — and corrected it |
| `gate-<image>.txt` | `tools/gate_in_container.sh` | the full 199-variant gate re-run natively inside another distribution's container |

Regenerate all four with `tools/wsl_sync_build.sh evidence` from Windows, or
by running the three scripts directly on a Linux host with the upstream
SUNDIALS 7.8.0 C tree as this workspace's parent directory.

`classify_diffs.txt` carries one row the gate does not count as a
divergence: `kinLaplace_picard_kry`, which is `WS same` and is handled by
`verify_examples.sh`'s symmetric noise filter, so it reports IDENTICAL in
`summary.txt`.
