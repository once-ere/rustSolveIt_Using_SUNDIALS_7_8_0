# KINSOL — Rust examples (`crates/kinsol_rs/examples`)

22 (example, argv) variants. Run one yourself with:

```bash
cargo run --release -p kinsol_rs --example <name> -- <argv>
```

That reproduces every row marked `OK`. It does **not** work for a
`NOT_PORTED` row: those examples have no `[[example]]` entry in any
`Cargo.toml`, because no Rust translation exists.

`seconds` is wall time **including harness overhead** — the runner
brackets each example with two `date` subprocesses and a subshell, a
floor of roughly 0.1 s. Treat it as a liveness signal, not a
benchmark: most of these examples finish in under 10 ms.

| # | example | argv | exit | status | seconds | stdout bytes | sha256 (first 16) | raw |
|---:|---|---|---:|---|---:|---:|---|---|
| 1 | `kinAnalytic_fp` | _(none)_ | 0 | OK | 0.106 | 708 | `98998bade74550e6` | [stdout](../raw/kinsol/serial/kinAnalytic_fp.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp.meta) |
| 2 | `kinAnalytic_fp` | `--damping_fn` | 0 | OK | 0.106 | 714 | `7afe99c8ef1e0b37` | [stdout](../raw/kinsol/serial/kinAnalytic_fp__--damping_fn.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp__--damping_fn.meta) |
| 3 | `kinAnalytic_fp` | `--damping_fp 0.5` | 0 | OK | 0.106 | 717 | `f2262e780b477be1` | [stdout](../raw/kinsol/serial/kinAnalytic_fp__--damping_fp_0.5.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp__--damping_fp_0.5.meta) |
| 4 | `kinAnalytic_fp` | `--m_aa 2` | 0 | OK | 0.106 | 707 | `1835ebe4be87f8ee` | [stdout](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2.meta) |
| 5 | `kinAnalytic_fp` | `--m_aa 2 --damping_aa 0.5` | 0 | OK | 0.106 | 722 | `6696251d2c61eb86` | [stdout](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--damping_aa_0.5.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--damping_aa_0.5.meta) |
| 6 | `kinAnalytic_fp` | `--m_aa 2 --damping_fn` | 0 | OK | 0.107 | 711 | `49768800e80847cd` | [stdout](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--damping_fn.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--damping_fn.meta) |
| 7 | `kinAnalytic_fp` | `--m_aa 2 --delay_aa 2` | 0 | OK | 0.106 | 708 | `370eaba2106470f6` | [stdout](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--delay_aa_2.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--delay_aa_2.meta) |
| 8 | `kinAnalytic_fp` | `--m_aa 2 --orth_aa 1` | 0 | OK | 0.106 | 707 | `5e6bc473ab82cbc0` | [stdout](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--orth_aa_1.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--orth_aa_1.meta) |
| 9 | `kinAnalytic_fp` | `--m_aa 2 --orth_aa 2` | 0 | OK | 0.107 | 707 | `ede30554c343b27f` | [stdout](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--orth_aa_2.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--orth_aa_2.meta) |
| 10 | `kinAnalytic_fp` | `--m_aa 2 --orth_aa 3` | 0 | OK | 0.108 | 707 | `b07bc414f32f23a3` | [stdout](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--orth_aa_3.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_2_--orth_aa_3.meta) |
| 11 | `kinAnalytic_fp` | `--m_aa 3 --depth_fn` | 0 | OK | 0.106 | 707 | `21f45b4b3887b860` | [stdout](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_3_--depth_fn.stdout) · [meta](../raw/kinsol/serial/kinAnalytic_fp__--m_aa_3_--depth_fn.meta) |
| 12 | `kinFerTron_dns` | _(none)_ | 0 | OK | 0.107 | 1503 | `6ac846fe2cdec1d5` | [stdout](../raw/kinsol/serial/kinFerTron_dns.stdout) · [meta](../raw/kinsol/serial/kinFerTron_dns.meta) |
| 13 | `kinFerTron_klu` | _(none)_ | 0 | OK | 0.107 | 1399 | `2cbbde2630ba8259` | [stdout](../raw/kinsol/serial/kinFerTron_klu.stdout) · [meta](../raw/kinsol/serial/kinFerTron_klu.meta) |
| 14 | `kinFoodWeb_kry` | _(none)_ | 0 | OK | 0.107 | 789 | `931d25d83cba3c00` | [stdout](../raw/kinsol/serial/kinFoodWeb_kry.stdout) · [meta](../raw/kinsol/serial/kinFoodWeb_kry.meta) |
| 15 | `kinKrylovDemo_ls` | _(none)_ | 0 | OK | 0.105 | 3420 | `27535204b5a07f97` | [stdout](../raw/kinsol/serial/kinKrylovDemo_ls.stdout) · [meta](../raw/kinsol/serial/kinKrylovDemo_ls.meta) |
| 16 | `kinLaplace_bnd` | _(none)_ | 0 | OK | 0.105 | 1816 | `54c7e2398c7384e8` | [stdout](../raw/kinsol/serial/kinLaplace_bnd.stdout) · [meta](../raw/kinsol/serial/kinLaplace_bnd.meta) |
| 17 | `kinLaplace_picard_bnd` | _(none)_ | 0 | OK | 0.105 | 1762 | `73402c23fd8ab615` | [stdout](../raw/kinsol/serial/kinLaplace_picard_bnd.stdout) · [meta](../raw/kinsol/serial/kinLaplace_picard_bnd.meta) |
| 18 | `kinLaplace_picard_kry` | _(none)_ | 0 | OK | 0.106 | 1761 | `02de672f876b0057` | [stdout](../raw/kinsol/serial/kinLaplace_picard_kry.stdout) · [meta](../raw/kinsol/serial/kinLaplace_picard_kry.meta) |
| 19 | `kinRoberts_fp` | _(none)_ | 0 | OK | 0.106 | 546 | `aad4d127168471e7` | [stdout](../raw/kinsol/serial/kinRoberts_fp.stdout) · [meta](../raw/kinsol/serial/kinRoberts_fp.meta) |
| 20 | `kinRoberts_fp` | `kinsol.m_aa 1` | 0 | OK | 0.106 | 546 | `fb18d2ee22ddf344` | [stdout](../raw/kinsol/serial/kinRoberts_fp__kinsol.m_aa_1.stdout) · [meta](../raw/kinsol/serial/kinRoberts_fp__kinsol.m_aa_1.meta) |
| 21 | `kinRoboKin_dns` | _(none)_ | 0 | OK | 0.107 | 1483 | `f73ec9e6acefec29` | [stdout](../raw/kinsol/serial/kinRoboKin_dns.stdout) · [meta](../raw/kinsol/serial/kinRoboKin_dns.meta) |
| 22 | `kinRoboKin_slu` | _(none)_ | - | NOT_PORTED | - | 0 | `-` | [stdout](../raw/kinsol/serial/kinRoboKin_slu.stdout) · [meta](../raw/kinsol/serial/kinRoboKin_slu.meta) |
