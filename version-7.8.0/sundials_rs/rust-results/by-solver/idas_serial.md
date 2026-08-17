# IDAS — Rust examples (`crates/idas_rs/examples`)

22 (example, argv) variants. Run one yourself with:

```bash
cargo run --release -p idas_rs --example <name> -- <argv>
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
| 1 | `idasAkzoNob_ASAi_dns` | _(none)_ | 0 | OK | 0.106 | 567 | `323408d02fe4d304` | [stdout](../raw/idas/serial/idasAkzoNob_ASAi_dns.stdout) · [meta](../raw/idas/serial/idasAkzoNob_ASAi_dns.meta) |
| 2 | `idasAkzoNob_dns` | _(none)_ | 0 | OK | 0.105 | 3036 | `dd69bfec8d242eca` | [stdout](../raw/idas/serial/idasAkzoNob_dns.stdout) · [meta](../raw/idas/serial/idasAkzoNob_dns.meta) |
| 3 | `idasAnalytic_mels` | _(none)_ | 0 | OK | 0.105 | 842 | `e67c6d65051515df` | [stdout](../raw/idas/serial/idasAnalytic_mels.stdout) · [meta](../raw/idas/serial/idasAnalytic_mels.meta) |
| 4 | `idasAnalytic_mels` | `idas.init_step 1e-5` | 0 | OK | 0.106 | 842 | `85c169b7a9d39601` | [stdout](../raw/idas/serial/idasAnalytic_mels__idas.init_step_1e-5.stdout) · [meta](../raw/idas/serial/idasAnalytic_mels__idas.init_step_1e-5.meta) |
| 5 | `idasFoodWeb_bnd` | _(none)_ | 0 | OK | 0.506 | 1541 | `f48a0923fb034160` | [stdout](../raw/idas/serial/idasFoodWeb_bnd.stdout) · [meta](../raw/idas/serial/idasFoodWeb_bnd.meta) |
| 6 | `idasHeat2D_bnd` | _(none)_ | 0 | OK | 0.105 | 1478 | `3dc0d5ea6d93ec20` | [stdout](../raw/idas/serial/idasHeat2D_bnd.stdout) · [meta](../raw/idas/serial/idasHeat2D_bnd.meta) |
| 7 | `idasHeat2D_kry` | _(none)_ | 0 | OK | 0.105 | 2665 | `0d9772e8099f0f4a` | [stdout](../raw/idas/serial/idasHeat2D_kry.stdout) · [meta](../raw/idas/serial/idasHeat2D_kry.meta) |
| 8 | `idasHessian_ASA_FSA` | _(none)_ | 0 | OK | 0.106 | 1258 | `e91863a731e78eb0` | [stdout](../raw/idas/serial/idasHessian_ASA_FSA.stdout) · [meta](../raw/idas/serial/idasHessian_ASA_FSA.meta) |
| 9 | `idasKrylovDemo_ls` | _(none)_ | 0 | OK | 0.105 | 4820 | `4dc9af3533dc886f` | [stdout](../raw/idas/serial/idasKrylovDemo_ls.stdout) · [meta](../raw/idas/serial/idasKrylovDemo_ls.meta) |
| 10 | `idasKrylovDemo_ls` | `1` | 0 | OK | 0.105 | 4820 | `4dc9af3533dc886f` | [stdout](../raw/idas/serial/idasKrylovDemo_ls__1.stdout) · [meta](../raw/idas/serial/idasKrylovDemo_ls__1.meta) |
| 11 | `idasKrylovDemo_ls` | `2` | 0 | OK | 0.105 | 4820 | `4dc9af3533dc886f` | [stdout](../raw/idas/serial/idasKrylovDemo_ls__2.stdout) · [meta](../raw/idas/serial/idasKrylovDemo_ls__2.meta) |
| 12 | `idasRoberts_ASAi_dns` | _(none)_ | 0 | OK | 0.105 | 4644 | `6393797cd4df94a3` | [stdout](../raw/idas/serial/idasRoberts_ASAi_dns.stdout) · [meta](../raw/idas/serial/idasRoberts_ASAi_dns.meta) |
| 13 | `idasRoberts_ASAi_klu` | _(none)_ | 0 | OK | 0.106 | 1200 | `64b19fd4a7704245` | [stdout](../raw/idas/serial/idasRoberts_ASAi_klu.stdout) · [meta](../raw/idas/serial/idasRoberts_ASAi_klu.meta) |
| 14 | `idasRoberts_ASAi_sps` | _(none)_ | - | NOT_PORTED | - | 0 | `-` | [stdout](../raw/idas/serial/idasRoberts_ASAi_sps.stdout) · [meta](../raw/idas/serial/idasRoberts_ASAi_sps.meta) |
| 15 | `idasRoberts_FSA_dns` | `-sensi stg t` | 0 | OK | 0.105 | 7142 | `803d81f721357fff` | [stdout](../raw/idas/serial/idasRoberts_FSA_dns__-sensi_stg_t.stdout) · [meta](../raw/idas/serial/idasRoberts_FSA_dns__-sensi_stg_t.meta) |
| 16 | `idasRoberts_FSA_klu` | `-sensi stg t` | 0 | OK | 0.105 | 5941 | `d42c79f11b30d884` | [stdout](../raw/idas/serial/idasRoberts_FSA_klu__-sensi_stg_t.stdout) · [meta](../raw/idas/serial/idasRoberts_FSA_klu__-sensi_stg_t.meta) |
| 17 | `idasRoberts_FSA_sps` | `-sensi stg t` | - | NOT_PORTED | - | 0 | `-` | [stdout](../raw/idas/serial/idasRoberts_FSA_sps__-sensi_stg_t.stdout) · [meta](../raw/idas/serial/idasRoberts_FSA_sps__-sensi_stg_t.meta) |
| 18 | `idasRoberts_dns` | _(none)_ | 0 | OK | 0.106 | 2692 | `74a2ad3da56b9a0c` | [stdout](../raw/idas/serial/idasRoberts_dns.stdout) · [meta](../raw/idas/serial/idasRoberts_dns.meta) |
| 19 | `idasRoberts_klu` | _(none)_ | 0 | OK | 0.106 | 1926 | `9fe77db586be1d71` | [stdout](../raw/idas/serial/idasRoberts_klu.stdout) · [meta](../raw/idas/serial/idasRoberts_klu.meta) |
| 20 | `idasRoberts_sps` | _(none)_ | - | NOT_PORTED | - | 0 | `-` | [stdout](../raw/idas/serial/idasRoberts_sps.stdout) · [meta](../raw/idas/serial/idasRoberts_sps.meta) |
| 21 | `idasSlCrank_FSA_dns` | _(none)_ | 0 | OK | 0.105 | 1027 | `b331e2f65e6ffffe` | [stdout](../raw/idas/serial/idasSlCrank_FSA_dns.stdout) · [meta](../raw/idas/serial/idasSlCrank_FSA_dns.meta) |
| 22 | `idasSlCrank_dns` | _(none)_ | 0 | OK | 0.105 | 2546 | `a10754d1065e7e74` | [stdout](../raw/idas/serial/idasSlCrank_dns.stdout) · [meta](../raw/idas/serial/idasSlCrank_dns.meta) |
