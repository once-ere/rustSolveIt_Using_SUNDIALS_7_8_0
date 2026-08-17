# CVODE — Rust examples (`crates/cvode_rs/examples`)

24 (example, argv) variants. Run one yourself with:

```bash
cargo run --release -p cvode_rs --example <name> -- <argv>
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
| 1 | `cvAdvDiff_bnd` | _(none)_ | 0 | OK | 0.105 | 848 | `2ea0937bb293ce55` | [stdout](../raw/cvode/serial/cvAdvDiff_bnd.stdout) · [meta](../raw/cvode/serial/cvAdvDiff_bnd.meta) |
| 2 | `cvAdvDiff_bndL` | _(none)_ | 0 | OK | 0.105 | 848 | `2ea0937bb293ce55` | [stdout](../raw/cvode/serial/cvAdvDiff_bndL.stdout) · [meta](../raw/cvode/serial/cvAdvDiff_bndL.meta) |
| 3 | `cvAnalytic_mels` | _(none)_ | 0 | OK | 0.105 | 770 | `c13a8a670343e246` | [stdout](../raw/cvode/serial/cvAnalytic_mels.stdout) · [meta](../raw/cvode/serial/cvAnalytic_mels.meta) |
| 4 | `cvDirectDemo_ls` | _(none)_ | 0 | OK | 0.106 | 17673 | `a19abd5087bb90d0` | [stdout](../raw/cvode/serial/cvDirectDemo_ls.stdout) · [meta](../raw/cvode/serial/cvDirectDemo_ls.meta) |
| 5 | `cvDisc_dns` | _(none)_ | 0 | OK | 0.106 | 3360 | `77b1dd2393e53661` | [stdout](../raw/cvode/serial/cvDisc_dns.stdout) · [meta](../raw/cvode/serial/cvDisc_dns.meta) |
| 6 | `cvDiurnal_kry` | _(none)_ | 0 | OK | 0.106 | 2860 | `3d5bdd944af9101c` | [stdout](../raw/cvode/serial/cvDiurnal_kry.stdout) · [meta](../raw/cvode/serial/cvDiurnal_kry.meta) |
| 7 | `cvDiurnal_kry_bp` | _(none)_ | 0 | OK | 0.106 | 6047 | `0651440486243145` | [stdout](../raw/cvode/serial/cvDiurnal_kry_bp.stdout) · [meta](../raw/cvode/serial/cvDiurnal_kry_bp.meta) |
| 8 | `cvKrylovDemo_ls` | _(none)_ | 0 | OK | 0.105 | 11712 | `24cbfc892a79f788` | [stdout](../raw/cvode/serial/cvKrylovDemo_ls.stdout) · [meta](../raw/cvode/serial/cvKrylovDemo_ls.meta) |
| 9 | `cvKrylovDemo_ls` | `0 1` | 0 | OK | 0.106 | 2472 | `3b2b30835739bd73` | [stdout](../raw/cvode/serial/cvKrylovDemo_ls__0_1.stdout) · [meta](../raw/cvode/serial/cvKrylovDemo_ls__0_1.meta) |
| 10 | `cvKrylovDemo_ls` | `1` | 0 | OK | 0.105 | 11712 | `24cbfc892a79f788` | [stdout](../raw/cvode/serial/cvKrylovDemo_ls__1.stdout) · [meta](../raw/cvode/serial/cvKrylovDemo_ls__1.meta) |
| 11 | `cvKrylovDemo_ls` | `2` | 0 | OK | 0.105 | 11712 | `24cbfc892a79f788` | [stdout](../raw/cvode/serial/cvKrylovDemo_ls__2.stdout) · [meta](../raw/cvode/serial/cvKrylovDemo_ls__2.meta) |
| 12 | `cvKrylovDemo_prec` | _(none)_ | 0 | OK | 0.105 | 26471 | `b1cf95a1d917a850` | [stdout](../raw/cvode/serial/cvKrylovDemo_prec.stdout) · [meta](../raw/cvode/serial/cvKrylovDemo_prec.meta) |
| 13 | `cvParticle_dns` | _(none)_ | 0 | OK | 0.105 | 885 | `fa77abe19cddd1f7` | [stdout](../raw/cvode/serial/cvParticle_dns.stdout) · [meta](../raw/cvode/serial/cvParticle_dns.meta) |
| 14 | `cvPendulum_dns` | _(none)_ | 0 | OK | 0.106 | 1900 | `66778152510991e4` | [stdout](../raw/cvode/serial/cvPendulum_dns.stdout) · [meta](../raw/cvode/serial/cvPendulum_dns.meta) |
| 15 | `cvRoberts_block_klu` | _(none)_ | 0 | OK | 0.105 | 1198 | `34fb2cfe2ef09780` | [stdout](../raw/cvode/serial/cvRoberts_block_klu.stdout) · [meta](../raw/cvode/serial/cvRoberts_block_klu.meta) |
| 16 | `cvRoberts_dns` | _(none)_ | 0 | OK | 0.106 | 2217 | `c23a663fee1d66b7` | [stdout](../raw/cvode/serial/cvRoberts_dns.stdout) · [meta](../raw/cvode/serial/cvRoberts_dns.meta) |
| 17 | `cvRoberts_dnsL` | _(none)_ | 0 | OK | 0.106 | 1261 | `abf33560958e0f9f` | [stdout](../raw/cvode/serial/cvRoberts_dnsL.stdout) · [meta](../raw/cvode/serial/cvRoberts_dnsL.meta) |
| 18 | `cvRoberts_dns_constraints` | _(none)_ | 0 | OK | 0.105 | 1261 | `106ba20145b7ce7c` | [stdout](../raw/cvode/serial/cvRoberts_dns_constraints.stdout) · [meta](../raw/cvode/serial/cvRoberts_dns_constraints.meta) |
| 19 | `cvRoberts_dns_negsol` | _(none)_ | 0 | OK | 0.106 | 2409 | `911bb744d63c8c33` | [stdout](../raw/cvode/serial/cvRoberts_dns_negsol.stdout) · [meta](../raw/cvode/serial/cvRoberts_dns_negsol.meta) |
| 20 | `cvRoberts_dns_uw` | _(none)_ | 0 | OK | 0.106 | 1261 | `abf33560958e0f9f` | [stdout](../raw/cvode/serial/cvRoberts_dns_uw.stdout) · [meta](../raw/cvode/serial/cvRoberts_dns_uw.meta) |
| 21 | `cvRoberts_klu` | _(none)_ | 0 | OK | 0.105 | 1245 | `e2319ee837288d96` | [stdout](../raw/cvode/serial/cvRoberts_klu.stdout) · [meta](../raw/cvode/serial/cvRoberts_klu.meta) |
| 22 | `cvRoberts_sps` | _(none)_ | - | NOT_PORTED | - | 0 | `-` | [stdout](../raw/cvode/serial/cvRoberts_sps.stdout) · [meta](../raw/cvode/serial/cvRoberts_sps.meta) |
| 23 | `cvRocket_dns` | _(none)_ | 0 | OK | 0.106 | 4212 | `2887cbbe9072a465` | [stdout](../raw/cvode/serial/cvRocket_dns.stdout) · [meta](../raw/cvode/serial/cvRocket_dns.meta) |
| 24 | `cvVdp_auto_nls` | _(none)_ | 0 | OK | 0.105 | 2403 | `2046a2142816e67a` | [stdout](../raw/cvode/serial/cvVdp_auto_nls.stdout) · [meta](../raw/cvode/serial/cvVdp_auto_nls.meta) |
