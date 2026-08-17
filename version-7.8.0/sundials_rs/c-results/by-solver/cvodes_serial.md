# CVODES — C examples (`examples/cvodes/serial`)

36 (example, argv) variants, executed on the machine described in
[`../README.md`](../README.md).

`stdout bytes` and `sha256` are of the captured stdout stream; re-run
`tools/c_examples_run.sh` and they must reproduce exactly.

| # | example | argv | exit | status | seconds | stdout bytes | sha256 (first 16) | raw |
|---:|---|---|---:|---|---:|---:|---|---|
| 1 | `cvsAdvDiff_ASAi_bnd` | _(none)_ | 0 | OK | 0.206 | 273 | `85d580b503344228` | [stdout](../raw/cvodes/serial/cvsAdvDiff_ASAi_bnd.stdout) · [meta](../raw/cvodes/serial/cvsAdvDiff_ASAi_bnd.meta) |
| 2 | `cvsAdvDiff_FSA_non` | `-sensi sim t` | 0 | OK | 0.207 | 3262 | `7e6b817908a5df5d` | [stdout](../raw/cvodes/serial/cvsAdvDiff_FSA_non__-sensi_sim_t.stdout) · [meta](../raw/cvodes/serial/cvsAdvDiff_FSA_non__-sensi_sim_t.meta) |
| 3 | `cvsAdvDiff_FSA_non` | `-sensi stg t` | 0 | OK | 0.206 | 3259 | `aba1080724c2adf5` | [stdout](../raw/cvodes/serial/cvsAdvDiff_FSA_non__-sensi_stg_t.stdout) · [meta](../raw/cvodes/serial/cvsAdvDiff_FSA_non__-sensi_stg_t.meta) |
| 4 | `cvsAdvDiff_bnd` | _(none)_ | 0 | OK | 0.206 | 850 | `24b74965dc903776` | [stdout](../raw/cvodes/serial/cvsAdvDiff_bnd.stdout) · [meta](../raw/cvodes/serial/cvsAdvDiff_bnd.meta) |
| 5 | `cvsAdvDiff_bndL` | _(none)_ | 0 | OK | 0.207 | 863 | `6b6ec3b83bbcaeb2` | [stdout](../raw/cvodes/serial/cvsAdvDiff_bndL.stdout) · [meta](../raw/cvodes/serial/cvsAdvDiff_bndL.meta) |
| 6 | `cvsAnalytic_mels` | _(none)_ | 0 | OK | 0.206 | 770 | `c13a8a670343e246` | [stdout](../raw/cvodes/serial/cvsAnalytic_mels.stdout) · [meta](../raw/cvodes/serial/cvsAnalytic_mels.meta) |
| 7 | `cvsAnalytic_mels` | `cvodes.max_order 3` | 0 | OK | 0.206 | 771 | `8b5710623edaecaf` | [stdout](../raw/cvodes/serial/cvsAnalytic_mels__cvodes.max_order_3.stdout) · [meta](../raw/cvodes/serial/cvsAnalytic_mels__cvodes.max_order_3.meta) |
| 8 | `cvsDirectDemo_ls` | _(none)_ | 0 | OK | 0.206 | 17713 | `7537f64339279cdf` | [stdout](../raw/cvodes/serial/cvsDirectDemo_ls.stdout) · [meta](../raw/cvodes/serial/cvsDirectDemo_ls.meta) |
| 9 | `cvsDiurnal_FSA_kry` | `-sensi sim t` | 0 | OK | 0.306 | 8944 | `f021beeff79601c6` | [stdout](../raw/cvodes/serial/cvsDiurnal_FSA_kry__-sensi_sim_t.stdout) · [meta](../raw/cvodes/serial/cvsDiurnal_FSA_kry__-sensi_sim_t.meta) |
| 10 | `cvsDiurnal_FSA_kry` | `-sensi stg t` | 0 | OK | 0.206 | 8941 | `5bbde57fae6067b2` | [stdout](../raw/cvodes/serial/cvsDiurnal_FSA_kry__-sensi_stg_t.stdout) · [meta](../raw/cvodes/serial/cvsDiurnal_FSA_kry__-sensi_stg_t.meta) |
| 11 | `cvsDiurnal_kry` | _(none)_ | 0 | OK | 0.206 | 2860 | `175cd73d3f78930b` | [stdout](../raw/cvodes/serial/cvsDiurnal_kry.stdout) · [meta](../raw/cvodes/serial/cvsDiurnal_kry.meta) |
| 12 | `cvsDiurnal_kry_bp` | _(none)_ | 0 | OK | 0.207 | 6047 | `ff9fe30eabbd17ba` | [stdout](../raw/cvodes/serial/cvsDiurnal_kry_bp.stdout) · [meta](../raw/cvodes/serial/cvsDiurnal_kry_bp.meta) |
| 13 | `cvsFoodWeb_ASAi_kry` | _(none)_ | 0 | OK | 0.507 | 991 | `dbb0f31483d7b0b9` | [stdout](../raw/cvodes/serial/cvsFoodWeb_ASAi_kry.stdout) · [meta](../raw/cvodes/serial/cvsFoodWeb_ASAi_kry.meta) |
| 14 | `cvsFoodWeb_ASAp_kry` | _(none)_ | 0 | OK | 0.707 | 961 | `ab4dd07a66053a31` | [stdout](../raw/cvodes/serial/cvsFoodWeb_ASAp_kry.stdout) · [meta](../raw/cvodes/serial/cvsFoodWeb_ASAp_kry.meta) |
| 15 | `cvsHessian_ASA_FSA` | _(none)_ | 0 | OK | 0.206 | 2307 | `b3079e8ddf9f5528` | [stdout](../raw/cvodes/serial/cvsHessian_ASA_FSA.stdout) · [meta](../raw/cvodes/serial/cvsHessian_ASA_FSA.meta) |
| 16 | `cvsKrylovDemo_ls` | _(none)_ | 0 | OK | 0.206 | 11712 | `6f0cda0e35d96b82` | [stdout](../raw/cvodes/serial/cvsKrylovDemo_ls.stdout) · [meta](../raw/cvodes/serial/cvsKrylovDemo_ls.meta) |
| 17 | `cvsKrylovDemo_ls` | `0 1` | 0 | OK | 0.206 | 11712 | `6f0cda0e35d96b82` | [stdout](../raw/cvodes/serial/cvsKrylovDemo_ls__0_1.stdout) · [meta](../raw/cvodes/serial/cvsKrylovDemo_ls__0_1.meta) |
| 18 | `cvsKrylovDemo_ls` | `1` | 0 | OK | 0.206 | 11712 | `6f0cda0e35d96b82` | [stdout](../raw/cvodes/serial/cvsKrylovDemo_ls__1.stdout) · [meta](../raw/cvodes/serial/cvsKrylovDemo_ls__1.meta) |
| 19 | `cvsKrylovDemo_ls` | `2` | 0 | OK | 0.206 | 11712 | `6f0cda0e35d96b82` | [stdout](../raw/cvodes/serial/cvsKrylovDemo_ls__2.stdout) · [meta](../raw/cvodes/serial/cvsKrylovDemo_ls__2.meta) |
| 20 | `cvsKrylovDemo_prec` | _(none)_ | 0 | OK | 0.206 | 26472 | `8e4b506a75e1c2f0` | [stdout](../raw/cvodes/serial/cvsKrylovDemo_prec.stdout) · [meta](../raw/cvodes/serial/cvsKrylovDemo_prec.meta) |
| 21 | `cvsLotkaVolterra_ASA` | _(none)_ | 0 | OK | 0.205 | 405 | `44d47a6d5cd349cc` | [stdout](../raw/cvodes/serial/cvsLotkaVolterra_ASA.stdout) · [meta](../raw/cvodes/serial/cvsLotkaVolterra_ASA.meta) |
| 22 | `cvsParticle_dns` | _(none)_ | 0 | OK | 0.206 | 885 | `fa77abe19cddd1f7` | [stdout](../raw/cvodes/serial/cvsParticle_dns.stdout) · [meta](../raw/cvodes/serial/cvsParticle_dns.meta) |
| 23 | `cvsPendulum_dns` | _(none)_ | 0 | OK | 0.206 | 1900 | `66778152510991e4` | [stdout](../raw/cvodes/serial/cvsPendulum_dns.stdout) · [meta](../raw/cvodes/serial/cvsPendulum_dns.meta) |
| 24 | `cvsRoberts_ASAi_dns` | _(none)_ | 0 | OK | 0.206 | 5336 | `9e3b3a9e1e06b09e` | [stdout](../raw/cvodes/serial/cvsRoberts_ASAi_dns.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_ASAi_dns.meta) |
| 25 | `cvsRoberts_ASAi_dns_constraints` | _(none)_ | 0 | OK | 0.206 | 1879 | `45a5a45c32e992f9` | [stdout](../raw/cvodes/serial/cvsRoberts_ASAi_dns_constraints.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_ASAi_dns_constraints.meta) |
| 26 | `cvsRoberts_ASAi_klu` | _(none)_ | 0 | OK | 0.206 | 1879 | `03d3bcf1662eae90` | [stdout](../raw/cvodes/serial/cvsRoberts_ASAi_klu.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_ASAi_klu.meta) |
| 27 | `cvsRoberts_FSA_dns` | `-sensi sim t` | 0 | OK | 0.206 | 6280 | `bd0b1b6688422332` | [stdout](../raw/cvodes/serial/cvsRoberts_FSA_dns__-sensi_sim_t.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_FSA_dns__-sensi_sim_t.meta) |
| 28 | `cvsRoberts_FSA_dns` | `-sensi stg1 t` | 0 | OK | 0.206 | 6513 | `928aa78347d646e0` | [stdout](../raw/cvodes/serial/cvsRoberts_FSA_dns__-sensi_stg1_t.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_FSA_dns__-sensi_stg1_t.meta) |
| 29 | `cvsRoberts_FSA_dns_Switch` | _(none)_ | 0 | OK | 0.206 | 1849 | `2e523db061c19214` | [stdout](../raw/cvodes/serial/cvsRoberts_FSA_dns_Switch.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_FSA_dns_Switch.meta) |
| 30 | `cvsRoberts_FSA_dns_constraints` | `-sensi stg1 t` | 0 | OK | 0.206 | 5277 | `3f2c9e3dcea16dc4` | [stdout](../raw/cvodes/serial/cvsRoberts_FSA_dns_constraints__-sensi_stg1_t.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_FSA_dns_constraints__-sensi_stg1_t.meta) |
| 31 | `cvsRoberts_FSA_klu` | `-sensi stg1 t` | 0 | OK | 0.206 | 5262 | `b92ee0b442c99fef` | [stdout](../raw/cvodes/serial/cvsRoberts_FSA_klu__-sensi_stg1_t.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_FSA_klu__-sensi_stg1_t.meta) |
| 32 | `cvsRoberts_dns` | _(none)_ | 0 | OK | 0.206 | 2217 | `c23a663fee1d66b7` | [stdout](../raw/cvodes/serial/cvsRoberts_dns.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_dns.meta) |
| 33 | `cvsRoberts_dnsL` | _(none)_ | 0 | OK | 0.206 | 1261 | `abf33560958e0f9f` | [stdout](../raw/cvodes/serial/cvsRoberts_dnsL.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_dnsL.meta) |
| 34 | `cvsRoberts_dns_constraints` | _(none)_ | 0 | OK | 0.205 | 1261 | `106ba20145b7ce7c` | [stdout](../raw/cvodes/serial/cvsRoberts_dns_constraints.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_dns_constraints.meta) |
| 35 | `cvsRoberts_dns_uw` | _(none)_ | 0 | OK | 0.206 | 1261 | `abf33560958e0f9f` | [stdout](../raw/cvodes/serial/cvsRoberts_dns_uw.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_dns_uw.meta) |
| 36 | `cvsRoberts_klu` | _(none)_ | 0 | OK | 0.206 | 1245 | `de88e6bc45119b34` | [stdout](../raw/cvodes/serial/cvsRoberts_klu.stdout) · [meta](../raw/cvodes/serial/cvsRoberts_klu.meta) |
