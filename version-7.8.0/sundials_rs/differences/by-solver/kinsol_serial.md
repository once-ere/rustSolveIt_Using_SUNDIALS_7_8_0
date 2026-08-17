# KINSOL — C vs Rust (`examples/kinsol/serial`)

| # | example | argv | class | diff lines / total | worst rel | worst ulp | diff |
|---:|---|---|---|---:|---:|---:|---|
| 1 | `kinAnalytic_fp` | _(none)_ | IDENTICAL | — | — | — | — |
| 2 | `kinAnalytic_fp` | `--damping_fn` | IDENTICAL | — | — | — | — |
| 3 | `kinAnalytic_fp` | `--damping_fp 0.5` | IDENTICAL | — | — | — | — |
| 4 | `kinAnalytic_fp` | `--m_aa 2` | IDENTICAL | — | — | — | — |
| 5 | `kinAnalytic_fp` | `--m_aa 2 --damping_aa 0.5` | IDENTICAL | — | — | — | — |
| 6 | `kinAnalytic_fp` | `--m_aa 2 --damping_fn` | IDENTICAL | — | — | — | — |
| 7 | `kinAnalytic_fp` | `--m_aa 2 --delay_aa 2` | IDENTICAL | — | — | — | — |
| 8 | `kinAnalytic_fp` | `--m_aa 2 --orth_aa 1` | IDENTICAL | — | — | — | — |
| 9 | `kinAnalytic_fp` | `--m_aa 2 --orth_aa 2` | IDENTICAL | — | — | — | — |
| 10 | `kinAnalytic_fp` | `--m_aa 2 --orth_aa 3` | IDENTICAL | — | — | — | — |
| 11 | `kinAnalytic_fp` | `--m_aa 3 --depth_fn` | IDENTICAL | — | — | — | — |
| 12 | `kinFerTron_dns` | _(none)_ | IDENTICAL | — | — | — | — |
| 13 | `kinFerTron_klu` | _(none)_ | IDENTICAL | — | — | — | — |
| 14 | `kinFoodWeb_kry` | _(none)_ | IDENTICAL | — | — | — | — |
| 15 | `kinKrylovDemo_ls` | _(none)_ | IDENTICAL | — | — | — | — |
| 16 | `kinLaplace_bnd` | _(none)_ | IDENTICAL | — | — | — | — |
| 17 | `kinLaplace_picard_bnd` | _(none)_ | IDENTICAL | — | — | — | — |
| 18 | `kinLaplace_picard_kry` | _(none)_ | IDENTICAL | — | — | — | — |
| 19 | `kinRoberts_fp` | _(none)_ | IDENTICAL | — | — | — | — |
| 20 | `kinRoberts_fp` | `kinsol.m_aa 1` | IDENTICAL | — | — | — | — |
| 21 | `kinRoboKin_dns` | _(none)_ | IDENTICAL | — | — | — | — |
| 22 | `kinRoboKin_slu` | _(none)_ | NOT_PORTED | — | — | — | — |

