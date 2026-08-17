# IDA — C examples (`examples/ida/serial`)

13 (example, argv) variants, executed on the machine described in
[`../README.md`](../README.md).

`stdout bytes` and `sha256` are of the captured stdout stream; re-run
`tools/c_examples_run.sh` and they must reproduce exactly.

| # | example | argv | exit | status | seconds | stdout bytes | sha256 (first 16) | raw |
|---:|---|---|---:|---|---:|---:|---|---|
| 1 | `idaAnalytic_mels` | _(none)_ | 0 | OK | 0.206 | 756 | `bf1750471b116227` | [stdout](../raw/ida/serial/idaAnalytic_mels.stdout) · [meta](../raw/ida/serial/idaAnalytic_mels.meta) |
| 2 | `idaAnalytic_mels` | `ida.scalar_tolerances 1e-3 1e-8` | 0 | OK | 0.206 | 751 | `a52123c7523ee23a` | [stdout](../raw/ida/serial/idaAnalytic_mels__ida.scalar_tolerances_1e-3_1e-8.stdout) · [meta](../raw/ida/serial/idaAnalytic_mels__ida.scalar_tolerances_1e-3_1e-8.meta) |
| 3 | `idaFoodWeb_bnd` | _(none)_ | 0 | OK | 0.206 | 1540 | `7a7b7e86364360d0` | [stdout](../raw/ida/serial/idaFoodWeb_bnd.stdout) · [meta](../raw/ida/serial/idaFoodWeb_bnd.meta) |
| 4 | `idaFoodWeb_kry` | _(none)_ | 0 | OK | 0.207 | 1516 | `94053386b480102c` | [stdout](../raw/ida/serial/idaFoodWeb_kry.stdout) · [meta](../raw/ida/serial/idaFoodWeb_kry.meta) |
| 5 | `idaHeat2D_bnd` | _(none)_ | 0 | OK | 0.206 | 1465 | `81d7aa7ac1c072d7` | [stdout](../raw/ida/serial/idaHeat2D_bnd.stdout) · [meta](../raw/ida/serial/idaHeat2D_bnd.meta) |
| 6 | `idaHeat2D_klu` | _(none)_ | 0 | OK | 0.206 | 1355 | `54f0d048c6173673` | [stdout](../raw/ida/serial/idaHeat2D_klu.stdout) · [meta](../raw/ida/serial/idaHeat2D_klu.meta) |
| 7 | `idaHeat2D_kry` | _(none)_ | 0 | OK | 0.206 | 2646 | `f7e8058dc419f94e` | [stdout](../raw/ida/serial/idaHeat2D_kry.stdout) · [meta](../raw/ida/serial/idaHeat2D_kry.meta) |
| 8 | `idaKrylovDemo_ls` | _(none)_ | 0 | OK | 0.206 | 4763 | `dcae85140f5954b1` | [stdout](../raw/ida/serial/idaKrylovDemo_ls.stdout) · [meta](../raw/ida/serial/idaKrylovDemo_ls.meta) |
| 9 | `idaKrylovDemo_ls` | `1` | 0 | OK | 0.206 | 4763 | `dcae85140f5954b1` | [stdout](../raw/ida/serial/idaKrylovDemo_ls__1.stdout) · [meta](../raw/ida/serial/idaKrylovDemo_ls__1.meta) |
| 10 | `idaKrylovDemo_ls` | `2` | 0 | OK | 0.206 | 4763 | `dcae85140f5954b1` | [stdout](../raw/ida/serial/idaKrylovDemo_ls__2.stdout) · [meta](../raw/ida/serial/idaKrylovDemo_ls__2.meta) |
| 11 | `idaRoberts_dns` | _(none)_ | 0 | OK | 0.206 | 2684 | `f6ce9a7b6597543c` | [stdout](../raw/ida/serial/idaRoberts_dns.stdout) · [meta](../raw/ida/serial/idaRoberts_dns.meta) |
| 12 | `idaRoberts_klu` | _(none)_ | 0 | OK | 0.206 | 1925 | `0e0dd050cf3b590c` | [stdout](../raw/ida/serial/idaRoberts_klu.stdout) · [meta](../raw/ida/serial/idaRoberts_klu.meta) |
| 13 | `idaSlCrank_dns` | _(none)_ | 0 | OK | 0.206 | 3551 | `a05c948cf9456ba7` | [stdout](../raw/ida/serial/idaSlCrank_dns.stdout) · [meta](../raw/ida/serial/idaSlCrank_dns.meta) |
