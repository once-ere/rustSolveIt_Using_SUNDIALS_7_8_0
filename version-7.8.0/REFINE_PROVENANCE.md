# Refine & Refactor Provenance

*What the 2026-08-06 refinement pass changed, exactly why, what it
deliberately declined to change, and the evidence that the result still
works. Written so a reviewer can re-check every claim rather than trust
it — the same standard as `EXPORT_PROVENANCE.md`.*

**Date:** 2026-08-06
**Base:** `once-ere/rustSimulate@e21ac82` (a fresh clone)
**Toolchain:** the workspace's pinned build (`cargo build --workspace`,
zero warnings, `#![forbid(unsafe_code)]`, zero crates.io dependencies)

---

## 1. Method

1. **Verify first.** The clone was verified exactly as shipped before
   anything was touched: workspace build (debug + release, 8 crates,
   zero warnings), the full test suite (563 green at base), all 6
   physical_object examples, both quantum examples, all 8
   special_functions examples, all 12 collision scripts, all 59
   dynamic notebooks (including the 34 Routh solutions), all 11
   StageNbks, the wire-protocol and Jupyter-kernel tests, the Index of
   Functions verifiers (1177/1177 posim fragments executed, 258/258
   Rust snippets compiled), and the index page live in a browser
   (buckets, Tier-C lazy load, search, entry pages). **Zero failures.**
2. **Review by independent agents.** Six reviewers each read one area
   (physical_object, posim core, posim front ends, quantum,
   special_functions, tooling + top-level docs) under the repository's
   hard rules. Every finding was then **adversarially verified** by an
   independent agent instructed to refute it — checking the claim
   against the actual file, the test suite, the captured index
   examples, and the pinned contracts in `ARCHITECTURE.md`/`CLAUDE.md`.
   33 findings were raised; 29 survived verification; 2 of the 4
   refuted-as-scoped findings were re-adjudicated and applied as
   deliberate contract-alignment fixes (§3), 2 were declined (§4).
3. **Apply in batches, test at every commit.** Changes landed in four
   commits, one per area, each leaving `cargo build --workspace`
   warning-free and the full suite green. Files were backed up into
   `.backups/2026-08-06/` before modification, per `CLAUDE.md`.

## 2. What changed

### Bug fixes (behavior change on defective paths only, each pinned by a new regression test)

| where | defect | fix | test |
|---|---|---|---|
| `posim/src/vm.rs` `execute()` | The transactional-NEW rollback assumed the failing object was appended last; a nested `DEF` call inside the initializer appends objects *after* it, so the rollback renumbered them while the name registry and wall indices kept stale indices — `get p.mass` could silently read a *different, later object*. | The rollback now does exactly the bookkeeping `Instr::Delete` does (`unregister_index`, wall-index retain/shift), and clears `pending_dumbbell` beside the other three pending slots. `ARCHITECTURE.md` §3.8 updated in lockstep. | `failing_new_with_nested_creation_renumbers_names` |
| `posim/src/machine.rs` JSON parser | `\uXXXX` escapes decoded each half of a UTF-16 surrogate pair separately; both halves fail `char::from_u32`, so every non-BMP character arrived as two U+FFFD. Python's `json.dumps` (default `ensure_ascii=True`) writes **all** non-BMP characters as surrogate pairs, so the shipped Jupyter kernel corrupted any emoji or astral math symbol. | RFC 8259 §7 pair recombination; unpaired halves stay U+FFFD exactly as before; `bad \u escape` error string unchanged. | `surrogate_pairs_decode_to_the_real_character` |
| `posim/src/parser.rs` | `DEL 1.9` silently truncated to `DEL 1` — deleting the wrong object *and* renumbering everything above it. Same silent `as usize` at `RUN … STEPS`, `LAPLACE`, `SCENE HIDE/SHOW`. The language's own documented policy (grammar.md, and `SCENE CREATE`'s port check) is refusal, not truncation. | New `expect_index` helper used at all four sites: fractional values are refused with an actionable message. Grammar productions unchanged (`NUMBER`), so no EBNF/HELP/grammar-doc lockstep was triggered. | `fractional_indices_are_refused_not_truncated` |
| `special_functions/src/bessel_complex.rs` `bessel_j_c` | The documented "Negative order" refusal sat *below* the four asymptotic routes, which carry no order guard — `bessel_j_c(-1, 30i)` returned a value while `bessel_j_c(-1, 1)` errored. (The leaked values were numerically correct via `J₋ₙ = (−1)ⁿJₙ`; returning them contradicted the crate-wide refusal convention and its own doc.) | Guard hoisted above the routes so the error contract holds on the whole domain, matching `bessel_i_c`/`bessel_y_c`/`bessel_k_c`. Applied only after the adversarial verifier ran the full crate suite on a patched scratch copy: 233 + 61 green, valid orders bit-identical. | two new assertions in `negative_whole_order_uses_the_gamma_poles` pinning the previously-leaking rotation and asymptotic regions |
| `tools/verify_tierb_examples.py` | A cargo build failure whose stderr contained no snippet-mappable error line left `bad` empty — and the script then stamped **every** snippet `verified` and printed `COMPILED 258/258`. A check that cannot fail, the exact failure mode `certify_clean.sh`'s header documents. | The run now exits nonzero when `returncode != 0` and no error mapped to a snippet; the declared 1800 s timeout is caught; the never-called `failing()` stub (always returned `{}`) is deleted. Re-run after the change: COMPILED 258/258. | the guard itself (fails loudly instead of vacuously passing) |

### Refinements (verified behavior-preserving)

- **physical_object** — `collide.rs` module doc matched to the pinned
  §3.8 contract (Dumbbell in the shape set, the canonical three tiers,
  the full five-family candidate-axis list); `support_axis_separation`
  doc now states the directed gap `g(±l)` the body evaluates;
  `is_collidable` doc describes the predicate it is (the old text
  described a radius accessor); `RunReport::nst/nfe` docs state what
  the SPRK path populates; `newtons_cradle.rs` and
  `bouncing_ball_restitution.rs` gain the two `allow` attributes hard
  rule 2 mandates (matching the other four examples).
- **posim** — one `component_suffix()` home for the `.x/.y/.z/.w`
  parse (was byte-identical in `path()` and `atom()`); one `qm_args()`
  method for the QM/QM2/QM3 argument-list closure (was triplicated,
  with the comma-rationale comment on only one copy); `sin("x")` now
  reports `sin() expects a number…` instead of the placeholder word
  `builtin` (rule 7); one `script_cells()` home for the brace-depth
  cell joiner (`%load` and `replay_into` had silently diverged copies);
  `loaded_hint` now recognises a 3-D quantum problem (`QM3 ANIMATE`
  hint) instead of signing off "nothing to show"; `scene.html`'s
  dumbbell comment says *four* silhouette lines (matching the code and
  the pinned UI facts) and the `state.entities` inventory lists the
  six fields the renderer actually reads.
- **quantum** — `NashPropagator` gets its own doc (its paragraphs were
  fused onto `Splitting`, making that enum's rustdoc summary wrong;
  same defusion for `step_with`/`kinetic`); the crate front page no
  longer claims "One-dimensional" — the 2-D/3-D ADI solvers, transfer
  matrix, absorber design and isosurface modules are on the module
  map; two dead self-assignments in `Propagator3::apply_dir` (and the
  comment asserting a false invariant) removed; `Grid2/Grid3::is_empty`
  compute `len() == 0` instead of hardcoding `false`; the
  `max_iters = 0` auto-budget sentinel of `bound_states` (called with
  `0` by posim in four places) is documented in 2-D and 3-D; one
  stranded mid-file import moved to the top block.
- **special_functions** — `bessel_y_nu`'s public doc was attached to
  the private `y_nu_loss` (split back; the public fn was undocumented);
  the stale NOTE claiming the loss guard "is not yet implemented" ten
  lines below the implemented Stage-2J guard now records history
  accurately; `no_method`'s refusal no longer claims DLMF 10.20 "is
  not implemented" (it is, in `airy_uniform`, and is offered as a
  candidate — the point is beyond it too); `bessel_cnu_large.rs`'s
  module doc records that complex Airy exists now; `airy_candidates`'
  doc defused from `debye_candidates`.
- **docs** — README/CLAUDE.md test counts refreshed to the measured
  566 (was a 5×-stale 104) in the `N passed workspace-wide` phrasing
  `certify_clean.sh` check 6 polices; the Routh catalogue counts 34
  notebooks, 17 per Part, matching the 34 `routh_*.posim` files and
  the README tables' own 17 rows (prose said 32/16);
  `PROJECT_STATUS.md` §1 counts refreshed and §3–§4 no longer claim
  Legendre/Laguerre/Hermite/Wigner/eigensolver/quadrature/root-finding
  are missing — each carries the file's own italic retirement note
  naming the module that built it; `jupyter/README.md`'s op list gains
  `{"op":"events"}` (implemented, tested, and relied on by the kernel).
  Running `scripts/certify_clean.sh` on the refined fresh clone then
  caught three more **live** count claims its gate polices —
  `CLEANROOM_PROVENANCE.md`, `EXPORT_PROVENANCE.md` and
  `SPECIAL_FUNCTIONS_PROVENANCE.md` all asserted "561 passed", stale
  even against the 563 of the base tree (pre-existing drift) — updated
  to 566, after which the certifier passes every check.

## 3. Re-adjudicated findings (applied as deliberate behavior changes)

Two findings were factually confirmed but failed the strict
"behavior-preserving" gate *because they are bug fixes*. Both were
applied, with the behavior change scoped to defective paths only and
pinned by tests: the surrogate-pair decoder and the `bessel_j_c` guard
hoist (see §2). In both cases the adversarial verifier had already
established that no test, captured index example, notebook output, or
asserted error string observes the old behavior.

## 4. Considered and declined

- **`scene/mod.rs` "window action" receipts** for unknown/rejected
  window commands: confirmed factually, but a coherent reading exists
  in which the receipt uniformly logs *what the window sent* (with the
  preceding `error:` event carrying the judgment), and the proposed fix
  left that semantics half-changed. Declined as not clearly an
  improvement.
- ~~**`tools/extract_rust_items.py` `#[cfg(test)]` exclusion** is dead
  code — declined as churn, deferred to the next index rebuild~~ —
  **done at that rebuild** (follow-up commit): the exclusion now arms
  on the attribute and fires only when its target is a `mod` (the
  docstring's stated policy; attribute-gated single items such as
  `hankel_ratio` stay indexed). Verified on a synthetic file (module
  items excluded, gated single items kept, scanning resumes after the
  module) and on the real tree: the regenerated `rust_items.jsonl`
  differs from the stale one by exactly the two recorded omissions
  (`QM2_SUBCOMMANDS`, `QM3_SUBCOMMANDS`), 163 line repositions, and
  the six doc texts this pass edited — nothing removed.
- **Parser ANIMATE/ISO argument parse** is repeated four times with
  per-family error strings and one structurally different site (ISO's
  optional `LEVEL`): extraction is only safe with every message
  preserved verbatim; declined in this pass in favor of the two
  clean dedupes (`component_suffix`, `qm_args`).
- **`sundials_rs/` and `vendor/spec_math`** were not touched, per the
  repository's hard rule: `sundials_rs` is byte-identical to
  `once-ere/SUNDIALS_7_8_Rust_port_for_Linux@780b916` and that identity
  is a verifiable
  property (`EXPORT_PROVENANCE.md` §4). Refinements to vendored code
  belong upstream.
- **Generated artifacts** (`index_of_entities.html`, `catalog-c.js`,
  `index_data/*.json`) were not hand-edited: they are build outputs of
  `tools/`, regenerated only by their own pipeline. Re-running the
  verifiers reproduces them modulo date stamps and one environmental
  message variant (headless vs browser capture), which were reverted.

## 5. Known defects deliberately preserved (pre-existing, documented)

- The parallel disk–disk face-on crossing is invisible to
  downward-crossing rootfinding — pinned as a documented limitation
  (`parallel_disk_disk_separation_is_the_documented_limitation`).
- ~~`DEL`/`BOX` executed on a *lower* index from inside a NEW
  initializer would leave `SimState.last_new` itself stale~~ — the
  adjacent defect the verifier flagged as follow-up was **fixed in a
  follow-up commit**: the Call instruction's NEW-context stash moved
  into `SimState.stashed_last_new` so `shift_new_targets` renumbers
  the active and every stashed rollback target on both deletion paths
  (`DEL`, `BOX` wall removal); deleting the half-built object itself
  disarms its rollback. Reproduced first (both repro tests failed on
  the old code), then pinned:
  `del_inside_a_new_initializer_keeps_the_rollback_on_target`,
  `box_off_inside_a_new_initializer_keeps_the_rollback_on_target`.
  Workspace count: 568.
- ~~The committed `index_data/rust_items.jsonl` carries minor
  pre-existing staleness (line-number drift, two missing constant
  entries)~~ — **reconciled**: the full index pipeline was re-run
  (extract → build_commands/tierb/tierc → build_catalog → build_app →
  both verifiers). Tier B grows 491 → 493 (`QM2_SUBCOMMANDS`,
  `QM3_SUBCOMMANDS`, both `reference` like every posim entry), the
  entity total 4,830 → 4,832, and every runnable example still passes:
  1177/1177 posim fragments executed, 258/258 Rust snippets compiled.
  Captured outputs in this rebuild come from headless runs, so the
  handful of `SCENE CREATE` captures now read "no browser was
  launched — open that address yourself" — the message the same
  command prints in any `POSIM_NO_BROWSER=1` environment.

## 6. Evidence (re-verification after the pass)

Every gate from §1 was re-run on the refined tree:

- `cargo build --workspace` (debug and release): zero warnings.
- `cargo clippy --workspace --all-targets`: zero lints.
- `cargo test --workspace`: **566 passed, 0 failed** (40
  physical_object lib + 19 collision + 9 conservation + 107 posim +
  92 quantum + 233 special_functions + 11 vendored identities + 55
  doctests) — 563 at base plus the three new regression tests.
- All 6 physical_object examples: SUCCESS with their analytic anchors
  (Pluto to 8 decimals, Kepler E/L/Laplace < 1e-6, gyroradius to 1e-4,
  L conserved exactly, cradle propagation, apex = e²h).
- 12/12 collision scripts, 59/59 dynamic notebooks (34 Routh), 11/11
  StageNbks: exit 0, zero failing cells.
- Wire-protocol test and 7-cell Jupyter-kernel test: all checks pass.
- Index verifiers: 1177/1177 posim fragments executed, 258/258 Rust
  snippets compiled — including under the new fail-loudly guard.
