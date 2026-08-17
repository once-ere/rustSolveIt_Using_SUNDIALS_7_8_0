# DIVERGENCES — where the prose and the code disagree

*Phase-2 deliverable of the Index of Entities (see `prompt_01.md` §1 defect 2,
§8 phase 2). Extraction date: **2026-07-30**.*

**Policy.** Code is authoritative; prose is corroborative. Every entity in the
index takes its definition from source and is checked against the
documentation. Where the two disagree, the disagreement is recorded here
rather than silently resolved — the index will carry the code's behaviour and
a note pointing at this file.

Each finding states the claim, the code truth, and — where behaviour is at
stake — the verbatim probe that settled it. **Nothing below is asserted from
reading alone.**

---

## D1 — `grammar.md` §2.2 omitted three keywords that the lexer defines — **FIXED**

**Claim.** `grammar.md:63-80` enumerates the reserved keywords in four
groups (core, scene, collision, names/functions).

**Code.** `posim/src/lexer.rs:36-38` defines `Qm`, `Qm2`, `Qm3` in the
`Keyword` enum, and `lexer.rs:104-106` maps the spellings `"qm"`, `"qm2"`,
`"qm3"` to them. They are reserved exactly as `NEW` is.

**Effect.** 57 keyword spellings exist; §2.2 lists 54. The three quantum
family heads are missing from the list, though §§5.10–5.12 document the
families themselves at length.

**Index treatment.** `QM`, `QM2`, `QM3` get keyword entries, sourced to
`lexer.rs`, cross-referenced to §§5.10–5.12.

### Fixed, 2026-07-30

Both keyword lists now carry a fourth group — `QM QM2 QM3` — in
`grammar.md` §2.2 and in `grammar.tex`, cross-referenced to the three family
sections. §2.2 also now states *why* the list is only three words long: every
sub-command after them (`grid`, `run`, `state`, `energy`, `reset`) is read as
an identifier-or-keyword and matched on its lowercased text, precisely so the
quantum vocabulary stays out of the global keyword namespace. Several of those
words are already keywords in their own right, and reserving the rest would
have cost `system.energy` and `obj0.state` for nothing.

---

## D2 — `parser.rs`'s own EBNF comment advertised a `QM2 ISO` the parser rejects — **FIXED**

**Claim.** `posim/src/parser.rs:136`, inside the module-level EBNF:

```
//! qm2cmd   := …
//!           | "ISO" STRING expr [ "FRAMES" expr ] [ "LEVEL" expr ]
```

**Code.** `Parser::qm2_command()` has no `iso` arm. The sub-commands it
accepts are `status, grid, potential, packet, step, run, norm, energy,
centroid|position, prob|probability, absorb, drive, states, state, reset,
animate`. `qm3_command()` *does* have `"iso" | "isosurface"`.

**Verified 2026-07-30** — `posim --script`:

```
In[3]:= qm2 iso "x.html" 0.1
Err[3]: QM2: unknown subcommand `iso` (grid, potential, packet, states, state, step, run, norm, energy, centroid, prob, absorb, animate, status, reset)
```

(That reply is the **pre-fix** text, captured before this was repaired; the
message is now generated from `QM2_SUBCOMMANDS`, so its wording and order
follow the constant. The transcript is left as it was observed.)

**Who is right.** `HELP_TEXT` (`vm.rs`) and `grammar.md` §5.11 are both
correct — neither documents `QM2 ISO`. The defect is confined to the EBNF
comment in `parser.rs`, which is the one place a reader would trust most.

**Index treatment.** No `QM2 ISO` entry. The `QM3 ISO` entry notes that the
2-D family has no isosurface command, and why (a 2-D density is already
drawable as a heat map).

### Fixed, 2026-07-30

The phantom production is gone from `parser.rs`, and — more to the point —
a gate now makes it uncatchable-by-reading no longer uncatchable at all.
`every_qm2_subcommand_is_documented_in_lockstep` and its `QM3` twin check
their production in **both** directions:

- *forward*, that every declared subcommand appears in `HELP_TEXT`, the
  EBNF, `grammar.md` and `grammar.tex`;
- *converse*, that every word the EBNF **quotes** is either a declared
  subcommand or a declared argument word.

The converse direction is the one that earns its keep, and it is the one
the pre-existing `QM` gate never had. A forward-only check asks whether
the documents mention what the code does; it can never ask whether the
code does what the documents promise. Re-injecting the deleted line makes
the test fail with:

```
QM2 grammar lockstep is broken:
  the qm2cmd EBNF quotes `ISO`, which is neither a declared subcommand nor a
  declared argument word — either the parser is missing an arm or the comment
  is promising a command that does not exist
  the qm2cmd EBNF quotes `LEVEL`, which is neither a declared subcommand nor a
  declared argument word — …
```

`LEVEL` is the tell: it was only ever an argument of the isosurface
command, so it had been orphaned by the same mistake and nothing noticed.

**Extending the gate found four more gaps**, all of the same shape — the
compressed spelling `QM2 NORM | ENERGY | CENTROID` in `HELP_TEXT` and
`grammar.tex`, which names the family once and then lists bare words. A
reader parses that correctly; a string search does not, and neither does
anyone grepping for `QM2 CENTROID`. Now spelled out in full in both files,
matching how `grammar.md` already wrote them.

The subcommand lists are also no longer duplicated: `parser.rs` builds its
`QM2`/`QM3: unknown subcommand` error messages from `QM2_SUBCOMMANDS` and
`QM3_SUBCOMMANDS`, exactly as it already did for `QM_SUBCOMMANDS`. A
duplicated list is a list that drifts.

---

## D3 — six path aliases worked but appeared in no documentation — **FIXED**

`posim/src/vm.rs` accepts a second spelling for several read paths. None of
these appear in `grammar.md` §5.2/§5.7, `physical_object_simulator.md` or
`collision_detection.md` as paths. (`approach` occurs in those files only as
an English word — `grammar.md:1228`, `:1401`, `:1420`; `impulse_n` occurs
once, at `collision_detection.md:343`, as a *Rust struct field*, not a path.)

| alias | canonical | source |
|---|---|---|
| `contactK.time` | `contactK.t` | `vm.rs:2392` |
| `contactK.approach` | `contactK.rel_vel_n` | `vm.rs:2396` |
| `contactK.impulse_n` | `contactK.impulse` | `vm.rs:2397` |
| `QM2 position` / `QM3 position` | `… centroid` | `parser.rs` qm2/qm3 `centroid \| position` |
| `QM2 probability` / `QM3 probability` | `… prob` | `parser.rs` qm2/qm3 `prob \| probability` |
| `QM3 isosurface` | `QM3 iso` | `parser.rs` qm3 `"iso" \| "isosurface"` |

**Verified 2026-07-30** — head-on exchange of two unit spheres, contact at
TOI 1.5:

```
In[5]:= get contact0.t
Out[5]= 1.5
In[6]:= get contact0.time
Out[6]= 1.5
In[7]:= get contact0.rel_vel_n
Out[7]= -2
In[8]:= get contact0.approach
Out[8]= -2
In[9]:= get contact0.impulse
Out[9]= 2
In[10]:= get contact0.impulse_n
Out[10]= 2
```

and, on a 16×16 2-D grid with a Gaussian packet:

```
In[8]:= qm2 position
Out[8]= [0.0000000000000000878205635495564, 0.00000000000000010353961926829087]
In[9]:= qm2 probability -1 1, -1 1
Out[9]= 0.432954441903126
```

**Index treatment.** Each is recorded in its canonical entry's `aliases`
array and resolves through search.

### Fixed, 2026-07-30

All six are now documented where a reader looks for them, in `grammar.md`
(and the two observable rows in `grammar.tex`):

- §5.7's contact-field table: `contactK.t` *(alias `time`)*,
  `contactK.rel_vel_n` *(alias `approach`)*, `contactK.impulse`
  *(alias `impulse_n`)*;
- §5.11 and §5.12: `CENTROID` also spells `POSITION`, `PROB` also spells
  `PROBABILITY`, and `QM3 ISO` also spells `ISOSURFACE` — with the note that
  **`QM2` has no `ISO`**, which is D2's other half stated where it will be
  read.

---

## D4 — `grammar.md` §3's EBNF predated the comparison operators — **FIXED**

**Claim.** `grammar.md:173`:

```ebnf
expr     := term { ("+" | "-") term } ;
```

**Code.** `posim/src/parser.rs:68,79`:

```
expr     := sum { ("<" | "<=" | ">" | ">=" | "==" | "!=") sum } ;
sum      := term { ("+" | "-") term } ;
```

**Effect.** The grammar block in §3 has no comparison level at all, so it
cannot derive `(x > 0) * (x < 1)` — the indicator idiom that §4.2, §5.10 and
Examples 17 and 18 are built on. §4.2 documents comparisons correctly,
including their lowest precedence; only the formal grammar block is stale.

**Index treatment.** The index quotes `parser.rs`'s EBNF as the syntax of
record for every expression-level entry.

### Fixed, 2026-07-30

§3's grammar block now has the comparison level, matching `parser.rs`:

```ebnf
expr     := sum { ("<" | "<=" | ">" | ">=" | "==" | "!=") sum } ;
sum      := term { ("+" | "-") term } ;
```

with an inline note that comparisons yield 1 or 0 at the lowest precedence and
that `(x > a) * (x < b)` is therefore an indicator function — the idiom §4.2,
§5.10 and Examples 17–18 are built on, which the old block could not derive.
The `atom` production gained `IMAGINARY` in the same pass, for the same
reason: it predated complex literals. `grammar.tex` carries both.

---

## D5 — `special_functions.md` called `wigner` "planned"; it had shipped — **FIXED**

**Claim.** `special_functions.md:33`:

| `wigner` — 3j, 6j, Clebsch–Gordan | native | **planned** |

and `special_functions.md:568`: *"A `wigner` module (3j, 6j,
Clebsch–Gordan) is planned and will be added in the same shape."*

**Code.** `special_functions/src/wigner.rs` is 871 lines and defines
`wigner_3j` (:133), `clebsch_gordan` (:239), `wigner_6j` (:275) and
`wigner_9j` (:398) — the last of which the doc's own "planned" list does not
even mention. All four are registered notebook builtins in
`posim/src/special.rs`, and `grammar.md` §4.1 documents them as available.

**Index treatment.** All four get full builtin entries.

### Fixed, 2026-07-30

The status row now reads *implemented, reachable from the notebook* and names
**9j**, which the old "planned" row did not even list. The closing sentence
that promised the module "will be added" is replaced by a real §10 covering
all four functions: the row-by-row argument order of `wigner_9j`, why
half-integer angular momenta are accepted as plain numbers, and why a
selection-rule violation returns **0** (the mathematically correct answer)
while a value that is not an angular momentum at all is an error.

Its two numeric claims were checked by running them, not asserted:

```
In[1]:= wigner_3j(1, 1, 5, 0, 0, 0)
Out[1]= 0
In[2]:= clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1, 0)
Out[2]= 0.7071067811865475
In[3]:= 1/sqrt(2)
Out[3]= 0.7071067811865475
```

---

## D6 — ten `special_functions` modules were absent from `special_functions.md` — **FIXED**

The document's "Status at a glance" table and section list cover
`sph_bessel`, `legendre`, `orthopoly`, `eigen`, `quadrature`, `complex`,
`tridiag`, `bessel`, `wigner`. The crate also contains:

`airy_complex` · `airy_uniform` · `bessel_cnu` · `bessel_cnu_large` ·
`bessel_complex` · `bessel_scaled` · `debye` · `gamma_complex` · `hankel` ·
`lanczos`

That is 10 of 20 modules undocumented in the file nominally dedicated to
them — 8,006 of the crate's lines. Their *notebook-facing* surface (`airy_z`,
`gamma_z`, `hankel_h1_z`, `bessel_j_nu`, `bessel_k_scaled`, …) **is**
thoroughly documented, in `grammar.md` §4.1 instead. The gap is at module
level, not at API level.

**Index treatment.** Module entries are sourced from the crate.

### Fixed, 2026-07-30

All ten now appear — in the status table, and in a new §11 that maps each
module to the entry points it holds: `gamma_complex`, `bessel_complex`,
`bessel_cnu`, `bessel_cnu_large`, `bessel_scaled`, `hankel`, `airy_complex`,
`airy_uniform`, `debye`, `lanczos`.

§11 is a map rather than a duplicate. The accuracy laws, the route selection
and the measured error surfaces stay in `grammar.md` §4.1, because that is the
document a notebook user actually reads; repeating them in two places would
create exactly the drift this file exists to record.

---

## D7 — `special_functions.md`'s own header states the policy D5 breaks

Recorded not as a separate defect but as the standard D5/D6 should be read
against — `special_functions.md:8-10`:

> "This document grows one module at a time, alongside the code. Sections
> marked **planned** are not implemented yet and say so rather than
> pretending otherwise."

The intent is right; D5 is the case where the marker was not updated after
the module landed.

---

## Non-divergences checked and cleared

Recorded so a later pass does not re-investigate them.

- **`system.collide` is read-only.** `grammar.md` §5.2 marks it `R`. A write
  arm exists at `vm.rs:2481` but returns `use COLLIDE ON / COLLIDE OFF to
  switch collision detection`. The doc is correct; the arm is a guiding
  refusal, not a setter.
- **`collisions` is deliberately not a keyword**, so `system.collisions`
  keeps its spelling (`grammar.md:85`). Confirmed absent from
  `lexer.rs::Keyword::from_ident`.
- **`DEF` is not a keyword.** Confirmed: no `"def"` arm in `from_ident`. It is
  a line form recognised before the grammar, exactly as documented.
- **All 61 registered special-function names** in `posim/src/special.rs` are
  present in `grammar.md` §4.1's tables. No omissions either way.
- **All seven shapes** (`POINT SPHERE CUBOID TORUS DISK CYLINDER DUMBBELL`)
  and all seven documented aliases (`DELETE CUBE DISC DESTROY SETTIMESTEP
  FUNCTIONS DUMBELL`) match `lexer.rs` exactly.
- **17 `SCENE` sub-commands** in `parser.rs` match `grammar.md` §5.6's table
  and `HELP_TEXT` one for one, `RESET` included.
- **23 `QM` sub-commands** match: `posim/src/qm.rs:102` exposes
  `QM_SUBCOMMANDS`, and the test `every_qm_subcommand_is_documented_in_lockstep`
  already checks each word against the parser, `HELP_TEXT`, the EBNF comment
  and both grammar documents. That gate is why the `QM` family has no
  D1-class defect — and its absence for `QM2`/`QM3` is why D2 survived.

---

## Status

| # | finding | state |
|---|---|---|
| D1 | `grammar.md` §2.2 omitted `QM`/`QM2`/`QM3` | **fixed 2026-07-30** |
| D2 | phantom `QM2 ISO` in the `parser.rs` EBNF | **fixed 2026-07-30**, with a gate |
| D3 | six undocumented path aliases | **fixed 2026-07-30** |
| D4 | `grammar.md` §3's EBNF had no comparison level | **fixed 2026-07-30** |
| D5 | `special_functions.md` called `wigner` "planned" | **fixed 2026-07-30** |
| D6 | ten `special_functions` modules undocumented | **fixed 2026-07-30** |

**All six are closed.** D2 was the only one that was a defect in
*code-adjacent* material rather than in prose, and the only one a reader could
act on and be wrong; it is the only one that needed a code change, and it got
a test that gates the production in both directions. The other five were
documentation drift, repaired in the documents themselves.

Re-probed after the fixes, by searching the files rather than trusting the
edit: every one of the seven strings D1/D3 needed is present in `grammar.md`,
both keyword lists carry `QM QM2 QM3`, both EBNF blocks carry the comparison
level, `special_functions.md` no longer calls anything shipped "planned", and
none of the ten modules is missing. `grammar.pdf` was recompiled (twice, per
the workflow rule).

The three lockstep gates still pass — they read `grammar.md` and `grammar.tex`
directly, so an edit that broke the agreement would have failed the build:
`cargo test -p posim`, 104 passed, 0 failed.
