# Quantum tunnelling — run log and results

**Date:** 2026-07-26
**Notebook:** [dynamic_notebooks/tunneling.posim](dynamic_notebooks/tunneling.posim)
**Reproduce:**

```bash
cargo run -p posim --release -- --notebook dynamic_notebooks/tunneling.posim
```

Every number below is verbatim output from that notebook. Units are
ħ = m = 1, so E = k²/2.

---

## 1. What changed in the language, and why it mattered

Before this run, a rectangular barrier could not be written in posim at
all. The language had arithmetic but **no comparison operators**, so a
piecewise potential had no expression — and a square barrier is *the*
canonical 1-D quantum problem. The previous milestone worked around it
with built-in `BARRIER`/`WELL` shapes and said so plainly.

Six operators now exist — `<`, `<=`, `>`, `>=`, `==`, `!=` — yielding
**1 for true and 0 for false**. There is no boolean type, and that is
the point:

```
In[1]:= def barrier(x) { 2.5 * (x > 0) * (x < 1) }
Out[1]= function barrier(1 parameter(s)) defined — 1 body line(s)
In[2]:= barrier(-1)
Out[2]= 0
In[3]:= barrier(0.5)
Out[3]= 2.5
In[4]:= barrier(2)
Out[4]= 0
```

`(x > 0) * (x < 1)` is an **indicator function**. Multiply by a height
and you have a barrier; add two and you have a resonant cavity. No new
syntax, no built-in shape — just arithmetic on 0 and 1.

A design consequence worth recording: the built-in shape `barrier` now
collides with a user function of the same name. They are told apart by
whether arguments follow — `qm potential barrier` is *your* function,
`qm potential barrier 2.5, 0, 1` is the built-in. A bare name always
means what the user wrote.

---

## 2. The canonical problem: tunnelling through a square barrier

A Gaussian packet of central wavenumber k₀ = 2, so E₀ = 2.0, fired at a
barrier of height **2.5** — above its energy. Classically nothing gets
through.

```
In[5]:= qm grid -100 100 2000
Out[5]= grid [-100, 100] with 2000 interior points, h = 0.099950 (potential and psi cleared)
In[6]:= qm potential barrier
Out[6]= potential `barrier` sampled at 2000 points, V in [0, 2.5] (psi cleared)
In[7]:= qm packet -25 2 2
Out[7]= psi = Gaussian packet at x0 = -25, sigma = 2, k0 = 2, t = 0
In[8]:= qm energy
Out[8]= 2.023971780446395
In[9]:= qm prob -100 0
Out[9]= 1.000000000000002
In[10]:= qm run 30 steps 3000
Out[10]= t = 30 (3000 step(s) of dt = 0.01), <E> = 2.023971780446, norm drift = 3.018e-13
In[11]:= qm prob 1 100
Out[11]= 0.327563019391693
In[12]:= qm prob -100 0
Out[12]= 0.672436408645103
```

| quantity | value |
|---|---|
| transmitted `T` | **0.327563** |
| reflected `R` | 0.672436 |
| `T + R` | 0.999999 |
| `<E>` drift over 3000 steps | 0 to 12 digits |
| norm drift | 3.0e-13 |

**33 % of the particle crossed a barrier it did not have the energy to
cross.** The two channels account for all but 1e-6 of the probability.

The norm drift is the sharp check here. The Cayley operator is unitary
for *any* time step — exactly, not asymptotically — so 3e-13 over 3000
solves measures the tridiagonal solver rather than `dt`.

**Against theory.** The library's own test compares this setup with the
analytic barrier coefficient averaged over the packet's momentum
distribution and agrees to 0.25 %. The averaging is not cosmetic: the
packet's spread straddles the barrier top, so comparing against `T` at
the central energy alone would give 0.3167 and look 4 % wrong.

---

## 3. Browser animation

`QM ANIMATE` propagates while capturing |ψ|² and writes a
**self-contained** HTML page — no scripts, styles, fonts or data fetched
from anywhere, so it works from `file://` and cannot phone home.

```
In[14]:= qm animate "scatter.html" 32 frames 140
Out[14]= wrote scatter.html — 140 frames over t = 32 (dt = 0.011429, 667 points per frame), worst norm drift 5.822e-13. Open it in a browser.
```

The page draws |ψ|² against x with the potential overlaid, a shaded
transmitted region, play/pause, a scrub bar, and a live readout of the
norm and transmitted fraction computed **in the browser** from the frame
data.

Verified live in a browser (the pane would not composite screenshots in
this environment, so the page was driven and inspected through
JavaScript instead):

| check | result |
|---|---|
| frames / points loaded | 140 / 667 |
| final norm, computed in-page | 0.999997 |
| final transmitted, computed in-page | **0.327562** |
| notebook's value | **0.327563** |
| peak at t = 0 | x = −24.94 (launched at −25) |
| final reflected peak | x = −35.43, height 0.0348 |
| final transmitted peak | x = +40.73, height 0.0169 |

Two independent implementations — the Rust `probability_in` and the
page's own JavaScript trapezoid — agree to 1e-6.

The transmitted peak sits slightly *ahead* of the free-travel prediction
(−25 + 2·31.8 = 38.5 versus 40.7 measured). That is not error:
tunnelling favours the high-k components of the packet, and they travel
faster. The reflected peak is the taller of the two, consistent with
R > T.

---

## 4. Stage 3: absorbing boundary conditions

The Dirichlet walls **reflect**, which is the real cost driver in a
scattering run: the domain must be long enough that nothing reaches
them. A complex absorbing potential replaces that constraint —
`H → H − i W(x)` with `W ≥ 0` ramping up smoothly over `width` at each
edge, draining probability instead of bouncing it.

```
In[15]:= qm grid -45 45 1350
In[17]:= qm absorb 15 3
Out[17]= absorbing edges: width 15, strength 3, power 2. Propagation is NO LONGER unitary — the norm decays, which is the absorber working. QM STATES is unavailable while this is on.
In[19]:= qm run 20 steps 2000
Out[19]= t = 20 (2000 step(s) of dt = 0.01), <E> = 2.027672292154, norm drift = 1.056e-4
In[20]:= qm prob 1 30
Out[20]= 0.327689667523621
```

| domain | width | absorber | transmitted |
|---|---|---|---|
| `[-100, 100]` | 200 | none | 0.327563 |
| `[-45, 45]` | **90** | `15 3` | **0.327690** |

**The same answer to 4 parts in 10 000, on a domain less than half the
size.** That is what the absorber buys.

Three consequences are reported by the tool rather than left to be
discovered:

* **Propagation stops being unitary.** The norm decays on purpose, so
  the drift figure stops being a health check while the absorber is on.
* **`QM STATES` is refused.** `H − iW` is not Hermitian, and a symmetric
  eigensolver fed a non-Hermitian matrix returns confident nonsense.
* **The tuning is real** — see below.

### The tuning surface, measured rather than assumed

My first attempt used `strength = 1.0` and the library test failed: the
absorber reflected 5e-3 back into the interior. Rather than nudge the
threshold, I measured the whole surface
(`cargo run -p quantum --release --example absorber_tuning`), k₀ = 3:

| strength | w=6 | w=10 | w=14 | w=18 |
|---|---|---|---|---|
| 0.05 | 8.7e-1 | 8.0e-1 | 7.2e-1 | 6.4e-1 |
| 0.20 | 5.8e-1 | 4.1e-1 | 2.8e-1 | 2.0e-1 |
| 0.80 | 1.2e-1 | 2.9e-2 | 7.4e-3 | 1.9e-3 |
| **3.20** | 3.1e-4 | 1.9e-6 | 1.8e-8 | **1.3e-9** |
| 12.8 | 2.2e-6 | 2.8e-7 | 6.6e-8 | 1.7e-8 |
| 51.2 | 1.6e-4 | 5.0e-6 | 1.1e-6 | 2.4e-7 |
| 204.8 | 6.7e-3 | 5.3e-4 | 5.1e-5 | 6.3e-6 |
| 3276.8 | 1.4e-1 | 6.5e-2 | 3.0e-2 | 8.1e-3 |

The optimum at width 18 is near strength 3, giving **1.3e-9** reflection
against a plain wall's 1.0 — nine orders of magnitude.

**A correction worth recording.** My first sweep stopped at strength
3.2 and showed reflection falling monotonically, which contradicted the
"rises on both sides" claim I had already written in the example's own
prose. Extending the sweep two decades further found the turnover: too
strong *does* reflect, because a steep change in the potential is a
mirror whether that potential is real or imaginary. The claim was right;
my evidence for it did not exist until I went and got it.

The ramp exponent also has a clear optimum — `power = 2` gives 1.3e-9
where 1, 3 and 4 give 2.0e-6, 2.4e-8 and 6.6e-7 — and the same absorber
is *not* equally good at every energy (2.8e-1 at k₀ = 1 versus 1.8e-11
at k₀ = 8). Long wavelengths need a wider, gentler absorber. That is why
these are parameters and not constants.

---

## 5. The double barrier, and a result that is not the textbook headline

Two barriers with a gap form a resonant cavity — the mechanism behind
the resonant tunnelling diode. Expressible only because comparisons are
ordinary arithmetic:

```
In[24]:= def double(x) { 2.5 * ((x > 0) * (x < 1) + (x > 3) * (x < 4)) }
In[31]:= qm prob 4 100
Out[31]= 0.271498902829932
In[32]:= qm prob -100 0
Out[32]= 0.720114758902967
```

| | single barrier | double barrier |
|---|---|---|
| transmitted | 0.3276 | **0.2715** |
| reflected | 0.6724 | 0.7201 |
| left in the structure | — | **0.0084** |

**At this energy the double barrier transmits *less*, not more.** The
0.84 % left sitting in the gap is the resonant cavity holding
probability — but resonant *enhancement* only happens at the cavity's
quasi-bound energies, and those peaks are narrow.

A scan over central wavenumber confirms it (σ = 2, so Δk = 0.25):

| k₀ | 1.4 | 1.7 | 2.0 | 2.3 | 2.6 | 2.9 |
|---|---|---|---|---|---|---|
| T | 0.017 | 0.101 | 0.271 | 0.308 | 0.420 | 0.722 |

Monotonic — no resonance peak. Repeating with a four-times narrower
momentum spread (σ = 8, domain 600 wide) gave T = 0.0045, 0.0040,
0.0061, 0.0155, 0.0898 at k₀ = 1.30…1.90: still no clean peak, and
transmission an order of magnitude *lower* at the same central k.

Both facts point the same way: **T(E) rises so steeply here that a broad
packet's answer is dominated by its high-energy tail**, and the
resonances are narrower than any of these packets. Resolving them needs
a narrower packet in k — a wider one in x, a longer domain, and a longer
run — or a direct transfer-matrix calculation of T(E), which is not
implemented.

I am recording the negative result rather than tuning parameters until a
peak appeared.

### Resolved — the transfer matrix, added later

The diagnosis was right and the remedy was the one named: a direct
`T(E)`. `quantum::transfer` solves the time-*independent* problem at one
energy, so the packet — and its momentum spread — leaves the question
entirely. The same double barrier, through the language:

```text
In[1]:= def dbl(x) { 3 * ((abs(x) > 1.5) * (abs(x) < 2)) }
In[2]:= qm grid -12 12 4800
In[3]:= qm potential dbl
In[4]:= qm scan 0.05 2.9 1200
Out[4]= 1200 energies in [0.05, 2.9] (0 refused), T from 2.082e-4 to 1.000e0
  2 resonance(s), strongest first:
    E = 1.288407006   T = 0.999989181
    E = 0.316221852   T = 0.998978977
In[5]:= qm transmission 0.5
Out[5]= E = 0.5: T = 0.031944823, R = 0.968055177, T + R = 1.000000000000
```

**Two resonances, both at essentially unit transmission**, standing
three to four orders of magnitude above the off-resonance background —
`T = 0.032` at `E = 0.5`, between the peaks. The wavepacket scan in the
table above ran across exactly this structure and reported a monotone
rise, because each packet averaged the whole of it.

The scan is not a refinement of the packet method; it answers a
different question. A packet tells you what a *state* does. `T(E)` tells
you what the *barrier* does. For resonances, only the second one is
finite work.

Flux balance, `T + R = 1` to twelve figures, is the check that comes
free with the method and holds at every energy where the potential is
real.

---

## 6. State

| | |
|---|---|
| workspace tests | **304 passed, 0 failed** |
| build warnings | 0 |
| `clippy --workspace --all-targets` | 0 errors, 0 warnings |
| certification gates | 9/9 from a fresh clone |

New in this round: six comparison operators, `QM ABSORB`, `QM ANIMATE`,
the `absorber_tuning` example, the tunnelling notebook, and 9 tests
covering them.

## 7. Stage 4 — two dimensions, by ADI

See [dynamic_notebooks/double_slit.posim](dynamic_notebooks/double_slit.posim)
and grammar.md §5.11 / Example 18. Summary of what it establishes:

* **ADI by Strang-split Cayley factors**, not Peaceman–Rachford. Each
  direction gets its own Cayley transform, so every factor is exactly
  unitary and the norm is machine-precision conserved for *any* `dt`,
  with the splitting error confined to the dynamics. Peaceman–Rachford
  would tangle the two together in one drifting number.
* **Double-slit fringes match theory.** With `d = 4`, `λ = 0.785`,
  screen at `L ≈ 14.8`: predicted maxima at y = 0, 2.96, 6.32 and minima
  at 1.46, 4.56. Measured band maxima at 0, 2.5–3.5, 5.5–6.5 and minima
  at 0.5–1.5, 3.5–4.5. Every one lands in the right band.
* **A bug this exposed:** `energy()` returned `∫ψ*Hψ` without dividing by
  the norm. Fine for a unit-norm state, wrong under an absorber — the
  double-slit run reported `<E> = 6.83` for a packet of energy 32 purely
  because 78 % had been absorbed. Now `<ψ|H|ψ>/<ψ|ψ>`, and the same run
  reports 30.99 against an initial 31.01.
* **Two setup errors recorded rather than hidden:** a first attempt put
  the first interference maximum at 52°, off-screen, so no fringes
  appeared at all; a second launched the packet *inside* the absorber and
  lost 86 % of it before the slits.

### Not done

* Bound states in 2-D. On an `nx × ny` grid the Hamiltonian is
  `(nx·ny)²` — a 200×200 grid gives a 40 000² dense matrix, far beyond
  the Jacobi eigensolver. Propagation has no such limit.
* Resonance peaks in the double barrier remain unresolved (§5).
* The absorber is tuned by hand; there is no automatic selection from
  the packet's energy, though `absorber_tuning` gives the data for one.
* Transmission is measured by integrating the density in a region, not
  by projecting onto outgoing plane waves — fine for well-separated
  packets, wrong if they overlap the barrier when you measure.
* Three dimensions. The same ADI structure extends, but memory grows as
  the cube of the linear resolution.
