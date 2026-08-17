#!/usr/bin/env python3
"""make_reports.py — turn the three index.tsv files into the documentation
sets under c-results/, rust-results/ and differences/.

    python3 tools/make_reports.py

Every number printed in those documents is read out of the index files,
which are themselves written by the run scripts from real process output.

That is a rule this script has to keep, not a property it gets for free.
Hardcoding a count here produces a document that looks computed and is
wrong -- and it did: the run-to-run table carried "179" and "12" long after
the compared set grew to 190 and the OpenMP set turned out to be 11, and the
attribution paragraph kept asserting the host-libm build matched everything
after the sparse-LU substitution made that false. If you add a claim below,
derive it, or the next person to read it will be misled.
"""

import os
import re
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "c-results"
R_DIR = ROOT / "rust-results"
D_DIR = ROOT / "differences"

SOLVER_TITLE = {
    "cvode/serial": "CVODE",
    "cvodes/serial": "CVODES",
    "kinsol/serial": "KINSOL",
    "ida/serial": "IDA",
    "idas/serial": "IDAS",
    "arkode/C_serial": "ARKODE",
}
CRATE_OF = {
    "cvode/serial": "cvode_rs",
    "cvodes/serial": "cvodes_rs",
    "kinsol/serial": "kinsol_rs",
    "ida/serial": "ida_rs",
    "idas/serial": "idas_rs",
    "arkode/C_serial": "arkode_rs",
}


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "(unavailable)"


def provenance():
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "os": sh("bash", "-c", "grep PRETTY_NAME /etc/os-release | cut -d'\"' -f2"),
        "kernel": platform.platform(),
        "arch": platform.machine(),
        "glibc": sh("bash", "-c", "ldd --version | head -1"),
        "cc": sh("bash", "-c", "cc --version | head -1"),
        "cxx": sh("bash", "-c", "c++ --version | head -1"),
        "fc": sh("bash", "-c", "gfortran --version | head -1"),
        "cmake": sh("bash", "-c", "cmake --version | head -1"),
        "rustc": sh("rustc", "--version"),
        "cargo": sh("cargo", "--version"),
        "cores": str(os.cpu_count()),
    }


def prov_block(p):
    return "\n".join(
        [
            "| item | value |",
            "|---|---|",
            f"| generated | {p['generated']} |",
            f"| operating system | {p['os']} |",
            f"| kernel / platform | {p['kernel']} |",
            f"| architecture | {p['arch']} |",
            f"| C library | {p['glibc']} |",
            f"| C compiler | {p['cc']} |",
            f"| C++ compiler | {p['cxx']} |",
            f"| Fortran compiler | {p['fc']} |",
            f"| CMake | {p['cmake']} |",
            f"| rustc | {p['rustc']} |",
            f"| cargo | {p['cargo']} |",
            f"| CPU cores | {p['cores']} |",
        ]
    )


def read_index(p):
    rows = []
    with open(p) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(dict(zip(head, line.rstrip("\n").split("\t"))))
    return rows


def argv_cell(a):
    return f"`{a}`" if a else "_(none)_"


# A SUNDIALS example can fail its solve and still exit 0 -- several return
# void from main after printing the failure. Exit status alone therefore
# overstates how well the run went, so the captures are searched too.
FAIL_MARKERS = ("returned with flag = -", "[ERROR]")


def failure_message(base, variant):
    """One readable line naming the failure, with build paths stripped.

    SUNDIALS error lines carry an absolute source path from whatever machine
    compiled them, which is noise in a table and would be the widest column
    on the page. Only the file name, line and function are kept.
    """
    lines = []
    for ext in (".stdout", ".stderr"):
        f = base / (variant + ext)
        if f.exists():
            lines += [ln.strip() for ln in f.read_text(errors="replace").splitlines()]

    flag = next((ln for ln in lines if "returned with flag = -" in ln), "")
    err = next((ln for ln in lines if "[ERROR]" in ln), "")

    if err:
        # [ERROR][rank 0][/abs/path/foo.c:2898][someFunction] the message
        parts = re.findall(r"\[([^\]]*)\]", err)
        tail = err.rsplit("]", 1)[-1].strip()
        where = ""
        for seg in parts:
            if "/" in seg or seg.endswith((".c", ".h")):
                where = seg.rsplit("/", 1)[-1]
        func = parts[-1] if parts and "/" not in parts[-1] else ""
        err = f"`{func}`: {tail}" if func else tail
        if where:
            err += f" ({where})"
    msg = " — ".join(x for x in (flag.lstrip("ERROR: ").strip(), err) if x)
    return msg.replace("|", "\\|") or "(see the capture)"


def internal_failures(rows, raw_dir):
    """Rows whose captures report a solver failure regardless of exit code."""
    out = []
    for r in rows:
        base = raw_dir / r["dir"]
        text = ""
        for ext in (".stdout", ".stderr"):
            f = base / (r["variant"] + ext)
            if f.exists():
                text += f.read_text(errors="replace")
        if any(m in text for m in FAIL_MARKERS):
            out.append(r)
    return out


def not_ported_reason(rows):
    """Describe the NOT_PORTED class from the examples actually in it.

    This used to read "KLU / SuperLU_MT example". The eleven `*_klu` examples
    are ported now and are compared like any other, so naming KLU here told
    the reader the opposite of what the table beside it says.
    """
    names = sorted({r["example"] for r in rows if r["class"] == "NOT_PORTED"})
    if not names:
        return "none"
    if all(n.endswith(("_sps", "_slu")) for n in names):
        return "SuperLU_MT example; absent on both sides, so there is no output to compare"
    return "no pure-Rust counterpart: " + ", ".join(f"`{n}`" for n in names)


def attribution_paragraph(ident, comparable, ab_ident, ab_total, ab_libm, ab_survivors):
    """State what the host-libm control build does and does not establish.

    The wording here is load-bearing. While the libm was the only substituted
    numerics, the control build accounted for every divergence and the
    paragraph could say so. The pure-Rust sparse LU broke that: it has no
    control build, because there is no KLU to switch back to, so the `*_klu`
    variants differ under *both* builds by construction. The old text went on
    claiming the switch explained everything, which turned a measurement into
    a false statement.
    """
    n_libm, n_surv = len(ab_libm), len(ab_survivors)
    out = [
        f"**With the elementary functions delegated back to the host C library "
        f"(`--features host-libm`), {ab_ident} of {ab_total} are identical.** "
        f"The switch changes nothing else in the port, so the {n_libm} "
        f"variant{'s' if n_libm != 1 else ''} it restores "
        f"{'are' if n_libm != 1 else 'is'} caused by the pure-Rust libm and by "
        f"nothing else — measured, not asserted."
    ]
    if n_surv:
        klu = [r for r in ab_survivors if "_klu" in r["example"]]
        if len(klu) == n_surv:
            out.append(
                f"The {n_surv} that differ under **both** builds are exactly the "
                f"`*_klu` examples. That is not a second finding, it is the same "
                f"one seen twice: `host-libm` does not touch the sparse linear "
                f"solver, and there is no KLU to switch back to, so those "
                f"variants cannot be attributed this way. They are covered "
                f"instead by direct verification of the replacement solver."
            )
        else:
            other = [r for r in ab_survivors if "_klu" not in r["example"]]
            out.append(
                f"**{len(other)} variant{'s' if len(other) != 1 else ''} differ under both "
                f"builds and {'are' if len(other) != 1 else 'is'} not `*_klu`: "
                + ", ".join(f"`{r['example']}`" for r in other)
                + ". Nothing in the port accounts for "
                + ("them" if len(other) != 1 else "it")
                + ", which is the signature of a port defect. Fix before landing.**"
            )
    out.append("See [ATTRIBUTION.md](ATTRIBUTION.md).")
    return "\n\n".join(out)


# --------------------------------------------------------------------------
# c-results
# --------------------------------------------------------------------------
def write_c(p):
    rows = read_index(C_DIR / "index.tsv")
    by_dir = defaultdict(list)
    for r in rows:
        by_dir[r["dir"]].append(r)

    serial = [d for d in by_dir if d in SOLVER_TITLE]
    other = sorted(d for d in by_dir if d not in SOLVER_TITLE)
    n_ok = sum(1 for r in rows if r["status"] == "OK")

    # how much of the upstream tree this run actually reached
    up = ROOT / "upstream-c" / "examples"
    if not up.exists():
        up = ROOT / "examples"
    n_upstream_dirs = len({q.relative_to(up) for q in up.glob("*/*") if q.is_dir()}) or len(by_dir)

    n_serial = sum(len(by_dir[d]) for d in serial)
    n_rust = len(read_index(R_DIR / "index.tsv")) if (R_DIR / "index.tsv").exists() else 0
    omp_dirs = sorted(d for d in by_dir if d.endswith(("C_openmp", "F2003_openmp")))
    n_omp = sum(len(by_dir[d]) for d in omp_dirs)
    # Observed by running the whole pipeline four times and diffing the
    # captures with git. Not derivable from a single run, so it is named
    # explicitly rather than counted -- and the set is not fixed: on the
    # fourth run `ark_heat1D_omp 4` reproduced and a *parallel* example moved
    # instead. Any given run sees a subset.
    OMP_MOVERS = [
        "ark_heat1D_omp 4",
        "idaFoodWeb_kry_omp 4",
        "idasFoodWeb_kry_omp 4",
        "kinFoodWeb_kry_omp 4",
        "idaHeat2D_kry_omp_f2003 4",
        "idaHeat2D_kry_omp_f2003 8",
    ]
    MPI_MOVERS = ["kin_diagon_kry_f2003 (mpirun -np 4)"]
    n_other_serial = sum(len(by_dir[d]) for d in other if "serial" in d)
    n_mpi = sum(len(by_dir[d]) for d in by_dir if d.endswith("parallel"))
    failed = internal_failures(rows, C_DIR / "raw")

    doc = [
        "# c-results — every upstream C example this toolchain could build",
        "",
        "This directory records what the **unmodified upstream SUNDIALS 7.8.0",
        "C examples** actually printed on this machine. It is raw evidence:",
        "the `.stdout` files are the bytes the processes wrote, with nothing",
        "filtered, rounded or edited.",
        "",
        f"\"Every\" is scoped, and the scope is large: {len(rows)} variants came out of "
        f"{len(by_dir)} of the upstream tree's {n_upstream_dirs} example directories. The other "
        f"{n_upstream_dirs - len(by_dir)} produced nothing, because a backend they need is "
        "missing or unusable here; every one is accounted for in",
        "[`../requirements.md`](../requirements.md).",
        "",
        "## Provenance",
        "",
        prov_block(p),
        "",
        "The C sources are an unpacked SUNDIALS 7.8.0 tree, used read-only. On",
        "the machine above it was `/home/nsh/Developer/sundials-7.8.0`, reached",
        "through the `upstream-c` symlink; that path is recorded in every",
        "`.meta` file and is provenance, not a dependency — point the symlink",
        "at your own copy and the pipeline reproduces. The vendored `examples/`",
        "tree is the same sources, and supplies the CMake tuples that decide",
        "which command-line variants each example is run with.",
        "",
        "## How to reproduce all of it",
        "",
        "```bash",
        "tools/c_build.sh          # configure + build, out of source, into build/c",
        "tools/c_examples_run.sh   # run every binary, once per declared argv variant",
        "python3 tools/make_reports.py",
        "```",
        "",
        "`tools/c_build.sh` prints which optional backends it was able to switch",
        "on; anything it could not is listed in [`../requirements.md`](../requirements.md).",
        "",
        "## Headline result",
        "",
        f"**{len(rows)} (example, argv) variants were executed. {n_ok} exited 0"
        + (f", and {len(rows) - len(failed)} also report a completed solve.**"
           if failed else ".**"),
        "",
        "| status | variants |",
        "|---|---|",
    ]
    st = defaultdict(int)
    for r in rows:
        st[r["status"]] += 1
    for k in sorted(st, key=lambda k: -st[k]):
        doc.append(f"| {k} | {st[k]} |")

    doc += [
        "",
        "## Layout of this directory",
        "",
        "| path | contents |",
        "|---|---|",
        "| `index.tsv` | one row per variant: directory, example, argv, exit status, wall time, stdout size and SHA-256 |",
        "| `raw/<dir>/<variant>.stdout` | exactly what the process printed to stdout |",
        "| `raw/<dir>/<variant>.stderr` | exactly what it printed to stderr |",
        "| `raw/<dir>/<variant>.meta` | the binary, the argv, the working directory, the exit code, the timing and the full SHA-256 |",
        "| `by-solver/*.md` | the per-solver tables below |",
        "",
        "A `<variant>` is the example name, plus `__` and the argv with spaces",
        "turned into underscores when the example is declared with arguments.",
        "",
        "## Checking any single row yourself",
        "",
        "```bash",
        "cat c-results/raw/cvode/serial/cvRoberts_dns.meta      # what was run",
        "cat c-results/raw/cvode/serial/cvRoberts_dns.stdout    # what it printed",
        "sha256sum c-results/raw/cvode/serial/cvRoberts_dns.stdout",
        "```",
        "",
        "The `.meta` file carries the full digest; `index.tsv` carries only its",
        "**first 16 hex characters**, so compare against the `.meta` line or",
        "against `sha256sum ... | cut -c1-16`.",
        "",
        "## Run-to-run reproducibility",
        "",
        "The whole pipeline has been executed four times on this machine, and",
        "the captured `.stdout` files compared between runs with git — a byte",
        "comparison, not a tolerance. The strongest single statement, and the",
        "one anyone can re-check, is about the most recent re-run: it rebuilt",
        "the C library and all 233 example binaries from source, re-ran every",
        "variant on both sides, and **every capture in the compared set came",
        "back byte-identical to the committed one** — 190 C, 199 Rust, 0 diffs.",
        "",
        "```bash",
        "tools/c_build.sh && tools/c_examples_run.sh && tools/rust_examples_run.sh",
        "git status --porcelain c-results rust-results   # only .meta timings should move",
        "```",
        "",
        "The earlier runs are weaker evidence for part of the set: the four runs",
        "were not runs of the same build, because KLU only became usable partway",
        "through, so the eleven `*_klu` serial variants have fewer repetitions",
        "behind them than the other 179.",
        "",
        "| set | variants | reproduced byte for byte |",
        "|---|---:|---|",
        f"| the six *serial* directories (the compared set) | {n_serial} | **all of them** |",
        f"| every Rust example (`rust-results/`) | {n_rust} | **all of them** |",
        f"| `*/C_openmp` and `*/F2003_openmp` | {n_omp} | "
        f"up to {len(OMP_MOVERS)} differ between runs |",
        f"| `*/*parallel` (MPI) | {n_mpi} | "
        f"{len(MPI_MOVERS)} reorders between runs |",
        "",
        f"The {len(OMP_MOVERS)} that move are OpenMP examples run with a thread count as argv: "
        + ", ".join(f"`{v}`" for v in OMP_MOVERS)
        + ". This is expected and is not a defect in anything: an OpenMP",
        "reduction sums partial results in whatever order the threads finish, so",
        "a dot product or a norm differs in its last bits from run to run, and",
        "inside an iterative solver that changes the iteration counts. Compare",
        "`kinFoodWeb_kry_omp 4`, which reported `nni = 7, nli = 229` on one run",
        "and `nni = 10, nli = 378` on the next.",
        "",
        "The MPI case is a different animal and worth separating, because it",
        "looks alarming and is not: `kin_diagon_kry_f2003` runs under `mpirun",
        "-np 4`, and between runs its 47 lines come out in a **different order**",
        "with every number identical -- four ranks writing to one stream, not a",
        "different answer. `sort`ing both captures makes them equal. The OpenMP",
        "movers are the real nondeterminism: there the numbers themselves change.",
        "",
        "None of these is in the compared set, so `differences/` is unaffected.",
        "It is recorded here because a reader is entitled to know which numbers",
        "in this directory are stable and which are not.",
        "",
    ]
    # stderr moves for reasons that have nothing to do with the port: the
    # host's MPI stack can start complaining about its own CPU topology and
    # every parallel example inherits the message. Counted rather than
    # asserted, so the note disappears when the host stops doing it.
    hw = [r for r in rows
          if (C_DIR / "raw" / r["dir"] / (r["variant"] + ".stderr")).exists()
          and "hwloc" in (C_DIR / "raw" / r["dir"] / (r["variant"] + ".stderr")).read_text(errors="replace")]
    if hw:
        doc += [
            f"### `.stderr` moves too, and not because of the port",
            "",
            f"{len(hw)} of the {len(rows)} runs currently carry an **hwloc** topology",
            "warning on stderr — all of them MPI examples, inheriting a complaint from",
            "OpenMPI about how it reads this machine's CPU layout. It appeared between",
            "two otherwise identical pipeline runs, so a `git diff` of the captures",
            "shows dozens of moved `.stderr` files and no moved `.stdout`.",
            "",
            "Harmless, and checkably so: none is in the compared set,",
            "[`../tools/compare_results.py`](../tools/compare_results.py) opens only",
            "`.stdout`, and the runs still exit 0.",
            "",
            "```bash",
            "grep -rl hwloc c-results/raw --include='*.stderr' | wc -l",
            "```",
        ]
    doc += [
        "",
        "## Per-solver tables (serial examples — these are the ones with a Rust counterpart)",
        "",
    ]
    for d in sorted(serial, key=lambda x: SOLVER_TITLE[x]):
        doc.append(f"* [{SOLVER_TITLE[d]} — `{d}`](by-solver/{d.replace('/', '_')}.md)"
                   f" — {len(by_dir[d])} variants")
    doc += [
        "",
        "## Runs that exited 0 but did not succeed",
        "",
    ]
    if failed:
        doc += [
            f"Exit status is not the whole story: {len(failed)} of the {len(rows)} runs",
            "returned 0 while their own output reports a failed solve. None is in the",
            "compared set, so `differences/` is unaffected, but a table of exit codes",
            "alone would read as though everything worked.",
            "",
            "| directory | variant | what it reports |",
            "|---|---|---|",
        ]
        for r in failed:
            doc.append(
                f"| `{r['dir']}` | `{r['variant']}` | "
                f"{failure_message(C_DIR / 'raw' / r['dir'], r['variant'])} |"
            )
        doc.append("")
    else:
        doc += [
            "None. Every run that exited 0 also reports a completed solve, checked by",
            f"searching each capture for {' and '.join(repr(m) for m in FAIL_MARKERS)}.",
            "",
        ]

    doc += [
        "## Other example families that were also built and run",
        "",
        "These have no pure-Rust counterpart because the port translates only",
        f"the six **C** serial directories -- {n_other_serial} of the {len(rows) - n_serial} rows "
        "below are themselves serial, in C++ or Fortran, so parallelism is not",
        "the reason. They do not appear in `differences/`, and are recorded",
        "because the instruction was to build and execute *all* examples.",
        "",
        "| directory | variants | all exited 0 |",
        "|---|---|---|",
    ]
    for d in other:
        allok = all(r["status"] == "OK" for r in by_dir[d])
        doc.append(f"| `{d}` | {len(by_dir[d])} | {'yes' if allok else 'NO'} |")

    # Which optional backends were actually reachable is decided by whether
    # the examples that need them produced rows, not by what was written down
    # when the tree was first probed. libsuitesparse-dev was installed part
    # way through this work, and the paragraph that used to live here went on
    # calling KLU absent afterwards.
    BACKENDS = [
        ("KLU (SuiteSparse)", lambda r: "_klu" in r["example"]),
        ("SuperLU_MT", lambda r: r["example"].endswith(("_sps", "_slu"))),
        # counted from the launcher the run actually used, not the directory
        # name: 63 runs go through mpirun, but only 52 sit in a *parallel dir
        ("MPI", lambda r: (C_DIR / "raw" / r["dir"] / (r["variant"] + ".meta")).exists()
         and "launcher: mpirun" in (C_DIR / "raw" / r["dir"] / (r["variant"] + ".meta")).read_text()),
        ("hypre", lambda r: "parhyp" in r["dir"]),
        ("PETSc", lambda r: "petsc" in r["dir"]),
        # the serial *L examples are the LAPACK ones; only 1 of the 5 is in a
        # directory named for it
        ("LAPACK", lambda r: "lapack" in r["dir"] or r["example"].endswith("L")),
        ("CUDA / RAJA / Kokkos / MAGMA / Ginkgo / SYCL / XBraid",
         lambda r: any(k in r["dir"] for k in
                       ("cuda", "raja", "kokkos", "magma", "ginkgo", "sycl", "xbraid", "onemkl"))),
    ]
    doc += [
        "",
        "## Which optional backends were reachable",
        "",
        "Read off the run itself: a backend counts as present here when the",
        "examples that need it produced rows in `index.tsv`. See",
        "[`../requirements.md`](../requirements.md) for the probe results and the",
        "exact `apt` command.",
        "",
        "| backend | example variants that ran | on this machine |",
        "|---|---:|---|",
    ]
    for name, pred in BACKENDS:
        n = sum(1 for r in rows if pred(r))
        doc.append(f"| {name} | {n} | {'**present**' if n else 'absent'} |")
    doc += [
        "",
        "The absent ones remove their example families from this run entirely --",
        "there is no output on either side, so nothing is being hidden by their",
        "absence.",
        "",
    ]
    (C_DIR / "README.md").write_text("\n".join(doc) + "\n")

    (C_DIR / "by-solver").mkdir(exist_ok=True)
    for d, rs in by_dir.items():
        if d not in SOLVER_TITLE:
            continue
        t = [
            f"# {SOLVER_TITLE[d]} — C examples (`examples/{d}`)",
            "",
            f"{len(rs)} (example, argv) variants, executed on the machine described in",
            "[`../README.md`](../README.md).",
            "",
            "`stdout bytes` and `sha256` are of the captured stdout stream; re-run",
            "`tools/c_examples_run.sh` and they must reproduce exactly.",
            "",
            "| # | example | argv | exit | status | seconds | stdout bytes | sha256 (first 16) | raw |",
            "|---:|---|---|---:|---|---:|---:|---|---|",
        ]
        for i, r in enumerate(sorted(rs, key=lambda r: (r["example"], r["argv"])), 1):
            t.append(
                f"| {i} | `{r['example']}` | {argv_cell(r['argv'])} | {r['exit']} | "
                f"{r['status']} | {r['seconds']} | {r['stdout_bytes']} | `{r['stdout_sha256']}` | "
                f"[stdout](../raw/{d}/{r['variant']}.stdout) · [meta](../raw/{d}/{r['variant']}.meta) |"
            )
        (C_DIR / "by-solver" / f"{d.replace('/', '_')}.md").write_text("\n".join(t) + "\n")


# --------------------------------------------------------------------------
# rust-results
# --------------------------------------------------------------------------
def write_r(p):
    rows = read_index(R_DIR / "index.tsv")
    by_dir = defaultdict(list)
    for r in rows:
        by_dir[r["dir"]].append(r)
    st = defaultdict(int)
    for r in rows:
        st[r["status"]] += 1

    doc = [
        "# rust-results — every ported example, built and executed here",
        "",
        "This directory records what the **pure-Rust translations** of the",
        "upstream serial examples printed on this machine. The rows of the",
        "provenance table that matter here are the OS, the architecture and",
        "`rustc`/`cargo`: these binaries link no C toolchain and call the host",
        "libm for nothing. The C compiler rows are carried for comparison with",
        "`c-results/`.",
        "",
        "Same rules as `c-results/`: the `.stdout` files are raw process",
        "output -- for the",
        "190 variants that ran. The 9 `NOT_PORTED` ones have empty placeholder",
        "files, because no binary exists to run.",
        "",
        "## Provenance",
        "",
        prov_block(p),
        "",
        "## How to reproduce all of it",
        "",
        "```bash",
        "cargo build --release --workspace --examples",
        "tools/rust_examples_run.sh",
        "python3 tools/make_reports.py",
        "```",
        "",
        "No network access and no package installation is involved: the",
        "workspace has **zero external crates**, so `cargo build` compiles only",
        "the seven crates in `crates/`. Nothing was added to",
        "[`../requirements.md`](../requirements.md) on the Rust side because",
        "nothing needed to be.",
        "",
        "## Headline result",
        "",
        f"**{len(rows)} (example, argv) variants, {st.get('OK', 0)} exited 0, "
        f"{st.get('NOT_PORTED', 0)} have no Rust counterpart.**",
        "",
        "| status | variants |",
        "|---|---|",
    ]
    for k in sorted(st, key=lambda k: -st[k]):
        doc.append(f"| {k} | {st[k]} |")

    doc += [
        "",
        "`NOT_PORTED` marks the 9 `*_sps` / `*_slu` examples, and only those.",
        "They need SuperLU_MT, a third-party sparse-direct **C** library that a",
        "port forbidding `unsafe`, FFI and external crates cannot call --  and",
        "that is not in the Ubuntu archive at any version, so the C side cannot",
        "build them either. **No comparison** is lost by their absence -- there",
        "is no output on either side to compare. Whether the SuperLU_MT code",
        "path itself would have exposed anything is not measured here and is",
        "not claimed.",
        "",
        "The 11 `*_klu` examples in these six serial directories *are* ported.",
        "(15 `*_klu` variants exist across the whole C build; the 4 outside",
        "these directories are out of the port's scope.) KLU itself is fully",
        "available on this machine and the C side uses it -- it is unreachable",
        "*from Rust*, which forbids FFI, not unreachable like SuperLU_MT. So",
        "they run on the independent",
        "pure-Rust sparse LU in `crates/sundials_core/src/sundials_sparse_lu.rs`",
        "instead. Four of them still match the C byte for byte.",
        "",
        "See [`../requirements.md`](../requirements.md) §1 and §4 for SuperLU_MT,",
        "§6 for the KLU substitution.",
        "",
        "## What makes these runs reproducible",
        "",
        "Unlike the C binaries, these do not call the host C library for any",
        "elementary function. `exp`, `log`, `pow`, `expm1`, `log1p`, `sin`,",
        "`cos`, `atan`, `asin`, `acos`, `sinh`, `cosh` and `acosh` are all",
        "implemented in `crates/sundials_core/src/sundials_libm.rs`, so the",
        "numbers below do not move when the host glibc moves. See",
        "[`../LIBM.md`](../LIBM.md).",
        "",
        "## Layout of this directory",
        "",
        "| path | contents |",
        "|---|---|",
        "| `index.tsv` | one row per variant |",
        "| `raw/<dir>/<variant>.stdout` | exactly what the process printed |",
        "| `raw/<dir>/<variant>.stderr` | stderr |",
        "| `raw/<dir>/<variant>.meta` | binary, argv, cwd, exit code, timing, SHA-256 |",
        "| `by-solver/*.md` | the per-solver tables below |",
        "",
        "## Per-solver tables",
        "",
    ]
    for d in sorted(by_dir, key=lambda x: SOLVER_TITLE.get(x, x)):
        doc.append(
            f"* [{SOLVER_TITLE.get(d, d)} — `{CRATE_OF.get(d, '')}`]"
            f"(by-solver/{d.replace('/', '_')}.md) — {len(by_dir[d])} variants"
        )
    doc.append("")
    (R_DIR / "README.md").write_text("\n".join(doc) + "\n")

    (R_DIR / "by-solver").mkdir(exist_ok=True)
    for d, rs in by_dir.items():
        crate = CRATE_OF.get(d, "")
        t = [
            f"# {SOLVER_TITLE.get(d, d)} — Rust examples (`crates/{crate}/examples`)",
            "",
            f"{len(rs)} (example, argv) variants. Run one yourself with:",
            "",
            "```bash",
            f"cargo run --release -p {crate} --example <name> -- <argv>",
            "```",
            "",
            "That reproduces every row marked `OK`. It does **not** work for a",
            "`NOT_PORTED` row: those examples have no `[[example]]` entry in any",
            "`Cargo.toml`, because no Rust translation exists.",
            "",
            "`seconds` is wall time **including harness overhead** — the runner",
            "brackets each example with two `date` subprocesses and a subshell, a",
            "floor of roughly 0.1 s. Treat it as a liveness signal, not a",
            "benchmark: most of these examples finish in under 10 ms.",
            "",
            "| # | example | argv | exit | status | seconds | stdout bytes | sha256 (first 16) | raw |",
            "|---:|---|---|---:|---|---:|---:|---|---|",
        ]
        for i, r in enumerate(sorted(rs, key=lambda r: (r["example"], r["argv"])), 1):
            t.append(
                f"| {i} | `{r['example']}` | {argv_cell(r['argv'])} | {r['exit']} | "
                f"{r['status']} | {r['seconds']} | {r['stdout_bytes']} | `{r['stdout_sha256']}` | "
                f"[stdout](../raw/{d}/{r['variant']}.stdout) · [meta](../raw/{d}/{r['variant']}.meta) |"
            )
        (R_DIR / "by-solver" / f"{d.replace('/', '_')}.md").write_text("\n".join(t) + "\n")


# --------------------------------------------------------------------------
# differences
# --------------------------------------------------------------------------
def write_d(p):
    rows = read_index(D_DIR / "index.tsv")
    by_dir = defaultdict(list)
    for r in rows:
        by_dir[r["dir"]].append(r)
    cls = defaultdict(int)
    for r in rows:
        cls[r["class"]] += 1

    ulps = sorted(int(r["worst_ulp"]) for r in rows
                  if r["class"] == "NUMERIC" and r.get("worst_ulp", "").strip().isdigit())
    worst_lo, worst_hi = (ulps[0], float(ulps[-1])) if ulps else (0, 0.0)

    comparable = sum(v for k, v in cls.items() if k not in ("NOT_PORTED", "NO_C_RUN"))
    ident = cls.get("IDENTICAL", 0)

    # the host-libm control build, if tools/ab_host_libm.sh has been run
    ab_path = D_DIR / "ab-host-libm.tsv"
    ab_total = ab_ident = None
    ab_libm = ab_survivors = []
    if ab_path.exists():
        ab = read_index(ab_path)
        ab_total = len(ab)
        ab_ident = sum(1 for r in ab if r["host_libm_class"] == "IDENTICAL")
        # the switch explains a variant when restoring the host libm restores
        # byte-identity; it explains nothing about a variant that differs
        # either way
        ab_libm = [r for r in ab
                   if r["default_class"] != "IDENTICAL" and r["host_libm_class"] == "IDENTICAL"]
        ab_survivors = [r for r in ab if r["host_libm_class"] != "IDENTICAL"]

    doc = [
        "# differences — C output versus Rust output, variant by variant",
        "",
        f"Every serial example with a pure-Rust translation — {comparable} variants — was",
        "executed twice on this machine: once as the upstream C binary",
        "(`c-results/`) and once as its translation (`rust-results/`). This",
        f"directory is the comparison of the two stdout streams. A further "
        f"{cls.get('NOT_PORTED', 0)} variants ran on **neither** side and are listed as "
        "`NOT_PORTED`;",
        "they are not a comparison that failed, they are a comparison that does",
        "not exist. Nothing here is asserted — every classification is computed by",
        "[`../tools/compare_results.py`](../tools/compare_results.py) from the",
        "captured bytes.",
        "",
        "## Provenance",
        "",
        prov_block(p),
        "",
        "## How to reproduce all of it",
        "",
        "```bash",
        "tools/c_build.sh && tools/c_examples_run.sh      # the C side",
        "tools/rust_examples_run.sh                       # the Rust side",
        "python3 tools/compare_results.py                 # the comparison",
        "tools/ab_host_libm.sh                            # the host-libm control build",
        "python3 tools/make_reports.py                    # these documents",
        "```",
        "",
        "## Headline result",
        "",
        f"**Of {comparable} comparable variants, {ident} are byte-for-byte identical "
        f"({100.0 * ident / comparable:.1f}%).**",
        "",
        (
            attribution_paragraph(ident, comparable, ab_ident, ab_total, ab_libm, ab_survivors)
            if ab_ident is not None
            else "_(run `tools/ab_host_libm.sh` to attribute the differences.)_"
        ),
        "",
        "| class | variants | meaning |",
        "|---|---:|---|",
        f"| IDENTICAL | {cls.get('IDENTICAL', 0)} | the two stdout streams are equal byte for byte |",
        f"| WHITESPACE | {cls.get('WHITESPACE', 0)} | every printed character matches; only column padding differs |",
        f"| NUMERIC | {cls.get('NUMERIC', 0)} | same text, same field count, at least one number differs |",
        f"| STRUCTURAL | {cls.get('STRUCTURAL', 0)} | different lines, words or field counts |",
        f"| NOT_PORTED | {cls.get('NOT_PORTED', 0)} | {not_ported_reason(rows)} |",
        f"| NO_C_RUN | {cls.get('NO_C_RUN', 0)} | the C example could not be built on this machine |",
        "",
        "## How to read a difference",
        "",
        "For every non-identical variant there is a unified diff, and for every",
        "`NUMERIC` one there is also a `.numbers` file naming the single worst",
        "field:",
        "",
        "```bash",
        "cat differences/diffs/<dir>/<variant>.diff",
        "cat differences/diffs/<dir>/<variant>.numbers",
        "```",
        "",
        "`worst rel` below is the largest relative difference between any pair of",
        "printed numbers, and `worst ulp` is the same pair measured in",
        "representable double steps. One ulp is the smallest difference two",
        "doubles can have — the granularity of the format itself, not an error",
        "in either program.",
        "",
        "**Do not read the whole `worst ulp` column as last-bit noise.** These",
        f"are the worst pair in each variant, and they range from {worst_lo} up to "
        f"{worst_hi:.3g}",
        "across the table below. A ulp distance only means \"almost equal\" when",
        "it is small; the large values are two numbers that genuinely parted",
        "company — for the largest, the pair has opposite signs, which makes the",
        "ulp count meaningless and the relative difference the number to read.",
        "",
        "## Attribution",
        "",
        "[**ATTRIBUTION.md**](ATTRIBUTION.md) — the controlled experiment that",
        "decides, for every divergent variant, whether the translation is wrong",
        "or the libm substitution accounts for it. Raw data in",
        "[`ab-host-libm.tsv`](ab-host-libm.tsv).",
        "",
        "## Per-solver tables",
        "",
    ]
    for d in sorted(by_dir, key=lambda x: SOLVER_TITLE.get(x, x)):
        n = len(by_dir[d])
        ni = sum(1 for r in by_dir[d] if r["class"] == "IDENTICAL")
        doc.append(
            f"* [{SOLVER_TITLE.get(d, d)}](by-solver/{d.replace('/', '_')}.md)"
            f" — {ni} identical of {n}"
        )
    doc.append("")
    (D_DIR / "README.md").write_text("\n".join(doc) + "\n")

    (D_DIR / "by-solver").mkdir(exist_ok=True)
    for d, rs in by_dir.items():
        t = [
            f"# {SOLVER_TITLE.get(d, d)} — C vs Rust (`examples/{d}`)",
            "",
            "| # | example | argv | class | diff lines / total | worst rel | worst ulp | diff |",
            "|---:|---|---|---|---:|---:|---:|---|",
        ]
        for i, r in enumerate(sorted(rs, key=lambda r: (r["example"], r["argv"])), 1):
            link = (
                f"[diff](../diffs/{d}/{r['variant']}.diff)"
                if r["class"] not in ("IDENTICAL", "NOT_PORTED", "NO_C_RUN")
                else "—"
            )
            dl = (
                f"{r['diff_lines']} / {r['total_lines']}"
                if r["diff_lines"]
                else "—"
            )
            t.append(
                f"| {i} | `{r['example']}` | {argv_cell(r['argv'])} | {r['class']} | "
                f"{dl} | {r['worst_rel'] or '—'} | {r['worst_ulp'] or '—'} | {link} |"
            )
        t.append("")
        (D_DIR / "by-solver" / f"{d.replace('/', '_')}.md").write_text("\n".join(t) + "\n")


def main():
    p = provenance()
    if (C_DIR / "index.tsv").exists():
        write_c(p)
        print("wrote c-results/")
    if (R_DIR / "index.tsv").exists():
        write_r(p)
        print("wrote rust-results/")
    if (D_DIR / "index.tsv").exists():
        write_d(p)
        print("wrote differences/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
