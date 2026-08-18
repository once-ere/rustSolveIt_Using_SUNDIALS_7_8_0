"""Invariant prose emitted VERBATIM into every generated notebook.

Requirement 2 from the brief is that a notebook must never send its reader
to another notebook to find an explanation. That is a hard rule, so every
explanation lives in this module and is written into *each* notebook in
full. Two notebooks therefore repeat each other word for word, on purpose.
Nothing here may ever say "see the other notebook", "as explained
elsewhere", or anything of that shape.
"""

# --------------------------------------------------------------------------
# 1. How to launch. Requirement 1: precise, CLI-first, in every notebook.
# --------------------------------------------------------------------------
LAUNCH = r"""## 1. How to open a notebook like this one, from a terminal

This section is repeated in full in every notebook in this project, so
that you never have to go looking in another file for it.

### 1.1 What you need

| you need | why | check it with |
|---|---|---|
| **Rust** (1.75 or newer) | the simulator is pure Rust and is built from source | `rustc --version` |
| **Python 3.8+** | Jupyter runs on Python, and this notebook is a Python notebook | `python3 --version` |
| **JupyterLab** or **Jupyter Notebook** | the program that opens `.ipynb` files | `jupyter --version` |
| the **posim** binary | the simulator itself; you build it once, below | `ls target/release/posim` |

You do **not** need SUNDIALS installed. You do **not** need a C or Fortran
compiler. You do **not** need any package from crates.io. The whole solver
suite — CVODE, CVODES, IDA, IDAS, KINSOL, ARKODE — is a pure-Rust
translation vendored inside this repository, and `cargo` builds it from
the source that is already on your disk. Nothing is downloaded at build
time.

### 1.2 Build the simulator, once

Open a terminal and run these four commands. `$` is the shell prompt —
do not type it.

```bash
$ git clone https://github.com/once-ere/rustSolveIt_Using_SUNDIALS_7_8_0.git
$ cd rustSolveIt_Using_SUNDIALS_7_8_0/version-7.8.0
$ cargo build --release -p posim
$ ls -l target/release/posim
```

The third command is the long one: it compiles the simulator and the whole
vendored SUNDIALS translation, and takes a few minutes the first time.
When it finishes, the fourth command must print a line describing an
executable file. If it prints `No such file or directory`, the build did
not succeed — scroll up in the terminal and read the first error, because
later errors are usually consequences of it.

### 1.3 Install Jupyter, if you do not have it

```bash
$ python3 -m pip install --user jupyterlab
$ jupyter --version
```

If `jupyter` is still "command not found" after this, your `pip --user`
scripts directory is not on your `PATH`. Print it and add it:

```bash
$ python3 -m site --user-base
$ export PATH="$(python3 -m site --user-base)/bin:$PATH"
```

To make that permanent, append that same `export` line to `~/.bashrc`
(bash) or `~/.zshrc` (zsh).

### 1.4 Start Jupyter and open a NEW notebook

From the `version-7.8.0` directory:

```bash
$ jupyter lab
```

That starts a small web server and prints a URL that contains a one-time
token, looking like:

```
    http://localhost:8888/lab?token=8f4c1e...
```

It normally opens your browser by itself. If it does not, copy that whole
URL — token included — into a browser. The token is the password; a URL
without it will be refused.

Then, inside JupyterLab:

1. **File → New → Notebook**.
2. When it asks you to *Select Kernel*, choose **Python 3 (ipykernel)**.
3. The empty notebook appears. Anything you type into a cell is Python.

Choose **Python 3**, and not the "posim" kernel that this repository also
ships, because this notebook is written in Python: it starts the simulator
itself, sends it commands, reads the replies back, and (at the end) opens
a graphical save dialog. Those are Python actions. The posim kernel speaks
only the simulator's own command language and cannot do them.

### 1.5 Running cells

- **Shift+Enter** runs the selected cell and moves to the next one.
- **Ctrl+Enter** runs the selected cell and stays put.
- Run the cells of this notebook **in order, from the top**. Each one
  builds on the state left by the ones above it, so skipping one will
  usually produce an error in a later cell.
- If things get into a confusing state, use **Kernel → Restart Kernel and
  Clear All Outputs**, then start again from the first cell.

### 1.6 If you would rather not use Jupyter at all

Everything this notebook does can also be typed straight into the
simulator's own prompt:

```bash
$ cargo run --release -p posim
```

That gives you an `In[1]:=` prompt where the simulator's commands are
typed directly. Type `HELP` for the command reference and `QUIT` to
leave. The Python in this notebook is a wrapper around exactly that
program.
"""

# --------------------------------------------------------------------------
# 2. What the simulator is, and the vocabulary used throughout.
# --------------------------------------------------------------------------
GLOSSARY = r"""## 2. The words used in this notebook

Repeated in full in every notebook of this project, so that this notebook
stands alone.

**Body.** One rigid object. It has a *shape* (which fixes its moments of
inertia), a mass, a position, a velocity, an orientation and an angular
velocity. Bodies are created with the `NEW` command.

**Boundary / shape.** One of `point`, `sphere`, `cuboid`, `cylinder`,
`disk`, `torus`, `dumbbell`. The shape is not decoration: the simulator
computes the body's inertia tensor from it analytically.

**Static body / anchor.** A body with `inverse_mass = 0`. No force can
move it. It is how a mechanism is bolted to the world. A `point` anchor
additionally has a zero inertia tensor, so no torque can turn it either.

**Joint / constraint.** An exact geometric or kinematic relation that the
solver holds true for all time — not a stiff spring. Each joint
contributes a fixed number of scalar equations, called *rows*:

| command | rows | what it holds | freedoms it leaves |
|---|---|---|---|
| `CONSTRAIN a b [len]` | 1 | a fixed distance between two centres | 5 |
| `GEAR a b <axis> <ratio>` | 1 | a fixed proportion between two turns | 5 |
| `RACK p b <axis> <dir> <r>` | 1 | a turn tied to a slide, `ds = r dtheta` | 5 |
| `BALL a b` | 3 | one shared point | 3 (any rotation about it) |
| `UNIVERSAL a b <u> <w>` | 4 | a shared point and a right angle | 2 |
| `HINGE a b <axis>` | 5 | a shared point and a shared axis | 1 (the swing) |
| `PRISMATIC a b <axis>` | 5 | a line to slide along, and no turning | 1 (the slide) |

**Degrees of freedom.** Each free body has 6 (three of position, three of
orientation). A mechanism of `N` free bodies has `6N`. Subtract the total
rows of all its joints and what remains is how many independent ways the
mechanism can still move. If the subtraction leaves zero the mechanism is
locked; if the rows are not independent of one another the system is
*redundant* and the solver will refuse it rather than guess.

**The pivot rule.** Every joint that holds a point places that point at
the **midpoint of the two bodies' centres, as they stand at the moment the
joint is created**, and then remembers it in each body's own frame. So you
position the parts first and join them second, and the geometry you
assemble is the geometry you get.

**`g` and `g_dot`.** `g` is the vector of constraint equations; the solver
holds `g = 0`. `g_dot` is its time derivative, which the solver also holds
at zero. Their sizes are the total row count. The `CONSTRAINTS` command
prints the worst absolute value of each, and those two numbers are how you
tell whether a mechanism is really being held together: they should stay
near the solver's tolerance and never grow steadily.

**`METHOD IDA`.** Any mechanism with a joint must be integrated by IDA,
the differential-algebraic solver, and the simulator refuses any other
method rather than silently integrating an unconstrained problem.
"""

# --------------------------------------------------------------------------
# 3. The state vector and the first-order reduction. Requirement 6.
# --------------------------------------------------------------------------
FIRST_ORDER = r"""## 3. How a second-order mechanics problem becomes a first-order system

Repeated in full in every notebook of this project.

SUNDIALS — like every general ODE/DAE library — integrates **first-order**
systems. Newton's and Euler's equations are **second order**. This section
is the bridge, and it is the same bridge in every example.

### 3.1 The physics, as written by hand

For each body `i`, with mass `m_i`, inertia tensor `I_i` (in the body's own
frame), position `x_i`, orientation `R_i`, and angular velocity `omega_i`:

```
m_i * d2(x_i)/dt2 = F_i                            (Newton)
I_i * d(omega_i)/dt + omega_i x (I_i omega_i) = T_i   (Euler)
```

Both are second order in the configuration variables: the first is second
order in `x_i`, and the second is second order in orientation, because
`omega_i` is itself the rate of change of `R_i`.

### 3.2 The standard trick: name the velocities

A second-order equation becomes two first-order equations by giving the
first derivative its own name and letting the solver carry it as an
unknown alongside the position. This simulator carries **momentum** rather
than velocity, which is the same trick with a better-conditioned variable:

```
d(x_i)/dt = p_i / m_i                    <- was the definition of velocity
d(p_i)/dt = F_i                          <- was Newton's second law
d(q_i)/dt = 0.5 * quat(omega_i) * q_i    <- was the definition of omega
d(L_i)/dt = T_i                          <- was Euler's equation
```

with `omega_i = R_i I_i^-1 R_i^T L_i`, and `q_i` a unit quaternion holding
the orientation. Orientation is stored as a quaternion rather than as
Euler angles because quaternions have no gimbal lock and renormalise
cheaply.

### 3.3 The packed state vector

Each body contributes exactly **13 numbers**, in this order:

```
index 0..2    x       position
index 3..5    p       linear momentum
index 6..9    q       orientation quaternion, w first
index 10..12  L       angular momentum
```

so a model of `N` bodies has a state vector `y` of length `13N`, and the
whole of mechanics above is one first-order system `dy/dt = f(t, y)`.
That is what is handed to the Rust translation of CVODE or ARKODE.

### 3.4 When there are joints: a DAE, not an ODE

A joint is an algebraic relation between coordinates, not a rate, so it
cannot be written as `dy/dt = ...`. The system becomes *differential-
algebraic*, and this simulator uses the **GGL (Gear-Gupta-Leimkuhler)**
index-2 formulation, which carries **both** the position-level constraint
and its derivative, with a Lagrange multiplier for each:

```
d(x)/dt = v - M^-1 J^T mu        position, corrected by the multiplier mu
d(p)/dt = F + J^T lambda         momentum, plus the constraint force
0       = g(q)                   the joints hold, at the position level
0       = J u                    the joints hold, at the velocity level
```

`J` is the constraint Jacobian: row `k` of `J` is the gradient of the
`k`-th constraint equation with respect to all the coordinates, so `J u`
is the rate of change of the constraints under the current velocities.
`lambda` are the constraint forces, `mu` the position-level correction,
and `M^-1` the inverse mass/inertia metric — which is **not** optional,
because `J` has rows in different units and the correction has to be
mass-weighted to be dimensionally consistent.

Carrying both `g` and `J u` is what keeps the joints exact. A formulation
that enforced only the acceleration level would let `g` drift away
quadratically in time, and a mechanism would slowly come apart.

The unknowns are therefore `y` (the `13N` numbers above) together with
`lambda` and `mu` (one of each per constraint row), and the whole thing is
handed to the Rust translation of **IDA**, which solves systems of the
implicit form `F(t, y, y') = 0`.

### 3.5 Consistent initial conditions

A DAE cannot be started from just any state. The starting state must
already satisfy `g = 0` **and** `J u = 0` — the parts must be assembled,
and their initial velocities must be compatible with the joints. If the
velocities are not compatible, this simulator projects them onto the
nearest compatible set before starting, and reports how much it had to
change them. A start that needs no projection is one where you specified
velocities the mechanism can actually have.
"""

# --------------------------------------------------------------------------
# 4. The driver cell: how this notebook talks to the simulator.
# --------------------------------------------------------------------------
DRIVER_MD = r"""## 4. How this notebook talks to the simulator

Repeated in full in every notebook of this project.

The next cell defines a small helper class. It launches the simulator as a
child process in **machine mode** (`posim --machine`), which makes it speak
JSON Lines: you send it one JSON object per line, and it answers with one
JSON object per line. This is the same interface the project's Jupyter
kernel uses internally.

The three calls used below are:

- `sim.do("...")` — run one simulator command and show what it printed.
  Adding `quiet=True` runs it without printing, which is what you want
  inside a loop that takes thousands of steps.
- `sim.get("obj.field")` — read one value back as a Python object.
- `sim.state()` — fetch the whole simulation state as a Python dictionary.

Nothing here integrates anything: every advance in time is performed by
the Rust SUNDIALS translation inside the child process.
"""

DRIVER_CODE = r'''import json, os, shutil, subprocess, sys
from pathlib import Path

def _find_posim():
    """Locate the posim binary, explaining clearly if it is missing."""
    env = os.environ.get("POSIM_BIN")
    if env and Path(env).is_file():
        return env
    onpath = shutil.which("posim")
    if onpath:
        return onpath
    here = Path.cwd()
    for base in [here, *here.parents]:
        for profile in ("release", "debug"):
            cand = base / "target" / profile / "posim"
            if cand.is_file():
                return str(cand)
    raise SystemExit(
        "Could not find the posim binary.\n"
        "Build it first, from the version-7.8.0 directory:\n"
        "    cargo build --release -p posim\n"
        "or set the POSIM_BIN environment variable to its full path."
    )

class Sim:
    """One posim child process, spoken to in JSON Lines."""

    def __init__(self):
        self.binary = _find_posim()
        self.proc = subprocess.Popen(
            [self.binary, "--machine"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
            env=dict(os.environ, POSIM_NO_BROWSER="1"),
        )
        print(f"simulator started: {self.binary}")

    def _rpc(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("the simulator closed the connection")
            reply = json.loads(line)
            if "event" in reply:      # asynchronous scene notice, not a reply
                continue
            return reply

    def do(self, code, quiet=False):
        """Run one simulator command; print and return what it printed.

        Pass quiet=True inside a loop, where printing a line per step
        would bury the result under thousands of lines of progress.
        """
        r = self._rpc({"op": "exec", "code": code})
        if not r.get("ok"):
            raise RuntimeError(f"the simulator refused {code!r}:\n  {r.get('error')}")
        out = r.get("display") or r.get("result")
        if out not in (None, "") and not quiet:
            print(out)
        return out

    def get(self, path):
        """Read one value, e.g. sim.get('piston.position')."""
        r = self._rpc({"op": "get", "path": path})
        if not r.get("ok"):
            raise RuntimeError(f"could not read {path!r}: {r.get('error')}")
        return r.get("result")

    def state(self):
        """The whole simulation state, as a Python dictionary."""
        r = self._rpc({"op": "state"})
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        return r["result"]

    def close(self):
        try:
            self.proc.stdin.write('{"op":"quit"}\n'); self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.wait(timeout=30)
        for pipe in (self.proc.stdin, self.proc.stdout):
            try: pipe.close()
            except Exception: pass
        print("simulator stopped")

sim = Sim()'''

# --------------------------------------------------------------------------
# 5. Naming and saving. Requirements 4 and 5.
# --------------------------------------------------------------------------
SAVE_MD = r"""## 9. Name this notebook, and choose where to save it

Repeated in full in every notebook of this project.

The next cell asks you for two things:

1. **A name for this notebook.** It is typed into a normal input box that
   appears just under the cell. Press Enter to accept the suggestion shown
   in brackets.
2. **A folder to save it in.** A graphical "Save As" window opens on top of
   your other windows. Pick a folder, adjust the file name if you like, and
   press Save.

The graphical window is drawn by `tkinter`, which is part of Python's own
standard library, so nothing needs installing. If you are running this
notebook on a machine with no graphical display — over a plain SSH
connection, for instance — that window cannot open. The cell detects this,
says so, and falls back to asking you to type the folder path instead. It
will not fail or hang.

Saving writes a **copy** of this notebook, exactly as it stands now,
including every output you have produced. The original file is left
untouched.
"""

SAVE_CODE = r'''import json, os, shutil, sys
from pathlib import Path

# ---- 1. Ask the user to name the notebook -------------------------------
suggested = "__DEFAULT_NAME__"
typed = input(f"Name for this notebook [{suggested}]: ").strip()
notebook_name = typed or suggested
if not notebook_name.endswith(".ipynb"):
    notebook_name += ".ipynb"
print(f"name chosen: {notebook_name}")

# ---- 2. Ask where to save it, with a pop-up dialog ----------------------
def ask_folder_graphically(initial_name):
    """Open a real Save-As window. Returns a path, or None if impossible."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"(no tkinter available: {exc})")
        return None
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("(no graphical display detected: $DISPLAY is not set)")
        return None
    try:
        root = tk.Tk()
        root.withdraw()                 # hide the empty parent window
        root.attributes("-topmost", True)   # put the dialog in front
        chosen = filedialog.asksaveasfilename(
            title="Save this notebook as...",
            initialfile=initial_name,
            defaultextension=".ipynb",
            filetypes=[("Jupyter notebook", "*.ipynb"), ("All files", "*.*")],
        )
        root.destroy()
        return chosen or None
    except Exception as exc:
        print(f"(could not open the dialog: {exc})")
        return None

target = ask_folder_graphically(notebook_name)

if target is None:
    # Fallback for a machine with no display: ask in plain text instead.
    print("Falling back to a typed folder path.")
    default_dir = str(Path.cwd())
    folder = input(f"Folder to save into [{default_dir}]: ").strip() or default_dir
    target = str(Path(folder).expanduser() / notebook_name)

target = Path(target).expanduser()
target.parent.mkdir(parents=True, exist_ok=True)

# ---- 3. Copy this notebook to the chosen place --------------------------
source = Path("__SOURCE_PATH__")
if not source.is_file():
    # The notebook may have been opened from somewhere else; look nearby.
    hits = list(Path.cwd().rglob(source.name))
    source = hits[0] if hits else None

if source is None:
    print(
        "Could not locate this notebook's own file on disk to copy it.\n"
        f"Use JupyterLab's  File -> Save Notebook As...  and save to:\n    {target}"
    )
else:
    shutil.copyfile(source, target)
    print(f"saved: {target}")'''
