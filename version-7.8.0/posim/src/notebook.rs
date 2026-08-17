//! Notebook-style REPL: numbered `In[n]`/`Out[n]` cells like a
//! Jupyter/Mathematica notebook. Enter executes the current line
//! (the terminal's equivalent of shift-enter); previous cells can be
//! revisited and edited with magics:
//!
//! - `%history`          — show all cells
//! - `%edit n <text>`    — replace cell n's input and re-execute it
//! - `%rerun n`          — execute cell n's input again
//! - `%save <file>`      — save all inputs as a replayable script
//! - `%load <file>`      — replay a script file
//! - `%reset`            — clear the simulator state
//! - `%quit` / `%exit`   — leave
//!
//! (Pure `std` has no raw terminal mode, so cursor-key cell navigation
//! is delegated to the JupyterLab front end via `posim --machine`.)

use std::io::{BufRead, Write};

use crate::vm::{execute_line, SimState, Value};

pub struct Cell {
    pub input: String,
    pub output: String,
    pub ok: bool,
}

#[derive(Default)]
pub struct Notebook {
    pub cells: Vec<Cell>,
    pub state: SimState,
}

impl Notebook {
    /// Executes one input line as a new numbered cell; returns the
    /// rendered output lines to display.
    pub fn execute_cell(&mut self, input: &str) -> String {
        let n = self.cells.len() + 1;
        let (output, ok) = match execute_line(input, &mut self.state) {
            Ok(Value::Unit) => (String::new(), true),
            Ok(v) => (v.to_string(), true),
            Err(e) => (e, false),
        };
        let rendered = if output.is_empty() {
            String::new()
        } else if ok {
            format!("Out[{n}]= {output}")
        } else {
            format!("Err[{n}]: {output}")
        };
        self.cells.push(Cell { input: input.to_string(), output, ok });
        rendered
    }

    /// Handles a `%magic` line; returns the text to display, or `None`
    /// if the notebook should quit.
    pub fn magic(&mut self, line: &str) -> Option<String> {
        let mut parts = line.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("");
        let rest = parts.next().unwrap_or("").trim().to_string();
        match cmd {
            "%quit" | "%exit" => None,
            "%history" => {
                let mut out = String::new();
                for (i, c) in self.cells.iter().enumerate() {
                    let marker = if c.ok { " " } else { "!" };
                    out.push_str(&format!("{marker}In[{}]:= {}\n", i + 1, c.input));
                    if !c.output.is_empty() {
                        let label = if c.ok { "Out" } else { "Err" };
                        out.push_str(&format!("  {label}[{}]= {}\n", i + 1, c.output));
                    }
                }
                if out.is_empty() {
                    out.push_str("(no history)\n");
                }
                out.pop();
                Some(out)
            }
            "%rerun" => match rest.parse::<usize>() {
                Ok(n) if n >= 1 && n <= self.cells.len() => {
                    let input = self.cells[n - 1].input.clone();
                    let echo = format!("In[{}]:= {}", self.cells.len() + 1, input);
                    let out = self.execute_cell(&input);
                    Some(if out.is_empty() { echo } else { format!("{echo}\n{out}") })
                }
                _ => Some(format!("%rerun: no cell {rest}")),
            },
            "%edit" => {
                let mut p = rest.splitn(2, char::is_whitespace);
                let idx = p.next().unwrap_or("").parse::<usize>();
                let new_text = p.next().unwrap_or("").trim().to_string();
                match idx {
                    Ok(n) if n >= 1 && n <= self.cells.len() && !new_text.is_empty() => {
                        self.cells[n - 1].input = new_text.clone();
                        let echo = format!("In[{}]:= {}", self.cells.len() + 1, new_text);
                        let out = self.execute_cell(&new_text);
                        Some(if out.is_empty() { echo } else { format!("{echo}\n{out}") })
                    }
                    Ok(n) if n >= 1 && n <= self.cells.len() => {
                        Some(format!("current In[{n}]:= {}\nusage: %edit {n} <new text>", self.cells[n - 1].input))
                    }
                    _ => Some("usage: %edit <cell number> <new text>".to_string()),
                }
            }
            "%save" => {
                if rest.is_empty() {
                    return Some("usage: %save <file>".to_string());
                }
                let mut body = String::new();
                let mut written = 0usize;
                for c in &self.cells {
                    if c.ok && !c.input.trim_start().starts_with('%') {
                        body.push_str(&c.input);
                        body.push('\n');
                        written += 1;
                    }
                }
                match std::fs::write(&rest, body) {
                    Ok(()) => Some(format!("saved {written} cell(s) to {rest}")),
                    Err(e) => Some(format!("%save failed: {e}")),
                }
            }
            "%load" => match std::fs::read_to_string(&rest) {
                Ok(text) => {
                    /* joins continuation lines by brace depth, exactly
                     * like script mode — a %saved multi-line DEF must
                     * replay as ONE cell */
                    let mut shown = Vec::new();
                    for cell in script_cells(&text) {
                        shown.push(format!("In[{}]:= {}", self.cells.len() + 1, cell));
                        let out = self.execute_cell(&cell);
                        if !out.is_empty() {
                            shown.push(out);
                        }
                    }
                    Some(shown.join("\n"))
                }
                Err(e) => Some(format!("%load {rest} failed: {e}")),
            },
            "%reset" => {
                self.state = SimState::default();
                Some("system reset".to_string())
            }
            other => Some(format!("unknown magic `{other}` — see HELP")),
        }
    }
}

/// Net `{`/`}` depth of a line, ignoring braces inside string
/// literals and after `#` comments — the notebook keeps reading
/// continuation lines while a `DEF ... {` block is open.
fn brace_delta(line: &str) -> i32 {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut prev_escape = false;
    for c in line.chars() {
        if in_str {
            if prev_escape {
                prev_escape = false;
            } else if c == '\\' {
                prev_escape = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '#' => break,
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
    }
    depth
}

/// Interactive notebook loop over stdin/stdout.
pub fn repl() {
    let mut nb = Notebook::default();
    println!("posim — physical_object simulator notebook (sundials_rs backend)");
    println!("type HELP for the command language, %quit to leave\n");
    repl_loop(&mut nb);
}

/// The interactive cell loop, shared by plain `posim` and
/// `posim --notebook` (which pre-loads a script into `nb` first, so
/// cell numbering and the scene window carry straight on).
fn repl_loop(nb: &mut Notebook) {
    let stdin = std::io::stdin();
    loop {
        print!("In[{}]:= ", nb.cells.len() + 1);
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, /* EOF */
            Ok(_) => {}
            Err(_) => break,
        }
        let mut line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        /* multi-line input: keep reading while braces stay open (a
         * DEF body); the prompt shows the continuation */
        let mut depth = brace_delta(&line);
        while depth > 0 {
            print!("  ...:= ");
            let _ = std::io::stdout().flush();
            let mut more = String::new();
            match stdin.lock().read_line(&mut more) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
            depth += brace_delta(&more);
            line.push('\n');
            line.push_str(more.trim_end());
        }
        let line = line.as_str();
        if line.starts_with('%') {
            match nb.magic(line) {
                Some(msg) => {
                    if !msg.is_empty() {
                        println!("{msg}");
                    }
                }
                None => break,
            }
        } else {
            let out = nb.execute_cell(line);
            if !out.is_empty() {
                println!("{out}");
            }
        }
    }
    println!("goodbye");
}

/// Batch mode: execute a script file, echoing cells.
pub fn run_script(path: &str) -> Result<(), String> {
    let mut nb = Notebook::default();
    if replay_into(&mut nb, path)? {
        Err("script had failing cells".to_string())
    } else {
        Ok(())
    }
}

/// Dynamic-notebook mode: execute a notebook file, then stay in the
/// interactive loop — the loaded cells keep their `In[n]` numbers and
/// the next prompt continues the numbering, exactly like opening a
/// saved notebook in Jupyter. The sign-off line reports what the file
/// actually left behind: a live scene window, bodies awaiting SCENE
/// CREATE, a quantum problem (which the 3-D scene cannot draw), or
/// pure numerics.
pub fn run_notebook(path: &str) -> Result<(), String> {
    let mut nb = Notebook::default();
    println!("posim — physical_object simulator notebook (sundials_rs backend)");
    println!("loading dynamic notebook {path}\n");
    if replay_into(&mut nb, path)? {
        return Err(format!("dynamic notebook {path} had failing cells"));
    }
    println!("\n{path} loaded — {}", loaded_hint(&nb));
    println!("type HELP for commands, %quit to leave\n");
    repl_loop(&mut nb);
    Ok(())
}

/// The state-dependent half of the `--notebook` sign-off message.
/// Never promises a scene window that does not exist.
fn loaded_hint(nb: &Notebook) -> String {
    let bodies = nb.state.system.objects.len();
    if nb.state.scene.is_some() {
        return "the simulation is ready: press Start in the scene\n\
                window (or type SCENE START)"
            .to_string();
    }
    if bodies > 0 {
        return format!(
            "{bodies} bod{} loaded; type SCENE CREATE to open the scene\n\
             window, then SCENE START (or STEP/RUN here at the prompt)",
            if bodies == 1 { "y is" } else { "ies are" },
        );
    }
    if nb.state.qm.grid.is_some() || nb.state.qm.psi.is_some() {
        return "a 1-D quantum problem is set up. The 3-D scene window draws\n\
                rigid bodies only — QM ANIMATE \"<file>.html\" <time> writes a\n\
                browser film of |psi|^2 instead (QM shows the configuration)"
            .to_string();
    }
    if nb.state.qm2.grid.is_some() || nb.state.qm2.psi.is_some() {
        return "a 2-D quantum problem is set up. The 3-D scene window draws\n\
                rigid bodies only — QM2 ANIMATE \"<file>.html\" <time> writes a\n\
                browser film of |psi|^2 instead (QM2 shows the configuration)"
            .to_string();
    }
    if nb.state.qm3.grid.is_some() || nb.state.qm3.psi.is_some() {
        return "a 3-D quantum problem is set up. The 3-D scene window draws\n\
                rigid bodies only — QM3 ANIMATE \"<file>.html\" <time> writes a\n\
                browser film of the marginals instead (QM3 shows the configuration)"
            .to_string();
    }
    "its results are printed above (no bodies were left in the\n\
     system, so there is nothing for the scene window to show)"
        .to_string()
}

/// Splits script text into executable cells: blank and `#` comment
/// lines are skipped, and continuation lines are joined by brace depth
/// so a multi-line DEF is ONE cell. An unterminated trailing block is
/// dropped, as the two former copies of this loop both did. One home
/// for the joining rule — the `%load` magic and `replay_into` had
/// separately maintained copies that had already diverged.
fn script_cells(text: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut pending = String::new();
    let mut depth = 0i32;
    for raw in text.lines() {
        let line = raw.trim_end();
        if depth == 0 {
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            pending = t.to_string();
            depth = brace_delta(t);
        } else {
            depth += brace_delta(line);
            pending.push('\n');
            pending.push_str(line);
        }
        if depth > 0 {
            continue; /* still inside a DEF block */
        }
        cells.push(std::mem::take(&mut pending));
    }
    cells
}

/// Replays a script file into an existing notebook, echoing cells;
/// returns whether any cell failed. Continuation lines are joined by
/// brace depth so a multi-line DEF replays as one cell.
fn replay_into(nb: &mut Notebook, path: &str) -> Result<bool, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut failed = false;
    for cell in script_cells(&text) {
        let line = cell.as_str();
        println!("In[{}]:= {}", nb.cells.len() + 1, line);
        if line.starts_with('%') {
            match nb.magic(line) {
                Some(msg) => {
                    if !msg.is_empty() {
                        println!("{msg}");
                    }
                }
                None => break,
            }
        } else {
            let out = nb.execute_cell(line);
            if !out.is_empty() {
                println!("{out}");
            }
            if let Some(c) = nb.cells.last() {
                if !c.ok {
                    failed = true;
                }
            }
        }
    }
    Ok(failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_number_and_capture_output() {
        let mut nb = Notebook::default();
        let out = nb.execute_cell("1 + 1");
        assert_eq!(out, "Out[1]= 2");
        let out = nb.execute_cell("new point { mass = 2 }");
        assert_eq!(out, "Out[2]= obj0");
        let out = nb.execute_cell("bogus syntax !!");
        assert!(out.starts_with("Err[3]:"), "{out}");
        assert_eq!(nb.cells.len(), 3);
        assert!(!nb.cells[2].ok);
    }

    #[test]
    fn save_load_round_trips_a_multiline_def() {
        let dir = std::env::temp_dir().join("posim_nb_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("roundtrip.posim");
        let path_s = path.to_string_lossy().to_string();

        let mut nb = Notebook::default();
        let def = "def probe(m = 2) {\n  new sphere { mass = m }\n}";
        assert!(nb.execute_cell(def).contains("defined"));
        nb.execute_cell("probe()");
        assert_eq!(nb.state.system.objects.len(), 1);
        let saved = nb.magic(&format!("%save {path_s}")).unwrap();
        assert!(saved.contains("saved"), "{saved}");

        /* a fresh notebook replays the file: the multi-line DEF must
         * come back as ONE cell and the call must work again */
        let mut nb2 = Notebook::default();
        let out = nb2.magic(&format!("%load {path_s}")).unwrap();
        assert!(out.contains("defined"), "DEF replayed: {out}");
        assert_eq!(nb2.state.system.objects.len(), 1, "probe() replayed");
        assert_eq!(
            nb2.state.system.objects[0].get_mass(),
            2.0,
            "default argument survived the round-trip"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn replay_into_keeps_state_and_numbering_for_the_repl() {
        // The dynamic-notebook contract: after loading a file, the
        // SAME notebook continues interactively — objects exist, the
        // loaded cells hold In[1..k], and the next cell is In[k+1].
        let dir = std::env::temp_dir().join("posim_nb_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("dynamic.posim");
        std::fs::write(
            &path,
            "# a tiny dynamic notebook\nnew sphere as ball { mass = 2, radius = 0.5 }\nenergy\n",
        )
        .unwrap();
        let mut nb = Notebook::default();
        let failed = replay_into(&mut nb, &path.to_string_lossy()).unwrap();
        assert!(!failed);
        assert_eq!(nb.cells.len(), 2, "comment skipped, two real cells");
        assert_eq!(nb.state.system.objects.len(), 1);
        let out = nb.execute_cell("get ball.mass");
        assert_eq!(out, "Out[3]= 2", "numbering continues after the load");
        // A missing file is a readable error, not a panic.
        assert!(replay_into(&mut Notebook::default(), "no_such_file.posim").is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn magics_edit_and_rerun() {
        let mut nb = Notebook::default();
        nb.execute_cell("new point { mass = 2 }");
        nb.execute_cell("get obj0.mass");
        assert_eq!(nb.cells[1].output, "2");
        let out = nb.magic("%edit 2 get obj0.inverse_mass").unwrap();
        assert!(out.contains("Out[3]= 0.5"), "{out}");
        let out = nb.magic("%rerun 1").unwrap();
        assert!(out.contains("obj1"), "{out}");
        let hist = nb.magic("%history").unwrap();
        assert!(hist.contains("In[1]:="), "{hist}");
        assert!(nb.magic("%quit").is_none());
    }
}
