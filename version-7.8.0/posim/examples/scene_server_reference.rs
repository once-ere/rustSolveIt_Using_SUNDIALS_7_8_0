//! Reference scene server: the smallest honest browser scene, in std alone.
//!
//! This is a REFERENCE, not the shipping scene. `posim`'s own window lives
//! in `posim/src/scene/` and is what `SCENE CREATE` opens; nothing here is
//! wired into it. The value of this file is that it is small enough to hold
//! in your head all at once — one world, one command path, one stream — so
//! the three architectural fixes below can be read directly off the code
//! rather than inferred from a larger system.
//!
//! Written against the three failure modes in the transcript:
//!
//!   1. `showing 0 entities` after a successful load.
//!      Cause: the loader fills one store, SCENE CREATE builds another.
//!      Fix here: there is exactly ONE world. `scene_create` never
//!      constructs entities; it only opens a view onto the world that
//!      already exists. Loading before or after scene creation both work.
//!
//!   2. Nothing moves in the browser.
//!      Cause: the page receives one static snapshot and never subscribes.
//!      Fix here: `GET /events` is a Server-Sent Events stream that pushes
//!      a snapshot every frame. SSE, not WebSocket, because it is one-way
//!      and needs no handshake, no framing, no dependencies.
//!
//!   3. Buttons do nothing.
//!      Cause: the control channel is never connected, so clicks post into
//!      the void. Fix here: `POST /cmd` takes the same command strings the
//!      REPL takes, so the buttons and `SCENE START` go through identical
//!      code. A button cannot drift from the command it mirrors.
//!
//! Also fixed: `START` with no scene auto-creates one instead of erroring.
//!
//! Run:
//!
//! ```text
//! cargo run --release --example scene_server_reference
//! ```
//!
//! It prints a `http://127.0.0.1:<port>/` URL on a port the OS chooses, and
//! does NOT open a browser — paste the URL yourself. The four buttons and
//! the stdin prompt accept the same command strings; type HELP for the list,
//! Ctrl-C to stop. Uses the standard library only: no dependency on any
//! workspace crate, so it builds and runs even if the rest of the tree does
//! not, which is what makes it usable as a bisection reference.
#![forbid(unsafe_code)]
#![deny(warnings)]

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

const G: f64 = 6.674_30e-11;
const TRAIL_MAX: usize = 240;

#[derive(Clone)]
struct Body {
    name: String,
    m: f64,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    radius_px: f64,
    color: &'static str,
    trail: VecDeque<(f64, f64)>,
}

impl Body {
    /// Position and velocity arrive as pairs rather than four loose f64s:
    /// eight positional arguments of the same type is a swap waiting to
    /// happen, and clippy refuses it at 8/7 anyway.
    fn new(
        name: &str,
        m: f64,
        (x, y): (f64, f64),
        (vx, vy): (f64, f64),
        radius_px: f64,
        color: &'static str,
    ) -> Self {
        Body {
            name: name.to_string(),
            m,
            x,
            y,
            vx,
            vy,
            radius_px,
            color,
            trail: VecDeque::with_capacity(TRAIL_MAX),
        }
    }
}

struct World {
    bodies: Vec<Body>,
    running: bool,
    tick: u64,
    sim_time: f64,
    dt: f64,
    scale: f64,
    source: String,
}

impl World {
    fn new() -> Self {
        World {
            bodies: Vec::new(),
            running: false,
            tick: 0,
            sim_time: 0.0,
            dt: 3600.0,
            scale: 2.6e-10,
            source: String::from("(empty)"),
        }
    }

    /// Stand-in for the `.posim` loader. The only thing that matters
    /// architecturally: it writes into the shared world, not a private one.
    fn load_stage(&mut self, label: &str) {
        self.bodies.clear();
        // Circular-orbit speeds at each aphelion-free radius, y-velocity only:
        // the whole system starts on the +x axis moving +y, so it orbits
        // counter-clockwise and stays visibly closed under Verlet.
        let b = &mut self.bodies;
        b.push(Body::new("Sol", 1.989e30, (0.0, 0.0), (0.0, 0.0), 12.0, "#ffd166"));
        b.push(Body::new("Mercury", 3.301e23, (5.79e10, 0.0), (0.0, 47_360.0), 3.0, "#b0a999"));
        b.push(Body::new("Venus", 4.867e24, (1.082e11, 0.0), (0.0, 35_020.0), 5.0, "#e0b070"));
        b.push(Body::new("Terra", 5.972e24, (1.496e11, 0.0), (0.0, 29_780.0), 5.0, "#6fb3d9"));
        b.push(Body::new("Mars", 6.417e23, (2.279e11, 0.0), (0.0, 24_070.0), 4.0, "#d97a5a"));
        self.tick = 0;
        self.sim_time = 0.0;
        self.source = label.to_string();
    }

    /// Velocity Verlet. Symplectic, so orbits stay closed over long runs
    /// instead of spiralling the way explicit Euler does.
    fn step(&mut self) {
        let n = self.bodies.len();
        if n == 0 {
            return;
        }
        let dt = self.dt;

        let a0 = self.accelerations();
        for (i, b) in self.bodies.iter_mut().enumerate() {
            b.x += b.vx * dt + 0.5 * a0[i].0 * dt * dt;
            b.y += b.vy * dt + 0.5 * a0[i].1 * dt * dt;
        }
        let a1 = self.accelerations();
        for (i, b) in self.bodies.iter_mut().enumerate() {
            b.vx += 0.5 * (a0[i].0 + a1[i].0) * dt;
            b.vy += 0.5 * (a0[i].1 + a1[i].1) * dt;
        }

        for b in self.bodies.iter_mut() {
            if b.trail.len() == TRAIL_MAX {
                b.trail.pop_front();
            }
            b.trail.push_back((b.x, b.y));
        }

        self.tick += 1;
        self.sim_time += dt;
    }

    fn accelerations(&self) -> Vec<(f64, f64)> {
        let n = self.bodies.len();
        let mut acc = vec![(0.0, 0.0); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.bodies[j].x - self.bodies[i].x;
                let dy = self.bodies[j].y - self.bodies[i].y;
                let r2 = dx * dx + dy * dy;
                if r2 == 0.0 {
                    continue;
                }
                let r = r2.sqrt();
                let inv = 1.0 / (r2 * r);
                acc[i].0 += G * self.bodies[j].m * dx * inv;
                acc[i].1 += G * self.bodies[j].m * dy * inv;
                acc[j].0 -= G * self.bodies[i].m * dx * inv;
                acc[j].1 -= G * self.bodies[i].m * dy * inv;
            }
        }
        acc
    }

    fn energy(&self) -> f64 {
        let mut e = 0.0;
        for b in &self.bodies {
            e += 0.5 * b.m * (b.vx * b.vx + b.vy * b.vy);
        }
        for i in 0..self.bodies.len() {
            for j in (i + 1)..self.bodies.len() {
                let dx = self.bodies[j].x - self.bodies[i].x;
                let dy = self.bodies[j].y - self.bodies[i].y;
                let r = (dx * dx + dy * dy).sqrt();
                if r > 0.0 {
                    e -= G * self.bodies[i].m * self.bodies[j].m / r;
                }
            }
        }
        e
    }

    /// Hand-rolled JSON. Positions go out already scaled to view units so
    /// the client does no unit arithmetic and cannot disagree with the server.
    fn snapshot_json(&self) -> String {
        let mut s = String::with_capacity(4096);
        s.push_str("{\"tick\":");
        s.push_str(&self.tick.to_string());
        s.push_str(",\"running\":");
        s.push_str(if self.running { "true" } else { "false" });
        s.push_str(",\"days\":");
        s.push_str(&format!("{:.2}", self.sim_time / 86_400.0));
        s.push_str(",\"energy\":");
        s.push_str(&format!("{:.6e}", self.energy()));
        s.push_str(",\"count\":");
        s.push_str(&self.bodies.len().to_string());
        s.push_str(",\"source\":\"");
        s.push_str(&escape(&self.source));
        s.push_str("\",\"bodies\":[");
        for (i, b) in self.bodies.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str("{\"name\":\"");
            s.push_str(&escape(&b.name));
            s.push_str("\",\"x\":");
            s.push_str(&format!("{:.3}", b.x * self.scale));
            s.push_str(",\"y\":");
            s.push_str(&format!("{:.3}", b.y * self.scale));
            s.push_str(",\"r\":");
            s.push_str(&format!("{:.2}", b.radius_px));
            s.push_str(",\"c\":\"");
            s.push_str(b.color);
            s.push_str("\",\"trail\":[");
            for (k, (tx, ty)) in b.trail.iter().enumerate() {
                if k > 0 {
                    s.push(',');
                }
                s.push_str(&format!("[{:.2},{:.2}]", tx * self.scale, ty * self.scale));
            }
            s.push_str("]}");
        }
        s.push_str("]}");
        s
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ---------------------------------------------------------------------------
// Commands — one implementation, shared by the REPL and the buttons
// ---------------------------------------------------------------------------

struct App {
    world: Mutex<World>,
    scene_open: AtomicBool,
    port: Mutex<u16>,
}

impl App {
    fn command(&self, raw: &str) -> String {
        let line = raw.trim();
        let upper = line.to_ascii_uppercase();
        let mut parts = upper.split_whitespace();
        let head = parts.next().unwrap_or("");
        let verb = parts.next().unwrap_or("");

        match (head, verb) {
            ("SCENE", "CREATE") => self.scene_create(),
            ("SCENE", "START") | ("START", _) => {
                // Failure mode 1 in the transcript: this used to error out.
                // A start request is a request to be running; if the scene
                // is missing, that is the server's problem to solve.
                let note = if self.scene_open.load(Ordering::SeqCst) {
                    String::new()
                } else {
                    format!("{}\n", self.scene_create())
                };
                let mut w = self.world.lock().unwrap();
                if w.bodies.is_empty() {
                    return format!("{}scene playback: running, but 0 entities — nothing is loaded. Run LOAD first.", note);
                }
                w.running = true;
                format!("{}scene playback: running", note)
            }
            ("SCENE", "PAUSE") | ("PAUSE", _) => {
                self.world.lock().unwrap().running = false;
                "scene playback: paused".into()
            }
            ("SCENE", "STEP") | ("STEP", _) => {
                let mut w = self.world.lock().unwrap();
                w.running = false;
                w.step();
                format!("stepped to tick {}", w.tick)
            }
            ("SCENE", "RESET") | ("RESET", _) => {
                let mut w = self.world.lock().unwrap();
                let src = w.source.clone();
                w.running = false;
                w.load_stage(&src);
                format!("reset to t=0; {} entities", w.bodies.len())
            }
            ("SCENE", "STATUS") | ("STATUS", _) => {
                let w = self.world.lock().unwrap();
                format!(
                    "scene {} | {} entities | tick {} | {:.1} days | {}",
                    if self.scene_open.load(Ordering::SeqCst) { "open" } else { "closed" },
                    w.bodies.len(),
                    w.tick,
                    w.sim_time / 86_400.0,
                    if w.running { "running" } else { "paused" }
                )
            }
            ("LOAD", _) => {
                let label = if verb.is_empty() { "Stage_2A.posim" } else { line.split_whitespace().nth(1).unwrap() };
                let mut w = self.world.lock().unwrap();
                w.load_stage(label);
                // The entities are now visible to any open scene, because
                // there is no second store for them to fail to reach.
                format!("{} loaded — {} entities", label, w.bodies.len())
            }
            ("HELP", _) => HELP.into(),
            ("", _) => String::new(),
            _ => format!("unknown command: {} (type HELP)", line),
        }
    }

    fn scene_create(&self) -> String {
        let port = *self.port.lock().unwrap();
        self.scene_open.store(true, Ordering::SeqCst);
        // Note what this does NOT do: it does not build an entity list.
        // The view is a window onto the world, never a copy of it.
        let n = self.world.lock().unwrap().bodies.len();
        format!(
            "scene window created: http://127.0.0.1:{}/\nshowing {} entities",
            port, n
        )
    }
}

const HELP: &str = "\
SCENE CREATE   open the scene window
SCENE START    begin evolution (creates the scene if needed)
SCENE PAUSE    halt evolution, keep state
SCENE STEP     advance exactly one timestep
SCENE RESET    reload the current stage at t=0
SCENE STATUS   print scene and world state
LOAD <file>    load a stage into the world
HELP           this list";

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

fn main() {
    let app = Arc::new(App {
        world: Mutex::new(World::new()),
        scene_open: AtomicBool::new(false),
        port: Mutex::new(0),
    });

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    *app.port.lock().unwrap() = port;

    // Physics runs on its own clock, independent of any client.
    // A browser that disconnects must not stall the integrator.
    {
        let app = Arc::clone(&app);
        thread::spawn(move || loop {
            {
                let mut w = app.world.lock().unwrap();
                if w.running {
                    for _ in 0..8 {
                        w.step();
                    }
                }
            }
            thread::sleep(Duration::from_millis(16));
        });
    }

    println!("{}", app.command("LOAD Stage_2A.posim"));
    println!("{}", app.command("SCENE CREATE"));
    println!("type HELP for commands, or use the buttons in the scene window");

    {
        let app = Arc::clone(&app);
        thread::spawn(move || {
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                match line {
                    Ok(l) => {
                        let out = app.command(&l);
                        if !out.is_empty() {
                            println!("{}", out);
                        }
                    }
                    Err(_) => break,
                }
            }
        });
    }

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let app = Arc::clone(&app);
                thread::spawn(move || handle(s, app));
            }
            Err(e) => eprintln!("accept: {}", e),
        }
    }
}

fn handle(mut stream: TcpStream, app: Arc<App>) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });

    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let mut it = request_line.split_whitespace();
    let method = it.next().unwrap_or("").to_string();
    let path = it.next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        match reader.read_line(&mut header) {
            Ok(0) => break,
            Ok(_) => {
                let h = header.trim_end();
                if h.is_empty() {
                    break;
                }
                let lower = h.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap_or(0);
                }
            }
            Err(_) => return,
        }
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/") => {
            send(&mut stream, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes());
        }
        ("GET", "/events") => stream_events(stream, app),
        ("POST", "/cmd") => {
            let mut buf = vec![0u8; content_length];
            if reader.read_exact(&mut buf).is_err() {
                return;
            }
            let cmd = String::from_utf8_lossy(&buf).to_string();
            let out = app.command(&cmd);
            println!("[scene] {} -> {}", cmd.trim(), out.replace('\n', " / "));
            send(&mut stream, "200 OK", "text/plain; charset=utf-8", out.as_bytes());
        }
        _ => send(&mut stream, "404 Not Found", "text/plain", b"not found"),
    }
}

fn send(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        status,
        ctype,
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

/// The piece the broken build was missing. Without an open stream the page
/// renders one frame and freezes, which looks exactly like a dead simulation.
fn stream_events(mut stream: TcpStream, app: Arc<App>) {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Cache-Control: no-store\r\n\
                Connection: keep-alive\r\n\r\n";
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }

    loop {
        let payload = {
            let w = app.world.lock().unwrap();
            w.snapshot_json()
        };
        let frame = format!("data: {}\n\n", payload);
        // A write error means the tab closed. Drop the thread; the world
        // keeps running for whoever else is watching.
        if stream.write_all(frame.as_bytes()).is_err() {
            return;
        }
        if stream.flush().is_err() {
            return;
        }
        thread::sleep(Duration::from_millis(33));
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

const PAGE: &str = r##"<!doctype html>
<meta charset="utf-8">
<title>orbit scene</title>
<style>
  :root {
    --void: #0a0e14;
    --panel: #121822;
    --rule: #253044;
    --ink: #c8d3e0;
    --dim: #64748b;
    --live: #7ee787;
    --held: #f0a868;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--void); color: var(--ink);
    font: 13px/1.5 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    display: grid; grid-template-rows: auto 1fr auto; height: 100vh;
  }
  header {
    display: flex; align-items: baseline; gap: 1rem;
    padding: .6rem 1rem; border-bottom: 1px solid var(--rule);
  }
  h1 { font-size: 13px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; margin: 0; }
  #src { color: var(--dim); }
  #state { margin-left: auto; letter-spacing: .1em; text-transform: uppercase; }
  #state.live { color: var(--live); }
  #state.held { color: var(--held); }
  main { position: relative; overflow: hidden; }
  canvas { display: block; width: 100%; height: 100%; }
  #empty {
    position: absolute; inset: 0; display: none;
    place-content: center; text-align: center; color: var(--dim);
  }
  #empty.show { display: grid; }
  footer {
    display: flex; gap: .5rem; align-items: center;
    padding: .6rem 1rem; border-top: 1px solid var(--rule); background: var(--panel);
  }
  button {
    font: inherit; letter-spacing: .08em; text-transform: uppercase;
    color: var(--ink); background: transparent;
    border: 1px solid var(--rule); border-radius: 2px;
    padding: .35rem .9rem; cursor: pointer;
  }
  button:hover { border-color: var(--ink); }
  button:focus-visible { outline: 2px solid var(--live); outline-offset: 2px; }
  .readout { margin-left: auto; display: flex; gap: 1.4rem; color: var(--dim); }
  .readout b { color: var(--ink); font-weight: 500; }
</style>

<header>
  <h1>Orbit scene</h1>
  <span id="src">—</span>
  <span id="state" class="held">connecting</span>
</header>

<main>
  <canvas id="c"></canvas>
  <div id="empty"><div>No entities in the world.<br>Run <b>LOAD Stage_2A.posim</b> to populate it.</div></div>
</main>

<footer>
  <button data-cmd="SCENE START">Start</button>
  <button data-cmd="SCENE PAUSE">Pause</button>
  <button data-cmd="SCENE STEP">Step</button>
  <button data-cmd="SCENE RESET">Reset</button>
  <div class="readout">
    <span>day <b id="days">0</b></span>
    <span>tick <b id="tick">0</b></span>
    <span>bodies <b id="count">0</b></span>
    <span>E <b id="energy">—</b></span>
  </div>
</footer>

<script>
const cv = document.getElementById('c');
const ctx = cv.getContext('2d');
let frame = null;

function resize() {
  const dpr = window.devicePixelRatio || 1;
  cv.width  = cv.clientWidth  * dpr;
  cv.height = cv.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
addEventListener('resize', resize);

// Every button routes through the same command strings the REPL accepts.
// If a command works when typed, the button works too — by construction.
for (const b of document.querySelectorAll('button[data-cmd]')) {
  b.addEventListener('click', async () => {
    try {
      const r = await fetch('/cmd', { method: 'POST', body: b.dataset.cmd });
      console.log(b.dataset.cmd, '->', await r.text());
    } catch (e) {
      console.error('control channel down:', e);
      setState('offline', 'held');
    }
  });
}

function setState(text, cls) {
  const el = document.getElementById('state');
  el.textContent = text;
  el.className = cls;
}

const es = new EventSource('/events');
es.onopen = () => setState('connected', 'held');
es.onerror = () => setState('stream lost', 'held');
es.onmessage = ev => {
  frame = JSON.parse(ev.data);
  document.getElementById('days').textContent   = frame.days;
  document.getElementById('tick').textContent   = frame.tick;
  document.getElementById('count').textContent  = frame.count;
  document.getElementById('energy').textContent = frame.energy;
  document.getElementById('src').textContent    = frame.source;
  document.getElementById('empty').classList.toggle('show', frame.count === 0);
  setState(frame.running ? 'running' : 'paused', frame.running ? 'live' : 'held');
  draw();
};

function draw() {
  const w = cv.clientWidth, h = cv.clientHeight;
  ctx.clearRect(0, 0, w, h);
  if (!frame || !frame.bodies.length) return;
  const cx = w / 2, cy = h / 2;

  for (const b of frame.bodies) {
    if (b.trail.length > 1) {
      ctx.beginPath();
      ctx.moveTo(cx + b.trail[0][0], cy + b.trail[0][1]);
      for (let i = 1; i < b.trail.length; i++) {
        ctx.lineTo(cx + b.trail[i][0], cy + b.trail[i][1]);
      }
      ctx.strokeStyle = b.c;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.beginPath();
    ctx.arc(cx + b.x, cy + b.y, b.r, 0, Math.PI * 2);
    ctx.fillStyle = b.c;
    ctx.fill();

    ctx.fillStyle = '#64748b';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(b.name, cx + b.x + b.r + 5, cy + b.y + 3);
  }
}

resize();
</script>
"##;
