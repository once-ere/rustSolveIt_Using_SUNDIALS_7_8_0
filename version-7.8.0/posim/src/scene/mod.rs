//! The graphical scene window subsystem.
//!
//! `SCENE CREATE` starts a tiny HTTP + WebSocket server (std-only, zero
//! external dependencies, zero `unsafe`) inside the posim process and
//! opens the scene page in the user's web browser. The page is a
//! self-contained HTML/canvas application (embedded at compile time via
//! `include_str!`) that renders every simulator entity, offers a
//! toolbar and a status bar, and maps the standard viewer gestures:
//! arrow keys translate, left-click drag rotates, the mouse wheel (or
//! `+`/`-`) zooms.
//!
//! Communication is asynchronous in both directions, PyBullet-style
//! (command -> status over a socket):
//! - notebook -> window: scene deltas, camera commands, playback state
//!   broadcast as JSON text frames;
//! - window -> notebook: errors, data requests and user actions queued
//!   as *events*, readable with `SCENE EVENTS` (and pushed as
//!   unsolicited `{"event": ...}` lines in `--machine` mode so the
//!   Jupyter kernel can stream them into the notebook).
//!
//! Time-stepped evolution runs on a playback thread that owns a
//! *synchronized copy* of the notebook's system (`SCENE REFRESH`
//! re-syncs it). All forward integration goes through
//! `physical_object::integrate` — never a hand-rolled stepper. Reverse
//! playback (`SCENE REVERSE`) replays a bounded ring buffer of
//! snapshots recorded while stepping forward (the Rapier
//! snapshot/restore idea).

pub mod ws;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::machine::{parse as json_parse, to_string as json_string, Json};
use ::physical_object::boundary::Boundary;
use ::physical_object::integrate::step as sundials_step;
use ::physical_object::PhysicalObjectSystem;

/// The embedded scene-window application (HTML + CSS + JS, one file).
const SCENE_HTML: &str = include_str!("scene.html");

/// Frames of forward history kept for reverse playback.
const HISTORY_CAP: usize = 20_000;

/// Playback tick period (~30 frames per second).
const TICK_MS: u64 = 33;

/// Playback state machine (Rapier `RunMode` analog).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RunMode {
    /// No evolution; history cleared. `SCENE START` begins fresh.
    Stopped,
    /// Evolving forward in time by `dt` per tick.
    Running,
    /// Frozen mid-run; `SCENE START` resumes, history kept.
    Paused,
    /// Replaying recorded history backward in time.
    Reversing,
}

impl RunMode {
    fn as_str(self) -> &'static str {
        match self {
            RunMode::Stopped => "stopped",
            RunMode::Running => "running",
            RunMode::Paused => "paused",
            RunMode::Reversing => "reversing",
        }
    }
}

/// Orbit camera: spherical coordinates around a target point, z-up.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Azimuth in degrees.
    pub yaw: f64,
    /// Elevation in degrees (90 = top-down onto the XY plane).
    pub pitch: f64,
    /// Distance from the target (> 0).
    pub dist: f64,
    /// Look-at point in world coordinates.
    pub target: [f64; 3],
}

impl Default for Camera {
    fn default() -> Self {
        Self { yaw: -60.0, pitch: 55.0, dist: 12.0, target: [0.0, 0.0, 0.0] }
    }
}

/// State shared between the VM, the playback thread and the socket
/// threads. Locked briefly; never held across blocking network I/O
/// (broadcasts go through non-blocking mpsc channels).
struct Shared {
    system: PhysicalObjectSystem,
    /// The playback copy's INITIAL state: what `SCENE CREATE` /
    /// `SCENE REFRESH` synced. `reset_playback` restores it exactly
    /// (every mutable value and the time), after which Start begins
    /// the simulation afresh.
    initial: PhysicalObjectSystem,
    mode: RunMode,
    dt: f64,
    hidden: BTreeSet<usize>,
    /// Inner side length of the rigid bounding box, when the VM created
    /// one with `BOX <size>` — the window draws its interior wireframe.
    box_size: Option<f64>,
    /// Entity indices of the six static wall slabs: drawn as the box
    /// wireframe instead of six giant cuboids.
    walls: BTreeSet<usize>,
    /// User names (index → name) for entity labels.
    names: BTreeMap<usize, String>,
    camera: Camera,
    history: VecDeque<PhysicalObjectSystem>,
    events: VecDeque<String>,
    outboxes: Vec<Sender<String>>,
    clients: usize,
    steps_done: u64,
    needs_init: bool,
    shutdown: bool,
    emit_async: bool,
}

impl Shared {
    /// Queues a message to every connected scene window.
    fn broadcast(&mut self, msg: &str) {
        self.outboxes.retain(|tx| tx.send(msg.to_string()).is_ok());
    }

    /// Records an asynchronous scene event for the notebook side and,
    /// in `--machine` mode, pushes it immediately as an unsolicited
    /// JSON line so the Jupyter kernel can stream it into the notebook.
    fn push_event(&mut self, text: &str) {
        if self.events.len() >= 1000 {
            self.events.pop_front();
        }
        self.events.push_back(text.to_string());
        if self.emit_async {
            let mut m = BTreeMap::new();
            m.insert("event".to_string(), Json::Str("scene".to_string()));
            m.insert("message".to_string(), Json::Str(text.to_string()));
            /* one full line per println! call: line-atomic on stdout */
            println!("{}", json_string(&Json::Obj(m)));
            let _ = std::io::stdout().flush();
        }
    }

    /// Advances the playback copy one step forward, recording history.
    fn step_forward(&mut self) {
        self.history.push_back(self.system.clone());
        if self.history.len() > HISTORY_CAP {
            self.history.pop_front();
        }
        let dt = self.dt;
        match sundials_step(&mut self.system, dt) {
            Ok(_) => self.steps_done += 1,
            Err(e) => {
                self.mode = RunMode::Paused;
                self.history.pop_back();
                self.push_event(&format!("error: solver failed at t = {}: {e}", self.system.time));
            }
        }
    }

    /// Re-initializes the playback: every mutable value returns to its
    /// initial value (the state last synced from the notebook) and the
    /// time returns to its initial value — a bit-identical clone of
    /// the `initial` snapshot. History and the step counter clear and
    /// the mode returns to Stopped; the Start button then re-starts
    /// the simulation from the beginning.
    fn reset_playback(&mut self) {
        self.system = self.initial.clone();
        self.system.contacts.clear(); /* defense in depth */
        self.history.clear();
        self.steps_done = 0;
        self.mode = RunMode::Stopped;
        self.needs_init = true;
    }

    /// Steps one recorded frame backward; pauses at the beginning.
    fn step_backward(&mut self) {
        match self.history.pop_back() {
            Some(prev) => self.system = prev,
            None => {
                self.mode = RunMode::Paused;
                self.push_event("reverse: reached the beginning of recorded history — paused");
            }
        }
    }
}

fn camera_json(c: &Camera) -> Json {
    let mut m = BTreeMap::new();
    m.insert("yaw".to_string(), Json::Num(c.yaw));
    m.insert("pitch".to_string(), Json::Num(c.pitch));
    m.insert("dist".to_string(), Json::Num(c.dist));
    m.insert(
        "target".to_string(),
        Json::Arr(c.target.iter().map(|x| Json::Num(*x)).collect()),
    );
    Json::Obj(m)
}

fn hidden_json(hidden: &BTreeSet<usize>) -> Json {
    Json::Arr(hidden.iter().map(|i| Json::Num(*i as f64)).collect())
}

/// Full scene description: entity geometry + camera + playback state.
/// Sent on connect, on `SCENE REFRESH`/`SCENE REDRAW`, and after
/// structural changes.
fn build_init(sh: &Shared) -> String {
    let mut ents = Vec::new();
    for (i, o) in sh.system.objects.iter().enumerate() {
        let mut m = BTreeMap::new();
        m.insert("i".to_string(), Json::Num(i as f64));
        m.insert("mass".to_string(), Json::Num(o.get_mass()));
        m.insert("charge".to_string(), Json::Num(o.get_charge()));
        match o.get_boundary() {
            Boundary::Point => {
                m.insert("shape".to_string(), Json::Str("point".to_string()));
            }
            Boundary::Sphere { radius } => {
                m.insert("shape".to_string(), Json::Str("sphere".to_string()));
                m.insert("radius".to_string(), Json::Num(radius));
            }
            Boundary::Cuboid { half_extents } => {
                m.insert("shape".to_string(), Json::Str("cuboid".to_string()));
                m.insert(
                    "half_extents".to_string(),
                    Json::Arr(half_extents.iter().map(|x| Json::Num(*x)).collect()),
                );
            }
            Boundary::Torus { ring_radius, tube_radius } => {
                m.insert("shape".to_string(), Json::Str("torus".to_string()));
                m.insert("ring_radius".to_string(), Json::Num(ring_radius));
                m.insert("tube_radius".to_string(), Json::Num(tube_radius));
            }
            Boundary::Disk { radius } => {
                m.insert("shape".to_string(), Json::Str("disk".to_string()));
                m.insert("radius".to_string(), Json::Num(radius));
            }
            Boundary::Cylinder { radius, half_height } => {
                m.insert("shape".to_string(), Json::Str("cylinder".to_string()));
                m.insert("radius".to_string(), Json::Num(radius));
                m.insert("half_height".to_string(), Json::Num(half_height));
            }
            Boundary::Dumbbell { r1, r2, rod_radius, z1, z2, .. } => {
                m.insert("shape".to_string(), Json::Str("dumbbell".to_string()));
                m.insert("r1".to_string(), Json::Num(r1));
                m.insert("r2".to_string(), Json::Num(r2));
                m.insert("rod_radius".to_string(), Json::Num(rod_radius));
                m.insert("z1".to_string(), Json::Num(z1));
                m.insert("z2".to_string(), Json::Num(z2));
            }
        }
        if sh.walls.contains(&i) {
            m.insert("wall".to_string(), Json::Bool(true));
        }
        if let Some(n) = sh.names.get(&i) {
            m.insert("name".to_string(), Json::Str(n.clone()));
        }
        ents.push(Json::Obj(m));
    }
    let mut m = BTreeMap::new();
    m.insert("type".to_string(), Json::Str("init".to_string()));
    m.insert("t".to_string(), Json::Num(sh.system.time));
    m.insert("dt".to_string(), Json::Num(sh.dt));
    m.insert("mode".to_string(), Json::Str(sh.mode.as_str().to_string()));
    m.insert("camera".to_string(), camera_json(&sh.camera));
    m.insert("hidden".to_string(), hidden_json(&sh.hidden));
    if let Some(size) = sh.box_size {
        m.insert("box".to_string(), Json::Num(size));
    }
    m.insert("entities".to_string(), Json::Arr(ents));
    json_string(&Json::Obj(m))
}

/// Per-tick pose update: positions + orientations of every entity.
fn build_frame(sh: &Shared) -> String {
    let mut bodies = Vec::new();
    for o in sh.system.objects.iter() {
        let p = o.get_position();
        let q = o.get_orientation();
        bodies.push(Json::Arr(vec![
            Json::Num(p.x),
            Json::Num(p.y),
            Json::Num(p.z),
            Json::Num(q.w),
            Json::Num(q.x),
            Json::Num(q.y),
            Json::Num(q.z),
        ]));
    }
    /* contacts recorded by the playback copy's most recent step: the
     * scene window draws each normal as an arrow at the contact point
     * (unit normal points from body i toward body j) */
    let mut contacts = Vec::new();
    for c in sh.system.contacts.iter() {
        let mut cm = BTreeMap::new();
        cm.insert("i".to_string(), Json::Num(c.i as f64));
        cm.insert("j".to_string(), Json::Num(c.j as f64));
        cm.insert(
            "point".to_string(),
            Json::Arr(vec![Json::Num(c.point.x), Json::Num(c.point.y), Json::Num(c.point.z)]),
        );
        cm.insert(
            "normal".to_string(),
            Json::Arr(vec![Json::Num(c.normal.x), Json::Num(c.normal.y), Json::Num(c.normal.z)]),
        );
        cm.insert("impulse".to_string(), Json::Num(c.impulse_n));
        contacts.push(Json::Obj(cm));
    }
    let mut m = BTreeMap::new();
    m.insert("type".to_string(), Json::Str("frame".to_string()));
    m.insert("t".to_string(), Json::Num(sh.system.time));
    m.insert("dt".to_string(), Json::Num(sh.dt));
    m.insert("mode".to_string(), Json::Str(sh.mode.as_str().to_string()));
    m.insert("steps".to_string(), Json::Num(sh.steps_done as f64));
    m.insert("history".to_string(), Json::Num(sh.history.len() as f64));
    m.insert("energy".to_string(), Json::Num(sh.system.total_energy()));
    let ptot = sh.system.total_momentum();
    let ltot = sh.system.total_angular_momentum(
        ::physical_object::linalg::Vec3::zeros(),
    );
    m.insert(
        "p".to_string(),
        Json::Arr(vec![Json::Num(ptot.x), Json::Num(ptot.y), Json::Num(ptot.z)]),
    );
    m.insert(
        "l".to_string(),
        Json::Arr(vec![Json::Num(ltot.x), Json::Num(ltot.y), Json::Num(ltot.z)]),
    );
    m.insert("hidden".to_string(), hidden_json(&sh.hidden));
    m.insert("contacts".to_string(), Json::Arr(contacts));
    m.insert("bodies".to_string(), Json::Arr(bodies));
    json_string(&Json::Obj(m))
}

fn build_camera_msg(sh: &Shared) -> String {
    let mut m = BTreeMap::new();
    m.insert("type".to_string(), Json::Str("camera".to_string()));
    m.insert("camera".to_string(), camera_json(&sh.camera));
    json_string(&Json::Obj(m))
}

/// Handle owned by the VM (`SimState.scene`). Dropping it shuts the
/// server down and disconnects every window.
pub struct SceneHandle {
    shared: Arc<Mutex<Shared>>,
    /// The scene window URL, e.g. `http://127.0.0.1:41234/`.
    pub url: String,
    /// Whether a browser-opening command was actually launched for this
    /// window. False when suppressed by `$POSIM_NO_BROWSER`, and false on
    /// any system whose platform opener -- `open` on macOS, `cmd /c start`
    /// on Windows, `xdg-open` on Linux/BSD -- is missing, where the spawn
    /// fails. Callers use it to avoid claiming a window was opened when
    /// none was.
    pub browser_launched: bool,
}

impl Drop for SceneHandle {
    fn drop(&mut self) {
        if let Ok(mut sh) = self.shared.lock() {
            sh.shutdown = true;
            sh.outboxes.clear();
        }
    }
}

impl SceneHandle {
    /// Starts the scene server: binds a local TCP port (an explicit
    /// `port`, or an OS-assigned one for 0), spawns the listener and
    /// playback threads, and (when `open_browser`) tries to open the
    /// page in the user's browser.
    pub fn start(
        system: PhysicalObjectSystem,
        port: u16,
        emit_async: bool,
        open_browser: bool,
    ) -> Result<SceneHandle, String> {
        let listener = TcpListener::bind(("127.0.0.1", port))
            .map_err(|e| format!("scene: cannot bind 127.0.0.1:{port}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| format!("scene: {e}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("scene: {e}"))?;
        let url = format!("http://{addr}/");

        let camera = Camera { dist: fit_distance(&system), ..Camera::default() };
        /* the playback copy starts with NO contact records: frames must
         * only ever carry contacts the playback itself resolved */
        let mut system = system;
        system.contacts.clear();
        let shared = Arc::new(Mutex::new(Shared {
            initial: system.clone(),
            system,
            mode: RunMode::Stopped,
            dt: 0.01,
            hidden: BTreeSet::new(),
            box_size: None,
            walls: BTreeSet::new(),
            names: BTreeMap::new(),
            camera,
            history: VecDeque::new(),
            events: VecDeque::new(),
            outboxes: Vec::new(),
            clients: 0,
            steps_done: 0,
            needs_init: false,
            shutdown: false,
            emit_async,
        }));

        /* listener thread: poll accept + shutdown flag */
        let sh_l = Arc::clone(&shared);
        thread::spawn(move || {
            loop {
                if sh_l.lock().map(|s| s.shutdown).unwrap_or(true) {
                    break;
                }
                match listener.accept() {
                    Ok((stream, _)) => {
                        let sh_c = Arc::clone(&sh_l);
                        thread::spawn(move || handle_connection(stream, sh_c));
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(_) => break,
                }
            }
        });

        /* playback thread: evolve + broadcast at ~30 fps */
        let sh_p = Arc::clone(&shared);
        thread::spawn(move || {
            loop {
                {
                    let mut sh = match sh_p.lock() {
                        Ok(g) => g,
                        Err(_) => break,
                    };
                    if sh.shutdown {
                        sh.outboxes.clear();
                        break;
                    }
                    match sh.mode {
                        RunMode::Running => sh.step_forward(),
                        RunMode::Reversing => sh.step_backward(),
                        RunMode::Stopped | RunMode::Paused => {}
                    }
                    if sh.needs_init {
                        sh.needs_init = false;
                        let init = build_init(&sh);
                        sh.broadcast(&init);
                    }
                    if !sh.outboxes.is_empty() {
                        let frame = build_frame(&sh);
                        sh.broadcast(&frame);
                    }
                }
                thread::sleep(Duration::from_millis(TICK_MS));
            }
        });

        /* Best-effort: open the user's browser on the scene page
         * (suppressed when $POSIM_NO_BROWSER is set, e.g. in tests).
         *
         * Whether we actually tried is REPORTED rather than assumed. Each
         * platform has its own opener and none of them is universal: if the
         * chosen binary is absent the spawn fails, and the old caller
         * announced "opened in your browser" regardless. A reader on such a
         * system then waits for a window that was never going to appear.
         * spawn() failing is exactly the signal needed to say so, so it is
         * kept instead of discarded. */
        let browser_launched = if open_browser && std::env::var_os("POSIM_NO_BROWSER").is_none() {
            /* `start` is a cmd builtin rather than an executable, and its
             * first quoted argument is the window title -- omit the empty
             * title and it would swallow the URL. */
            let (program, args): (&str, &[&str]) = if cfg!(target_os = "macos") {
                ("open", &[])
            } else if cfg!(target_os = "windows") {
                ("cmd", &["/C", "start", ""])
            } else {
                ("xdg-open", &[])
            };
            std::process::Command::new(program)
                .args(args)
                .arg(&url)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .is_ok()
        } else {
            false
        };

        Ok(SceneHandle {
            shared,
            url,
            browser_launched,
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Shared>, String> {
        self.shared
            .lock()
            .map_err(|_| "scene: internal lock poisoned".to_string())
    }

    /// Replaces the playback copy with the notebook's current system,
    /// clears history, and redraws every window.
    pub fn sync(&self, system: &PhysicalObjectSystem) -> Result<(), String> {
        let mut sh = self.lock()?;
        sh.system = system.clone();
        sh.system.contacts.clear(); /* only playback-resolved contacts */
        /* the synced state becomes the new "initial values" that the
         * Reset button / SCENE RESET restore */
        sh.initial = sh.system.clone();
        sh.history.clear();
        sh.steps_done = 0;
        sh.needs_init = true;
        Ok(())
    }

    /// Re-initializes the playback to its initial state (see
    /// `Shared::reset_playback`) — the `SCENE RESET` command and the
    /// window's Reset button both land here.
    pub fn reset_playback(&self) -> Result<String, String> {
        let mut sh = self.lock()?;
        sh.reset_playback();
        Ok(format!(
            "scene playback reset to its initial state (t = {}, {} entit{}); \
             Start runs the simulation again from the beginning",
            sh.system.time,
            sh.system.objects.len(),
            if sh.system.objects.len() == 1 { "y" } else { "ies" }
        ))
    }

    /// Forces a full scene re-description on every window.
    pub fn redraw(&self) -> Result<(), String> {
        let mut sh = self.lock()?;
        sh.needs_init = true;
        Ok(())
    }

    /// Tells the window about the rigid bounding box (inner size, wall
    /// slab indices — drawn as the interior wireframe, not as giant
    /// cuboids) and the user-name registry for entity labels.
    pub fn set_box(
        &self,
        box_size: Option<f64>,
        walls: &[usize],
        names: &std::collections::BTreeMap<String, usize>,
    ) -> Result<(), String> {
        let mut sh = self.lock()?;
        sh.box_size = box_size;
        sh.walls = walls.iter().copied().collect();
        sh.names = names.iter().map(|(n, i)| (*i, n.clone())).collect();
        sh.needs_init = true;
        Ok(())
    }

    /// Moves the camera target (world units).
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> Result<String, String> {
        let mut sh = self.lock()?;
        sh.camera.target[0] += dx;
        sh.camera.target[1] += dy;
        sh.camera.target[2] += dz;
        let msg = build_camera_msg(&sh);
        sh.broadcast(&msg);
        let t = sh.camera.target;
        Ok(format!("camera target = [{}, {}, {}]", t[0], t[1], t[2]))
    }

    /// Orbits the camera: yaw (azimuth) and pitch (elevation), degrees.
    pub fn rotate(&self, dyaw: f64, dpitch: f64) -> Result<String, String> {
        let mut sh = self.lock()?;
        sh.camera.yaw = (sh.camera.yaw + dyaw) % 360.0;
        sh.camera.pitch = (sh.camera.pitch + dpitch).clamp(-89.0, 89.0);
        let msg = build_camera_msg(&sh);
        sh.broadcast(&msg);
        Ok(format!("camera yaw = {}°, pitch = {}°", sh.camera.yaw, sh.camera.pitch))
    }

    /// Zooms by a factor > 0 (factor > 1 zooms in, < 1 zooms out).
    pub fn zoom(&self, factor: f64) -> Result<String, String> {
        if !(factor.is_finite() && factor > 0.0) {
            return Err("scene: zoom factor must be a positive number".to_string());
        }
        let mut sh = self.lock()?;
        sh.camera.dist = (sh.camera.dist / factor).clamp(1e-3, 1e9);
        let msg = build_camera_msg(&sh);
        sh.broadcast(&msg);
        Ok(format!("camera distance = {}", sh.camera.dist))
    }

    /// Hides (`hide = true`) or shows one entity, or all of them.
    pub fn set_visibility(&self, which: Option<usize>, hide: bool) -> Result<String, String> {
        let mut sh = self.lock()?;
        let n = sh.system.objects.len();
        match which {
            Some(i) => {
                if i >= n {
                    return Err(format!("no object obj{i}"));
                }
                if hide {
                    sh.hidden.insert(i);
                } else {
                    sh.hidden.remove(&i);
                }
            }
            None => {
                if hide {
                    sh.hidden = (0..n).collect();
                } else {
                    sh.hidden.clear();
                }
            }
        }
        let count = sh.hidden.len();
        sh.needs_init = true;
        Ok(format!("{} object(s) hidden", count))
    }

    /// Sets the playback mode (start / stop / pause / reverse).
    pub fn set_mode(&self, mode: RunMode) -> Result<String, String> {
        let mut sh = self.lock()?;
        if mode == RunMode::Stopped {
            sh.history.clear();
        }
        if mode == RunMode::Reversing && sh.history.is_empty() {
            return Err(
                "scene: nothing to reverse — no forward history recorded yet (SCENE START first)"
                    .to_string(),
            );
        }
        sh.mode = mode;
        Ok(format!("scene playback: {}", mode.as_str()))
    }

    /// Sets the playback time step (must be positive and finite).
    pub fn set_dt(&self, dt: f64) -> Result<String, String> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err("scene: set_time_step needs a positive, finite dt".to_string());
        }
        let mut sh = self.lock()?;
        sh.dt = dt;
        Ok(format!("scene time step dt = {dt}"))
    }

    /// Human-readable status line for `SCENE STATUS`.
    pub fn status(&self) -> Result<String, String> {
        let sh = self.lock()?;
        let c = sh.camera;
        let hidden = if sh.hidden.is_empty() {
            "none".to_string()
        } else {
            sh.hidden.iter().map(|i| format!("obj{i}")).collect::<Vec<_>>().join(", ")
        };
        Ok(format!(
            "scene: {}  ({} window(s) connected)\n\
             mode = {}, t = {}, dt = {}, steps = {}, history = {} frame(s)\n\
             entities = {} (hidden: {hidden})\n\
             camera: yaw = {}°, pitch = {}°, dist = {}, target = [{}, {}, {}]",
            self.url,
            sh.clients,
            sh.mode.as_str(),
            sh.system.time,
            sh.dt,
            sh.steps_done,
            sh.history.len(),
            sh.system.objects.len(),
            c.yaw,
            c.pitch,
            c.dist,
            c.target[0],
            c.target[1],
            c.target[2],
        ))
    }

    /// Drains queued window->notebook events (errors, requests, user
    /// actions), oldest first.
    pub fn drain_events(&self) -> Result<Vec<String>, String> {
        let mut sh = self.lock()?;
        Ok(sh.events.drain(..).collect())
    }

    /// Number of connected scene windows (used in tests).
    #[cfg(test)]
    pub fn client_count(&self) -> usize {
        self.lock().map(|sh| sh.clients).unwrap_or(0)
    }

    /// Current camera (used in tests).
    #[cfg(test)]
    pub fn camera(&self) -> Camera {
        self.lock().map(|sh| sh.camera).unwrap_or_default()
    }

    /// Current playback mode (used in tests).
    #[cfg(test)]
    pub fn mode(&self) -> RunMode {
        self.lock().map(|sh| sh.mode).unwrap_or(RunMode::Stopped)
    }

    /// Forward steps taken by the playback copy (used in tests).
    #[cfg(test)]
    pub fn steps_done(&self) -> u64 {
        self.lock().map(|sh| sh.steps_done).unwrap_or(0)
    }

    /// The playback copy's simulation time (used in tests).
    #[cfg(test)]
    pub fn system_time(&self) -> f64 {
        self.lock().map(|sh| sh.system.time).unwrap_or(f64::NAN)
    }

    /// The playback copy's packed 13N state (used in tests).
    #[cfg(test)]
    pub fn packed_state(&self) -> Vec<f64> {
        self.lock().map(|sh| sh.system.pack_state()).unwrap_or_default()
    }

    /// Contact records currently on the playback copy (used in tests).
    #[cfg(test)]
    pub fn contacts_len(&self) -> usize {
        self.lock().map(|sh| sh.system.contacts.len()).unwrap_or(0)
    }
}

/// Initial camera distance that frames the whole scene (Rapier's
/// `FRAME_SCENE` idea): 2.5x the farthest entity, at least 12.
fn fit_distance(system: &PhysicalObjectSystem) -> f64 {
    let mut r: f64 = 0.0;
    for o in system.objects.iter() {
        let p = o.get_position();
        r = r.max((p.x * p.x + p.y * p.y + p.z * p.z).sqrt());
    }
    (2.5 * r).max(12.0)
}

/// One HTTP connection: either serves the page or upgrades to a
/// WebSocket and becomes a scene-window session.
fn handle_connection(mut stream: TcpStream, shared: Arc<Mutex<Shared>>) {
    /* read the request head (bounded) */
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => return,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 16 * 1024 {
                    break;
                }
            }
            /* EINTR is not a failure: the read was interrupted by a signal
             * before any byte moved, and retrying is the correct response.
             * Lumping it in with the fatal arm closed the connection with no
             * reply at all, which the peer sees as an EOF in mid-handshake. */
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return,
        }
    }
    let head = String::from_utf8_lossy(&buf).to_string();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or_default();
    let path = request_line.split_whitespace().nth(1).unwrap_or("/");
    let ws_key = lines
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("sec-websocket-key"))
        .map(|(_, v)| v.trim().to_string());

    match (path, ws_key) {
        ("/ws", Some(key)) => {
            let accept = ws::accept_key(&key);
            let response = format!(
                "HTTP/1.1 101 Switching Protocols\r\n\
                 Upgrade: websocket\r\n\
                 Connection: Upgrade\r\n\
                 Sec-WebSocket-Accept: {accept}\r\n\r\n"
            );
            if stream.write_all(response.as_bytes()).is_err() {
                return;
            }
            serve_ws_client(stream, shared);
        }
        ("/" | "/index.html", _) => {
            let body = SCENE_HTML.as_bytes();
            let response = format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: text/html; charset=utf-8\r\n\
                 Cache-Control: no-store\r\n\
                 Content-Length: {}\r\n\
                 Connection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.write_all(body);
        }
        _ => {
            let _ = stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found",
            );
        }
    }
}

/// A connected scene window: registers an outbox for broadcasts, sends
/// the full scene description, then routes incoming messages until the
/// window disconnects or the scene shuts down.
fn serve_ws_client(mut stream: TcpStream, shared: Arc<Mutex<Shared>>) {
    let (tx, rx) = channel::<String>();
    {
        let mut sh = match shared.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if sh.shutdown {
            return;
        }
        sh.outboxes.push(tx.clone());
        sh.clients += 1;
        let init = build_init(&sh);
        let _ = tx.send(init);
        let n = sh.clients;
        sh.push_event(&format!("window connected ({n} total)"));
    }

    /* writer thread: outbox -> socket; the write half is mutex-shared
     * so the reader can send pong/close without interleaving frames */
    let write_half = match stream.try_clone() {
        Ok(s) => Arc::new(Mutex::new(s)),
        Err(_) => return,
    };
    let write_for_writer = Arc::clone(&write_half);
    let writer = thread::spawn(move || {
        while let Ok(msg) = rx.recv() {
            let mut s = match write_for_writer.lock() {
                Ok(g) => g,
                Err(_) => break,
            };
            if ws::write_text(&mut s, &msg).is_err() {
                break;
            }
        }
        if let Ok(mut s) = write_for_writer.lock() {
            let _ = ws::write_close(&mut s);
        }
    });

    /* reader loop: poll with a timeout so shutdown is honored */
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    loop {
        if shared.lock().map(|s| s.shutdown).unwrap_or(true) {
            break;
        }
        match ws::read_frame(&mut stream) {
            Ok(None) => continue,
            Ok(Some(ws::WsMessage::Text(text))) => {
                if let Ok(mut sh) = shared.lock() {
                    handle_client_message(&mut sh, &text, &tx);
                }
            }
            Ok(Some(ws::WsMessage::Ping(payload))) => {
                if let Ok(mut s) = write_half.lock() {
                    let _ = ws::write_pong(&mut s, &payload);
                }
            }
            Ok(Some(ws::WsMessage::Pong)) => {}
            Ok(Some(ws::WsMessage::Close)) | Err(_) => break,
        }
    }
    drop(tx);
    if let Ok(mut sh) = shared.lock() {
        sh.clients = sh.clients.saturating_sub(1);
        let n = sh.clients;
        sh.push_event(&format!("window disconnected ({n} remain)"));
    }
    let _ = writer.join();
}

/// Routes one window -> simulator JSON message.
fn handle_client_message(sh: &mut Shared, text: &str, own_tx: &Sender<String>) {
    let msg = match json_parse(text) {
        Ok(j) => j,
        Err(e) => {
            sh.push_event(&format!("error: bad message from window: {e}"));
            return;
        }
    };
    let kind = msg.get("type").and_then(|j| j.as_str()).unwrap_or("");
    let num = |key: &str| -> Option<f64> {
        match msg.get(key) {
            Some(Json::Num(n)) => Some(*n),
            _ => None,
        }
    };
    match kind {
        /* window camera moved (drag / keys / wheel): keep server copy
         * in sync so SCENE STATUS reflects what the user sees */
        "camera" => {
            if let Some(v) = num("yaw") {
                sh.camera.yaw = v;
            }
            if let Some(v) = num("pitch") {
                sh.camera.pitch = v.clamp(-89.0, 89.0);
            }
            if let Some(v) = num("dist") {
                if v.is_finite() && v > 0.0 {
                    sh.camera.dist = v;
                }
            }
            if let Some(Json::Arr(t)) = msg.get("target") {
                for (i, item) in t.iter().take(3).enumerate() {
                    if let Json::Num(x) = item {
                        sh.camera.target[i] = *x;
                    }
                }
            }
        }
        /* toolbar commands */
        "cmd" => {
            let action = msg.get("action").and_then(|j| j.as_str()).unwrap_or("");
            match action {
                "start" => sh.mode = RunMode::Running,
                "pause" => sh.mode = RunMode::Paused,
                "stop" => {
                    sh.mode = RunMode::Stopped;
                    sh.history.clear();
                }
                "reverse" => {
                    if sh.history.is_empty() {
                        sh.push_event("reverse requested with no history — ignored");
                    } else {
                        sh.mode = RunMode::Reversing;
                    }
                }
                "step" => sh.step_forward(),
                "step_back" => sh.step_backward(),
                "reset" => sh.reset_playback(),
                "set_dt" => {
                    if let Some(v) = num("value") {
                        if v.is_finite() && v > 0.0 {
                            sh.dt = v;
                        } else {
                            sh.push_event("error: window sent a non-positive dt — ignored");
                        }
                    }
                }
                "refresh" => sh.needs_init = true,
                other => sh.push_event(&format!("error: unknown window command `{other}`")),
            }
            sh.push_event(&format!("window action: {action}"));
        }
        /* the window asks for a full scene description */
        "request_state" => {
            let init = build_init(sh);
            let _ = own_tx.send(init);
            sh.push_event("window requested a state refresh");
        }
        /* free-form window -> notebook event (errors, notes) */
        "event" => {
            let level = msg.get("level").and_then(|j| j.as_str()).unwrap_or("info");
            let m = msg.get("message").and_then(|j| j.as_str()).unwrap_or("");
            sh.push_event(&format!("{level}: {m}"));
        }
        other => sh.push_event(&format!("error: unknown message type `{other}` from window")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::physical_object::linalg::Vec3;
    use ::physical_object::physical_object::physical_object;

    /// Minimal WebSocket *client* for tests (the server tolerates
    /// unmasked client frames, so no masking is needed here).
    fn ws_send_text(stream: &mut TcpStream, text: &str) {
        let mut frame = vec![0x81u8];
        assert!(text.len() < 126, "test frames stay short");
        frame.push(text.len() as u8);
        frame.extend_from_slice(text.as_bytes());
        stream.write_all(&frame).unwrap();
        stream.flush().unwrap();
    }

    /// Reads one server frame (unmasked text) and returns its payload.
    fn ws_read_text(stream: &mut TcpStream) -> String {
        let mut head = [0u8; 2];
        stream.read_exact(&mut head).unwrap();
        assert_eq!(head[0] & 0x0F, 0x1, "expected a text frame");
        let mut len = (head[1] & 0x7F) as u64;
        if len == 126 {
            let mut ext = [0u8; 2];
            stream.read_exact(&mut ext).unwrap();
            len = u16::from_be_bytes(ext) as u64;
        } else if len == 127 {
            let mut ext = [0u8; 8];
            stream.read_exact(&mut ext).unwrap();
            len = u64::from_be_bytes(ext);
        }
        let mut payload = vec![0u8; len as usize];
        stream.read_exact(&mut payload).unwrap();
        String::from_utf8(payload).unwrap()
    }

    /// Polls a condition for up to ~3 s (server threads tick at 33 ms).
    fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
        for _ in 0..120 {
            if cond() {
                return true;
            }
            thread::sleep(Duration::from_millis(25));
        }
        false
    }

    fn test_system() -> PhysicalObjectSystem {
        let mut sys = PhysicalObjectSystem::new(Vec::new(), 1.0);
        sys.add_object(physical_object::new_point(
            0,
            2.0,
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ));
        sys
    }

    fn port_of(url: &str) -> u16 {
        url.trim_start_matches("http://127.0.0.1:")
            .trim_end_matches('/')
            .parse()
            .unwrap()
    }

    /// Connect to the scene server and complete the RFC 6455 handshake,
    /// returning the socket and the response head.
    ///
    /// Retried, deliberately. `handle_connection` gives a connection five
    /// seconds to deliver its request head and closes it otherwise. Under a
    /// full `cargo test --workspace` this thread can be descheduled past that
    /// budget between `connect()` and `write_all()`, whereupon the server
    /// closes with no reply and the handshake read here fails with
    /// `UnexpectedEof`. Nothing is wrong with the server in that case -- the
    /// test simply lost a scheduling race -- so the race is retried rather
    /// than asserted away.
    fn ws_handshake(url: &str) -> (TcpStream, String) {
        let mut last = String::new();
        for _ in 0..5 {
            match try_ws_handshake(url) {
                Ok(pair) => return pair,
                Err(e) => last = e,
            }
        }
        panic!("websocket handshake never completed after 5 attempts: {last}");
    }

    fn try_ws_handshake(url: &str) -> Result<(TcpStream, String), String> {
        let mut stream =
            TcpStream::connect(("127.0.0.1", port_of(url))).map_err(|e| e.to_string())?;
        stream
            .write_all(
                b"GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n\
                  Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
                  Sec-WebSocket-Version: 13\r\n\r\n",
            )
            .map_err(|e| e.to_string())?;
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.windows(4).any(|w| w == b"\r\n\r\n") {
            stream.read_exact(&mut byte).map_err(|e| e.to_string())?;
            head.push(byte[0]);
        }
        Ok((stream, String::from_utf8_lossy(&head).to_string()))
    }

    #[test]
    fn serves_the_scene_page_over_http() {
        let handle = SceneHandle::start(test_system(), 0, false, false).unwrap();
        let mut stream = TcpStream::connect(("127.0.0.1", port_of(&handle.url))).unwrap();
        stream
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut body = String::new();
        stream.read_to_string(&mut body).unwrap();
        assert!(body.starts_with("HTTP/1.1 200 OK"), "{}", &body[..60.min(body.len())]);
        assert!(body.contains("posim scene"), "page body missing");
        assert!(body.contains("id=\"toolbar\""), "toolbar missing");
        assert!(body.contains("id=\"statusbar\""), "statusbar missing");
        /* collision UI: normal-arrow toggle + statusbar counter */
        assert!(body.contains("id=\"bt-contacts\""), "Contacts toggle missing");
        assert!(body.contains("id=\"bt-reset\""), "permanent Reset button missing");
        assert!(body.contains("id=\"hud\""), "conserved-quantities HUD missing");
        assert!(body.contains("conserved quantities"), "HUD label missing");
        assert!(body.contains("id=\"st-contacts\""), "contacts statusbar missing");
        /* the three required gestures are wired in the page script */
        assert!(body.contains("ArrowLeft") && body.contains("ArrowRight"), "arrow keys");
        assert!(body.contains("mousedown") && body.contains("mousemove"), "drag rotate");
        assert!(body.contains("wheel"), "wheel zoom");
    }

    #[test]
    fn websocket_session_end_to_end() {
        let handle = SceneHandle::start(test_system(), 0, false, false).unwrap();
        /* handshake reply carries the RFC 6455 accept key */
        let (mut stream, head) = ws_handshake(&handle.url);
        assert!(head.starts_with("HTTP/1.1 101"), "{head}");
        assert!(head.contains("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="), "{head}");

        /* first message is the full scene description */
        let init = ws_read_text(&mut stream);
        assert!(init.contains("\"type\":\"init\""), "{init}");
        assert!(init.contains("\"entities\":["), "{init}");
        assert!(wait_for(|| handle.client_count() == 1));

        /* what the arrow keys / left-drag / wheel produce: a camera
         * sync message — the server copy must follow it */
        ws_send_text(
            &mut stream,
            r#"{"type":"camera","yaw":33.0,"pitch":44.0,"dist":7.5,"target":[1.0,2.0,3.0]}"#,
        );
        assert!(wait_for(|| {
            let c = handle.camera();
            c.yaw == 33.0 && c.pitch == 44.0 && c.dist == 7.5 && c.target == [1.0, 2.0, 3.0]
        }));

        /* toolbar: start / pause the time-stepped evolution */
        ws_send_text(&mut stream, r#"{"type":"cmd","action":"start"}"#);
        assert!(wait_for(|| handle.mode() == RunMode::Running));
        ws_send_text(&mut stream, r#"{"type":"cmd","action":"pause"}"#);
        assert!(wait_for(|| handle.mode() == RunMode::Paused));

        /* window -> notebook async event path */
        ws_send_text(
            &mut stream,
            r#"{"type":"event","level":"error","message":"canvas exploded"}"#,
        );
        assert!(wait_for(|| {
            handle
                .drain_events()
                .unwrap()
                .iter()
                .any(|e| e.contains("error: canvas exploded"))
        }));

        /* frames keep flowing while connected */
        let mut saw_frame = false;
        for _ in 0..40 {
            let msg = ws_read_text(&mut stream);
            if msg.contains("\"type\":\"frame\"") && msg.contains("\"bodies\":[") {
                assert!(msg.contains("\"contacts\":["), "frame carries the contacts array");
                assert!(msg.contains("\"p\":["), "frame carries total momentum");
                assert!(msg.contains("\"l\":["), "frame carries total angular momentum");
                saw_frame = true;
                break;
            }
        }
        assert!(saw_frame, "no frame broadcast received");
    }

    #[test]
    fn box_shapes_and_wall_flags_reach_the_init_message() {
        /* a torus body plus one "wall" cuboid; the VM-side BOX info is
         * pushed via set_box and must appear in the init message */
        let mut sys = PhysicalObjectSystem::new(Vec::new(), 1.0);
        let mut wall = ::physical_object::physical_object::physical_object::new_from_shape(
            0,
            1.0,
            0.0,
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::zeros(),
            Vec3::zeros(),
            ::physical_object::boundary::Boundary::Cuboid { half_extents: [1.0, 4.0, 4.0] },
        );
        wall.set_inverse_mass(0.0);
        sys.add_object(wall);
        sys.add_object(::physical_object::physical_object::physical_object::new_from_shape(
            1,
            1.0,
            0.0,
            Vec3::zeros(),
            Vec3::zeros(),
            Vec3::zeros(),
            ::physical_object::boundary::Boundary::Torus { ring_radius: 1.5, tube_radius: 0.5 },
        ));
        let handle = SceneHandle::start(sys, 0, false, false).unwrap();
        handle
            .set_box(
                Some(4.0),
                &[0],
                &std::collections::BTreeMap::from([("ringo".to_string(), 1usize)]),
            )
            .unwrap();

        let (mut stream, _head) = ws_handshake(&handle.url);
        /* read messages until an init carrying the box arrives (the
         * pre-set_box init may race ahead of the flagged one) */
        let mut seen = String::new();
        for _ in 0..40 {
            let msg = ws_read_text(&mut stream);
            if msg.contains("\"type\":\"init\"") && msg.contains("\"box\":4.0") {
                seen = msg;
                break;
            }
        }
        assert!(!seen.is_empty(), "an init with the box arrived");
        assert!(seen.contains("\"shape\":\"torus\""), "{seen}");
        assert!(seen.contains("\"ring_radius\":1.5"), "{seen}");
        assert!(seen.contains("\"tube_radius\":0.5"), "{seen}");
        assert!(seen.contains("\"wall\":true"), "{seen}");
        assert!(seen.contains("\"name\":\"ringo\""), "{seen}");
    }

    #[test]
    fn stale_notebook_contacts_never_reach_the_playback_copy() {
        // A system arriving with contact records from a notebook
        // STEP/RUN must not re-flash arrows in the window: the
        // playback copy only ever carries contacts IT resolved.
        let mut sys = test_system();
        sys.contacts.push(::physical_object::collide::Contact {
            i: 0,
            j: 0,
            t: 1.0,
            point: Vec3::zeros(),
            normal: Vec3::new(1.0, 0.0, 0.0),
            depth: 0.0,
            rel_vel_n: -1.0,
            impulse_n: 2.0,
        });
        let handle = SceneHandle::start(sys.clone(), 0, false, false).unwrap();
        assert_eq!(handle.contacts_len(), 0, "cleared at CREATE");
        handle.sync(&sys).unwrap();
        assert_eq!(handle.contacts_len(), 0, "cleared at REFRESH sync");
        handle.reset_playback().unwrap();
        assert_eq!(handle.contacts_len(), 0, "cleared at RESET");
    }

    #[test]
    fn reset_restores_the_initial_state_and_start_reruns() {
        let sys = test_system();
        let initial_pack = sys.pack_state();
        let handle = SceneHandle::start(sys, 0, false, false).unwrap();
        handle.set_dt(0.05).unwrap();
        handle.set_mode(RunMode::Running).unwrap();
        assert!(wait_for(|| handle.steps_done() >= 10), "playback advanced");
        handle.set_mode(RunMode::Paused).unwrap();
        assert!(handle.system_time() > 0.0, "time moved");

        /* Reset: every mutable value and the time return to their
         * initial values — bit-identically. */
        let msg = handle.reset_playback().unwrap();
        assert!(msg.contains("initial state"), "{msg}");
        assert_eq!(handle.mode(), RunMode::Stopped);
        assert_eq!(handle.system_time(), 0.0, "time back to its initial value");
        assert_eq!(handle.packed_state(), initial_pack, "bit-identical initial state");
        assert_eq!(handle.steps_done(), 0);

        /* Start re-starts the simulation after a reset. */
        handle.set_mode(RunMode::Running).unwrap();
        assert!(wait_for(|| handle.steps_done() >= 5), "re-started after reset");
        assert!(handle.system_time() > 0.0);
    }

    #[test]
    fn playback_forward_then_reverse_restores_state() {
        let handle = SceneHandle::start(test_system(), 0, false, false).unwrap();
        handle.set_dt(0.05).unwrap();
        handle.set_mode(RunMode::Running).unwrap();
        assert!(wait_for(|| handle.lock().unwrap().history.len() >= 10));
        handle.set_mode(RunMode::Paused).unwrap();
        let t_mid = handle.lock().unwrap().system.time;
        assert!(t_mid > 0.0);
        handle.set_mode(RunMode::Reversing).unwrap();
        /* reverse consumes all history, then auto-pauses at t = 0 */
        assert!(wait_for(|| {
            let sh = handle.lock().unwrap();
            sh.history.is_empty() && sh.mode == RunMode::Paused
        }));
        let sh = handle.lock().unwrap();
        assert_eq!(sh.system.time, 0.0, "reverse must land back on the start state");
        drop(sh);
        assert!(handle
            .drain_events()
            .unwrap()
            .iter()
            .any(|e| e.contains("beginning of recorded history")));
    }
}
