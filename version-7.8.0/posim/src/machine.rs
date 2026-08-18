//! Machine mode (`posim --machine`): a line-delimited JSON protocol
//! over stdin/stdout for front ends such as the JupyterLab wrapper
//! kernel in `jupyter/`. One JSON document per line, flushed per reply.
//!
//! Requests:
//!   {"op":"exec","code":"<command line>"}
//!   {"op":"get","path":"obj0.position"}
//!   {"op":"set","path":"obj0.mass","value":2.5}
//!   {"op":"state"}
//!   {"op":"events"}        drain queued scene-window events
//!   {"op":"help"}
//!   {"op":"quit"}
//! Replies:
//!   {"ok":true,"result":<json>,"display":"<human text>"}
//!   {"ok":false,"error":"<message>"}
//! Asynchronous lines (only while a scene window is open — the window
//! pushes errors / data requests / user actions at any time, not in
//! reply to a request; front ends must route them separately):
//!   {"event":"scene","message":"<text>"}
//!
//! The JSON reader/writer below is a minimal hand-rolled subset
//! (objects, arrays, strings, numbers, booleans, null) — no external
//! dependencies.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};

use crate::vm::{execute_line, SimState, Value};
use ::physical_object::boundary::Boundary;

/// Minimal JSON document.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn get<'a>(&'a self, key: &str) -> Option<&'a Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Serializes a JSON document to a single line.
pub fn to_string(j: &Json) -> String {
    let mut s = String::new();
    write_json(j, &mut s);
    s
}

fn write_json(j: &Json, out: &mut String) {
    match j {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            if n.is_finite() {
                /* {:?} is Rust's shortest round-trip float formatting */
                out.push_str(&format!("{n:?}"));
            } else {
                out.push_str("null");
            }
        }
        Json::Str(s) => write_json_string(s, out),
        Json::Arr(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(it, out);
            }
            out.push(']');
        }
        Json::Obj(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(k, out);
                out.push(':');
                write_json(v, out);
            }
            out.push('}');
        }
    }
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parses a JSON document from a string.
pub fn parse(text: &str) -> Result<Json, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pos = 0usize;
    let v = parse_value(&chars, &mut pos)?;
    skip_ws(&chars, &mut pos);
    if pos != chars.len() {
        return Err(format!("trailing characters at offset {pos}"));
    }
    Ok(v)
}

fn skip_ws(c: &[char], pos: &mut usize) {
    while *pos < c.len() && c[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn parse_value(c: &[char], pos: &mut usize) -> Result<Json, String> {
    skip_ws(c, pos);
    match c.get(*pos) {
        None => Err("unexpected end of JSON".to_string()),
        Some('{') => {
            *pos += 1;
            let mut map = BTreeMap::new();
            skip_ws(c, pos);
            if c.get(*pos) == Some(&'}') {
                *pos += 1;
                return Ok(Json::Obj(map));
            }
            loop {
                skip_ws(c, pos);
                let key = match parse_value(c, pos)? {
                    Json::Str(s) => s,
                    _ => return Err("object key must be a string".to_string()),
                };
                skip_ws(c, pos);
                if c.get(*pos) != Some(&':') {
                    return Err(format!("expected `:` at offset {pos}"));
                }
                *pos += 1;
                let val = parse_value(c, pos)?;
                map.insert(key, val);
                skip_ws(c, pos);
                match c.get(*pos) {
                    Some(',') => {
                        *pos += 1;
                    }
                    Some('}') => {
                        *pos += 1;
                        return Ok(Json::Obj(map));
                    }
                    _ => return Err(format!("expected `,` or `}}` at offset {pos}")),
                }
            }
        }
        Some('[') => {
            *pos += 1;
            let mut items = Vec::new();
            skip_ws(c, pos);
            if c.get(*pos) == Some(&']') {
                *pos += 1;
                return Ok(Json::Arr(items));
            }
            loop {
                items.push(parse_value(c, pos)?);
                skip_ws(c, pos);
                match c.get(*pos) {
                    Some(',') => {
                        *pos += 1;
                    }
                    Some(']') => {
                        *pos += 1;
                        return Ok(Json::Arr(items));
                    }
                    _ => return Err(format!("expected `,` or `]` at offset {pos}")),
                }
            }
        }
        Some('"') => {
            *pos += 1;
            let mut s = String::new();
            loop {
                match c.get(*pos) {
                    None => return Err("unterminated string".to_string()),
                    Some('"') => {
                        *pos += 1;
                        return Ok(Json::Str(s));
                    }
                    Some('\\') => {
                        *pos += 1;
                        match c.get(*pos) {
                            Some('"') => s.push('"'),
                            Some('\\') => s.push('\\'),
                            Some('/') => s.push('/'),
                            Some('n') => s.push('\n'),
                            Some('r') => s.push('\r'),
                            Some('t') => s.push('\t'),
                            Some('b') => s.push('\u{0008}'),
                            Some('f') => s.push('\u{000c}'),
                            Some('u') => {
                                /* RFC 8259 §7: a \uXXXX escape may be
                                 * half of a UTF-16 surrogate pair —
                                 * Python's json.dumps (the default
                                 * ensure_ascii=True) writes EVERY
                                 * non-BMP character that way, so the
                                 * pair must be recombined; an unpaired
                                 * half stays U+FFFD. */
                                let hex4 = |c: &[char], pos: &mut usize| -> Result<u32, String> {
                                    let mut code = 0u32;
                                    for _ in 0..4 {
                                        *pos += 1;
                                        let d = c
                                            .get(*pos)
                                            .and_then(|ch| ch.to_digit(16))
                                            .ok_or("bad \\u escape")?;
                                        code = code * 16 + d;
                                    }
                                    Ok(code)
                                };
                                let mut code = hex4(c, pos)?;
                                while (0xD800..=0xDBFF).contains(&code)
                                    && c.get(*pos + 1) == Some(&'\\')
                                    && c.get(*pos + 2) == Some(&'u')
                                {
                                    *pos += 2; // onto the second escape's 'u'
                                    let next = hex4(c, pos)?;
                                    if (0xDC00..=0xDFFF).contains(&next) {
                                        code = 0x10000 + ((code - 0xD800) << 10) + (next - 0xDC00);
                                        break;
                                    }
                                    s.push('\u{fffd}'); // the lone high half
                                    code = next; // may itself start a pair
                                }
                                s.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                            }
                            other => return Err(format!("bad escape {other:?}")),
                        }
                        *pos += 1;
                    }
                    Some(ch) => {
                        s.push(*ch);
                        *pos += 1;
                    }
                }
            }
        }
        Some('t') => expect_lit(c, pos, "true", Json::Bool(true)),
        Some('f') => expect_lit(c, pos, "false", Json::Bool(false)),
        Some('n') => expect_lit(c, pos, "null", Json::Null),
        Some(_) => {
            let start = *pos;
            if c.get(*pos) == Some(&'-') {
                *pos += 1;
            }
            while *pos < c.len()
                && (c[*pos].is_ascii_digit()
                    || c[*pos] == '.'
                    || c[*pos] == 'e'
                    || c[*pos] == 'E'
                    || c[*pos] == '+'
                    || c[*pos] == '-')
            {
                *pos += 1;
            }
            let text: String = c[start..*pos].iter().collect();
            text.parse::<f64>()
                .map(Json::Num)
                .map_err(|_| format!("bad JSON number `{text}` at offset {start}"))
        }
    }
}

fn expect_lit(c: &[char], pos: &mut usize, lit: &str, val: Json) -> Result<Json, String> {
    let end = *pos + lit.len();
    if end <= c.len() && c[*pos..end].iter().collect::<String>() == lit {
        *pos = end;
        Ok(val)
    } else {
        Err(format!("bad literal at offset {pos}"))
    }
}

/// Converts a VM value to JSON.
pub fn value_to_json(v: &Value) -> Json {
    match v {
        Value::Num(n) => Json::Num(*n),
        // A 2-element [re, im] array: JSON has no complex type, and a
        // bare pair is what every consumer of this bridge already
        // understands. Matches how `sph_harm` reports its value.
        Value::Complex(z) => Json::Arr(vec![Json::Num(z.re), Json::Num(z.im)]),
        Value::Vec3(x) => Json::Arr(vec![Json::Num(x.x), Json::Num(x.y), Json::Num(x.z)]),
        Value::Quat(q) => Json::Arr(vec![
            Json::Num(q.w),
            Json::Num(q.x),
            Json::Num(q.y),
            Json::Num(q.z),
        ]),
        Value::Mat3(m) => Json::Arr(
            m.0.iter()
                .map(|row| Json::Arr(row.iter().map(|x| Json::Num(*x)).collect()))
                .collect(),
        ),
        Value::List(items) => Json::Arr(items.iter().map(value_to_json).collect()),
        Value::Str(s) => Json::Str(s.clone()),
        Value::Unit => Json::Null,
    }
}

/// Converts a JSON value to a `SET`-compatible command-line fragment.
fn json_to_literal(j: &Json) -> Result<String, String> {
    match j {
        Json::Num(n) => Ok(format!("{n:?}")),
        Json::Arr(items) => {
            let parts: Result<Vec<String>, String> = items.iter().map(json_to_literal).collect();
            Ok(format!("[{}]", parts?.join(", ")))
        }
        other => Err(format!("unsupported SET value {other:?} (use numbers or arrays)")),
    }
}

fn ok_reply(result: Json, display: String) -> String {
    let mut m = BTreeMap::new();
    m.insert("ok".to_string(), Json::Bool(true));
    m.insert("result".to_string(), result);
    m.insert("display".to_string(), Json::Str(display));
    to_string(&Json::Obj(m))
}

fn err_reply(msg: &str) -> String {
    let mut m = BTreeMap::new();
    m.insert("ok".to_string(), Json::Bool(false));
    m.insert("error".to_string(), Json::Str(msg.to_string()));
    to_string(&Json::Obj(m))
}

/// Full-system JSON dump for `{"op":"state"}`.
fn state_dump(state: &SimState) -> Json {
    let s = &state.system;
    let vec3 = |v: ::physical_object::linalg::Vec3| {
        Json::Arr(vec![Json::Num(v.x), Json::Num(v.y), Json::Num(v.z)])
    };
    let mut objs = Vec::new();
    for (i, o) in s.objects.iter().enumerate() {
        let mut m = BTreeMap::new();
        m.insert("index".to_string(), Json::Num(i as f64));
        m.insert("id".to_string(), Json::Num(o.get_id() as f64));
        m.insert("mass".to_string(), Json::Num(o.get_mass()));
        m.insert("charge".to_string(), Json::Num(o.get_charge()));
        m.insert("position".to_string(), vec3(o.get_position()));
        m.insert("velocity".to_string(), vec3(o.get_velocity()));
        m.insert("momentum".to_string(), vec3(o.get_momentum()));
        m.insert("angular_momentum".to_string(), vec3(o.get_angular_momentum()));
        let q = o.get_orientation();
        m.insert(
            "orientation".to_string(),
            Json::Arr(vec![Json::Num(q.w), Json::Num(q.x), Json::Num(q.y), Json::Num(q.z)]),
        );
        m.insert(
            "boundary".to_string(),
            Json::Str(match o.get_boundary() {
                Boundary::Point => "point".to_string(),
                Boundary::Sphere { radius } => format!("sphere r={radius}"),
                Boundary::Cuboid { half_extents } => {
                    format!("cuboid he=[{},{},{}]", half_extents[0], half_extents[1], half_extents[2])
                }
                Boundary::Torus { ring_radius, tube_radius } => {
                    format!("torus ring={ring_radius} tube={tube_radius}")
                }
                Boundary::Disk { radius } => format!("disk r={radius}"),
                Boundary::Cylinder { radius, half_height } => {
                    format!("cylinder r={radius} h={}", 2.0 * half_height)
                }
                Boundary::Dumbbell { r1, r2, rod_radius, z1, z2, .. } => {
                    format!("dumbbell r1={r1} r2={r2} rod_r={rod_radius} len={}", z2 - z1)
                }
            }),
        );
        m.insert("kinetic_energy".to_string(), Json::Num(o.kinetic_energy()));
        m.insert("restitution".to_string(), Json::Num(o.get_restitution()));
        m.insert("inverse_mass".to_string(), Json::Num(o.get_inverse_mass()));
        m.insert("wall".to_string(), Json::Bool(state.wall_indices.contains(&i)));
        if let Some(n) = state.names.iter().find(|(_, oi)| **oi == i).map(|(n, _)| n) {
            m.insert("name".to_string(), Json::Str(n.clone()));
        }
        objs.push(Json::Obj(m));
    }
    /* contacts of the last STEP/RUN: pair, event time, point, the
     * contact normal (unit, from i toward j — the action-reaction
     * line), penetration depth and applied impulse */
    let mut contacts = Vec::new();
    for c in s.contacts.iter() {
        let mut cm = BTreeMap::new();
        cm.insert("i".to_string(), Json::Num(c.i as f64));
        cm.insert("j".to_string(), Json::Num(c.j as f64));
        cm.insert("t".to_string(), Json::Num(c.t));
        cm.insert("point".to_string(), vec3(c.point));
        cm.insert("normal".to_string(), vec3(c.normal));
        cm.insert("depth".to_string(), Json::Num(c.depth));
        cm.insert("rel_vel_n".to_string(), Json::Num(c.rel_vel_n));
        cm.insert("impulse".to_string(), Json::Num(c.impulse_n));
        contacts.push(Json::Obj(cm));
    }
    let mut m = BTreeMap::new();
    m.insert("time".to_string(), Json::Num(s.time));
    m.insert("g_constant".to_string(), Json::Num(s.g_constant));
    m.insert("softening".to_string(), Json::Num(s.softening));
    m.insert("uniform_gravity".to_string(), vec3(s.uniform_gravity));
    m.insert("e_field".to_string(), vec3(s.e_field));
    m.insert("b_field".to_string(), vec3(s.b_field));
    m.insert("method".to_string(), Json::Str(format!("{:?}", s.method)));
    m.insert("total_energy".to_string(), Json::Num(s.total_energy()));
    m.insert("collide_enabled".to_string(), Json::Bool(s.collide_enabled));
    m.insert("collision_count".to_string(), Json::Num(s.collision_count as f64));
    m.insert(
        "box".to_string(),
        match state.box_size {
            Some(size) => Json::Num(size),
            None => Json::Null,
        },
    );
    /* Rigid joints, with their world-frame geometry resolved at this
     * instant: the attachment point every front end wants to draw, and
     * the hinge axis it turns about. The body-frame arms the joint
     * stores are not directly useful to a viewer. */
    let mut joints = Vec::new();
    for (k, joint) in state.system.constraints.joints.iter().enumerate() {
        let (i, j) = joint.bodies();
        let mut jm = BTreeMap::new();
        jm.insert("index".to_string(), Json::Num(k as f64));
        jm.insert("kind".to_string(), Json::Str(joint.kind().to_string()));
        jm.insert("i".to_string(), Json::Num(i as f64));
        jm.insert("j".to_string(), Json::Num(j as f64));
        jm.insert("rows".to_string(), Json::Num(joint.rows() as f64));
        let world = |body: usize, local: ::physical_object::linalg::Vec3| {
            s.objects[body].get_position()
                + s.objects[body].get_orientation().normalize().rotate(local)
        };
        let dir = |body: usize, local: ::physical_object::linalg::Vec3| {
            s.objects[body].get_orientation().normalize().rotate(local)
        };
        use ::physical_object::constrain::Joint;
        match *joint {
            Joint::Distance { .. } => {
                jm.insert("point".to_string(), vec3(s.objects[i].get_position()));
                jm.insert("point_j".to_string(), vec3(s.objects[j].get_position()));
            }
            Joint::Ball { a_i, a_j, .. } => {
                jm.insert("point".to_string(), vec3(world(i, a_i)));
                jm.insert("point_j".to_string(), vec3(world(j, a_j)));
            }
            Joint::Hinge { a_i, a_j, h_i, .. } => {
                jm.insert("point".to_string(), vec3(world(i, a_i)));
                jm.insert("point_j".to_string(), vec3(world(j, a_j)));
                jm.insert("axis".to_string(), vec3(dir(i, h_i)));
            }
            Joint::Universal { a_i, a_j, u_i, u_j, .. } => {
                jm.insert("point".to_string(), vec3(world(i, a_i)));
                jm.insert("point_j".to_string(), vec3(world(j, a_j)));
                jm.insert("axis".to_string(), vec3(dir(i, u_i)));
                jm.insert("axis_j".to_string(), vec3(dir(j, u_j)));
            }
            /* A gear holds no point at all — only a proportion — so it
             * has no pivot to report. Each body's own centre is given,
             * which is where a viewer would draw the wheels, and the
             * shared axis and the ratio go alongside. */
            /* A rack holds no pivot either. The pinion's centre and
             * the rack's are given, with the axis it turns about, the
             * direction it slides along, and the pitch radius. */
            Joint::Prismatic { d_i, .. } => {
                jm.insert("point".to_string(), vec3(s.objects[i].get_position()));
                jm.insert("point_j".to_string(), vec3(s.objects[j].get_position()));
                jm.insert("axis".to_string(), vec3(dir(i, d_i)));
            }
            Joint::Rack { h_i, d_j, radius, .. } => {
                jm.insert("point".to_string(), vec3(s.objects[i].get_position()));
                jm.insert("point_j".to_string(), vec3(s.objects[j].get_position()));
                jm.insert("axis".to_string(), vec3(dir(i, h_i)));
                jm.insert("axis_j".to_string(), vec3(dir(j, d_j)));
                jm.insert("radius".to_string(), Json::Num(radius));
            }
            Joint::Gear { axis, p, q, .. } => {
                jm.insert("point".to_string(), vec3(s.objects[i].get_position()));
                jm.insert("point_j".to_string(), vec3(s.objects[j].get_position()));
                jm.insert("axis".to_string(), vec3(axis));
                jm.insert("ratio".to_string(), Json::Num(f64::from(p) / f64::from(q)));
            }
        }
        joints.push(Json::Obj(jm));
    }
    let (gdrift, gddrift) = state.system.constraints.drift(&state.system);
    m.insert("joints".to_string(), Json::Arr(joints));
    m.insert("joint_drift".to_string(), Json::Num(gdrift));
    m.insert("joint_rate_drift".to_string(), Json::Num(gddrift));
    m.insert("contacts".to_string(), Json::Arr(contacts));
    m.insert("objects".to_string(), Json::Arr(objs));
    Json::Obj(m)
}

/// Processes one request line; returns the reply line, or `None` for
/// `{"op":"quit"}`.
pub fn handle_request(line: &str, state: &mut SimState) -> Option<String> {
    let req = match parse(line) {
        Ok(j) => j,
        Err(e) => return Some(err_reply(&format!("bad JSON: {e}"))),
    };
    let op = match req.get("op").and_then(|j| j.as_str()) {
        Some(op) => op.to_string(),
        None => return Some(err_reply("missing \"op\"")),
    };
    match op.as_str() {
        "quit" => None,
        "help" => Some(ok_reply(Json::Str(crate::vm::HELP_TEXT.to_string()), crate::vm::HELP_TEXT.to_string())),
        "state" => Some(ok_reply(state_dump(state), String::new())),
        "events" => {
            let events = match &state.scene {
                Some(s) => s.drain_events().unwrap_or_default(),
                None => Vec::new(),
            };
            let display = events.join("\n");
            let arr = Json::Arr(events.into_iter().map(Json::Str).collect());
            Some(ok_reply(arr, display))
        }
        "exec" => {
            let code = match req.get("code").and_then(|j| j.as_str()) {
                Some(c) => c,
                None => return Some(err_reply("exec needs \"code\"")),
            };
            match execute_line(code, state) {
                Ok(v) => Some(ok_reply(value_to_json(&v), v.to_string())),
                Err(e) => Some(err_reply(&e)),
            }
        }
        "get" => {
            let path = match req.get("path").and_then(|j| j.as_str()) {
                Some(p) => p,
                None => return Some(err_reply("get needs \"path\"")),
            };
            match execute_line(&format!("get {path}"), state) {
                Ok(v) => Some(ok_reply(value_to_json(&v), v.to_string())),
                Err(e) => Some(err_reply(&e)),
            }
        }
        "set" => {
            let path = match req.get("path").and_then(|j| j.as_str()) {
                Some(p) => p,
                None => return Some(err_reply("set needs \"path\"")),
            };
            let value = match req.get("value") {
                Some(v) => v,
                None => return Some(err_reply("set needs \"value\"")),
            };
            let lit = match json_to_literal(value) {
                Ok(l) => l,
                Err(e) => return Some(err_reply(&e)),
            };
            match execute_line(&format!("set {path} = {lit}"), state) {
                Ok(_) => Some(ok_reply(Json::Null, String::new())),
                Err(e) => Some(err_reply(&e)),
            }
        }
        other => Some(err_reply(&format!("unknown op \"{other}\""))),
    }
}

/// The `--machine` stdin/stdout server loop.
pub fn serve() {
    let mut state = SimState::default();
    /* scene events are pushed as unsolicited {"event":...} lines */
    state.machine_mode = true;
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        match handle_request(&line, &mut state) {
            Some(reply) => {
                let _ = writeln!(stdout, "{reply}");
                let _ = stdout.flush();
            }
            None => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let doc = r#"{"op":"set","path":"obj0.mass","value":[1.5e0,-2,3]}"#;
        let j = parse(doc).unwrap();
        assert_eq!(j.get("op").unwrap().as_str().unwrap(), "set");
        let back = parse(&to_string(&j)).unwrap();
        assert_eq!(j, back);
        assert!(parse("{\"a\":}").is_err());
        assert_eq!(parse("\"a\\u0041b\"").unwrap(), Json::Str("aAb".to_string()));
    }

    /// RFC 8259 surrogate pairs decode to the real character — this is
    /// how Python's json.dumps (default ensure_ascii=True) writes every
    /// non-BMP char, so the shipped kernel hits it for any emoji or
    /// astral math symbol. Unpaired halves stay U+FFFD, exactly as
    /// before.
    #[test]
    fn surrogate_pairs_decode_to_the_real_character() {
        // U+1F600 GRINNING FACE as json.dumps writes it.
        assert_eq!(parse("\"\\ud83d\\ude00\"").unwrap(), Json::Str("\u{1F600}".to_string()));
        // U+1D4A9 MATHEMATICAL SCRIPT CAPITAL N, mid-string.
        assert_eq!(
            parse("\"a\\ud835\\udca9b\"").unwrap(),
            Json::Str(format!("a{}b", '\u{1D4A9}'))
        );
        // A lone high half stays U+FFFD (previous behavior kept)...
        assert_eq!(parse("\"\\ud800x\"").unwrap(), Json::Str("\u{fffd}x".to_string()));
        // ...also when the following escape is not a low surrogate,
        assert_eq!(parse("\"\\ud800\\u0041\"").unwrap(), Json::Str("\u{fffd}A".to_string()));
        // ...and a lone low half likewise.
        assert_eq!(parse("\"\\ude00\"").unwrap(), Json::Str("\u{fffd}".to_string()));
        // A high half chained before a real pair still yields the pair.
        assert_eq!(
            parse("\"\\ud800\\ud83d\\ude00\"").unwrap(),
            Json::Str(format!("\u{fffd}{}", '\u{1F600}'))
        );
    }

    #[test]
    fn exec_get_set_state_flow() {
        let mut st = SimState::default();
        let r = handle_request(
            r#"{"op":"exec","code":"new sphere { mass = 2, radius = 0.5 }"}"#,
            &mut st,
        )
        .unwrap();
        assert!(r.contains("\"ok\":true"), "{r}");
        assert!(r.contains("obj0"), "{r}");

        let r = handle_request(r#"{"op":"set","path":"obj0.velocity","value":[1,0,-0.5]}"#, &mut st)
            .unwrap();
        assert!(r.contains("\"ok\":true"), "{r}");

        let r = handle_request(r#"{"op":"get","path":"obj0.momentum"}"#, &mut st).unwrap();
        assert!(r.contains("[2.0,0.0,-1.0]"), "{r}");

        let r = handle_request(r#"{"op":"state"}"#, &mut st).unwrap();
        assert!(r.contains("\"objects\":["), "{r}");
        assert!(r.contains("\"mass\":2.0"), "{r}");

        let r = handle_request(r#"{"op":"exec","code":"get obj9.mass"}"#, &mut st).unwrap();
        assert!(r.contains("\"ok\":false"), "{r}");

        assert!(handle_request(r#"{"op":"quit"}"#, &mut st).is_none());
    }

    #[test]
    fn state_reports_box_walls_and_inverse_mass() {
        let mut st = SimState::default();
        /* no box: the state carries an explicit null */
        let r = handle_request(r#"{"op":"state"}"#, &mut st).unwrap();
        assert!(r.contains("\"box\":null"), "{r}");

        handle_request(r#"{"op":"exec","code":"box 4"}"#, &mut st).unwrap();
        let r = handle_request(r#"{"op":"state"}"#, &mut st).unwrap();
        assert!(r.contains("\"box\":4.0"), "{r}");
        assert!(r.contains("\"wall\":true"), "{r}");
        assert!(r.contains("\"inverse_mass\":0.0"), "{r}");

        handle_request(r#"{"op":"exec","code":"box off"}"#, &mut st).unwrap();
        let r = handle_request(r#"{"op":"state"}"#, &mut st).unwrap();
        assert!(r.contains("\"box\":null"), "{r}");
    }
}
