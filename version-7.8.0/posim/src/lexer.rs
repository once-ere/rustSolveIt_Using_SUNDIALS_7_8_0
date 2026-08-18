//! Lexer ("flex" analog): turns a command line into a token stream that
//! feeds the stack machine's parser. Keywords are matched
//! case-insensitively with longest-match identifier rules; every token
//! carries its column for precise error messages.

use std::fmt;

/// Reserved keywords of the command grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keyword {
    New,
    Set,
    Get,
    Del,
    List,
    Step,
    Run,
    Steps,
    Method,
    Adams,
    Bdf,
    Sprk,
    Energy,
    Com,
    Momentum,
    Angmom,
    Laplace,
    Help,
    Point,
    Sphere,
    Cuboid,
    Torus,
    Disk,
    Cylinder,
    Reset,
    /* the rigid bounding box */
    Box,
    /* one- and two-dimensional quantum mechanics */
    Qm,
    Qm2,
    Qm3,
    /* graphical scene commands */
    Scene,
    Create,
    Close,
    Translate,
    Rotate,
    Zoom,
    In,
    Out,
    Hide,
    Show,
    Refresh,
    Redraw,
    Start,
    Stop,
    Pause,
    Reverse,
    SetTimeStep,
    Status,
    Events,
    All,
    /* user definitions and names */
    As,
    Let,
    Funcs,
    Dumbbell,
    /* collision commands */
    Collide,
    Contacts,
    On,
    Off,
    /* constrained dynamics, equilibrium and sensitivity (IDA, KINSOL,
     * CVODES, IDAS) */
    Constrain,
    Constraints,
    Ball,
    Hinge,
    Gear,
    Rack,
    Prismatic,
    Universal,
    Equilibrium,
    Sensitivity,
    Ida,
}

impl Keyword {
    fn from_ident(s: &str) -> Option<Keyword> {
        match s.to_ascii_lowercase().as_str() {
            "new" => Some(Keyword::New),
            "set" => Some(Keyword::Set),
            "get" => Some(Keyword::Get),
            "del" | "delete" => Some(Keyword::Del),
            "list" => Some(Keyword::List),
            "step" => Some(Keyword::Step),
            "run" => Some(Keyword::Run),
            "steps" => Some(Keyword::Steps),
            "method" => Some(Keyword::Method),
            "adams" => Some(Keyword::Adams),
            "bdf" => Some(Keyword::Bdf),
            "sprk" => Some(Keyword::Sprk),
            "energy" => Some(Keyword::Energy),
            "com" => Some(Keyword::Com),
            "momentum" => Some(Keyword::Momentum),
            "angmom" => Some(Keyword::Angmom),
            "laplace" => Some(Keyword::Laplace),
            "help" => Some(Keyword::Help),
            "point" => Some(Keyword::Point),
            "sphere" => Some(Keyword::Sphere),
            "cuboid" | "cube" => Some(Keyword::Cuboid),
            "torus" => Some(Keyword::Torus),
            "disk" | "disc" => Some(Keyword::Disk),
            "cylinder" => Some(Keyword::Cylinder),
            "box" => Some(Keyword::Box),
            "reset" => Some(Keyword::Reset),
            "qm" => Some(Keyword::Qm),
            "qm2" => Some(Keyword::Qm2),
            "qm3" => Some(Keyword::Qm3),
            "scene" => Some(Keyword::Scene),
            "create" => Some(Keyword::Create),
            "close" | "destroy" => Some(Keyword::Close),
            "translate" => Some(Keyword::Translate),
            "rotate" => Some(Keyword::Rotate),
            "zoom" => Some(Keyword::Zoom),
            "in" => Some(Keyword::In),
            "out" => Some(Keyword::Out),
            "hide" => Some(Keyword::Hide),
            "show" => Some(Keyword::Show),
            "refresh" => Some(Keyword::Refresh),
            "redraw" => Some(Keyword::Redraw),
            "start" => Some(Keyword::Start),
            "stop" => Some(Keyword::Stop),
            "pause" => Some(Keyword::Pause),
            "reverse" => Some(Keyword::Reverse),
            "set_time_step" | "settimestep" => Some(Keyword::SetTimeStep),
            "status" => Some(Keyword::Status),
            "events" => Some(Keyword::Events),
            "all" => Some(Keyword::All),
            "as" => Some(Keyword::As),
            "let" => Some(Keyword::Let),
            "funcs" | "functions" => Some(Keyword::Funcs),
            "dumbbell" | "dumbell" => Some(Keyword::Dumbbell),
            "collide" => Some(Keyword::Collide),
            "contacts" => Some(Keyword::Contacts),
            "constrain" => Some(Keyword::Constrain),
            "constraints" => Some(Keyword::Constraints),
            "ball" => Some(Keyword::Ball),
            "hinge" => Some(Keyword::Hinge),
            "gear" => Some(Keyword::Gear),
            "rack" => Some(Keyword::Rack),
            "prismatic" => Some(Keyword::Prismatic),
            "universal" | "cardan" => Some(Keyword::Universal),
            "equilibrium" | "equil" => Some(Keyword::Equilibrium),
            "sensitivity" | "sens" => Some(Keyword::Sensitivity),
            "ida" => Some(Keyword::Ida),
            "on" => Some(Keyword::On),
            "off" => Some(Keyword::Off),
            _ => None,
        }
    }
}

/// Token kinds fed to the parser.
/// A number immediately followed by `i` is an IMAGINARY literal.
///
/// The `i` must not be followed by another identifier character, or
/// `2in` would silently become `2i` then `n` — so `2i` is imaginary
/// while `2intercept` stays a number beside an identifier, exactly as
/// it did before complex numbers existed.
fn imaginary_suffix(
    chars: &[char],
    i: usize,
    len: usize,
    num: f64,
) -> (TokKind, usize) {
    let j = i + len;
    if j < chars.len()
        && chars[j] == 'i'
        && !(j + 1 < chars.len() && (chars[j + 1].is_alphanumeric() || chars[j + 1] == '_'))
    {
        (TokKind::Imaginary(num), len + 1)
    } else {
        (TokKind::Number(num), len)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokKind {
    Keyword(Keyword),
    Ident(String),
    Number(f64),
    /// An imaginary literal: a number with an `i` suffix, e.g. `3i`.
    /// `2 + 3i` is then ordinary addition of a real and an imaginary.
    Imaginary(f64),
    /// A double-quoted string literal (escapes: \" \\ \n).
    Str(String),
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    LParen,
    RParen,
    Comma,
    Dot,
    Equals,
    Plus,
    Minus,
    Star,
    Slash,
    /* comparisons — they yield 1 for true and 0 for false, since this
     * language has numbers but no boolean type */
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
}

/// A token plus its 1-based source column.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokKind,
    pub col: usize,
}

impl fmt::Display for TokKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokKind::Keyword(k) => write!(f, "{k:?}"),
            TokKind::Ident(s) => write!(f, "identifier `{s}`"),
            TokKind::Number(n) => write!(f, "number {n}"),
            TokKind::Imaginary(n) => write!(f, "imaginary {n}i"),
            TokKind::Str(st) => write!(f, "string \"{st}\""),
            TokKind::LBracket => write!(f, "`[`"),
            TokKind::RBracket => write!(f, "`]`"),
            TokKind::LBrace => write!(f, "`{{`"),
            TokKind::RBrace => write!(f, "`}}`"),
            TokKind::LParen => write!(f, "`(`"),
            TokKind::RParen => write!(f, "`)`"),
            TokKind::Comma => write!(f, "`,`"),
            TokKind::Dot => write!(f, "`.`"),
            TokKind::Equals => write!(f, "`=`"),
            TokKind::Plus => write!(f, "`+`"),
            TokKind::Minus => write!(f, "`-`"),
            TokKind::Star => write!(f, "`*`"),
            TokKind::Slash => write!(f, "`/`"),
            TokKind::Lt => write!(f, "`<`"),
            TokKind::Le => write!(f, "`<=`"),
            TokKind::Gt => write!(f, "`>`"),
            TokKind::Ge => write!(f, "`>=`"),
            TokKind::EqEq => write!(f, "`==`"),
            TokKind::Ne => write!(f, "`!=`"),
        }
    }
}

/// Tokenizes a command line. `#` starts a comment running to the end of
/// the line.
pub fn tokenize(line: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = line.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        let col = i + 1;
        match c {
            '#' => break,
            c if c.is_whitespace() => {
                i += 1;
            }
            '[' => {
                toks.push(Token { kind: TokKind::LBracket, col });
                i += 1;
            }
            ']' => {
                toks.push(Token { kind: TokKind::RBracket, col });
                i += 1;
            }
            '{' => {
                toks.push(Token { kind: TokKind::LBrace, col });
                i += 1;
            }
            '}' => {
                toks.push(Token { kind: TokKind::RBrace, col });
                i += 1;
            }
            '(' => {
                toks.push(Token { kind: TokKind::LParen, col });
                i += 1;
            }
            ')' => {
                toks.push(Token { kind: TokKind::RParen, col });
                i += 1;
            }
            ',' => {
                toks.push(Token { kind: TokKind::Comma, col });
                i += 1;
            }
            '.' if i + 1 < chars.len() && chars[i + 1].is_ascii_digit() => {
                /* a number like .5 */
                let (num, len) = lex_number(&chars[i..], col)?;
                let (kind, len) = imaginary_suffix(&chars, i, len, num);
                toks.push(Token { kind, col });
                i += len;
            }
            '.' => {
                toks.push(Token { kind: TokKind::Dot, col });
                i += 1;
            }
            /* `==` must be tried before the single `=`, which is
             * assignment in SET and in NEW initialiser lists */
            '=' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                toks.push(Token { kind: TokKind::EqEq, col });
                i += 2;
            }
            '=' => {
                toks.push(Token { kind: TokKind::Equals, col });
                i += 1;
            }
            '<' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                toks.push(Token { kind: TokKind::Le, col });
                i += 2;
            }
            '<' => {
                toks.push(Token { kind: TokKind::Lt, col });
                i += 1;
            }
            '>' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                toks.push(Token { kind: TokKind::Ge, col });
                i += 2;
            }
            '>' => {
                toks.push(Token { kind: TokKind::Gt, col });
                i += 1;
            }
            '!' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                toks.push(Token { kind: TokKind::Ne, col });
                i += 2;
            }
            '+' => {
                toks.push(Token { kind: TokKind::Plus, col });
                i += 1;
            }
            '-' => {
                toks.push(Token { kind: TokKind::Minus, col });
                i += 1;
            }
            '*' => {
                toks.push(Token { kind: TokKind::Star, col });
                i += 1;
            }
            '/' => {
                toks.push(Token { kind: TokKind::Slash, col });
                i += 1;
            }
            '"' => {
                /* string literal with minimal escapes */
                let mut out = String::new();
                let mut j = i + 1;
                let mut closed = false;
                while j < chars.len() {
                    match chars[j] {
                        '"' => {
                            closed = true;
                            j += 1;
                            break;
                        }
                        '\\' if j + 1 < chars.len() => {
                            out.push(match chars[j + 1] {
                                'n' => '\n',
                                other => other, /* \" and \\ and anything else verbatim */
                            });
                            j += 2;
                        }
                        other => {
                            out.push(other);
                            j += 1;
                        }
                    }
                }
                if !closed {
                    return Err(format!("lexical error at column {col}: unterminated string"));
                }
                toks.push(Token { kind: TokKind::Str(out), col });
                i = j;
            }
            c if c.is_ascii_digit() => {
                let (num, len) = lex_number(&chars[i..], col)?;
                let (kind, len) = imaginary_suffix(&chars, i, len, num);
                toks.push(Token { kind, col });
                i += len;
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                /* BALL, HINGE and UNIVERSAL are *contextual* keywords:
                 * they name commands, and only in command position. Any
                 * other word does not need this, but `ball` is exactly
                 * the sort of thing a physics simulator's user calls an
                 * object — `new sphere as ball` and `get ball.mass` were
                 * both already in the tests and the docs — and reserving
                 * it outright would have broken them. Off the front of
                 * the line these three lex as ordinary identifiers. */
                let kind = match Keyword::from_ident(&word) {
                    Some(k)
                        if !toks.is_empty()
                            && matches!(
                                k,
                                Keyword::Ball
                                    | Keyword::Hinge
                                    | Keyword::Universal
                                    | Keyword::Gear
                                    | Keyword::Rack
                                    | Keyword::Prismatic
                            ) =>
                    {
                        TokKind::Ident(word)
                    }
                    Some(k) => TokKind::Keyword(k),
                    None => TokKind::Ident(word),
                };
                toks.push(Token { kind, col });
            }
            other => {
                return Err(format!("lexical error at column {col}: unexpected character `{other}`"));
            }
        }
    }
    Ok(toks)
}

/// Lexes a number `123`, `1.5`, `.5`, `1e-3`, `2.5E+4`.
fn lex_number(chars: &[char], col: usize) -> Result<(f64, usize), String> {
    let mut len = 0usize;
    while len < chars.len() && chars[len].is_ascii_digit() {
        len += 1;
    }
    if len < chars.len() && chars[len] == '.' {
        len += 1;
        while len < chars.len() && chars[len].is_ascii_digit() {
            len += 1;
        }
    }
    if len < chars.len() && (chars[len] == 'e' || chars[len] == 'E') {
        let mut j = len + 1;
        if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
            j += 1;
        }
        if j < chars.len() && chars[j].is_ascii_digit() {
            len = j;
            while len < chars.len() && chars[len].is_ascii_digit() {
                len += 1;
            }
        }
    }
    let text: String = chars[..len].iter().collect();
    text.parse::<f64>()
        .map(|v| (v, len))
        .map_err(|_| format!("lexical error at column {col}: bad number `{text}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords_case_insensitive_and_idents() {
        let t = tokenize("NEW sphere { mass = 2.5e-1 }").unwrap();
        assert_eq!(t[0].kind, TokKind::Keyword(Keyword::New));
        assert_eq!(t[1].kind, TokKind::Keyword(Keyword::Sphere));
        assert_eq!(t[2].kind, TokKind::LBrace);
        assert_eq!(t[3].kind, TokKind::Ident("mass".to_string()));
        assert_eq!(t[4].kind, TokKind::Equals);
        assert_eq!(t[5].kind, TokKind::Number(0.25));
        assert_eq!(t[6].kind, TokKind::RBrace);
    }

    #[test]
    fn paths_vectors_arith() {
        let t = tokenize("set obj0.position = [1, -2.5, .5] * 2").unwrap();
        let kinds: Vec<&TokKind> = t.iter().map(|x| &x.kind).collect();
        assert!(matches!(kinds[1], TokKind::Ident(s) if s == "obj0"));
        assert_eq!(*kinds[2], TokKind::Dot);
        assert!(matches!(kinds[3], TokKind::Ident(s) if s == "position"));
        assert_eq!(*kinds[4], TokKind::Equals);
        assert_eq!(*kinds[5], TokKind::LBracket);
        assert_eq!(*kinds[7], TokKind::Comma);
        assert_eq!(*kinds[8], TokKind::Minus);
        assert_eq!(*kinds[9], TokKind::Number(2.5));
        assert_eq!(*kinds[11], TokKind::Number(0.5));
        assert_eq!(*kinds[13], TokKind::Star);
    }

    #[test]
    fn scene_keywords() {
        let t = tokenize("SCENE zoom In out set_time_step ALL").unwrap();
        assert_eq!(t[0].kind, TokKind::Keyword(Keyword::Scene));
        assert_eq!(t[1].kind, TokKind::Keyword(Keyword::Zoom));
        assert_eq!(t[2].kind, TokKind::Keyword(Keyword::In));
        assert_eq!(t[3].kind, TokKind::Keyword(Keyword::Out));
        assert_eq!(t[4].kind, TokKind::Keyword(Keyword::SetTimeStep));
        assert_eq!(t[5].kind, TokKind::Keyword(Keyword::All));
        /* both spellings of CLOSE */
        assert_eq!(tokenize("close").unwrap()[0].kind, TokKind::Keyword(Keyword::Close));
        assert_eq!(tokenize("destroy").unwrap()[0].kind, TokKind::Keyword(Keyword::Close));
    }

    #[test]
    fn comments_and_errors() {
        assert!(tokenize("list # everything after is ignored ???").unwrap().len() == 1);
        let err = tokenize("get obj0 @").unwrap_err();
        assert!(err.contains("column 10"), "{err}");
    }
}
