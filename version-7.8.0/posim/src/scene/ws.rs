//! Minimal RFC 6455 WebSocket support (server side) plus the SHA-1 and
//! base64 primitives its opening handshake requires. Hand-rolled on
//! purpose: the workspace allows zero external dependencies and zero
//! `unsafe`, so `std` I/O + integer arithmetic is all we may use.
//!
//! Scope: text, close, ping and pong frames — exactly what the scene
//! window protocol needs. Fragmented messages and binary frames are
//! rejected with an error so a misbehaving client cannot wedge the
//! server.

use std::io::{Read, Write};
use std::net::TcpStream;

/// The GUID nailed down by RFC 6455 §1.3 for the accept-key digest.
const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// SHA-1 digest (FIPS 180-4). Needed only for the WebSocket handshake —
/// this is not a security boundary, it is a protocol checksum.
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for (i, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let tmp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, word) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Standard base64 (RFC 4648, with padding).
pub fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Computes the `Sec-WebSocket-Accept` value for a client key.
pub fn accept_key(client_key: &str) -> String {
    let joined = format!("{}{}", client_key.trim(), WS_GUID);
    base64_encode(&sha1(joined.as_bytes()))
}

/// A single parsed WebSocket message from the client.
#[derive(Debug, PartialEq)]
pub enum WsMessage {
    Text(String),
    Ping(Vec<u8>),
    Pong,
    Close,
}

/// Reads one complete frame from a client (client frames are masked per
/// RFC 6455 §5.1). Returns `Ok(None)` when the stream's read timeout
/// expires before a frame starts (so callers can poll a shutdown flag);
/// returns `Close` on a close frame; errors on I/O failure,
/// fragmentation, or binary frames.
pub fn read_frame(stream: &mut TcpStream) -> Result<Option<WsMessage>, String> {
    let mut head = [0u8; 2];
    if let Err(e) = stream.read_exact(&mut head) {
        /* a timeout before the frame header is a poll tick, not an error */
        if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut {
            return Ok(None);
        }
        return Err(format!("ws read: {e}"));
    }
    let fin = head[0] & 0x80 != 0;
    let opcode = head[0] & 0x0F;
    let masked = head[1] & 0x80 != 0;
    let mut len = (head[1] & 0x7F) as u64;
    if !fin {
        return Err("ws: fragmented frames are not supported".to_string());
    }
    if len == 126 {
        let mut ext = [0u8; 2];
        stream.read_exact(&mut ext).map_err(|e| format!("ws read: {e}"))?;
        len = u16::from_be_bytes(ext) as u64;
    } else if len == 127 {
        let mut ext = [0u8; 8];
        stream.read_exact(&mut ext).map_err(|e| format!("ws read: {e}"))?;
        len = u64::from_be_bytes(ext);
    }
    if len > 1 << 20 {
        return Err("ws: frame too large".to_string());
    }
    let mask = if masked {
        let mut m = [0u8; 4];
        stream.read_exact(&mut m).map_err(|e| format!("ws read: {e}"))?;
        Some(m)
    } else {
        None
    };
    let mut payload = vec![0u8; len as usize];
    stream.read_exact(&mut payload).map_err(|e| format!("ws read: {e}"))?;
    if let Some(m) = mask {
        for (i, byte) in payload.iter_mut().enumerate() {
            *byte ^= m[i % 4];
        }
    }
    match opcode {
        0x1 => String::from_utf8(payload)
            .map(|s| Some(WsMessage::Text(s)))
            .map_err(|_| "ws: invalid UTF-8 in text frame".to_string()),
        0x8 => Ok(Some(WsMessage::Close)),
        0x9 => Ok(Some(WsMessage::Ping(payload))),
        0xA => Ok(Some(WsMessage::Pong)),
        0x2 => Err("ws: binary frames are not supported".to_string()),
        other => Err(format!("ws: unsupported opcode {other}")),
    }
}

/// Builds an unmasked server frame with the given opcode and payload.
fn build_frame(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(payload.len() + 10);
    frame.push(0x80 | opcode);
    let len = payload.len();
    if len < 126 {
        frame.push(len as u8);
    } else if len <= u16::MAX as usize {
        frame.push(126);
        frame.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(len as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    frame
}

/// Sends a text frame to the client.
pub fn write_text(stream: &mut TcpStream, text: &str) -> Result<(), String> {
    stream
        .write_all(&build_frame(0x1, text.as_bytes()))
        .and_then(|_| stream.flush())
        .map_err(|e| format!("ws write: {e}"))
}

/// Sends a pong frame (in reply to a ping).
pub fn write_pong(stream: &mut TcpStream, payload: &[u8]) -> Result<(), String> {
    stream
        .write_all(&build_frame(0xA, payload))
        .and_then(|_| stream.flush())
        .map_err(|e| format!("ws write: {e}"))
}

/// Sends a close frame.
pub fn write_close(stream: &mut TcpStream) -> Result<(), String> {
    stream
        .write_all(&build_frame(0x8, &[]))
        .and_then(|_| stream.flush())
        .map_err(|e| format!("ws write: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_known_vectors() {
        /* FIPS 180-4 test vectors */
        let hex = |d: [u8; 20]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(hex(sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(
            hex(sha1(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }

    #[test]
    fn base64_known_vectors() {
        /* RFC 4648 §10 test vectors */
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn rfc6455_accept_key() {
        /* the worked example from RFC 6455 §1.3 */
        assert_eq!(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }
}
