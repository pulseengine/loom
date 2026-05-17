//! Minimal HTTP/1.1 request-line and header parser written from scratch.
//!
//! This is **not** the `httparse` crate. It is a deliberately small, all-Rust,
//! zero-allocation, byte-at-a-time state machine that parses the request line
//!
//!     METHOD SP request-target SP HTTP/1.1 CRLF
//!     (header-name ":" OWS field-value OWS CRLF)*
//!     CRLF
//!
//! and returns the byte offset where the body begins. This file exists to
//! provide a small "real-world-ish" Rust → wasm corpus fixture for the LOOM
//! optimizer; correctness against the full RFC is **not** the goal — being a
//! plausible mix of branches, range checks and ASCII comparisons is.
//!
//! The exported entry point is `parse(ptr, len) -> u32`, returning either the
//! byte offset of the start of the body, or `0` on parse failure.

#![no_std]
#![allow(clippy::missing_safety_doc)]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Status returned by inner parsing helpers.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Status {
    Ok,
    Bad,
    Partial,
}

/// Byte classification — keeping this as `match` over `u8` produces a fair
/// chunk of branchy code in wasm, which is what we want for the corpus.
#[inline(always)]
fn is_tchar(b: u8) -> bool {
    // RFC 7230 token character set.
    matches!(
        b,
        b'!' | b'#' | b'$' | b'%' | b'&' | b'\'' | b'*' | b'+' | b'-' | b'.'
            | b'^' | b'_' | b'`' | b'|' | b'~'
            | b'0'..=b'9'
            | b'A'..=b'Z'
            | b'a'..=b'z'
    )
}

#[inline(always)]
fn is_vchar(b: u8) -> bool {
    // visible ASCII, no controls
    b > 0x20 && b < 0x7f
}

#[inline(always)]
fn is_ows(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

/// Parse the request line: METHOD SP target SP HTTP/1.1 CRLF.
/// On success, returns (Ok, offset just after CRLF). On partial, returns
/// (Partial, original i). On bad, returns (Bad, _).
fn parse_request_line(buf: &[u8], mut i: usize) -> (Status, usize) {
    let n = buf.len();

    // METHOD
    let mut saw_method = 0usize;
    while i < n {
        let b = buf[i];
        if b == b' ' {
            break;
        }
        if !is_tchar(b) {
            return (Status::Bad, i);
        }
        i += 1;
        saw_method += 1;
        if saw_method > 16 {
            return (Status::Bad, i);
        }
    }
    if saw_method == 0 || i >= n {
        return (Status::Partial, i);
    }
    // Skip the SP.
    i += 1;

    // request-target — any vchar, until SP.
    let mut saw_target = 0usize;
    while i < n {
        let b = buf[i];
        if b == b' ' {
            break;
        }
        if !is_vchar(b) {
            return (Status::Bad, i);
        }
        i += 1;
        saw_target += 1;
        if saw_target > 8192 {
            return (Status::Bad, i);
        }
    }
    if saw_target == 0 || i >= n {
        return (Status::Partial, i);
    }
    i += 1;

    // HTTP-version: literal "HTTP/1.1" or "HTTP/1.0".
    if i + 8 > n {
        return (Status::Partial, i);
    }
    let v = &buf[i..i + 8];
    if v[0] != b'H'
        || v[1] != b'T'
        || v[2] != b'T'
        || v[3] != b'P'
        || v[4] != b'/'
        || v[5] != b'1'
        || v[6] != b'.'
        || (v[7] != b'1' && v[7] != b'0')
    {
        return (Status::Bad, i);
    }
    i += 8;

    // CRLF
    if i + 2 > n {
        return (Status::Partial, i);
    }
    if buf[i] != b'\r' || buf[i + 1] != b'\n' {
        return (Status::Bad, i);
    }
    i += 2;
    (Status::Ok, i)
}

/// Parse a single header line. On Ok, advances past the trailing CRLF.
fn parse_header(buf: &[u8], mut i: usize) -> (Status, usize) {
    let n = buf.len();

    // name
    let start = i;
    while i < n {
        let b = buf[i];
        if b == b':' {
            break;
        }
        if !is_tchar(b) {
            return (Status::Bad, i);
        }
        i += 1;
    }
    if i == start || i >= n {
        return (Status::Partial, i);
    }
    // colon
    i += 1;
    // OWS
    while i < n && is_ows(buf[i]) {
        i += 1;
    }
    // value — vchar or OWS internal, terminated by CRLF.
    while i < n {
        let b = buf[i];
        if b == b'\r' {
            break;
        }
        if !is_vchar(b) && !is_ows(b) {
            return (Status::Bad, i);
        }
        i += 1;
    }
    if i + 2 > n {
        return (Status::Partial, i);
    }
    if buf[i] != b'\r' || buf[i + 1] != b'\n' {
        return (Status::Bad, i);
    }
    i += 2;
    (Status::Ok, i)
}

/// Walk the full message and return the offset where the body starts, or 0
/// on any parse error / partial buffer.
fn parse_message(buf: &[u8]) -> u32 {
    let n = buf.len();
    if n == 0 {
        return 0;
    }
    let (st, mut i) = parse_request_line(buf, 0);
    if st != Status::Ok {
        return 0;
    }

    // Headers — bounded loop to keep the wasm small but realistic.
    let mut headers_seen = 0usize;
    loop {
        if i + 2 <= n && buf[i] == b'\r' && buf[i + 1] == b'\n' {
            i += 2;
            return i as u32;
        }
        let (st, j) = parse_header(buf, i);
        match st {
            Status::Ok => {
                i = j;
                headers_seen += 1;
                if headers_seen > 128 {
                    return 0;
                }
            }
            _ => return 0,
        }
    }
}

/// Public C ABI entry point: returns the byte offset where the body begins
/// in the buffer pointed to by `ptr`, or 0 on parse failure / partial input.
#[no_mangle]
pub unsafe extern "C" fn parse(ptr: *const u8, len: usize) -> u32 {
    if ptr.is_null() || len == 0 {
        return 0;
    }
    let buf = core::slice::from_raw_parts(ptr, len);
    parse_message(buf)
}

/// A second, slightly different exported entry point so the wasm has more
/// than one externally-visible function (helps LOOM exercise its export
/// table handling and produces a larger code section).
#[no_mangle]
pub unsafe extern "C" fn parse_strict(ptr: *const u8, len: usize) -> u32 {
    if ptr.is_null() || len < 16 {
        return 0;
    }
    let buf = core::slice::from_raw_parts(ptr, len);
    // Reject a leading SP / CRLF aggressively in the "strict" variant.
    if buf[0] == b' ' || buf[0] == b'\r' || buf[0] == b'\n' {
        return 0;
    }
    parse_message(buf)
}
