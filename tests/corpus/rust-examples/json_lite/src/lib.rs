//! Minimal JSON tokenizer — emits a stream of single-byte token tags counted
//! and returned. No allocations; the tokenizer is a pure forward scan over
//! a `&[u8]`. Strings and numbers are validated but not materialized.
//!
//! This is intentionally **not** a JSON parser. The fixture's role is to
//! exercise the LOOM optimizer on a realistic-shaped Rust → wasm body with
//! a state machine over an ASCII byte stream.

#![no_std]
#![allow(clippy::missing_safety_doc)]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Token tags (each fits in a u8, returned to a caller via memory if they
/// want; the public entry point only returns the *count*).
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq)]
enum Tag {
    ObjectStart = 1,
    ObjectEnd = 2,
    ArrayStart = 3,
    ArrayEnd = 4,
    Comma = 5,
    Colon = 6,
    String = 7,
    Number = 8,
    True = 9,
    False = 10,
    Null = 11,
    Eof = 12,
}

#[inline(always)]
fn is_ws(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\n' || b == b'\r'
}

#[inline(always)]
fn is_digit(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}

#[inline(always)]
fn is_hex(b: u8) -> bool {
    is_digit(b)
        || (b >= b'a' && b <= b'f')
        || (b >= b'A' && b <= b'F')
}

/// Advance past whitespace.
#[inline]
fn skip_ws(buf: &[u8], mut i: usize) -> usize {
    while i < buf.len() && is_ws(buf[i]) {
        i += 1;
    }
    i
}

/// Skip a JSON string (after opening `"`). Returns offset after the closing
/// `"`, or `None` on error.
fn skip_string(buf: &[u8], mut i: usize) -> Option<usize> {
    while i < buf.len() {
        let b = buf[i];
        if b == b'"' {
            return Some(i + 1);
        }
        if b == b'\\' {
            if i + 1 >= buf.len() {
                return None;
            }
            let esc = buf[i + 1];
            match esc {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    i += 2;
                }
                b'u' => {
                    if i + 6 > buf.len() {
                        return None;
                    }
                    if !is_hex(buf[i + 2])
                        || !is_hex(buf[i + 3])
                        || !is_hex(buf[i + 4])
                        || !is_hex(buf[i + 5])
                    {
                        return None;
                    }
                    i += 6;
                }
                _ => return None,
            }
        } else if b < 0x20 {
            return None;
        } else {
            i += 1;
        }
    }
    None
}

/// Skip a JSON number. Returns offset just past the last digit.
fn skip_number(buf: &[u8], mut i: usize) -> Option<usize> {
    if i >= buf.len() {
        return None;
    }
    if buf[i] == b'-' {
        i += 1;
    }
    let int_start = i;
    while i < buf.len() && is_digit(buf[i]) {
        i += 1;
    }
    if i == int_start {
        return None;
    }
    if i < buf.len() && buf[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < buf.len() && is_digit(buf[i]) {
            i += 1;
        }
        if i == frac_start {
            return None;
        }
    }
    if i < buf.len() && (buf[i] == b'e' || buf[i] == b'E') {
        i += 1;
        if i < buf.len() && (buf[i] == b'+' || buf[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < buf.len() && is_digit(buf[i]) {
            i += 1;
        }
        if i == exp_start {
            return None;
        }
    }
    Some(i)
}

/// Match a literal keyword (`true`, `false`, `null`). Returns offset just
/// after the keyword on match.
fn match_keyword(buf: &[u8], i: usize, kw: &[u8]) -> Option<usize> {
    if i + kw.len() > buf.len() {
        return None;
    }
    let mut k = 0;
    while k < kw.len() {
        if buf[i + k] != kw[k] {
            return None;
        }
        k += 1;
    }
    Some(i + kw.len())
}

/// Scan the full document and return the token count. On parse error, the
/// count seen so far is still returned (so callers can do partial-parse
/// diagnostics) but with the top bit set to mark error.
fn tokenize_inner(buf: &[u8]) -> u32 {
    let mut count: u32 = 0;
    let mut i = 0usize;
    while i < buf.len() {
        i = skip_ws(buf, i);
        if i >= buf.len() {
            break;
        }
        let b = buf[i];
        let (tag, ni) = match b {
            b'{' => (Tag::ObjectStart, Some(i + 1)),
            b'}' => (Tag::ObjectEnd, Some(i + 1)),
            b'[' => (Tag::ArrayStart, Some(i + 1)),
            b']' => (Tag::ArrayEnd, Some(i + 1)),
            b',' => (Tag::Comma, Some(i + 1)),
            b':' => (Tag::Colon, Some(i + 1)),
            b'"' => (Tag::String, skip_string(buf, i + 1)),
            b't' => (Tag::True, match_keyword(buf, i, b"true")),
            b'f' => (Tag::False, match_keyword(buf, i, b"false")),
            b'n' => (Tag::Null, match_keyword(buf, i, b"null")),
            b'-' | b'0'..=b'9' => (Tag::Number, skip_number(buf, i)),
            _ => (Tag::Eof, None),
        };
        match ni {
            Some(ni) => {
                // Touch the tag value so the compiler keeps the branches.
                let _ = tag as u8;
                count = count.saturating_add(1);
                i = ni;
            }
            None => return count | 0x8000_0000,
        }
    }
    count
}

#[no_mangle]
pub unsafe extern "C" fn tokenize(ptr: *const u8, len: usize) -> u32 {
    if ptr.is_null() || len == 0 {
        return 0;
    }
    let buf = core::slice::from_raw_parts(ptr, len);
    tokenize_inner(buf)
}

/// A second, slightly different entry point: counts only string and number
/// tokens. Helps keep multiple branches alive after optimization.
#[no_mangle]
pub unsafe extern "C" fn count_values(ptr: *const u8, len: usize) -> u32 {
    if ptr.is_null() || len == 0 {
        return 0;
    }
    let buf = core::slice::from_raw_parts(ptr, len);
    let mut count: u32 = 0;
    let mut i = 0usize;
    while i < buf.len() {
        i = skip_ws(buf, i);
        if i >= buf.len() {
            break;
        }
        let b = buf[i];
        let ni = match b {
            b'{' | b'}' | b'[' | b']' | b',' | b':' => Some(i + 1),
            b'"' => {
                let r = skip_string(buf, i + 1);
                if r.is_some() {
                    count = count.saturating_add(1);
                }
                r
            }
            b't' => match_keyword(buf, i, b"true"),
            b'f' => match_keyword(buf, i, b"false"),
            b'n' => match_keyword(buf, i, b"null"),
            b'-' | b'0'..=b'9' => {
                let r = skip_number(buf, i);
                if r.is_some() {
                    count = count.saturating_add(1);
                }
                r
            }
            _ => None,
        };
        match ni {
            Some(ni) => i = ni,
            None => return count,
        }
    }
    count
}
