//! Helpers shared by the loom-cli integration tests.

/// `wasm` with its element section (id 9) and its custom sections removed.
/// Every other section's contents are copied byte for byte under a fresh
/// header, and the result is checked with wasmparser.
pub fn without_element_section(wasm: &[u8]) -> Vec<u8> {
    let mut out = wasm[..8].to_vec();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        let payload = payload.expect("parse fixture");
        let Some((id, range)) = payload.as_section() else {
            continue;
        };
        if id == 9 || id == 0 {
            continue;
        }
        let contents = &wasm[range.start as usize..range.end as usize];
        out.push(id);
        let mut len = contents.len();
        loop {
            let byte = (len & 0x7f) as u8;
            len >>= 7;
            if len == 0 {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
        out.extend_from_slice(contents);
    }
    wasmparser::Validator::new()
        .validate_all(&out)
        .expect("the element-free fixture must validate");
    out
}
