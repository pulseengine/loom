//! Fuzz target for WASM parse/encode roundtrip
//!
//! This target checks that the parser and encoder correctly handle arbitrary
//! WebAssembly input. It verifies:
//! 1. Parse -> Encode produces valid WASM
//! 2. Parse -> Encode -> Parse produces structurally equivalent module

#![no_main]

use libfuzzer_sys::fuzz_target;
use loom_core::{encode, parse};

fuzz_target!(|data: &[u8]| {
    // Try to parse the input as WASM
    let module = match parse::parse_wasm(data) {
        Ok(m) => m,
        Err(_) => return, // Invalid input, skip
    };

    // Encode the parsed module
    let encoded = match encode::encode_wasm(&module) {
        Ok(bytes) => bytes,
        Err(_) => {
            // Encoding failed - check if this is expected
            // Some modules might have features we don't support encoding
            return;
        }
    };

    // The encoded output should be valid WASM
    if wasmparser::validate(&encoded).is_err() {
        panic!("Parser -> Encoder produced invalid WASM!");
    }

    // Re-parse the encoded output
    let reparsed = match parse::parse_wasm(&encoded) {
        Ok(m) => m,
        Err(e) => {
            // We encoded it, wasmparser validated it, but we can't parse it?
            // This suggests a bug in our parser or encoder
            panic!("Failed to re-parse encoded WASM: {}", e);
        }
    };

    // Check structural equivalence
    if module.functions.len() != reparsed.functions.len() {
        panic!(
            "Function count mismatch: {} -> {} -> {}",
            module.functions.len(),
            encoded.len(),
            reparsed.functions.len()
        );
    }

    if module.globals.len() != reparsed.globals.len() {
        panic!(
            "Global count mismatch: {} -> {}",
            module.globals.len(),
            reparsed.globals.len()
        );
    }

    if module.memories.len() != reparsed.memories.len() {
        panic!(
            "Memory count mismatch: {} -> {}",
            module.memories.len(),
            reparsed.memories.len()
        );
    }

    // Check function signatures match
    for (i, (orig, repr)) in module
        .functions
        .iter()
        .zip(reparsed.functions.iter())
        .enumerate()
    {
        if orig.signature.params.len() != repr.signature.params.len() {
            panic!(
                "Function {} param count mismatch: {} -> {}",
                i,
                orig.signature.params.len(),
                repr.signature.params.len()
            );
        }
        if orig.signature.results.len() != repr.signature.results.len() {
            panic!(
                "Function {} result count mismatch: {} -> {}",
                i,
                orig.signature.results.len(),
                repr.signature.results.len()
            );
        }
    }
});
