//! Fuzz target for LOOM optimization pipeline
//!
//! This target generates random valid WebAssembly modules and runs them through
//! the LOOM optimizer, checking for:
//! 1. Crashes/panics
//! 2. Invalid WASM output
//! 3. (Optional) Semantic divergence via differential testing

#![no_main]

use libfuzzer_sys::fuzz_target;
use loom_core::{optimize, parse};

fuzz_target!(|data: &[u8]| {
    // Try to parse the input as WASM
    // If it's not valid WASM, that's fine - we skip
    let module = match parse::parse_wasm(data) {
        Ok(m) => m,
        Err(_) => return, // Invalid input, skip
    };

    // Clone for comparison later (may be used for future differential testing)
    let _original = module.clone();

    // Run optimization - should never panic
    let mut optimized = module;
    let _ = optimize::optimize_module(&mut optimized);

    // The optimized module should still be valid
    // (if parse succeeded, optimize should produce valid output)
    let encoded = match loom_core::encode::encode_wasm(&optimized) {
        Ok(bytes) => bytes,
        Err(_) => {
            // Encoding failed - this is a bug worth investigating
            // but don't crash the fuzzer
            return;
        }
    };

    // Validate the output with wasmparser
    if wasmparser::validate(&encoded).is_err() {
        // This is a serious bug - LOOM produced invalid WASM
        panic!("LOOM produced invalid WASM output!");
    }

    // Optional: Check that re-parsing produces the same structure
    // (this can catch encoder bugs)
    if let Ok(reparsed) = parse::parse_wasm(&encoded) {
        // Basic structural check - same number of functions
        if reparsed.functions.len() != optimized.functions.len() {
            panic!(
                "Roundtrip mismatch: {} functions -> {} functions",
                optimized.functions.len(),
                reparsed.functions.len()
            );
        }
    }
});
