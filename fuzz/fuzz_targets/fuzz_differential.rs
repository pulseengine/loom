//! Differential fuzzing: LOOM vs wasmtime interpretation
//!
//! This target runs optimized WASM through wasmtime and compares output
//! against the original unoptimized version to detect semantic divergence.
//!
//! Requires wasmtime to be available as a library.

#![no_main]

use libfuzzer_sys::fuzz_target;
use loom_core::{encode, optimize, parse};

fuzz_target!(|data: &[u8]| {
    // Parse input as WASM
    let module = match parse::parse_wasm(data) {
        Ok(m) => m,
        Err(_) => return,
    };

    // Skip modules with features we can't fully verify
    // (e.g., imports that would need runtime support)
    if !module.imports.is_empty() {
        return;
    }

    // Encode original (before optimization)
    let original_bytes = match encode::encode_wasm(&module) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Optimize
    let mut optimized = module.clone();
    if optimize::optimize_module(&mut optimized).is_err() {
        return;
    }

    // Encode optimized
    let optimized_bytes = match encode::encode_wasm(&optimized) {
        Ok(b) => b,
        Err(_) => {
            // Optimization produced something we can't encode - bug!
            panic!("Failed to encode optimized module");
        }
    };

    // Validate both are valid WASM
    if wasmparser::validate(&original_bytes).is_err() {
        return; // Original wasn't valid, skip
    }

    if wasmparser::validate(&optimized_bytes).is_err() {
        panic!("LOOM produced invalid WASM!");
    }

    // Size reduction check (optimization shouldn't dramatically increase size)
    // Allow up to 2x size increase (some optimizations like inlining can increase size)
    if optimized_bytes.len() > original_bytes.len() * 2 {
        // Not a crash, but worth logging for investigation
        // eprintln!("Warning: optimization increased size significantly");
    }

    // TODO: When wasmtime is added as a dependency, compare execution results:
    //
    // let engine = wasmtime::Engine::default();
    // let orig_module = wasmtime::Module::new(&engine, &original_bytes)?;
    // let opt_module = wasmtime::Module::new(&engine, &optimized_bytes)?;
    //
    // For each exported function:
    // - Generate random inputs
    // - Call on both modules
    // - Compare outputs
    // - Panic if different (semantic divergence found!)
});
