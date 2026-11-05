//! Differential fuzzing: LOOM vs wasmtime interpretation
//!
//! This target runs optimized WASM through wasmtime and compares output
//! against the original unoptimized version to detect semantic divergence.
//!
//! The comparison itself lives in `fuzz/src/lib.rs` (`loom_fuzz::compare_modules`)
//! so it can be unit-tested. Build with `--features negative-control` to inject
//! a deliberate fault after optimization; the target must then crash.

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

    // Test-only fault injection (negative control for the oracle, #474).
    #[cfg(feature = "negative-control")]
    for f in &mut optimized.functions {
        for i in &mut f.instructions {
            if let loom_core::Instruction::I32Const(v) = i {
                *v = v.wrapping_add(1);
            }
        }
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

    // Execute both under wasmtime (bounded by fuel) and compare return
    // values, trap-vs-no-trap, and exported memory/globals.
    // Seed argument generation from the input so runs are reproducible.
    let seed = data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
    });
    if let loom_fuzz::Verdict::Mismatch(why) =
        loom_fuzz::compare_modules(&original_bytes, &optimized_bytes, seed)
    {
        panic!("semantic divergence between original and optimized module: {why}");
    }
});
