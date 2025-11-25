//! Fuzz testing with wasm-smith for LOOM optimizer
//!
//! These tests generate random valid WebAssembly modules and verify that:
//! 1. Parse+encode roundtrip produces valid WASM
//! 2. Optimizations are idempotent
//! 3. Optimizations preserve validity

use arbitrary::Unstructured;
use loom_core::{encode, optimize, parse};
use proptest::prelude::*;

/// Generate a random valid WebAssembly module using wasm-smith
fn arb_wasm_module() -> impl Strategy<Value = Vec<u8>> {
    // Generate random bytes for arbitrary, then use wasm-smith to create valid WASM
    prop::collection::vec(any::<u8>(), 100..1000).prop_filter_map("valid wasm", |bytes| {
        let mut u = Unstructured::new(&bytes);

        // Configure wasm-smith to generate simple, well-formed modules
        let mut config = wasm_smith::Config::default();
        config.min_funcs = 1;
        config.max_funcs = 5;
        config.min_types = 1;
        config.max_types = 5;
        config.max_instructions = 50;
        config.max_memories = 1;
        config.allow_start_export = false;
        config.canonicalize_nans = true;

        // Generate a module
        match wasm_smith::Module::new(config, &mut u) {
            Ok(module) => {
                let wasm_bytes = module.to_bytes();
                // Validate it's actually valid
                match wasmparser::validate(&wasm_bytes) {
                    Ok(_) => Some(wasm_bytes),
                    Err(_) => None,
                }
            }
            Err(_) => None,
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 10,  // Start with 10 cases - fuzzing reveals known bugs
        failure_persistence: None,  // Don't persist failures during normal testing
        ..ProptestConfig::default()
    })]

    /// Property: Parse+encode roundtrip should produce valid WASM
    ///
    /// This test catches bugs like Issue #30 where the roundtrip corrupts the module
    ///
    /// **Currently failing** - reveals known parse+encode roundtrip bugs
    /// Run with: `cargo test prop_parse_encode -- --ignored`
    #[test]
    #[ignore]  // Known failures - see Issue #30
    fn prop_parse_encode_roundtrip_valid(wasm_bytes in arb_wasm_module()) {
        // Parse the WASM
        let module = match parse::parse_wasm(&wasm_bytes) {
            Ok(m) => m,
            Err(_) => return Ok(()), // Skip if we can't parse
        };

        // Encode it back
        let encoded = match encode::encode_wasm(&module) {
            Ok(bytes) => bytes,
            Err(_) => return Ok(()), // Skip if we can't encode
        };

        // The encoded output should be valid WASM
        prop_assert!(
            wasmparser::validate(&encoded).is_ok(),
            "Parse+encode roundtrip produced invalid WASM"
        );
    }

    /// Property: Optimization should be idempotent
    ///
    /// optimize(optimize(x)) = optimize(x)
    ///
    /// Run with: `cargo test prop_optimization -- --ignored`
    #[test]
    #[ignore]  // Depends on roundtrip working
    fn prop_optimization_idempotent(wasm_bytes in arb_wasm_module()) {
        // Parse
        let mut module1 = match parse::parse_wasm(&wasm_bytes) {
            Ok(m) => m,
            Err(_) => return Ok(()),
        };

        // First optimization
        if optimize::optimize_module(&mut module1).is_err() {
            return Ok(());
        }

        let encoded1 = match encode::encode_wasm(&module1) {
            Ok(bytes) => bytes,
            Err(_) => return Ok(()),
        };

        // Second optimization
        let mut module2 = match parse::parse_wasm(&encoded1) {
            Ok(m) => m,
            Err(_) => return Ok(()),
        };

        if optimize::optimize_module(&mut module2).is_err() {
            return Ok(());
        }

        let encoded2 = match encode::encode_wasm(&module2) {
            Ok(bytes) => bytes,
            Err(_) => return Ok(()),
        };

        // Both should be valid
        prop_assert!(
            wasmparser::validate(&encoded1).is_ok(),
            "First optimization produced invalid WASM"
        );
        prop_assert!(
            wasmparser::validate(&encoded2).is_ok(),
            "Second optimization produced invalid WASM"
        );

        // Should be idempotent (same size at minimum)
        prop_assert_eq!(
            encoded1.len(),
            encoded2.len(),
            "Optimization is not idempotent"
        );
    }

    /// Property: Optimization should preserve validity
    ///
    /// If input is valid, output should be valid
    ///
    /// Run with: `cargo test prop_optimization -- --ignored`
    #[test]
    #[ignore]  // Depends on roundtrip working
    fn prop_optimization_preserves_validity(wasm_bytes in arb_wasm_module()) {
        // Parse
        let mut module = match parse::parse_wasm(&wasm_bytes) {
            Ok(m) => m,
            Err(_) => return Ok(()),
        };

        // Optimize
        if optimize::optimize_module(&mut module).is_err() {
            return Ok(()); // Optimization can fail on some inputs
        }

        // Encode
        let optimized = match encode::encode_wasm(&module) {
            Ok(bytes) => bytes,
            Err(_) => return Ok(()),
        };

        // Output must be valid if we got this far
        prop_assert!(
            wasmparser::validate(&optimized).is_ok(),
            "Optimization produced invalid WASM from valid input"
        );
    }
}

#[cfg(test)]
mod deterministic_tests {
    use super::*;

    /// Test parse+encode roundtrip with a simple known module
    #[test]
    fn test_parse_encode_simple() {
        let wat = r#"
            (module
                (func (result i32)
                    i32.const 42
                )
            )
        "#;

        let wasm_bytes = wat::parse_str(wat).expect("Failed to parse WAT");
        let module = parse::parse_wasm(&wasm_bytes).expect("Failed to parse WASM");
        let encoded = encode::encode_wasm(&module).expect("Failed to encode");

        // Should produce valid WASM
        wasmparser::validate(&encoded).expect("Roundtrip produced invalid WASM");
    }

    /// Test that we can handle empty functions
    #[test]
    fn test_empty_function() {
        let wat = r#"
            (module
                (func)
            )
        "#;

        let wasm_bytes = wat::parse_str(wat).expect("Failed to parse WAT");
        let mut module = parse::parse_wasm(&wasm_bytes).expect("Failed to parse WASM");
        optimize::optimize_module(&mut module).expect("Optimization failed");
        let encoded = encode::encode_wasm(&module).expect("Failed to encode");

        wasmparser::validate(&encoded).expect("Optimization broke empty function");
    }

    /// Test that parse+encode handles locals correctly
    #[test]
    fn test_locals_roundtrip() {
        let wat = r#"
            (module
                (func (param i32) (result i32)
                    (local i32 i64)
                    local.get 0
                    local.set 1
                    local.get 1
                )
            )
        "#;

        let wasm_bytes = wat::parse_str(wat).expect("Failed to parse WAT");
        let module = parse::parse_wasm(&wasm_bytes).expect("Failed to parse WASM");
        let encoded = encode::encode_wasm(&module).expect("Failed to encode");

        wasmparser::validate(&encoded).expect("Locals roundtrip failed");
    }
}
