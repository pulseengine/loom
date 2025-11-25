//! Verification Tests for LOOM Optimizations
//!
//! Phase 5: Property-Based Verification
//!
//! This module uses property-based testing to verify the correctness of
//! optimization rules. While not as rigorous as SMT-based verification (Crocus),
//! property testing provides strong confidence by testing thousands of random cases.
//!
//! Future Work: Integrate Crocus for full SMT verification using Z3 solver.

use loom_core::{encode, optimize, parse};
use loom_core::{Function, FunctionSignature, Instruction, Module, ValueType};
use loom_isle::{iadd32, iconst32, simplify, Imm32, ValueData};
use proptest::prelude::*;

/// Property: Constant folding preserves WebAssembly semantics
///
/// For any two i32 constants x and y:
///   eval(i32.add(i32.const(x), i32.const(y))) = eval(i32.const(x + y))
///
/// This is the fundamental correctness property for constant folding.
#[test]
fn prop_constant_folding_correctness() {
    let config = ProptestConfig::with_cases(256);
    proptest!(config, |(x: i32, y: i32)| {
        // Build the unoptimized term: I32Add(I32Const(x), I32Const(y))
        let term = iadd32(iconst32(Imm32::from(x)), iconst32(Imm32::from(y)));

        // Apply optimization
        let optimized = simplify(term);

        // The result should be I32Const(x + y) with wrapping
        let expected = x.wrapping_add(y);

        match optimized.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), expected,
                    "Constant folding failed: {} + {} should equal {}",
                    x, y, expected);
            }
            _ => {
                panic!("Optimization should produce I32Const");
            }
        }
    });
}

/// Property: Optimization is idempotent
///
/// For any value v:
///   simplify(simplify(v)) = simplify(v)
///
/// Applying optimization twice should give the same result as applying it once.
#[test]
fn prop_optimization_idempotent() {
    let config = ProptestConfig::with_cases(256);
    proptest!(config, |(x: i32, y: i32)| {
        let term = iadd32(iconst32(Imm32::from(x)), iconst32(Imm32::from(y)));

        let optimized_once = simplify(term.clone());
        let optimized_twice = simplify(optimized_once.clone());

        assert_eq!(optimized_once, optimized_twice,
            "Optimization should be idempotent");
    });
}

/// Property: Constants are unchanged by optimization
///
/// For any constant c:
///   simplify(I32Const(c)) = I32Const(c)
///
/// Constants are already in simplest form.
#[test]
fn prop_constants_unchanged() {
    let config = ProptestConfig::with_cases(256);
    proptest!(config, |(c: i32)| {
        let term = iconst32(Imm32::from(c));
        let optimized = simplify(term.clone());

        assert_eq!(term, optimized,
            "Constants should be unchanged by optimization");
    });
}

/// Property: Round-trip through optimization preserves executability
///
/// For any module M:
///   encode(parse(encode(optimize(M)))) should succeed
///
/// Optimized code should produce valid WebAssembly.
#[test]
fn prop_optimization_produces_valid_wasm() {
    let config = ProptestConfig::with_cases(256);
    proptest!(config, |(x: i32, y: i32)| {
        // Create a module with i32.add constant folding opportunity
        let mut module = Module {
            functions: vec![Function {
                name: None,
                signature: FunctionSignature {
                    params: vec![],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![
                    Instruction::I32Const(x),
                    Instruction::I32Const(y),
                    Instruction::I32Add,
                    Instruction::End,
                ],
            }],
            memories: vec![],
            tables: vec![],
            globals: vec![],
            types: vec![],
            exports: vec![],
        };

        // Apply optimization
        optimize::optimize_module(&mut module).expect("Optimization failed");

        // Encode to WASM
        let wasm_bytes = encode::encode_wasm(&module).expect("Encoding failed");

        // Parse back
        let module2 = parse::parse_wasm(&wasm_bytes).expect("Re-parsing failed");

        // Should have valid structure
        assert_eq!(module2.functions.len(), 1);
    });
}

/// Property: Overflow behavior matches WebAssembly spec
///
/// Specifically test boundary cases:
/// - i32::MAX + 1 = i32::MIN
/// - i32::MIN + (-1) = i32::MAX
/// - 0 + x = x
#[test]
fn test_overflow_boundary_cases() {
    // Test case 1: i32::MAX + 1 = i32::MIN
    let term1 = iadd32(iconst32(Imm32::from(i32::MAX)), iconst32(Imm32::from(1)));
    let result1 = simplify(term1);
    match result1.data() {
        ValueData::I32Const { val } => {
            assert_eq!(val.value(), i32::MIN);
        }
        _ => panic!("Expected I32Const"),
    }

    // Test case 2: i32::MIN + (-1) = i32::MAX
    let term2 = iadd32(iconst32(Imm32::from(i32::MIN)), iconst32(Imm32::from(-1)));
    let result2 = simplify(term2);
    match result2.data() {
        ValueData::I32Const { val } => {
            assert_eq!(val.value(), i32::MAX);
        }
        _ => panic!("Expected I32Const"),
    }

    // Test case 3: 0 + x = x (identity)
    for x in [0, 1, -1, 42, i32::MAX, i32::MIN] {
        let term = iadd32(iconst32(Imm32::from(0)), iconst32(Imm32::from(x)));
        let result = simplify(term);
        match result.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), x);
            }
            _ => panic!("Expected I32Const"),
        }
    }
}

/// Property: Nested operations are fully optimized
///
/// For any x, y, z:
///   simplify(I32Add(I32Const(x), I32Add(I32Const(y), I32Const(z))))
///     = I32Const(x + y + z)
#[test]
fn prop_nested_optimization() {
    let config = ProptestConfig::with_cases(256);
    proptest!(config, |(x: i32, y: i32, z: i32)| {
        // Create nested expression: x + (y + z)
        let inner = iadd32(iconst32(Imm32::from(y)), iconst32(Imm32::from(z)));
        let outer = iadd32(iconst32(Imm32::from(x)), inner);

        let result = simplify(outer);

        let expected = x.wrapping_add(y).wrapping_add(z);

        match result.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), expected,
                    "Nested folding failed: {} + ({} + {}) should equal {}",
                    x, y, z, expected);
            }
            _ => {
                panic!("Nested optimization should produce I32Const");
            }
        }
    });
}

/// Verification Summary Test
///
/// Runs a comprehensive suite of test cases to build confidence in correctness.
#[test]
fn test_verification_summary() {
    println!("\n=== LOOM Verification Summary ===\n");

    let test_cases = vec![
        (10, 32, 42, "Basic addition"),
        (0, 0, 0, "Identity (0+0)"),
        (i32::MAX, 1, i32::MIN, "Overflow (MAX+1)"),
        (i32::MIN, -1, i32::MAX, "Underflow (MIN-1)"),
        (-5, 5, 0, "Cancellation"),
        (100, -50, 50, "Mixed signs"),
    ];

    for (x, y, expected, description) in test_cases {
        let term = iadd32(iconst32(Imm32::from(x)), iconst32(Imm32::from(y)));
        let result = simplify(term);

        match result.data() {
            ValueData::I32Const { val } => {
                assert_eq!(
                    val.value(),
                    expected,
                    "Failed for {}: {} + {} should be {}, got {}",
                    description,
                    x,
                    y,
                    expected,
                    val.value()
                );
                println!("✓ {}: {} + {} = {} ✓", description, x, y, expected);
            }
            _ => panic!("Expected I32Const for {}", description),
        }
    }

    println!("\n=== All Verification Tests Passed ===\n");
}
