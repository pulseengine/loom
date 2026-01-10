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
            imports: vec![],
            data_segments: vec![],
            element_section_bytes: None,
            start_function: None,
            custom_sections: vec![],
            type_section_bytes: None,
            global_section_bytes: None,
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

// ============================================================================
// Z3 Translation Validation Tests
// ============================================================================

/// Test Z3 verification of constant folding
///
/// Demonstrates that Z3 can prove: i32.const 2 + i32.const 3 ≡ i32.const 5
#[test]
#[cfg(feature = "verification")]
fn test_z3_constant_folding_verification() {
    use loom_core::verify::verify_function_equivalence;

    // Original: 2 + 3
    let original = Function {
        name: Some("test_add".to_string()),
        signature: FunctionSignature {
            params: vec![],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            Instruction::I32Const(2),
            Instruction::I32Const(3),
            Instruction::I32Add,
            Instruction::End,
        ],
    };

    // Optimized: 5
    let optimized = Function {
        name: Some("test_add".to_string()),
        signature: FunctionSignature {
            params: vec![],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![Instruction::I32Const(5), Instruction::End],
    };

    // Z3 should prove these are equivalent
    let result = verify_function_equivalence(&original, &optimized, "test");
    assert!(result.is_ok(), "Verification should succeed");
    assert!(result.unwrap(), "Functions should be proven equivalent");
}

/// Test Z3 verification of dead code elimination
///
/// Demonstrates that Z3 can prove removing dead code is semantically equivalent
#[test]
#[cfg(feature = "verification")]
fn test_z3_dead_code_verification() {
    use loom_core::verify::verify_function_equivalence;

    // Original: const 42, const 99, drop (dead code)
    let original = Function {
        name: Some("test_dce".to_string()),
        signature: FunctionSignature {
            params: vec![],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            Instruction::I32Const(42),
            Instruction::I32Const(99),
            Instruction::Drop, // Dead code pattern
            Instruction::End,
        ],
    };

    // Optimized: just const 42
    let optimized = Function {
        name: Some("test_dce".to_string()),
        signature: FunctionSignature {
            params: vec![],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![Instruction::I32Const(42), Instruction::End],
    };

    let result = verify_function_equivalence(&original, &optimized, "test_dce");
    assert!(result.is_ok(), "Verification should succeed");
    assert!(result.unwrap(), "DCE should be proven equivalent");
}

/// Test Z3 verification with parameters
///
/// Demonstrates Z3 verifies parametric functions correctly
#[test]
#[cfg(feature = "verification")]
fn test_z3_parametric_verification() {
    use loom_core::verify::verify_function_equivalence;

    // Original: x + 0
    let original = Function {
        name: Some("test_param".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            Instruction::LocalGet(0),
            Instruction::I32Const(0),
            Instruction::I32Add,
            Instruction::End,
        ],
    };

    // Optimized: x (identity - adding zero is no-op)
    let optimized = Function {
        name: Some("test_param".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![Instruction::LocalGet(0), Instruction::End],
    };

    let result = verify_function_equivalence(&original, &optimized, "test_param");
    assert!(result.is_ok(), "Verification should succeed");
    assert!(result.unwrap(), "x + 0 = x should be proven");
}

/// Test Z3 catches semantic differences
///
/// Demonstrates Z3 can detect when an "optimization" is incorrect
#[test]
#[cfg(feature = "verification")]
fn test_z3_catches_bad_optimization() {
    use loom_core::verify::verify_function_equivalence;

    // Original: x + 1
    let original = Function {
        name: Some("test_bad".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            Instruction::LocalGet(0),
            Instruction::I32Const(1),
            Instruction::I32Add,
            Instruction::End,
        ],
    };

    // Bad "optimization": just returns x (incorrect!)
    let bad_optimized = Function {
        name: Some("test_bad".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![Instruction::LocalGet(0), Instruction::End],
    };

    let result = verify_function_equivalence(&original, &bad_optimized, "test_bad");
    // This should return Ok(false) - Z3 finds a counterexample
    assert!(result.is_ok(), "Verification should complete");
    assert!(
        !result.unwrap(),
        "Z3 should detect x + 1 ≠ x (bad optimization)"
    );
}

/// Test TranslationValidator RAII pattern
#[test]
#[cfg(feature = "verification")]
fn test_translation_validator_raii() {
    use loom_core::verify::TranslationValidator;

    let mut func = Function {
        name: Some("test_validator".to_string()),
        signature: FunctionSignature {
            params: vec![],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            Instruction::I32Const(10),
            Instruction::I32Const(20),
            Instruction::I32Add,
            Instruction::End,
        ],
    };

    // Capture original state
    let validator = TranslationValidator::new(&func, "test_pass");

    // Simulate optimization: fold 10 + 20 -> 30
    func.instructions = vec![Instruction::I32Const(30), Instruction::End];

    // Verify equivalence
    let result = validator.verify(&func);
    assert!(result.is_ok(), "Validator should prove equivalence");
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

/// Test VerificationCoverage tracking
#[test]
fn test_verification_coverage_tracking() {
    use loom_core::verify::{VerificationCoverage, VerificationResult};

    let mut coverage = VerificationCoverage::new();

    // Initially empty
    assert_eq!(coverage.total(), 0);
    assert_eq!(coverage.coverage_percent(), 100.0); // 0/0 = 100%

    // Add some verified functions
    VerificationResult::Verified.update_coverage(&mut coverage);
    VerificationResult::Verified.update_coverage(&mut coverage);
    VerificationResult::Verified.update_coverage(&mut coverage);

    assert_eq!(coverage.verified, 3);
    assert_eq!(coverage.coverage_percent(), 100.0); // 3/3 = 100%

    // Add some skipped functions
    VerificationResult::SkippedLoop.update_coverage(&mut coverage);
    VerificationResult::SkippedMemory.update_coverage(&mut coverage);

    assert_eq!(coverage.skipped_loops, 1);
    assert_eq!(coverage.skipped_memory, 1);
    assert_eq!(coverage.total(), 5);
    // 3 verified / (3 verified + 2 skipped) = 60%
    assert!((coverage.coverage_percent() - 60.0).abs() < 0.1);

    // Test summary output
    let summary = coverage.summary();
    assert!(summary.contains("3/5 functions"));
    assert!(summary.contains("60.0% Z3-proven"));

    println!("Coverage tracking test passed: {}", summary);
}

/// Test that VerificationResult correctly identifies equivalence
#[test]
fn test_verification_result_equivalence() {
    use loom_core::verify::VerificationResult;

    // These should all be considered "equivalent" (optimization accepted)
    assert!(VerificationResult::Verified.is_equivalent());
    assert!(VerificationResult::SkippedLoop.is_equivalent());
    assert!(VerificationResult::SkippedMemory.is_equivalent());
    assert!(VerificationResult::SkippedUnknown.is_equivalent());

    // Only Verified should be "fully verified"
    assert!(VerificationResult::Verified.is_verified());
    assert!(!VerificationResult::SkippedLoop.is_verified());
    assert!(!VerificationResult::SkippedMemory.is_verified());

    // Failed/Error should not be equivalent
    assert!(!VerificationResult::Failed("test".to_string()).is_equivalent());
    assert!(!VerificationResult::Error("test".to_string()).is_equivalent());
}

/// Test module-level coverage computation
#[test]
#[cfg(feature = "verification")]
fn test_compute_verification_coverage() {
    use loom_core::verify::compute_verification_coverage;

    // Create a simple module with one function that can be verified
    let original = Module {
        functions: vec![Function {
            name: Some("simple".to_string()),
            signature: FunctionSignature {
                params: vec![],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![
                Instruction::I32Const(10),
                Instruction::I32Const(20),
                Instruction::I32Add,
                Instruction::End,
            ],
        }],
        memories: vec![],
        tables: vec![],
        globals: vec![],
        types: vec![],
        exports: vec![],
        imports: vec![],
        data_segments: vec![],
        element_section_bytes: None,
        start_function: None,
        custom_sections: vec![],
        type_section_bytes: None,
        global_section_bytes: None,
    };

    // Optimized version
    let optimized = Module {
        functions: vec![Function {
            name: Some("simple".to_string()),
            signature: FunctionSignature {
                params: vec![],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::I32Const(30), Instruction::End],
        }],
        ..original.clone()
    };

    let coverage = compute_verification_coverage(&original, &optimized, "test");

    // Should have 1 verified function
    assert_eq!(coverage.verified, 1);
    assert_eq!(coverage.total(), 1);
    assert_eq!(coverage.coverage_percent(), 100.0);

    println!("Module coverage: {}", coverage.summary());
}

/// Test that memory load/store operations are now verified with Z3 Array theory
///
/// This test verifies that functions using I32Load/I32Store are NO LONGER skipped
/// and are properly verified using the Z3 Array memory model.
#[test]
#[cfg(feature = "verification")]
fn test_memory_load_store_verification() {
    use loom_core::verify::{
        compute_verification_coverage, verify_function_equivalence_with_result,
    };

    // Function with I32Load - should now be verified (not skipped)
    let func_with_load = Function {
        name: Some("load_test".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32], // address parameter
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            Instruction::LocalGet(0), // get address
            Instruction::I32Load {
                align: 2,
                offset: 0,
            },
            Instruction::End,
        ],
    };

    // Verify the function with itself (identity transform should be equivalent)
    let result =
        verify_function_equivalence_with_result(&func_with_load, &func_with_load, "memory_test");

    // Should be Verified (not SkippedMemory) since we now have Z3 Array theory
    assert!(
        result.is_verified(),
        "Memory load functions should now be verified, got: {:?}",
        result
    );

    // Function with I32Store followed by I32Load (store-load forwarding pattern)
    let func_store_load = Function {
        name: Some("store_load_test".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32, ValueType::I32], // address, value
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            Instruction::LocalGet(0), // address
            Instruction::LocalGet(1), // value
            Instruction::I32Store {
                align: 2,
                offset: 0,
            },
            Instruction::LocalGet(0), // address again
            Instruction::I32Load {
                align: 2,
                offset: 0,
            },
            Instruction::End,
        ],
    };

    let result2 = verify_function_equivalence_with_result(
        &func_store_load,
        &func_store_load,
        "store_load_test",
    );

    assert!(
        result2.is_verified(),
        "Store-load function should be verified, got: {:?}",
        result2
    );

    // Test full module coverage
    let module = Module {
        functions: vec![func_with_load.clone(), func_store_load.clone()],
        memories: vec![loom_core::Memory {
            min: 1,
            max: None,
            shared: false,
            memory64: false,
        }],
        tables: vec![],
        globals: vec![],
        types: vec![],
        exports: vec![],
        imports: vec![],
        data_segments: vec![],
        element_section_bytes: None,
        start_function: None,
        custom_sections: vec![],
        type_section_bytes: None,
        global_section_bytes: None,
    };

    let coverage = compute_verification_coverage(&module, &module, "memory_module_test");

    // Both functions should be verified (not skipped)
    assert_eq!(
        coverage.verified,
        2,
        "Both memory functions should be verified, got coverage: {}",
        coverage.summary()
    );
    assert_eq!(
        coverage.skipped_memory, 0,
        "No functions should be skipped for memory anymore"
    );

    println!(
        "Memory verification test passed with coverage: {}",
        coverage.summary()
    );
}

/// Test that simple loops (no nesting, no unverifiable ops) are now verified
///
/// This tests the bounded loop unrolling verification for simple loops.
#[test]
#[cfg(feature = "verification")]
fn test_simple_loop_verification() {
    use loom_core::verify::verify_function_equivalence_with_result;

    // Simple loop: just increments a counter
    let func_with_simple_loop = Function {
        name: Some("simple_loop".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32], // initial counter value
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            // Simple loop that adds 1 three times
            Instruction::Block {
                block_type: loom_core::BlockType::Empty,
                body: vec![Instruction::Loop {
                    block_type: loom_core::BlockType::Empty,
                    body: vec![
                        Instruction::LocalGet(0),
                        Instruction::I32Const(1),
                        Instruction::I32Add,
                        Instruction::LocalSet(0),
                        // Exit after one iteration (simplified - real loop would have condition)
                        Instruction::Br(1), // Break out of block
                    ],
                }],
            },
            Instruction::LocalGet(0),
            Instruction::End,
        ],
    };

    // Verify the function with itself (identity should be equivalent)
    let result = verify_function_equivalence_with_result(
        &func_with_simple_loop,
        &func_with_simple_loop,
        "simple_loop_test",
    );

    // Simple loops should now be verified (not skipped)
    assert!(
        result.is_verified(),
        "Simple loop functions should now be verified, got: {:?}",
        result
    );

    println!("Simple loop verification test passed: {:?}", result);
}

// ============================================================================
// Function Summary Tests
// ============================================================================

/// Test function summary analysis - pure function
#[test]
#[cfg(feature = "verification")]
fn test_function_summary_pure() {
    use loom_core::verify::FunctionSummary;

    // A pure function: just adds two parameters
    let pure_func = Function {
        name: Some("pure_add".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32, ValueType::I32],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            Instruction::LocalGet(0),
            Instruction::LocalGet(1),
            Instruction::I32Add,
            Instruction::End,
        ],
    };

    let summary = FunctionSummary::analyze(&pure_func);
    assert!(summary.is_pure(), "Function should be pure (no side effects)");
    assert!(!summary.has_side_effects(), "Function should have no side effects");
    assert!(summary.globals_read.is_empty(), "Function should not read globals");
    assert!(summary.globals_written.is_empty(), "Function should not write globals");
    assert!(!summary.reads_memory, "Function should not read memory");
    assert!(!summary.writes_memory, "Function should not write memory");
    assert!(!summary.has_calls, "Function should not have calls");
}

/// Test function summary analysis - function with global write
#[test]
#[cfg(feature = "verification")]
fn test_function_summary_global_write() {
    use loom_core::verify::FunctionSummary;

    // Function that writes to a global
    let impure_func = Function {
        name: Some("write_global".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![],
        },
        locals: vec![],
        instructions: vec![
            Instruction::LocalGet(0),
            Instruction::GlobalSet(0), // Write to global 0
            Instruction::End,
        ],
    };

    let summary = FunctionSummary::analyze(&impure_func);
    assert!(!summary.is_pure(), "Function should not be pure (writes global)");
    assert!(summary.has_side_effects(), "Function should have side effects");
    assert!(summary.globals_written.contains(&0), "Function should write global 0");
    assert!(!summary.writes_memory, "Function should not write memory");
}

/// Test function summary analysis - function with memory write
#[test]
#[cfg(feature = "verification")]
fn test_function_summary_memory_write() {
    use loom_core::verify::FunctionSummary;

    // Function that writes to memory
    let store_func = Function {
        name: Some("store_memory".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32, ValueType::I32], // addr, value
            results: vec![],
        },
        locals: vec![],
        instructions: vec![
            Instruction::LocalGet(0), // address
            Instruction::LocalGet(1), // value
            Instruction::I32Store { align: 2, offset: 0 },
            Instruction::End,
        ],
    };

    let summary = FunctionSummary::analyze(&store_func);
    assert!(!summary.is_pure(), "Function should not be pure (writes memory)");
    assert!(summary.has_side_effects(), "Function should have side effects");
    assert!(summary.writes_memory, "Function should write memory");
}

/// Test function summary analysis - function with calls
#[test]
#[cfg(feature = "verification")]
fn test_function_summary_with_calls() {
    use loom_core::verify::FunctionSummary;

    // Function that calls another function
    let calling_func = Function {
        name: Some("caller".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            Instruction::LocalGet(0),
            Instruction::Call(1), // Call function index 1
            Instruction::End,
        ],
    };

    let summary = FunctionSummary::analyze(&calling_func);
    assert!(summary.has_calls, "Function should have calls");
    assert!(summary.called_functions.contains(&1), "Function should call function 1");
    assert!(!summary.has_indirect_calls, "Function should not have indirect calls");
}

/// Test function summary analysis - function with indirect calls
#[test]
#[cfg(feature = "verification")]
fn test_function_summary_with_indirect_calls() {
    use loom_core::verify::FunctionSummary;

    // Function with indirect call
    let indirect_call_func = Function {
        name: Some("indirect_caller".to_string()),
        signature: FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        },
        locals: vec![],
        instructions: vec![
            Instruction::LocalGet(0), // table index
            Instruction::CallIndirect {
                type_idx: 0,
                table_idx: 0,
            },
            Instruction::End,
        ],
    };

    let summary = FunctionSummary::analyze(&indirect_call_func);
    assert!(summary.has_calls, "Function should have calls");
    assert!(summary.has_indirect_calls, "Function should have indirect calls");
}

/// Test building function summaries from a module
#[test]
#[cfg(feature = "verification")]
fn test_build_function_summaries() {
    use loom_core::verify::build_function_summaries;

    let module = Module {
        functions: vec![
            // Function 0: pure
            Function {
                name: Some("pure".to_string()),
                signature: FunctionSignature {
                    params: vec![],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![Instruction::I32Const(42), Instruction::End],
            },
            // Function 1: reads global
            Function {
                name: Some("read_global".to_string()),
                signature: FunctionSignature {
                    params: vec![],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![Instruction::GlobalGet(0), Instruction::End],
            },
            // Function 2: writes global
            Function {
                name: Some("write_global".to_string()),
                signature: FunctionSignature {
                    params: vec![ValueType::I32],
                    results: vec![],
                },
                locals: vec![],
                instructions: vec![
                    Instruction::LocalGet(0),
                    Instruction::GlobalSet(0),
                    Instruction::End,
                ],
            },
        ],
        types: vec![],
        imports: vec![],
        exports: vec![],
        tables: vec![],
        memories: vec![],
        globals: vec![],
        start_function: None,
        element_section_bytes: None,
        data_segments: vec![],
        custom_sections: vec![],
        type_section_bytes: None,
        global_section_bytes: None,
    };

    let summaries = build_function_summaries(&module);

    // Function 0 should be pure
    assert!(summaries.get(&0).unwrap().is_pure(), "Function 0 should be pure");

    // Function 1 reads global but doesn't write
    let summary1 = summaries.get(&1).unwrap();
    assert!(summary1.globals_read.contains(&0), "Function 1 should read global 0");
    assert!(summary1.is_pure(), "Function 1 should be pure (only reads)");

    // Function 2 writes global
    let summary2 = summaries.get(&2).unwrap();
    assert!(summary2.globals_written.contains(&0), "Function 2 should write global 0");
    assert!(!summary2.is_pure(), "Function 2 should not be pure");
}
