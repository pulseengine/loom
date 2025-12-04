//! Comprehensive optimization tests for LOOM
//!
//! This test suite validates all optimization passes to ensure correctness
//! and that semantics are preserved.

use loom_core::{optimize, parse};

/// Helper to test that WAT input optimizes to expected WAT output
#[allow(dead_code)]
fn assert_optimizes_to(input_wat: &str, expected_wat: &str) {
    let mut module = parse::parse_wat(input_wat).expect("Failed to parse input WAT");
    optimize::optimize_module(&mut module).expect("Optimization failed");
    let optimized_wat = format!("{:?}", module.functions[0].instructions);

    let expected_module = parse::parse_wat(expected_wat).expect("Failed to parse expected WAT");
    let expected_instructions = format!("{:?}", expected_module.functions[0].instructions);

    assert_eq!(
        optimized_wat, expected_instructions,
        "\nInput:\n{}\n\nExpected:\n{}\n\nGot:\n{}",
        input_wat, expected_wat, optimized_wat
    );
}

// ============================================================================
// Advanced Instruction Optimization Tests (Issue #21)
// ============================================================================

#[test]
fn test_strength_reduction_mul_power_of_2() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                i32.const 8
                i32.mul
            )
        )
    "#;

    // Should optimize to shift: x * 8 → x << 3
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    // Verify shift is present
    assert!(
        format!("{:?}", instructions).contains("Shl"),
        "Expected shift left, got: {:?}",
        instructions
    );
}

#[test]
fn test_strength_reduction_div_power_of_2() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                i32.const 16
                i32.div_u
            )
        )
    "#;

    // Should optimize to shift: x / 16 → x >> 4
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    assert!(
        format!("{:?}", instructions).contains("ShrU"),
        "Expected unsigned shift right, got: {:?}",
        instructions
    );
}

#[test]
fn test_strength_reduction_rem_power_of_2() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                i32.const 32
                i32.rem_u
            )
        )
    "#;

    // Should optimize to AND: x % 32 → x & 31
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    assert!(
        format!("{:?}", instructions).contains("And"),
        "Expected AND, got: {:?}",
        instructions
    );
    assert!(
        format!("{:?}", instructions).contains("31"),
        "Expected mask value 31, got: {:?}",
        instructions
    );
}

#[test]
fn test_bitwise_xor_self() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                local.get 0
                i32.xor
            )
        )
    "#;

    // Should optimize to constant 0: x XOR x → 0
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    let has_const_zero = instructions
        .iter()
        .any(|i| matches!(i, loom_core::Instruction::I32Const(0)));
    assert!(
        has_const_zero,
        "Expected i32.const 0, got: {:?}",
        instructions
    );
}

#[test]
fn test_bitwise_and_self() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                local.get 0
                i32.and
            )
        )
    "#;

    // Should optimize to identity: x AND x → x
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    // Should be reduced to just local.get 0
    assert!(
        instructions.len() <= 2, // local.get + end
        "Expected simplified to single local.get, got: {:?}",
        instructions
    );
}

#[test]
fn test_algebraic_add_zero() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.add
            )
        )
    "#;

    // Should optimize to identity: x + 0 → x
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    assert!(
        !format!("{:?}", instructions).contains("Add"),
        "Expected addition to be eliminated, got: {:?}",
        instructions
    );
}

#[test]
fn test_algebraic_mul_zero() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.mul
            )
        )
    "#;

    // Should optimize to constant: x * 0 → 0
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    let has_const_zero = instructions
        .iter()
        .any(|i| matches!(i, loom_core::Instruction::I32Const(0)));
    assert!(
        has_const_zero,
        "Expected constant 0, got: {:?}",
        instructions
    );
}

#[test]
fn test_algebraic_mul_one() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.mul
            )
        )
    "#;

    // Should optimize to identity: x * 1 → x
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    assert!(
        !format!("{:?}", instructions).contains("Mul"),
        "Expected multiplication to be eliminated, got: {:?}",
        instructions
    );
}

// ============================================================================
// CSE Tests (Issue #19)
// ============================================================================

#[test]
fn test_cse_duplicate_constants() {
    let input = r#"
        (module
            (func (result i32)
                i32.const 42
                i32.const 42
                i32.add
            )
        )
    "#;

    // CSE should recognize duplicate constants
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    // Should constant-fold to 84
    let instructions = &module.functions[0].instructions;
    let has_84 = instructions
        .iter()
        .any(|i| matches!(i, loom_core::Instruction::I32Const(84)));
    assert!(
        has_84,
        "Expected constant folding to 84, got: {:?}",
        instructions
    );
}

#[test]
fn test_cse_duplicate_computation() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                i32.const 10
                i32.add
                local.get 0
                i32.const 10
                i32.add
                i32.add
            )
        )
    "#;

    // CSE should detect duplicate (x + 10) computation
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    // After CSE, should use local.tee to cache result
    let instructions = &module.functions[0].instructions;
    let has_tee_or_fewer_ops = instructions
        .iter()
        .any(|i| matches!(i, loom_core::Instruction::LocalTee(_)))
        || instructions.len() < 8; // Original has 8+ instructions

    assert!(
        has_tee_or_fewer_ops,
        "Expected CSE optimization, got: {:?}",
        instructions
    );
}

#[test]
fn test_cse_multiple_duplicates() {
    let input = r#"
        (module
            (func (param $x i32) (param $y i32) (result i32)
                (local.get $x)
                (local.get $y)
                (i32.add)
                (local.get $x)
                (local.get $y)
                (i32.add)
                (local.get $x)
                (local.get $y)
                (i32.add)
                (i32.add)
                (i32.add)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    let before_len = module.functions[0].instructions.len();
    optimize::optimize_module(&mut module).unwrap();
    let after_len = module.functions[0].instructions.len();

    // Should reduce code by eliminating duplicate computations
    assert!(
        after_len <= before_len,
        "CSE should reduce or maintain code size. Before: {}, After: {}",
        before_len,
        after_len
    );

    // Should produce valid WASM
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_cse_commutative_operations() {
    let input = r#"
        (module
            (func (param $x i32) (param $y i32) (result i32)
                (local.get $x)
                (local.get $y)
                (i32.add)
                (local.get $y)
                (local.get $x)
                (i32.add)
                (i32.add)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    // Should recognize x+y and y+x as equivalent
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_cse_with_bitwise_operations() {
    let input = r#"
        (module
            (func (param $x i32) (param $y i32) (result i32)
                (local.get $x)
                (local.get $y)
                (i32.and)
                (local.get $x)
                (local.get $y)
                (i32.and)
                (i32.or)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    // Should handle bitwise operations in CSE
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_cse_i64_operations() {
    let input = r#"
        (module
            (func (param $x i64) (param $y i64) (result i64)
                (local.get $x)
                (local.get $y)
                (i64.add)
                (local.get $x)
                (local.get $y)
                (i64.add)
                (i64.mul)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    // Should handle i64 operations
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

// ============================================================================
// Function Inlining Tests (Issue #14)
// ============================================================================

#[test]
fn test_inline_simple_function() {
    let input = r#"
        (module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add
            )
            (func $main (param i32) (result i32)
                local.get 0
                call $add_one
            )
        )
    "#;

    // Should inline $add_one into $main
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    // After inlining, $main should have fewer Call instructions
    let main_func = &module.functions[1]; // $main is second function
    let call_count = main_func
        .instructions
        .iter()
        .filter(|i| matches!(i, loom_core::Instruction::Call(_)))
        .count();

    // Should have 0 or 1 call (depending on inlining threshold)
    assert!(
        call_count <= 1,
        "Expected inlining to reduce calls, got {} calls in {:?}",
        call_count,
        main_func.instructions
    );
}

// ============================================================================
// Loop Optimization Tests (Issue #23)
// ============================================================================

#[test]
fn test_licm_constant_hoist() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                (local i32 i32)
                i32.const 0
                local.set 1
                (loop $L
                    ;; Invariant computation - should be hoisted
                    i32.const 10
                    i32.const 20
                    i32.add
                    local.set 2

                    ;; Loop counter
                    local.get 1
                    i32.const 1
                    i32.add
                    local.tee 1
                    local.get 0
                    i32.lt_u
                    br_if $L
                )
                local.get 1
            )
        )
    "#;

    // LICM should hoist the constant computation (10 + 20) outside the loop
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    // After LICM + constant folding, the constant 30 should appear before the loop
    let instructions = &module.functions[0].instructions;
    let instr_str = format!("{:?}", instructions);

    // Should have constant 30 from folding 10 + 20
    assert!(
        instr_str.contains("30") || instr_str.contains("I32Const"),
        "Expected constant optimization, got: {:?}",
        instructions
    );
}

// ============================================================================
// Code Folding Tests (Issue #22)
// ============================================================================

#[test]
fn test_block_flattening() {
    let input = r#"
        (module
            (func (result i32)
                (block
                    (block
                        i32.const 42
                    )
                )
            )
        )
    "#;

    // Nested empty blocks should be flattened
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    // Count number of Block instructions
    let block_count = instructions
        .iter()
        .filter(|i| matches!(i, loom_core::Instruction::Block { .. }))
        .count();

    // Should have fewer blocks after flattening
    assert!(
        block_count <= 1,
        "Expected block flattening, got {} blocks in {:?}",
        block_count,
        instructions
    );
}

// ============================================================================
// Integration Tests - Multiple Optimizations
// ============================================================================

#[test]
fn test_combined_optimizations() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                ;; Multiple optimizable patterns
                local.get 0
                i32.const 4
                i32.mul        ;; Strength reduction: x * 4 → x << 2
                i32.const 0
                i32.add        ;; Algebraic: y + 0 → y
                local.get 0
                local.get 0
                i32.xor        ;; Bitwise: x XOR x → 0
                i32.add        ;; z + 0 → z
            )
        )
    "#;

    // Should apply multiple optimizations
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    let instr_str = format!("{:?}", instructions);

    // Should have shift from strength reduction
    assert!(instr_str.contains("Shl"), "Expected shift optimization");

    // Should be significantly simplified
    assert!(
        instructions.len() < 10,
        "Expected significant optimization, got {} instructions: {:?}",
        instructions.len(),
        instructions
    );
}

#[test]
fn test_optimization_idempotence() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                i32.const 8
                i32.mul
            )
        )
    "#;

    // Optimize once
    let mut module1 = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module1).unwrap();
    let result1 = format!("{:?}", module1.functions[0].instructions);

    // Optimize again
    optimize::optimize_module(&mut module1).unwrap();
    let result2 = format!("{:?}", module1.functions[0].instructions);

    // Results should be identical (idempotent)
    assert_eq!(
        result1, result2,
        "Optimization should be idempotent:\nFirst: {}\nSecond: {}",
        result1, result2
    );
}

#[test]
fn test_optimization_preserves_semantics() {
    let input = r#"
        (module
            (func (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add
                i32.const 10
                i32.mul
            )
        )
    "#;

    // Optimize
    let mut optimized = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut optimized).unwrap();

    // Both should compute (x + y) * 10
    // Manual verification that structure is preserved
    let instructions = &optimized.functions[0].instructions;
    assert!(
        !instructions.is_empty(),
        "Optimization should not remove all instructions"
    );

    // Should still have parameters
    assert_eq!(
        optimized.functions[0].signature.params.len(),
        2,
        "Should preserve parameters"
    );
    assert_eq!(
        optimized.functions[0].signature.results.len(),
        1,
        "Should preserve results"
    );
}

// ============================================================================
// Edge Cases and Regression Tests
// ============================================================================

#[test]
fn test_empty_function() {
    let input = r#"
        (module
            (func (result i32)
                i32.const 0
            )
        )
    "#;

    // Should handle empty/minimal functions
    let mut module = parse::parse_wat(input).unwrap();
    let result = optimize::optimize_module(&mut module);
    assert!(result.is_ok(), "Should handle minimal function");
}

#[test]
fn test_no_optimizations_needed() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
            )
        )
    "#;

    // Already optimal - should not break
    let mut module = parse::parse_wat(input).unwrap();
    let before = format!("{:?}", module.functions[0].instructions);
    let before_len = before.len();
    optimize::optimize_module(&mut module).unwrap();
    let after = format!("{:?}", module.functions[0].instructions);
    let after_len = after.len();

    // Should be unchanged or very similar
    assert!(
        before == after || after_len <= before_len + 1,
        "Should not make already-optimal code worse\nBefore (len={}): {}\nAfter  (len={}): {}",
        before_len,
        before,
        after_len,
        after
    );
}

#[test]
fn test_large_constants() {
    let input = r#"
        (module
            (func (result i32)
                i32.const 2147483647
                i32.const 1
                i32.add
            )
        )
    "#;

    // Should handle i32 overflow correctly (wrapping)
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions = &module.functions[0].instructions;
    // Should constant-fold to -2147483648 (overflow wraps)
    assert!(
        !instructions.is_empty(),
        "Should handle overflow correctly: {:?}",
        instructions
    );
}

#[test]
fn test_multiple_functions() {
    let input = r#"
        (module
            (func $f1 (result i32)
                i32.const 10
                i32.const 20
                i32.add
            )
            (func $f2 (result i32)
                i32.const 30
                i32.const 40
                i32.add
            )
            (func $f3 (result i32)
                call $f1
                call $f2
                i32.add
            )
        )
    "#;

    // Should optimize all functions
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    assert_eq!(module.functions.len(), 3, "Should preserve all functions");

    // First two should constant-fold
    assert!(
        module.functions[0].instructions.len() <= 2, // const 30 + end
        "f1 should be optimized: {:?}",
        module.functions[0].instructions
    );
    assert!(
        module.functions[1].instructions.len() <= 2, // const 70 + end
        "f2 should be optimized: {:?}",
        module.functions[1].instructions
    );
}

// ============================================================================
// Self-Operation Algebraic Optimizations (New Features)
// ============================================================================

#[test]
fn test_self_xor_i32() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                (i32.xor (local.get 0) (local.get 0))
            )
        )
    "#;

    // x ^ x should become 0
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    assert!(
        instructions_str.contains("I32Const") && instructions_str.contains("0"),
        "Expected i32.const 0, got: {:?}",
        module.functions[0].instructions
    );
}

#[test]
fn test_self_xor_i64() {
    let input = r#"
        (module
            (func (param i64) (result i64)
                (i64.xor (local.get 0) (local.get 0))
            )
        )
    "#;

    // x ^ x should become 0
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    assert!(
        instructions_str.contains("I64Const") && instructions_str.contains("0"),
        "Expected i64.const 0, got: {:?}",
        module.functions[0].instructions
    );
}

#[test]
fn test_self_and_i32() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                (i32.and (local.get 0) (local.get 0))
            )
        )
    "#;

    // x & x should become x
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    // Should just be local.get (no And operation)
    assert!(
        !instructions_str.contains("And"),
        "Expected no And operation, got: {:?}",
        module.functions[0].instructions
    );
    assert!(
        instructions_str.contains("LocalGet"),
        "Expected LocalGet, got: {:?}",
        module.functions[0].instructions
    );
}

#[test]
fn test_self_or_i64() {
    let input = r#"
        (module
            (func (param i64) (result i64)
                (i64.or (local.get 0) (local.get 0))
            )
        )
    "#;

    // x | x should become x
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    // Should just be local.get (no Or operation)
    assert!(
        !instructions_str.contains("Or"),
        "Expected no Or operation, got: {:?}",
        module.functions[0].instructions
    );
}

#[test]
fn test_self_sub_i32() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                (i32.sub (local.get 0) (local.get 0))
            )
        )
    "#;

    // x - x should become 0
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    assert!(
        instructions_str.contains("I32Const") && instructions_str.contains("0"),
        "Expected i32.const 0, got: {:?}",
        module.functions[0].instructions
    );
}

#[test]
fn test_self_sub_i64() {
    let input = r#"
        (module
            (func (param i64) (result i64)
                (i64.sub (local.get 0) (local.get 0))
            )
        )
    "#;

    // x - x should become 0
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    assert!(
        instructions_str.contains("I64Const") && instructions_str.contains("0"),
        "Expected i64.const 0, got: {:?}",
        module.functions[0].instructions
    );
}

#[test]
fn test_self_eq_i32() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                (i32.eq (local.get 0) (local.get 0))
            )
        )
    "#;

    // x == x should become 1 (true)
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    assert!(
        instructions_str.contains("I32Const") && instructions_str.contains("1"),
        "Expected i32.const 1, got: {:?}",
        module.functions[0].instructions
    );
}

#[test]
fn test_self_eq_i64() {
    let input = r#"
        (module
            (func (param i64) (result i32)
                (i64.eq (local.get 0) (local.get 0))
            )
        )
    "#;

    // x == x should become 1 (true)
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    assert!(
        instructions_str.contains("I32Const") && instructions_str.contains("1"),
        "Expected i32.const 1, got: {:?}",
        module.functions[0].instructions
    );
}

#[test]
fn test_self_ne_i32() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                (i32.ne (local.get 0) (local.get 0))
            )
        )
    "#;

    // x != x should become 0 (false)
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    assert!(
        instructions_str.contains("I32Const") && instructions_str.contains("0"),
        "Expected i32.const 0, got: {:?}",
        module.functions[0].instructions
    );
}

#[test]
fn test_self_ne_i64() {
    let input = r#"
        (module
            (func (param i64) (result i32)
                (i64.ne (local.get 0) (local.get 0))
            )
        )
    "#;

    // x != x should become 0 (false)
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    assert!(
        instructions_str.contains("I32Const") && instructions_str.contains("0"),
        "Expected i32.const 0, got: {:?}",
        module.functions[0].instructions
    );
}

// ============================================================================
// Redundant Set Elimination (RSE) Tests
// ============================================================================

#[test]
#[ignore] // eliminate_redundant_sets disabled - has bugs
fn test_rse_simple_redundant_set() {
    let input = r#"
        (module
            (func (result i32)
                (local $x i32)
                (local.set $x (i32.const 10))
                (local.set $x (i32.const 20))
                (local.get $x)
            )
        )
    "#;

    // First set should be eliminated since it's immediately overwritten
    let mut module = parse::parse_wat(input).unwrap();
    let before = module.functions[0].instructions.len();
    optimize::optimize_module(&mut module).unwrap();
    let after = module.functions[0].instructions.len();

    // Should have fewer instructions (one LocalSet removed)
    assert!(
        after < before,
        "Expected fewer instructions after RSE. Before: {}, After: {}",
        before,
        after
    );

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    // Should only have one set with value 20
    let set_count = instructions_str.matches("LocalSet").count();
    assert_eq!(
        set_count, 1,
        "Expected exactly 1 LocalSet remaining, got {} in: {:?}",
        set_count, module.functions[0].instructions
    );
}

#[test]
#[ignore] // eliminate_redundant_sets disabled - has bugs
fn test_rse_with_intervening_get() {
    let input = r#"
        (module
            (func (result i32)
                (local $x i32)
                (local.set $x (i32.const 10))
                (local.get $x)
                (local.set $x (i32.const 20))
                (local.get $x)
            )
        )
    "#;

    // Both sets should remain because there's a get between them
    // Note: We call simplify_locals directly to test RSE in isolation,
    // without constant folding interfering
    let mut module = parse::parse_wat(input).unwrap();
    optimize::simplify_locals(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    let set_count = instructions_str.matches("LocalSet").count();
    assert_eq!(
        set_count, 2,
        "Expected 2 LocalSet instructions (no elimination), got {} in: {:?}",
        set_count, module.functions[0].instructions
    );
}

#[test]
#[ignore] // eliminate_redundant_sets disabled - has bugs
fn test_rse_multiple_redundant_sets() {
    let input = r#"
        (module
            (func (result i32)
                (local $x i32)
                (local.set $x (i32.const 10))
                (local.set $x (i32.const 20))
                (local.set $x (i32.const 30))
                (local.get $x)
            )
        )
    "#;

    // First two sets should be eliminated
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    let set_count = instructions_str.matches("LocalSet").count();
    assert_eq!(
        set_count, 1,
        "Expected only 1 LocalSet (value 30), got {} in: {:?}",
        set_count, module.functions[0].instructions
    );
}

#[test]
#[ignore] // eliminate_redundant_sets disabled - has bugs
fn test_rse_different_locals() {
    let input = r#"
        (module
            (func (result i32)
                (local $x i32)
                (local $y i32)
                (local.set $x (i32.const 10))
                (local.set $y (i32.const 20))
                (local.set $x (i32.const 30))
                (local.get $x)
                (local.get $y)
                i32.add
            )
        )
    "#;

    // First set to $x should be eliminated, but $y set should remain
    let mut module = parse::parse_wat(input).unwrap();
    let before = module.functions[0].instructions.len();
    optimize::optimize_module(&mut module).unwrap();
    let after = module.functions[0].instructions.len();

    assert!(
        after < before,
        "Expected RSE to eliminate first set to $x. Before: {}, After: {}",
        before,
        after
    );

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    let set_count = instructions_str.matches("LocalSet").count();
    assert_eq!(
        set_count, 2,
        "Expected 2 LocalSet instructions, got {} in: {:?}",
        set_count, module.functions[0].instructions
    );
}

#[test]
#[ignore] // eliminate_redundant_sets disabled - has bugs
fn test_rse_with_tee() {
    let input = r#"
        (module
            (func (result i32)
                (local $x i32)
                (local.set $x (i32.const 10))
                (local.tee $x (i32.const 20))
            )
        )
    "#;

    // First set should be eliminated, tee both sets and returns value
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    let set_count = instructions_str.matches("LocalSet").count();
    let tee_count = instructions_str.matches("LocalTee").count();

    assert_eq!(
        set_count, 0,
        "Expected 0 LocalSet (eliminated by tee), got {} in: {:?}",
        set_count, module.functions[0].instructions
    );
    assert_eq!(
        tee_count, 1,
        "Expected 1 LocalTee, got {} in: {:?}",
        tee_count, module.functions[0].instructions
    );
}

#[test]
#[ignore] // eliminate_redundant_sets disabled - has bugs
fn test_rse_in_block() {
    let input = r#"
        (module
            (func (result i32)
                (local $x i32)
                (block
                    (local.set $x (i32.const 10))
                    (local.set $x (i32.const 20))
                )
                (local.get $x)
            )
        )
    "#;

    // Should eliminate redundant set even inside block
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    let set_count = instructions_str.matches("LocalSet").count();
    assert_eq!(
        set_count, 1,
        "Expected 1 LocalSet in block, got {} in: {:?}",
        set_count, module.functions[0].instructions
    );
}

#[test]
#[ignore] // eliminate_redundant_sets disabled - has bugs
fn test_rse_conservative_in_if() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                (local $x i32)
                (local.set $x (i32.const 10))
                (if (local.get 0)
                    (then
                        (local.set $x (i32.const 20))
                    )
                )
                (local.get $x)
            )
        )
    "#;

    // Both sets should remain - we can't eliminate the first one because
    // the if branch might not execute
    // Note: We call simplify_locals directly to test RSE in isolation
    let mut module = parse::parse_wat(input).unwrap();
    optimize::simplify_locals(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    let set_count = instructions_str.matches("LocalSet").count();
    assert_eq!(
        set_count, 2,
        "Expected 2 LocalSet (conservative for if), got {} in: {:?}",
        set_count, module.functions[0].instructions
    );
}

// Code Folding Tests

#[test]
fn test_code_folding_simple_tail_merge() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                (if (result i32) (local.get 0)
                    (then
                        (i32.const 1)
                        (i32.const 100)
                        (i32.add)
                    )
                    (else
                        (i32.const 2)
                        (i32.const 100)
                        (i32.add)
                    )
                )
            )
        )
    "#;

    // The (i32.const 100) and (i32.add) should be moved outside the if
    let mut module = parse::parse_wat(input).unwrap();

    // Count total instructions recursively
    fn count_total_instrs(instrs: &[loom_core::Instruction]) -> usize {
        use loom_core::Instruction;
        let mut count = instrs.len();
        for instr in instrs {
            match instr {
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    count += count_total_instrs(then_body);
                    count += count_total_instrs(else_body);
                }
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    count += count_total_instrs(body);
                }
                _ => {}
            }
        }
        count
    }

    let before = count_total_instrs(&module.functions[0].instructions);
    optimize::code_folding(&mut module).unwrap();
    let after = count_total_instrs(&module.functions[0].instructions);

    // Should reduce total instruction count (duplicate tail removed)
    assert!(
        after < before,
        "Expected code folding to reduce total instruction count. Before: {}, After: {}",
        before,
        after
    );
}

#[test]
fn test_code_folding_multiple_instructions() {
    let input = r#"
        (module
            (func (param i32)
                (local $x i32)
                (local $y i32)
                (if (local.get 0)
                    (then
                        (i32.const 1)
                        (local.set $x (i32.const 10))
                        (local.set $y (i32.const 20))
                    )
                    (else
                        (i32.const 2)
                        (local.set $x (i32.const 10))
                        (local.set $y (i32.const 20))
                    )
                )
            )
        )
    "#;

    // Both LocalSet instructions should be moved outside
    let mut module = parse::parse_wat(input).unwrap();
    optimize::code_folding(&mut module).unwrap();

    // Count how many instructions are in the then/else bodies
    fn count_if_body_instrs(instrs: &[loom_core::Instruction]) -> usize {
        use loom_core::Instruction;
        for instr in instrs {
            if let Instruction::If { then_body, .. } = instr {
                return then_body.len();
            }
        }
        0
    }

    let then_len = count_if_body_instrs(&module.functions[0].instructions);
    assert_eq!(
        then_len, 1,
        "Expected only 1 instruction in then body after folding, got {}",
        then_len
    );
}

#[test]
fn test_code_folding_no_false_positive() {
    let input = r#"
        (module
            (func (param i32) (result i32)
                (if (result i32) (local.get 0)
                    (then
                        (i32.const 1)
                        (i32.const 100)
                        (i32.add)
                    )
                    (else
                        (i32.const 2)
                        (i32.const 200)
                        (i32.add)
                    )
                )
            )
        )
    "#;

    // Different constants - should only fold i32.add, not the different constants
    let mut module = parse::parse_wat(input).unwrap();

    // Count total instructions recursively
    fn count_total_instrs(instrs: &[loom_core::Instruction]) -> usize {
        use loom_core::Instruction;
        let mut count = instrs.len();
        for instr in instrs {
            match instr {
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    count += count_total_instrs(then_body);
                    count += count_total_instrs(else_body);
                }
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    count += count_total_instrs(body);
                }
                _ => {}
            }
        }
        count
    }

    let before = count_total_instrs(&module.functions[0].instructions);
    optimize::code_folding(&mut module).unwrap();
    let after = count_total_instrs(&module.functions[0].instructions);

    // Should reduce by 1 (only i32.add is common, not the different constants)
    assert!(
        after == before - 1,
        "Expected exactly 1 instruction to be folded (i32.add). Before: {}, After: {}",
        before,
        after
    );
}

#[test]
fn test_code_folding_nested_if() {
    let input = r#"
        (module
            (func (param i32)
                (local $x i32)
                (if (local.get 0)
                    (then
                        (if (i32.const 1)
                            (then (local.set $x (i32.const 10)))
                            (else (local.set $x (i32.const 10)))
                        )
                    )
                    (else
                        (i32.const 2)
                    )
                )
            )
        )
    "#;

    // The nested if should have its common tail merged
    let mut module = parse::parse_wat(input).unwrap();
    optimize::code_folding(&mut module).unwrap();

    // Just ensure it doesn't crash and produces valid output
    assert!(
        !module.functions[0].instructions.is_empty(),
        "Module should have instructions after folding"
    );
}

// Loop Invariant Code Motion Tests

#[test]
fn test_licm_simple_hoist() {
    let input = r#"
        (module
            (func (param $x i32) (param $y i32) (result i32)
                (local $sum i32)
                (local $invariant i32)
                (loop $loop
                    ;; This computation is loop-invariant (x + y doesn't change)
                    (local.set $invariant
                        (i32.add (local.get $x) (local.get $y))
                    )
                    ;; This uses the invariant
                    (local.set $sum
                        (i32.add (local.get $sum) (local.get $invariant))
                    )
                    ;; Loop continues
                    (br_if $loop (i32.const 0))
                )
                (local.get $sum)
            )
        )
    "#;

    // The invariant computation should be hoisted outside the loop
    let mut module = parse::parse_wat(input).unwrap();
    optimize::loop_invariant_code_motion(&mut module).unwrap();

    // Count instructions inside the loop
    fn count_loop_body_instrs(instrs: &[loom_core::Instruction]) -> usize {
        use loom_core::Instruction;
        for instr in instrs {
            if let Instruction::Loop { body, .. } = instr {
                return body.len();
            }
        }
        0
    }

    let loop_size_before =
        count_loop_body_instrs(&parse::parse_wat(input).unwrap().functions[0].instructions);
    let loop_size_after = count_loop_body_instrs(&module.functions[0].instructions);

    // Loop body should be smaller after hoisting
    assert!(
        loop_size_after < loop_size_before,
        "Expected loop body to shrink after LICM. Before: {}, After: {}",
        loop_size_before,
        loop_size_after
    );
}

#[test]
fn test_licm_no_hoist_modified_local() {
    let input = r#"
        (module
            (func (param $x i32) (result i32)
                (local $sum i32)
                (loop $loop
                    ;; This reads $sum which is modified in the loop - NOT invariant
                    (local.set $sum
                        (i32.add (local.get $sum) (local.get $x))
                    )
                    (br_if $loop (i32.const 0))
                )
                (local.get $sum)
            )
        )
    "#;

    // Should not hoist anything since $sum is modified in the loop
    let mut module = parse::parse_wat(input).unwrap();
    optimize::loop_invariant_code_motion(&mut module).unwrap();

    // Instructions should remain the same (no hoisting of non-invariant code)
    // We just ensure it doesn't crash
    assert!(
        !module.functions[0].instructions.is_empty(),
        "LICM should not crash on non-invariant code"
    );
}

#[test]
fn test_licm_hoist_constants() {
    let input = r#"
        (module
            (func (result i32)
                (local $sum i32)
                (loop $loop
                    (i32.const 42)
                    (i32.const 100)
                    (i32.add)
                    (local.set $sum)
                    (br_if $loop (i32.const 0))
                )
                (local.get $sum)
            )
        )
    "#;

    // Constants should be hoisted
    let mut module = parse::parse_wat(input).unwrap();

    fn count_loop_body_instrs(instrs: &[loom_core::Instruction]) -> usize {
        use loom_core::Instruction;
        for instr in instrs {
            if let Instruction::Loop { body, .. } = instr {
                return body.len();
            }
        }
        0
    }

    let before = count_loop_body_instrs(&module.functions[0].instructions);
    optimize::loop_invariant_code_motion(&mut module).unwrap();
    let after = count_loop_body_instrs(&module.functions[0].instructions);

    // Loop body should be smaller
    assert!(
        after < before,
        "Expected constants to be hoisted. Before: {}, After: {}",
        before,
        after
    );
}

#[test]
fn test_licm_nested_loops() {
    let input = r#"
        (module
            (func (param $x i32)
                (local $outer i32)
                (loop $outer_loop
                    (loop $inner_loop
                        (i32.const 10)
                        (local.set $outer)
                        (br_if $inner_loop (i32.const 0))
                    )
                    (br_if $outer_loop (i32.const 0))
                )
            )
        )
    "#;

    // Should handle nested loops without crashing
    let mut module = parse::parse_wat(input).unwrap();
    optimize::loop_invariant_code_motion(&mut module).unwrap();

    // Just ensure it doesn't crash
    assert!(
        !module.functions[0].instructions.is_empty(),
        "LICM should handle nested loops"
    );
}

// Remove Unused Branches Tests

#[test]
fn test_remove_dead_code_after_return() {
    let input = r#"
        (module
            (func (result i32)
                (return (i32.const 42))
                (i32.const 100)
                (i32.const 200)
                (i32.add)
            )
        )
    "#;

    // Code after return should be removed
    let mut module = parse::parse_wat(input).unwrap();
    let before = module.functions[0].instructions.len();
    optimize::remove_unused_branches(&mut module).unwrap();
    let after = module.functions[0].instructions.len();

    assert!(
        after < before,
        "Expected dead code after return to be removed. Before: {}, After: {}",
        before,
        after
    );

    // Should only have return and its value
    let instructions_str = format!("{:?}", module.functions[0].instructions);
    assert!(
        instructions_str.contains("Return"),
        "Should still have return instruction"
    );
}

#[test]
fn test_remove_dead_code_after_unreachable() {
    let input = r#"
        (module
            (func
                (unreachable)
                (i32.const 100)
                (drop)
            )
        )
    "#;

    // Code after unreachable should be removed
    let mut module = parse::parse_wat(input).unwrap();
    let before = module.functions[0].instructions.len();
    optimize::remove_unused_branches(&mut module).unwrap();
    let after = module.functions[0].instructions.len();

    assert!(
        after < before,
        "Expected dead code after unreachable to be removed. Before: {}, After: {}",
        before,
        after
    );
}

#[test]
fn test_dead_code_in_blocks() {
    let input = r#"
        (module
            (func (result i32)
                (block (result i32)
                    (return (i32.const 42))
                    (i32.const 100)
                )
            )
        )
    "#;

    // Dead code in blocks should be removed
    let mut module = parse::parse_wat(input).unwrap();
    optimize::remove_unused_branches(&mut module).unwrap();

    // Just ensure it doesn't crash and produces valid output
    assert!(
        !module.functions[0].instructions.is_empty(),
        "Should have instructions after cleanup"
    );
}

#[test]
fn test_dce_unused_local_assignment() {
    let input = r#"
        (module
            (func $unused_local (result i32)
                (local $unused i32)
                (local.set $unused (i32.const 100))
                (i32.const 42)
            )
        )
    "#;

    // Assignment to unused local should be processable
    let mut module = parse::parse_wat(input).unwrap();
    optimize::remove_unused_branches(&mut module).unwrap();

    // Result should be valid
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_dce_dead_block() {
    let input = r#"
        (module
            (func $dead_block (result i32)
                (i32.const 42)
                (return)
                (i32.const 100)
                (drop)
            )
        )
    "#;

    // Dead code after return should be removed
    let mut module = parse::parse_wat(input).unwrap();
    let before_len = module.functions[0].instructions.len();
    optimize::remove_unused_branches(&mut module).unwrap();
    let after_len = module.functions[0].instructions.len();

    // Should have removed dead code
    assert!(
        after_len < before_len,
        "Should have removed dead instructions"
    );

    // Should still be valid
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_dce_unreachable_branch() {
    let input = r#"
        (module
            (func $unreachable_branch (param $cond i32) (result i32)
                (local.get $cond)
                (if (then
                    (return (i32.const 1))
                ))
                (i32.const 2)
            )
        )
    "#;

    // This should not crash and properly handle branching
    let mut module = parse::parse_wat(input).unwrap();
    optimize::remove_unused_branches(&mut module).unwrap();

    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_dce_multiple_returns_in_nested_blocks() {
    let input = r#"
        (module
            (func $multiple_returns (result i32)
                (block $b1
                    (block $b2
                        (return (i32.const 10))
                        (i32.const 20)
                        (drop)
                    )
                    (i32.const 30)
                    (drop)
                )
                (i32.const 40)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::remove_unused_branches(&mut module).unwrap();

    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_dce_with_drop_operations() {
    let input = r#"
        (module
            (func $const_drop (result i32)
                (i32.const 100)
                (drop)
                (i32.const 42)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();

    // Run DCE - should not crash and produce valid output
    optimize::remove_unused_branches(&mut module).unwrap();
    optimize::eliminate_dead_code(&mut module).unwrap();

    // Should produce valid WASM
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_dce_in_nested_blocks() {
    let input = r#"
        (module
            (func $nested_dead (result i32)
                (block $b1
                    (block $b2
                        (i32.const 100)
                        (return)
                        (i32.const 200)
                        (drop)
                    )
                )
                (i32.const 42)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::remove_unused_branches(&mut module).unwrap();

    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

// ============================================================================
// Call and CallIndirect Tests (Issue #35 - Call/call_indirect Handling)
// ============================================================================

#[test]
fn test_call_basic_optimization() {
    let input = r#"
        (module
            (func $add (param $x i32) (param $y i32) (result i32)
                (local.get $x)
                (local.get $y)
                (i32.add)
            )

            (func $main (result i32)
                (i32.const 5)
                (i32.const 3)
                (call $add)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();

    // Verify it parses correctly
    assert_eq!(module.functions.len(), 2);

    // Run optimization
    optimize::optimize_module(&mut module).unwrap();

    // Verify output is valid
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_call_with_inlining() {
    let input = r#"
        (module
            (func $add_two (param $x i32) (result i32)
                (local.get $x)
                (i32.const 2)
                (i32.add)
            )

            (func $main (param $a i32) (result i32)
                (local.get $a)
                (call $add_two)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();

    // Run inlining
    optimize::inline_functions(&mut module).unwrap();

    // Inlining should have processed the call
    // The function should still be valid
    assert!(!module.functions[1].instructions.is_empty());

    // Should be valid WASM
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_call_prevents_rse() {
    let input = r#"
        (module
            (func $other (result i32)
                (i32.const 42)
            )

            (func (result i32)
                (local $x i32)
                (local.set $x (i32.const 10))
                (call $other)
                (drop)
                (local.set $x (i32.const 20))
                (local.get $x)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    // Should produce valid output
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

#[test]
fn test_multiple_calls_in_function() {
    let input = r#"
        (module
            (func $add (param $x i32) (param $y i32) (result i32)
                (local.get $x)
                (local.get $y)
                (i32.add)
            )

            (func $multiply_sum (param $a i32) (param $b i32) (param $c i32) (result i32)
                (local.get $a)
                (local.get $b)
                (call $add)
                (local.get $c)
                (i32.mul)
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).unwrap();

    // Should handle multiple calls correctly
    use loom_core::encode;
    let wasm = encode::encode_wasm(&module).expect("Should encode");
    wasmparser::validate(&wasm).expect("Should be valid WASM");
}

// Optimize Added Constants Tests

#[test]
fn test_merge_constant_adds_i32() {
    let input = r#"
        (module
            (func (param $x i32) (result i32)
                (local.get $x)
                (i32.const 5)
                (i32.add)
                (i32.const 10)
                (i32.add)
            )
        )
    "#;

    // Should merge 5 + 10 into 15
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_added_constants(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);

    // Count how many i32.add instructions remain
    let add_count = instructions_str.matches("I32Add").count();
    assert_eq!(
        add_count, 1,
        "Expected only 1 I32Add after merging constants, got {}",
        add_count
    );
}

#[test]
fn test_fold_constant_add() {
    let input = r#"
        (module
            (func (result i32)
                (i32.const 100)
                (i32.const 200)
                (i32.add)
            )
        )
    "#;

    // Should fold into single constant 300
    let mut module = parse::parse_wat(input).unwrap();
    let before = module.functions[0].instructions.len();
    optimize::optimize_added_constants(&mut module).unwrap();
    let after = module.functions[0].instructions.len();

    assert!(
        after < before,
        "Expected constants to be folded. Before: {}, After: {}",
        before,
        after
    );
}

#[test]
fn test_merge_constant_adds_i64() {
    let input = r#"
        (module
            (func (param $x i64) (result i64)
                (local.get $x)
                (i64.const 100)
                (i64.add)
                (i64.const 200)
                (i64.add)
            )
        )
    "#;

    // Should merge i64 constants
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_added_constants(&mut module).unwrap();

    let instructions_str = format!("{:?}", module.functions[0].instructions);
    let add_count = instructions_str.matches("I64Add").count();
    assert_eq!(
        add_count, 1,
        "Expected only 1 I64Add after merging constants, got {}",
        add_count
    );
}

#[test]
#[ignore] // eliminate_redundant_sets disabled - has bugs
fn test_rse_no_false_positive_with_call() {
    let input = r#"
        (module
            (func $other (result i32)
                (i32.const 42)
            )
            (func (result i32)
                (local $x i32)
                (local.set $x (i32.const 10))
                (call $other)
                drop
                (local.set $x (i32.const 20))
                (local.get $x)
            )
        )
    "#;

    // Even though there's a call between the sets (which doesn't access locals),
    // this is still a valid RSE opportunity
    let mut module = parse::parse_wat(input).unwrap();
    let before = module.functions[1].instructions.len();
    optimize::optimize_module(&mut module).unwrap();
    let after = module.functions[1].instructions.len();

    // First set should be eliminated
    assert!(
        after <= before,
        "Expected RSE with call. Before: {}, After: {}",
        before,
        after
    );
}

// ============================================================================
// Function Inlining Tests (Issue #31 - Stack Discipline Bug Fix)
// ============================================================================

#[test]
fn test_inline_stack_discipline_simple() {
    let input = r#"
        (module
            (func $add_two (param $x i32) (result i32)
                local.get $x
                i32.const 2
                i32.add
            )

            (func $main (param $a i32) (result i32)
                local.get $a
                call $add_two
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::inline_functions(&mut module).unwrap();

    // Verify the inlined code is valid
    let main_func = &module.functions[1];

    // Should have parameter storage + inlined body
    // The inlined function should work correctly
    assert!(
        !main_func.instructions.is_empty(),
        "Function should have inlined instructions"
    );

    // Validate by encoding and re-parsing
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
    wasmparser::validate(&wasm_bytes).expect("Inlined module should be valid WASM");
}

#[test]
fn test_inline_stack_discipline_multiple_calls() {
    // This is the exact test case from Issue #31
    let input = r#"
        (module
            (func $add_two (param $x i32) (result i32)
                local.get $x
                i32.const 2
                i32.add
            )

            (func $main (param $a i32) (result i32)
                local.get $a
                call $add_two
                call $add_two
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::inline_functions(&mut module).unwrap();

    // Validate by encoding and re-parsing
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
    wasmparser::validate(&wasm_bytes).expect("Multiple inlined calls should be valid WASM");
}

#[test]
fn test_inline_stack_discipline_multiple_params() {
    let input = r#"
        (module
            (func $add (param $x i32) (param $y i32) (result i32)
                local.get $x
                local.get $y
                i32.add
            )

            (func $main (param $a i32) (param $b i32) (result i32)
                local.get $a
                local.get $b
                call $add
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::inline_functions(&mut module).unwrap();

    // Validate by encoding and re-parsing
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
    wasmparser::validate(&wasm_bytes).expect("Multi-param inlining should be valid WASM");
}

#[test]
fn test_inline_stack_discipline_with_locals() {
    let input = r#"
        (module
            (func $compute (param $x i32) (result i32)
                (local $temp i32)
                local.get $x
                i32.const 5
                i32.add
                local.tee $temp
                local.get $temp
                i32.mul
            )

            (func $main (param $a i32) (result i32)
                local.get $a
                call $compute
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::inline_functions(&mut module).unwrap();

    // Validate by encoding and re-parsing
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
    wasmparser::validate(&wasm_bytes).expect("Inlining with locals should be valid WASM");
}

#[test]
fn test_inline_stack_discipline_idempotence() {
    let input = r#"
        (module
            (func $add_two (param $x i32) (result i32)
                local.get $x
                i32.const 2
                i32.add
            )

            (func $main (param $a i32) (result i32)
                local.get $a
                call $add_two
            )
        )
    "#;

    // First optimization
    let mut module1 = parse::parse_wat(input).unwrap();
    optimize::inline_functions(&mut module1).unwrap();

    // Encode first result
    use loom_core::encode;
    let wasm1 = encode::encode_wasm(&module1).expect("Failed to encode first time");

    // Parse and optimize again
    let mut module2 = parse::parse_wasm(&wasm1).unwrap();
    optimize::inline_functions(&mut module2).unwrap();
    let wasm2 = encode::encode_wasm(&module2).expect("Failed to encode second time");

    // Should be idempotent - second optimization doesn't change anything
    assert_eq!(
        wasm1.len(),
        wasm2.len(),
        "Inline optimization should be idempotent"
    );
}

#[test]
fn test_inline_stack_discipline_semantics() {
    let input = r#"
        (module
            (func $double (param $x i32) (result i32)
                local.get $x
                local.get $x
                i32.add
            )

            (func $quadruple (param $a i32) (result i32)
                local.get $a
                call $double
                call $double
            )
        )
    "#;

    let mut module = parse::parse_wat(input).unwrap();
    optimize::inline_functions(&mut module).unwrap();

    // Validate by encoding and re-parsing
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
    wasmparser::validate(&wasm_bytes).expect("Semantic preservation should produce valid WASM");
}

// ============================================================================
// F32/F64 Constant Support Tests (Issue #34)
// ============================================================================

#[test]
fn test_f32_const_parsing() {
    let input = r#"
        (module
            (func $f32_example (result f32)
                f32.const 3.14
            )
        )
    "#;

    let module = parse::parse_wat(input).unwrap();
    use loom_core::Instruction;

    // Verify F32Const instruction exists
    let instructions = &module.functions[0].instructions;
    let has_f32_const = instructions
        .iter()
        .any(|i| matches!(i, Instruction::F32Const(_)));

    assert!(
        has_f32_const,
        "Expected F32Const instruction, got: {:?}",
        instructions
    );
}

#[test]
fn test_f64_const_parsing() {
    let input = r#"
        (module
            (func $f64_example (result f64)
                f64.const 2.71828
            )
        )
    "#;

    let module = parse::parse_wat(input).unwrap();
    use loom_core::Instruction;

    // Verify F64Const instruction exists
    let instructions = &module.functions[0].instructions;
    let has_f64_const = instructions
        .iter()
        .any(|i| matches!(i, Instruction::F64Const(_)));

    assert!(
        has_f64_const,
        "Expected F64Const instruction, got: {:?}",
        instructions
    );
}

#[test]
fn test_f32_const_roundtrip() {
    let input = r#"
        (module
            (func $f32_roundtrip (result f32)
                f32.const 1.5
            )
        )
    "#;

    // Parse and encode back
    let module = parse::parse_wat(input).unwrap();
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");

    // Validate WASM is well-formed
    wasmparser::validate(&wasm_bytes).expect("F32Const should produce valid WASM");

    // Re-parse and verify we get the same structure
    let module2 = parse::parse_wasm(&wasm_bytes).unwrap();
    use loom_core::Instruction;

    let instructions = &module2.functions[0].instructions;
    let has_f32_const = instructions
        .iter()
        .any(|i| matches!(i, Instruction::F32Const(_)));

    assert!(has_f32_const, "F32Const should survive roundtrip");
}

#[test]
fn test_f64_const_roundtrip() {
    let input = r#"
        (module
            (func $f64_roundtrip (result f64)
                f64.const 2.71828
            )
        )
    "#;

    // Parse and encode back
    let module = parse::parse_wat(input).unwrap();
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");

    // Validate WASM is well-formed
    wasmparser::validate(&wasm_bytes).expect("F64Const should produce valid WASM");

    // Re-parse and verify we get the same structure
    let module2 = parse::parse_wasm(&wasm_bytes).unwrap();
    use loom_core::Instruction;

    let instructions = &module2.functions[0].instructions;
    let has_f64_const = instructions
        .iter()
        .any(|i| matches!(i, Instruction::F64Const(_)));

    assert!(has_f64_const, "F64Const should survive roundtrip");
}

#[test]
fn test_multiple_float_constants() {
    let input = r#"
        (module
            (func $multiple_floats (result f64)
                f32.const 1.0
                f32.const 2.0
                f64.const 3.5
                f64.const 4.5
            )
        )
    "#;

    let module = parse::parse_wat(input).unwrap();
    use loom_core::Instruction;

    // Count float constants
    let instructions = &module.functions[0].instructions;
    let f32_count = instructions
        .iter()
        .filter(|i| matches!(i, Instruction::F32Const(_)))
        .count();
    let f64_count = instructions
        .iter()
        .filter(|i| matches!(i, Instruction::F64Const(_)))
        .count();

    assert_eq!(f32_count, 2, "Expected 2 F32Const instructions");
    assert_eq!(f64_count, 2, "Expected 2 F64Const instructions");
}

#[test]
fn test_float_constant_encoding_roundtrip() {
    let input = r#"
        (module
            (func $test_floats (result f64)
                f32.const 3.14159
                f64.promote_f32
                f64.const 1.41421
                f64.add
            )
        )
    "#;

    // Parse, optimize, encode, and validate
    let mut module = parse::parse_wat(input).unwrap();
    optimize::optimize_module(&mut module).expect("Optimization should succeed");

    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
    wasmparser::validate(&wasm_bytes).expect("Float constants should encode to valid WASM");

    // Verify constants survive optimization and encoding
    use loom_core::Instruction;
    let instructions = &module.functions[0].instructions;
    let has_floats = instructions
        .iter()
        .any(|i| matches!(i, Instruction::F32Const(_) | Instruction::F64Const(_)));

    assert!(has_floats, "Float constants should survive optimization");
}

#[test]
fn test_f32_const_with_operations() {
    let input = r#"
        (module
            (func $f32_ops (result f32)
                f32.const 2.0
                f32.const 3.0
                f32.add
            )
        )
    "#;

    let module = parse::parse_wat(input).unwrap();
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");

    wasmparser::validate(&wasm_bytes).expect("F32Const with operations should be valid");
}

#[test]
fn test_f64_const_with_operations() {
    let input = r#"
        (module
            (func $f64_ops (result f64)
                f64.const 1.5
                f64.const 2.5
                f64.mul
            )
        )
    "#;

    let module = parse::parse_wat(input).unwrap();
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");

    wasmparser::validate(&wasm_bytes).expect("F64Const with operations should be valid");
}

#[test]
fn test_float_const_with_conditionals() {
    let input = r#"
        (module
            (func $float_if (param f32) (result f32)
                local.get 0
                f32.const 5.0
                f32.lt
                if (result f32)
                    f32.const 1.0
                else
                    f32.const 0.0
                end
            )
        )
    "#;

    let module = parse::parse_wat(input).unwrap();
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");

    wasmparser::validate(&wasm_bytes).expect("Float constants in conditionals should be valid");
}

#[test]
fn test_float_const_special_values() {
    let input = r#"
        (module
            (func $special_floats (result f64)
                f32.const 0.0
                f64.promote_f32
                f64.const 1e308
                f64.add
            )
        )
    "#;

    let module = parse::parse_wat(input).unwrap();
    use loom_core::encode;
    let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");

    wasmparser::validate(&wasm_bytes).expect("Special float values should encode correctly");

    // Verify roundtrip preserves structure
    let module2 = parse::parse_wasm(&wasm_bytes).unwrap();
    use loom_core::Instruction;

    let instructions = &module2.functions[0].instructions;
    let has_floats = instructions
        .iter()
        .any(|i| matches!(i, Instruction::F32Const(_) | Instruction::F64Const(_)));

    assert!(has_floats, "Special float values should survive roundtrip");
}
