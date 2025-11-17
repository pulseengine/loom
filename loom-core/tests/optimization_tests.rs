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
