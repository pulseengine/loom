//! Formal Verification Module for LOOM
//!
//! This module implements translation validation using the Z3 SMT solver to prove
//! that optimizations preserve program semantics.
//!
//! # Overview
//!
//! Translation validation works by encoding both the original and optimized programs
//! as SMT formulas and asking Z3 to prove they are semantically equivalent for all
//! possible inputs.
//!
//! # Example
//!
//! ```rust,ignore
//! use loom_core::verify::verify_optimization;
//!
//! let original = parse_wat("(module (func (result i32) (i32.add (i32.const 2) (i32.const 3))))");
//! let optimized = parse_wat("(module (func (result i32) (i32.const 5)))");
//!
//! // Verify that 2 + 3 = 5 optimization is correct
//! assert!(verify_optimization(&original, &optimized).unwrap());
//! ```

#[cfg(feature = "verification")]
use z3::ast::{Ast, BV};
#[cfg(feature = "verification")]
use z3::{Config, Context, SatResult, Solver};

#[cfg(not(feature = "verification"))]
use crate::Module;
#[cfg(feature = "verification")]
use crate::{Function, Instruction, Module};
#[cfg(feature = "verification")]
use anyhow::Context as AnyhowContext;
use anyhow::{anyhow, Result};

/// Verify that an optimization preserves program semantics
///
/// This function uses Z3 SMT solver to prove that the optimized program is semantically
/// equivalent to the original program for all possible inputs.
///
/// # Arguments
///
/// * `original` - The original unoptimized module
/// * `optimized` - The optimized module
///
/// # Returns
///
/// * `Ok(true)` - Programs are proven equivalent
/// * `Ok(false)` - Found a counterexample (programs differ)
/// * `Err(_)` - Verification error or timeout
///
/// # Examples
///
/// ```rust,ignore
/// let original = parse_wat("...");
/// let optimized = optimize_module(original.clone());
/// assert!(verify_optimization(&original, &optimized)?);
/// ```
#[cfg(feature = "verification")]
pub fn verify_optimization(original: &Module, optimized: &Module) -> Result<bool> {
    // Check basic structural equivalence first
    if original.functions.len() != optimized.functions.len() {
        return Ok(false);
    }

    // Create Z3 context and solver
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let solver = Solver::new(&ctx);

    // Verify each function pair
    for (orig_func, opt_func) in original.functions.iter().zip(optimized.functions.iter()) {
        // Check signatures match
        if orig_func.signature.params != opt_func.signature.params
            || orig_func.signature.results != opt_func.signature.results
        {
            return Ok(false);
        }

        // Encode both functions to SMT
        let orig_formula = encode_function_to_smt(&ctx, orig_func)?;
        let opt_formula = encode_function_to_smt(&ctx, opt_func)?;

        // Assert they are NOT equal (looking for counterexample)
        solver.push();
        solver.assert(&orig_formula._eq(&opt_formula).not());

        // UNSAT means equivalent (no counterexample exists)
        match solver.check() {
            SatResult::Unsat => {
                // Functions are equivalent
                solver.pop(1);
                continue;
            }
            SatResult::Sat => {
                // Found counterexample - not equivalent!
                let model = solver.get_model().context("Failed to get counterexample")?;
                eprintln!("Counterexample found:");
                eprintln!("{}", model);
                return Ok(false);
            }
            SatResult::Unknown => {
                return Err(anyhow!(
                    "SMT solver returned unknown (timeout or too complex)"
                ));
            }
        }
    }

    Ok(true)
}

/// Encode a WebAssembly function to an SMT formula
///
/// This converts the instruction sequence into a symbolic execution that Z3 can reason about.
#[cfg(feature = "verification")]
fn encode_function_to_smt<'ctx>(ctx: &'ctx Context, func: &Function) -> Result<BV<'ctx>> {
    // Create symbolic variables for parameters
    let mut stack: Vec<BV<'ctx>> = Vec::new();
    let mut locals: Vec<BV<'ctx>> = Vec::new();

    // Initialize parameters as symbolic inputs
    for (idx, param_type) in func.signature.params.iter().enumerate() {
        let width = match param_type {
            crate::ValueType::I32 => 32,
            crate::ValueType::I64 => 64,
            crate::ValueType::F32 | crate::ValueType::F64 => {
                // For now, skip floating point (would need different encoding)
                return Err(anyhow!("Floating point verification not yet supported"));
            }
        };
        let param = BV::new_const(ctx, format!("param{}", idx), width);
        locals.push(param);
    }

    // Initialize local variables to zero
    for local_type in func.locals.iter() {
        let width = match local_type.1 {
            crate::ValueType::I32 => 32,
            crate::ValueType::I64 => 64,
            crate::ValueType::F32 | crate::ValueType::F64 => {
                return Err(anyhow!("Floating point verification not yet supported"));
            }
        };
        locals.push(BV::from_u64(ctx, 0, width));
    }

    // Symbolically execute instructions
    for instr in &func.instructions {
        match instr {
            // Constants
            Instruction::I32Const(n) => {
                stack.push(BV::from_i64(ctx, *n as i64, 32));
            }
            Instruction::I64Const(n) => {
                stack.push(BV::from_i64(ctx, *n, 64));
            }
            Instruction::F32Const(bits) => {
                // Float constants are treated as bit patterns for now
                // We don't perform floating-point arithmetic verification yet
                stack.push(BV::from_i64(ctx, *bits as i64, 32));
            }
            Instruction::F64Const(bits) => {
                // Float constants are treated as bit patterns for now
                // We don't perform floating-point arithmetic verification yet
                stack.push(BV::from_i64(ctx, *bits as i64, 64));
            }

            // Arithmetic operations (i32)
            Instruction::I32Add => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Add"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvadd(&rhs));
            }
            Instruction::I32Sub => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Sub"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvsub(&rhs));
            }
            Instruction::I32Mul => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Mul"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvmul(&rhs));
            }

            // Arithmetic operations (i64)
            Instruction::I64Add => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Add"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvadd(&rhs));
            }
            Instruction::I64Sub => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Sub"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvsub(&rhs));
            }
            Instruction::I64Mul => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Mul"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvmul(&rhs));
            }

            // Bitwise operations (i32)
            Instruction::I32And => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32And"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvand(&rhs));
            }
            Instruction::I32Or => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Or"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvor(&rhs));
            }
            Instruction::I32Xor => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Xor"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvxor(&rhs));
            }
            Instruction::I32Shl => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Shl"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvshl(&rhs));
            }
            Instruction::I32ShrU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32ShrU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvlshr(&rhs));
            }
            Instruction::I32ShrS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32ShrS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvashr(&rhs));
            }

            // Bitwise operations (i64)
            Instruction::I64And => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64And"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvand(&rhs));
            }
            Instruction::I64Or => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Or"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvor(&rhs));
            }
            Instruction::I64Xor => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Xor"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvxor(&rhs));
            }
            Instruction::I64Shl => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Shl"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvshl(&rhs));
            }
            Instruction::I64ShrU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64ShrU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvlshr(&rhs));
            }
            Instruction::I64ShrS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64ShrS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvashr(&rhs));
            }

            // Local operations
            Instruction::LocalGet(idx) => {
                if *idx as usize >= locals.len() {
                    return Err(anyhow!("LocalGet index out of bounds: {}", idx));
                }
                stack.push(locals[*idx as usize].clone());
            }
            Instruction::LocalSet(idx) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in LocalSet"));
                }
                if *idx as usize >= locals.len() {
                    return Err(anyhow!("LocalSet index out of bounds: {}", idx));
                }
                let value = stack.pop().unwrap();
                locals[*idx as usize] = value;
            }
            Instruction::LocalTee(idx) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in LocalTee"));
                }
                if *idx as usize >= locals.len() {
                    return Err(anyhow!("LocalTee index out of bounds: {}", idx));
                }
                let value = stack.last().unwrap().clone();
                locals[*idx as usize] = value;
            }

            // Control flow - simplified for now
            Instruction::End => {
                // End of function
                break;
            }

            _ => {
                // Unsupported instruction for verification
                return Err(anyhow!(
                    "Unsupported instruction for verification: {:?}",
                    instr
                ));
            }
        }
    }

    // Return value should be on stack
    if stack.is_empty() {
        return Err(anyhow!("Function returned no value (stack empty)"));
    }

    Ok(stack.pop().unwrap())
}

/// Non-verification stub when Z3 feature is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_optimization(_original: &Module, _optimized: &Module) -> Result<bool> {
    Err(anyhow!(
        "Verification support not enabled. Rebuild with --features verification"
    ))
}

#[cfg(all(test, feature = "verification"))]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn test_verify_constant_folding() {
        // Original: (i32.add (i32.const 2) (i32.const 3))
        let original_wat = r#"
            (module
                (func (result i32)
                    i32.const 2
                    i32.const 3
                    i32.add
                )
            )
        "#;

        // Optimized: (i32.const 5)
        let optimized_wat = r#"
            (module
                (func (result i32)
                    i32.const 5
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        // Verify equivalence
        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "Programs should be equivalent");
    }

    #[test]
    fn test_verify_strength_reduction() {
        // Original: (i32.mul x 4)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 4
                    i32.mul
                )
            )
        "#;

        // Optimized: (i32.shl x 2)
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 2
                    i32.shl
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        // Verify equivalence
        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(
            result.unwrap(),
            "Strength reduction should preserve semantics"
        );
    }

    #[test]
    fn test_verify_bitwise_identity() {
        // Original: (i32.xor x x)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    local.get 0
                    i32.xor
                )
            )
        "#;

        // Optimized: (i32.const 0)
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 0
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        // Verify equivalence
        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "x XOR x = 0 should be proven");
    }

    #[test]
    fn test_verify_detects_incorrect_optimization() {
        // Original: (i32.add x 1)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                )
            )
        "#;

        // Incorrectly optimized to: (i32.const 2) - WRONG!
        let wrong_optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 2
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let wrong_optimized = parse::parse_wat(wrong_optimized_wat).unwrap();

        // Verification should detect this is wrong
        let result = verify_optimization(&original, &wrong_optimized);
        assert!(result.is_ok(), "Verification should complete");
        assert!(!result.unwrap(), "Should detect incorrect optimization");
    }
}
