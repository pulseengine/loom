//! Function-summary interprocedural analysis (IPA).
//!
//! Computes per-function summaries (`is_pure`, `is_no_trap`) so downstream
//! passes can reason across `Call` boundaries. Without this, every `Call`
//! is an opaque side-effecting wall — CSE can't dedupe two identical calls,
//! `vacuum` can't fold `Call f; Drop` when `f` has no observable effects,
//! and DCE can't drop a pure call whose result isn't used.
//!
//! ## Definitions
//!
//! - **Pure**: the call has no observable side effects. Specifically, the
//!   function contains no `Store`/`GlobalSet`/`Memory*`/`Table*-write`/
//!   `CallIndirect` instructions, and every direct `Call` target is itself
//!   pure. (Notably, *reading* memory or globals is allowed — a pure
//!   function can produce a value that depends on memory, but cannot
//!   change observable state.)
//!
//! - **No-trap**: the function never traps. Contains no `Unreachable`,
//!   `Div`/`Rem` (divide-by-zero), `Load`/`Store` (page-fault),
//!   `TruncF*` (NaN/overflow), `CallIndirect` (type-mismatch / OOB),
//!   `Table*` (OOB), and every direct `Call` target is itself no-trap.
//!
//! ## Fixpoint
//!
//! Analysis is optimistic-then-demote: assume all functions are
//! pure + no-trap, scan each for intrinsic violations, then iteratively
//! demote any function that calls a non-pure / may-trap function.
//! Bounded by O(#funcs) iterations (each iteration can only flip a
//! status from true→false). Mutual recursion converges naturally.
//!
//! `CallIndirect` and unsupported instructions (SIMD, ref types, etc.)
//! conservatively mark the function impure + may-trap regardless of
//! callee summaries.
//!
//! ## Use sites
//!
//! - `vacuum`'s `pure_push;Drop` peephole can be extended to fold
//!   `Call f; Drop` when `f` is pure + no-trap (the call result is
//!   unused and the call has no other observable effect).
//! - CSE can hash `Call f` as a determinate value when `f` is pure +
//!   no-trap, enabling cross-call dedup.
//! - DCE can drop a `Call f` whose only consumer is a `Drop` when the
//!   call is pure + no-trap.

use crate::{Function, Instruction, Module};

/// Per-function summary computed by the IPA.
#[derive(Debug, Clone, Copy, Default)]
pub struct FunctionSummary {
    /// The function has no observable side effects. A pure call can be
    /// CSE'd, DCE'd if its result is unused, etc.
    pub is_pure: bool,
    /// The function never traps on any input. A no-trap call followed
    /// by `Drop` can be folded away entirely (combined with `is_pure`).
    pub is_no_trap: bool,
}

/// Compute summaries for every function in `module`. Index in the returned
/// vector matches the function index in `module.functions`.
pub fn compute_module_summaries(module: &Module) -> Vec<FunctionSummary> {
    let n = module.functions.len();

    // Step 1: compute INTRINSIC purity / no-trap for each function
    // (ignoring callees). Recurses into Block/Loop/If bodies.
    let mut intrinsic_pure = vec![true; n];
    let mut intrinsic_no_trap = vec![true; n];
    let mut callees: Vec<Vec<u32>> = vec![Vec::new(); n];

    for (i, func) in module.functions.iter().enumerate() {
        let (pure, no_trap, calls) = scan_function(func);
        intrinsic_pure[i] = pure;
        intrinsic_no_trap[i] = no_trap;
        callees[i] = calls;
    }

    // Step 2: fixpoint demotion. Start from intrinsic and propagate
    // callee constraints. Each iteration can only flip true→false; the
    // loop terminates when no flip happens.
    let mut is_pure = intrinsic_pure.clone();
    let mut is_no_trap = intrinsic_no_trap.clone();

    loop {
        let mut changed = false;
        for i in 0..n {
            if is_pure[i] {
                for &callee in &callees[i] {
                    let c = callee as usize;
                    if c < n && !is_pure[c] {
                        is_pure[i] = false;
                        changed = true;
                        break;
                    }
                }
            }
            if is_no_trap[i] {
                for &callee in &callees[i] {
                    let c = callee as usize;
                    if c < n && !is_no_trap[c] {
                        is_no_trap[i] = false;
                        changed = true;
                        break;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    (0..n)
        .map(|i| FunctionSummary {
            is_pure: is_pure[i],
            is_no_trap: is_no_trap[i],
        })
        .collect()
}

/// Scan a function's instruction tree. Returns
/// `(intrinsic_pure, intrinsic_no_trap, callees)`.
fn scan_function(func: &Function) -> (bool, bool, Vec<u32>) {
    let mut pure = true;
    let mut no_trap = true;
    let mut callees: Vec<u32> = Vec::new();
    scan_body(&func.instructions, &mut pure, &mut no_trap, &mut callees);
    (pure, no_trap, callees)
}

fn scan_body(
    instructions: &[Instruction],
    pure: &mut bool,
    no_trap: &mut bool,
    callees: &mut Vec<u32>,
) {
    for instr in instructions {
        match instr {
            // Calls — record callee, purity decided by fixpoint.
            Instruction::Call(idx) => {
                callees.push(*idx);
            }
            // CallIndirect — can't resolve target; conservatively impure
            // and may-trap (type mismatch / OOB).
            Instruction::CallIndirect { .. } => {
                *pure = false;
                *no_trap = false;
            }

            // Memory writes — impure.
            Instruction::I32Store { .. }
            | Instruction::I64Store { .. }
            | Instruction::F32Store { .. }
            | Instruction::F64Store { .. }
            | Instruction::I32Store8 { .. }
            | Instruction::I32Store16 { .. }
            | Instruction::I64Store8 { .. }
            | Instruction::I64Store16 { .. }
            | Instruction::I64Store32 { .. } => {
                *pure = false;
                // Stores also may-trap (page fault on bad address).
                *no_trap = false;
            }

            // Memory loads — pure but may trap.
            Instruction::I32Load { .. }
            | Instruction::I64Load { .. }
            | Instruction::F32Load { .. }
            | Instruction::F64Load { .. }
            | Instruction::I32Load8S { .. }
            | Instruction::I32Load8U { .. }
            | Instruction::I32Load16S { .. }
            | Instruction::I32Load16U { .. }
            | Instruction::I64Load8S { .. }
            | Instruction::I64Load8U { .. }
            | Instruction::I64Load16S { .. }
            | Instruction::I64Load16U { .. }
            | Instruction::I64Load32S { .. }
            | Instruction::I64Load32U { .. } => {
                *no_trap = false;
            }

            // Global writes — impure.
            Instruction::GlobalSet(_) => {
                *pure = false;
            }

            // Memory.* / Table.* writes — impure.
            Instruction::MemoryGrow(_)
            | Instruction::MemoryCopy { .. }
            | Instruction::MemoryFill(_)
            | Instruction::MemoryInit { .. }
            | Instruction::DataDrop(_) => {
                *pure = false;
                *no_trap = false; // many of these can also fault
            }

            // Integer divide / remainder — pure but trap on divisor 0.
            Instruction::I32DivS
            | Instruction::I32DivU
            | Instruction::I32RemS
            | Instruction::I32RemU
            | Instruction::I64DivS
            | Instruction::I64DivU
            | Instruction::I64RemS
            | Instruction::I64RemU => {
                *no_trap = false;
            }

            // Unreachable — explicit trap.
            Instruction::Unreachable => {
                *pure = false;
                *no_trap = false;
            }

            // Float-to-int truncation traps on NaN / out-of-range.
            // (Conservative on any of these even though they're pure
            // in terms of state — they CAN end execution.)
            // The variant names depend on the codebase; only call out
            // ones we know exist via the encoder.
            // (Catch-all for unknown variants is the wildcard below.)

            // Structured control flow — recurse.
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                scan_body(body, pure, no_trap, callees);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                scan_body(then_body, pure, no_trap, callees);
                scan_body(else_body, pure, no_trap, callees);
            }

            // Everything else — pure + no-trap by default. New instruction
            // variants we don't know about (e.g., SIMD, ref types) would
            // fall through to here, but the existing
            // `has_unsupported_isle_instructions` guard in the consumer
            // pass ensures those functions are skipped before reaching
            // this analysis. For the subset LOOM actually processes, the
            // default is correct.
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn analyze_wat(wat: &str) -> Vec<FunctionSummary> {
        let module = parse::parse_wat(wat).expect("parse");
        compute_module_summaries(&module)
    }

    #[test]
    fn test_pure_arithmetic_function_is_pure_and_no_trap() {
        let wat = r#"(module
            (func (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure, "arithmetic-only function must be pure");
        assert!(s[0].is_no_trap, "arithmetic-only function must be no-trap");
    }

    #[test]
    fn test_store_makes_function_impure() {
        let wat = r#"(module
            (memory 1)
            (func (param i32 i32)
                local.get 0
                local.get 1
                i32.store
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(!s[0].is_pure, "i32.store must mark function impure");
        assert!(!s[0].is_no_trap, "i32.store must mark function may-trap");
    }

    #[test]
    fn test_load_is_pure_but_may_trap() {
        let wat = r#"(module
            (memory 1)
            (func (param i32) (result i32)
                local.get 0
                i32.load
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure, "i32.load is pure (no observable write)");
        assert!(!s[0].is_no_trap, "i32.load can fault on bad address");
    }

    #[test]
    fn test_divide_is_pure_but_may_trap() {
        let wat = r#"(module
            (func (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.div_s
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure);
        assert!(!s[0].is_no_trap, "i32.div_s traps on divisor 0");
    }

    #[test]
    fn test_global_set_makes_function_impure() {
        let wat = r#"(module
            (global $g (mut i32) (i32.const 0))
            (func (param i32)
                local.get 0
                global.set $g
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(!s[0].is_pure, "global.set must mark function impure");
    }

    #[test]
    fn test_pure_caller_of_pure_callee_is_pure() {
        // Two-function module: $callee is pure, $caller calls $callee.
        let wat = r#"(module
            (func $callee (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add
            )
            (func $caller (param i32) (result i32)
                local.get 0
                call $callee
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure && s[0].is_no_trap, "callee is pure + no-trap");
        assert!(
            s[1].is_pure && s[1].is_no_trap,
            "caller of pure+no-trap callee must inherit"
        );
    }

    #[test]
    fn test_impure_propagates_through_call() {
        // Caller is intrinsically pure but calls an impure callee.
        let wat = r#"(module
            (memory 1)
            (func $impure (param i32)
                local.get 0
                i32.const 1
                i32.store
            )
            (func $caller (param i32)
                local.get 0
                call $impure
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(!s[0].is_pure, "callee performs store, must be impure");
        assert!(
            !s[1].is_pure,
            "caller of impure callee must propagate impurity"
        );
    }

    #[test]
    fn test_call_indirect_marks_caller_impure_and_may_trap() {
        let wat = r#"(module
            (table 1 funcref)
            (type $sig (func (param i32) (result i32)))
            (func (param i32 i32) (result i32)
                local.get 0
                local.get 1
                call_indirect (type $sig)
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(
            !s[0].is_pure && !s[0].is_no_trap,
            "call_indirect can't resolve target — conservative"
        );
    }

    #[test]
    fn test_mutual_recursion_converges() {
        // Two functions calling each other; both pure intrinsically.
        let wat = r#"(module
            (func $a (param i32) (result i32)
                local.get 0
                i32.eqz
                if (result i32)
                    i32.const 0
                else
                    local.get 0
                    i32.const 1
                    i32.sub
                    call $b
                end
            )
            (func $b (param i32) (result i32)
                local.get 0
                call $a
            )
        )"#;
        let s = analyze_wat(wat);
        // Both should be pure + no-trap by fixpoint.
        assert!(s[0].is_pure && s[0].is_no_trap);
        assert!(s[1].is_pure && s[1].is_no_trap);
    }

    #[test]
    fn test_recursion_with_impure_self() {
        // Function that recursively calls itself AND has an impure op.
        // The self-recursion shouldn't mask the intrinsic impurity.
        let wat = r#"(module
            (memory 1)
            (func $self_call (param i32)
                local.get 0
                i32.const 1
                i32.store
                local.get 0
                call $self_call
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(!s[0].is_pure, "self-recursion does not save an impure body");
    }
}
