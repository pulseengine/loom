//! Peephole synthesis (Souper-shaped, v0.9.0 PR-L2).
//!
//! Hand-curated extension of the algorithmic-solver direction from
//! `docs/research/v0.7.0/algorithmic-solver-feasibility.md`. The full
//! Souper analog is ~1500 LOC; PR-L shipped the harness with 3 rules,
//! PR-L2 grows the set to 12 hand-proven arithmetic identities and
//! wires the pass into the optimizer pipeline.
//!
//! ## What's here
//!
//! A small set of `Candidate` rules, each of which is:
//!   - A `match_pattern` (sequence of `Instruction` to find).
//!   - A `replacement` (sequence to substitute, typically empty).
//!   - A formal-proof comment documenting WHY the rule is sound.
//!
//! The `apply_peephole_synth` pass walks each function and replaces
//! each match. The existing per-function stack validator wraps the
//! call site so any rewrite that breaks stack typing is caught and
//! reverted. The candidates themselves are right-identity patterns
//! whose `pattern` consumes `(stack-top: const, below: x)` and leaves
//! `x` on the stack, so the stack effect of `pattern` and `replacement`
//! (empty) match by construction.
//!
//! ## Operand-order audit
//!
//! WebAssembly binary ops pop in reverse: for `iN.<op>`, the
//! second-pushed operand `c2` is popped first, then `c1`, then the
//! result `c1 op c2` is pushed. Every candidate below puts the
//! constant on the stack-top (the position of `c2`), so:
//!
//!   - `x op 0`  for op ∈ {add, sub, or, shl, shr_s, shr_u}: identity in x
//!   - `x op 1`  for op = mul: identity in x
//!   - `x & -1`: identity in x (all-ones AND)
//!
//! Note: `0 - x` (= -x) is NOT folded — that would require the constant
//! on the bottom of the stack, not the top.
//!
//! ## Why no runtime Z3 gate
//!
//! For the 12 hand-curated identities in this PR, the proofs are
//! well-known algebraic identities. A future PR will add the
//! startup-time Z3 verifier once the candidate set grows past what
//! can be hand-audited (e.g., once we start harvesting from real
//! corpora).
//!
//! ## Candidates shipped
//!
//! All twelve are RIGHT-IDENTITY arithmetic rewrites. They take the
//! form `<x> ; iN.const I ; iN.op` → `<x>`.
//!
//! i32:
//!   1. `<x> ; i32.const 0  ; i32.add`    →  `<x>`   (x + 0 = x)
//!   2. `<x> ; i32.const 0  ; i32.or`     →  `<x>`   (x | 0 = x)
//!   3. `<x> ; i32.const -1 ; i32.and`    →  `<x>`   (x & -1 = x)
//!   4. `<x> ; i32.const 1  ; i32.mul`    →  `<x>`   (x * 1 = x)
//!   5. `<x> ; i32.const 0  ; i32.sub`    →  `<x>`   (x - 0 = x)
//!   6. `<x> ; i32.const 0  ; i32.shl`    →  `<x>`   (x << 0 = x)
//!   7. `<x> ; i32.const 0  ; i32.shr_s`  →  `<x>`   (x >>s 0 = x)
//!   8. `<x> ; i32.const 0  ; i32.shr_u`  →  `<x>`   (x >>u 0 = x)
//!
//! i64:
//!   9. `<x> ; i64.const 0  ; i64.add`    →  `<x>`   (x + 0 = x)
//!  10. `<x> ; i64.const 0  ; i64.or`     →  `<x>`   (x | 0 = x)
//!  11. `<x> ; i64.const -1 ; i64.and`    →  `<x>`   (x & -1 = x)
//!  12. `<x> ; i64.const 1  ; i64.mul`    →  `<x>`   (x * 1 = x)
//!
//! Proofs are documented per-candidate below.

use crate::{Instruction, Module};
use anyhow::Result;

/// A single peephole rewrite candidate. Both pattern and replacement
/// are slices of consecutive instructions. The match is linear (no
/// reordering, no cross-block reach).
struct Candidate {
    /// Human-readable name (used in revert telemetry — currently unread,
    /// but reserved for a future startup gate that will log per-rule
    /// admission events).
    #[allow(dead_code)]
    name: &'static str,
    /// Instructions to match exactly.
    pattern: &'static [Instruction],
    /// Replacement (often empty: the pattern is a no-op on the stack).
    replacement: &'static [Instruction],
}

/// The shipped candidate set. Each entry's `pattern` must, when
/// preceded by a value of the appropriate type on the stack, leave
/// the same value on the stack — i.e., the entire `pattern` is a
/// no-op for the right-hand operand. The `replacement` (empty here)
/// achieves the same identity at zero cost.
const CANDIDATES: &[Candidate] = &[
    // ─── i32 ────────────────────────────────────────────────────────────
    Candidate {
        name: "i32_add_zero_identity",
        // Proof: ∀x: BV32. x + 0 = x (additive identity in Z/2^32).
        // wasm i32.add is two's-complement addition mod 2^32, for which
        // 0 is the unique additive identity.
        pattern: &[Instruction::I32Const(0), Instruction::I32Add],
        replacement: &[],
    },
    Candidate {
        name: "i32_or_zero_identity",
        // Proof: ∀x: BV32. x | 0 = x (bitwise-OR identity element is 0).
        // Trivially true bit-by-bit.
        pattern: &[Instruction::I32Const(0), Instruction::I32Or],
        replacement: &[],
    },
    Candidate {
        name: "i32_and_neg_one_identity",
        // Proof: ∀x: BV32. x & 0xFFFFFFFF = x (bitwise-AND identity
        // element is all-ones). In two's-complement i32, -1 = 0xFFFFFFFF.
        // Bit-by-bit: each bit of x AND 1 yields the bit of x.
        pattern: &[Instruction::I32Const(-1), Instruction::I32And],
        replacement: &[],
    },
    Candidate {
        name: "i32_mul_one_identity",
        // Proof: ∀x: BV32. x * 1 = x (multiplicative identity in Z/2^32).
        // wasm i32.mul is two's-complement multiplication mod 2^32, for
        // which 1 is the unique multiplicative identity.
        pattern: &[Instruction::I32Const(1), Instruction::I32Mul],
        replacement: &[],
    },
    Candidate {
        name: "i32_sub_zero_identity",
        // Proof: ∀x: BV32. x - 0 = x.
        // Operand-order check: wasm pops c2 (top) then c1 (below), and
        // pushes c1 - c2. With c2 = 0, the result is c1 = x. Note this
        // is the RIGHT-zero subtraction; 0 - x = -x is NOT folded.
        pattern: &[Instruction::I32Const(0), Instruction::I32Sub],
        replacement: &[],
    },
    Candidate {
        name: "i32_shl_zero_identity",
        // Proof: ∀x: BV32. x << 0 = x.
        // wasm i32.shl is defined as `c1 << (c2 mod 32)`. With c2 = 0,
        // the shift amount is 0, and shifting by 0 is the identity.
        // (For c2 = 32 the shift would also be 0 by the mod rule, but
        // we do NOT fold that case — only the literal constant 0 is
        // guaranteed to be the shift-by-0 we want.)
        pattern: &[Instruction::I32Const(0), Instruction::I32Shl],
        replacement: &[],
    },
    Candidate {
        name: "i32_shr_s_zero_identity",
        // Proof: ∀x: BV32. x >>s 0 = x.
        // wasm i32.shr_s is arithmetic right shift by (c2 mod 32). With
        // c2 = 0, the shift amount is 0; arithmetic shift by 0 yields
        // x unchanged (sign bit is replicated zero times).
        pattern: &[Instruction::I32Const(0), Instruction::I32ShrS],
        replacement: &[],
    },
    Candidate {
        name: "i32_shr_u_zero_identity",
        // Proof: ∀x: BV32. x >>u 0 = x.
        // wasm i32.shr_u is logical right shift by (c2 mod 32). With
        // c2 = 0, the shift amount is 0; logical shift by 0 yields x
        // unchanged (no bits are dropped or zero-extended).
        pattern: &[Instruction::I32Const(0), Instruction::I32ShrU],
        replacement: &[],
    },
    // ─── i64 ────────────────────────────────────────────────────────────
    Candidate {
        name: "i64_add_zero_identity",
        // Proof: ∀x: BV64. x + 0 = x (additive identity in Z/2^64).
        // wasm i64.add is two's-complement addition mod 2^64, for which
        // 0 is the unique additive identity.
        pattern: &[Instruction::I64Const(0), Instruction::I64Add],
        replacement: &[],
    },
    Candidate {
        name: "i64_or_zero_identity",
        // Proof: ∀x: BV64. x | 0 = x (bitwise-OR identity element is 0).
        // Trivially true bit-by-bit on 64 bits.
        pattern: &[Instruction::I64Const(0), Instruction::I64Or],
        replacement: &[],
    },
    Candidate {
        name: "i64_and_neg_one_identity",
        // Proof: ∀x: BV64. x & 0xFFFFFFFFFFFFFFFF = x (bitwise-AND
        // identity element is all-ones). In two's-complement i64,
        // -1 = 0xFFFFFFFFFFFFFFFF. Bit-by-bit on 64 bits.
        pattern: &[Instruction::I64Const(-1), Instruction::I64And],
        replacement: &[],
    },
    Candidate {
        name: "i64_mul_one_identity",
        // Proof: ∀x: BV64. x * 1 = x (multiplicative identity in Z/2^64).
        // wasm i64.mul is two's-complement multiplication mod 2^64, for
        // which 1 is the unique multiplicative identity.
        pattern: &[Instruction::I64Const(1), Instruction::I64Mul],
        replacement: &[],
    },
];

/// Apply all shipped peephole candidates to every function in the
/// module. Uses the existing per-function stack validator as a safety
/// net: if any rewrite somehow changes stack typing, the validator
/// rejects and reverts.
pub fn apply_peephole_synth(module: &mut Module) -> Result<usize> {
    let mut total_folds = 0;

    for func in &mut module.functions {
        let original = func.instructions.clone();
        let folds = apply_to_body(&mut func.instructions);
        total_folds += folds;

        // No Z3 hook for this MVP — the candidates are hand-proven
        // identity rules. The existing stack validator catches any
        // byte-level divergence. A future PR will run a startup-time
        // Z3 gate here, but for now we trust the rules + the validator.
        if folds > 0 {
            // Quick sanity: the function body must still validate.
            // If for some reason it doesn't (e.g., a future rule
            // changes stack effect), revert.
            if let Err(e) = crate::stack::validation::validate_function(func) {
                eprintln!(
                    "peephole_synth: validation failed after {} folds, reverting: {}",
                    folds, e
                );
                crate::stats::record_revert("peephole_synth/stack-invalid");
                func.instructions = original;
                total_folds -= folds;
            }
        }
    }

    Ok(total_folds)
}

/// Apply all candidates repeatedly to `body` until no further matches
/// happen. Recurses into Block/Loop/If bodies. Returns total folds.
fn apply_to_body(body: &mut Vec<Instruction>) -> usize {
    let mut total = 0;

    // Step 1: recurse into nested bodies.
    for instr in body.iter_mut() {
        match instr {
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                total += apply_to_body(body);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                total += apply_to_body(then_body);
                total += apply_to_body(else_body);
            }
            _ => {}
        }
    }

    // Step 2: linear scan, replace each match with the candidate's
    // replacement. Iterate until no further matches.
    loop {
        let mut matched_any = false;
        let mut i = 0;
        while i < body.len() {
            let mut hit: Option<&Candidate> = None;
            for c in CANDIDATES {
                if i + c.pattern.len() <= body.len()
                    && body[i..i + c.pattern.len()] == *c.pattern
                {
                    hit = Some(c);
                    break;
                }
            }
            if let Some(c) = hit {
                let n = c.pattern.len();
                // Replace pattern with replacement (typically empty).
                body.splice(i..i + n, c.replacement.iter().cloned());
                matched_any = true;
                total += 1;
                // Don't advance i — the replacement could itself enable
                // an adjacent match.
            } else {
                i += 1;
            }
        }
        if !matched_any {
            break;
        }
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn test_i32_add_zero_identity() {
        // (local.get 0; i32.const 0; i32.add) → (local.get 0)
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.add
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i32.add 0 identity must fire");

        let has_add = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Add));
        let has_zero = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Const(0)));
        assert!(!has_add, "i32.add must be gone after fold");
        assert!(!has_zero, "i32.const 0 must be gone after fold");
    }

    #[test]
    fn test_i32_or_zero_identity() {
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.or
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i32.or 0 identity must fire");

        let has_or = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Or));
        assert!(!has_or, "i32.or must be gone after fold");
    }

    #[test]
    fn test_i32_and_neg_one_identity() {
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const -1
                i32.and
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i32.and -1 identity must fire");

        let has_and = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32And));
        assert!(!has_and, "i32.and must be gone after fold");
    }

    #[test]
    fn test_does_not_fold_add_nonzero() {
        // x + 1 must NOT be folded — the constant is non-identity.
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 0, "i32.add 1 must NOT fold");

        let has_add = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Add));
        assert!(has_add, "i32.add 1 must survive");
    }

    #[test]
    fn test_chained_identities_fold_iteratively() {
        // Two adjacent identities should both fire.
        // (x; 0; or; 0; add) → (x)
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.or
                i32.const 0
                i32.add
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 2, "both identities must fire");

        // Only LocalGet should remain.
        assert_eq!(
            module.functions[0].instructions.len(),
            1,
            "only local.get should survive"
        );
    }

    #[test]
    fn test_recurses_into_blocks() {
        // Pattern inside a block body — recursion must reach it.
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                block (result i32)
                    local.get 0
                    i32.const 0
                    i32.add
                end
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "block-nested identity must fold");
    }

    // ─── PR-L2: new candidate fire tests ──────────────────────────────

    #[test]
    fn test_i32_mul_one_identity() {
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.mul
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i32.mul 1 identity must fire");

        let has_mul = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Mul));
        assert!(!has_mul, "i32.mul must be gone after fold");
    }

    #[test]
    fn test_i32_sub_zero_identity() {
        // x - 0 = x. Operand-order check baked in: the const 0 is the
        // RHS (top of stack), so this is the right-identity case.
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.sub
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i32.sub 0 identity must fire");

        let has_sub = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Sub));
        assert!(!has_sub, "i32.sub must be gone after fold");
    }

    #[test]
    fn test_i32_shl_zero_identity() {
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.shl
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i32.shl 0 identity must fire");

        let has_shl = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Shl));
        assert!(!has_shl, "i32.shl must be gone after fold");
    }

    #[test]
    fn test_i32_shr_s_zero_identity() {
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.shr_s
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i32.shr_s 0 identity must fire");

        let has_shr = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32ShrS));
        assert!(!has_shr, "i32.shr_s must be gone after fold");
    }

    #[test]
    fn test_i32_shr_u_zero_identity() {
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.shr_u
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i32.shr_u 0 identity must fire");

        let has_shr = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32ShrU));
        assert!(!has_shr, "i32.shr_u must be gone after fold");
    }

    #[test]
    fn test_i64_add_zero_identity() {
        let wat = r#"(module
            (func (export "test") (param i64) (result i64)
                local.get 0
                i64.const 0
                i64.add
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i64.add 0 identity must fire");

        let has_add = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Add));
        assert!(!has_add, "i64.add must be gone after fold");
    }

    #[test]
    fn test_i64_or_zero_identity() {
        let wat = r#"(module
            (func (export "test") (param i64) (result i64)
                local.get 0
                i64.const 0
                i64.or
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i64.or 0 identity must fire");

        let has_or = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Or));
        assert!(!has_or, "i64.or must be gone after fold");
    }

    #[test]
    fn test_i64_and_neg_one_identity() {
        let wat = r#"(module
            (func (export "test") (param i64) (result i64)
                local.get 0
                i64.const -1
                i64.and
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i64.and -1 identity must fire");

        let has_and = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64And));
        assert!(!has_and, "i64.and must be gone after fold");
    }

    #[test]
    fn test_i64_mul_one_identity() {
        let wat = r#"(module
            (func (export "test") (param i64) (result i64)
                local.get 0
                i64.const 1
                i64.mul
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 1, "i64.mul 1 identity must fire");

        let has_mul = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Mul));
        assert!(!has_mul, "i64.mul must be gone after fold");
    }

    // ─── PR-L2: negative cases (constant must be the identity element) ──

    #[test]
    fn test_does_not_fold_mul_non_one() {
        // x * 2 must NOT fold — 2 is not the multiplicative identity.
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 2
                i32.mul
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 0, "i32.mul 2 must NOT fold");
    }

    #[test]
    fn test_does_not_fold_sub_nonzero() {
        // x - 1 must NOT fold.
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.sub
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 0, "i32.sub 1 must NOT fold");
    }

    #[test]
    fn test_does_not_fold_shl_nonzero() {
        // x << 1 must NOT fold (it doubles x).
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.shl
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 0, "i32.shl 1 must NOT fold");
    }

    #[test]
    fn test_does_not_fold_shr_s_nonzero() {
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.shr_s
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 0, "i32.shr_s 1 must NOT fold");
    }

    #[test]
    fn test_does_not_fold_shr_u_nonzero() {
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.shr_u
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 0, "i32.shr_u 1 must NOT fold");
    }

    #[test]
    fn test_does_not_fold_i64_add_nonzero() {
        let wat = r#"(module
            (func (export "test") (param i64) (result i64)
                local.get 0
                i64.const 1
                i64.add
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 0, "i64.add 1 must NOT fold");
    }

    #[test]
    fn test_does_not_fold_i64_or_nonzero() {
        let wat = r#"(module
            (func (export "test") (param i64) (result i64)
                local.get 0
                i64.const 1
                i64.or
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 0, "i64.or 1 must NOT fold");
    }

    #[test]
    fn test_does_not_fold_i64_and_non_neg_one() {
        // x & 0 = 0, NOT x — must not fold.
        let wat = r#"(module
            (func (export "test") (param i64) (result i64)
                local.get 0
                i64.const 0
                i64.and
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 0, "i64.and 0 must NOT fold (it's x & 0 = 0, not identity)");
    }

    #[test]
    fn test_does_not_fold_i64_mul_non_one() {
        let wat = r#"(module
            (func (export "test") (param i64) (result i64)
                local.get 0
                i64.const 2
                i64.mul
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let folds = apply_peephole_synth(&mut module).expect("apply");
        assert_eq!(folds, 0, "i64.mul 2 must NOT fold");
    }
}
