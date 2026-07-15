// The Z3 backend's fluent builder takes borrowed args (`&lower(b)`); this is the
// idiomatic Z3 pattern and matches `verify_rules.rs`.
#![allow(clippy::needless_borrows_for_generic_args)]

//! Swappable solver backend for algebraic rule verification (issue #277).
//!
//! `verify_rules.rs` proves LOOM's algebraic rewrite rules correct: for each
//! rule it builds an LHS and an RHS bitvector term, asserts they are NOT equal,
//! and expects `Unsat` — i.e. the rule holds for *all* inputs. Historically the
//! terms were built directly against Z3's `z3::ast::BV` and discharged with
//! `z3::Solver`.
//!
//! This module is the **migration boundary** for moving that verification off
//! Z3 and onto `ordeal` — the org's pure-Rust, certificate-checked QF_BV
//! decision procedure. It introduces:
//!
//!   1. A backend-neutral term DSL ([`RuleTerm`] / [`RuleBool`]) that captures
//!      exactly the QF_BV fragment the rules use (widths 32/64, no arrays, no
//!      uninterpreted functions).
//!   2. A [`RuleSolver`] trait: `prove_rule_equiv(lhs, rhs) -> RuleVerdict`.
//!   3. Two implementations — [`Z3RuleSolver`] (the original Z3 logic) and
//!      [`OrdealRuleSolver`] (via `ordeal::Solver::prove_equiv`, with a
//!      defence-in-depth certificate `recheck`).
//!
//! The backend is selected by the `LOOM_VERIFY_BACKEND` environment variable:
//!
//!   * `z3`     — Z3 only (default; keeps existing behaviour while ordeal
//!                is burned in).
//!   * `ordeal` — ordeal only.
//!   * `both`   — run **both** and assert the verdicts AGREE. This is the
//!                differential burn-in: it proves ordeal reaches the same
//!                verdict as Z3 on every rule LOOM already proves. A
//!                disagreement is a hard error (either a real solver bug or a
//!                fragment gap to file upstream on ordeal).
//!
//! Tier-2 (`verify.rs`, the whole-function translation validator) will grow the
//! same boundary later; keeping the seam small and explicit here is what makes
//! that migration tractable.

use std::sync::Arc;

// ============================================================================
// Backend-neutral term DSL
// ============================================================================

/// A backend-neutral bitvector term.
///
/// This is the closed QF_BV fragment the algebraic rules actually exercise —
/// deliberately minimal so both the Z3 and the ordeal lowering are total and
/// obviously faithful. Every variant maps to a single well-defined operation in
/// each backend.
#[derive(Clone, Debug)]
pub enum RuleTerm {
    /// A concrete bitvector constant of the given width.
    Const {
        /// The constant value (low `width` bits are significant).
        value: u128,
        /// Bit width.
        width: u32,
    },
    /// A free bitvector variable of the given width.
    Var {
        /// Variable name.
        name: String,
        /// Bit width.
        width: u32,
    },

    // Arithmetic
    /// Modular addition (`bvadd`).
    Add(Box<RuleTerm>, Box<RuleTerm>),
    /// Modular subtraction (`bvsub`).
    Sub(Box<RuleTerm>, Box<RuleTerm>),
    /// Modular multiplication (`bvmul`).
    Mul(Box<RuleTerm>, Box<RuleTerm>),
    /// Unsigned division (`bvudiv`).
    Udiv(Box<RuleTerm>, Box<RuleTerm>),
    /// Unsigned remainder (`bvurem`).
    Urem(Box<RuleTerm>, Box<RuleTerm>),

    // Bitwise
    /// Bitwise AND (`bvand`).
    And(Box<RuleTerm>, Box<RuleTerm>),
    /// Bitwise OR (`bvor`).
    Or(Box<RuleTerm>, Box<RuleTerm>),
    /// Bitwise XOR (`bvxor`).
    Xor(Box<RuleTerm>, Box<RuleTerm>),

    // Shifts
    /// Logical shift left (`bvshl`).
    Shl(Box<RuleTerm>, Box<RuleTerm>),
    /// Logical shift right (`bvlshr`).
    Lshr(Box<RuleTerm>, Box<RuleTerm>),
    /// Arithmetic shift right (`bvashr`).
    Ashr(Box<RuleTerm>, Box<RuleTerm>),

    // Structural
    /// Bit extraction `[hi:lo]` (inclusive), yielding a `(hi - lo + 1)`-bit term.
    Extract {
        /// High bit index (inclusive).
        hi: u32,
        /// Low bit index (inclusive).
        lo: u32,
        /// Operand.
        arg: Box<RuleTerm>,
    },
    /// Sign-extension by `by` bits (`sign_ext`).
    SignExt {
        /// Number of bits to add.
        by: u32,
        /// Operand.
        arg: Box<RuleTerm>,
    },

    /// If-then-else: `cond ? then_ : else_` (both arms share the result width).
    Ite {
        /// Boolean condition.
        cond: Box<RuleBool>,
        /// Value when `cond` holds.
        then_: Box<RuleTerm>,
        /// Value when `cond` does not hold.
        else_: Box<RuleTerm>,
    },
}

/// A backend-neutral boolean term (used only as `Ite` conditions in the rules).
#[derive(Clone, Debug)]
pub enum RuleBool {
    /// Bitvector equality (`=`).
    Eq(Box<RuleTerm>, Box<RuleTerm>),
    /// Logical negation.
    Not(Box<RuleBool>),
}

// --- Ergonomic constructors (mirror the fluent Z3 builder style) ------------

impl RuleTerm {
    /// A bitvector constant from an unsigned value.
    pub fn from_u64(value: u64, width: u32) -> Self {
        RuleTerm::Const {
            value: value as u128,
            width,
        }
    }

    /// A bitvector constant from a signed value (two's-complement, masked to
    /// `width`).
    pub fn from_i64(value: i64, width: u32) -> Self {
        let mask: u128 = if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };
        RuleTerm::Const {
            value: (value as u128) & mask,
            width,
        }
    }

    /// A free bitvector variable.
    pub fn new_const(name: &str, width: u32) -> Self {
        RuleTerm::Var {
            name: name.to_string(),
            width,
        }
    }

    /// The bit width of this term.
    pub fn width(&self) -> u32 {
        match self {
            RuleTerm::Const { width, .. } | RuleTerm::Var { width, .. } => *width,
            RuleTerm::Add(a, _)
            | RuleTerm::Sub(a, _)
            | RuleTerm::Mul(a, _)
            | RuleTerm::Udiv(a, _)
            | RuleTerm::Urem(a, _)
            | RuleTerm::And(a, _)
            | RuleTerm::Or(a, _)
            | RuleTerm::Xor(a, _)
            | RuleTerm::Shl(a, _)
            | RuleTerm::Lshr(a, _)
            | RuleTerm::Ashr(a, _) => a.width(),
            RuleTerm::Extract { hi, lo, .. } => hi - lo + 1,
            RuleTerm::SignExt { by, arg } => arg.width() + by,
            RuleTerm::Ite { then_, .. } => then_.width(),
        }
    }

    /// `self + rhs`
    pub fn add(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Add(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// `self - rhs`
    pub fn sub(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Sub(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// `self * rhs`
    pub fn mul(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Mul(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// `self /u rhs`
    pub fn udiv(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Udiv(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// `self %u rhs`
    pub fn urem(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Urem(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// `self & rhs`
    pub fn and(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::And(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// `self | rhs`
    pub fn or(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Or(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// `self ^ rhs`
    pub fn xor(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Xor(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// `self << rhs`
    pub fn shl(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Shl(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// `self >>u rhs`
    pub fn lshr(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Lshr(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// `self >>s rhs`
    pub fn ashr(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Ashr(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// Sign-extend by `by` bits.
    pub fn sign_ext(&self, by: u32) -> RuleTerm {
        RuleTerm::SignExt {
            by,
            arg: Box::new(self.clone()),
        }
    }
    /// Extract bits `[hi:lo]`.
    pub fn extract(&self, hi: u32, lo: u32) -> RuleTerm {
        RuleTerm::Extract {
            hi,
            lo,
            arg: Box::new(self.clone()),
        }
    }
    /// Equality condition `self == rhs`.
    pub fn eq_bool(&self, rhs: &RuleTerm) -> RuleBool {
        RuleBool::Eq(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl RuleBool {
    /// Negate this boolean condition.
    pub fn not(&self) -> RuleBool {
        RuleBool::Not(Box::new(self.clone()))
    }
    /// Build `cond ? then_ : else_`.
    pub fn ite(&self, then_: &RuleTerm, else_: &RuleTerm) -> RuleTerm {
        RuleTerm::Ite {
            cond: Box::new(self.clone()),
            then_: Box::new(then_.clone()),
            else_: Box::new(else_.clone()),
        }
    }
}

// ============================================================================
// Solver trait + verdict
// ============================================================================

/// The verdict of proving a rule (`lhs == rhs` for all inputs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuleVerdict {
    /// Proven: `lhs == rhs` holds for all inputs (the negation is `Unsat`).
    Proven,
    /// Disproven: a counterexample exists. The string is a human-readable
    /// rendering of the model.
    Disproven(String),
    /// The solver could not decide (or the query was ill-formed). Treated
    /// conservatively — never as a proof.
    Unknown,
}

impl RuleVerdict {
    /// Whether this verdict is [`RuleVerdict::Proven`].
    pub fn is_proven(&self) -> bool {
        matches!(self, RuleVerdict::Proven)
    }

    /// The counterexample string, if disproven.
    pub fn counterexample(&self) -> Option<&str> {
        match self {
            RuleVerdict::Disproven(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Whether two verdicts agree for the purposes of the `both`-mode
    /// differential. Proven==Proven and Unknown==Unknown must match exactly;
    /// two `Disproven` verdicts agree even if the counterexample text differs
    /// (different solvers legitimately pick different witnesses).
    fn agrees_with(&self, other: &RuleVerdict) -> bool {
        matches!(
            (self, other),
            (RuleVerdict::Proven, RuleVerdict::Proven)
                | (RuleVerdict::Disproven(_), RuleVerdict::Disproven(_))
                | (RuleVerdict::Unknown, RuleVerdict::Unknown)
        )
    }
}

/// A swappable rule-verification backend.
///
/// The single obligation `prove_rule_equiv(lhs, rhs)` asks: does `lhs == rhs`
/// hold for *all* assignments to the free variables? Implementations assert the
/// negation and map the solver result onto [`RuleVerdict`].
pub trait RuleSolver {
    /// Prove `lhs == rhs` for all inputs.
    fn prove_rule_equiv(&self, lhs: &RuleTerm, rhs: &RuleTerm) -> RuleVerdict;

    /// A short human-readable backend name (for diagnostics).
    fn backend_name(&self) -> &'static str;
}

// ============================================================================
// Backend selection
// ============================================================================

/// Which backend(s) the rule verifier should use, from `LOOM_VERIFY_BACKEND`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyBackend {
    /// Z3 only (default).
    Z3,
    /// ordeal only.
    Ordeal,
    /// Both, asserting the verdicts agree (the differential burn-in).
    Both,
}

impl VerifyBackend {
    /// Read the backend from `LOOM_VERIFY_BACKEND` (default: [`VerifyBackend::Z3`]).
    ///
    /// Unrecognized values fall back to the default with no panic — the
    /// verifier must never be *harder* to run than before.
    pub fn from_env() -> Self {
        match std::env::var("LOOM_VERIFY_BACKEND")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("ordeal") => VerifyBackend::Ordeal,
            Some("both") => VerifyBackend::Both,
            _ => VerifyBackend::Z3,
        }
    }
}

/// Resolve the active [`RuleSolver`] from the environment.
///
/// In `both` mode the returned solver runs Z3 and ordeal and asserts they
/// agree on every query.
pub fn active_solver() -> Arc<dyn RuleSolver + Send + Sync> {
    match VerifyBackend::from_env() {
        VerifyBackend::Z3 => Arc::new(Z3RuleSolver),
        VerifyBackend::Ordeal => Arc::new(OrdealRuleSolver),
        VerifyBackend::Both => Arc::new(DifferentialRuleSolver),
    }
}

// ============================================================================
// Z3 backend (the original logic, now behind the trait)
// ============================================================================

/// The Z3-backed rule solver — the historical verification path.
pub struct Z3RuleSolver;

#[cfg(feature = "verification")]
mod z3_backend {
    use super::{RuleBool, RuleSolver, RuleTerm, RuleVerdict, Z3RuleSolver};
    use z3::ast::{BV, Bool};
    use z3::{SatResult, Solver};

    fn lower(term: &RuleTerm) -> BV {
        match term {
            RuleTerm::Const { value, width } => {
                // Widths in the rule fragment are <= 128; the low `width` bits
                // carry the value. `from_u64` covers everything the rules use
                // (all constants fit in 64 bits); wider constants would need a
                // string path, but the fragment never builds them.
                BV::from_u64(*value as u64, *width)
            }
            RuleTerm::Var { name, width } => BV::new_const(name.as_str(), *width),
            RuleTerm::Add(a, b) => lower(a).bvadd(&lower(b)),
            RuleTerm::Sub(a, b) => lower(a).bvsub(&lower(b)),
            RuleTerm::Mul(a, b) => lower(a).bvmul(&lower(b)),
            RuleTerm::Udiv(a, b) => lower(a).bvudiv(&lower(b)),
            RuleTerm::Urem(a, b) => lower(a).bvurem(&lower(b)),
            RuleTerm::And(a, b) => lower(a).bvand(&lower(b)),
            RuleTerm::Or(a, b) => lower(a).bvor(&lower(b)),
            RuleTerm::Xor(a, b) => lower(a).bvxor(&lower(b)),
            RuleTerm::Shl(a, b) => lower(a).bvshl(&lower(b)),
            RuleTerm::Lshr(a, b) => lower(a).bvlshr(&lower(b)),
            RuleTerm::Ashr(a, b) => lower(a).bvashr(&lower(b)),
            RuleTerm::Extract { hi, lo, arg } => lower(arg).extract(*hi, *lo),
            RuleTerm::SignExt { by, arg } => lower(arg).sign_ext(*by),
            RuleTerm::Ite { cond, then_, else_ } => {
                lower_bool(cond).ite(&lower(then_), &lower(else_))
            }
        }
    }

    fn lower_bool(cond: &RuleBool) -> Bool {
        match cond {
            RuleBool::Eq(a, b) => lower(a).eq(&lower(b)),
            RuleBool::Not(inner) => lower_bool(inner).not(),
        }
    }

    impl RuleSolver for Z3RuleSolver {
        fn prove_rule_equiv(&self, lhs: &RuleTerm, rhs: &RuleTerm) -> RuleVerdict {
            let solver = Solver::new();
            let l = lower(lhs);
            let r = lower(rhs);
            // Look for a counterexample: assert lhs != rhs.
            solver.assert(&l.eq(&r).not());
            match solver.check() {
                SatResult::Unsat => RuleVerdict::Proven,
                SatResult::Sat => {
                    let model = solver.get_model().unwrap();
                    RuleVerdict::Disproven(format!("{}", model))
                }
                SatResult::Unknown => RuleVerdict::Unknown,
            }
        }

        fn backend_name(&self) -> &'static str {
            "z3"
        }
    }
}

/// When verification is disabled, the Z3 backend is unavailable and always
/// reports `Unknown` (conservative — never a proof).
#[cfg(not(feature = "verification"))]
impl RuleSolver for Z3RuleSolver {
    fn prove_rule_equiv(&self, _lhs: &RuleTerm, _rhs: &RuleTerm) -> RuleVerdict {
        RuleVerdict::Unknown
    }
    fn backend_name(&self) -> &'static str {
        "z3(disabled)"
    }
}

// ============================================================================
// ordeal backend (certificate-checked QF_BV)
// ============================================================================

/// The ordeal-backed rule solver — pure-Rust, certificate-checked QF_BV.
pub struct OrdealRuleSolver;

#[cfg(feature = "verification")]
mod ordeal_backend {
    use super::{OrdealRuleSolver, RuleBool, RuleSolver, RuleTerm, RuleVerdict};
    use ordeal::lowering;
    use ordeal::{BoolTerm, BvTerm, CheckResult, Solver, Sort};

    fn lower(term: &RuleTerm) -> BvTerm {
        match term {
            RuleTerm::Const { value, width } => BvTerm::Const {
                value: *value,
                sort: Sort::new(*width),
            },
            RuleTerm::Var { name, width } => BvTerm::Var {
                name: name.clone(),
                sort: Sort::new(*width),
            },
            RuleTerm::Add(a, b) => BvTerm::Add(Box::new(lower(a)), Box::new(lower(b))),
            RuleTerm::Sub(a, b) => BvTerm::Sub(Box::new(lower(a)), Box::new(lower(b))),
            RuleTerm::Mul(a, b) => BvTerm::Mul(Box::new(lower(a)), Box::new(lower(b))),
            RuleTerm::Udiv(a, b) => BvTerm::Udiv(Box::new(lower(a)), Box::new(lower(b))),
            // bvurem is a blessed derived op in ordeal::lowering.
            RuleTerm::Urem(a, b) => lowering::bvurem(lower(a), lower(b), a.width()),
            RuleTerm::And(a, b) => BvTerm::And(Box::new(lower(a)), Box::new(lower(b))),
            RuleTerm::Or(a, b) => BvTerm::Or(Box::new(lower(a)), Box::new(lower(b))),
            RuleTerm::Xor(a, b) => BvTerm::Xor(Box::new(lower(a)), Box::new(lower(b))),
            RuleTerm::Shl(a, b) => BvTerm::Shl(Box::new(lower(a)), Box::new(lower(b))),
            RuleTerm::Lshr(a, b) => BvTerm::Lshr(Box::new(lower(a)), Box::new(lower(b))),
            RuleTerm::Ashr(a, b) => BvTerm::Ashr(Box::new(lower(a)), Box::new(lower(b))),
            RuleTerm::Extract { hi, lo, arg } => BvTerm::Extract {
                hi: *hi,
                lo: *lo,
                arg: Box::new(lower(arg)),
            },
            RuleTerm::SignExt { by, arg } => BvTerm::SignExt {
                by: *by,
                arg: Box::new(lower(arg)),
            },
            RuleTerm::Ite { cond, then_, else_ } => BvTerm::Ite {
                cond: Box::new(lower_bool(cond)),
                then_: Box::new(lower(then_)),
                else_: Box::new(lower(else_)),
            },
        }
    }

    fn lower_bool(cond: &RuleBool) -> BoolTerm {
        match cond {
            RuleBool::Eq(a, b) => BoolTerm::Eq(Box::new(lower(a)), Box::new(lower(b))),
            RuleBool::Not(inner) => BoolTerm::Not(Box::new(lower_bool(inner))),
        }
    }

    fn render_model(m: &ordeal::Model) -> String {
        if m.assignments.is_empty() {
            return "(no free variables)".to_string();
        }
        m.assignments
            .iter()
            .map(|(name, value)| format!("{} -> {:#x}", name, value))
            .collect::<Vec<_>>()
            .join("\n")
    }

    impl RuleSolver for OrdealRuleSolver {
        fn prove_rule_equiv(&self, lhs: &RuleTerm, rhs: &RuleTerm) -> RuleVerdict {
            let l = lower(lhs);
            let r = lower(rhs);
            // prove_equiv asserts the negation (lhs != rhs) and runs the
            // certificate-checked pipeline.
            match Solver::prove_equiv(l, r) {
                CheckResult::Unsat(cert) => {
                    // Defence in depth: re-validate the LRAT certificate. A
                    // rejection here would mean an ordeal bug (a certificate
                    // that the checker itself won't accept) — treat it as
                    // "not proven" rather than silently trusting.
                    match cert.recheck() {
                        Ok(()) => RuleVerdict::Proven,
                        Err(_) => RuleVerdict::Unknown,
                    }
                }
                CheckResult::Sat(model) => RuleVerdict::Disproven(render_model(&model)),
                CheckResult::Unknown => RuleVerdict::Unknown,
            }
        }

        fn backend_name(&self) -> &'static str {
            "ordeal"
        }
    }
}

/// When verification is disabled, ordeal is unavailable and always reports
/// `Unknown` (conservative — never a proof).
#[cfg(not(feature = "verification"))]
impl RuleSolver for OrdealRuleSolver {
    fn prove_rule_equiv(&self, _lhs: &RuleTerm, _rhs: &RuleTerm) -> RuleVerdict {
        RuleVerdict::Unknown
    }
    fn backend_name(&self) -> &'static str {
        "ordeal(disabled)"
    }
}

// ============================================================================
// Differential backend (`both`) — the burn-in
// ============================================================================

/// Runs Z3 and ordeal on every query and asserts they AGREE.
///
/// This is how a solver migration is de-risked: for every rule LOOM already
/// proves via Z3, ordeal must reach the same verdict. A disagreement panics —
/// it is either a real solver bug or a fragment gap, and must be investigated
/// (and, if it is an ordeal gap, filed upstream) rather than silently absorbed.
pub struct DifferentialRuleSolver;

impl RuleSolver for DifferentialRuleSolver {
    fn prove_rule_equiv(&self, lhs: &RuleTerm, rhs: &RuleTerm) -> RuleVerdict {
        let z3 = Z3RuleSolver.prove_rule_equiv(lhs, rhs);
        let ordeal = OrdealRuleSolver.prove_rule_equiv(lhs, rhs);
        assert!(
            z3.agrees_with(&ordeal),
            "LOOM_VERIFY_BACKEND=both: solver disagreement!\n  z3     = {:?}\n  ordeal = {:?}\n  lhs    = {:?}\n  rhs    = {:?}",
            z3,
            ordeal,
            lhs,
            rhs
        );
        // Verdicts agree; return the Z3 verdict (they are equivalent for the
        // caller's purposes, and Z3's counterexample string is the historical
        // format the rest of the pipeline already parses).
        z3
    }

    fn backend_name(&self) -> &'static str {
        "both(z3+ordeal)"
    }
}

#[cfg(all(test, feature = "verification"))]
mod tests {
    use super::*;

    fn all_backends() -> Vec<Box<dyn RuleSolver>> {
        vec![
            Box::new(Z3RuleSolver),
            Box::new(OrdealRuleSolver),
            Box::new(DifferentialRuleSolver),
        ]
    }

    #[test]
    fn proves_add_zero_identity_i32() {
        let x = RuleTerm::new_const("x", 32);
        let zero = RuleTerm::from_u64(0, 32);
        let lhs = x.add(&zero);
        for s in all_backends() {
            assert_eq!(
                s.prove_rule_equiv(&lhs, &x),
                RuleVerdict::Proven,
                "{}: x + 0 == x should be proven",
                s.backend_name()
            );
        }
    }

    #[test]
    fn proves_mul_pow2_is_shift_i32() {
        let x = RuleTerm::new_const("x", 32);
        let lhs = x.mul(&RuleTerm::from_u64(8, 32));
        let rhs = x.shl(&RuleTerm::from_u64(3, 32));
        for s in all_backends() {
            assert_eq!(
                s.prove_rule_equiv(&lhs, &rhs),
                RuleVerdict::Proven,
                "{}: x * 8 == x << 3 should be proven",
                s.backend_name()
            );
        }
    }

    #[test]
    fn disproves_false_rule() {
        // x + 1 == x is false; both solvers must find a counterexample.
        let x = RuleTerm::new_const("x", 32);
        let lhs = x.add(&RuleTerm::from_u64(1, 32));
        for s in [
            Box::new(Z3RuleSolver) as Box<dyn RuleSolver>,
            Box::new(OrdealRuleSolver),
        ] {
            let v = s.prove_rule_equiv(&lhs, &x);
            assert!(
                v.counterexample().is_some(),
                "{}: x + 1 == x should be disproven, got {:?}",
                s.backend_name(),
                v
            );
        }
    }

    #[test]
    fn urem_pow2_is_mask_ordeal_and_z3() {
        // x %u 8 == x & 7
        let x = RuleTerm::new_const("x", 32);
        let lhs = x.urem(&RuleTerm::from_u64(8, 32));
        let rhs = x.and(&RuleTerm::from_u64(7, 32));
        assert_eq!(
            DifferentialRuleSolver.prove_rule_equiv(&lhs, &rhs),
            RuleVerdict::Proven
        );
    }
}
