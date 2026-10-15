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
//! Tier-2 slice 1 (issue #313) now routes the *translation validator's* pure
//! bit-vector obligations through this same trait — see
//! [`crate::verify_solver`]. That slice widened the term DSL below from the
//! algebraic-rule fragment to the fragment `verify.rs` actually emits for
//! straight-line integer code (bitwise NOT/NEG, concat, zero-extend, the
//! variable-amount rotates, and the full set of BV comparisons and boolean
//! connectives). Every added variant is still a single well-defined operation
//! in both backends, and each has a differential test below.

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
    /// Signed division (`bvsdiv`) — the TOTAL bitvector operation, not wasm
    /// `idiv_s`.
    ///
    /// This is a pure VALUE operator and carries no trap semantics, by
    /// design. Wasm's `div_s` traps on `b == 0` and on `INT_MIN / -1`; the
    /// bitvector `bvsdiv` is defined everywhere. The trap clause is NOT
    /// represented here because it is not representable here: a trap
    /// predicate has no corresponding node in the solver term, so it could
    /// not be covered by the seam's reflect-and-re-lower self-validation
    /// (#313 slice 1). Putting it in the DSL would mean the seam advertised a
    /// round-trip guarantee over a field the round-trip cannot see.
    ///
    /// Trap semantics are discharged where they already live, by
    /// [`crate::trap_gate`]'s `DefineOrTrap`, and enter the obligation as the
    /// `¬may_trap ⇒ value_eq` guard rather than as part of the value term.
    Sdiv(Box<RuleTerm>, Box<RuleTerm>),
    /// Signed remainder (`bvsrem`) — truncated toward zero, matching wasm
    /// `irem_s` on its non-trapping domain. See [`RuleTerm::Sdiv`] on why no
    /// trap clause appears here.
    Srem(Box<RuleTerm>, Box<RuleTerm>),

    /// Two's-complement negation (`bvneg`).
    Neg(Box<RuleTerm>),

    // Bitwise
    /// Bitwise AND (`bvand`).
    And(Box<RuleTerm>, Box<RuleTerm>),
    /// Bitwise OR (`bvor`).
    Or(Box<RuleTerm>, Box<RuleTerm>),
    /// Bitwise XOR (`bvxor`).
    Xor(Box<RuleTerm>, Box<RuleTerm>),
    /// Bitwise NOT / one's complement (`bvnot`).
    Not(Box<RuleTerm>),

    // Shifts
    /// Logical shift left (`bvshl`).
    Shl(Box<RuleTerm>, Box<RuleTerm>),
    /// Logical shift right (`bvlshr`).
    Lshr(Box<RuleTerm>, Box<RuleTerm>),
    /// Arithmetic shift right (`bvashr`).
    Ashr(Box<RuleTerm>, Box<RuleTerm>),
    /// Rotate left by a *variable* amount (SMT-LIB `ext_rotate_left`); the
    /// amount is taken modulo the width.
    Rotl(Box<RuleTerm>, Box<RuleTerm>),
    /// Rotate right by a *variable* amount (SMT-LIB `ext_rotate_right`); the
    /// amount is taken modulo the width.
    Rotr(Box<RuleTerm>, Box<RuleTerm>),

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
    /// Zero-extension by `by` bits (`zero_ext`).
    ZeroExt {
        /// Number of bits to add.
        by: u32,
        /// Operand.
        arg: Box<RuleTerm>,
    },
    /// Concatenation (`concat`): the first operand becomes the HIGH bits.
    Concat(Box<RuleTerm>, Box<RuleTerm>),

    /// `select(array, index)` — read the byte at `index` of a linear-memory
    /// array. Always 8 bits wide, because loom models linear memory as
    /// `Array(BV32 -> BV8)` and nothing else (#313 slice 4).
    ///
    /// This is the one term whose children are not all bitvectors, so it is
    /// also the one place the fragment stops being closed under the pure-BV
    /// backends: see [`RuleArray`].
    Select {
        /// The array read from.
        array: Box<RuleArray>,
        /// The BV32 byte address read.
        index: Box<RuleTerm>,
    },

    /// An uninterpreted-function application — loom's `pure_call_<F>(args)`
    /// encoding of a call to a provably pure callee (#313 Tier-2 slice 3).
    ///
    /// The function is opaque: nothing is assumed about what it computes.
    /// The ONE thing asserted is congruence — equal argument tuples yield
    /// equal results — and that is exactly what licenses CSE across calls.
    /// It must be asserted positively rather than assumed, because an engine
    /// that treated the call as a constant would also satisfy congruence
    /// while being unsound; see the slice-3 tests.
    ///
    /// Like [`RuleTerm::Select`], this is a term the pure-BV core cannot
    /// express: ordeal's `BvTerm` has no UF node, so a `PureCall` is
    /// *always* extended and never [`ExtBvTerm::Core`].
    PureCall {
        /// The function's name — its identity for congruence.
        name: String,
        /// The argument terms.
        args: Vec<RuleTerm>,
        /// Width of the result bitvector.
        width: u32,
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

/// A backend-neutral **array-sorted** term: wasm linear memory, modelled as
/// the non-extensional `Array(BV32 -> BV8)` both engines already agree on
/// (#313 slice 4).
///
/// Deliberately tiny. There is no array equality, no array-valued `ite` and
/// no array parameter — only a named base array and a chain of byte stores
/// over it. That is exactly what `verify.rs` builds: a fresh
/// `Array::new_const` for the initial memory and for each havoc point, with
/// `store` applied per byte by the little-endian store encoders.
///
/// The narrowness is the point. An array-sorted `ite` (which a control-flow
/// merge over two memory states would produce) is **refused** by the
/// reflection rather than approximated, because neither the round-trip gate
/// nor ordeal's sliver can take it.
#[derive(Clone, Debug)]
pub enum RuleArray {
    /// A free array variable: the initial memory, or a havoc point. Two
    /// distinct havoc points carry distinct names and must never be unified.
    Var {
        /// The array's name — its identity for both backends.
        name: String,
    },
    /// `store(array, index, value)`: a single byte written at a BV32 address.
    Store {
        /// The array being updated.
        array: Box<RuleArray>,
        /// The BV32 byte address written.
        index: Box<RuleTerm>,
        /// The BV8 value written.
        value: Box<RuleTerm>,
    },
}

/// A backend-neutral boolean term.
///
/// In Tier-1 this was only ever an `Ite` condition on the algebraic rules
/// (`Eq` / `Not`). Tier-2 slice 1 (#313) widened it to the comparisons and
/// connectives `verify.rs` emits when it lowers WebAssembly's `i32.lt_s`,
/// `i64.ge_u`, `select`, `if`/`br_if` and friends.
#[derive(Clone, Debug)]
pub enum RuleBool {
    /// Bitvector equality (`=`).
    Eq(Box<RuleTerm>, Box<RuleTerm>),
    /// Unsigned less-than (`bvult`).
    Ult(Box<RuleTerm>, Box<RuleTerm>),
    /// Unsigned less-or-equal (`bvule`).
    Ule(Box<RuleTerm>, Box<RuleTerm>),
    /// Unsigned greater-than (`bvugt`).
    Ugt(Box<RuleTerm>, Box<RuleTerm>),
    /// Unsigned greater-or-equal (`bvuge`).
    Uge(Box<RuleTerm>, Box<RuleTerm>),
    /// Signed less-than (`bvslt`).
    Slt(Box<RuleTerm>, Box<RuleTerm>),
    /// Signed less-or-equal (`bvsle`).
    Sle(Box<RuleTerm>, Box<RuleTerm>),
    /// Signed greater-than (`bvsgt`).
    Sgt(Box<RuleTerm>, Box<RuleTerm>),
    /// Signed greater-or-equal (`bvsge`).
    Sge(Box<RuleTerm>, Box<RuleTerm>),
    /// Logical negation.
    Not(Box<RuleBool>),
    /// Logical conjunction (binary).
    BoolAnd(Box<RuleBool>, Box<RuleBool>),
    /// Logical disjunction (binary).
    BoolOr(Box<RuleBool>, Box<RuleBool>),
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
            RuleTerm::Const { width, .. }
            | RuleTerm::Var { width, .. }
            | RuleTerm::PureCall { width, .. } => *width,
            RuleTerm::Add(a, _)
            | RuleTerm::Sub(a, _)
            | RuleTerm::Mul(a, _)
            | RuleTerm::Udiv(a, _)
            | RuleTerm::Urem(a, _)
            | RuleTerm::Sdiv(a, _)
            | RuleTerm::Srem(a, _)
            | RuleTerm::And(a, _)
            | RuleTerm::Or(a, _)
            | RuleTerm::Xor(a, _)
            | RuleTerm::Shl(a, _)
            | RuleTerm::Lshr(a, _)
            | RuleTerm::Ashr(a, _)
            | RuleTerm::Rotl(a, _)
            | RuleTerm::Rotr(a, _)
            | RuleTerm::Neg(a)
            | RuleTerm::Not(a) => a.width(),
            RuleTerm::Extract { hi, lo, .. } => hi - lo + 1,
            RuleTerm::SignExt { by, arg } | RuleTerm::ZeroExt { by, arg } => arg.width() + by,
            RuleTerm::Concat(a, b) => a.width() + b.width(),
            RuleTerm::Ite { then_, .. } => then_.width(),
            // Linear memory is byte-addressed: a select is always one byte.
            RuleTerm::Select { .. } => 8,
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
    /// `-self` (two's-complement negation)
    pub fn neg(&self) -> RuleTerm {
        RuleTerm::Neg(Box::new(self.clone()))
    }
    /// `!self` (bitwise NOT / one's complement)
    pub fn bit_not(&self) -> RuleTerm {
        RuleTerm::Not(Box::new(self.clone()))
    }
    /// Rotate left by a variable amount.
    pub fn rotl(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Rotl(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// Rotate right by a variable amount.
    pub fn rotr(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Rotr(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// Concatenate: `self` becomes the HIGH bits.
    pub fn concat(&self, rhs: &RuleTerm) -> RuleTerm {
        RuleTerm::Concat(Box::new(self.clone()), Box::new(rhs.clone()))
    }
    /// Zero-extend by `by` bits.
    pub fn zero_ext(&self, by: u32) -> RuleTerm {
        RuleTerm::ZeroExt {
            by,
            arg: Box::new(self.clone()),
        }
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
    pub(crate) fn agrees_with(&self, other: &RuleVerdict) -> bool {
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
pub(crate) mod z3_backend {
    use super::{RuleArray, RuleBool, RuleSolver, RuleTerm, RuleVerdict, Z3RuleSolver};
    use z3::ast::{Array, BV, Bool};
    use z3::{FuncDecl, SatResult, Solver, Sort};

    /// Lower a neutral term to a Z3 bitvector.
    ///
    /// Exposed within the crate because Tier-2's reflection
    /// ([`crate::verify_solver`]) re-lowers every term it reflects and demands
    /// the result be the **identical** Z3 AST node as the obligation it came
    /// from. That round-trip is what makes the reflection self-validating, and
    /// it is only meaningful if it goes through exactly the lowering the Z3
    /// backend itself uses.
    pub(crate) fn lower_to_z3(term: &RuleTerm) -> BV {
        lower(term)
    }

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
            // Deliberately the SAME nodes the reflection reads back
            // (`BSDIV`/`BSREM`). Lowering a reflected `bvsdiv` into
            // trap_gate's abs/udiv/negate formulation instead would produce a
            // different AST and fail the round-trip on every division, even
            // though both terms are equivalent.
            RuleTerm::Sdiv(a, b) => lower(a).bvsdiv(&lower(b)),
            RuleTerm::Srem(a, b) => lower(a).bvsrem(&lower(b)),
            RuleTerm::Neg(a) => lower(a).bvneg(),
            RuleTerm::And(a, b) => lower(a).bvand(&lower(b)),
            RuleTerm::Or(a, b) => lower(a).bvor(&lower(b)),
            RuleTerm::Xor(a, b) => lower(a).bvxor(&lower(b)),
            RuleTerm::Not(a) => lower(a).bvnot(),
            RuleTerm::Shl(a, b) => lower(a).bvshl(&lower(b)),
            RuleTerm::Lshr(a, b) => lower(a).bvlshr(&lower(b)),
            RuleTerm::Ashr(a, b) => lower(a).bvashr(&lower(b)),
            RuleTerm::Rotl(a, b) => lower(a).bvrotl(&lower(b)),
            RuleTerm::Rotr(a, b) => lower(a).bvrotr(&lower(b)),
            RuleTerm::Extract { hi, lo, arg } => lower(arg).extract(*hi, *lo),
            RuleTerm::SignExt { by, arg } => lower(arg).sign_ext(*by),
            RuleTerm::ZeroExt { by, arg } => lower(arg).zero_ext(*by),
            RuleTerm::Concat(a, b) => lower(a).concat(&lower(b)),
            RuleTerm::Ite { cond, then_, else_ } => {
                lower_bool(cond).ite(&lower(then_), &lower(else_))
            }
            // #313 slice 4. `as_bv()` cannot fail here: `lower_array` only
            // ever builds `Array(BV32 -> BV8)`, so a select off it is a BV8.
            // The `expect` is a sort invariant of this very function, not a
            // claim about anything the reflection handed us.
            // Rebuild the SAME Z3 function decl. Z3 symbol-deduplicates
            // decls: the same name + domain + range in one context resolves
            // to the identical decl, which is both what gives congruence its
            // teeth and what lets the reflect-and-re-lower self-validation
            // gate see a byte-identical AST. `verify.rs` builds the forward
            // encoding the same way.
            RuleTerm::PureCall { name, args, width } => {
                let lowered: Vec<BV> = args.iter().map(lower).collect();
                // Domain sorts come from the ARGUMENT WIDTHS, not from the
                // lowered nodes, so this matches the forward encoding even if
                // a lowering ever introduces a differently-sorted temporary.
                let domain: Vec<Sort> = args.iter().map(|a| Sort::bitvector(a.width())).collect();
                let domain_refs: Vec<&Sort> = domain.iter().collect();
                let range = Sort::bitvector(*width);
                let decl = FuncDecl::new(name.as_str(), &domain_refs, &range);
                let arg_refs: Vec<&dyn z3::ast::Ast> =
                    lowered.iter().map(|a| a as &dyn z3::ast::Ast).collect();
                // Infallible: `range` is a bitvector sort by construction one
                // line above, so the application is a bitvector.
                decl.apply(&arg_refs)
                    .as_bv()
                    .expect("pure_call range sort is a bitvector by construction")
            }
            RuleTerm::Select { array, index } => lower_array(array)
                .select(&lower(index))
                .as_bv()
                .expect("select on an Array(BV32 -> BV8) is a BV8 by construction"),
        }
    }

    /// Lower a neutral array term to the Z3 `Array(BV32 -> BV8)` that
    /// `verify.rs` builds, node for node.
    ///
    /// The sorts are hard-coded rather than carried on [`RuleArray`] because
    /// the fragment admits exactly one array sort; a second one would need a
    /// reflection change here, not a silent widening.
    fn lower_array(array: &RuleArray) -> Array {
        match array {
            RuleArray::Var { name } => {
                Array::new_const(name.as_str(), &Sort::bitvector(32), &Sort::bitvector(8))
            }
            RuleArray::Store {
                array,
                index,
                value,
            } => lower_array(array).store(&lower(index), &lower(value)),
        }
    }

    fn lower_bool(cond: &RuleBool) -> Bool {
        match cond {
            RuleBool::Eq(a, b) => lower(a).eq(&lower(b)),
            RuleBool::Ult(a, b) => lower(a).bvult(&lower(b)),
            RuleBool::Ule(a, b) => lower(a).bvule(&lower(b)),
            RuleBool::Ugt(a, b) => lower(a).bvugt(&lower(b)),
            RuleBool::Uge(a, b) => lower(a).bvuge(&lower(b)),
            RuleBool::Slt(a, b) => lower(a).bvslt(&lower(b)),
            RuleBool::Sle(a, b) => lower(a).bvsle(&lower(b)),
            RuleBool::Sgt(a, b) => lower(a).bvsgt(&lower(b)),
            RuleBool::Sge(a, b) => lower(a).bvsge(&lower(b)),
            RuleBool::Not(inner) => lower_bool(inner).not(),
            RuleBool::BoolAnd(a, b) => Bool::and(&[lower_bool(a), lower_bool(b)]),
            RuleBool::BoolOr(a, b) => Bool::or(&[lower_bool(a), lower_bool(b)]),
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
pub(crate) mod ordeal_backend {
    use super::{OrdealRuleSolver, RuleArray, RuleBool, RuleSolver, RuleTerm, RuleVerdict};
    use ordeal::lowering;
    use ordeal::sliver::{ArrayTerm, ExtBoolTerm, ExtBvTerm, ExtOp};
    use ordeal::{BoolTerm, BvTerm, CheckResult, Solver, Sort};

    /// A gap in the **extended** (sliver) language: an operator that exists
    /// in the closed core but has no [`ExtOp`] variant, so it cannot sit
    /// above a `select`.
    ///
    /// `ExtOp` lifts most of the core over extended children, but not
    /// `Ite`, `Urem`, `Neg`, `Not`, `Rotl`, `Sdiv` or `Srem`. For a
    /// select-free subtree that costs nothing — it collapses into a single
    /// [`ExtBvTerm::Core`] node and the operator is used in its blessed core
    /// form. It only bites when memory is genuinely underneath, and then the
    /// honest answer is to refuse: deriving, say, `bvnot` by hand in the
    /// extended language would introduce a SECOND derivation beside
    /// `ordeal::lowering`'s blessed one, which is exactly the kind of
    /// silently-diverging duplicate the seam exists to avoid.
    ///
    /// `Ite` is the one that matters in practice and it is tracked upstream
    /// as ordeal#231.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct ExtGap(pub(crate) &'static str);

    /// Combine two lowered children under an operator that **is** liftable:
    /// stay in the core when both children did, lift otherwise.
    ///
    /// Keeping the core node whenever possible is not an optimisation — it
    /// is what preserves the invariant that [`lower_ext`] returns
    /// [`ExtBvTerm::Core`] *exactly* when the subtree is select-free, which
    /// is in turn how [`lower_to_ordeal`] decides whether a term is pure.
    fn lift(
        a: ExtBvTerm,
        b: ExtBvTerm,
        core: impl FnOnce(Box<BvTerm>, Box<BvTerm>) -> BvTerm,
        ext: impl FnOnce(ExtBvTerm, ExtBvTerm) -> ExtOp,
    ) -> ExtBvTerm {
        match (a, b) {
            (ExtBvTerm::Core(x), ExtBvTerm::Core(y)) => {
                ExtBvTerm::Core(core(Box::new(x), Box::new(y)))
            }
            (x, y) => ExtBvTerm::Op(Box::new(ext(x, y))),
        }
    }

    /// Combine two lowered children under an operator that is **not**
    /// liftable: only the all-core case is expressible.
    fn core_only(
        a: ExtBvTerm,
        b: ExtBvTerm,
        what: &'static str,
        core: impl FnOnce(BvTerm, BvTerm) -> BvTerm,
    ) -> Result<ExtBvTerm, ExtGap> {
        match (a, b) {
            (ExtBvTerm::Core(x), ExtBvTerm::Core(y)) => Ok(ExtBvTerm::Core(core(x, y))),
            _ => Err(ExtGap(what)),
        }
    }

    /// Lower a neutral term into ordeal's extended (array/UF sliver)
    /// language.
    ///
    /// **Invariant:** the result is [`ExtBvTerm::Core`] if and only if the
    /// term contains no [`RuleTerm::Select`] **and** no
    /// [`RuleTerm::PureCall`]. Both are terms the pure-BV core cannot
    /// express — ordeal's `BvTerm` has neither an array nor a UF node — so
    /// either one forces the extended language. Everything that depends on
    /// "is this obligation pure?" reads that off the result rather than
    /// walking the tree a second time.
    pub(crate) fn lower_ext(term: &RuleTerm) -> Result<ExtBvTerm, ExtGap> {
        use ExtBvTerm::Core;
        let w = term.width();
        Ok(match term {
            RuleTerm::Const { value, width } => Core(BvTerm::Const {
                value: *value,
                sort: Sort::new(*width),
            }),
            RuleTerm::Var { name, width } => Core(BvTerm::Var {
                name: name.clone(),
                sort: Sort::new(*width),
            }),

            RuleTerm::Add(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Add, ExtOp::Add),
            RuleTerm::Sub(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Sub, ExtOp::Sub),
            RuleTerm::Mul(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Mul, ExtOp::Mul),
            RuleTerm::Udiv(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Udiv, ExtOp::Udiv),
            RuleTerm::And(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::And, ExtOp::And),
            RuleTerm::Or(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Or, ExtOp::Or),
            RuleTerm::Xor(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Xor, ExtOp::Xor),
            RuleTerm::Shl(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Shl, ExtOp::Shl),
            RuleTerm::Lshr(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Lshr, ExtOp::Lshr),
            RuleTerm::Ashr(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Ashr, ExtOp::Ashr),
            RuleTerm::Rotr(a, b) => lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Rotr, ExtOp::Rotr),
            RuleTerm::Concat(a, b) => {
                lift(lower_ext(a)?, lower_ext(b)?, BvTerm::Concat, ExtOp::Concat)
            }

            // Core-only, so memory under one of these still defers.
            //
            // They are no longer "derived" in the sense the old note here
            // claimed. As of ordeal 0.30.0 `lowering::bvurem` and friends
            // ignore their `width` argument and build the primitive
            // `BvTerm::{Urem,Sdiv,Srem,Rotl}` node — exactly what
            // `ExtOp::{Urem,Sdiv,Srem,Rotl}` lowers to — so lifting them
            // would hand no derivation to anyone and could not disagree
            // about the div_s/rem_s asymmetry.
            //
            // They are left core-only anyway, deliberately: `core_only` is
            // the ONLY remaining constructor of `ExtGap`, so lifting these
            // four would make `ExtGap` uninstantiable and render
            // `GoalError::Gap` and `DeferReason::MemoryUnderUnliftedOp`
            // unreachable. Removing that deferral path is a larger change
            // than this one, and leaving a provably-dead error arm in place
            // is the shape this codebase keeps having to correct. See #416.
            RuleTerm::Urem(a, b) => {
                core_only(lower_ext(a)?, lower_ext(b)?, "memory under urem", |x, y| {
                    lowering::bvurem(x, y, a.width())
                })?
            }
            RuleTerm::Sdiv(a, b) => {
                core_only(lower_ext(a)?, lower_ext(b)?, "memory under sdiv", |x, y| {
                    lowering::bvsdiv(x, y, a.width())
                })?
            }
            RuleTerm::Srem(a, b) => {
                core_only(lower_ext(a)?, lower_ext(b)?, "memory under srem", |x, y| {
                    lowering::bvsrem(x, y, a.width())
                })?
            }
            RuleTerm::Rotl(a, b) => {
                core_only(lower_ext(a)?, lower_ext(b)?, "memory under rotl", |x, y| {
                    lowering::bvrotl(x, y, a.width())
                })?
            }

            RuleTerm::Neg(a) => match lower_ext(a)? {
                Core(x) => Core(lowering::bvneg(x, a.width())),
                ext => ExtBvTerm::Op(Box::new(ExtOp::Neg(ext))),
            },
            RuleTerm::Not(a) => match lower_ext(a)? {
                Core(x) => Core(lowering::bvnot(x, a.width())),
                ext => ExtBvTerm::Op(Box::new(ExtOp::Not(ext))),
            },

            RuleTerm::Extract { hi, lo, arg } => match lower_ext(arg)? {
                Core(x) => Core(BvTerm::Extract {
                    hi: *hi,
                    lo: *lo,
                    arg: Box::new(x),
                }),
                x => ExtBvTerm::Op(Box::new(ExtOp::Extract {
                    hi: *hi,
                    lo: *lo,
                    arg: x,
                })),
            },
            RuleTerm::SignExt { by, arg } => match lower_ext(arg)? {
                Core(x) => Core(BvTerm::SignExt {
                    by: *by,
                    arg: Box::new(x),
                }),
                x => ExtBvTerm::Op(Box::new(ExtOp::SignExt { by: *by, arg: x })),
            },
            RuleTerm::ZeroExt { by, arg } => match lower_ext(arg)? {
                Core(x) => Core(BvTerm::ZeroExt {
                    by: *by,
                    arg: Box::new(x),
                }),
                x => ExtBvTerm::Op(Box::new(ExtOp::ZeroExt { by: *by, arg: x })),
            },

            // `ExtOp` has no `Ite`, so a select under a conditional cannot be
            // expressed. That is ordeal#231, and it gates memory under
            // control flow as well as the slice-3 UF work.
            // Stay in the core when the whole conditional is select-free;
            // otherwise build `ExtOp::Ite`, which ordeal 0.29.0 added (#416,
            // closing ordeal#231). The core branch is not an optimisation —
            // it is what preserves the `Core` iff select-free-and-call-free
            // invariant that
            // `lower_to_ordeal` reads purity off.
            RuleTerm::Ite { cond, then_, else_ } => {
                match (lower_bool_ext(cond)?, lower_ext(then_)?, lower_ext(else_)?) {
                    (Some(c), Core(t), Core(e)) => Core(BvTerm::Ite {
                        cond: Box::new(c),
                        then_: Box::new(t),
                        else_: Box::new(e),
                    }),
                    (_, t, e) => ExtBvTerm::Op(Box::new(ExtOp::Ite {
                        cond: lower_bool_to_ext(cond)?,
                        then_: t,
                        else_: e,
                    })),
                }
            }

            // Always extended, never `Core`: ordeal's core `BvTerm` has no
            // UF node, so there is nothing to stay in the core AS. This is
            // why the invariant above reads "select-free AND call-free".
            RuleTerm::PureCall { name, args, width } => {
                let mut lowered = Vec::with_capacity(args.len());
                for a in args {
                    lowered.push(lower_ext(a)?);
                }
                ExtBvTerm::PureCall {
                    name: name.clone(),
                    args: lowered,
                    sort: Sort::new(*width),
                }
            }
            RuleTerm::Select { array, index } => {
                debug_assert_eq!(w, 8, "a select is a byte read");
                ExtBvTerm::Select {
                    array: Box::new(lower_array_ext(array)?),
                    index: Box::new(lower_ext(index)?),
                }
            }
        })
    }

    /// Lower a neutral array term to ordeal's [`ArrayTerm`].
    fn lower_array_ext(array: &RuleArray) -> Result<ArrayTerm, ExtGap> {
        Ok(match array {
            RuleArray::Var { name } => ArrayTerm::Var { name: name.clone() },
            RuleArray::Store {
                array,
                index,
                value,
            } => ArrayTerm::Store {
                array: Box::new(lower_array_ext(array)?),
                index: Box::new(lower_ext(index)?),
                value: Box::new(lower_ext(value)?),
            },
        })
    }

    /// Lower a boolean, which is expressible only when every operand stayed
    /// in the core — `ExtBoolTerm` exists, but it is only reachable from the
    /// top-level goal, never from an `Ite` condition inside a term.
    ///
    /// `Ok(None)` means "a select appeared under this condition", which the
    /// caller turns into the `ite` gap.
    fn lower_bool_ext(cond: &RuleBool) -> Result<Option<BoolTerm>, ExtGap> {
        let pair = |a: &RuleTerm, b: &RuleTerm| -> Result<Option<(BvTerm, BvTerm)>, ExtGap> {
            match (lower_ext(a)?, lower_ext(b)?) {
                (ExtBvTerm::Core(x), ExtBvTerm::Core(y)) => Ok(Some((x, y))),
                _ => Ok(None),
            }
        };
        macro_rules! cmp {
            ($a:expr, $b:expr, $ctor:path) => {
                match pair($a, $b)? {
                    Some((x, y)) => Some($ctor(Box::new(x), Box::new(y))),
                    None => None,
                }
            };
        }
        Ok(match cond {
            RuleBool::Eq(a, b) => cmp!(a, b, BoolTerm::Eq),
            RuleBool::Ult(a, b) => cmp!(a, b, BoolTerm::Ult),
            RuleBool::Ule(a, b) => cmp!(a, b, BoolTerm::Ule),
            RuleBool::Ugt(a, b) => cmp!(a, b, BoolTerm::Ugt),
            RuleBool::Uge(a, b) => cmp!(a, b, BoolTerm::Uge),
            RuleBool::Slt(a, b) => cmp!(a, b, BoolTerm::Slt),
            RuleBool::Sle(a, b) => cmp!(a, b, BoolTerm::Sle),
            RuleBool::Sgt(a, b) => cmp!(a, b, BoolTerm::Sgt),
            RuleBool::Sge(a, b) => cmp!(a, b, BoolTerm::Sge),
            RuleBool::Not(inner) => lower_bool_ext(inner)?.map(|c| BoolTerm::Not(Box::new(c))),
            RuleBool::BoolAnd(a, b) => match (lower_bool_ext(a)?, lower_bool_ext(b)?) {
                (Some(x), Some(y)) => Some(BoolTerm::And(Box::new(x), Box::new(y))),
                _ => None,
            },
            RuleBool::BoolOr(a, b) => match (lower_bool_ext(a)?, lower_bool_ext(b)?) {
                (Some(x), Some(y)) => Some(BoolTerm::Or(Box::new(x), Box::new(y))),
                _ => None,
            },
        })
    }

    /// Lower a neutral condition into the **extended** boolean language.
    ///
    /// Distinct from [`lower_bool_ext`], which produces a *core* [`BoolTerm`]
    /// and reports `Ok(None)` the moment a `select` appears. That is not a
    /// stepping stone to this one: [`ExtBoolTerm`] has **no `Core` variant**
    /// — unlike [`ExtBvTerm`], which embeds a pure core term verbatim — so an
    /// already-lowered core `BoolTerm` cannot be wrapped. The comparison has
    /// to be rebuilt over [`ExtBvTerm`] children instead.
    ///
    /// The mapping is total: every `RuleBool` variant has an `ExtBoolTerm`
    /// counterpart, so this never widens the refusal surface. It can still
    /// fail, but only by propagating an [`ExtGap`] from a child term.
    fn lower_bool_to_ext(cond: &RuleBool) -> Result<ExtBoolTerm, ExtGap> {
        macro_rules! cmp {
            ($a:expr, $b:expr, $ctor:path) => {
                $ctor(lower_ext($a)?, lower_ext($b)?)
            };
        }
        Ok(match cond {
            RuleBool::Eq(a, b) => cmp!(a, b, ExtBoolTerm::Eq),
            RuleBool::Ult(a, b) => cmp!(a, b, ExtBoolTerm::Ult),
            RuleBool::Ule(a, b) => cmp!(a, b, ExtBoolTerm::Ule),
            RuleBool::Ugt(a, b) => cmp!(a, b, ExtBoolTerm::Ugt),
            RuleBool::Uge(a, b) => cmp!(a, b, ExtBoolTerm::Uge),
            RuleBool::Slt(a, b) => cmp!(a, b, ExtBoolTerm::Slt),
            RuleBool::Sle(a, b) => cmp!(a, b, ExtBoolTerm::Sle),
            RuleBool::Sgt(a, b) => cmp!(a, b, ExtBoolTerm::Sgt),
            RuleBool::Sge(a, b) => cmp!(a, b, ExtBoolTerm::Sge),
            RuleBool::Not(inner) => ExtBoolTerm::Not(Box::new(lower_bool_to_ext(inner)?)),
            RuleBool::BoolAnd(a, b) => ExtBoolTerm::And(
                Box::new(lower_bool_to_ext(a)?),
                Box::new(lower_bool_to_ext(b)?),
            ),
            RuleBool::BoolOr(a, b) => ExtBoolTerm::Or(
                Box::new(lower_bool_to_ext(a)?),
                Box::new(lower_bool_to_ext(b)?),
            ),
        })
    }

    /// Lower a neutral term to an ordeal `BvTerm`.
    ///
    /// Exposed within the crate so Tier-2 ([`crate::verify_solver`]) can drive
    /// a *deadline-bounded* ordeal solve (`check_with_deadline`) rather than
    /// the unbounded `Solver::prove_equiv` the rule verifier uses: a
    /// whole-function obligation can be far larger than an algebraic rule, and
    /// an unbounded solve would turn a slow query into a hang instead of a
    /// conservative revert.
    /// `None` when the term reads memory, because a `select` has no closed-core
    /// form — such an obligation belongs on the sliver path
    /// ([`lower_ext`]), never on this one. Returning an `Option` rather than
    /// panicking keeps the pure path total and makes the memory case an
    /// explicit, conservative `Unknown` at every call site.
    pub(crate) fn lower_to_ordeal(term: &RuleTerm) -> Option<BvTerm> {
        match lower_ext(term) {
            // The `lower_ext` invariant: `Core` iff select-free AND
            // call-free. A `pure_call` has no closed-core form either, so
            // it belongs on the sliver path exactly as a `select` does.
            Ok(ExtBvTerm::Core(bv)) => Some(bv),
            _ => None,
        }
    }

    /// A lowered equivalence goal, ready to assert on a [`Solver`].
    pub(crate) enum GoalLowering {
        /// Neither side reads memory: the closed-core `Ne` goal, exactly as
        /// slices 1-2 built it.
        Core(BoolTerm),
        /// A memory obligation, already eliminated into the closed core by
        /// the array sliver (eager read-over-write + array congruence). No
        /// array construct remains; the ordinary blast pipeline decides it.
        Sliver(Vec<BoolTerm>),
    }

    /// Why an equivalence goal could not be lowered.
    #[derive(Debug)]
    pub(crate) enum GoalError {
        /// An operator with no extended form sits above a `select`.
        Gap(ExtGap),
        /// ordeal's sliver refused the query outright.
        Sliver(ordeal::sliver::SliverError),
    }

    /// Lower the goal `lhs != rhs`, routing through the array sliver when
    /// either side reads memory.
    ///
    /// Deliberately NOT `Solver::prove_equiv_sliver`/`check_sliver`. Those
    /// are one-shot helpers that run the **unbounded** `Solver::check` and
    /// collapse every `SliverError` into `Unknown`. Both are wrong here: the
    /// deadline is what turns a slow memory query into a conservative revert
    /// instead of a hang (loom#347 is exactly that failure mode), and an
    /// out-of-sliver query must surface as a REFUSAL the seam can defer to
    /// the incumbent on, not as an undecided verdict the call site turns
    /// into a revert. Lowering here and asserting the core ourselves keeps
    /// both properties.
    pub(crate) fn lower_goal_ne(lhs: &RuleTerm, rhs: &RuleTerm) -> Result<GoalLowering, GoalError> {
        let l = lower_ext(lhs).map_err(GoalError::Gap)?;
        let r = lower_ext(rhs).map_err(GoalError::Gap)?;
        match (l, r) {
            // The `lower_ext` invariant: both `Core` iff neither side reads
            // memory, so the pure path is byte-identical to slices 1-2.
            (ExtBvTerm::Core(a), ExtBvTerm::Core(b)) => {
                Ok(GoalLowering::Core(BoolTerm::Ne(Box::new(a), Box::new(b))))
            }
            (a, b) => ordeal::sliver::lower(&[ExtBoolTerm::Ne(a, b)])
                .map(GoalLowering::Sliver)
                .map_err(GoalError::Sliver),
        }
    }

    /// Render an ordeal model the way the rule solver does (shared with
    /// Tier-2's counterexample reporting).
    pub(crate) fn render_ordeal_model(m: &ordeal::Model) -> String {
        render_model(m)
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
            // Tier-1's algebraic rules never touch memory. If one ever does,
            // `Unknown` is the conservative answer — the rule is simply not
            // proven, and the caller keeps the original.
            let (Some(l), Some(r)) = (lower_to_ordeal(lhs), lower_to_ordeal(rhs)) else {
                return RuleVerdict::Unknown;
            };
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

    // ------------------------------------------------------------------
    // #313 Tier-2 slice 1 widened this DSL. Each variant added there needs
    // its own evidence, because the round-trip gate in `verify_solver` only
    // pins the *Z3* reading of a neutral term — it says nothing about how
    // ordeal reads it. These tests close that gap the same way Tier-1 does:
    // a known identity, proven under BOTH engines, with `DifferentialRuleSolver`
    // panicking if the verdicts disagree.
    // ------------------------------------------------------------------

    /// Every variant's declared `width()` must equal the width Z3 gives its
    /// lowering. A wrong width on `Concat`/`ZeroExt` would silently change
    /// which formula gets proven.
    #[test]
    fn declared_width_matches_the_z3_lowering() {
        let x = RuleTerm::new_const("x", 32);
        let y = RuleTerm::new_const("y", 32);
        let w = RuleTerm::new_const("w", 64);
        let terms = vec![
            x.add(&y),
            x.sub(&y),
            x.mul(&y),
            x.udiv(&y),
            x.urem(&y),
            x.neg(),
            x.and(&y),
            x.or(&y),
            x.xor(&y),
            x.bit_not(),
            x.shl(&y),
            x.lshr(&y),
            x.ashr(&y),
            x.rotl(&y),
            x.rotr(&y),
            x.extract(15, 0),
            w.extract(47, 16),
            x.sign_ext(32),
            x.zero_ext(32),
            x.concat(&y),
            x.eq_bool(&y).ite(&x, &y),
        ];
        for t in &terms {
            assert_eq!(
                t.width(),
                z3_backend::lower_to_z3(t).get_size(),
                "width mismatch for {:?}",
                t
            );
        }
    }

    /// Prove `lhs == rhs` on Z3, on ordeal, and differentially.
    fn proven_everywhere(lhs: &RuleTerm, rhs: &RuleTerm, what: &str) {
        for s in all_backends() {
            assert_eq!(
                s.prove_rule_equiv(lhs, rhs),
                RuleVerdict::Proven,
                "{}: {} should be proven",
                s.backend_name(),
                what
            );
        }
    }

    #[test]
    fn bitwise_not_double_negation_is_identity() {
        let x = RuleTerm::new_const("x", 32);
        proven_everywhere(&x.bit_not().bit_not(), &x, "~~x == x");
        // ~x == -x - 1 pins the SIGN of the derivation, which `~~x == x`
        // (true for any involution) would not.
        proven_everywhere(
            &x.bit_not(),
            &x.neg().sub(&RuleTerm::from_u64(1, 32)),
            "~x == -x - 1",
        );
    }

    #[test]
    fn negation_matches_twos_complement() {
        let x = RuleTerm::new_const("x", 32);
        proven_everywhere(
            &x.neg(),
            &x.bit_not().add(&RuleTerm::from_u64(1, 32)),
            "-x == ~x + 1",
        );
    }

    #[test]
    fn concat_and_extract_round_trip_a_64_bit_value() {
        let w = RuleTerm::new_const("w", 64);
        proven_everywhere(
            &w.extract(63, 32).concat(&w.extract(31, 0)),
            &w,
            "concat(hi, lo) == w",
        );
    }

    #[test]
    fn zero_extend_is_not_sign_extend() {
        let x = RuleTerm::new_const("x", 32);
        // Narrowing back recovers the original under BOTH extensions...
        proven_everywhere(&x.zero_ext(32).extract(31, 0), &x, "trunc(zext(x)) == x");
        proven_everywhere(&x.sign_ext(32).extract(31, 0), &x, "trunc(sext(x)) == x");
        // ...but the HIGH half distinguishes them: zext's is zero, sext's is
        // the replicated sign bit. Without this the two variants could be
        // swapped and the tests above would still pass.
        proven_everywhere(
            &x.zero_ext(32).extract(63, 32),
            &RuleTerm::from_u64(0, 32),
            "hi(zext(x)) == 0",
        );
        proven_everywhere(
            &x.sign_ext(32).extract(63, 32),
            &x.ashr(&RuleTerm::from_u64(31, 32)),
            "hi(sext(x)) == x >>s 31",
        );
    }

    #[test]
    fn rotates_are_inverse_at_complementary_amounts() {
        let x = RuleTerm::new_const("x", 32);
        let k = RuleTerm::from_u64(8, 32);
        let k_bar = RuleTerm::from_u64(24, 32);
        proven_everywhere(&x.rotl(&k), &x.rotr(&k_bar), "rotl(x,8) == rotr(x,24)");
        // A variable amount exercises ordeal's derived `rotl = rotr(a, 0 - b)`
        // rather than a constant-folded shape.
        let n = RuleTerm::new_const("n", 32);
        proven_everywhere(&x.rotl(&n).rotr(&n), &x, "rotr(rotl(x,n),n) == x for all n");
    }

    #[test]
    fn comparisons_agree_with_their_negations() {
        let x = RuleTerm::new_const("x", 32);
        let y = RuleTerm::new_const("y", 32);
        let one = RuleTerm::from_u64(1, 32);
        let zero = RuleTerm::from_u64(0, 32);
        // Each comparison is pinned against the negation of its complement,
        // which is what would break if a signed variant were lowered as an
        // unsigned one (or an operand order were flipped).
        let pairs: Vec<(RuleBool, RuleBool)> = vec![
            (RuleBool::Ult(bx(&x), bx(&y)), RuleBool::Uge(bx(&x), bx(&y))),
            (RuleBool::Ule(bx(&x), bx(&y)), RuleBool::Ugt(bx(&x), bx(&y))),
            (RuleBool::Slt(bx(&x), bx(&y)), RuleBool::Sge(bx(&x), bx(&y))),
            (RuleBool::Sle(bx(&x), bx(&y)), RuleBool::Sgt(bx(&x), bx(&y))),
        ];
        for (a, b) in &pairs {
            proven_everywhere(
                &a.ite(&one, &zero),
                &b.ite(&zero, &one),
                "cmp == not(complement)",
            );
        }
        // Signed and unsigned must NOT coincide: this is disproven, which is
        // what stops the four identities above from being vacuous.
        let signed = RuleBool::Slt(bx(&x), bx(&y)).ite(&one, &zero);
        let unsigned = RuleBool::Ult(bx(&x), bx(&y)).ite(&one, &zero);
        for s in [
            Box::new(Z3RuleSolver) as Box<dyn RuleSolver>,
            Box::new(OrdealRuleSolver),
        ] {
            assert!(
                s.prove_rule_equiv(&signed, &unsigned)
                    .counterexample()
                    .is_some(),
                "{}: slt and ult must differ",
                s.backend_name()
            );
        }
    }

    #[test]
    fn boolean_connectives_follow_de_morgan() {
        let x = RuleTerm::new_const("x", 32);
        let y = RuleTerm::new_const("y", 32);
        let one = RuleTerm::from_u64(1, 32);
        let zero = RuleTerm::from_u64(0, 32);
        let p = RuleBool::Ult(bx(&x), bx(&y));
        let q = RuleBool::Eq(bx(&x), bx(&zero));
        let and = RuleBool::BoolAnd(Box::new(p.clone()), Box::new(q.clone()));
        let nor_of_nots = RuleBool::BoolOr(
            Box::new(RuleBool::Not(Box::new(p.clone()))),
            Box::new(RuleBool::Not(Box::new(q.clone()))),
        );
        proven_everywhere(
            &and.ite(&one, &zero),
            &nor_of_nots.ite(&zero, &one),
            "p and q == not(not p or not q)",
        );
    }

    fn bx(t: &RuleTerm) -> Box<RuleTerm> {
        Box::new(t.clone())
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
