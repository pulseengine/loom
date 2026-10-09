//! Tier-2 slice 1 (issue #313): routing the **translation validator's** pure
//! bit-vector obligations through the swappable solver seam established for
//! the algebraic rule verifier in Tier-1 (issue #277).
//!
//! # What the obligation is
//!
//! `verify.rs` encodes the original and the optimized function to a single
//! Z3 bitvector each and asks whether they can differ:
//!
//! ```text
//! assert (not (= orig opt));  check()   // Unsat ⟹ equivalent
//! ```
//!
//! That is *exactly* the shape [`RuleSolver::prove_rule_equiv`] already has,
//! so this module reuses Tier-1's [`RuleTerm`]/[`RuleBool`] DSL and
//! [`RuleVerdict`] verbatim rather than inventing a parallel one. What it adds
//! is the part Tier-1 did not need: a way to get a **already-built Z3 AST**
//! into the neutral DSL.
//!
//! # Why reflection rather than a second encoder
//!
//! The encoder in `verify.rs` (`encode_function_to_smt_impl_inner`) is ~3000
//! lines and builds `z3::ast::BV` directly. Writing a second, neutral encoder
//! beside it would double the surface on which a translation-validator bug —
//! i.e. a silent miscompile — could hide, and the two would drift. Instead
//! this module *reflects* the finished Z3 AST back into [`RuleTerm`]:
//!
//!   1. Walk the Z3 AST. Every node must be one of the pure-BV operations in
//!      the closed fragment. **Anything else aborts the whole reflection** —
//!      a memory `select`/`store`, an uninterpreted-function application
//!      (`pure_call` congruence), a trapping `bvudiv`/`bvurem`/`bvsdiv`/
//!      `bvsrem`, a float, a bool constant, an n-ary node. There is no partial
//!      or approximate reflection.
//!   2. **Re-lower the reflected term back to Z3 with the very lowering the
//!      Z3 backend uses, and require the result to be the IDENTICAL AST node**
//!      (`Z3_is_eq_ast`, which on Z3's hash-consed AST is exact structural
//!      identity). If it is not identical, the reflection is not trusted and
//!      the obligation stays on the incumbent.
//!
//! Step 2 is the load-bearing soundness gate. It means a mis-reflection cannot
//! silently turn a hard obligation into an easy one: the neutral term either
//! denotes precisely the query Z3 was going to be asked, or it is discarded.
//!
//! # What is deliberately NOT here (slice 1 scope)
//!
//! Slice 1 does **not** change the correctness relation. For pure BV there are
//! no traps and no nondeterminism, so trap-equivalence + value-refinement
//! degenerates to the equality the validator already proves; this is a backend
//! swap that validates the harness. Partial operations (div/rem/trunc, loads,
//! stores), uninterpreted `pure_call` congruence, havoc, memory arrays and
//! floats are all **refused** by [`reflect_bv`] and stay on the incumbent —
//! they need the relation change that lands in slice 2.
//!
//! # Backend selection
//!
//! The **existing** `LOOM_VERIFY_BACKEND` variable selects the engine, with the
//! default (`z3`) unchanged: no obligation is routed at all, and `verify.rs`
//! takes byte-identical code paths to before this module existed.
//!
//!   * `z3`     — default. Nothing is routed; the incumbent decides everything.
//!   * `ordeal` — reflectable obligations are decided by ordeal, whose `Unsat`
//!                carries an LRAT certificate that is re-checked before the
//!                verdict is believed.
//!   * `both`   — both engines run on every routed obligation and the verdicts
//!                must AGREE; a disagreement panics. A green suite under
//!                `LOOM_VERIFY_BACKEND=both` IS the no-divergence assertion.

#[cfg(feature = "verification")]
use crate::rule_solver::{
    RuleBool, RuleSolver, RuleTerm, RuleVerdict, VerifyBackend, Z3RuleSolver,
};

#[cfg(feature = "verification")]
use std::cell::Cell;

// ============================================================================
// Budgets
// ============================================================================

/// Maximum number of nodes in a reflected [`RuleTerm`].
///
/// Z3's AST is a hash-consed **DAG**; [`RuleTerm`] is a **tree**. Reflecting a
/// heavily-shared DAG therefore expands it, in the worst case exponentially.
/// The budget is checked *during* the walk, so a blow-up costs a bounded amount
/// of work and then defers to the incumbent. It also bounds bit-blasting time,
/// which ordeal's wall-clock deadline explicitly does not cover (the deadline
/// governs the SAT search).
#[cfg(feature = "verification")]
const MAX_TERM_NODES: usize = 4096;

/// Default per-obligation wall-clock budget for the ordeal solve, in ms.
///
/// Mirrors the incumbent's own `LOOM_Z3_TIMEOUT_MS` default (5000 ms) so a slow
/// solve degrades to a fast, safe revert on either engine rather than a hang.
#[cfg(feature = "verification")]
const DEFAULT_ORDEAL_TIMEOUT_MS: u64 = 5000;

/// Read the ordeal per-obligation deadline from `LOOM_ORDEAL_TIMEOUT_MS`
/// (default [`DEFAULT_ORDEAL_TIMEOUT_MS`]).
#[cfg(feature = "verification")]
fn ordeal_timeout_ms() -> u64 {
    std::env::var("LOOM_ORDEAL_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_ORDEAL_TIMEOUT_MS)
}

// ============================================================================
// Outcome of offering an obligation to the seam
// ============================================================================

/// Why an obligation was NOT routed through the neutral seam.
///
/// Every variant is a *conservative* outcome: the obligation goes back to the
/// incumbent solver, which decides it exactly as it did before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeferReason {
    /// The active backend is the incumbent (`LOOM_VERIFY_BACKEND` unset or
    /// `z3`). Nothing is attempted — the default path is unchanged.
    IncumbentBackend,
    /// The two sides have different result widths. This is the validator's own
    /// soundness bail (loom#145) and is left entirely to the call site; the
    /// seam never decides a width-mismatched equality.
    WidthMismatch,
    /// A node outside the closed pure-BV fragment was reached. The `&'static
    /// str` names the class, for diagnostics.
    OutOfFragment(&'static str),
    /// The reflected term exceeded [`MAX_TERM_NODES`].
    TooLarge,
    /// The reflected term did not re-lower to the identical Z3 AST, so the
    /// reflection could not be self-validated and is not trusted.
    RoundTripFailed,
    /// A division appeared inside an `Ite` arm, so its trap condition is
    /// path-dependent and this seam will not approximate it. See
    /// [`trap_condition`].
    TrapUnderConditional,
    /// One variable name occurred at two different widths, or two distinct Z3
    /// constants shared a name.
    ///
    /// ordeal interns bitvector variables by **name alone**
    /// (`Solver::var_word`), so such a term could conflate two distinct
    /// variables. Z3 keys constants by `(symbol, sort)` and would not. Rather
    /// than rely on this never happening, the seam refuses the obligation.
    VariableNameCollision,
}

impl DeferReason {
    /// A short human-readable label (for diagnostics and test messages).
    pub fn label(&self) -> &'static str {
        match self {
            DeferReason::IncumbentBackend => "incumbent backend selected",
            DeferReason::WidthMismatch => "result width mismatch",
            DeferReason::OutOfFragment(what) => what,
            DeferReason::TrapUnderConditional => "division under a conditional",
            DeferReason::TooLarge => "term exceeds node budget",
            DeferReason::RoundTripFailed => "reflection round-trip not identical",
            DeferReason::VariableNameCollision => "variable name collision",
        }
    }
}

/// What the seam did with an obligation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeamOutcome {
    /// The seam decided it. The caller must map this verdict onto whatever it
    /// previously did for the corresponding `SatResult`.
    Decided(RuleVerdict),
    /// The seam declined; the caller runs the incumbent path unchanged.
    Deferred(DeferReason),
}

// ============================================================================
// Route accounting — reachability is asserted, not assumed
// ============================================================================
//
// A seam that silently stops matching would leave every obligation on the
// incumbent and every test still green. These counters make "a real obligation
// flowed through ordeal" a property a test can ASSERT. They are thread-local so
// a test observes only its own thread's obligations, which is what `cargo test`
// (one thread per test) needs for a deterministic assertion.

#[cfg(feature = "verification")]
thread_local! {
    static ROUTED: Cell<u64> = const { Cell::new(0) };
    static DEFERRED: Cell<u64> = const { Cell::new(0) };
}

/// Reset this thread's route counters.
#[cfg(feature = "verification")]
pub fn reset_route_counts() {
    ROUTED.with(|c| c.set(0));
    DEFERRED.with(|c| c.set(0));
}

/// `(routed, deferred)` obligation counts for this thread since the last
/// [`reset_route_counts`].
#[cfg(feature = "verification")]
pub fn route_counts() -> (u64, u64) {
    (ROUTED.with(|c| c.get()), DEFERRED.with(|c| c.get()))
}

#[cfg(feature = "verification")]
fn note_routed() {
    ROUTED.with(|c| c.set(c.get() + 1));
}

#[cfg(feature = "verification")]
fn note_deferred(reason: DeferReason) {
    DEFERRED.with(|c| c.set(c.get() + 1));
    // Same switch the validator already uses for per-function revert detail
    // (loom#145). During the migration an operator needs to be able to see
    // WHY coverage is what it is, not just that it is partial.
    if std::env::var_os("LOOM_VERBOSE_REVERTS").is_some() {
        eprintln!(
            "verify_solver: obligation left on the incumbent ({})",
            reason.label()
        );
    }
}

// ============================================================================
// The seam entry point
// ============================================================================

/// Offer the equivalence obligation `orig == opt` to the neutral seam.
///
/// Returns [`SeamOutcome::Decided`] only when the obligation was reflected into
/// the closed pure-BV fragment, the reflection re-lowered to the identical Z3
/// AST, and the selected backend reached a verdict. In every other case the
/// caller must fall through to the incumbent path unchanged.
///
/// `backend` is an explicit parameter rather than an env read so callers *and
/// tests* can drive a specific engine deterministically; `verify.rs` passes
/// [`VerifyBackend::from_env`].
#[cfg(feature = "verification")]
pub fn decide_bv_equivalence(
    orig: &z3::ast::BV,
    opt: &z3::ast::BV,
    backend: VerifyBackend,
) -> SeamOutcome {
    let outcome = decide_inner(orig, opt, backend);
    match &outcome {
        SeamOutcome::Decided(_) => note_routed(),
        SeamOutcome::Deferred(reason) => note_deferred(*reason),
    }
    outcome
}

#[cfg(feature = "verification")]
fn decide_inner(orig: &z3::ast::BV, opt: &z3::ast::BV, backend: VerifyBackend) -> SeamOutcome {
    // Default backend: nothing is routed. The validator's behaviour is
    // byte-identical to before this module existed.
    if backend == VerifyBackend::Z3 {
        return SeamOutcome::Deferred(DeferReason::IncumbentBackend);
    }

    // The width bail is the call site's own soundness boundary (loom#145).
    // Re-checked here so the seam can never be handed an ill-sorted equality.
    if orig.get_size() != opt.get_size() {
        return SeamOutcome::Deferred(DeferReason::WidthMismatch);
    }

    // One reflector for BOTH sides: the variable-name/width table has to be
    // shared, or a name used at two widths across the two sides would slip
    // through.
    let mut r = reflect::Reflector::new(MAX_TERM_NODES);
    let lhs = match r.reflect_bv(orig) {
        Ok(t) => t,
        Err(reason) => return SeamOutcome::Deferred(reason),
    };
    let rhs = match r.reflect_bv(opt) {
        Ok(t) => t,
        Err(reason) => return SeamOutcome::Deferred(reason),
    };

    // The self-validation gate: the neutral terms must re-lower to the
    // IDENTICAL Z3 AST nodes we were handed. Anything else and we do not
    // believe the reflection.
    if !reflect::round_trips(&lhs, orig) || !reflect::round_trips(&rhs, opt) {
        return SeamOutcome::Deferred(DeferReason::RoundTripFailed);
    }

    // #313 slice 2 / #300: the obligation is the TRAP-EQUIVALENCE +
    // VALUE-REFINEMENT relation, not a bare value equality.
    //
    // Derived from the reflected terms rather than the Z3 ASTs on purpose:
    // these are the terms the round trip above validated, so the trap
    // condition is read off the same representation the proof is discharged
    // on. Reading it from the Z3 side instead would reintroduce a second,
    // unvalidated path into the obligation.
    let lhs_trap = match trap_condition(&lhs) {
        Ok(t) => t,
        Err(reason) => return SeamOutcome::Deferred(reason),
    };
    let rhs_trap = match trap_condition(&rhs) {
        Ok(t) => t,
        Err(reason) => return SeamOutcome::Deferred(reason),
    };

    // When NEITHER side can trap this is byte-identical to slice 1's
    // obligation modulo a constant 0 flag on both sides, so the pure-BV path
    // is unchanged in substance.
    let goal_lhs = trap_guarded(&lhs, lhs_trap.as_ref());
    let goal_rhs = trap_guarded(&rhs, rhs_trap.as_ref());

    SeamOutcome::Decided(prove(&goal_lhs, &goal_rhs, backend))
}

/// The wasm trap condition of every division in a reflected term, or a
/// refusal.
///
/// wasm's `div`/`rem` are PARTIAL where the bitvector operators are total:
///
/// | op | traps when |
/// |----|------------|
/// | `div_u`, `rem_u` | `b == 0` |
/// | `rem_s` | `b == 0` |
/// | `div_s` | `b == 0` **or** `a == INT_MIN && b == -1` |
///
/// The `div_s` / `rem_s` asymmetry is the one detail most likely to be
/// flattened by someone tidying this function: wasm defines
/// `rem_s(INT_MIN, -1) = 0`, no trap. ordeal itself got this wrong before
/// 0.10.0 (ordeal#72/#84) and the resulting over-approximation would have
/// falsely REJECTED a sound fold. There are tests that fail if the two are
/// made uniform.
///
/// # Why a division under a conditional is REFUSED
///
/// Returns [`DeferReason::TrapUnderConditional`] if a division appears inside
/// an `Ite` arm. A division in an untaken wasm `if` arm does not execute and
/// therefore does not trap, so the honest trap condition is
/// `branch_taken && divisor_is_zero` — not the flat disjunction this function
/// would otherwise build. Using the flat version would over-approximate: it
/// claims a trap on inputs where none occurs, which is not unsound for the
/// equivalence check but silently rejects correct transforms, and makes the
/// seam's claim ("the trap conditions match") untrue as stated.
///
/// Refusing is the same scope discipline slice 1 applied to memory: the bound
/// is narrow, exact, and named, rather than wide and approximate.
#[cfg(feature = "verification")]
fn trap_condition(term: &RuleTerm) -> Result<Option<RuleBool>, DeferReason> {
    fn walk(t: &RuleTerm, under_ite: bool, acc: &mut Vec<RuleBool>) -> Result<(), DeferReason> {
        use RuleTerm as T;
        // The division arms first: these are the nodes that can trap.
        let divisor_and_kind = match t {
            T::Udiv(a, b) | T::Urem(a, b) | T::Srem(a, b) => Some((a, b, false)),
            T::Sdiv(a, b) => Some((a, b, true)),
            _ => None,
        };
        if let Some((a, b, is_signed_div)) = divisor_and_kind {
            if under_ite {
                return Err(DeferReason::TrapUnderConditional);
            }
            let width = b.width();
            let zero = T::Const { value: 0, width };
            // Every kind traps on a zero divisor.
            let mut cond = RuleBool::Eq(Box::new((**b).clone()), Box::new(zero.clone()));
            if is_signed_div {
                // `div_s` ALSO traps on INT_MIN / -1 (signed overflow).
                // `rem_s` deliberately does NOT — see the doc comment.
                let int_min = T::Const {
                    value: 1u128 << (width - 1),
                    width,
                };
                let minus_one = T::Const {
                    value: (!0u128) >> (128 - width),
                    width,
                };
                let overflow = RuleBool::BoolAnd(
                    Box::new(RuleBool::Eq(Box::new((**a).clone()), Box::new(int_min))),
                    Box::new(RuleBool::Eq(Box::new((**b).clone()), Box::new(minus_one))),
                );
                cond = RuleBool::BoolOr(Box::new(cond), Box::new(overflow));
            }
            acc.push(cond);
        }

        // Recurse. `Ite` arms set `under_ite` for everything beneath them;
        // the CONDITION is not an arm and is evaluated unconditionally, so it
        // keeps the current flag.
        match t {
            T::Const { .. } | T::Var { .. } => {}
            T::Add(a, b)
            | T::Sub(a, b)
            | T::Mul(a, b)
            | T::Udiv(a, b)
            | T::Urem(a, b)
            | T::Sdiv(a, b)
            | T::Srem(a, b)
            | T::And(a, b)
            | T::Or(a, b)
            | T::Xor(a, b)
            | T::Shl(a, b)
            | T::Lshr(a, b)
            | T::Ashr(a, b)
            | T::Rotl(a, b)
            | T::Rotr(a, b)
            | T::Concat(a, b) => {
                walk(a, under_ite, acc)?;
                walk(b, under_ite, acc)?;
            }
            T::Neg(a) | T::Not(a) => walk(a, under_ite, acc)?,
            T::ZeroExt { arg, .. } | T::SignExt { arg, .. } | T::Extract { arg, .. } => {
                walk(arg, under_ite, acc)?
            }
            T::Ite { cond, then_, else_ } => {
                walk_bool(cond, under_ite, acc)?;
                walk(then_, true, acc)?;
                walk(else_, true, acc)?;
            }
        }
        Ok(())
    }

    fn walk_bool(
        b: &RuleBool,
        under_ite: bool,
        acc: &mut Vec<RuleBool>,
    ) -> Result<(), DeferReason> {
        use RuleBool as B;
        match b {
            B::Eq(x, y)
            | B::Ult(x, y)
            | B::Ule(x, y)
            | B::Ugt(x, y)
            | B::Uge(x, y)
            | B::Slt(x, y)
            | B::Sle(x, y)
            | B::Sgt(x, y)
            | B::Sge(x, y) => {
                walk(x, under_ite, acc)?;
                walk(y, under_ite, acc)
            }
            B::Not(inner) => walk_bool(inner, under_ite, acc),
            B::BoolAnd(x, y) | B::BoolOr(x, y) => {
                walk_bool(x, under_ite, acc)?;
                walk_bool(y, under_ite, acc)
            }
        }
    }

    let mut acc = Vec::new();
    walk(term, false, &mut acc)?;
    Ok(acc
        .into_iter()
        .reduce(|l, r| RuleBool::BoolOr(Box::new(l), Box::new(r))))
}

/// Fold a side's trap condition and value into ONE bitvector term, so the
/// slice-2 relation becomes a plain equality the existing solver API can
/// discharge.
///
/// ```text
///   [ 1-bit trap flag ] ++ [ value, forced to 0 when trapping ]
/// ```
///
/// Comparing two of these for equality is EXACTLY the relation #300 asks for:
///
/// * trap conditions differ  -> flags differ -> terms differ -> REJECTED,
///   which catches trap REMOVAL and trap ADDITION symmetrically. A bare value
///   equality sees neither.
/// * both trap               -> both sides are `1 ++ 0` -> equal. Correct:
///   wasm traps are observable but a trapped computation has no value, so the
///   values are genuinely don't-care there. Without this normalisation the
///   total `bvsdiv`'s arbitrary value at `b == 0` would leak into the
///   obligation and reject sound transforms.
/// * neither traps           -> `0 ++ v` vs `0 ++ v'` -> equal iff the values
///   are equal, which is the relation the validator already proved.
///
/// Values are a singleton set here, so refinement degenerates to equality.
/// That is correct for integer div/rem — the places wasm is genuinely
/// nondeterministic (NaN payloads, `memory.grow`) are floats and memory,
/// slices 5 and 6.
///
/// Expressing the relation inside the existing `prove_rule_equiv` rather than
/// adding a trait method is deliberate: every construct used here
/// (`Ite`, `Concat`, `Const`, the comparisons) is already covered by the
/// reflect-and-re-lower round trip, so the encoding inherits that validation
/// instead of needing its own.
#[cfg(feature = "verification")]
fn trap_guarded(term: &RuleTerm, trap: Option<&RuleBool>) -> RuleTerm {
    let Some(trap) = trap else {
        // Nothing in this term can trap; the relation reduces to the value
        // equality the seam already discharged. Prefixing a constant 0 flag
        // keeps both sides the same shape.
        return RuleTerm::Concat(
            Box::new(RuleTerm::Const { value: 0, width: 1 }),
            Box::new(term.clone()),
        );
    };
    let width = term.width();
    let flag = RuleTerm::Ite {
        cond: Box::new(trap.clone()),
        then_: Box::new(RuleTerm::Const { value: 1, width: 1 }),
        else_: Box::new(RuleTerm::Const { value: 0, width: 1 }),
    };
    let value = RuleTerm::Ite {
        cond: Box::new(trap.clone()),
        then_: Box::new(RuleTerm::Const { value: 0, width }),
        else_: Box::new(term.clone()),
    };
    RuleTerm::Concat(Box::new(flag), Box::new(value))
}

/// Discharge a reflected obligation on the selected backend.
#[cfg(feature = "verification")]
fn prove(lhs: &RuleTerm, rhs: &RuleTerm, backend: VerifyBackend) -> RuleVerdict {
    match backend {
        // Unreachable in practice (`decide_inner` returns early), but keeping
        // the arm total means adding a backend cannot silently fall through.
        VerifyBackend::Z3 => Z3RuleSolver.prove_rule_equiv(lhs, rhs),
        VerifyBackend::Ordeal => BoundedOrdealSolver::from_env().prove_rule_equiv(lhs, rhs),
        VerifyBackend::Both => {
            let z3 = Z3RuleSolver.prove_rule_equiv(lhs, rhs);
            let ordeal = BoundedOrdealSolver::from_env().prove_rule_equiv(lhs, rhs);
            assert!(
                z3.agrees_with(&ordeal),
                "LOOM_VERIFY_BACKEND=both: translation-validation solver disagreement!\n  \
                 z3     = {:?}\n  ordeal = {:?}\n  lhs    = {:?}\n  rhs    = {:?}",
                z3,
                ordeal,
                lhs,
                rhs
            );
            // Verdicts agree; return the Z3 one so the counterexample text
            // stays in the historical format the pipeline already logs.
            z3
        }
    }
}

// ============================================================================
// ordeal backend, wall-clock bounded
// ============================================================================

/// ordeal, driven under a per-obligation wall-clock deadline.
///
/// Tier-1's [`crate::rule_solver::OrdealRuleSolver`] uses the unbounded
/// `Solver::prove_equiv`, which is right for a handful of tiny algebraic rules.
/// A whole-function obligation is a different size class, and an unbounded
/// solve would turn a slow query into a hang rather than the conservative
/// revert the validator is built around — so this variant asserts the same
/// `Ne(a, b)` goal and drives `check_with_deadline`.
///
/// The soundness gate is unchanged and is ordeal's own: on engine-UNSAT the
/// LRAT certificate is validated by the checker before `Unsat` is returned. We
/// then `recheck()` it a second time here, exactly as Tier-1 does — an `Unsat`
/// whose certificate we cannot re-validate is downgraded to `Unknown`, never
/// believed.
#[cfg(feature = "verification")]
pub struct BoundedOrdealSolver {
    /// Wall-clock budget for the SAT search, in milliseconds.
    pub timeout_ms: u64,
}

#[cfg(feature = "verification")]
impl BoundedOrdealSolver {
    /// Build one with the deadline from `LOOM_ORDEAL_TIMEOUT_MS`.
    pub fn from_env() -> Self {
        BoundedOrdealSolver {
            timeout_ms: ordeal_timeout_ms(),
        }
    }
}

#[cfg(feature = "verification")]
impl RuleSolver for BoundedOrdealSolver {
    fn prove_rule_equiv(&self, lhs: &RuleTerm, rhs: &RuleTerm) -> RuleVerdict {
        use crate::rule_solver::ordeal_backend::{lower_to_ordeal, render_ordeal_model};
        use ordeal::{BoolTerm, CheckResult, Solver};

        let mut solver = Solver::new();
        solver.assert(BoolTerm::Ne(
            Box::new(lower_to_ordeal(lhs)),
            Box::new(lower_to_ordeal(rhs)),
        ));
        match solver.check_with_deadline(self.timeout_ms) {
            CheckResult::Unsat(cert) => match cert.recheck() {
                Ok(()) => RuleVerdict::Proven,
                Err(_) => RuleVerdict::Unknown,
            },
            CheckResult::Sat(model) => RuleVerdict::Disproven(render_ordeal_model(&model)),
            CheckResult::Unknown => RuleVerdict::Unknown,
        }
    }

    fn backend_name(&self) -> &'static str {
        "ordeal(bounded)"
    }
}

// ============================================================================
// Z3 AST → neutral term reflection
// ============================================================================

#[cfg(feature = "verification")]
mod reflect {
    use super::DeferReason;
    use crate::rule_solver::{RuleBool, RuleTerm};
    use std::collections::HashMap;
    use z3::ast::{Ast, BV, Bool, Dynamic};
    use z3::{AstKind, DeclKind};

    type Reflected<T> = Result<T, DeferReason>;

    /// Re-lower `term` and require the identical Z3 AST node.
    ///
    /// `Z3_is_eq_ast` (which is what `PartialEq` on a `BV` calls) is pointer
    /// equality on Z3's hash-consed AST — so `true` here means the neutral term
    /// denotes *precisely* the formula we were handed, not merely an equivalent
    /// one. That is what makes reflection safe to trust without asking a solver
    /// to check it.
    pub(super) fn round_trips(term: &RuleTerm, original: &BV) -> bool {
        crate::rule_solver::z3_backend::lower_to_z3(term) == *original
    }

    /// Walks a Z3 AST and rebuilds it in the closed pure-BV fragment.
    pub(super) struct Reflector {
        /// name → (width, the Z3 constant it came from).
        vars: HashMap<String, (u32, BV)>,
        budget: usize,
    }

    impl Reflector {
        pub(super) fn new(max_nodes: usize) -> Self {
            Reflector {
                vars: HashMap::new(),
                budget: max_nodes,
            }
        }

        fn spend(&mut self) -> Reflected<()> {
            if self.budget == 0 {
                return Err(DeferReason::TooLarge);
            }
            self.budget -= 1;
            Ok(())
        }

        /// Record a free variable, refusing any name reuse that ordeal's
        /// name-keyed interning could conflate.
        fn intern(&mut self, name: String, width: u32, ast: &BV) -> Reflected<RuleTerm> {
            match self.vars.get(&name) {
                Some((w, prev)) => {
                    // Two distinct Z3 constants must never share a neutral
                    // name, and one name must never carry two widths.
                    if *w != width || prev != ast {
                        return Err(DeferReason::VariableNameCollision);
                    }
                }
                None => {
                    self.vars.insert(name.clone(), (width, ast.clone()));
                }
            }
            Ok(RuleTerm::Var { name, width })
        }

        pub(super) fn reflect_bv(&mut self, node: &BV) -> Reflected<RuleTerm> {
            self.spend()?;
            let width = node.get_size();

            // Constants.
            if node.kind() == AstKind::Numeral {
                // The neutral `Const` carries a u128 but every backend lowering
                // reads it back through a 64-bit path, so refuse anything
                // wider rather than silently truncate.
                if width > 64 {
                    return Err(DeferReason::OutOfFragment("numeral wider than 64 bits"));
                }
                let value = node
                    .as_u64()
                    .ok_or(DeferReason::OutOfFragment("numeral not readable as u64"))?;
                return Ok(RuleTerm::Const {
                    value: value as u128,
                    width,
                });
            }

            if node.kind() != AstKind::App {
                return Err(DeferReason::OutOfFragment("non-application AST node"));
            }
            let decl = node
                .safe_decl()
                .map_err(|_| DeferReason::OutOfFragment("AST node is not an application"))?;
            let kind = decl.kind();
            let n = node.num_children();

            // Free variables are 0-ary uninterpreted constants. An
            // uninterpreted decl WITH arguments is a `pure_call`-style
            // uninterpreted function — congruence reasoning, explicitly slice 2.
            if kind == DeclKind::UNINTERPRETED {
                if n == 0 {
                    return self.intern(decl.name(), width, node);
                }
                return Err(DeferReason::OutOfFragment(
                    "uninterpreted function application",
                ));
            }

            match kind {
                // --- still refused ---
                //
                // `bvsmod` is floored modulo. Wasm has no such operator, so a
                // `bsmod` node in a loom-built term would mean the encoder
                // produced something the spec does not define; refusing is
                // the honest answer to a term we cannot attribute to a wasm
                // instruction.
                //
                // The `*0` and `*_I` kinds are Z3's INTERNAL representations
                // of the partial operators — the uninterpreted
                // division-by-zero function and the interpreted form. They
                // are not what loom builds, and their semantics at the
                // undefined point are Z3's own business rather than
                // something the wasm spec pins, so the seam does not reflect
                // them.
                DeclKind::BSMOD
                | DeclKind::BUDIV0
                | DeclKind::BUREM0
                | DeclKind::BSDIV0
                | DeclKind::BSREM0
                | DeclKind::BSMOD0
                | DeclKind::BUDIV_I
                | DeclKind::BUREM_I
                | DeclKind::BSDIV_I
                | DeclKind::BSREM_I
                | DeclKind::BSMOD_I => Err(DeferReason::OutOfFragment(
                    "floored modulo, or a Z3-internal partial-op node",
                )),

                // --- slice 2: signed/unsigned division and remainder ---
                //
                // Reflected as PURE VALUE operators. `bvsdiv`/`bvsrem` and
                // `bvudiv`/`bvurem` are total bitvector operations; wasm's
                // `div_s`/`div_u`/`rem_s`/`rem_u` are partial, and the trap
                // clause is NOT carried here. It could not be: a trap
                // predicate has no node in the Z3 AST, so it would be the one
                // part of the reflection that reflect-and-re-lower cannot
                // validate, in a seam whose trustworthiness is exactly that
                // check. Traps are discharged by `trap_gate`'s `DefineOrTrap`
                // and enter the obligation as the `¬may_trap ⇒ value_eq`
                // guard.
                //
                // `bvsmod` is NOT included. Wasm has no floored-modulo
                // operator, so a `bsmod` node in a loom-built term would mean
                // the encoder produced something the wasm spec does not
                // define, and refusing is the honest response to a term we
                // cannot attribute to a wasm instruction.
                DeclKind::BUDIV => self.binop(node, n, RuleTerm::Udiv),
                DeclKind::BUREM => self.binop(node, n, RuleTerm::Urem),
                DeclKind::BSDIV => self.binop(node, n, RuleTerm::Sdiv),
                DeclKind::BSREM => self.binop(node, n, RuleTerm::Srem),

                // --- memory: Array theory, slice 2 ---
                DeclKind::SELECT | DeclKind::STORE => {
                    Err(DeferReason::OutOfFragment("memory array select/store"))
                }

                // --- the closed pure-BV fragment ---
                DeclKind::BADD => self.binop(node, n, RuleTerm::Add),
                DeclKind::BSUB => self.binop(node, n, RuleTerm::Sub),
                DeclKind::BMUL => self.binop(node, n, RuleTerm::Mul),
                DeclKind::BAND => self.binop(node, n, RuleTerm::And),
                DeclKind::BOR => self.binop(node, n, RuleTerm::Or),
                DeclKind::BXOR => self.binop(node, n, RuleTerm::Xor),
                DeclKind::BSHL => self.binop(node, n, RuleTerm::Shl),
                DeclKind::BLSHR => self.binop(node, n, RuleTerm::Lshr),
                DeclKind::BASHR => self.binop(node, n, RuleTerm::Ashr),
                DeclKind::EXT_ROTATE_RIGHT => self.binop(node, n, RuleTerm::Rotr),
                DeclKind::EXT_ROTATE_LEFT => {
                    // ordeal derives `bvrotl` as `rotr(a, 0 - b)`, which is
                    // exact only when the width is a power of two. Every WASM
                    // width is (32/64), but refuse anything else rather than
                    // depend on that.
                    if !width.is_power_of_two() {
                        return Err(DeferReason::OutOfFragment(
                            "rotate-left at a non-power-of-two width",
                        ));
                    }
                    self.binop(node, n, RuleTerm::Rotl)
                }
                DeclKind::CONCAT => self.binop(node, n, RuleTerm::Concat),
                DeclKind::BNOT => self.unop(node, n, RuleTerm::Not),
                DeclKind::BNEG => self.unop(node, n, RuleTerm::Neg),

                DeclKind::SIGN_EXT | DeclKind::ZERO_EXT => {
                    if n != 1 {
                        return Err(DeferReason::OutOfFragment("extend with != 1 argument"));
                    }
                    let child = child_bv(node, 0)?;
                    let cw = child.get_size();
                    if width < cw {
                        return Err(DeferReason::OutOfFragment("extend narrows"));
                    }
                    // The extension amount is a decl PARAMETER, which the safe
                    // binding does not expose — but it is fully determined by
                    // the two sorts, and the round-trip gate re-checks it.
                    let by = width - cw;
                    let arg = Box::new(self.reflect_bv(&child)?);
                    Ok(if kind == DeclKind::SIGN_EXT {
                        RuleTerm::SignExt { by, arg }
                    } else {
                        RuleTerm::ZeroExt { by, arg }
                    })
                }

                DeclKind::EXTRACT => {
                    if n != 1 {
                        return Err(DeferReason::OutOfFragment("extract with != 1 argument"));
                    }
                    let child = child_bv(node, 0)?;
                    let cw = child.get_size();
                    if width > cw || width == 0 {
                        return Err(DeferReason::OutOfFragment("extract wider than its operand"));
                    }
                    // `lo` is a decl parameter the safe binding does not expose.
                    // Recover it by construction: only one `lo` can rebuild the
                    // identical hash-consed node.
                    let lo = (0..=(cw - width))
                        .find(|lo| child.extract(lo + width - 1, *lo) == *node)
                        .ok_or(DeferReason::OutOfFragment("extract bounds not recoverable"))?;
                    Ok(RuleTerm::Extract {
                        hi: lo + width - 1,
                        lo,
                        arg: Box::new(self.reflect_bv(&child)?),
                    })
                }

                DeclKind::ITE => {
                    if n != 3 {
                        return Err(DeferReason::OutOfFragment("ite with != 3 arguments"));
                    }
                    let cond = node
                        .nth_child(0)
                        .and_then(|d| d.as_bool())
                        .ok_or(DeferReason::OutOfFragment("ite condition is not a Bool"))?;
                    let cond = Box::new(self.reflect_bool(&cond)?);
                    let then_ = Box::new(self.reflect_bv(&child_bv(node, 1)?)?);
                    let else_ = Box::new(self.reflect_bv(&child_bv(node, 2)?)?);
                    Ok(RuleTerm::Ite { cond, then_, else_ })
                }

                _ => Err(DeferReason::OutOfFragment(
                    "bitvector op outside the slice-1 fragment",
                )),
            }
        }

        pub(super) fn reflect_bool(&mut self, node: &Bool) -> Reflected<RuleBool> {
            self.spend()?;
            if node.kind() != AstKind::App {
                return Err(DeferReason::OutOfFragment("non-application boolean node"));
            }
            let decl = node
                .safe_decl()
                .map_err(|_| DeferReason::OutOfFragment("boolean node is not an application"))?;
            let n = node.num_children();

            // A boolean free variable or an uninterpreted predicate has no
            // place in the slice-1 fragment (the neutral DSL has no boolean
            // leaves), so refuse it rather than invent an encoding.
            if decl.kind() == DeclKind::UNINTERPRETED {
                return Err(DeferReason::OutOfFragment("uninterpreted boolean"));
            }

            match decl.kind() {
                DeclKind::EQ => self.cmp(node, n, RuleBool::Eq),
                DeclKind::ULT => self.cmp(node, n, RuleBool::Ult),
                DeclKind::ULEQ => self.cmp(node, n, RuleBool::Ule),
                DeclKind::UGT => self.cmp(node, n, RuleBool::Ugt),
                DeclKind::UGEQ => self.cmp(node, n, RuleBool::Uge),
                DeclKind::SLT => self.cmp(node, n, RuleBool::Slt),
                DeclKind::SLEQ => self.cmp(node, n, RuleBool::Sle),
                DeclKind::SGT => self.cmp(node, n, RuleBool::Sgt),
                DeclKind::SGEQ => self.cmp(node, n, RuleBool::Sge),
                DeclKind::NOT => {
                    if n != 1 {
                        return Err(DeferReason::OutOfFragment("not with != 1 argument"));
                    }
                    Ok(RuleBool::Not(Box::new(
                        self.reflect_bool(&child_bool(node, 0)?)?,
                    )))
                }
                DeclKind::AND | DeclKind::OR => {
                    // Z3's and/or are n-ary. The neutral DSL is binary and the
                    // round-trip gate rebuilds a 2-ary node, so anything else
                    // could not round-trip anyway — refuse it up front.
                    if n != 2 {
                        return Err(DeferReason::OutOfFragment("n-ary boolean connective"));
                    }
                    let a = Box::new(self.reflect_bool(&child_bool(node, 0)?)?);
                    let b = Box::new(self.reflect_bool(&child_bool(node, 1)?)?);
                    Ok(if decl.kind() == DeclKind::AND {
                        RuleBool::BoolAnd(a, b)
                    } else {
                        RuleBool::BoolOr(a, b)
                    })
                }
                // `distinct` is deliberately absent. The neutral DSL has no
                // disequality that lowers back to a Z3 `distinct` node, so a
                // `Ne` variant could never satisfy the round-trip gate — it
                // would be an unreachable arm. The encoder does not build
                // `distinct` either; it writes `(not (= a b))`, which the
                // `NOT`/`EQ` arms already cover.
                //
                // `true` / `false` have no neutral leaf (ordeal's BoolTerm has
                // no constant), and encoding them via a dummy equality would be
                // a lowering that is not a single well-defined operation in both
                // backends. Refused; the incumbent handles those obligations.
                DeclKind::TRUE | DeclKind::FALSE => Err(DeferReason::OutOfFragment(
                    "boolean constant (no neutral leaf)",
                )),
                _ => Err(DeferReason::OutOfFragment(
                    "boolean op outside the slice-1 fragment",
                )),
            }
        }

        fn binop(
            &mut self,
            node: &BV,
            n: usize,
            build: fn(Box<RuleTerm>, Box<RuleTerm>) -> RuleTerm,
        ) -> Reflected<RuleTerm> {
            if n != 2 {
                return Err(DeferReason::OutOfFragment("n-ary bitvector op"));
            }
            let a = Box::new(self.reflect_bv(&child_bv(node, 0)?)?);
            let b = Box::new(self.reflect_bv(&child_bv(node, 1)?)?);
            Ok(build(a, b))
        }

        fn unop(
            &mut self,
            node: &BV,
            n: usize,
            build: fn(Box<RuleTerm>) -> RuleTerm,
        ) -> Reflected<RuleTerm> {
            if n != 1 {
                return Err(DeferReason::OutOfFragment("unary op with != 1 argument"));
            }
            Ok(build(Box::new(self.reflect_bv(&child_bv(node, 0)?)?)))
        }

        fn cmp(
            &mut self,
            node: &Bool,
            n: usize,
            build: fn(Box<RuleTerm>, Box<RuleTerm>) -> RuleBool,
        ) -> Reflected<RuleBool> {
            if n != 2 {
                return Err(DeferReason::OutOfFragment("n-ary comparison"));
            }
            // Equality/distinct also exist at Bool and Array sorts; `as_bv`
            // returning None is exactly the refusal we want there.
            let a = bv_child(node, 0)?;
            let b = bv_child(node, 1)?;
            let a = Box::new(self.reflect_bv(&a)?);
            let b = Box::new(self.reflect_bv(&b)?);
            Ok(build(a, b))
        }
    }

    fn child_bv(node: &BV, idx: usize) -> Reflected<BV> {
        as_bv(node.nth_child(idx))
    }

    fn bv_child(node: &Bool, idx: usize) -> Reflected<BV> {
        as_bv(node.nth_child(idx))
    }

    fn as_bv(child: Option<Dynamic>) -> Reflected<BV> {
        child
            .and_then(|d| d.as_bv())
            .ok_or(DeferReason::OutOfFragment("operand is not a bitvector"))
    }

    fn child_bool(node: &Bool, idx: usize) -> Reflected<Bool> {
        node.nth_child(idx)
            .and_then(|d| d.as_bool())
            .ok_or(DeferReason::OutOfFragment("operand is not a boolean"))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "verification"))]
mod tests {
    use super::*;
    use z3::ast::{Array, BV, Bool};
    use z3::{Config, Sort, with_z3_config};

    fn cfg() -> Config {
        Config::new()
    }

    /// Reflect `bv` and require the round-trip to be the identical AST.
    fn reflect_ok(bv: &BV) -> RuleTerm {
        let mut r = reflect::Reflector::new(MAX_TERM_NODES);
        let t = r
            .reflect_bv(bv)
            .unwrap_or_else(|e| panic!("expected {} to reflect, got {:?}", bv, e));
        assert!(
            reflect::round_trips(&t, bv),
            "round-trip must be the identical AST for {}",
            bv
        );
        t
    }

    fn reflect_err(bv: &BV) -> DeferReason {
        let mut r = reflect::Reflector::new(MAX_TERM_NODES);
        match r.reflect_bv(bv) {
            Ok(t) => panic!("expected {} to be refused, but it reflected to {:?}", bv, t),
            Err(e) => e,
        }
    }

    /// Do the two engines agree about division at the points wasm never
    /// reaches?
    ///
    /// `bvsdiv`/`bvudiv` are TOTAL in SMT-LIB: they are defined at `b == 0`.
    /// Wasm traps there, so that point is a don't-care for loom — but the
    /// seam currently routes a bare value equality, which quantifies over it.
    /// If Z3 and ordeal happen to define the divide-by-zero result
    /// differently, two correct engines would return different verdicts on
    /// the same obligation, and `LOOM_VERIFY_BACKEND=both` would panic on a
    /// disagreement that means nothing about loom.
    ///
    /// ordeal reaches `bvsdiv` through a BLESSED DERIVED op
    /// (`lowering::bvsdiv`) rather than a primitive, so its behaviour at the
    /// undefined point is a property of that construction, not something the
    /// two engines share by assumption.
    ///
    /// MEASURED (ordeal 0.27.0, this tree): they agree on the VERDICT in
    /// every case probed.
    ///
    /// ```text
    /// sdiv by 0 vs all-ones: z3=Disproven(a -> #xffffffff)
    ///                    ordeal=Disproven(a -> 0x80000000)   agree
    /// udiv by 0 vs all-ones: z3=Proven   ordeal=Proven        agree
    /// control: sdiv == itself: z3=Proven ordeal=Proven        agree
    /// ```
    ///
    /// Both define `bvudiv(a, 0)` as all-ones, per SMT-LIB. The `sdiv` row
    /// repays a careful read: the verdicts match while the COUNTEREXAMPLES
    /// differ. That is not a divergence — a counterexample witnesses an
    /// existential, and two different witnesses are both valid.
    /// `agrees_with` compares verdicts, which is the right granularity.
    ///
    /// So the `both`-mode panic risk on routed division is lower than
    /// feared. It does NOT remove the need for the trap guard: a bare value
    /// equality still quantifies over `b == 0`, so a correct transform
    /// differing only there would be REJECTED — conservative, but a
    /// precision loss — and the slice-2 relation needs the guard regardless,
    /// since it must reject trap REMOVAL and trap ADDITION, neither of which
    /// a value equality can see. What this test buys is a regression guard on
    /// the agreement itself.
    #[test]
    fn the_engines_are_asked_whether_they_agree_on_division_by_zero() {
        with_z3_config(&cfg(), || {
            let a = RuleTerm::Var {
                name: "a".to_string(),
                width: 32,
            };
            let zero = RuleTerm::Const {
                value: 0,
                width: 32,
            };
            // Each pair is (name, lhs, rhs). The first two are the don't-care
            // points; the third is a control that must hold on any engine.
            let cases: Vec<(&str, RuleTerm, RuleTerm)> = vec![
                (
                    "sdiv by zero vs all-ones",
                    RuleTerm::Sdiv(Box::new(a.clone()), Box::new(zero.clone())),
                    RuleTerm::Const {
                        value: u32::MAX as u128,
                        width: 32,
                    },
                ),
                (
                    "udiv by zero vs all-ones",
                    RuleTerm::Udiv(Box::new(a.clone()), Box::new(zero.clone())),
                    RuleTerm::Const {
                        value: u32::MAX as u128,
                        width: 32,
                    },
                ),
                (
                    "control: sdiv is equal to itself",
                    RuleTerm::Sdiv(Box::new(a.clone()), Box::new(a.clone())),
                    RuleTerm::Sdiv(Box::new(a.clone()), Box::new(a.clone())),
                ),
            ];
            let mut disagreements = Vec::new();
            for (name, lhs, rhs) in &cases {
                let z3 = Z3RuleSolver.prove_rule_equiv(lhs, rhs);
                let ordeal = BoundedOrdealSolver::from_env().prove_rule_equiv(lhs, rhs);
                let agree = z3.agrees_with(&ordeal);
                println!("  {name}: z3={z3:?} ordeal={ordeal:?} agree={agree}");
                if !agree {
                    disagreements.push(*name);
                }
            }
            assert!(
                disagreements.is_empty(),
                "the engines disagree about division at a point wasm never \
                 reaches: {disagreements:?}. A bare value equality over a \
                 total division therefore cannot be routed under `both` \
                 without the trap guard, because the panic would report a \
                 difference that says nothing about loom."
            );
        });
    }

    fn v(name: &str, width: u32) -> RuleTerm {
        RuleTerm::Var {
            name: name.to_string(),
            width,
        }
    }
    fn k(value: u128, width: u32) -> RuleTerm {
        RuleTerm::Const { value, width }
    }
    /// The relation as the seam discharges it, for a hand-built pair.
    fn decide_terms(lhs: &RuleTerm, rhs: &RuleTerm) -> RuleVerdict {
        let lt = trap_condition(lhs).expect("no conditional division in these fixtures");
        let rt = trap_condition(rhs).expect("no conditional division in these fixtures");
        Z3RuleSolver.prove_rule_equiv(
            &trap_guarded(lhs, lt.as_ref()),
            &trap_guarded(rhs, rt.as_ref()),
        )
    }

    /// Trap REMOVAL must be rejected — and this is the case that proves the
    /// guard is doing work a value equality cannot.
    ///
    /// `div_u(a, 0)` traps on every input. SMT-LIB's TOTAL `bvudiv` defines
    /// `bvudiv(a, 0)` as all-ones. So replacing the division with the
    /// constant all-ones is, to a bare bitvector equality, a correct
    /// transform — the two terms are equal everywhere. To wasm it is a
    /// miscompile: one traps unconditionally and the other returns a value.
    ///
    /// The test asserts BOTH halves, because only the pair is evidence:
    /// the unguarded equality ACCEPTS it (so the guard is not redundant), and
    /// the guarded relation REJECTS it (so the guard works).
    #[test]
    fn trap_removal_is_rejected_where_a_value_equality_would_accept_it() {
        with_z3_config(&cfg(), || {
            let always_traps = RuleTerm::Udiv(Box::new(v("a", 32)), Box::new(k(0, 32)));
            let all_ones = k(u32::MAX as u128, 32);

            // What slice 1's obligation would have said.
            let unguarded = Z3RuleSolver.prove_rule_equiv(&always_traps, &all_ones);
            assert!(
                matches!(unguarded, RuleVerdict::Proven),
                "the premise of this test is that a BARE value equality accepts \
                 this transform (SMT-LIB bvudiv(a,0) == all-ones). If this ever \
                 stops holding, the test below no longer demonstrates that the \
                 trap guard adds anything. got {unguarded:?}"
            );

            // What the slice-2 relation says.
            let guarded = decide_terms(&always_traps, &all_ones);
            assert!(
                matches!(guarded, RuleVerdict::Disproven(_)),
                "removing a mandatory trap must be REJECTED; got {guarded:?}"
            );
        });
    }

    /// Trap ADDITION must be rejected too. One-directional checks are how
    /// half a bug class survives: a gate that only refuses trap removal
    /// happily accepts an optimizer that introduces a trap.
    #[test]
    fn trap_addition_is_rejected() {
        with_z3_config(&cfg(), || {
            let all_ones = k(u32::MAX as u128, 32);
            let always_traps = RuleTerm::Udiv(Box::new(v("a", 32)), Box::new(k(0, 32)));
            let guarded = decide_terms(&all_ones, &always_traps);
            assert!(
                matches!(guarded, RuleVerdict::Disproven(_)),
                "introducing a trap must be REJECTED; got {guarded:?}"
            );
        });
    }

    /// The div_s / rem_s asymmetry, asserted as the pair that makes it an
    /// asymmetry rather than two separate facts.
    ///
    /// wasm: `div_s(INT_MIN, -1)` TRAPS (signed overflow);
    /// `rem_s(INT_MIN, -1)` is `0` and does NOT trap. Flattening the two —
    /// which ordeal itself did before 0.10.0 (ordeal#72/#84) — makes the
    /// `rem_s` row fail here.
    #[test]
    fn div_s_traps_on_int_min_over_minus_one_and_rem_s_does_not() {
        with_z3_config(&cfg(), || {
            let int_min = k(1u128 << 31, 32);
            let minus_one = k(u32::MAX as u128, 32);

            // rem_s(INT_MIN, -1) does not trap and is 0, so folding it to 0
            // must be ACCEPTED.
            let rem = RuleTerm::Srem(Box::new(int_min.clone()), Box::new(minus_one.clone()));
            let rem_verdict = decide_terms(&rem, &k(0, 32));
            assert!(
                matches!(rem_verdict, RuleVerdict::Proven),
                "rem_s(INT_MIN, -1) = 0 with NO trap, so this fold is sound and \
                 must be accepted. A rejection here means the overflow disjunct \
                 leaked into rem_s (ordeal#84's bug). got {rem_verdict:?}"
            );

            // div_s(INT_MIN, -1) TRAPS, so folding it to any value must be
            // REJECTED. INT_MIN is the two's-complement wrap-around result a
            // naive folder would produce, which is why it is the value used.
            let div = RuleTerm::Sdiv(Box::new(int_min.clone()), Box::new(minus_one));
            let div_verdict = decide_terms(&div, &int_min);
            assert!(
                matches!(div_verdict, RuleVerdict::Disproven(_)),
                "div_s(INT_MIN, -1) traps, so folding it to a value is a trap \
                 removal and must be rejected; got {div_verdict:?}"
            );
        });
    }

    /// A division under a conditional is REFUSED rather than approximated.
    ///
    /// A division in an untaken wasm `if` arm does not execute and so does
    /// not trap; the honest condition is path-dependent. The flat
    /// disjunction this seam builds would over-approximate and silently
    /// reject correct transforms, so the obligation is declined instead —
    /// the same scope discipline slice 1 applied to memory.
    #[test]
    fn a_division_under_a_conditional_is_refused_not_approximated() {
        let cond = RuleBool::Eq(Box::new(v("c", 32)), Box::new(k(0, 32)));
        let conditional_div = RuleTerm::Ite {
            cond: Box::new(cond),
            then_: Box::new(RuleTerm::Udiv(Box::new(v("a", 32)), Box::new(v("b", 32)))),
            else_: Box::new(k(0, 32)),
        };
        assert!(
            matches!(
                trap_condition(&conditional_div),
                Err(DeferReason::TrapUnderConditional)
            ),
            "a division under an Ite arm must be refused, not approximated"
        );

        // The control: the same division NOT under a conditional is accepted
        // by the predicate, so the refusal above is about the conditional and
        // not about division generally.
        let plain = RuleTerm::Udiv(Box::new(v("a", 32)), Box::new(v("b", 32)));
        assert!(
            trap_condition(&plain)
                .expect("plain division is fine")
                .is_some(),
            "an unconditional division must produce a trap condition"
        );
    }

    #[test]
    fn every_routed_op_reflects_and_round_trips_identically() {
        // This is the reflection's faithfulness test: for each operation in
        // the closed fragment, the neutral term must re-lower to the SAME Z3
        // AST node. A wrong match arm (e.g. reflecting bvsub as bvadd) fails
        // here rather than silently proving a different query.
        with_z3_config(&cfg(), || {
            let a = BV::new_const("a", 32);
            let b = BV::new_const("b", 32);
            let wide = BV::new_const("w", 64);
            let ops: Vec<BV> = vec![
                a.bvadd(&b),
                a.bvsub(&b),
                a.bvmul(&b),
                a.bvand(&b),
                a.bvor(&b),
                a.bvxor(&b),
                a.bvshl(&b),
                a.bvlshr(&b),
                a.bvashr(&b),
                a.bvrotl(&b),
                a.bvrotr(&b),
                a.bvnot(),
                a.bvneg(),
                a.zero_ext(32),
                a.sign_ext(32),
                a.extract(15, 0),
                a.extract(31, 16),
                wide.extract(47, 16),
                a.concat(&b),
                BV::from_u64(0xdead_beef, 32),
                a.eq(&b).ite(&a, &b),
                // slice 2: the four partial operators, reflected as pure
                // value operators. These are the arms most likely to be got
                // wrong, because trap_gate builds signed division as
                // abs/udiv/negate — a term that is EQUIVALENT to `bvsdiv`
                // but is not the same AST. Reflecting into that shape would
                // fail here rather than silently prove a different query,
                // which is what this test is for.
                a.bvudiv(&b),
                a.bvurem(&b),
                a.bvsdiv(&b),
                a.bvsrem(&b),
                a.bvslt(&b).ite(&a, &b),
                a.bvult(&b).ite(&a, &b),
                a.bvule(&b).ite(&a, &b),
                a.bvugt(&b).ite(&a, &b),
                a.bvuge(&b).ite(&a, &b),
                a.bvsle(&b).ite(&a, &b),
                a.bvsgt(&b).ite(&a, &b),
                a.bvsge(&b).ite(&a, &b),
                a.eq(&b).not().ite(&a, &b),
                Bool::and(&[a.eq(&b), a.bvult(&b)]).ite(&a, &b),
                Bool::or(&[a.eq(&b), a.bvult(&b)]).ite(&a, &b),
            ];
            for op in &ops {
                let t = reflect_ok(op);
                assert_eq!(
                    t.width(),
                    op.get_size(),
                    "neutral width must match Z3's for {}",
                    op
                );
            }
        });
    }

    #[test]
    fn out_of_fragment_obligations_are_refused() {
        with_z3_config(&cfg(), || {
            let a = BV::new_const("a", 32);
            let b = BV::new_const("b", 32);

            // Slice 2 moved div/rem INTO the fragment, so the old
            // assertion here (that they are refused) is gone — deliberately,
            // and replaced by its dual rather than deleted. These must now
            // reflect, and `every_routed_op_reflects_and_round_trips_identically`
            // additionally pins that they round-trip to the same AST.
            for op in [a.bvudiv(&b), a.bvurem(&b), a.bvsdiv(&b), a.bvsrem(&b)] {
                let mut r = reflect::Reflector::new(MAX_TERM_NODES);
                assert!(
                    r.reflect_bv(&op).is_ok(),
                    "slice 2 routes this partial op; it must reflect: {}",
                    op
                );
            }

            // What is still OUT: floored modulo has no wasm operator, so a
            // `bvsmod` node would be a term loom cannot attribute to any
            // instruction. Refusing it is the honest answer, and keeping this
            // assertion is what stops "slice 2 opened the division arms" from
            // drifting into "the seam accepts anything division-shaped".
            let smod = a.bvsmod(&b);
            assert!(
                matches!(reflect_err(&smod), DeferReason::OutOfFragment(_)),
                "bvsmod has no wasm counterpart and must stay refused"
            );

            // Memory: Array select.
            let mem = Array::new_const("memory", &Sort::bitvector(32), &Sort::bitvector(8));
            let load = mem.select(&a).as_bv().unwrap();
            assert!(matches!(reflect_err(&load), DeferReason::OutOfFragment(_)));

            // Uninterpreted function application (the `pure_call` shape).
            let f = z3::FuncDecl::new("pure_call_f", &[&Sort::bitvector(32)], &Sort::bitvector(32));
            let app = f.apply(&[&a]).as_bv().unwrap();
            assert_eq!(
                reflect_err(&app),
                DeferReason::OutOfFragment("uninterpreted function application")
            );

            // Boolean constant condition — no neutral leaf.
            let t = Bool::from_bool(true).ite(&a, &b);
            assert!(matches!(reflect_err(&t), DeferReason::OutOfFragment(_)));
        });
    }

    #[test]
    fn node_budget_defers_instead_of_expanding_forever() {
        with_z3_config(&cfg(), || {
            // A shared DAG: doubling `t` 40 times is 40 Z3 nodes but 2^40
            // tree nodes. The budget must stop the walk.
            let mut t = BV::new_const("a", 32);
            for _ in 0..40 {
                t = t.bvadd(&t);
            }
            let mut r = reflect::Reflector::new(MAX_TERM_NODES);
            assert_eq!(r.reflect_bv(&t).unwrap_err(), DeferReason::TooLarge);
        });
    }

    #[test]
    fn variable_name_reuse_at_two_widths_is_refused() {
        // ordeal interns bitvector variables by NAME ALONE, so a term with
        // `x:32` and `x:64` could conflate them. Z3 keys on (symbol, sort) and
        // would not. The seam must refuse rather than depend on this never
        // arising.
        with_z3_config(&cfg(), || {
            let narrow = BV::new_const("x", 32);
            let wide = BV::new_const("x", 64);
            let term = narrow.zero_ext(32).bvadd(&wide);
            assert_eq!(reflect_err(&term), DeferReason::VariableNameCollision);
        });
    }

    #[test]
    fn variable_name_reuse_across_the_two_sides_is_refused() {
        // The intra-side test above puts both widths in ONE term. THIS is the
        // harder case: each side round-trips perfectly against its own
        // original, so the per-side round-trip gate cannot see the conflict —
        // only the reflector's shared name table can. `decide_bv_equivalence`
        // therefore uses ONE reflector for both sides; if that ever became two,
        // ordeal (which interns by name alone) could be handed a term where
        // `x` means two different variables while Z3 saw two distinct
        // constants. This test is what fails if that sharing is lost.
        with_z3_config(&cfg(), || {
            let narrow = BV::new_const("x", 32);
            let wide = BV::new_const("x", 64);
            let lhs = narrow.zero_ext(32);
            // Each side alone reflects and round-trip fine...
            assert!(reflect::round_trips(&reflect_ok(&lhs), &lhs));
            assert!(reflect::round_trips(&reflect_ok(&wide), &wide));
            // ...but the obligation that pairs them must be refused.
            for backend in [VerifyBackend::Ordeal, VerifyBackend::Both] {
                assert_eq!(
                    decide_bv_equivalence(&lhs, &wide, backend),
                    SeamOutcome::Deferred(DeferReason::VariableNameCollision),
                    "cross-side name collision must be refused under {:?}",
                    backend
                );
            }
        });
    }

    #[test]
    fn default_backend_routes_nothing() {
        // The default path must be untouched by this module.
        with_z3_config(&cfg(), || {
            let a = BV::new_const("a", 32);
            let lhs = a.bvadd(BV::from_u64(0, 32));
            assert_eq!(
                decide_bv_equivalence(&lhs, &a, VerifyBackend::Z3),
                SeamOutcome::Deferred(DeferReason::IncumbentBackend)
            );
        });
    }

    #[test]
    fn width_mismatch_is_never_decided_by_the_seam() {
        with_z3_config(&cfg(), || {
            let a = BV::new_const("a", 32);
            let w = BV::new_const("w", 64);
            for backend in [VerifyBackend::Ordeal, VerifyBackend::Both] {
                assert_eq!(
                    decide_bv_equivalence(&a, &w, backend),
                    SeamOutcome::Deferred(DeferReason::WidthMismatch)
                );
            }
        });
    }

    #[test]
    fn seam_proves_a_true_equivalence_on_both_engines() {
        with_z3_config(&cfg(), || {
            let a = BV::new_const("a", 32);
            // (a << 3) == a * 8, and ((a + 1) - 1) == a.
            let pairs = [
                (a.bvshl(BV::from_u64(3, 32)), a.bvmul(BV::from_u64(8, 32))),
                (
                    a.bvadd(BV::from_u64(1, 32)).bvsub(BV::from_u64(1, 32)),
                    a.clone(),
                ),
                (a.bvnot().bvnot(), a.clone()),
                (a.bvneg().bvneg(), a.clone()),
                (a.zero_ext(32).extract(31, 0), a.clone()),
            ];
            for (l, r) in &pairs {
                for backend in [VerifyBackend::Ordeal, VerifyBackend::Both] {
                    assert_eq!(
                        decide_bv_equivalence(l, r, backend),
                        SeamOutcome::Decided(RuleVerdict::Proven),
                        "{} == {} must be proven by {:?}",
                        l,
                        r,
                        backend
                    );
                }
            }
        });
    }

    #[test]
    fn seam_disproves_a_false_equivalence_on_both_engines() {
        with_z3_config(&cfg(), || {
            let a = BV::new_const("a", 32);
            let l = a.bvadd(BV::from_u64(1, 32));
            for backend in [VerifyBackend::Ordeal, VerifyBackend::Both] {
                match decide_bv_equivalence(&l, &a, backend) {
                    SeamOutcome::Decided(RuleVerdict::Disproven(_)) => {}
                    other => panic!(
                        "a + 1 == a must be disproven by {:?}, got {:?}",
                        backend, other
                    ),
                }
            }
        });
    }

    #[test]
    fn defer_reasons_render_a_diagnostic_label() {
        // `label()` is what the LOOM_VERBOSE_REVERTS diagnostic prints; pin
        // the text so the operator-facing reason cannot silently become
        // uninformative.
        assert_eq!(
            DeferReason::IncumbentBackend.label(),
            "incumbent backend selected"
        );
        assert_eq!(DeferReason::WidthMismatch.label(), "result width mismatch");
        assert_eq!(DeferReason::TooLarge.label(), "term exceeds node budget");
        assert_eq!(
            DeferReason::RoundTripFailed.label(),
            "reflection round-trip not identical"
        );
        assert_eq!(
            DeferReason::VariableNameCollision.label(),
            "variable name collision"
        );
        assert_eq!(
            DeferReason::OutOfFragment("memory array select/store").label(),
            "memory array select/store"
        );
    }

    #[test]
    fn route_counts_track_routed_and_deferred() {
        with_z3_config(&cfg(), || {
            reset_route_counts();
            let a = BV::new_const("a", 32);
            let b = BV::new_const("b", 32);
            // Routed: pure BV.
            let _ = decide_bv_equivalence(&a.bvadd(&b), &b.bvadd(&a), VerifyBackend::Ordeal);
            // Routed since slice 2: division is in the fragment now. This
            // used to be the DEFERRED case in this test, and swapping it to
            // the routed side is the point — a counter that only ever ticked
            // "deferred" for division would keep passing after slice 2 while
            // describing slice 1.
            let _ = decide_bv_equivalence(&a.bvsdiv(&b), &a.bvsdiv(&b), VerifyBackend::Ordeal);
            // Deferred: floored modulo has no wasm counterpart.
            let _ = decide_bv_equivalence(&a.bvsmod(&b), &a.bvsmod(&b), VerifyBackend::Ordeal);
            assert_eq!(
                route_counts(),
                (2, 1),
                "two routed (add, sdiv) and one deferred (smod)"
            );
        });
    }
}
