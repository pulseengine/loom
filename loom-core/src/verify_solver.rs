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
    RuleArray, RuleBool, RuleSolver, RuleTerm, RuleVerdict, VerifyBackend, Z3RuleSolver,
};

#[cfg(feature = "verification")]
use std::cell::{Cell, RefCell};
#[cfg(feature = "verification")]
use std::collections::BTreeMap;

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

/// Maximum number of memory accesses (`select`s plus `store`s, both sides
/// together) in a routed obligation.
///
/// [`MAX_TERM_NODES`] bounds the reflected TREE; it does not bound what the
/// array elimination expands that tree into. Resolving a read walks the whole
/// store chain, and ordeal then relates every pair of reads of one base array
/// with an aliasing constraint — so the lowered core grows as roughly
/// `reads x stores + reads^2 / 2`. All of that is built, and bit-blasted,
/// before `check_with_deadline` starts counting, which is precisely the shape
/// of blow-up loom#347 was: a cost in the ENCODING, not in the search, so a
/// timeout does not bound it.
///
/// The count is over the reflected TREE, which is the Z3 DAG already
/// expanded, so it is far larger than the number of wasm reads a function
/// performs: each of a load's four `select`s carries its own copy of the
/// store chain beneath it. One `i32` store followed by one `i32` load is
/// four selects over a four-store chain — about 20 accesses, not 8.
///
/// That expansion also makes the count a good PROXY for what it is meant to
/// bound: the tree access count is roughly `reads x stores`, which is the
/// size of the lowered core after read-over-write and array congruence.
///
/// MEASURED on this tree, `n` `i32` store/load pairs against the 5000 ms
/// deadline (`slice4_the_memory_budget_is_measured` prints it, release):
///
/// ```text
///   pairs  accesses  solve
///       1        20    0 ms
///       2        72    3 ms
///       4       272   15 ms
///       6       600   37 ms
///       8         -   reflection refuses first (MAX_TERM_NODES)
/// ```
///
/// The first default here was 64, which admitted exactly ONE pair while the
/// solver was finishing 600 accesses in 37 ms — discarding almost all real
/// coverage to bound a cost that was not there. 1024 sits above every shape
/// that reflects at all, so [`MAX_TERM_NODES`] is the binding constraint for
/// ordinary code, while this still caps the quadratic for a shape with an
/// unusual read/store ratio.
#[cfg(feature = "verification")]
const DEFAULT_MAX_MEMORY_ACCESSES: usize = 1024;

/// Read the memory-access bound from `LOOM_ORDEAL_MAX_MEM_ACCESSES`.
#[cfg(feature = "verification")]
fn max_memory_accesses() -> usize {
    std::env::var("LOOM_ORDEAL_MAX_MEM_ACCESSES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_MEMORY_ACCESSES)
}

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
    /// Two `pure_call` sites shared a function name but disagreed on its
    /// argument widths or result width. ordeal keys an uninterpreted
    /// function by NAME for congruence, so conflating two different
    /// functions under one name would let unequal results be proven equal.
    /// ordeal would refuse this itself (`SliverError::InconsistentCall`);
    /// loom refuses first, with a reason that names the cause.
    CallSignatureCollision,
    /// A `select` sits beneath an operator that ordeal's extended language
    /// cannot lift over extended children. The `&'static str` names the
    /// operator. ordeal 0.29.0 lifted `ite`, `not` and `neg` (ordeal#231,
    /// taken in #416), so what remains are the four core-only derived ops —
    /// `urem`, `sdiv`, `srem`, `rotl`. See
    /// `rule_solver::ordeal_backend::ExtGap`.
    MemoryUnderUnliftedOp(&'static str),
    /// The obligation touches more distinct memory bytes than
    /// [`max_memory_accesses`] allows. Eliminating the array is quadratic —
    /// every read walks the whole store chain, and every pair of reads of one
    /// base array contributes an aliasing constraint — and that work happens
    /// BEFORE the solver deadline can intervene, so it needs its own bound.
    TooManyMemoryAccesses,
    /// ordeal's sliver refused the lowered query outright. The `&'static
    /// str` names which refusal, because the three mean very different
    /// things: a bad array sort would be a loom encoding bug, an
    /// inconsistent call signature belongs to slice 3, and a non-concrete
    /// index is vestigial in ordeal 0.27 (symbolic indices were decided from
    /// 0.11.0 / TR-028) — seeing it would mean that changed.
    SliverRefused(&'static str),
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
            DeferReason::CallSignatureCollision => "pure_call signature collision",
            DeferReason::MemoryUnderUnliftedOp(what) => what,
            DeferReason::TooManyMemoryAccesses => "too many memory accesses",
            DeferReason::SliverRefused(what) => what,
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
    /// Deferrals broken down by reason.
    ///
    /// The totals say how much coverage there is; this says what is in the
    /// way, which is the number that decides what the next slice should be.
    /// Keyed by [`DeferReason::label`] so the breakdown reads the same in a
    /// test as in `LOOM_VERBOSE_REVERTS` output.
    static DEFERRED_BY_REASON: RefCell<BTreeMap<&'static str, u64>> =
        const { RefCell::new(BTreeMap::new()) };
}

/// Reset this thread's route counters.
#[cfg(feature = "verification")]
pub fn reset_route_counts() {
    ROUTED.with(|c| c.set(0));
    DEFERRED.with(|c| c.set(0));
    DEFERRED_BY_REASON.with(|m| m.borrow_mut().clear());
}

/// Deferrals since the last [`reset_route_counts`], by reason label,
/// most frequent first.
#[cfg(feature = "verification")]
pub fn deferrals_by_reason() -> Vec<(&'static str, u64)> {
    let mut v: Vec<_> =
        DEFERRED_BY_REASON.with(|m| m.borrow().iter().map(|(k, n)| (*k, *n)).collect());
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    v
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
    DEFERRED_BY_REASON.with(|m| *m.borrow_mut().entry(reason.label()).or_insert(0) += 1);
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

    // #313 slice 4: a memory obligation leaves the closed core, so ordeal
    // decides it through the array sliver. Whether the sliver can TAKE it is
    // settled here, before anything is routed.
    //
    // That ordering is the point. `prove` returns a verdict, and the call
    // sites in `verify.rs` map `Unknown` onto a revert — so an obligation
    // routed and then found inexpressible would cost an optimization that
    // the incumbent could have proven. Deferring instead hands it back
    // untouched. The cost is that the lowering runs twice on the memory
    // path; it is a tree rebuild against a SAT solve, and it buys an exact
    // defer reason.
    if let Err(reason) = sliver_admits(&goal_lhs, &goal_rhs) {
        return SeamOutcome::Deferred(reason);
    }

    SeamOutcome::Decided(prove(&goal_lhs, &goal_rhs, backend))
}

/// Count the memory accesses (`select`s and `store`s) in a term.
#[cfg(feature = "verification")]
fn memory_accesses(term: &RuleTerm) -> usize {
    use RuleTerm as T;
    fn arr(a: &RuleArray) -> usize {
        match a {
            RuleArray::Var { .. } => 0,
            RuleArray::Store {
                array,
                index,
                value,
            } => 1 + arr(array) + memory_accesses(index) + memory_accesses(value),
        }
    }
    fn boolean(b: &RuleBool) -> usize {
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
            | B::Sge(x, y) => memory_accesses(x) + memory_accesses(y),
            B::Not(x) => boolean(x),
            B::BoolAnd(x, y) | B::BoolOr(x, y) => boolean(x) + boolean(y),
        }
    }
    match term {
        T::Const { .. } | T::Var { .. } => 0,
        T::Select { array, index } => 1 + arr(array) + memory_accesses(index),
        // The call itself touches no memory, but its arguments can:
        // `f(load(p))` is one access. Counting it is what keeps the
        // budget honest for call-carrying obligations.
        T::PureCall { args, .. } => args.iter().map(memory_accesses).sum(),
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
        | T::Concat(a, b) => memory_accesses(a) + memory_accesses(b),
        T::Neg(a) | T::Not(a) => memory_accesses(a),
        T::Extract { arg, .. } | T::SignExt { arg, .. } | T::ZeroExt { arg, .. } => {
            memory_accesses(arg)
        }
        T::Ite { cond, then_, else_ } => {
            boolean(cond) + memory_accesses(then_) + memory_accesses(else_)
        }
    }
}

/// Can ordeal's array sliver take this obligation?
///
/// `Ok(())` for a pure obligation (nothing changes — no sliver is involved)
/// and for a memory obligation that lowers within budget. Every other answer
/// is a named refusal the caller defers on.
#[cfg(feature = "verification")]
fn sliver_admits(lhs: &RuleTerm, rhs: &RuleTerm) -> Result<(), DeferReason> {
    use crate::rule_solver::ordeal_backend::{GoalError, lower_goal_ne};

    let accesses = memory_accesses(lhs) + memory_accesses(rhs);
    if accesses == 0 {
        // Pure: slices 1-2 unchanged, and no array elimination to bound.
        return Ok(());
    }
    if accesses > max_memory_accesses() {
        return Err(DeferReason::TooManyMemoryAccesses);
    }
    match lower_goal_ne(lhs, rhs) {
        Ok(_) => Ok(()),
        Err(GoalError::Gap(gap)) => Err(DeferReason::MemoryUnderUnliftedOp(gap.0)),
        Err(GoalError::Sliver(e)) => Err(DeferReason::SliverRefused(match e {
            ordeal::sliver::SliverError::NonConcreteIndex => {
                "array sliver: non-concrete index (unexpected on ordeal >= 0.11)"
            }
            ordeal::sliver::SliverError::BadArraySort => "array sliver: array sort rejected",
            ordeal::sliver::SliverError::InconsistentCall { .. } => {
                "array sliver: inconsistent call signature"
            } // Deliberately exhaustive, with no catch-all. `SliverError` is
              // not `#[non_exhaustive]`, so a new refusal reason in a future
              // ordeal breaks this build — which is the right outcome: a new
              // way for the sliver to refuse is a new thing the seam has to
              // classify, not something to absorb under a wildcard.
        })),
    }
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
            // A `select` contributes NO trap clause of its own: loom's
            // memory model is a total `Array(BV32 -> BV8)` with no bounds
            // check, so an out-of-bounds load is not representable here, let
            // alone trapping. That is a REAL gap in what the obligation
            // covers and it is recorded as such in the slice-4 artifact —
            // but it is the incumbent's gap too, inherited unchanged, not
            // one this slice introduces.
            //
            // The children still have to be walked: an address can be
            // computed with a division, and so can a stored value.
            T::Select { array, index } => {
                walk_array(array, under_ite, acc)?;
                walk(index, under_ite, acc)?;
            }
            // A `pure_call` contributes no trap clause of its own — the
            // callee was proven pure and total before this encoding was
            // chosen, which is what licenses treating it as an uninterpreted
            // function at all. Its ARGUMENTS still have to be walked: an
            // argument can be computed with a division, and `under_ite` has
            // to carry into them unchanged because an argument is evaluated
            // exactly when the call is.
            T::PureCall { args, .. } => {
                for a in args {
                    walk(a, under_ite, acc)?;
                }
            }
        }
        Ok(())
    }

    fn walk_array(
        a: &RuleArray,
        under_ite: bool,
        acc: &mut Vec<RuleBool>,
    ) -> Result<(), DeferReason> {
        match a {
            RuleArray::Var { .. } => Ok(()),
            RuleArray::Store {
                array,
                index,
                value,
            } => {
                walk_array(array, under_ite, acc)?;
                walk(index, under_ite, acc)?;
                walk(value, under_ite, acc)
            }
        }
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
        use crate::rule_solver::ordeal_backend::{render_ordeal_model, solver_for_goal_ne};
        use ordeal::CheckResult;

        let solver = match solver_for_goal_ne(lhs, rhs) {
            Ok(solver) => solver,
            // `decide_bv_equivalence` gates this case and defers instead, so
            // reaching it means the obligation came from somewhere else.
            // `Unknown` is the conservative answer either way.
            Err(_) => return RuleVerdict::Unknown,
        };
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
    use crate::rule_solver::{RuleArray, RuleBool, RuleTerm};
    use std::collections::HashMap;
    use z3::ast::{Array, Ast, BV, Bool, Dynamic};
    use z3::{AstKind, DeclKind, Sort};

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
        /// Array name -> the Z3 array constant it came from. Separate from
        /// `vars` because the sorts are disjoint, but guarded identically:
        /// ordeal keys arrays by name alone, so two distinct Z3 memories
        /// sharing a name would be conflated — which is exactly the hazard
        /// that would let two HAVOC points be proven equal (#313 slice 4).
        arrays: HashMap<String, Array>,
        /// `pure_call` name -> (argument widths, result width). ordeal keys
        /// an uninterpreted function by name alone, so one name carrying two
        /// signatures would conflate two different functions — the same
        /// hazard `vars` and `arrays` guard against, one level up.
        calls: HashMap<String, (Vec<u32>, u32)>,
        budget: usize,
    }

    impl Reflector {
        pub(super) fn new(max_nodes: usize) -> Self {
            Reflector {
                vars: HashMap::new(),
                arrays: HashMap::new(),
                calls: HashMap::new(),
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
            reserved_name_check(&name)?;
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

        /// Reflect a `pure_call_<F>(args)` application into
        /// [`RuleTerm::PureCall`], refusing any name reuse that ordeal's
        /// name-keyed congruence could conflate.
        ///
        /// The arguments are reflected first so that a refusal from deeper in
        /// the tree wins over the signature check — a call whose argument is
        /// out of fragment is out of fragment regardless of its name.
        fn reflect_call(
            &mut self,
            name: String,
            width: u32,
            node: &BV,
            n: usize,
        ) -> Reflected<RuleTerm> {
            reserved_name_check(&name)?;
            let mut args = Vec::with_capacity(n);
            for i in 0..n {
                args.push(self.reflect_bv(&child_bv(node, i)?)?);
            }
            let signature: Vec<u32> = args.iter().map(RuleTerm::width).collect();
            match self.calls.get(&name) {
                Some((prev_sig, prev_width)) => {
                    if *prev_sig != signature || *prev_width != width {
                        return Err(DeferReason::CallSignatureCollision);
                    }
                }
                None => {
                    self.calls.insert(name.clone(), (signature, width));
                }
            }
            Ok(RuleTerm::PureCall { name, args, width })
        }

        /// Record a free ARRAY variable under the same name discipline as
        /// [`Reflector::intern`].
        fn intern_array(&mut self, name: String, ast: &Array) -> Reflected<RuleArray> {
            reserved_name_check(&name)?;
            match self.arrays.get(&name) {
                Some(prev) => {
                    if prev != ast {
                        return Err(DeferReason::VariableNameCollision);
                    }
                }
                None => {
                    self.arrays.insert(name.clone(), ast.clone());
                }
            }
            Ok(RuleArray::Var { name })
        }

        /// Reflect an array-sorted node into the neutral array sub-language.
        ///
        /// Admits exactly two shapes — a named `Array(BV32 -> BV8)` constant
        /// and a `store` over one. Everything else is refused, and the
        /// refusals are load-bearing rather than incidental:
        ///
        /// * an array-sorted **`ite`** is what a control-flow merge over two
        ///   memory states produces. ordeal's extended language has no array
        ///   `ite`, so there is nothing to lower it to (ordeal#231);
        /// * a **const-array** (`((as const (Array ...)) v)`) has no neutral
        ///   form either, and silently treating it as a fresh variable would
        ///   DROP the constraint that every byte equals `v` — unsound in the
        ///   direction that matters, because it would make more things
        ///   provable, not fewer.
        fn reflect_array(&mut self, node: &Array) -> Reflected<RuleArray> {
            self.reflect_array_at(node, 0)
        }

        /// Maximum `store` chain depth.
        ///
        /// Separate from the node budget because they bound different
        /// things. [`super::MAX_TERM_NODES`] caps TOTAL nodes; a store chain
        /// is walked by RECURSION, so a long one costs stack proportional to
        /// its length before the node budget has been spent. 4096 nodes of
        /// chain is several megabytes of debug frames, which overflows a
        /// test thread's 2 MiB stack — an abort, not the conservative revert
        /// this seam is built around.
        ///
        /// 256 byte-stores is 64 `i32` stores in one straight-line run, and
        /// such a function exhausts the node budget long before this, so
        /// this is a safety floor rather than a coverage limit.
        const MAX_ARRAY_CHAIN: usize = 256;

        fn reflect_array_at(&mut self, node: &Array, depth: usize) -> Reflected<RuleArray> {
            self.spend()?;
            if depth >= Self::MAX_ARRAY_CHAIN {
                return Err(DeferReason::TooLarge);
            }

            // One array sort, checked rather than assumed: the neutral
            // `RuleArray` hard-codes BV32 -> BV8 at both lowerings, so a
            // differently-sorted array must never reach it.
            if node.get_sort() != Sort::array(&Sort::bitvector(32), &Sort::bitvector(8)) {
                return Err(DeferReason::OutOfFragment(
                    "array whose sort is not BV32 -> BV8",
                ));
            }
            if node.kind() != AstKind::App {
                return Err(DeferReason::OutOfFragment("non-application array node"));
            }
            let decl = node
                .safe_decl()
                .map_err(|_| DeferReason::OutOfFragment("array node is not an application"))?;
            let n = node.num_children();

            if decl.kind() == DeclKind::UNINTERPRETED {
                if n == 0 {
                    return self.intern_array(decl.name(), node);
                }
                return Err(DeferReason::OutOfFragment(
                    "uninterpreted array-valued function application",
                ));
            }

            if decl.kind() == DeclKind::STORE {
                if n != 3 {
                    return Err(DeferReason::OutOfFragment("store with != 3 arguments"));
                }
                let inner = node
                    .nth_child(0)
                    .and_then(|d| d.as_array())
                    .ok_or(DeferReason::OutOfFragment("store target is not an array"))?;
                let index = child_bv_of(node, 1)?;
                let value = child_bv_of(node, 2)?;
                // Implied by the sort check above; asserted so a future sort
                // widening cannot quietly change what gets stored.
                if index.get_size() != 32 || value.get_size() != 8 {
                    return Err(DeferReason::OutOfFragment(
                        "store index/value is not BV32/BV8",
                    ));
                }
                return Ok(RuleArray::Store {
                    array: Box::new(self.reflect_array_at(&inner, depth + 1)?),
                    index: Box::new(self.reflect_bv(&index)?),
                    value: Box::new(self.reflect_bv(&value)?),
                });
            }

            Err(DeferReason::OutOfFragment(
                "array term that is neither a named memory nor a store",
            ))
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
            // uninterpreted decl WITH arguments is loom's
            // `pure_call_<F>(args)` encoding of a call to a provably pure
            // callee, which the sliver carries as an uninterpreted function
            // (#313 Tier-2 slice 3). The comment here used to say "slice 2";
            // the artifact has always said slice 3.
            if kind == DeclKind::UNINTERPRETED {
                if n == 0 {
                    return self.intern(decl.name(), width, node);
                }
                return self.reflect_call(decl.name(), width, node, n);
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

                // --- memory: Array theory, slice 4 ---
                //
                // A `select` is the only array-theory node that is
                // bitvector-sorted, so it is the only one that can appear
                // here; a `store` is array-sorted and is reached through
                // `reflect_array` instead. A BV-sorted `store` cannot exist,
                // and refusing it costs nothing.
                DeclKind::SELECT => {
                    if n != 2 {
                        return Err(DeferReason::OutOfFragment("select with != 2 arguments"));
                    }
                    if width != 8 {
                        return Err(DeferReason::OutOfFragment("select result is not a byte"));
                    }
                    let array = node
                        .nth_child(0)
                        .and_then(|d| d.as_array())
                        .ok_or(DeferReason::OutOfFragment("select source is not an array"))?;
                    let index = child_bv(node, 1)?;
                    if index.get_size() != 32 {
                        return Err(DeferReason::OutOfFragment("select index is not BV32"));
                    }
                    Ok(RuleTerm::Select {
                        array: Box::new(self.reflect_array(&array)?),
                        index: Box::new(self.reflect_bv(&index)?),
                    })
                }
                DeclKind::STORE => Err(DeferReason::OutOfFragment(
                    "bitvector-sorted store (not a shape Z3 builds)",
                )),

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

    fn child_bv_of(node: &Array, idx: usize) -> Reflected<BV> {
        as_bv(node.nth_child(idx))
    }

    /// ordeal's sliver mints its own core variables while eliminating
    /// arrays, named `$sel:<array>:<index>`. It interns every variable by
    /// name alone, so a loom-supplied name in that namespace could be
    /// conflated with a lowering-internal one. loom never produces such a
    /// name, but "never produces" is an assumption about a 3000-line
    /// encoder, so it is checked rather than relied upon.
    fn reserved_name_check(name: &str) -> Reflected<()> {
        if name.starts_with('$') {
            return Err(DeferReason::OutOfFragment(
                "name in the solver's reserved `$` namespace",
            ));
        }
        Ok(())
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

    /// Every trapping operator against a HAND-AUTHORED table of the wasm
    /// spec, at the points where the four kinds disagree.
    ///
    /// # Why a table and not a differential
    ///
    /// Slice 2 introduced a SECOND trap-condition implementation: this
    /// module's `trap_condition` over the neutral term, beside
    /// `trap_gate`'s predicates over ordeal terms for the constant-fold
    /// backstop. That duplication is recorded as outstanding in
    /// `TEST-313-TIER2-SLICE2-PARTIAL-OPS`, and the obvious mitigation is a
    /// differential between the two.
    ///
    /// This is deliberately not that. Two implementations agreeing with each
    /// other can both be wrong in the same way — and they would be, because
    /// they were written from the same reading. A table written from the
    /// spec pins each of them to the spec instead, which is the claim that
    /// actually matters. ordeal's own `rem_s` bug (ordeal#72/#84) was exactly
    /// a case where a second implementation would have agreed.
    ///
    /// `trap` rows assert REJECTION of any fold to a value, because a
    /// trapping op has no value to fold to. `no trap` rows assert the fold
    /// to the spec's value is ACCEPTED, which checks the trap condition is
    /// not over-approximated as well as the value being right.
    #[test]
    fn the_four_trapping_ops_match_a_spec_table() {
        // (name, build, expected): `expected` is None for "traps", or
        // Some(bits) for "does not trap, and the value is these bits".
        type Build = fn(RuleTerm, RuleTerm) -> RuleTerm;
        let udiv: Build = |a, b| RuleTerm::Udiv(Box::new(a), Box::new(b));
        let urem: Build = |a, b| RuleTerm::Urem(Box::new(a), Box::new(b));
        let sdiv: Build = |a, b| RuleTerm::Sdiv(Box::new(a), Box::new(b));
        let srem: Build = |a, b| RuleTerm::Srem(Box::new(a), Box::new(b));

        const W: u32 = 32;
        let int_min: u128 = 1u128 << 31;
        let neg = |x: i64| -> u128 { (x as u32) as u128 };

        #[allow(clippy::type_complexity)]
        let table: Vec<(&str, Build, u128, u128, Option<u128>)> = vec![
            // divide-by-zero traps for ALL FOUR kinds.
            ("div_u by zero", udiv, 7, 0, None),
            ("rem_u by zero", urem, 7, 0, None),
            ("div_s by zero", sdiv, 7, 0, None),
            ("rem_s by zero", srem, 7, 0, None),
            // signed overflow: div_s TRAPS, rem_s does NOT. This is the row
            // pair that makes the asymmetry an asymmetry.
            ("div_s INT_MIN / -1", sdiv, int_min, neg(-1), None),
            ("rem_s INT_MIN / -1", srem, int_min, neg(-1), Some(0)),
            // ordinary arithmetic: no trap, and the spec's value.
            ("div_u 7/3", udiv, 7, 3, Some(2)),
            ("rem_u 7/3", urem, 7, 3, Some(1)),
            // signed division truncates TOWARD ZERO (not floor), so
            // -6/3 = -2 and -7 % 3 = -1 (sign follows the dividend).
            ("div_s -6/3", sdiv, neg(-6), 3, Some(neg(-2))),
            ("rem_s -7/3", srem, neg(-7), 3, Some(neg(-1))),
            ("div_s 7/-2", sdiv, 7, neg(-2), Some(neg(-3))),
            ("rem_s 7/-2", srem, 7, neg(-2), Some(1)),
            // INT_MIN with a divisor that does NOT overflow.
            ("div_s INT_MIN / 1", sdiv, int_min, 1, Some(int_min)),
            ("rem_s INT_MIN / 1", srem, int_min, 1, Some(0)),
        ];

        with_z3_config(&cfg(), || {
            for (name, build, a, b, expected) in &table {
                let op = build(k(*a, W), k(*b, W));
                match expected {
                    None => {
                        // Traps: folding it to ANY value removes a mandatory
                        // trap. Two different targets, so the rejection
                        // cannot be an accident of one chosen constant.
                        for target in [k(0, W), k(u32::MAX as u128, W)] {
                            let verdict = decide_terms(&op, &target);
                            assert!(
                                matches!(verdict, RuleVerdict::Disproven(_)),
                                "{name}: traps per the spec, so folding it to a \
                                 value must be REJECTED; got {verdict:?}"
                            );
                        }
                    }
                    Some(bits) => {
                        let verdict = decide_terms(&op, &k(*bits, W));
                        assert!(
                            matches!(verdict, RuleVerdict::Proven),
                            "{name}: does NOT trap per the spec and evaluates to \
                             {bits:#x}, so this fold is sound and must be \
                             ACCEPTED. A rejection means the trap condition is \
                             over-approximated or the value is wrong; got \
                             {verdict:?}"
                        );
                        // And the dual: folding to the WRONG value must be
                        // rejected, or "accepted" above would also hold for a
                        // relation that accepts everything.
                        let wrong = k(bits.wrapping_add(1) & 0xffff_ffff, W);
                        let verdict = decide_terms(&op, &wrong);
                        assert!(
                            matches!(verdict, RuleVerdict::Disproven(_)),
                            "{name}: folding to a wrong value must be REJECTED; \
                             got {verdict:?}"
                        );
                    }
                }
            }
        });
    }

    /// A division under a conditional is REFUSED rather than approximated.
    ///
    /// A division in an untaken wasm `if` arm does not execute and so does
    /// not trap; the honest condition is path-dependent. The flat
    /// disjunction this seam builds would over-approximate and silently
    /// reject correct transforms, so the obligation is declined instead —
    /// the same scope discipline slice 1 applied to memory.
    /// #433 / ordeal#247: a query variable must never be identified with the
    /// sliver's own fresh variable for a read.
    ///
    /// ordeal < 0.32.1 named the fresh variable for a concrete read
    /// `$sel:<array>:<index>` without checking the query's own names, so a
    /// query variable spelled that way was silently unified with the read and
    /// `select(memory, 5) != $sel:memory:5` — satisfiable, the two are
    /// independent — came back as a CERTIFIED `Unsat`. This drives the exact
    /// counterexample through the production solver.
    ///
    /// loom's own variable names cannot start with `$` (every Z3 name it
    /// creates begins with a literal letter), so production was not exposed;
    /// this pins the fix so a future pin or name scheme cannot reopen it.
    #[test]
    fn a_query_variable_cannot_masquerade_as_a_sliver_read() {
        use crate::rule_solver::RuleArray;
        let read = RuleTerm::Select {
            array: Box::new(RuleArray::Var {
                name: "memory".into(),
            }),
            index: Box::new(k(5, 32)),
        };
        let impostor = v("$sel:memory:5", 8);
        let verdict = BoundedOrdealSolver::from_env().prove_rule_equiv(&read, &impostor);
        assert!(
            matches!(verdict, RuleVerdict::Disproven(_)),
            "a read and an independent variable are NOT equal; got {verdict:?}"
        );
    }

    /// #433: a memory or call obligation must be decided by a solver built
    /// with `Solver::from_sliver`. From ordeal 0.33.0 only such a solver runs
    /// the trusted, Lean-proven `check_sliver_lowering` before it returns a
    /// verdict, which proves that the closed core it decides is exactly the
    /// read-over-write and congruence elimination of the query loom asked.
    /// Lowering by hand and asserting the core skips that check.
    /// `is_sliver()` reports which entry point built the solver. A pure
    /// obligation has nothing to eliminate and must stay on the plain path.
    #[test]
    fn memory_obligations_are_decided_on_the_proven_sliver_path() {
        use crate::rule_solver::RuleArray;
        use crate::rule_solver::ordeal_backend::solver_for_goal_ne;
        let read = |at: u128| RuleTerm::Select {
            array: Box::new(RuleArray::Var {
                name: "memory".into(),
            }),
            index: Box::new(k(at, 32)),
        };
        let memory = solver_for_goal_ne(&read(5), &read(6)).expect("in the sliver");
        assert!(
            memory.is_sliver(),
            "a memory obligation was decided on a solver that skips check_sliver_lowering"
        );
        let pure = solver_for_goal_ne(&v("x", 32), &v("y", 32)).expect("closed core");
        assert!(
            !pure.is_sliver(),
            "a pure obligation has no sliver to check"
        );
    }

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

            // Memory: slice 4 moved `select`/`store` INTO the fragment, so
            // this assertion is inverted rather than deleted — same
            // discipline slice 2 applied to division above. A load must now
            // reflect.
            let memory = Array::new_const("memory", &Sort::bitvector(32), &Sort::bitvector(8));
            let load = memory.select(&a).as_bv().unwrap();
            assert!(
                reflect::Reflector::new(MAX_TERM_NODES)
                    .reflect_bv(&load)
                    .is_ok(),
                "slice 4 routes memory reads; a select must reflect"
            );

            // What is still OUT on the memory side: an array-sorted `ite`,
            // which is what a control-flow merge over two memory states
            // produces. Neither the round-trip gate nor ordeal's extended
            // language has a form for it.
            let merged = a.eq(&b).ite(
                &memory,
                &Array::new_const("other", &Sort::bitvector(32), &Sort::bitvector(8)),
            );
            let load_merged = merged.select(&a).as_bv().unwrap();
            assert!(matches!(
                reflect_err(&load_merged),
                DeferReason::OutOfFragment(_)
            ));

            // An uninterpreted function application (the `pure_call` shape)
            // used to belong here. It is OUT of this list as of #313 slice 3
            // — it now reflects to `RuleTerm::PureCall` and routes. Moved
            // rather than deleted: see
            // `slice3_a_pure_call_is_routed_and_congruence_is_decided` and
            // `slice3_a_pure_call_is_not_a_constant`, which assert the
            // positive behaviour this line used to deny. What IS still
            // refused on the call side is a name reused with a different
            // signature — see `slice3_one_name_two_signatures_is_refused`.

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

    // ========================================================================
    // #313 slice 4 — havoc and memory
    //
    // The artifact `TEST-313-TIER2-SLICE4-HAVOC-CONCRETE-MEM` states four
    // acceptance criteria. Each has a test below named after it, plus the
    // aliasing pair that is the real content of a memory model.
    //
    // Every one of these asserts `route_counts().0 > 0`. Without that, a
    // regression that made the seam refuse memory again would leave each
    // test passing — the verdicts would still be right, because Z3 would
    // still be deciding them. "Decided by the new engine, not merely offered
    // to it" is the property (#289), so it is asserted, not assumed.
    // ========================================================================

    /// The shared `Array(BV32 -> BV8)` sort, as `verify.rs` builds it.
    fn mem(name: &str) -> Array {
        Array::new_const(name, &Sort::bitvector(32), &Sort::bitvector(8))
    }

    /// Decide `orig == opt`, asserting it was actually routed — on ordeal
    /// AND under `both`. Returns the verdict.
    ///
    /// Running both backends here rather than in a separate helper is
    /// deliberate. `LOOM_VERIFY_BACKEND` is not consulted by these tests
    /// (they pass the backend explicitly, so they are deterministic), which
    /// means a test that only ever passed `Ordeal` would leave the
    /// no-divergence claim resting on whichever cases happened to remember
    /// to ask for `Both`. Putting it in the shared helper makes every
    /// verdict-asserting slice-4 test a differential by construction.
    ///
    /// Under `Both` the seam runs Z3 and ordeal on the same obligation and
    /// PANICS on disagreement, so these calls passing IS the assertion.
    fn decide_routed(orig: &BV, opt: &BV) -> RuleVerdict {
        let mut first: Option<RuleVerdict> = None;
        for backend in [VerifyBackend::Ordeal, VerifyBackend::Both] {
            reset_route_counts();
            let outcome = decide_bv_equivalence(orig, opt, backend);
            let (routed, deferred) = route_counts();
            match outcome {
                SeamOutcome::Decided(v) => {
                    assert_eq!(
                        (routed, deferred),
                        (1, 0),
                        "the obligation must be DECIDED by the seam on \
                         {backend:?}, not left on the incumbent — otherwise \
                         this test would still pass with the memory path \
                         removed"
                    );
                    if first.is_none() {
                        first = Some(v);
                    }
                }
                SeamOutcome::Deferred(reason) => panic!(
                    "expected the memory obligation to route on {backend:?}, \
                     but it deferred: {}",
                    reason.label()
                ),
            }
        }
        first.expect("at least one backend was run")
    }

    /// Criterion 3: a store then a load of the SAME address reads back what
    /// was stored.
    ///
    /// Written at byte granularity, which is the granularity the model
    /// actually has — a wider load is four of these, and proving the byte
    /// case is what makes the wide case work.
    #[test]
    fn slice4_a_load_of_a_stored_address_is_provably_the_stored_value() {
        with_z3_config(&cfg(), || {
            let m = mem("memory");
            let addr = BV::new_const("addr", 32);
            let val = BV::new_const("val", 8);
            let stored = m.store(&addr, &val);
            let loaded: BV = stored.select(&addr).as_bv().unwrap();

            let verdict = decide_routed(&loaded, &val);
            assert!(
                matches!(verdict, RuleVerdict::Proven),
                "select(store(m, a, v), a) must be provably v — this is \
                 read-over-write, the whole basis of the memory encoding; \
                 got {verdict:?}"
            );
        });
    }

    /// Criterion 4, and the load-bearing one: a load at a DIFFERENT address
    /// is NOT forced to the stored value.
    ///
    /// This is the aliasing question, and it is where a memory model is
    /// usually wrong. Forwarding a store to a load at an unrelated address
    /// is only valid when the two addresses are equal — which is not known
    /// here, so the fold must be REJECTED.
    ///
    /// A model that collapsed all addresses to one cell, or that dropped the
    /// index from the read variable, would prove this and miscompile every
    /// program with two live pointers.
    #[test]
    fn slice4_a_load_of_a_different_address_is_not_forced_to_the_stored_value() {
        with_z3_config(&cfg(), || {
            let m = mem("memory");
            let p = BV::new_const("p", 32);
            let q = BV::new_const("q", 32);
            let val = BV::new_const("val", 8);
            let stored = m.store(&p, &val);
            let loaded_elsewhere: BV = stored.select(&q).as_bv().unwrap();

            let verdict = decide_routed(&loaded_elsewhere, &val);
            assert!(
                matches!(verdict, RuleVerdict::Disproven(_)),
                "select(store(m, p, v), q) with p and q unconstrained must NOT \
                 be provably v — p and q may differ. A `Proven` here means the \
                 model cannot distinguish two addresses; got {verdict:?}"
            );
        });
    }

    /// The same shape at CONCRETE indices, which is the case the artifact
    /// names and the one ordeal settles statically rather than handing to
    /// the SAT core.
    ///
    /// Both rows together: the matching index must be forced, the
    /// neighbouring one must not. The pair is the evidence — a model that
    /// always answered "forced" would pass the first alone, and one that
    /// always answered "not forced" would pass the second alone.
    #[test]
    fn slice4_concrete_indices_resolve_statically_in_both_directions() {
        with_z3_config(&cfg(), || {
            let m = mem("memory");
            let val = BV::new_const("val", 8);
            let at_100 = m.store(&BV::from_u64(100, 32), &val);

            let same: BV = at_100.select(&BV::from_u64(100, 32)).as_bv().unwrap();
            let other: BV = at_100.select(&BV::from_u64(101, 32)).as_bv().unwrap();

            assert!(
                matches!(decide_routed(&same, &val), RuleVerdict::Proven),
                "the byte written at 100 must read back at 100"
            );
            let verdict = decide_routed(&other, &val);
            assert!(
                matches!(verdict, RuleVerdict::Disproven(_)),
                "the byte at 101 is untouched by a write to 100 and must NOT \
                 be forced to it; got {verdict:?}"
            );
        });
    }

    /// Criterion 2: two distinct HAVOC points must not be unified.
    ///
    /// `verify.rs` havocs memory after an impure call by replacing it with a
    /// fresh `Array::new_const("mem_after_call_<f>_<n>", ..)`. Two different
    /// calls get different names and are genuinely unrelated, so a load from
    /// one must not be provably equal to a load from the other at the same
    /// address.
    ///
    /// The failure mode this guards is concrete: ordeal interns arrays by
    /// NAME ALONE. If the reflection let two distinct Z3 arrays share a
    /// neutral name, they would become one array in the lowered query and
    /// this would come out `Proven` — licensing a transform that moves a
    /// load across a call. That is the shape #283 suspects elsewhere, and
    /// the reason `intern_array` refuses a name collision.
    #[test]
    fn slice4_two_distinct_havoc_memories_are_not_provably_equal() {
        with_z3_config(&cfg(), || {
            let addr = BV::new_const("addr", 32);
            let first: BV = mem("mem_after_call_7_0").select(&addr).as_bv().unwrap();
            let second: BV = mem("mem_after_call_7_1").select(&addr).as_bv().unwrap();

            let verdict = decide_routed(&first, &second);
            assert!(
                matches!(verdict, RuleVerdict::Disproven(_)),
                "two distinct havoc memories are unrelated, so reading the same \
                 address from each must NOT be provably equal. `Proven` here \
                 would mean the two arrays were conflated — the exact hazard of \
                 name-keyed interning; got {verdict:?}"
            );

            // The control: the SAME havoc memory read twice IS equal. Without
            // this row the test above would also pass if every memory read
            // came back as a fresh unconstrained value.
            let again: BV = mem("mem_after_call_7_0").select(&addr).as_bv().unwrap();
            assert!(
                matches!(decide_routed(&first, &again), RuleVerdict::Proven),
                "the same address in the same memory must read the same value"
            );
        });
    }

    /// Criterion 1: a transform that assumes anything about a havoc value is
    /// rejected.
    ///
    /// A havoc'd byte is unconstrained, so `v / v` is not 1 (it traps when
    /// `v` is 0), `v | 1` is not `v`, and `v` is not any particular
    /// constant. The last is the one asserted here, with the control that an
    /// identity on the same value IS accepted.
    #[test]
    fn slice4_a_transform_assuming_a_havoc_value_is_rejected() {
        with_z3_config(&cfg(), || {
            let addr = BV::new_const("addr", 32);
            let havoc: BV = mem("mem_after_call_3_0").select(&addr).as_bv().unwrap();

            // Both directions, because only the PAIR says "unconstrained".
            // Rejecting the fold to 0 alone would also hold for a value
            // known to be non-zero, and rejecting the fold to 1 alone would
            // hold for a value known to be zero. Asserting both is what
            // states that NO constant is forced — which is the criterion
            // ("a transform that assumes a havoc value is non-zero is
            // rejected") and its generalisation at once.
            for assumed in [0u64, 1u64] {
                let verdict = decide_routed(&havoc, &BV::from_u64(assumed, 8));
                assert!(
                    matches!(verdict, RuleVerdict::Disproven(_)),
                    "a havoc'd byte is unconstrained; folding it to {assumed} \
                     must be rejected. got {verdict:?}"
                );
            }

            // Control: a real identity over the same havoc value is accepted,
            // so the rejection above is about the ASSUMPTION, not about the
            // seam refusing everything that mentions memory.
            assert!(
                matches!(
                    decide_routed(&havoc.bvor(&havoc), &havoc),
                    RuleVerdict::Proven
                ),
                "`v | v == v` holds for any v, havoc'd or not"
            );
        });
    }

    /// Criterion 1 proper: a transform correct for every value EXCEPT zero
    /// must be rejected on a havoc'd byte.
    ///
    /// The `fold to 0` / `fold to 1` rows above are necessary but not
    /// sufficient. A havoc value secretly constrained to, say, `2..=255`
    /// would satisfy both of them — and that constraint is exactly the
    /// failure the criterion exists to catch. Neither row ever shows the
    /// byte can BE zero.
    ///
    /// This one does. `v | ((zext(v,24) - 1) >>u 31)[7:0]` equals `v` for
    /// every non-zero `v`, and equals `1` when `v` is `0` (the subtraction
    /// wraps, setting the sign bit). So folding it to `v` is sound exactly
    /// when `v != 0`, and a REJECTION is the proof that zero is reachable.
    /// Built from `or`/`zero_ext`/`sub`/`lshr`/`extract` only — all
    /// liftable, so it routes on the pinned ordeal rather than deferring.
    #[test]
    fn slice4_a_transform_assuming_a_havoc_value_is_nonzero_is_rejected() {
        with_z3_config(&cfg(), || {
            let addr = BV::new_const("addr", 32);
            let havoc: BV = mem("mem_after_call_9_0").select(&addr).as_bv().unwrap();

            let is_zero = havoc
                .zero_ext(24)
                .bvsub(BV::from_u64(1, 32))
                .bvlshr(BV::from_u64(31, 32))
                .extract(7, 0);
            let same_unless_zero = havoc.bvor(&is_zero);

            let verdict = decide_routed(&same_unless_zero, &havoc);
            assert!(
                matches!(verdict, RuleVerdict::Disproven(_)),
                "a fold that is correct for every NON-ZERO byte must be \
                 rejected on a havoc'd byte, because nothing rules zero out. \
                 `Proven` here would mean the havoc value is secretly \
                 constrained away from 0; got {verdict:?}"
            );
        });
    }

    /// Where the memory-access budget actually bites, measured.
    ///
    /// [`DEFAULT_MAX_MEMORY_ACCESSES`] counts nodes of the reflected tree,
    /// which is the Z3 DAG expanded — so the bound is far tighter in wasm
    /// terms than the raw number suggests. Rather than leave that as a
    /// comment, this walks `i32` store/load PAIRS until the seam defers and
    /// prints where the edge is, asserting only the two things that must
    /// hold: one pair fits, and the budget does eventually bite.
    #[test]
    fn slice4_the_memory_budget_is_measured() {
        with_z3_config(&cfg(), || {
            let mut last_routed = 0usize;
            let mut first_deferred = None;
            let mut timings: Vec<(usize, usize, u128, String)> = Vec::new();
            for pairs in [1usize, 2, 4, 8, 10, 12, 14, 16] {
                let m = mem("memory");
                let value = BV::new_const("value", 32);
                let mut stored = m.clone();
                for p in 0..pairs as i64 {
                    let base = BV::new_const(format!("addr{p}"), 32);
                    for i in 0..4i64 {
                        let byte = value.extract((i * 8 + 7) as u32, (i * 8) as u32);
                        stored = stored.store(&base.bvadd(BV::from_i64(i, 32)), &byte);
                    }
                }
                let mut acc = BV::from_i64(0, 32);
                for p in 0..pairs as i64 {
                    let base = BV::new_const(format!("addr{p}"), 32);
                    for i in 0..4i64 {
                        let b: BV = stored
                            .select(&base.bvadd(BV::from_i64(i, 32)))
                            .as_bv()
                            .unwrap();
                        acc = acc.bvor(b.zero_ext(24).bvshl(BV::from_i64(i * 8, 32)));
                    }
                }
                reset_route_counts();
                match decide_bv_equivalence(&acc, &acc, VerifyBackend::Ordeal) {
                    SeamOutcome::Decided(_) => last_routed = pairs,
                    SeamOutcome::Deferred(r) => {
                        if first_deferred.is_none() {
                            first_deferred = Some((pairs, r.label()));
                        }
                    }
                }

                // What the bound is protecting against, measured directly:
                // reflect and solve with a generous deadline, bypassing the
                // budget, and time it. A bound chosen without this is a
                // guess; with it, the default can be defended or moved on
                // evidence.
                let mut r = reflect::Reflector::new(MAX_TERM_NODES);
                if let Ok(t) = r.reflect_bv(&acc) {
                    let goal = trap_guarded(&t, None);
                    let started = std::time::Instant::now();
                    let verdict =
                        BoundedOrdealSolver { timeout_ms: 60_000 }.prove_rule_equiv(&goal, &goal);
                    timings.push((
                        pairs,
                        memory_accesses(&goal),
                        started.elapsed().as_millis(),
                        format!("{verdict:?}")
                            .split('(')
                            .next()
                            .unwrap()
                            .to_string(),
                    ));
                }
            }
            println!(
                "memory budget (default {DEFAULT_MAX_MEMORY_ACCESSES}): largest routed = {last_routed} i32 store/load pair(s); first deferral = {first_deferred:?}"
            );
            for (pairs, accesses, ms, verdict) in &timings {
                println!("  {pairs} pair(s): {accesses} accesses, solved in {ms} ms -> {verdict}");
            }
            assert!(
                last_routed >= 4,
                "at least four i32 store/load pairs must fit, or the budget is \
                 discarding coverage the solver demonstrably handles in \
                 milliseconds (got {last_routed})"
            );
            assert!(
                first_deferred.is_some(),
                "nothing bounded the obligation within the probed range, so \
                 this test measures no edge at all — widen the range"
            );
        });
    }

    /// A full little-endian `i32` load forwarded from the four stores that
    /// wrote it — the shape `encode_i32_load_from` and
    /// `encode_i32_store_into` actually produce.
    ///
    /// The byte-level tests above are the semantics; this is the evidence
    /// that the operators loom really emits (`bvadd`, `zero_ext`, `bvshl`,
    /// `bvor`, `select`, `store`, `extract`) all survive the reflection, the
    /// round trip, and the extended lowering together.
    #[test]
    fn slice4_a_full_i32_store_then_load_round_trips_through_the_seam() {
        with_z3_config(&cfg(), || {
            let m = mem("memory");
            let addr = BV::new_const("addr", 32);
            let value = BV::new_const("value", 32);

            let mut stored = m.clone();
            for i in 0..4i64 {
                let byte = value.extract((i * 8 + 7) as u32, (i * 8) as u32);
                stored = stored.store(&addr.bvadd(BV::from_i64(i, 32)), &byte);
            }
            let mut loaded = BV::from_i64(0, 32);
            for i in 0..4i64 {
                let b: BV = stored
                    .select(&addr.bvadd(BV::from_i64(i, 32)))
                    .as_bv()
                    .unwrap();
                loaded = loaded.bvor(b.zero_ext(24).bvshl(BV::from_i64(i * 8, 32)));
            }

            let verdict = decide_routed(&loaded, &value);
            assert!(
                matches!(verdict, RuleVerdict::Proven),
                "storing an i32 and loading it back at the same address must be \
                 provably the identity; got {verdict:?}"
            );
        });
    }

    // ───────────────────────── #313 Tier-2 slice 3 ─────────────────────────
    //
    // `pure_call_<F>(args)` is loom's encoding of a call to a provably pure
    // callee. The sliver carries it as an uninterpreted function, and the ONE
    // thing asserted about it is congruence: equal argument tuples yield
    // equal results. That is exactly what licenses CSE across calls, so it is
    // asserted positively below rather than assumed.
    //
    // Before this slice the reflector refused the shape outright
    // (`OutOfFragment("uninterpreted function application")`); that assertion
    // lived in `out_of_fragment_obligations_are_refused` and is now inverted.

    /// Criterion [1]: equal arguments force equal results, and the obligation
    /// is DECIDED by the new engine rather than merely offered to it
    /// (`decide_routed` asserts routed == 1 / deferred == 0 on both `Ordeal`
    /// and `Both`).
    #[test]
    fn slice3_a_pure_call_is_routed_and_congruence_is_decided() {
        with_z3_config(&cfg(), || {
            let f = z3::FuncDecl::new("pure_call_f", &[&Sort::bitvector(32)], &Sort::bitvector(32));
            let x = BV::new_const("x", 32);

            // f(x) vs f(x) — the same argument term, so congruence alone
            // settles it.
            let lhs = f.apply(&[&x]).as_bv().unwrap();
            let rhs = f.apply(&[&x]).as_bv().unwrap();

            let verdict = decide_routed(&lhs, &rhs);
            assert!(
                matches!(verdict, RuleVerdict::Proven),
                "equal arguments must force equal results; got {verdict:?}"
            );
        });
    }

    /// Criterion [2], and the load-bearing half: an uninterpreted function is
    /// NOT a constant.
    ///
    /// Criterion [1] on its own is satisfied by an engine that answers
    /// `Proven` to everything, and equally by one that models the call as a
    /// single opaque constant — both would "prove" congruence while being
    /// unsound for CSE. Unconstrained arguments must therefore NOT yield
    /// provably equal results. This is the test that distinguishes a real UF
    /// from either degenerate reading.
    #[test]
    fn slice3_a_pure_call_is_not_a_constant() {
        with_z3_config(&cfg(), || {
            let f = z3::FuncDecl::new("pure_call_f", &[&Sort::bitvector(32)], &Sort::bitvector(32));
            let x = BV::new_const("x", 32);
            let y = BV::new_const("y", 32);

            let fx = f.apply(&[&x]).as_bv().unwrap();
            let fy = f.apply(&[&y]).as_bv().unwrap();

            // DISPROVEN, not merely "not Proven". `Unknown` would also
            // satisfy a `!Proven` assertion while telling us nothing, so this
            // demands an actual counterexample — a model in which the two
            // applications differ. Measured: the engine returns e.g.
            //   $uf:pure_call_f:0 -> 0x0,  $uf:pure_call_f:1 -> 0x80000000
            //   x -> 0x0,                  y -> 0x80000000
            // i.e. it genuinely constructs a function separating the two
            // arguments. (`$uf:…` is ordeal's own reserved namespace for UF
            // results, which is why `reserved_name_check` refuses `$` names
            // coming from loom.)
            let verdict = decide_routed(&fx, &fy);
            assert!(
                matches!(verdict, RuleVerdict::Disproven(_)),
                "f(x) and f(y) with x and y unconstrained must be DISPROVEN \
                 with a counterexample — proving them equal would mean the \
                 call is modelled as a constant and would license unsound \
                 CSE, and `Unknown` would leave that untested; got {verdict:?}"
            );
        });
    }

    /// One name must not carry two signatures.
    ///
    /// ordeal keys an uninterpreted function by NAME for congruence, so two
    /// genuinely different functions sharing a name would be conflated and
    /// their unequal results could be proven equal. ordeal would refuse this
    /// itself (`SliverError::InconsistentCall`); loom refuses first so the
    /// deferral carries a reason that names the cause.
    #[test]
    fn slice3_one_name_two_signatures_is_refused() {
        with_z3_config(&cfg(), || {
            let x = BV::new_const("x", 32);
            let narrow = BV::new_const("n", 8);

            let g32 =
                z3::FuncDecl::new("pure_call_g", &[&Sort::bitvector(32)], &Sort::bitvector(32));
            let g8 = z3::FuncDecl::new("pure_call_g", &[&Sort::bitvector(8)], &Sort::bitvector(32));

            // Both applications in ONE term, so a single reflection sees the
            // conflict. Two separate obligations would each reflect fine.
            let wide = g32.apply(&[&x]).as_bv().unwrap();
            let narrow_app = g8.apply(&[&narrow]).as_bv().unwrap();
            let combined = wide.bvadd(narrow_app);

            assert_eq!(
                reflect_err(&combined),
                DeferReason::CallSignatureCollision,
                "one name at two signatures must be refused by NAME, not \
                 silently conflated"
            );
        });
    }

    /// ordeal#231 is CLOSED (ordeal 0.29.0, taken in #416): the extended
    /// language lifts `ite` over extended children, so memory under a
    /// conditional now ROUTES AND IS DECIDED instead of deferring.
    ///
    /// This test asserted the opposite until #416. It is inverted rather
    /// than deleted, because an inverted test is the only thing that shows
    /// the lift actually took effect. Bumping the dependency alone does not:
    /// the gap was produced by loom's own `lower_ext`, and with ordeal
    /// 0.30.0 already in the lockfile but the `Ite` arm unchanged, this test
    /// still passed in its old deferring form.
    #[test]
    fn slice4_memory_under_a_conditional_is_routed_and_decided() {
        with_z3_config(&cfg(), || {
            let m = mem("memory");
            let p = BV::new_const("p", 32);
            let q = BV::new_const("q", 32);

            // An `ite` ABOVE two selects — `wasm select`, or an `if`
            // choosing between two LOADED values. This is the shape that
            // was inexpressible.
            let left: BV = m.select(&p).as_bv().unwrap();
            let right: BV = m.select(&q).as_bv().unwrap();
            let chosen = p.bvult(&q).ite(&left, &right);

            reset_route_counts();
            let outcome = decide_bv_equivalence(&chosen, &chosen, VerifyBackend::Ordeal);
            assert!(
                matches!(outcome, SeamOutcome::Decided(RuleVerdict::Proven)),
                "memory under a conditional must now route and be decided, \
                 not {outcome:?}"
            );
            assert_eq!(
                route_counts(),
                (1, 0),
                "the obligation must be routed rather than deferred"
            );

            // ... and the decision has to discriminate. Reflexivity is
            // trivially provable, so the assertion above would also pass
            // against an engine that answered Proven to everything —
            // exactly the failure mode the slice-4 review found three times.
            // Swapping the arms changes the value whenever the two loads
            // differ, so it must NOT be provable.
            let swapped = p.bvult(&q).ite(&right, &left);
            let verdict = decide_routed(&chosen, &swapped);
            assert!(
                !matches!(verdict, RuleVerdict::Proven),
                "swapping the arms of a conditional over two DIFFERENT loads \
                 changes the value, so it must not be proven equivalent; \
                 got {verdict:?}"
            );
        });
    }

    /// The other side of ordeal#231, and the reason the wall is narrower
    /// than it first looks: an `ite` BENEATH a select routes.
    ///
    /// A load at a conditionally computed address — `base + (c ? i : j)` —
    /// has its conditional entirely inside the index, where the extended
    /// language embeds it verbatim as a closed-core subterm. Only an `ite`
    /// whose arms or condition CONTAIN a select is inexpressible.
    ///
    /// This test is what stops the refusal above from being widened to "any
    /// memory near a conditional", which would defer a large and routable
    /// class for no reason.
    #[test]
    fn slice4_a_load_at_a_conditionally_computed_address_still_routes() {
        with_z3_config(&cfg(), || {
            let m = mem("memory");
            let p = BV::new_const("p", 32);
            let q = BV::new_const("q", 32);
            let addr = p.bvult(&q).ite(&p, &q);
            let loaded: BV = m.select(&addr).as_bv().unwrap();

            let verdict = decide_routed(&loaded, &loaded);
            assert!(
                matches!(verdict, RuleVerdict::Proven),
                "a load at a conditionally computed address is expressible and \
                 must route; got {verdict:?}"
            );
        });
    }

    /// An oversized memory obligation DEFERS instead of building a lowered
    /// core nothing bounds.
    ///
    /// Two budgets can stop it — [`MAX_TERM_NODES`] on the reflected tree and
    /// [`DEFAULT_MAX_MEMORY_ACCESSES`] on the array elimination — and which
    /// fires first depends on the shape, because both scale with the
    /// expanded tree. The property worth asserting is that one of them does,
    /// not which; `slice4_the_memory_budget_is_measured` reports the edge.
    #[test]
    fn slice4_an_oversized_memory_obligation_defers() {
        with_z3_config(&cfg(), || {
            let addr = BV::new_const("addr", 32);
            let mut m = mem("memory");
            for i in 0..2000i64 {
                m = m.store(&addr.bvadd(BV::from_i64(i, 32)), &BV::from_u64(0, 8));
            }
            let loaded: BV = m.select(&addr).as_bv().unwrap();

            reset_route_counts();
            let outcome = decide_bv_equivalence(&loaded, &loaded, VerifyBackend::Ordeal);
            match outcome {
                SeamOutcome::Deferred(
                    DeferReason::TooManyMemoryAccesses | DeferReason::TooLarge,
                ) => {}
                other => panic!(
                    "a 2000-store chain must defer on one of the two budgets, \
                     not {other:?} — otherwise nothing bounds the lowered core"
                ),
            }
            assert_eq!(route_counts(), (0, 1), "nothing may be routed");
        });
    }

    /// An array-sorted `ite` — what a control-flow merge over two memory
    /// states produces — is refused at REFLECTION, before any lowering.
    #[test]
    fn slice4_an_array_valued_conditional_is_refused() {
        with_z3_config(&cfg(), || {
            let a = mem("mem_a");
            let b = mem("mem_b");
            let c = Bool::new_const("c");
            let merged = c.ite(&a, &b);
            let loaded: BV = merged.select(&BV::new_const("addr", 32)).as_bv().unwrap();

            reset_route_counts();
            let outcome = decide_bv_equivalence(&loaded, &loaded, VerifyBackend::Ordeal);
            assert!(
                matches!(outcome, SeamOutcome::Deferred(_)),
                "an array-valued ite has no neutral form and must be refused; \
                 got {outcome:?}"
            );
        });
    }

    /// The reserved-namespace guard. ordeal mints `$sel:<array>:<index>`
    /// variables while eliminating arrays and interns every variable by name
    /// alone, so a loom name in that namespace could be conflated with one
    /// of them.
    #[test]
    fn slice4_a_variable_in_the_solvers_reserved_namespace_is_refused() {
        with_z3_config(&cfg(), || {
            let shady = BV::new_const("$sel:memory:0", 8);
            let reason = reflect_err(&shady.bvadd(BV::from_u64(1, 8)));
            assert!(
                matches!(reason, DeferReason::OutOfFragment(what) if what.contains('$')),
                "a `$`-prefixed name must be refused; got {reason:?}"
            );
        });
    }
}
