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
//! # Slice 2: the relation changes (PARTIAL operations)
//!
//! Slice 1 did **not** change the correctness relation: for pure BV there are no
//! traps and no nondeterminism, so trap-equivalence + value-refinement
//! degenerates to the equality the validator already proves.
//!
//! Slice 2 admits WASM's integer `div`/`rem` — operations that **trap** — and
//! with them the real relation:
//!
//! ```text
//! UNSAT(  orig.trap ≠ opt.trap
//!       ∨ (¬orig.trap ∧ opt.value ∉ orig.valueSet) )
//! ```
//!
//! Two halves, deliberately asymmetric:
//!
//!   * **Traps are an EQUIVALENCE, not a refinement.** Alive2's relation says
//!     "where the original was UB, the optimized may do anything" — correct for
//!     LLVM, where UB is a licence. WASM has no such UB: a trap is a *defined,
//!     deterministic, observable* outcome. Transplanting refinement would
//!     licence *deleting a mandatory trap*, which is exactly the miscompile
//!     class (#273/#274/#276/#278/#281) this work exists to close. So neither
//!     dropping nor adding a trap is sound, in either direction, and the clause
//!     is `≠` — not `⇒`.
//!   * **Values are refinement-shaped** (`opt.value ∈ orig.valueSet`). WASM's
//!     genuine value nondeterminism is narrow — NaN payloads, `memory.grow`
//!     failure, relaxed SIMD — and none of it can occur in this slice, so every
//!     [`ValueSet`] the reflection builds here is a singleton and the clause
//!     degenerates to equality. See [`ValueSet`] for why the general shape
//!     exists anyway and what test keeps it honest.
//!
//! The `¬orig.trap` guard is not decoration. `UNSAT` of the disjunction gives
//! `orig.trap ⇔ opt.trap` **and** `¬orig.trap ⟹ member` independently, so the
//! value clause is only ever evaluated where every div/rem node on *both* sides
//! is inside its WASM-defined domain. The proof therefore never depends on the
//! two engines agreeing about SMT-LIB's `÷0` values.
//!
//! ## What makes the trap condition EXACT
//!
//! A trap condition that is merely an over-approximation would be unsound here,
//! not conservative: it appears *negated* in the value clause, so claiming a
//! trap the spec does not have would switch off the value comparison on real
//! states. The seam therefore refuses any obligation it cannot make exact —
//! see [`TrapContext`] for the two conditions the caller must establish, and
//! [`DeferReason::PartialOpTallyMismatch`] for the invariant that is *checked*
//! rather than assumed.
//!
//! ## Still refused after slice 2
//!
//! `trunc` (float→int), loads/stores, uninterpreted `pure_call` congruence,
//! havoc and memory arrays stay on the incumbent:
//!
//!   * **`trunc`** is not blocked by the solver — ordeal has an exact QF_BV
//!     `trap_trunc` — but by loom's own encoder, which erases
//!     `i32.trunc_f32_s` into a fresh `BV::new_const("i32_trunc_f32_s_result")`
//!     when the operand is not a literal. There is no trunc node left in the
//!     AST to reflect, so its trap condition cannot be recovered here at all.
//!     Making it routable is encoder work, not solver work.
//!   * **load/store** need the symbolic memory bound the current unbounded
//!     `Array(BV32 → BV8)` model does not carry (that is what #278 slipped
//!     through, and what [`crate::trap_gate::check_mem_transform`] currently
//!     supplies out-of-band). Their OOB trap condition belongs to the slice
//!     that introduces the bound.
//!
//! ## Relationship to the existing trap gates
//!
//! [`crate::trap_gate`] and [`crate::trap_backstop`] (#279/#288/#290) are live
//! safety gates on the runtime path and slice 2 neither removes, weakens nor
//! bypasses them. This module reuses their vocabulary
//! ([`crate::trap_gate::DivKind`]) and is checked *against* them
//! (`trap_gate_and_the_seam_agree_on_every_div_kind`), which is how a claim of
//! subsumption gets evidence instead of a deletion.
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
use crate::rule_solver::{RuleBool, RuleSolver, RuleTerm, RuleVerdict, VerifyBackend, Z3RuleSolver};

// Slice 2 shares the incumbent trap gate's vocabulary rather than inventing a
// parallel one: the same four kinds, so a divergence between this seam and
// `trap_gate` is a divergence about semantics, not about spelling.
#[cfg(feature = "verification")]
use crate::trap_gate::DivKind;

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
    /// One variable name occurred at two different widths, or two distinct Z3
    /// constants shared a name.
    ///
    /// ordeal interns bitvector variables by **name alone**
    /// (`Solver::var_word`), so such a term could conflate two distinct
    /// variables. Z3 keys constants by `(symbol, sort)` and would not. Rather
    /// than rely on this never happening, the seam refuses the obligation.
    VariableNameCollision,
    /// A partial (trapping) operation was reached without a [`TrapContext`]
    /// establishing that its trap condition can be stated exactly.
    ///
    /// This was slice 1's behaviour for *every* div/rem, and remains the
    /// default: with nothing known about **where** the op executes, the
    /// disjunction of per-node trap conditions is neither an upper nor a lower
    /// bound on the function's trap condition, so no sound obligation exists to
    /// build.
    PartialOpWithoutTrapContext,
    /// The partial ops found in the reflected term do not match the partial-op
    /// instructions the caller declared (the same multiset of `(kind, width)`).
    ///
    /// The trap clause is exact only if every trapping instruction contributes
    /// exactly one occurrence to the value term and nothing else does. That is
    /// an argument about the encoder; this reason is what turns it into a
    /// **checked invariant**. It fires if a future Z3 constant-folds
    /// `bvudiv(10, 0)` at construction (today it does not — the node survives),
    /// if the encoder starts sharing or duplicating a div node, or if an
    /// operand reaches the op at a width other than the instruction's — which
    /// would put the `INT_MIN`/`-1` constants of the `div_s` overflow disjunct
    /// at the wrong width.
    PartialOpTallyMismatch,
}

impl DeferReason {
    /// A short human-readable label (for diagnostics and test messages).
    pub fn label(&self) -> &'static str {
        match self {
            DeferReason::IncumbentBackend => "incumbent backend selected",
            DeferReason::WidthMismatch => "result width mismatch",
            DeferReason::OutOfFragment(what) => what,
            DeferReason::TooLarge => "term exceeds node budget",
            DeferReason::RoundTripFailed => "reflection round-trip not identical",
            DeferReason::VariableNameCollision => "variable name collision",
            DeferReason::PartialOpWithoutTrapContext => {
                "partial op with no established trap context"
            }
            DeferReason::PartialOpTallyMismatch => {
                "partial ops in the term do not match the instructions"
            }
        }
    }
}

// ============================================================================
// The trap dimension (slice 2)
// ============================================================================

/// The multiset of partial-op **instructions** on one side, keyed by
/// `(kind, width)`.
///
/// Built by the caller from the instruction list and compared against what the
/// reflection actually found in the Z3 term. See
/// [`DeferReason::PartialOpTallyMismatch`] for why this is checked rather than
/// argued.
#[cfg(feature = "verification")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartialOpTally {
    counts: std::collections::HashMap<(DivKind, u32), usize>,
}

#[cfg(feature = "verification")]
impl PartialOpTally {
    /// An empty tally (a side with no partial ops).
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one occurrence of `kind` at `width`.
    pub fn add(&mut self, kind: DivKind, width: u32) {
        *self.counts.entry((kind, width)).or_insert(0) += 1;
    }

    /// Whether this side has no partial ops at all.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

/// What the caller has established about **where** a function's partial
/// operations execute.
///
/// The seam reads a *value* expression. That expression says what a function
/// computes, not what it executes: a div under an `Ite` arm may be a WASM `if`
/// (executed conditionally) or a WASM `select` (both arms always executed), and
/// a div whose result was dropped or stored to a dead local does not appear in
/// it at all. None of that is recoverable from the AST, and each way of getting
/// it wrong is unsound in a different direction — an over-approximated trap
/// condition switches off the value clause on states that do not trap, an
/// under-approximated one lets a mandatory trap be deleted.
///
/// So the seam does not guess. The caller — which has the instruction list —
/// either establishes the property or the obligation stays on the incumbent.
#[cfg(feature = "verification")]
#[derive(Clone, Debug)]
pub enum TrapContext {
    /// Nothing established. Any partial op in the term is refused
    /// ([`DeferReason::PartialOpWithoutTrapContext`]) — this is exactly slice
    /// 1's behaviour and the default for every caller that has not done the
    /// analysis.
    Unestablished,
    /// **Every** partial op on both sides executes unconditionally, exactly
    /// once, and appears exactly once in the encoded value term.
    ///
    /// The caller establishes this by admitting only single-result bodies built
    /// from total, control-flow-free, value-preserving instructions plus
    /// div/rem — see `verify::trap_context_for`. Under that discipline:
    ///
    ///   * there is no control flow, so no instruction is skipped and no path
    ///     is dropped from the encoding;
    ///   * nothing discards a value (no `drop`, no `local.set`), and WASM's
    ///     validation requires the body to end with exactly the result on the
    ///     stack, so **every** computed value — every div — flows into the
    ///     encoded term;
    ///   * `select` is admitted deliberately: it is *eager*, both arms always
    ///     execute (WASM Core §4.4.1), which is precisely the unconditional
    ///     reading this context asserts. It is also the #281 shape.
    ///
    /// Under those conditions "the function traps" is exactly the disjunction
    /// of the per-op trap conditions, because in straight-line code an op
    /// traps-if-reached and every op is reached unless an earlier one trapped —
    /// which the same disjunction already covers.
    ///
    /// The tallies are the caller's instruction-level count per side; the seam
    /// refuses unless the reflection finds exactly them.
    Unconditional {
        /// Partial-op instructions in the original function.
        orig: PartialOpTally,
        /// Partial-op instructions in the optimized function.
        opt: PartialOpTally,
    },
}

#[cfg(feature = "verification")]
impl TrapContext {
    /// Whether partial ops may be reflected at all.
    fn admits_partial_ops(&self) -> bool {
        matches!(self, TrapContext::Unconditional { .. })
    }

    /// The declared tally for one side, if established.
    fn tally(&self, side: Side) -> Option<&PartialOpTally> {
        match (self, side) {
            (TrapContext::Unestablished, _) => None,
            (TrapContext::Unconditional { orig, .. }, Side::Orig) => Some(orig),
            (TrapContext::Unconditional { opt, .. }, Side::Opt) => Some(opt),
        }
    }
}

/// Which side of the obligation a reflection belongs to.
#[cfg(feature = "verification")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Orig,
    Opt,
}

/// The set of values the ORIGINAL may produce — the right-hand side of
/// `opt.value ∈ orig.valueSet`.
///
/// # Why a set at all, when slice 2 only ever builds singletons
///
/// WASM value nondeterminism is real but narrow: NaN payload selection,
/// `memory.grow` failure, relaxed-SIMD lane behaviour. None of it can occur in
/// the div/rem fragment this slice routes, so [`reflect`] only ever calls
/// [`ValueSet::singleton`] and the membership clause degenerates to equality.
///
/// Shipping a set type that is only ever built with one element would be
/// indistinguishable from equality plus dead code — a defect this repository has
/// found repeatedly (a gate with tests and no callers, a premise hook with no
/// producers, an unreachable match arm). The decision on #313 was that the
/// non-singleton path ships **with a test that fails if the path is removed**:
/// `synthetic_non_singleton_value_set_accepts_a_member_and_rejects_a_non_member`
/// builds a genuine two-member set, drives it through the *same*
/// [`Obligation::goal`] production uses, and requires one member accepted and a
/// non-member rejected. That test is honest about what it is: it exercises
/// machinery **the optimizer does not yet drive**, so that the slice which can
/// drive it (floats/NaN payloads — slice 6) inherits a proven shape rather than
/// a retrofit.
#[cfg(feature = "verification")]
#[derive(Clone, Debug)]
pub struct ValueSet {
    /// Non-empty by construction.
    members: Vec<RuleTerm>,
}

#[cfg(feature = "verification")]
impl ValueSet {
    /// The deterministic case: exactly one possible value.
    pub fn singleton(value: RuleTerm) -> Self {
        ValueSet {
            members: vec![value],
        }
    }

    /// A set of possible values. `None` if `members` is empty — an empty value
    /// set would make membership unsatisfiable and would silently turn the
    /// obligation into "reject everything".
    pub fn of(members: Vec<RuleTerm>) -> Option<Self> {
        if members.is_empty() {
            return None;
        }
        Some(ValueSet { members })
    }

    /// How many values the original may produce.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Always false — the set is non-empty by construction. Present so `len`
    /// does not stand alone.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// `value ∈ self`, as a neutral boolean: the disjunction of the equalities.
    fn contains(&self, value: &RuleTerm) -> RuleBool {
        let mut it = self.members.iter();
        let first = it
            .next()
            .expect("ValueSet is non-empty by construction")
            .eq_bool(value);
        it.fold(first, |acc, m| {
            RuleBool::BoolOr(Box::new(acc), Box::new(m.eq_bool(value)))
        })
    }
}

/// The slice-2 correctness obligation for one function pair, in neutral terms.
///
/// Assembled from round-tripped reflections and discharged by
/// [`prove_obligation`]. Keeping it a value (rather than inlining the formula
/// into the prover) is what lets the synthetic non-singleton test drive exactly
/// the production goal assembly.
#[cfg(feature = "verification")]
#[derive(Clone, Debug)]
pub struct Obligation {
    /// The original's trap condition. `None` means *statically* false — the
    /// side has no partial ops at all. It is not "unknown": the neutral DSL has
    /// no boolean constant to spell `false` with (slice 1 refused boolean
    /// leaves precisely because neither backend has one that round-trips), so
    /// the absent case is carried in the type and simplified away in
    /// [`Obligation::violation`].
    pub orig_trap: Option<RuleBool>,
    /// The optimized side's trap condition, same convention.
    pub opt_trap: Option<RuleBool>,
    /// The values the original may produce.
    pub orig_values: ValueSet,
    /// The value the optimized program produces.
    pub opt_value: RuleTerm,
}

/// Width of the 0/1 flag the assembled goal compares. Any width works; 32 keeps
/// the extra bit-blasting negligible next to the terms it guards.
#[cfg(feature = "verification")]
const FLAG_WIDTH: u32 = 32;

#[cfg(feature = "verification")]
impl Obligation {
    /// The `bad` predicate: the states in which the transform is WRONG.
    ///
    /// ```text
    /// bad = (T_o ≠ T_p) ∨ (¬T_o ∧ opt ∉ orig.valueSet)
    /// ```
    ///
    /// With `None` meaning "statically false", three of the four cases
    /// simplify — and the simplification is not cosmetic, it is what lets the
    /// formula be written at all without a boolean constant:
    ///
    /// ```text
    /// (None, None)    bad = ¬member
    /// (Some t, None)  bad = (t ≠ false) ∨ (¬t ∧ ¬member) = t ∨ ¬member
    /// (None, Some t)  bad = (false ≠ t) ∨ (true ∧ ¬member) = t ∨ ¬member
    /// (Some a,Some b) bad = (a ∧ ¬b) ∨ (¬a ∧ b) ∨ (¬a ∧ ¬member)
    /// ```
    ///
    /// The two middle cases collapsing to the same shape is a coincidence of
    /// the algebra, not a copy-paste: dropping a mandatory trap and inventing
    /// one are both "the trap conditions differ", and in each case the surviving
    /// value obligation is exactly the non-trapping one.
    fn violation(&self) -> RuleBool {
        let not_member = self.orig_values.contains(&self.opt_value).not();
        match (&self.orig_trap, &self.opt_trap) {
            (None, None) => not_member,
            (Some(t), None) | (None, Some(t)) => {
                RuleBool::BoolOr(Box::new(t.clone()), Box::new(not_member))
            }
            (Some(to), Some(tp)) => {
                let only_orig =
                    RuleBool::BoolAnd(Box::new(to.clone()), Box::new(tp.clone().not()));
                let only_opt = RuleBool::BoolAnd(Box::new(to.clone().not()), Box::new(tp.clone()));
                let value_wrong =
                    RuleBool::BoolAnd(Box::new(to.clone().not()), Box::new(not_member));
                RuleBool::BoolOr(
                    Box::new(RuleBool::BoolOr(Box::new(only_orig), Box::new(only_opt))),
                    Box::new(value_wrong),
                )
            }
        }
    }

    /// Whether the relation degenerates to the plain equality slice 1 emits:
    /// no traps on either side and a single possible value.
    fn is_pure_equality(&self) -> bool {
        self.orig_trap.is_none() && self.opt_trap.is_none() && self.orig_values.len() == 1
    }

    /// The `(lhs, rhs)` pair to hand [`RuleSolver::prove_rule_equiv`], whose
    /// `Proven` means `lhs == rhs` for all inputs.
    ///
    /// For the degenerate case this is *literally* the slice-1 query — same two
    /// terms, byte for byte — so an obligation with no partial ops is decided by
    /// exactly the formula it was decided by before this slice.
    ///
    /// Otherwise the relation is encoded as `ite(bad, 1, 0) == 0`, which is
    /// `Unsat(bad)`: the prover asserts `ite(bad,1,0) ≠ 0`, satisfiable exactly
    /// when `bad` is. Encoding it this way reuses the [`RuleSolver`] trait
    /// unchanged, so both backends receive the identical neutral goal and
    /// `both` mode remains a real differential.
    pub fn goal(&self) -> (RuleTerm, RuleTerm) {
        if self.is_pure_equality() {
            return (self.orig_values.members[0].clone(), self.opt_value.clone());
        }
        let zero = RuleTerm::from_u64(0, FLAG_WIDTH);
        let one = RuleTerm::from_u64(1, FLAG_WIDTH);
        (self.violation().ite(&one, &zero), zero)
    }
}

/// Discharge a slice-2 [`Obligation`] on the selected backend.
///
/// `Proven` means the relation holds for every input: traps preserved in both
/// directions, and — where the original does not trap — a value the original
/// could have produced.
#[cfg(feature = "verification")]
pub fn prove_obligation(obligation: &Obligation, backend: VerifyBackend) -> RuleVerdict {
    let (lhs, rhs) = obligation.goal();
    prove(&lhs, &rhs, backend)
}

/// The **exact** trap condition of one WASM div/rem, as a neutral boolean.
///
/// Checked against the spec, op by op (WASM Core §4.4.1 `idiv_u`, `idiv_s`,
/// `irem_u`, `irem_s`), not against intuition:
///
/// | op      | `÷0` | `INT_MIN / -1` |
/// |---------|------|----------------|
/// | `div_u` | trap | n/a (unsigned) |
/// | `div_s` | trap | **trap** — the true quotient `2^(N-1)` is not representable |
/// | `rem_u` | trap | n/a (unsigned) |
/// | `rem_s` | trap | **defined**, result `0` |
///
/// The `rem_s` row is the one that is easy to get wrong by symmetry with
/// `div_s`, and getting it wrong is not conservative — it claims a trap the
/// spec does not have, which rejects correct folds *and*, because the condition
/// is negated in the value clause, switches off the value comparison on states
/// that really do execute. It was ordeal#84/#72 and loom#288; here it is pinned
/// three ways: against ordeal's independently-written `trap::trap_div`
/// (`trap_gate_and_the_seam_agree_on_every_div_kind`), against a real WASM
/// engine (`wasm_engine_agrees_with_the_encoded_trap_conditions`), and
/// end-to-end through the validator (`slice2_rem_s_int_min_fold_is_accepted`).
#[cfg(feature = "verification")]
fn trap_condition(
    kind: DivKind,
    dividend: &RuleTerm,
    divisor: &RuleTerm,
    width: u32,
) -> RuleBool {
    let zero = RuleTerm::from_u64(0, width);
    let div_by_zero = divisor.eq_bool(&zero);
    if kind != DivKind::DivS {
        return div_by_zero;
    }
    let int_min = RuleTerm::Const {
        value: 1u128 << (width - 1),
        width,
    };
    let all_ones = RuleTerm::from_i64(-1, width);
    let overflow = RuleBool::BoolAnd(
        Box::new(dividend.eq_bool(&int_min)),
        Box::new(divisor.eq_bool(&all_ones)),
    );
    RuleBool::BoolOr(Box::new(div_by_zero), Box::new(overflow))
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

/// Offer the equivalence obligation `orig == opt` to the neutral seam, with no
/// trap context — so any partial (trapping) operation is refused.
///
/// This is the slice-1 entry point, kept exactly as it was. A caller that has
/// not established *where* a function's trapping operations execute has no
/// business deciding an obligation that contains one; see
/// [`decide_equivalence`] for the general form and [`TrapContext`] for what has
/// to be established.
#[cfg(feature = "verification")]
pub fn decide_bv_equivalence(
    orig: &z3::ast::BV,
    opt: &z3::ast::BV,
    backend: VerifyBackend,
) -> SeamOutcome {
    decide_equivalence(orig, opt, backend, &TrapContext::Unestablished)
}

/// Offer the correctness obligation for `orig` vs `opt` to the neutral seam.
///
/// Returns [`SeamOutcome::Decided`] only when both sides were reflected into
/// the closed fragment, each reflection re-lowered to the identical Z3 AST, the
/// partial ops found matched the ones `ctx` declared, and the selected backend
/// reached a verdict. In every other case the caller must fall through to the
/// incumbent path unchanged.
///
/// What is proven is the slice-2 relation (see the module docs): trap
/// **equivalence** in both directions, plus value membership wherever the
/// original does not trap. With no partial ops on either side that is literally
/// the slice-1 equality.
///
/// `backend` is an explicit parameter rather than an env read so callers *and
/// tests* can drive a specific engine deterministically; `verify.rs` passes
/// [`VerifyBackend::from_env`].
#[cfg(feature = "verification")]
pub fn decide_equivalence(
    orig: &z3::ast::BV,
    opt: &z3::ast::BV,
    backend: VerifyBackend,
    ctx: &TrapContext,
) -> SeamOutcome {
    let outcome = decide_inner(orig, opt, backend, ctx);
    match &outcome {
        SeamOutcome::Decided(_) => note_routed(),
        SeamOutcome::Deferred(reason) => note_deferred(*reason),
    }
    outcome
}

#[cfg(feature = "verification")]
fn decide_inner(
    orig: &z3::ast::BV,
    opt: &z3::ast::BV,
    backend: VerifyBackend,
    ctx: &TrapContext,
) -> SeamOutcome {
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
    let mut r = reflect::Reflector::new(MAX_TERM_NODES, ctx.admits_partial_ops());
    let lhs = match r.reflect_bv(orig) {
        Ok(t) => t,
        Err(reason) => return SeamOutcome::Deferred(reason),
    };
    // Taken per side: the trap clause of the ORIGINAL and of the OPTIMIZED are
    // different formulas and the whole relation is about them differing.
    let lhs_ops = r.take_partial_ops();
    let rhs = match r.reflect_bv(opt) {
        Ok(t) => t,
        Err(reason) => return SeamOutcome::Deferred(reason),
    };
    let rhs_ops = r.take_partial_ops();

    // The self-validation gate: the neutral terms must re-lower to the
    // IDENTICAL Z3 AST nodes we were handed. Anything else and we do not
    // believe the reflection.
    //
    // This also validates the trap clause, without a second gate: lowering is
    // structural and Z3's AST is hash-consed, so if the whole term re-lowers to
    // the identical node then every sub-term does too — including the dividend
    // and divisor operands the trap conditions are built from.
    if !reflect::round_trips(&lhs, orig) || !reflect::round_trips(&rhs, opt) {
        return SeamOutcome::Deferred(DeferReason::RoundTripFailed);
    }

    // The partial ops found must be exactly the ones the caller declared —
    // same kinds, same widths, same counts. See
    // `DeferReason::PartialOpTallyMismatch`.
    let orig_trap = match trap_clause(&lhs_ops, ctx, Side::Orig) {
        Ok(t) => t,
        Err(reason) => return SeamOutcome::Deferred(reason),
    };
    let opt_trap = match trap_clause(&rhs_ops, ctx, Side::Opt) {
        Ok(t) => t,
        Err(reason) => return SeamOutcome::Deferred(reason),
    };

    SeamOutcome::Decided(prove_obligation(
        &Obligation {
            orig_trap,
            opt_trap,
            // Deterministic: the div/rem fragment has no value nondeterminism,
            // so the original's value set is a singleton. See `ValueSet`.
            orig_values: ValueSet::singleton(lhs),
            opt_value: rhs,
        },
        backend,
    ))
}

/// Build one side's trap clause from the partial ops the reflection found,
/// after checking them against the tally the caller declared.
///
/// `Ok(None)` means the side has no partial ops, i.e. a *statically false* trap
/// condition.
#[cfg(feature = "verification")]
fn trap_clause(
    found: &[reflect::PartialOp],
    ctx: &TrapContext,
    side: Side,
) -> Result<Option<RuleBool>, DeferReason> {
    if found.is_empty() {
        // Nothing found. If the caller declared partial ops for this side they
        // are missing from the term (the encoder folded or dropped one), which
        // would make the trap clause an under-approximation — refuse.
        if ctx.tally(side).is_some_and(|t| !t.is_empty()) {
            return Err(DeferReason::PartialOpTallyMismatch);
        }
        return Ok(None);
    }
    let mut seen = PartialOpTally::new();
    for op in found {
        seen.add(op.kind, op.width);
    }
    match ctx.tally(side) {
        Some(declared) if *declared == seen => {}
        // Unreachable while `admits_partial_ops` gates the reflection (no
        // context ⟹ no occurrences ⟹ the early return above), but a total
        // arm means a future caller cannot open a hole by accident.
        _ => return Err(DeferReason::PartialOpTallyMismatch),
    }
    let mut clause: Option<RuleBool> = None;
    for op in found {
        let c = trap_condition(op.kind, &op.dividend, &op.divisor, op.width);
        clause = Some(match clause {
            None => c,
            Some(prev) => RuleBool::BoolOr(Box::new(prev), Box::new(c)),
        });
    }
    Ok(clause)
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
    use super::{DeferReason, DivKind};
    use crate::rule_solver::{RuleBool, RuleTerm};
    use std::collections::HashMap;
    use z3::ast::{Ast, BV, Bool, Dynamic};
    use z3::{AstKind, DeclKind};

    type Reflected<T> = Result<T, DeferReason>;

    /// One PARTIAL-operation occurrence found while reflecting a side.
    ///
    /// The operands are the reflected sub-terms, so the trap condition built
    /// from them is stated over exactly the terms the value clause uses — and
    /// they inherit the round-trip guarantee of the whole side (lowering is
    /// structural over a hash-consed AST, so an identical whole implies
    /// identical parts).
    pub(super) struct PartialOp {
        pub(super) kind: DivKind,
        pub(super) width: u32,
        pub(super) dividend: RuleTerm,
        pub(super) divisor: RuleTerm,
    }

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

    /// Walks a Z3 AST and rebuilds it in the closed fragment.
    pub(super) struct Reflector {
        /// name → (width, the Z3 constant it came from).
        vars: HashMap<String, (u32, BV)>,
        budget: usize,
        /// Whether the caller established a [`super::TrapContext`] that makes a
        /// partial op's trap condition exactly stateable.
        admit_partial_ops: bool,
        /// Partial ops found since the last [`Reflector::take_partial_ops`].
        partial_ops: Vec<PartialOp>,
    }

    impl Reflector {
        pub(super) fn new(max_nodes: usize, admit_partial_ops: bool) -> Self {
            Reflector {
                vars: HashMap::new(),
                budget: max_nodes,
                admit_partial_ops,
                partial_ops: Vec::new(),
            }
        }

        /// Take the partial ops found so far, leaving the reflector ready for
        /// the next side. The *variable* table is deliberately NOT reset — it
        /// has to span both sides (see `decide_inner`).
        pub(super) fn take_partial_ops(&mut self) -> Vec<PartialOp> {
            std::mem::take(&mut self.partial_ops)
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
                // --- PARTIAL operations (slice 2): admitted only with a trap
                // context, and always paired with an exact trap condition.
                DeclKind::BUDIV => self.partial(node, n, DivKind::DivU, width, RuleTerm::Udiv),
                DeclKind::BUREM => self.partial(node, n, DivKind::RemU, width, RuleTerm::Urem),
                DeclKind::BSDIV => self.partial(node, n, DivKind::DivS, width, RuleTerm::Sdiv),
                DeclKind::BSREM => self.partial(node, n, DivKind::RemS, width, RuleTerm::Srem),

                // `bvsmod` (floored modulo) is not a WASM operation and the
                // encoder never builds one; there is no `iN` instruction whose
                // trap condition it would carry, so it stays refused.
                DeclKind::BSMOD
                // The `_I` / `0` families are Z3's INTERNAL div/rem forms —
                // the interpreted variants and the uninterpreted
                // divide-by-zero functions its rewriter introduces. They do
                // not denote the SMT-LIB operation the neutral DSL lowers to,
                // so reflecting one as `Udiv`/`Sdiv`/... would state a
                // different query. (The round-trip gate would also catch it;
                // refusing by name keeps the diagnostic honest.)
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
                    "internal or non-WASM division form",
                )),

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

        /// Reflect a PARTIAL binary op, recording the occurrence so its exact
        /// trap condition can be added to the obligation.
        ///
        /// Refused outright without a trap context: a partial op whose trap
        /// condition cannot be stated exactly has no sound obligation at all
        /// (see [`DeferReason::PartialOpWithoutTrapContext`]).
        fn partial(
            &mut self,
            node: &BV,
            n: usize,
            kind: DivKind,
            width: u32,
            build: fn(Box<RuleTerm>, Box<RuleTerm>) -> RuleTerm,
        ) -> Reflected<RuleTerm> {
            if !self.admit_partial_ops {
                return Err(DeferReason::PartialOpWithoutTrapContext);
            }
            if n != 2 {
                return Err(DeferReason::OutOfFragment("n-ary division"));
            }
            let dividend = self.reflect_bv(&child_bv(node, 0)?)?;
            let divisor = self.reflect_bv(&child_bv(node, 1)?)?;
            // Z3 sorts both operands and the result identically for div/rem, so
            // this is a re-assertion rather than a filter — but the trap
            // constants are built at `width`, and a silent width difference is
            // exactly the failure the tally check exists to catch.
            if dividend.width() != width || divisor.width() != width {
                return Err(DeferReason::PartialOpTallyMismatch);
            }
            self.partial_ops.push(PartialOp {
                kind,
                width,
                dividend: dividend.clone(),
                divisor: divisor.clone(),
            });
            Ok(build(Box::new(dividend), Box::new(divisor)))
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

    /// Reflect `bv` (no trap context) and require the round-trip to be the
    /// identical AST.
    fn reflect_ok(bv: &BV) -> RuleTerm {
        reflect_ok_with(bv, false)
    }

    /// Reflect `bv`, optionally admitting partial ops, and require the
    /// round-trip to be the identical AST.
    fn reflect_ok_with(bv: &BV, admit_partial: bool) -> RuleTerm {
        let mut r = reflect::Reflector::new(MAX_TERM_NODES, admit_partial);
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
        reflect_err_with(bv, false)
    }

    fn reflect_err_with(bv: &BV, admit_partial: bool) -> DeferReason {
        let mut r = reflect::Reflector::new(MAX_TERM_NODES, admit_partial);
        match r.reflect_bv(bv) {
            Ok(t) => panic!("expected {} to be refused, but it reflected to {:?}", bv, t),
            Err(e) => e,
        }
    }

    #[test]
    fn every_slice1_op_reflects_and_round_trips_identically() {
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

            // Trapping / partial ops. Slice 2 admits these, but ONLY with a
            // trap context; `reflect_err` supplies none, so the refusal is
            // still absolute here — only its attributed reason moved from "not
            // in the fragment" to "no context in which its trap condition can
            // be stated exactly".
            for op in [a.bvudiv(&b), a.bvurem(&b), a.bvsdiv(&b), a.bvsrem(&b)] {
                assert_eq!(
                    reflect_err(&op),
                    DeferReason::PartialOpWithoutTrapContext,
                    "partial op must be refused without a trap context: {}",
                    op
                );
            }

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
            let mut r = reflect::Reflector::new(MAX_TERM_NODES, false);
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
            // Deferred: trapping op.
            let _ = decide_bv_equivalence(&a.bvudiv(&b), &a.bvudiv(&b), VerifyBackend::Ordeal);
            assert_eq!(route_counts(), (1, 1));
        });
    }
}
