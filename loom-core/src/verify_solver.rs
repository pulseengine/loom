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
use crate::rule_solver::{RuleSolver, RuleTerm, RuleVerdict, VerifyBackend, Z3RuleSolver};

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

    SeamOutcome::Decided(prove(&lhs, &rhs, backend))
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
                // --- trapping / partial operations: slice 2, refused here ---
                DeclKind::BUDIV
                | DeclKind::BUREM
                | DeclKind::BSDIV
                | DeclKind::BSREM
                | DeclKind::BSMOD
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
                    "division/remainder (partial op — slice 2)",
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

            // Trapping / partial ops — deliberately deferred to slice 2.
            for op in [a.bvudiv(&b), a.bvurem(&b), a.bvsdiv(&b), a.bvsrem(&b)] {
                assert!(
                    matches!(reflect_err(&op), DeferReason::OutOfFragment(_)),
                    "partial op must be refused: {}",
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
            // Deferred: trapping op.
            let _ = decide_bv_equivalence(&a.bvudiv(&b), &a.bvudiv(&b), VerifyBackend::Ordeal);
            assert_eq!(route_counts(), (1, 1));
        });
    }
}
