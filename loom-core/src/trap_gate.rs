//! Systemic **trap-equivalence** gate (issue #279).
//!
//! # Why this exists
//!
//! loom's Z3 translation validator ([`crate::verify`]) proves *value*
//! equivalence over a model in which every WASM operation is **total** — traps
//! are unmodeled (SMT-LIB QF_BV `bvsdiv`/`bvudiv` are total, division by zero
//! just yields all-ones; a symbolic `Array(BV32 → BV8)` load never faults).
//! That is exactly why loom has shipped five trap-elimination miscompiles —
//! #273, #274, #276, #278, #281 — each patched by an ad-hoc `is_no_trap` static
//! guard on one pass. Deleting a trapping op looks *value-equal* to a total
//! model, so the value-only verifier waved it through.
//!
//! This module is the durable cure: a **trap-equivalence** obligation decided by
//! the [`ordeal`] certificate-checked QF_BV pipeline. An optimization over a
//! partial op (div/rem, load/store, `select` with a trapping discarded arm) is
//! accepted **only** when trap-and-value equivalence is proven `Unsat` — i.e.
//! the transform preserves traps AND values. This is the general backstop that
//! supersedes the per-pass `is_no_trap` stopgaps (they remain sound as a cheap
//! fast-path, but this gate is the authority).
//!
//! # The boundary (loom supplies values, ordeal supplies traps)
//!
//! ordeal's closed fragment has `Udiv` only (no `Sdiv`/`Srem`/`Urem`), so the
//! VC's **value** clause is supplied here, by loom — which already models signed
//! div/rem values in its Z3 encoding (cf. #163). ordeal contributes the
//! trap-condition predicates ([`ordeal::trap::trap_div`],
//! [`ordeal::trap::trap_mem_oob`], …) and the VC assembly; loom supplies value +
//! trap terms; they meet in [`ordeal::trap::DefineOrTrap`]. Modeling signed
//! div/rem *values* natively inside ordeal is a separate, later decision — not
//! needed here, because [`signed_div_value`] / [`signed_rem_value`] build the
//! value `BvTerm` from the primitive fragment (`Ite`/`Slt`/`Sub`/`Udiv`).
//!
//! # Backend
//!
//! This gate runs on the **ordeal** backend (its own AIG → CNF → CDCL → LRAT
//! pipeline), NOT Z3. That is deliberate: an ordeal `Unsat` carries an
//! independently re-checkable LRAT certificate ([`ordeal::Certificate::recheck`])
//! — the trap gate accepts a fold only on a certificate that re-validates. Value
//! equivalence for the non-partial bulk of a function stays on the existing Z3
//! validator; this gate adds the trap dimension the Z3 model cannot express.
//!
//! # Decision rule (never `!Sat`)
//!
//! Accept a transform **only** on `Unsat`. `Sat` → reject with a counterexample
//! input. `Unknown` → reject conservatively (ordeal returns `Unknown` for any
//! query it will not stand behind). This matches ordeal's soundness contract:
//! only a checked `Unsat` certificate authorizes a transformation.

#![cfg(feature = "verification")]

use ordeal::term::{BoolTerm, BvTerm, Sort};
use ordeal::trap::{
    self, DefineOrTrap, DivOp, prove_trap_condition_equivalence, prove_trap_equivalence,
};
use ordeal::{CheckResult, Solver};

fn bx(t: BvTerm) -> Box<BvTerm> {
    Box::new(t)
}
fn bb(t: BoolTerm) -> Box<BoolTerm> {
    Box::new(t)
}

/// The verdict of a trap-equivalence query, reduced to loom's accept/reject
/// decision. Only [`Accept`](TrapVerdict::Accept) authorizes the transform.
#[derive(Debug)]
pub enum TrapVerdict {
    /// Trap-and-value equivalence proven; the LRAT certificate re-checks. Safe
    /// to apply the fold.
    Accept,
    /// A concrete input distinguishes the two sides (a dropped/added trap or a
    /// value mismatch on a non-trapping input). Reject; carries the offending
    /// variable assignment for diagnostics.
    RejectCounterexample(Vec<(String, u128)>),
    /// The solver would not decide (or the certificate failed to re-check).
    /// Reject conservatively — never accept on `Unknown`.
    RejectUnknown,
}

impl TrapVerdict {
    /// True only for [`Accept`](TrapVerdict::Accept). The single predicate a
    /// pass consults to decide whether the fold is trap-safe.
    pub fn is_accepted(&self) -> bool {
        matches!(self, TrapVerdict::Accept)
    }
}

/// Reduce an ordeal [`CheckResult`] to a [`TrapVerdict`], re-checking the LRAT
/// certificate on `Unsat`. A certificate that does not re-check degrades to
/// [`RejectUnknown`](TrapVerdict::RejectUnknown) — no unchecked accept, ever.
fn verdict_of(result: CheckResult) -> TrapVerdict {
    match result {
        CheckResult::Unsat(cert) => match cert.recheck() {
            Ok(()) => TrapVerdict::Accept,
            Err(_) => TrapVerdict::RejectUnknown,
        },
        CheckResult::Sat(model) => TrapVerdict::RejectCounterexample(model.assignments),
        CheckResult::Unknown => TrapVerdict::RejectUnknown,
    }
}

/// A free bitvector variable of the given width.
pub fn var(name: &str, width: u32) -> BvTerm {
    BvTerm::Var {
        name: name.into(),
        sort: Sort::new(width),
    }
}

/// A bitvector constant of the given width.
pub fn constant(value: u128, width: u32) -> BvTerm {
    BvTerm::Const {
        value,
        sort: Sort::new(width),
    }
}

/// Signed division **value** `BvTerm` matching WASM `iN.div_s` on non-trapping
/// inputs, built from ordeal's closed fragment (it has `Udiv` only).
///
/// `div_s(a, b) = sign(a) == sign(b) ? |a|/|b| : -(|a|/|b|)`, where `|x|` is the
/// two's-complement absolute value. The trap cases (`b == 0`, `INT_MIN / -1`)
/// are handled by the *trap clause* ([`ordeal::trap::trap_div`] with
/// [`DivOp::DivS`]); this term only needs to agree with WASM on the
/// non-trapping domain, which the VC's `¬may_trap ⇒ value_eq` guard enforces.
pub fn signed_div_value(a: &BvTerm, b: &BvTerm, width: u32) -> BvTerm {
    let abs = |x: &BvTerm| -> BvTerm {
        let neg = BvTerm::Sub(bx(constant(0, width)), bx(x.clone()));
        BvTerm::Ite {
            cond: bb(BoolTerm::Slt(bx(x.clone()), bx(constant(0, width)))),
            then_: bx(neg),
            else_: bx(x.clone()),
        }
    };
    let mag = BvTerm::Udiv(bx(abs(a)), bx(abs(b)));
    let neg_mag = BvTerm::Sub(bx(constant(0, width)), bx(mag.clone()));
    // Result is negative iff exactly one operand is negative.
    let a_neg = BoolTerm::Slt(bx(a.clone()), bx(constant(0, width)));
    let b_neg = BoolTerm::Slt(bx(b.clone()), bx(constant(0, width)));
    let diff_sign = BoolTerm::And(
        bb(BoolTerm::Or(bb(a_neg.clone()), bb(b_neg.clone()))),
        bb(BoolTerm::Not(bb(BoolTerm::And(bb(a_neg), bb(b_neg))))),
    );
    BvTerm::Ite {
        cond: bb(diff_sign),
        then_: bx(neg_mag),
        else_: bx(mag),
    }
}

/// Signed remainder **value** `BvTerm` matching WASM `iN.rem_s` on non-trapping
/// inputs. WASM defines `rem_s(a, b) = a - (a div_s b) * b` with the *truncated*
/// quotient, which means the remainder takes the sign of the dividend `a`.
/// Built as `sign(a) ? -(|a| rem |b|) : (|a| rem |b|)`.
pub fn signed_rem_value(a: &BvTerm, b: &BvTerm, width: u32) -> BvTerm {
    let abs = |x: &BvTerm| -> BvTerm {
        let neg = BvTerm::Sub(bx(constant(0, width)), bx(x.clone()));
        BvTerm::Ite {
            cond: bb(BoolTerm::Slt(bx(x.clone()), bx(constant(0, width)))),
            then_: bx(neg),
            else_: bx(x.clone()),
        }
    };
    // Unsigned remainder |a| % |b| = |a| - (|a| / |b|) * |b|; ordeal has no
    // Urem, so derive it from Udiv/Mul/Sub.
    let abs_a = abs(a);
    let abs_b = abs(b);
    let q = BvTerm::Udiv(bx(abs_a.clone()), bx(abs_b.clone()));
    let prod = BvTerm::Mul(bx(q), bx(abs_b));
    let umag = BvTerm::Sub(bx(abs_a), bx(prod));
    let neg_umag = BvTerm::Sub(bx(constant(0, width)), bx(umag.clone()));
    let a_neg = BoolTerm::Slt(bx(a.clone()), bx(constant(0, width)));
    BvTerm::Ite {
        cond: bb(a_neg),
        then_: bx(neg_umag),
        else_: bx(umag),
    }
}

/// The signedness/kind of a div/rem op, mapping loom instructions onto ordeal's
/// [`DivOp`] plus the correct **value** builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DivKind {
    /// `iN.div_u`.
    DivU,
    /// `iN.div_s`.
    DivS,
    /// `iN.rem_u`.
    RemU,
    /// `iN.rem_s`.
    RemS,
}

impl DivKind {
    fn ordeal_op(self) -> DivOp {
        match self {
            DivKind::DivU => DivOp::DivU,
            DivKind::DivS => DivOp::DivS,
            DivKind::RemU => DivOp::RemU,
            DivKind::RemS => DivOp::RemS,
        }
    }

    /// The WASM value `BvTerm` for this op on operands `a`, `b`.
    fn value(self, a: &BvTerm, b: &BvTerm, width: u32) -> BvTerm {
        match self {
            DivKind::DivU => BvTerm::Udiv(bx(a.clone()), bx(b.clone())),
            DivKind::DivS => signed_div_value(a, b, width),
            DivKind::RemU => {
                // |a| % |b| with unsigned operands = a - (a/b)*b.
                let q = BvTerm::Udiv(bx(a.clone()), bx(b.clone()));
                let prod = BvTerm::Mul(bx(q), bx(b.clone()));
                BvTerm::Sub(bx(a.clone()), bx(prod))
            }
            DivKind::RemS => signed_rem_value(a, b, width),
        }
    }
}

/// Build the [`DefineOrTrap`] for a div/rem op: loom's value term + ordeal's
/// trap clause (`÷0` for all four; `INT_MIN / -1` overflow for the signed ops).
pub fn div_define(kind: DivKind, a: &BvTerm, b: &BvTerm, width: u32) -> DefineOrTrap {
    DefineOrTrap {
        value: kind.value(a, b, width),
        may_trap: trap::trap_div(kind.ordeal_op(), a, b, width),
    }
}

/// **Trap-and-value equivalence** gate for a div/rem fold (issue #279 —
/// #273/#274/#276 class). Proves `(orig.trap ⇔ opt.trap) ∧ (¬orig.trap ⇒
/// value_eq)` via ordeal and returns the accept/reject verdict.
///
/// A fold that drops the `÷0` or `INT_MIN/-1` trap (e.g. constant-folding
/// `div_s(INT_MIN, -1)`, or dead-eliminating a trapping div) is caught as `Sat`.
/// A value-preserving, trap-preserving lowering proves `Unsat` and is accepted.
pub fn check_div_transform(
    orig_kind: DivKind,
    opt_kind: DivKind,
    a: &BvTerm,
    b: &BvTerm,
    width: u32,
) -> TrapVerdict {
    let orig = div_define(orig_kind, a, b, width);
    let opt = div_define(opt_kind, a, b, width);
    verdict_of(prove_trap_equivalence(&orig, &opt))
}

/// **Trap-condition equivalence** gate (conjunct-1 only) for a memory-op fold
/// (issue #279 — #278 OOB-load / zero-annihilation class). loom has no
/// memory-*contents* value model, so this proves only that the OOB trap was
/// neither dropped nor spuriously added: `orig.may_trap ⇔ opt.may_trap`.
///
/// `mem_bound` is loom's symbolic linear-memory extent (page count × 64 KiB, or
/// the `memory.size` term × 65536) in the same width as `addr`/`size` — the
/// bound the current unbounded `Array(BV32 → BV8)` model omits (the reason #278
/// slipped through). A fold that discards a load whose address may be OOB
/// (`mul(load(oob), 0) → 0`) is caught as `Sat`.
pub fn check_mem_transform(
    orig_addr: &BvTerm,
    orig_size: &BvTerm,
    opt_traps: bool,
    mem_bound: &BvTerm,
) -> TrapVerdict {
    let orig_trap = trap::trap_mem_oob(orig_addr, orig_size, mem_bound);
    // The optimized side either preserves the SAME OOB trap or drops it
    // entirely (the discard folds we are guarding). `opt_traps == true` means
    // the optimized program still performs the (identical) access.
    let opt_trap = if opt_traps {
        trap::trap_mem_oob(orig_addr, orig_size, mem_bound)
    } else {
        // Trivially-false trap condition (`0 != 0`): the access was removed.
        BoolTerm::Ne(bx(constant(0, 8)), bx(constant(0, 8)))
    };
    verdict_of(prove_trap_condition_equivalence(&orig_trap, &opt_trap))
}

/// The symbolic linear-memory byte bound for [`check_mem_transform`], from a
/// `memory.size` term (in 64 KiB pages) of `width` bits: `pages × 65536`.
///
/// Threads loom's `memory.size` symbolic value into the OOB trap predicate so
/// the bound is the same on both sides of a transform (the value cancels in the
/// `⇔`). WASM's page size is fixed at 64 KiB.
pub fn mem_bound_from_pages(pages: &BvTerm, width: u32) -> BvTerm {
    BvTerm::Mul(bx(pages.clone()), bx(constant(65536, width)))
}

/// **Div-is-trap-free** gate for a div/rem → **constant** fold (issue #279 —
/// #273 class). Constant-folding `div_s/div_u/rem_s/rem_u` replaces the op with a
/// literal that carries NO trap, so the fold is sound **only** when the op
/// provably CANNOT trap for these operands. This proves exactly that obligation:
/// `orig.may_trap ⇔ false` — the `÷0` disjunct (all four kinds) and the
/// `INT_MIN / -1` overflow disjunct (signed kinds) are both unsatisfiable.
///
/// `a`/`b` are the operand `BvTerm`s at the fold site. At a const-fold site they
/// are [`constant`] terms, so ordeal decides the trap condition exactly (no
/// symbolic slack). A folded `div_s(INT_MIN, -1)` (which the #273 static guard
/// already refuses) is caught here as `Sat` (the overflow disjunct holds); a
/// fold of `20 / 4` proves `Unsat` and is accepted.
pub fn check_div_discard(kind: DivKind, a: &BvTerm, b: &BvTerm, width: u32) -> TrapVerdict {
    let orig_trap = trap::trap_div(kind.ordeal_op(), a, b, width);
    let no_trap = BoolTerm::Ne(bx(constant(0, 8)), bx(constant(0, 8))); // false
    verdict_of(prove_trap_condition_equivalence(&orig_trap, &no_trap))
}

/// **Select-with-trapping-arm** gate (issue #279 — #281 class). WASM `select` is
/// *eager*: both value arms are evaluated before the condition picks one, so
/// folding `select(const_cond, a, b)` to the taken arm is sound **only** if the
/// discarded arm cannot trap. This proves the discarded arm's trap condition is
/// preserved: dropping it must not drop a trap.
///
/// `discarded_arm_trap` is the trap condition of the arm the fold would delete
/// (e.g. an OOB `load` or a `÷0` div). The optimized side no longer evaluates it
/// (`false`). If the arm may trap, `orig.trap ⇔ false` is `Sat` (rejected); a
/// provably trap-free arm (`discarded_arm_trap` ≡ `false`) proves `Unsat`.
pub fn check_select_discard(discarded_arm_trap: &BoolTerm) -> TrapVerdict {
    let opt_trap = BoolTerm::Ne(bx(constant(0, 8)), bx(constant(0, 8))); // false
    verdict_of(prove_trap_condition_equivalence(
        discarded_arm_trap,
        &opt_trap,
    ))
}

// ---------------------------------------------------------------------------
// Reusable "prove a transform under an asserted hypothesis" machinery.
//
// #231 (proof-carrying facts) will reuse THIS helper to prove fact-driven
// rewrites under an assumed fact: e.g. a range premise `0 <= x < N` lets a
// bounds check fold away, but the rewrite is only sound *given* that premise.
// The pattern is identical to the trap gate — assert premises, then prove the
// obligation UNSAT-under-negation — so it is factored out here as the single
// certificate-checked entry point both features share.
// ---------------------------------------------------------------------------

/// Prove an `obligation` (a `BoolTerm` goal we want **valid**) under a set of
/// asserted `premises`, decided by ordeal's certificate-checked pipeline.
///
/// Semantics: prove `(⋀ premises) ⇒ obligation` is valid for every input, by
/// asserting the premises together with `¬obligation` and checking for
/// unsatisfiability. `Unsat(cert)` (re-checked) ⟹ the obligation holds under the
/// premises ⟹ [`Accept`](TrapVerdict::Accept). `Sat` ⟹ a counterexample
/// satisfying the premises but violating the obligation. `Unknown` ⟹
/// conservative reject.
///
/// With `premises` empty this is exactly "prove `obligation` valid" — the
/// unconditional case. With premises it is "prove the transform under the
/// asserted hypothesis", the machinery #231 reuses for fact-driven rewrites.
pub fn prove_under_premises(premises: &[BoolTerm], obligation: &BoolTerm) -> TrapVerdict {
    let mut s = Solver::new();
    for p in premises {
        s.assert(p.clone());
    }
    s.assert(BoolTerm::Not(bb(obligation.clone())));
    verdict_of(s.check())
}

/// Trap-equivalence VC evaluated **under premises** — the composition of
/// [`ordeal::trap::trap_equivalence_vc`] with [`prove_under_premises`]. This is
/// the shape #231 needs: prove that a fold over a partial op preserves traps and
/// values *given* an assumed fact (e.g. "the divisor is known non-zero here").
///
/// With no premises this is equivalent to [`check_div_transform`]'s core check;
/// premises let a caller discharge the trap under a proven fact (a known-nonzero
/// divisor makes the `÷0` disjunct unreachable, so both sides may legitimately
/// never trap).
pub fn check_trap_equivalence_under(
    premises: &[BoolTerm],
    orig: &DefineOrTrap,
    opt: &DefineOrTrap,
) -> TrapVerdict {
    let obligation = trap::trap_equivalence_vc(orig, opt);
    prove_under_premises(premises, &obligation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(name: &str, w: u32) -> BvTerm {
        var(name, w)
    }
    fn c(value: u128, w: u32) -> BvTerm {
        constant(value, w)
    }

    // ---- div/rem: value model agrees with ordeal's own Udiv reference ----

    /// signed_div_value must equal a two's-complement reference on the
    /// non-trapping 8-bit domain (every a, every b != 0, excluding INT_MIN/-1).
    #[test]
    fn signed_div_value_matches_reference_8bit() {
        use ordeal::eval::{Env, eval_bv};
        let a = v("a", 8);
        let b = v("b", 8);
        let term = signed_div_value(&a, &b, 8);
        for av in 0u128..256 {
            for bv in 1u128..256 {
                if av == 0x80 && bv == 0xFF {
                    continue; // trapping overflow case, handled by trap clause
                }
                let mut e = Env::new();
                e.insert("a".into(), av);
                e.insert("b".into(), bv);
                let got = eval_bv(&term, &e).unwrap() & 0xFF;
                let ai = av as i8 as i32;
                let bi = bv as i8 as i32;
                let want = (ai / bi) as i8 as u8 as u128;
                assert_eq!(got, want, "div_s a={av} b={bv}");
            }
        }
    }

    #[test]
    fn signed_rem_value_matches_reference_8bit() {
        use ordeal::eval::{Env, eval_bv};
        let a = v("a", 8);
        let b = v("b", 8);
        let term = signed_rem_value(&a, &b, 8);
        for av in 0u128..256 {
            for bv in 1u128..256 {
                if av == 0x80 && bv == 0xFF {
                    continue;
                }
                let mut e = Env::new();
                e.insert("a".into(), av);
                e.insert("b".into(), bv);
                let got = eval_bv(&term, &e).unwrap() & 0xFF;
                let ai = av as i8 as i32;
                let bi = bv as i8 as i32;
                let want = (ai % bi) as i8 as u8 as u128;
                assert_eq!(got, want, "rem_s a={av} b={bv}");
            }
        }
    }

    // ---- ACCEPT: a value-and-trap-preserving div lowering proves Unsat ----

    #[test]
    fn identity_div_transform_is_accepted() {
        let a = v("a", 32);
        let b = v("b", 32);
        // Same op on both sides: trap-equivalent and value-equivalent.
        let verdict = check_div_transform(DivKind::DivS, DivKind::DivS, &a, &b, 32);
        assert!(
            verdict.is_accepted(),
            "identity div_s transform must be accepted, got {verdict:?}"
        );
    }

    // ---- REJECT: div overflow / ÷0 trap-dropping folds are caught ----

    /// #273: constant-folding `div_s(INT_MIN, -1)` to a constant drops the
    /// signed-overflow trap. The optimized side is `INT_MIN` with no trap; the
    /// gate must REJECT with a counterexample.
    #[test]
    fn div_s_overflow_fold_is_rejected() {
        let int_min = c(0x8000_0000, 32);
        let neg_one = c(0xFFFF_FFFF, 32);
        // orig: div_s(INT_MIN, -1) — traps (overflow). opt: constant INT_MIN,
        // no trap. Model the opt as a no-trap define of the folded constant.
        let orig = div_define(DivKind::DivS, &int_min, &neg_one, 32);
        let opt = DefineOrTrap {
            value: c(0x8000_0000, 32),
            may_trap: BoolTerm::Ne(bx(c(0, 8)), bx(c(0, 8))), // false
        };
        let verdict = check_trap_equivalence_under(&[], &orig, &opt);
        assert!(
            matches!(verdict, TrapVerdict::RejectCounterexample(_)),
            "div_s(INT_MIN,-1) const-fold must be rejected (trap dropped), got {verdict:?}"
        );
    }

    /// #274: dead-eliminating a trapping div drops its ÷0 trap. Symbolic
    /// operands; the opt side has no trap. Must be rejected with divisor 0.
    #[test]
    fn dead_div_elimination_is_rejected() {
        let a = v("a", 32);
        let b = v("b", 32);
        let orig = div_define(DivKind::DivU, &a, &b, 32);
        let opt = DefineOrTrap {
            value: BvTerm::Udiv(bx(a.clone()), bx(b.clone())),
            may_trap: BoolTerm::Ne(bx(c(0, 8)), bx(c(0, 8))), // false — op deleted
        };
        match check_trap_equivalence_under(&[], &orig, &opt) {
            TrapVerdict::RejectCounterexample(m) => {
                let bval = m.iter().find(|(n, _)| n == "b").map(|(_, x)| *x);
                assert_eq!(bval, Some(0), "counterexample must set divisor to 0");
            }
            other => panic!("dead div elimination must be rejected, got {other:?}"),
        }
    }

    // ---- div → constant fold: check_div_discard (trap-free obligation, #273) ----

    /// A non-trapping constant div (`20 / 4`) proves trap-free ⇒ the const-fold
    /// is accepted. This is the ACCEPT half of the #273 backstop.
    #[test]
    fn div_discard_trap_free_const_is_accepted() {
        let a = c(20, 32);
        let b = c(4, 32);
        let verdict = check_div_discard(DivKind::DivS, &a, &b, 32);
        assert!(
            verdict.is_accepted(),
            "trap-free const div_s(20,4) fold must be accepted, got {verdict:?}"
        );
    }

    /// #273: `div_s(INT_MIN, -1)` overflows — the trap is real, so folding it to
    /// a constant drops a mandatory trap. `check_div_discard` must REJECT.
    #[test]
    fn div_discard_int_min_overflow_is_rejected() {
        let int_min = c(0x8000_0000, 32);
        let neg_one = c(0xFFFF_FFFF, 32);
        let verdict = check_div_discard(DivKind::DivS, &int_min, &neg_one, 32);
        assert!(
            matches!(verdict, TrapVerdict::RejectCounterexample(_)),
            "div_s(INT_MIN,-1) const-fold must be rejected (overflow trap), got {verdict:?}"
        );
    }

    /// A ÷0 constant div traps — folding it to a constant drops the trap.
    /// `check_div_discard` must REJECT.
    #[test]
    fn div_discard_div_by_zero_is_rejected() {
        let a = c(7, 32);
        let zero = c(0, 32);
        let verdict = check_div_discard(DivKind::DivU, &a, &zero, 32);
        assert!(
            matches!(verdict, TrapVerdict::RejectCounterexample(_)),
            "div_u(7,0) const-fold must be rejected (÷0 trap), got {verdict:?}"
        );
    }

    /// `rem_u(7, 0)` traps (÷0) — folding it to a constant drops the trap, so it
    /// must be REJECTED. `rem_u` is unsigned so ordeal's `trap_div` models it
    /// exactly (÷0 only, no overflow disjunct).
    #[test]
    fn rem_u_div_by_zero_is_rejected() {
        let a = c(7, 32);
        let zero = c(0, 32);
        let verdict = check_div_discard(DivKind::RemU, &a, &zero, 32);
        assert!(
            matches!(verdict, TrapVerdict::RejectCounterexample(_)),
            "rem_u(7,0) const-fold must be rejected (÷0 trap), got {verdict:?}"
        );
    }

    /// KNOWN OVER-APPROXIMATION (documented gap). Per the WASM spec,
    /// `rem_s(INT_MIN, -1)` does NOT trap — the result is 0. But ordeal's
    /// `trap_div(RemS, …)` shares the signed-overflow disjunct with `div_s`, so
    /// it reports this case as TRAPPING. `check_div_discard(RemS, …)` therefore
    /// (conservatively, but imprecisely) REJECTS a genuinely-safe `rem_s`
    /// overflow fold. Because a REJECT here would revert a valid optimization
    /// (a capability regression, not a soundness bug), the #288 backstop
    /// deliberately does NOT route `RemS` folds — they stay on their exact
    /// static #273 guard. This test PINS that over-approximation so a future
    /// ordeal fix (dropping the overflow disjunct for `RemS`) is noticed here
    /// and the backstop can then be extended to cover `RemS`.
    #[test]
    fn rem_s_int_min_is_over_approximated_rejected() {
        let int_min = c(0x8000_0000, 32);
        let neg_one = c(0xFFFF_FFFF, 32);
        let verdict = check_div_discard(DivKind::RemS, &int_min, &neg_one, 32);
        assert!(
            matches!(verdict, TrapVerdict::RejectCounterexample(_)),
            "ordeal over-approximates rem_s(INT_MIN,-1) as trapping (documented \
             gap; backstop must not route RemS), got {verdict:?}"
        );
    }

    // ---- REJECT: OOB load discard (zero-annihilation, #278) ----

    #[test]
    fn oob_load_discard_is_rejected() {
        // mul(load(addr), 0) → 0 discards the load. If addr may be OOB, the
        // load's trap is dropped. mem_bound = symbolic pages * 64KiB.
        let addr = v("addr", 32);
        let size = c(4, 32);
        let pages = v("pages", 32);
        let mem_bound = mem_bound_from_pages(&pages, 32);
        let verdict = check_mem_transform(&addr, &size, /* opt_traps */ false, &mem_bound);
        assert!(
            matches!(verdict, TrapVerdict::RejectCounterexample(_)),
            "discarding a possibly-OOB load must be rejected, got {verdict:?}"
        );
    }

    #[test]
    fn preserved_load_is_accepted() {
        // The access is preserved on both sides (same addr/size/bound): the OOB
        // trap condition is identical ⇒ trap-equivalent ⇒ accept.
        let addr = v("addr", 32);
        let size = c(4, 32);
        let pages = v("pages", 32);
        let mem_bound = mem_bound_from_pages(&pages, 32);
        let verdict = check_mem_transform(&addr, &size, /* opt_traps */ true, &mem_bound);
        assert!(
            verdict.is_accepted(),
            "preserved load must be accepted, got {verdict:?}"
        );
    }

    // ---- select with trapping discarded arm (#281) ----

    #[test]
    fn select_discard_trapping_div_arm_is_rejected() {
        // Discarded arm is div_u(x, 0)-shaped: its trap condition is ÷0, which
        // is satisfiable ⇒ dropping the arm may drop a trap ⇒ reject.
        let x = v("x", 32);
        let zero = c(0, 32);
        let arm_trap = trap::trap_div(DivOp::DivU, &x, &zero, 32);
        let verdict = check_select_discard(&arm_trap);
        assert!(
            matches!(verdict, TrapVerdict::RejectCounterexample(_)),
            "discarding a trapping div arm must be rejected, got {verdict:?}"
        );
    }

    #[test]
    fn select_discard_trapping_load_arm_is_rejected() {
        let addr = v("addr", 32);
        let size = c(4, 32);
        let pages = v("pages", 32);
        let mem_bound = mem_bound_from_pages(&pages, 32);
        let arm_trap = trap::trap_mem_oob(&addr, &size, &mem_bound);
        let verdict = check_select_discard(&arm_trap);
        assert!(
            matches!(verdict, TrapVerdict::RejectCounterexample(_)),
            "discarding a possibly-OOB load arm must be rejected, got {verdict:?}"
        );
    }

    #[test]
    fn select_discard_trap_free_arm_is_accepted() {
        // A provably trap-free discarded arm (constant trap condition = false)
        // may be dropped: trap-equivalent to `false`.
        let never_traps = BoolTerm::Ne(bx(c(0, 8)), bx(c(0, 8)));
        let verdict = check_select_discard(&never_traps);
        assert!(
            verdict.is_accepted(),
            "dropping a trap-free arm must be accepted, got {verdict:?}"
        );
    }

    // ---- the reusable under-hypothesis helper (#231 machinery) ----

    /// The divisor-nonzero PREMISE discharges the ÷0 trap: under `b != 0`, a
    /// div that has been proven safe to treat as total is trap-equivalent to a
    /// no-trap lowering — the #231 fact-driven-rewrite shape. Without the
    /// premise this same query is rejected (see `dead_div_elimination`).
    #[test]
    fn div_transform_accepted_under_nonzero_premise() {
        let a = v("a", 8);
        let b = v("b", 8);
        // Both sides carry the SAME div value; orig keeps the trap clause, opt
        // asserts no-trap. Under the premise `b != 0` (and no signed overflow
        // for DivU), the ÷0 disjunct is unreachable, so the trap clauses agree
        // (both false) and the values match.
        let orig = div_define(DivKind::DivU, &a, &b, 8);
        let opt = DefineOrTrap {
            value: BvTerm::Udiv(bx(a.clone()), bx(b.clone())),
            may_trap: BoolTerm::Ne(bx(c(0, 8)), bx(c(0, 8))), // false
        };
        let premise = BoolTerm::Ne(bx(b.clone()), bx(c(0, 8)));
        let verdict = check_trap_equivalence_under(&[premise], &orig, &opt);
        assert!(
            verdict.is_accepted(),
            "under b != 0, dropping the ÷0 trap must be accepted, got {verdict:?}"
        );
    }

    /// The same obligation WITHOUT the premise is rejected — confirming the
    /// premise is load-bearing (the helper is not vacuously accepting).
    #[test]
    fn div_transform_rejected_without_premise() {
        let a = v("a", 8);
        let b = v("b", 8);
        let orig = div_define(DivKind::DivU, &a, &b, 8);
        let opt = DefineOrTrap {
            value: BvTerm::Udiv(bx(a.clone()), bx(b.clone())),
            may_trap: BoolTerm::Ne(bx(c(0, 8)), bx(c(0, 8))),
        };
        let verdict = check_trap_equivalence_under(&[], &orig, &opt);
        assert!(
            matches!(verdict, TrapVerdict::RejectCounterexample(_)),
            "without the b != 0 premise the ÷0 trap-drop must be rejected, got {verdict:?}"
        );
    }

    /// `prove_under_premises` proves a plain validity: `x + 0 == x` for all x.
    #[test]
    fn prove_under_premises_proves_plain_validity() {
        let x = v("x", 32);
        let lhs = BvTerm::Add(bx(x.clone()), bx(c(0, 32)));
        let obligation = BoolTerm::Eq(bx(lhs), bx(x));
        assert!(prove_under_premises(&[], &obligation).is_accepted());
    }
}
