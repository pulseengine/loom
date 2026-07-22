//! Runtime wiring of the [`crate::trap_gate`] trap-equivalence gate onto the
//! optimization passes (issue #288).
//!
//! # Why this exists
//!
//! [`crate::trap_gate`] (issue #279/#284) is a certificate-checked
//! trap-equivalence oracle, but until #288 its transform-checking entry points
//! had **zero callers** — every trap-relevant fold was guarded only by a
//! hand-written static predicate (#273/#274/#276/#278/#281). Those static guards
//! remain the feature-independent floor and are NOT touched here. This module is
//! purely **additive**: it invokes the gate as a proof BACKSTOP at each
//! trap-relevant fold site, so a fold that slipped past a static guard (or a
//! future guard regression) is caught by a checked `Unsat` certificate before it
//! ships.
//!
//! # The contract
//!
//! Each entry point takes the ORIGINAL and OPTIMIZED instruction stream of one
//! function, detects the trap-relevant folds that FIRED, and returns `true` iff
//! **every** such fold is accepted by the gate (trap-and-value equivalence
//! proven, LRAT certificate re-checked). A `false` return means the pass MUST
//! revert this function to its original instructions. With the `verification`
//! feature OFF this module is compiled out entirely and passes behave exactly as
//! before (byte-identical).
//!
//! # Scope (honest coverage)
//!
//! Wired: the `div_s` / `div_u` / `rem_u` / `rem_s` → **constant** fold (#273
//! class), which is soundly and exactly modelable because the fold's operands are
//! literal constants — ordeal decides the trap condition with no symbolic slack.
//! The gate proves the folded op could not have trapped
//! ([`crate::trap_gate::check_div_discard`]).
//!
//! `rem_s` joined the routed set with ordeal 0.14.0 (ordeal#84 / loom#288):
//! ordeal < 0.10.0's `trap_div(RemS)` over-approximated `rem_s(INT_MIN, -1)` as
//! trapping (it shared the `INT_MIN / -1` overflow disjunct with `div_s`, but
//! WASM defines `rem_s(INT_MIN, -1) = 0`, no trap), so routing it would have
//! falsely REJECTED that sound fold and reverted a valid optimization. ordeal
//! 0.14.0 drops the disjunct for remainder, so `rem_s` is now modeled exactly and
//! routed like the other three kinds. See `div_meta` and the un-pinning tests
//! `trap_gate::tests::rem_s_int_min_neg1_fold_is_accepted` (+ its wrong-value
//! dual).
//!
//! Documented remaining gaps (static guard is sole authority, see PR #288):
//! - The vacuum `load; drop` discard (#278) already never fires — `is_pure_pusher`
//!   excludes loads — so there is no fired fold to back-check at that site; the
//!   [`crate::trap_gate::check_mem_transform`] API is exercised by its own unit
//!   tests but not on a live fold, because loom carries no symbolic memory bound
//!   at the instruction layer to feed it soundly.
//! - The `select(const, a, b)` arm-discard (#281) fold lives in `loom-shared`'s
//!   `rewrite_pure`, below the ordeal dependency boundary, and its discarded arm
//!   is a *value term* whose trap condition loom does not reconstruct at the
//!   instruction layer; modeling it soundly would require lifting arbitrary arm
//!   expressions into `BvTerm`s, which is out of scope for this additive
//!   backstop. Its static `is_no_trap_expr` guard remains sole authority.

#![cfg(feature = "verification")]

use crate::Instruction;
use crate::trap_gate::{DivKind, TrapVerdict, check_div_discard, constant};

/// Width + kind of a div/rem instruction, for the trap-gate query.
///
/// All four div/rem kinds are routed. `rem_s` was previously EXCLUDED because
/// ordeal < 0.10.0's `trap_div` shared the `INT_MIN / -1` signed-overflow
/// disjunct between `div_s` and `rem_s`, but WASM `rem_s(INT_MIN, -1)` does NOT
/// trap (result 0) — so routing it would have REJECTED a genuinely-safe fold and
/// reverted a valid optimization. ordeal 0.14.0 (ordeal#84 / loom#288) drops that
/// disjunct for remainder, so `rem_s` is now modeled exactly and joins the routed
/// set. See `trap_gate::tests::rem_s_int_min_neg1_fold_is_accepted`.
fn div_meta(instr: &Instruction) -> Option<(DivKind, u32)> {
    use Instruction::*;
    Some(match instr {
        I32DivS => (DivKind::DivS, 32),
        I32DivU => (DivKind::DivU, 32),
        I32RemU => (DivKind::RemU, 32),
        I32RemS => (DivKind::RemS, 32),
        I64DivS => (DivKind::DivS, 64),
        I64DivU => (DivKind::DivU, 64),
        I64RemU => (DivKind::RemU, 64),
        I64RemS => (DivKind::RemS, 64),
        _ => return None,
    })
}

/// The unsigned bit pattern of an integer constant instruction (width-masked),
/// for feeding the trap-gate as an ordeal `constant`.
fn const_bits(instr: &Instruction) -> Option<(u128, u32)> {
    match instr {
        Instruction::I32Const(v) => Some(((*v as u32) as u128, 32)),
        Instruction::I64Const(v) => Some(((*v as u64) as u128, 64)),
        _ => None,
    }
}

/// Count the constant-operand div/rem fold CANDIDATES in a straight-line-or-
/// nested instruction stream: a `<const> <const> <div/rem>` triple whose two
/// preceding pushers are both constants of the matching width. Recurses into
/// `Block`/`Loop`/`If` bodies. Returns one entry `(kind, width, a_bits, b_bits)`
/// per candidate; ordering is a stable pre-order so two streams can be compared
/// positionally.
fn div_const_fold_candidates(instrs: &[Instruction]) -> Vec<(DivKind, u32, u128, u128)> {
    let mut out = Vec::new();
    collect_candidates(instrs, &mut out);
    out
}

fn collect_candidates(instrs: &[Instruction], out: &mut Vec<(DivKind, u32, u128, u128)>) {
    for (i, instr) in instrs.iter().enumerate() {
        match instr {
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                collect_candidates(body, out);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                collect_candidates(then_body, out);
                collect_candidates(else_body, out);
            }
            _ => {
                if let Some((kind, width)) = div_meta(instr) {
                    // The two immediately-preceding instructions must be the
                    // constant operands (postfix order: a, b, op).
                    if i >= 2 {
                        if let (Some((a, aw)), Some((b, bw))) =
                            (const_bits(&instrs[i - 2]), const_bits(&instrs[i - 1]))
                        {
                            if aw == width && bw == width {
                                out.push((kind, width, a, b));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Back-check every div/rem → constant fold that FIRED between `original` and
/// `optimized`, using [`crate::trap_gate::check_div_discard`].
///
/// A fold is deemed to have FIRED when a `<const> <const> <div/rem>` candidate
/// present in `original` is no longer present in `optimized` (the pure rewriter
/// replaced it with a single constant). For every such fired fold the gate must
/// prove the op was trap-free; if any is not accepted, this returns `false` and
/// the caller MUST revert the function.
///
/// Candidates that survive into `optimized` were NOT folded (the static #273
/// guard held the op in place) and need no proof — the op still traps correctly
/// at runtime.
///
/// Returns `true` when every fired fold is proven trap-free (or none fired).
pub fn accept_div_const_folds(original: &[Instruction], optimized: &[Instruction]) -> bool {
    use std::collections::HashMap;

    let before = div_const_fold_candidates(original);
    if before.is_empty() {
        return true; // no trap-relevant div/rem fold could have fired
    }
    let after = div_const_fold_candidates(optimized);

    // Multiset of surviving candidates — a candidate present in `after` was not
    // folded. Anything in `before` beyond the `after` count is a FIRED fold.
    let mut after_counts: HashMap<(DivKind, u32, u128, u128), usize> = HashMap::new();
    for cand in &after {
        *after_counts.entry(*cand).or_insert(0) += 1;
    }

    let mut before_counts: HashMap<(DivKind, u32, u128, u128), usize> = HashMap::new();
    for cand in &before {
        *before_counts.entry(*cand).or_insert(0) += 1;
    }

    for (cand, n_before) in &before_counts {
        let n_after = after_counts.get(cand).copied().unwrap_or(0);
        if *n_before > n_after {
            // At least one instance of this candidate was folded away. PROVE the
            // op is trap-free for these (constant) operands; if not, reject.
            let (kind, width, a, b) = *cand;
            let a_term = constant(a, width);
            let b_term = constant(b, width);
            let verdict: TrapVerdict = check_div_discard(kind, &a_term, &b_term, width);
            if !verdict.is_accepted() {
                return false;
            }
        }
    }
    true
}

/// Test-only observability: how many div/rem → constant folds fired between the
/// two streams. Lets the anti-regression test assert the backstop was actually
/// consulted (a fold was seen and checked), not vacuously satisfied.
#[cfg(test)]
pub fn fired_div_const_fold_count(original: &[Instruction], optimized: &[Instruction]) -> usize {
    use std::collections::HashMap;
    let before = div_const_fold_candidates(original);
    let after = div_const_fold_candidates(optimized);
    let mut after_counts: HashMap<(DivKind, u32, u128, u128), usize> = HashMap::new();
    for c in &after {
        *after_counts.entry(*c).or_insert(0) += 1;
    }
    let mut before_counts: HashMap<(DivKind, u32, u128, u128), usize> = HashMap::new();
    for c in &before {
        *before_counts.entry(*c).or_insert(0) += 1;
    }
    let mut fired = 0;
    for (c, nb) in &before_counts {
        let na = after_counts.get(c).copied().unwrap_or(0);
        if *nb > na {
            fired += nb - na;
        }
    }
    fired
}
