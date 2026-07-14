//! #281 end-to-end trap-preservation for the constant-condition `select` fold.
//!
//! gale #281: `loom optimize` folded `select(const_cond, a, b)` to the taken
//! arm, DISCARDING the other arm. But WASM `select` is EAGER — BOTH value
//! operands are evaluated before the condition selects one (WASM Core §4.4.1) —
//! so if the DISCARDED arm contains a trapping op (out-of-bounds load, div, …),
//! dropping it silently deletes a trap: a miscompile.
//!
//! Fix (shared with #278): the constant-condition select fold in the algebraic
//! simplifier is gated on the DISCARDED arm being provably trap-free
//! (`is_no_trap_expr`, the `is_forwardable_expr` whitelist). If the discarded
//! arm may trap, the `select` is kept intact and the trap is preserved.
//!
//! These cases exercise the fold via `constant_folding` — the pass that drives
//! the ISLE `rewrite_pure` term simplifier where the constant-condition select
//! fold lives. We cover the trapping op being on EITHER discarded side, and
//! confirm a trap-free discarded arm STILL folds.

use loom_core::{encode, optimize, parse};
use wasmtime::{Engine, Instance, Module as WtModule, Store, Val};

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Trapped,
    Returned(i32),
}

fn run(wasm: &[u8]) -> Outcome {
    let engine = Engine::default();
    let module = WtModule::new(&engine, wasm).expect("wasmtime failed to load module");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate failed");
    let func = instance
        .get_func(&mut store, "f")
        .expect("export `f` missing");
    let mut results = [Val::I32(0)];
    match func.call(&mut store, &[], &mut results) {
        Ok(()) => match results[0] {
            Val::I32(x) => Outcome::Returned(x),
            other => panic!("unexpected result {other:?}"),
        },
        Err(_) => Outcome::Trapped,
    }
}

fn optimize_wasm(wasm: &[u8]) -> Vec<u8> {
    let mut module = parse::parse_wasm(wasm).expect("parse failed");
    optimize::constant_folding(&mut module).expect("optimize failed");
    encode::encode_wasm(&module).expect("encode failed")
}

/// The original module traps (the trapping op is reached under WASM's eager
/// evaluation of both select arms). Assert the optimized module ALSO traps —
/// i.e. the constant-condition fold did NOT discard the trapping arm.
fn assert_trap_preserved(wat: &str) {
    let orig = wat::parse_str(wat).expect("wat parse failed");
    assert_eq!(
        run(&orig),
        Outcome::Trapped,
        "fixture must trap in the ORIGINAL module"
    );

    let opt = optimize_wasm(&orig);
    let opt_outcome = run(&opt);
    assert_eq!(
        opt_outcome,
        Outcome::Trapped,
        "#281 REGRESSION: optimized module did NOT trap (constant-condition \
         select fold discarded a trapping arm). optimized wat:\n{}",
        wasmprinter::print_bytes(&opt).unwrap_or_default()
    );
}

// 1-page (64 KiB) memory; address 1_000_000 is far out of bounds → i32.load traps.
const MEM: &str = "(memory 1)";

#[test]
fn t281_true_const_discards_trapping_false_arm() {
    // select(true=7, false=load_OOB, cond=1): cond!=0 picks the TRUE arm (7),
    // but WASM already evaluated the false arm (load_OOB) → traps. The fold must
    // NOT drop the discarded (false) load arm.
    assert_trap_preserved(&format!(
        "(module {MEM}(func (export \"f\")(result i32)
            (select (i32.const 7) (i32.load (i32.const 1000000)) (i32.const 1))))"
    ));
}

#[test]
fn t281_false_const_discards_trapping_true_arm() {
    // select(true=load_OOB, false=7, cond=0): cond==0 picks the FALSE arm (7),
    // but the true arm (load_OOB) is still evaluated → traps. The fold must NOT
    // drop the discarded (true) load arm.
    assert_trap_preserved(&format!(
        "(module {MEM}(func (export \"f\")(result i32)
            (select (i32.load (i32.const 1000000)) (i32.const 7) (i32.const 0))))"
    ));
}

#[test]
fn t281_true_const_discards_trapping_div_false_arm() {
    // Non-memory trap: (i32.div_u 1 0) in the discarded false arm must survive.
    assert_trap_preserved(
        "(module (func (export \"f\")(result i32)
            (select (i32.const 7) (i32.div_u (i32.const 1) (i32.const 0)) (i32.const 1))))",
    );
}

// ---- No over-suppression: a trap-free discarded arm STILL folds away ----

#[test]
fn t281_trap_free_arms_fold_to_taken() {
    // select(true=local.get 0, false=local.get 1, cond=1) → local.get 0.
    // Both arms are trap-free (locals), so the fold is sound: no `select`
    // survives and the taken arm is the whole body.
    let wat = "(module (func (export \"f\")(param i32 i32)(result i32)
        (select (local.get 0) (local.get 1) (i32.const 1))))";
    let orig = wat::parse_str(wat).unwrap();
    let opt = optimize_wasm(&orig);
    let opt_wat = wasmprinter::print_bytes(&opt).unwrap();
    assert!(
        !opt_wat.contains("select"),
        "trap-free select(1, a, b) must fold to `a` (no select left), got:\n{opt_wat}"
    );

    // And it must actually return the taken (true) arm's value.
    let engine = Engine::default();
    let module = WtModule::new(&engine, &opt).unwrap();
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).unwrap();
    let func = instance.get_func(&mut store, "f").unwrap();
    let mut results = [Val::I32(0)];
    func.call(&mut store, &[Val::I32(11), Val::I32(22)], &mut results)
        .unwrap();
    assert_eq!(
        results[0].i32(),
        Some(11),
        "folded select(1, a, b) must yield the true arm `a`"
    );
}

#[test]
fn t281_trap_free_false_cond_folds_to_false_arm() {
    // select(true=local.get 0, false=local.get 1, cond=0) → local.get 1.
    let wat = "(module (func (export \"f\")(param i32 i32)(result i32)
        (select (local.get 0) (local.get 1) (i32.const 0))))";
    let orig = wat::parse_str(wat).unwrap();
    let opt = optimize_wasm(&orig);
    let opt_wat = wasmprinter::print_bytes(&opt).unwrap();
    assert!(
        !opt_wat.contains("select"),
        "trap-free select(0, a, b) must fold to `b` (no select left), got:\n{opt_wat}"
    );

    let engine = Engine::default();
    let module = WtModule::new(&engine, &opt).unwrap();
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).unwrap();
    let func = instance.get_func(&mut store, "f").unwrap();
    let mut results = [Val::I32(0)];
    func.call(&mut store, &[Val::I32(11), Val::I32(22)], &mut results)
        .unwrap();
    assert_eq!(
        results[0].i32(),
        Some(22),
        "folded select(0, a, b) must yield the false arm `b`"
    );
}
