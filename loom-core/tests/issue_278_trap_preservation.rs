//! #278 end-to-end trap-preservation: zero-annihilation / absorption folds
//! (`x*0→0`, `0*x→0`, `x&0→0`, `x|-1→-1`, `select` with a constant condition,
//! …) must NOT discard a TRAPPING operand.
//!
//! Root cause (same class as #273/#274/#276): LOOM's value-equivalence / Z3
//! checking does not model traps. When the discarded operand is an out-of-bounds
//! `i32.load`/`i64.load` or a `div`/`rem` by zero, the operand FAULTS at runtime;
//! dropping it turns a trap into a returned value — a miscompile.
//!
//! Each case below is a module whose exported `f` computes `<trap> OP <const>`
//! where the fold would annihilate `<trap>`. We instantiate BOTH the original
//! and the LOOM-optimized module in wasmtime and assert they are
//! TRAP-EQUIVALENT: the original traps (OOB / div-by-zero), so the optimized one
//! must trap too — not return 0 / -1.

use loom_core::{encode, optimize_fused_module, parse};
use wasmtime::{Engine, Instance, Module as WtModule, Store, Val, ValType};

/// Outcome of running exported `f` with no args.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Trapped,
    Returned(i64),
}

fn run(wasm: &[u8]) -> Outcome {
    let engine = Engine::default();
    let module = WtModule::new(&engine, wasm).expect("wasmtime failed to load module");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate failed");
    let func = instance
        .get_func(&mut store, "f")
        .expect("export `f` missing");
    let ty = func.ty(&store);
    let result_ty = ty.results().next().expect("f must return a value");
    let mut results = [match result_ty {
        ValType::I64 => Val::I64(0),
        _ => Val::I32(0),
    }];
    match func.call(&mut store, &[], &mut results) {
        Ok(()) => match results[0] {
            Val::I32(x) => Outcome::Returned(x as i64),
            Val::I64(x) => Outcome::Returned(x),
            other => panic!("unexpected result {other:?}"),
        },
        Err(_) => Outcome::Trapped,
    }
}

fn optimize(wasm: &[u8]) -> Vec<u8> {
    let mut module = parse::parse_wasm(wasm).expect("parse failed");
    optimize_fused_module(&mut module).expect("optimize failed");
    encode::encode_wasm(&module).expect("encode failed")
}

/// Assert: the original module TRAPS, and the optimized module TRAPS identically
/// (i.e. the fold did NOT eliminate the trap). Also print the optimized wat so a
/// human can confirm the trapping op survives.
fn assert_trap_preserved(wat: &str) {
    let orig = wat::parse_str(wat).expect("wat parse failed");
    let opt = optimize(&orig);

    let orig_outcome = run(&orig);
    assert_eq!(
        orig_outcome,
        Outcome::Trapped,
        "test fixture must trap in the ORIGINAL module; got {orig_outcome:?}"
    );

    let opt_outcome = run(&opt);
    assert_eq!(
        opt_outcome,
        Outcome::Trapped,
        "#278 REGRESSION: optimized module did NOT trap (fold discarded a \
         trapping operand). optimized wat:\n{}",
        wasmprinter::print_bytes(&opt).unwrap_or_default()
    );
}

// A 1-page (64 KiB) memory; address 1_000_000 is far out of bounds.
const MEM: &str = "(memory 1)";

#[test]
fn t278_i32_mul_load_by_zero() {
    assert_trap_preserved(&format!(
        "(module {MEM}(func (export \"f\")(result i32)
            (i32.mul (i32.load (i32.const 1000000)) (i32.const 0))))"
    ));
}

#[test]
fn t278_i32_mul_zero_by_load() {
    assert_trap_preserved(&format!(
        "(module {MEM}(func (export \"f\")(result i32)
            (i32.mul (i32.const 0) (i32.load (i32.const 1000000)))))"
    ));
}

#[test]
fn t278_i32_and_load_with_zero() {
    assert_trap_preserved(&format!(
        "(module {MEM}(func (export \"f\")(result i32)
            (i32.and (i32.load (i32.const 1000000)) (i32.const 0))))"
    ));
}

#[test]
fn t278_i32_or_load_with_allones() {
    assert_trap_preserved(&format!(
        "(module {MEM}(func (export \"f\")(result i32)
            (i32.or (i32.load (i32.const 1000000)) (i32.const -1))))"
    ));
}

#[test]
fn t278_i64_mul_load_by_zero() {
    assert_trap_preserved(&format!(
        "(module {MEM}(func (export \"f\")(result i64)
            (i64.mul (i64.load (i32.const 1000000)) (i64.const 0))))"
    ));
}

#[test]
fn t278_i64_and_load_with_zero() {
    assert_trap_preserved(&format!(
        "(module {MEM}(func (export \"f\")(result i64)
            (i64.and (i64.load (i32.const 1000000)) (i64.const 0))))"
    ));
}

#[test]
fn t278_i32_mul_div_by_zero_operand() {
    // (i32.div_u 1 0) traps; multiplying it by 0 must NOT annihilate the trap.
    assert_trap_preserved(
        "(module (func (export \"f\")(result i32)
            (i32.mul (i32.div_u (i32.const 1) (i32.const 0)) (i32.const 0))))",
    );
}

#[test]
fn t278_select_const_cond_discards_trapping_arm() {
    // select(0, <trap>, 7) picks the false arm (7) but WASM evaluates BOTH arms,
    // so the trapping true arm must still fault.
    assert_trap_preserved(&format!(
        "(module {MEM}(func (export \"f\")(result i32)
            (select (i32.load (i32.const 1000000)) (i32.const 7) (i32.const 0))))"
    ));
}

// ---- No over-suppression: a TRAP-FREE operand still folds to the constant ----

#[test]
fn t278_trap_free_mul_by_zero_still_folds_to_zero() {
    let wat = "(module (func (export \"f\")(param i32)(result i32)
        (i32.mul (local.get 0) (i32.const 0))))";
    let orig = wat::parse_str(wat).unwrap();
    let opt = optimize(&orig);
    let opt_wat = wasmprinter::print_bytes(&opt).unwrap();
    // The trap-free operand must be discarded and folded away: no i32.mul / no
    // local.get survives — the body is just `i32.const 0`.
    assert!(
        !opt_wat.contains("i32.mul") && !opt_wat.contains("local.get"),
        "trap-free (local.get 0)*0 must fold to a constant, got:\n{opt_wat}"
    );

    // And it must actually behave as 0 for any argument.
    let engine = Engine::default();
    let module = WtModule::new(&engine, &opt).unwrap();
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).unwrap();
    let func = instance.get_func(&mut store, "f").unwrap();
    let mut results = [Val::I32(123)];
    func.call(&mut store, &[Val::I32(999)], &mut results)
        .unwrap();
    assert_eq!(results[0].i32(), Some(0), "folded x*0 must yield 0");
}
