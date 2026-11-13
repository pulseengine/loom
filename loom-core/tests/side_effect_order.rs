//! Two miscompiles found by running original and optimized modules side by side.
//!
//! 1. Statement order through the term IR. `instructions_to_terms` emitted its
//!    `side_effects` list before its value `stack`, and a result-less `if` was
//!    kept on the stack. So in `if (x) { g = 7 }; g = 3` the `g = 3` was emitted
//!    first: with x = 1, g ended at 7 instead of 3. Both `constant_folding` and
//!    `optimize_advanced_instructions` go through the term IR.
//!
//! 2. `forward_carrier_locals` re-evaluated a carrier's defining expression
//!    (`local.get $x`) at the use even though `$x` was reassigned in between.

use loom_core::{encode, optimize, parse};
use wasmtime::{Engine, Instance, Module as WtModule, Store, Val};

const IF_THEN_SET: &str = r#"
(module
  (global $g (mut i32) (i32.const 0))
  (func (export "f") (param $x i32)
    (if (local.get $x) (then (global.set $g (i32.const 7))))
    (global.set $g (i32.const 3)))
  (export "g" (global $g)))
"#;

const CARRIER: &str = r#"
(module
  (func (export "h") (param $x f32) (param $c i32) (result f32) (local $y f32)
    local.get $x
    local.set $y
    local.get $c
    if
      local.get $x
      f32.neg
      local.set $x
    end
    local.get $x
    local.get $y
    f32.add))
"#;

/// Call `f(x)` and return the global `g` afterwards.
fn global_after_f(wasm: &[u8], x: i32) -> i32 {
    let engine = Engine::default();
    let module = WtModule::new(&engine, wasm).expect("load");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let f = instance.get_func(&mut store, "f").expect("export f");
    f.call(&mut store, &[Val::I32(x)], &mut []).expect("call f");
    let g = instance.get_global(&mut store, "g").expect("export g");
    g.get(&mut store).unwrap_i32()
}

fn h(wasm: &[u8], x: f32, c: i32) -> f32 {
    let engine = Engine::default();
    let module = WtModule::new(&engine, wasm).expect("load");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let h = instance
        .get_typed_func::<(f32, i32), f32>(&mut store, "h")
        .expect("export h");
    h.call(&mut store, (x, c)).expect("call h")
}

fn optimized(wat: &str, pass: fn(&mut loom_core::Module) -> anyhow::Result<()>) -> Vec<u8> {
    let wasm = wat::parse_str(wat).expect("wat");
    let mut module = parse::parse_wasm(&wasm).expect("parse");
    pass(&mut module).expect("pass");
    let out = encode::encode_wasm(&module).expect("encode");
    wasmparser::validate(&out).expect("output validates");
    out
}

fn check_if_then_set(out: &[u8]) {
    let orig = wat::parse_str(IF_THEN_SET).unwrap();
    for x in [0, 1] {
        assert_eq!(global_after_f(&orig, x), 3, "fixture: original, x = {x}");
        assert_eq!(global_after_f(out, x), 3, "optimized, x = {x}");
    }
}

#[test]
fn constant_folding_keeps_if_before_following_store() {
    check_if_then_set(&optimized(IF_THEN_SET, optimize::constant_folding));
}

#[test]
fn advanced_keeps_if_before_following_store() {
    check_if_then_set(&optimized(
        IF_THEN_SET,
        optimize::optimize_advanced_instructions,
    ));
}

#[test]
fn full_pipeline_keeps_if_before_following_store() {
    check_if_then_set(&optimized(IF_THEN_SET, optimize::optimize_module));
}

fn check_carrier(out: &[u8]) {
    let orig = wat::parse_str(CARRIER).unwrap();
    for (x, c) in [(1.5f32, 1), (1.5, 0), (-2.0, 1)] {
        let want = h(&orig, x, c);
        assert_eq!(h(out, x, c).to_bits(), want.to_bits(), "h({x}, {c})");
    }
    assert_eq!(h(&orig, 1.5, 1), 0.0, "fixture: original h(1.5, 1)");
}

#[test]
fn forward_carrier_does_not_read_a_reassigned_local() {
    check_carrier(&optimized(CARRIER, optimize::forward_carrier_locals));
}

#[test]
fn full_pipeline_does_not_read_a_reassigned_local() {
    check_carrier(&optimized(CARRIER, optimize::optimize_module));
}
