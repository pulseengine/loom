//! `eliminate_dead_stores` must judge each store in an `if`/`else` by its own
//! liveness.
//!
//! The liveness walk numbered stores while walking backward and visited the
//! `then` arm first, but the stores are removed in forward order, `then` arm
//! first. With a dead store in `then` and a live one in `else`, the verdicts
//! swapped: the live `else` store was removed and `f(0)` returned 0 instead
//! of 7. The value goes through a second memory, as in the fused module this
//! was found on.

use loom_core::{encode, optimize, parse};
use wasmtime::{Engine, Instance, Module as WtModule, Store};

const DEAD_THEN_LIVE_ELSE: &str = r#"
(module
  (memory 1)
  (memory $m1 1)
  (func (export "f") (param i32) (result i32) (local i32 i32)
    local.get 0
    if
      i32.const 5
      local.set 1
    else
      i32.const 7
      local.set 2
    end
    i32.const 0
    local.get 2
    i32.store $m1
    i32.const 0
    i32.load $m1))
"#;

/// The mirror case: live store in `then`, dead one in `else`.
const LIVE_THEN_DEAD_ELSE: &str = r#"
(module
  (memory 1)
  (memory $m1 1)
  (func (export "f") (param i32) (result i32) (local i32 i32)
    local.get 0
    if
      i32.const 7
      local.set 2
    else
      i32.const 5
      local.set 1
    end
    i32.const 0
    local.get 2
    i32.store $m1
    i32.const 0
    i32.load $m1))
"#;

fn run(wasm: &[u8], x: i32) -> i32 {
    let engine = Engine::default();
    let module = WtModule::new(&engine, wasm).expect("load");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let f = instance
        .get_typed_func::<i32, i32>(&mut store, "f")
        .expect("export f");
    f.call(&mut store, x).expect("call f")
}

fn check(wat: &str) {
    let orig = wat::parse_str(wat).expect("wat");
    let mut module = parse::parse_wasm(&orig).expect("parse");
    optimize::eliminate_dead_stores(&mut module).expect("pass");
    let out = encode::encode_wasm(&module).expect("encode");
    wasmparser::validate(&out).expect("output validates");
    for x in [0, 1] {
        assert_eq!(run(&out, x), run(&orig, x), "f({x})");
    }
}

#[test]
fn live_store_in_else_is_kept() {
    check(DEAD_THEN_LIVE_ELSE);
}

#[test]
fn live_store_in_then_is_kept() {
    check(LIVE_THEN_DEAD_ELSE);
}
