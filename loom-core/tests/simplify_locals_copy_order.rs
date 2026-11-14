//! `simplify_locals` must rewrite a read of a copied local only after the
//! copy, and only until either local is written again.
//!
//! It collected `local.get s; local.set d` copies over the whole body and
//! then rewrote EVERY read of `d` to `s`, including reads before the copy.
//! Below, the parameter is read first and later overwritten with a copy of
//! local 1; the first read became a read of local 1 (still 0), so `f(5)`
//! returned 31 instead of 36. The load goes through a second memory, as in
//! the module this was found on.

use loom_core::{encode, optimize, parse};
use wasmtime::{Engine, Instance, Module as WtModule, Store};

const READ_BEFORE_COPY: &str = r#"
(module
  (memory 1)
  (memory $m1 1)
  (func (export "f") (param i64) (result i64) (local i64)
    local.get 0
    i32.const 0
    i64.load $m1
    i64.const 31
    i64.add
    local.tee 1
    drop
    local.get 1
    local.set 0
    local.get 0
    i64.add))
"#;

/// The source is written after the copy: the copy's reads keep the old value.
const SOURCE_WRITTEN_AFTER_COPY: &str = r#"
(module
  (memory 1)
  (memory $m1 1)
  (func (export "f") (param i64) (result i64) (local i64)
    local.get 0
    local.set 1
    i32.const 0
    i64.load $m1
    i64.const 7
    i64.add
    local.set 0
    local.get 1
    local.get 0
    i64.add))
"#;

fn run(wasm: &[u8], x: i64) -> i64 {
    let engine = Engine::default();
    let module = WtModule::new(&engine, wasm).expect("load");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let f = instance
        .get_typed_func::<i64, i64>(&mut store, "f")
        .expect("export f");
    f.call(&mut store, x).expect("call f")
}

fn check(wat: &str) {
    let orig = wat::parse_str(wat).expect("wat");
    let mut module = parse::parse_wasm(&orig).expect("parse");
    optimize::simplify_locals(&mut module).expect("pass");
    let out = encode::encode_wasm(&module).expect("encode");
    wasmparser::validate(&out).expect("output validates");
    for x in [0, 5, -3] {
        assert_eq!(run(&out, x), run(&orig, x), "f({x})");
    }
}

#[test]
fn read_before_the_copy_keeps_its_local() {
    check(READ_BEFORE_COPY);
}

#[test]
fn copy_ends_when_its_source_is_written() {
    check(SOURCE_WRITTEN_AFTER_COPY);
}
