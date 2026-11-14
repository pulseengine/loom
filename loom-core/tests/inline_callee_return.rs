//! Inlining must keep a callee's early exit inside the inlined copy.
//!
//! The inliner spliced a callee's body into the caller unchanged, so a
//! `return` in the callee (or a branch to the callee's function label)
//! returned from the CALLER. `scan` below calls a saturating add inside a
//! loop; after inlining, the add's early `return` ended `scan` with the
//! saturated value instead of letting the loop continue.

use loom_core::{encode, optimize, parse};
use wasmtime::{Engine, Instance, Module as WtModule, Store};

/// Saturating add with early returns in nested `if`s, called in a loop.
const SCAN: &str = r#"
(module
  (func $sat (param $a i32) (param $b i32) (result i32)
    (local $s i32)
    (local.set $s (i32.add (local.get $a) (local.get $b)))
    (if (i32.ge_s (local.get $a) (i32.const 0))
      (then
        (if (i32.ge_s (local.get $b) (i32.const 0))
          (then (if (i32.lt_s (local.get $s) (i32.const 0))
            (then (return (i32.const 2147483647)))))))
      (else
        (if (i32.lt_s (local.get $b) (i32.const 0))
          (then (if (i32.ge_s (local.get $s) (i32.const 0))
            (then (return (i32.const -2147483648))))))))
    (local.get $s))
  (func (export "f") (param $n i32) (result i32)
    (local $i i32) (local $acc i32)
    (loop $l
      (if (i32.ge_s (local.get $i) (local.get $n)) (then (return (local.get $acc))))
      (local.set $acc (call $sat (local.get $acc)
        (i32.mul (i32.mul (local.get $i) (local.get $i)) (i32.const 1000003))))
      (if (i32.eq (local.get $acc) (i32.const 2147483647)) (then (return (local.get $i))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l))
    (unreachable)))
"#;

/// An early `return` in a callee called from straight-line code.
const STRAIGHT: &str = r#"
(module
  (func $g (param i32) (result i32)
    local.get 0
    i32.const 0
    i32.lt_s
    if
      local.get 0
      return
    end
    i32.const 100)
  (func (export "f") (param i32) (result i32)
    local.get 0
    call $g
    local.get 0
    i32.add))
"#;

/// A callee that leaves through a branch to its own function label.
const BRANCH_OUT: &str = r#"
(module
  (func $g (param i32) (result i32)
    i32.const 7
    local.get 0
    br_if 0
    drop
    i32.const 9)
  (func (export "f") (param i32) (result i32)
    local.get 0
    call $g
    i32.const 1
    i32.add))
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

fn check(wat: &str, pass: fn(&mut loom_core::Module) -> anyhow::Result<()>, inputs: &[i32]) {
    let orig = wat::parse_str(wat).expect("wat");
    let mut module = parse::parse_wasm(&orig).expect("parse");
    pass(&mut module).expect("pass");
    let out = encode::encode_wasm(&module).expect("encode");
    wasmparser::validate(&out).expect("output validates");
    for &x in inputs {
        assert_eq!(run(&out, x), run(&orig, x), "f({x})");
    }
}

#[test]
fn inlined_return_inside_a_loop_stays_in_the_callee() {
    check(SCAN, optimize::inline_functions, &[0, 5, 19, 100]);
}

#[test]
fn inlined_return_in_straight_line_code_stays_in_the_callee() {
    check(STRAIGHT, optimize::inline_functions, &[-5, 0, 5]);
}

#[test]
fn inlined_branch_to_the_function_label_stays_in_the_callee() {
    check(BRANCH_OUT, optimize::inline_functions, &[0, 1]);
}

#[test]
fn full_pipeline_keeps_an_inlined_return_in_the_callee() {
    check(SCAN, optimize::optimize_module, &[0, 5, 19, 100]);
}
