//! `coalesce_locals` must not merge two locals whose values are both needed.
//!
//! Each case runs the original and the coalesced module in wasmtime and
//! compares results. Two cases fail on the previous code:
//! - colors were used directly as local indices, so color 0 landed on
//!   parameter 0 (a copy `$y = $x` was merged into `$x`, which is then
//!   reassigned while `$y` is still needed);
//! - locals of different types could share a slot.
//!
//! The other cases pass on the previous code too. They pin the results the
//! now more conservative live ranges must keep: a range spans every
//! occurrence, a loop extends it to the whole loop, and a local first touched
//! anywhere but a top-level `local.set` / `local.tee` gets its own slot.

use loom_core::{encode, optimize, parse};
use wasmtime::{Engine, Instance, Module as WtModule, Store, Val};

fn call(wasm: &[u8], name: &str, args: &[Val]) -> Val {
    let engine = Engine::default();
    let module = WtModule::new(&engine, wasm).expect("load");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let f = instance.get_func(&mut store, name).expect("export");
    let mut out = [Val::I32(0)];
    f.call(&mut store, args, &mut out).expect("call");
    out[0]
}

fn coalesced(wat: &str) -> Vec<u8> {
    let wasm = wat::parse_str(wat).expect("wat");
    let mut module = parse::parse_wasm(&wasm).expect("parse");
    optimize::coalesce_locals(&mut module).expect("coalesce");
    let out = encode::encode_wasm(&module).expect("encode");
    wasmparser::validate(&out).expect("output validates");
    out
}

fn same(wat: &str, name: &str, inputs: &[Vec<Val>]) {
    let orig = wat::parse_str(wat).unwrap();
    let out = coalesced(wat);
    for args in inputs {
        let want = call(&orig, name, args);
        let got = call(&out, name, args);
        assert_eq!(
            format!("{got:?}"),
            format!("{want:?}"),
            "{name}({args:?}): coalesced differs from original"
        );
    }
}

/// `$y` copies parameter `$x`; `$x` is then reassigned while `$y` is live.
#[test]
fn copy_of_reassigned_parameter_is_kept() {
    let wat = r#"
      (module (func (export "h") (param $x f32) (param $c i32) (result f32) (local $y f32)
        local.get $x local.set $y
        local.get $c if local.get $x f32.neg local.set $x end
        local.get $x local.get $y f32.add))"#;
    same(
        wat,
        "h",
        &[
            vec![Val::F32(1.5f32.to_bits()), Val::I32(1)],
            vec![Val::F32(1.5f32.to_bits()), Val::I32(0)],
        ],
    );
}

/// `$z` is written only inside the `if`; with `$c = 0` it is read while still
/// zero, so it must not take `$t`'s slot.
#[test]
fn conditionally_written_local_stays_zero_on_the_other_path() {
    let wat = r#"
      (module (func (export "c") (param $p i32) (param $c i32) (result i32)
          (local $t i32) (local $z i32)
        local.get $p i32.const 5 i32.add local.set $t
        local.get $t drop
        local.get $c if i32.const 9 local.set $z end
        local.get $z))"#;
    same(
        wat,
        "c",
        &[
            vec![Val::I32(1), Val::I32(0)],
            vec![Val::I32(1), Val::I32(1)],
        ],
    );
}

/// Loops are extended to their whole extent. (`coalesce_locals` skips any
/// function with a non-tail branch, so a loop here runs once; this pins the
/// result, it is not a reachable failure on the previous code.)
/// `$a` is last read at the top of the loop and written later in the body;
/// the write is read in the next iteration. `$b` lives between that write
/// and the back edge, so it must not share `$a`'s slot.
#[test]
fn value_carried_around_loop_is_kept() {
    let wat = r#"
      (module (func (export "lp") (param $n i32) (result i32)
          (local $a i32) (local $b i32) (local $acc i32)
        i32.const 1 local.set $a
        i32.const 0 local.set $acc
        loop $L
          local.get $acc local.get $a i32.add local.set $acc
          local.get $a i32.const 2 i32.mul local.set $a
          local.get $n i32.const 1 i32.sub local.set $b
          local.get $b local.set $n
          local.get $b br_if $L
        end
        local.get $acc))"#;
    same(
        wat,
        "lp",
        &[vec![Val::I32(1)], vec![Val::I32(3)], vec![Val::I32(5)]],
    );
}

/// `$z` is read before it is written, so it must still be 0 there even
/// though `$t` was written earlier.
#[test]
fn zero_initialised_read_is_kept() {
    let wat = r#"
      (module (func (export "z") (param $p i32) (result i32) (local $t i32) (local $z i32)
        local.get $p i32.const 5 i32.add local.set $t
        local.get $t drop
        local.get $z))"#;
    same(wat, "z", &[vec![Val::I32(1)], vec![Val::I32(9)]]);
}

/// An i32 and an f32 local with disjoint ranges must not share a slot.
#[test]
fn locals_of_different_types_are_kept_apart() {
    let wat = r#"
      (module (func (export "m") (param $p i32) (result f32) (local $i i32) (local $f f32)
        local.get $p i32.const 1 i32.add local.set $i
        local.get $i drop
        f32.const 2.5 local.set $f
        local.get $f))"#;
    same(wat, "m", &[vec![Val::I32(1)]]);
}
