//! `eliminate_dead_stores` must keep a store whose value is read after a
//! branch out of a loop.
//!
//! The liveness walk set the live set before a loop to "live after the loop
//! plus the locals the body reads", discarding what the walk of the body
//! found. A branch from inside the loop to an enclosing label carries that
//! label's live locals, and those were lost. Two shapes:
//! - a loop counter whose only later read follows a `br_if` out of an inner
//!   loop: the update was removed and `f(1)` no longer terminated;
//! - a `local.tee` at entry whose value is read after a branch out of the
//!   loop: the tee was removed.

use loom_core::{encode, optimize, parse};
use wasmtime::{Config, Engine, Instance, Module as WtModule, Store};

const COUNTER: &str = r#"
(module
  (func (export "f") (param i32) (result i32)
    (local i32)
    loop $outer
      local.get 1
      i32.const 1
      i32.add
      local.set 1
      block $b
        block $c
          loop $inner
            local.get 0
            br_if 2
          end
        end
        i32.const 10
        return
      end
      local.get 1
      local.get 0
      i32.ne
      br_if 0
    end
    i32.const 0))
"#;

const TEE: &str = r#"
(module
  (func (export "f") (param i32) (result i32)
    (local i32 i32)
    local.get 0
    local.tee 1
    local.set 2
    block
      block
        loop
          local.get 2
          i32.eqz
          br_if 2
          local.get 2
          i32.const -1
          i32.add
          local.tee 2
          i32.eqz
          br_if 1
          br 0
        end
        i32.const -2
        return
      end
      local.get 1
      local.set 2
    end
    local.get 2))
"#;

/// Result of `f(x)`, or `None` if it runs out of fuel (does not terminate).
fn run(wasm: &[u8], x: i32) -> Option<i32> {
    let mut config = Config::new();
    config.consume_fuel(true);
    let engine = Engine::new(&config).expect("engine");
    let module = WtModule::new(&engine, wasm).expect("load");
    let mut store = Store::new(&engine, ());
    store.set_fuel(1_000_000).expect("fuel");
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let f = instance
        .get_typed_func::<i32, i32>(&mut store, "f")
        .expect("export f");
    f.call(&mut store, x).ok()
}

fn check(wat: &str) {
    let orig = wat::parse_str(wat).expect("wat");
    let mut module = parse::parse_wasm(&orig).expect("parse");
    optimize::eliminate_dead_stores(&mut module).expect("pass");
    let out = encode::encode_wasm(&module).expect("encode");
    wasmparser::validate(&out).expect("output validates");
    for x in [0, 1, 3] {
        let want = run(&orig, x);
        assert!(want.is_some(), "fixture: f({x}) terminates");
        assert_eq!(run(&out, x), want, "f({x})");
    }
}

#[test]
fn loop_counter_read_after_a_branch_out_is_kept() {
    check(COUNTER);
}

#[test]
fn tee_read_after_a_branch_out_of_a_loop_is_kept() {
    check(TEE);
}
