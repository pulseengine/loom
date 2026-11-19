//! The translation validator must decide functions with loops, not accept
//! them on a check that compares only part of the state.
//!
//! The loop verifier compared the locals and stack of each loop body after
//! one iteration and nothing else: code outside the loops, globals, memory
//! and traps were never compared, loops it found complex were skipped, and
//! two functions with different numbers of loops were accepted outright.

use loom_core::{parse, verify::TranslationValidator, verify::VerificationSignatureContext};

fn accepted(orig: &str, new: &str) -> bool {
    let a = parse::parse_wat(orig).expect("parse original");
    let b = parse::parse_wat(new).expect("parse rewrite");
    let ctx = VerificationSignatureContext::from_module(&a);
    TranslationValidator::new_with_context(&a.functions[0], "test", ctx)
        .verify(&b.functions[0])
        .is_ok()
}

/// A loop counter whose update is read only after a branch out of a nested
/// loop (the #516 shape). Dropping the update makes f(1) never terminate.
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

const COUNTER_UPDATE_DROPPED: &str = r#"
(module
  (func (export "f") (param i32) (result i32)
    (local i32)
    loop $outer
      local.get 1
      i32.const 1
      i32.add
      drop
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

/// A store after the loop, which the loop verifier never looked at.
const STORE_AFTER_LOOP: &str = r#"
(module
  (memory 1)
  (func (export "f") (param i32) (result i32)
    (local i32)
    loop $l
      local.get 1
      i32.const 1
      i32.add
      local.tee 1
      local.get 0
      i32.lt_u
      br_if $l
    end
    i32.const 0
    local.get 1
    i32.store
    local.get 1))
"#;

const STORE_AFTER_LOOP_DROPPED: &str = r#"
(module
  (memory 1)
  (func (export "f") (param i32) (result i32)
    (local i32)
    loop $l
      local.get 1
      i32.const 1
      i32.add
      local.tee 1
      local.get 0
      i32.lt_u
      br_if $l
    end
    local.get 1))
"#;

/// A global written inside the loop body.
const GLOBAL_IN_LOOP: &str = r#"
(module
  (global $g (mut i32) (i32.const 0))
  (func (export "f") (param i32) (result i32)
    (local i32)
    loop $l
      global.get $g
      i32.const 1
      i32.add
      global.set $g
      local.get 1
      i32.const 1
      i32.add
      local.tee 1
      local.get 0
      i32.lt_u
      br_if $l
    end
    local.get 1))
"#;

const GLOBAL_IN_LOOP_DROPPED: &str = r#"
(module
  (global $g (mut i32) (i32.const 0))
  (func (export "f") (param i32) (result i32)
    (local i32)
    loop $l
      local.get 1
      i32.const 1
      i32.add
      local.tee 1
      local.get 0
      i32.lt_u
      br_if $l
    end
    local.get 1))
"#;

/// Folding a constant inside a loop body, which must stay provable.
const FOLDABLE: &str = r#"
(module
  (func (export "f") (param i32) (result i32)
    (local i32)
    loop $l
      local.get 1
      i32.const 2
      i32.const 3
      i32.add
      i32.add
      local.tee 1
      local.get 0
      i32.lt_u
      br_if $l
    end
    local.get 1))
"#;

const FOLDED: &str = r#"
(module
  (func (export "f") (param i32) (result i32)
    (local i32)
    loop $l
      local.get 1
      i32.const 5
      i32.add
      local.tee 1
      local.get 0
      i32.lt_u
      br_if $l
    end
    local.get 1))
"#;

#[test]
fn dropped_loop_counter_update_is_rejected() {
    assert!(!accepted(COUNTER, COUNTER_UPDATE_DROPPED));
}

#[test]
fn dropped_store_after_a_loop_is_rejected() {
    assert!(!accepted(STORE_AFTER_LOOP, STORE_AFTER_LOOP_DROPPED));
}

#[test]
fn dropped_global_write_inside_a_loop_is_rejected() {
    assert!(!accepted(GLOBAL_IN_LOOP, GLOBAL_IN_LOOP_DROPPED));
}

#[test]
fn folding_inside_a_loop_is_accepted() {
    assert!(accepted(FOLDABLE, FOLDED));
}

#[test]
fn unchanged_loops_are_accepted() {
    for f in [COUNTER, STORE_AFTER_LOOP, GLOBAL_IN_LOOP, FOLDABLE] {
        assert!(accepted(f, f));
    }
}
