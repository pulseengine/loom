//! `--stats` counts an accepted transform as "proven" only when the validator
//! compared everything the function can change.
//!
//! The validator compares return values and globals. An accepted transform of
//! a function that writes memory, may trap, or branches (which the main
//! encoder approximates) is kept, but it is not proven, and the counts must
//! say so with the reason. One test per binary: the counters are
//! process-global.
#![cfg(feature = "verification")]

use loom_core::{parse, stats, verify::TranslationValidator};

fn func(wat: &str) -> loom_core::Function {
    let wasm = wat::parse_str(wat).expect("wat");
    let module = parse::parse_wasm(&wasm).expect("parse");
    module.functions[0].clone()
}

fn accepted_reason(wat: &str) -> (u64, Vec<String>) {
    let f = func(wat);
    let _ = stats::take_coverage_summary();
    TranslationValidator::new(&f, "identity")
        .verify(&f)
        .expect("a function is equivalent to itself");
    let s = stats::take_coverage_summary();
    (s.proven, s.kept_unproven.keys().cloned().collect())
}

#[test]
fn only_fully_compared_acceptances_count_as_proven() {
    // Pure arithmetic: everything observable is the return value.
    let (proven, kept) = accepted_reason(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0 local.get 1 i32.add))"#,
    );
    assert_eq!((proven, kept.len()), (1, 0), "pure function: {kept:?}");

    // A store: the final memory is not compared.
    let (proven, kept) = accepted_reason(
        r#"(module (memory 1) (func (param i32) (result i32)
             local.get 0 i32.const 7 i32.store
             local.get 0))"#,
    );
    assert_eq!(proven, 0);
    assert_eq!(kept, vec!["final memory not compared".to_string()]);

    // A division: the trap on a zero divisor is not compared.
    let (proven, kept) = accepted_reason(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0 local.get 1 i32.div_u))"#,
    );
    assert_eq!(proven, 0);
    assert_eq!(kept, vec!["traps not compared".to_string()]);

    // A br_if: the main encoder approximates it.
    let (proven, kept) = accepted_reason(
        r#"(module (func (param i32) (result i32)
             (block (result i32)
               i32.const 1 local.get 0 br_if 0 drop i32.const 2)))"#,
    );
    assert_eq!(proven, 0);
    assert_eq!(kept, vec!["branches approximated".to_string()]);
}
