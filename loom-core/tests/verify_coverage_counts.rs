//! `--stats` counts an accepted transform as "proven" only when the validator
//! compared everything the function can change.
//!
//! Functions are decided by the exact executor, which compares results,
//! globals, memory and traps (loops one iteration at a time), so their
//! acceptances are proven. A body over the solver size threshold is kept
//! without a check, and the counts say so with the reason. One test per binary: the counters are
//! process-global.
#![cfg(feature = "verification")]

use loom_core::{parse, stats, verify::TranslationValidator, verify::VerificationSignatureContext};

fn accepted_counts(wat: &str) -> (u64, Vec<String>) {
    let module = parse::parse_wat(wat).expect("parse");
    let f = module.functions[0].clone();
    let ctx = VerificationSignatureContext::from_module(&module);
    let _ = stats::take_coverage_summary();
    TranslationValidator::new_with_context(&f, "identity", ctx)
        .verify(&f)
        .expect("a function is equivalent to itself");
    let s = stats::take_coverage_summary();
    (s.proven, s.kept_unproven.keys().cloned().collect())
}

#[test]
fn only_fully_compared_acceptances_count_as_proven() {
    for wat in [
        // Pure arithmetic.
        r#"(module (func (param i32 i32) (result i32)
             local.get 0 local.get 1 i32.add))"#,
        // A store: memory is compared.
        r#"(module (memory 1) (func (param i32) (result i32)
             local.get 0 i32.const 7 i32.store
             local.get 0))"#,
        // A division: its trap is compared.
        r#"(module (func (param i32 i32) (result i32)
             local.get 0 local.get 1 i32.div_u))"#,
        // A br_if carrying a value: modelled exactly.
        r#"(module (func (param i32) (result i32)
             (block (result i32)
               i32.const 1 local.get 0 br_if 0 drop i32.const 2)))"#,
    ] {
        let (proven, kept) = accepted_counts(wat);
        assert_eq!((proven, kept.len()), (1, 0), "{wat}: kept {kept:?}");
    }

    // A loop is proven too: against itself, one iteration at a time.
    let (proven, kept) = accepted_counts(
        r#"(module (func (param i32) (result i32) (local i32)
             (loop
               local.get 1 i32.const 1 i32.add local.set 1
               local.get 1 local.get 0 i32.lt_u br_if 0)
             local.get 1))"#,
    );
    assert_eq!((proven, kept.len()), (1, 0), "loop: kept {kept:?}");

    // A body over the solver size threshold is kept without a check, and
    // the counts say so.
    let big = format!(
        "(module (func (param i32) (result i32) local.get 0 {}))",
        "i32.const 1 i32.add ".repeat(1100)
    );
    let (proven, kept) = accepted_counts(&big);
    assert_eq!(proven, 0);
    assert_eq!(
        kept,
        vec!["body over the solver size threshold".to_string()]
    );
}
