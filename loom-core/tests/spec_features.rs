//! Spec-feature coverage tests: post-MVP WebAssembly features.
//!
//! v0.4.0 audit found LOOM's test corpus had zero fixtures using post-MVP
//! features (SIMD, ref types, GC, tail calls, EH, threads, multi-memory).
//! The parser is expected to *reject* unsupported instructions cleanly
//! (return `Err`, never panic). Where LOOM does support the feature, we
//! exercise the full optimize + round-trip path.
//!
//! Per CLAUDE.md: rejection paths matter as much as happy paths. We assert
//! every fixture (a) does not panic the parser, and (b) either succeeds
//! end-to-end or fails with a recognizable diagnostic.

use loom_core::{encode, optimize, parse};
use std::panic;

/// Outcome bucket for a single fixture run.
#[derive(Debug)]
enum Outcome {
    /// Parser panicked — should never happen.
    Panicked,
    /// Parser returned a clean error.
    Rejected(String),
    /// Parser succeeded; full module is available.
    Accepted(Box<loom_core::Module>),
}

fn classify(wat: &str) -> Outcome {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| parse::parse_wat(wat)));
    match result {
        Err(_) => Outcome::Panicked,
        Ok(Ok(module)) => Outcome::Accepted(Box::new(module)),
        Ok(Err(e)) => Outcome::Rejected(format!("{e:#}")),
    }
}

/// For features LOOM does not support, we require:
///   * no panic
///   * a clean Err (or, if the parser unexpectedly accepts, a successful
///     re-encode — anything except a panic is acceptable; we just want to
///     pin the contract).
fn assert_no_panic(name: &str, wat: &str) {
    match classify(wat) {
        Outcome::Panicked => panic!("parser panicked on {name}"),
        Outcome::Rejected(msg) => {
            // Sanity: error string should be non-empty.
            assert!(
                !msg.is_empty(),
                "{name}: rejection produced empty error message"
            );
            eprintln!("{name}: rejected cleanly: {msg}");
        }
        Outcome::Accepted(module) => {
            // Surprise — parser accepted. Make sure encode also doesn't panic.
            let encode_result =
                panic::catch_unwind(panic::AssertUnwindSafe(|| encode::encode_wasm(&module)));
            assert!(
                encode_result.is_ok(),
                "{name}: parser accepted but encoder panicked"
            );
            eprintln!("{name}: parser accepted (likely partial support)");
        }
    }
}

/// For LOOM-supported features, we want full optimize + re-encode round-trip.
fn assert_optimize_roundtrip(name: &str, wat: &str) {
    let outcome = classify(wat);
    let mut module = match outcome {
        Outcome::Panicked => panic!("parser panicked on {name}"),
        Outcome::Rejected(msg) => panic!("{name}: expected acceptance, got rejection: {msg}"),
        Outcome::Accepted(m) => *m,
    };

    let opt_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        optimize::optimize_module(&mut module)
    }));
    assert!(opt_result.is_ok(), "{name}: optimizer panicked");
    opt_result
        .unwrap()
        .unwrap_or_else(|e| panic!("{name}: optimize_module returned Err: {e:#}"));

    let enc_result = panic::catch_unwind(panic::AssertUnwindSafe(|| encode::encode_wasm(&module)));
    assert!(enc_result.is_ok(), "{name}: encoder panicked");
    let bytes = enc_result
        .unwrap()
        .unwrap_or_else(|e| panic!("{name}: encode_wasm returned Err: {e:#}"));

    // Re-parse the encoded bytes to confirm round-trip stability.
    let reparse = panic::catch_unwind(panic::AssertUnwindSafe(|| parse::parse_wasm(&bytes)));
    assert!(reparse.is_ok(), "{name}: re-parser panicked");
    reparse
        .unwrap()
        .unwrap_or_else(|e| panic!("{name}: re-parse failed: {e:#}"));
}

// ---------------------------------------------------------------------------
// Post-MVP features LOOM does NOT yet support — assert clean rejection only.
// ---------------------------------------------------------------------------

#[test]
fn spec_feature_simd_v128_does_not_panic() {
    let wat = include_str!("../../tests/fixtures/spec-features/simd_v128_minimal.wat");
    assert_no_panic("simd_v128", wat);
}

#[test]
fn spec_feature_ref_types_does_not_panic() {
    let wat = include_str!("../../tests/fixtures/spec-features/ref_types_minimal.wat");
    assert_no_panic("ref_types", wat);
}

#[test]
fn spec_feature_tail_calls_does_not_panic() {
    let wat = include_str!("../../tests/fixtures/spec-features/tail_calls_minimal.wat");
    assert_no_panic("tail_calls", wat);
}

#[test]
fn spec_feature_exception_handling_does_not_panic() {
    let wat = include_str!("../../tests/fixtures/spec-features/exception_handling_minimal.wat");
    assert_no_panic("exception_handling", wat);
}

// ---------------------------------------------------------------------------
// Post-MVP features LOOM partially supports — accept either rejection or
// successful round-trip; never panic.
// ---------------------------------------------------------------------------

#[test]
fn spec_feature_bulk_memory_does_not_panic() {
    let wat = include_str!("../../tests/fixtures/spec-features/bulk_memory_minimal.wat");
    assert_no_panic("bulk_memory", wat);
}

#[test]
fn spec_feature_multi_memory_does_not_panic() {
    let wat = include_str!("../../tests/fixtures/spec-features/multi_memory_minimal.wat");
    assert_no_panic("multi_memory", wat);
}

// ---------------------------------------------------------------------------
// Standardized features LOOM fully supports — full optimize + round-trip.
// ---------------------------------------------------------------------------

#[test]
fn spec_feature_sign_extension_ops_round_trip() {
    let wat = include_str!("../../tests/fixtures/spec-features/sign_extension_ops.wat");
    assert_optimize_roundtrip("sign_extension_ops", wat);
}

#[test]
fn spec_feature_saturating_trunc_round_trip() {
    let wat = include_str!("../../tests/fixtures/spec-features/saturating_trunc.wat");
    assert_optimize_roundtrip("saturating_trunc", wat);
}
