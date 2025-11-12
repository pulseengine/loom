//! Integration tests for WebAssembly I/O
//! Phase 2: Testing with fixture files

use loom_core::{encode, parse, Module};
use std::fs;

#[test]
fn test_parse_simple_module_fixture() {
    let wat_content =
        fs::read_to_string("tests/fixtures/simple_module.wat").expect("Failed to read fixture");

    let module = parse::parse_wat(&wat_content).expect("Failed to parse fixture WAT");

    assert_eq!(module.functions.len(), 1);
    let func = &module.functions[0];
    assert_eq!(func.signature.params.len(), 0);
    assert_eq!(func.signature.results.len(), 1);
}

#[test]
fn test_parse_test_input_fixture() {
    let wat_content =
        fs::read_to_string("tests/fixtures/test_input.wat").expect("Failed to read fixture");

    let module = parse::parse_wat(&wat_content).expect("Failed to parse test input");

    assert_eq!(module.functions.len(), 1);
    let func = &module.functions[0];

    // Should have i32.const 10, i32.const 32, i32.add
    let has_const_10 = func
        .instructions
        .iter()
        .any(|i| matches!(i, loom_core::Instruction::I32Const(10)));
    let has_const_32 = func
        .instructions
        .iter()
        .any(|i| matches!(i, loom_core::Instruction::I32Const(32)));
    let has_add = func
        .instructions
        .iter()
        .any(|i| matches!(i, loom_core::Instruction::I32Add));

    assert!(has_const_10, "Missing i32.const 10");
    assert!(has_const_32, "Missing i32.const 32");
    assert!(has_add, "Missing i32.add");
}

#[test]
fn test_round_trip_with_fixture() {
    let wat_content =
        fs::read_to_string("tests/fixtures/simple_module.wat").expect("Failed to read fixture");

    // Parse WAT -> Module
    let module1 = parse::parse_wat(&wat_content).expect("Failed to parse WAT");

    // Encode Module -> WASM
    let wasm_bytes = encode::encode_wasm(&module1).expect("Failed to encode WASM");

    // Parse WASM -> Module
    let module2 = parse::parse_wasm(&wasm_bytes).expect("Failed to parse WASM");

    // Verify modules match
    assert_eq!(module1.functions.len(), module2.functions.len());
    assert_eq!(
        module1.functions[0].signature.params.len(),
        module2.functions[0].signature.params.len()
    );
    assert_eq!(
        module1.functions[0].signature.results.len(),
        module2.functions[0].signature.results.len()
    );
}
