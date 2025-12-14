//! EMI integration tests
//!
//! These tests run EMI testing on the test fixtures to verify
//! that LOOM's optimizer doesn't have miscompilation bugs.

use loom_testing::emi::{analyze_dead_code, emi_test, EmiConfig};

/// Load a fixture as WASM bytes
fn load_fixture(name: &str) -> Vec<u8> {
    let path = format!("../tests/fixtures/{}", name);
    let content = std::fs::read(&path).expect(&format!("Failed to read fixture: {}", path));

    if name.ends_with(".wat") {
        wat::parse_bytes(&content)
            .expect(&format!("Failed to parse WAT: {}", path))
            .into_owned()
    } else {
        content
    }
}

/// This test currently finds real bugs in LOOM's optimizer.
/// The bugs are tracked and will be fixed as part of stack analysis improvements.
#[test]
fn test_emi_branch_simplification() {
    let wasm = load_fixture("branch_simplification_test.wat");

    // Should find dead regions
    let regions = analyze_dead_code(&wasm).expect("Analysis failed");
    assert!(
        !regions.is_empty(),
        "Should find dead regions in branch_simplification_test.wat"
    );

    // Run EMI test with conservative settings
    let config = EmiConfig {
        iterations: 20,
        ..EmiConfig::conservative()
    };

    let result = emi_test(&wasm, config).expect("EMI test failed");

    println!("Dead regions: {}", result.dead_regions_found);
    println!("Variants tested: {}", result.variants_tested);
    println!("Bugs found: {}", result.bugs_found.len());

    // EMI should find no bugs - optimizer has been fixed to handle dead code correctly
    assert!(
        result.bugs_found.is_empty(),
        "EMI found {} bugs in branch_simplification_test.wat",
        result.bugs_found.len()
    );
}

#[test]
fn test_emi_dce_test() {
    let wasm = load_fixture("dce_test.wat");

    let config = EmiConfig {
        iterations: 20,
        ..EmiConfig::conservative()
    };

    let result = emi_test(&wasm, config).expect("EMI test failed");

    assert!(
        result.bugs_found.is_empty(),
        "Found {} EMI bugs in dce_test.wat",
        result.bugs_found.len()
    );
}

#[test]
fn test_emi_vacuum_test() {
    let wasm = load_fixture("vacuum_test.wat");

    let config = EmiConfig {
        iterations: 20,
        ..EmiConfig::conservative()
    };

    let result = emi_test(&wasm, config).expect("EMI test failed");

    assert!(
        result.bugs_found.is_empty(),
        "Found {} EMI bugs in vacuum_test.wat",
        result.bugs_found.len()
    );
}

#[test]
fn test_emi_simplify_locals() {
    let wasm = load_fixture("simplify_locals_test.wat");

    let config = EmiConfig {
        iterations: 20,
        ..EmiConfig::conservative()
    };

    let result = emi_test(&wasm, config).expect("EMI test failed");

    assert!(
        result.bugs_found.is_empty(),
        "Found {} EMI bugs in simplify_locals_test.wat",
        result.bugs_found.len()
    );
}

/// Test that all fixtures can be analyzed for dead code
#[test]
fn test_analyze_all_fixtures() {
    let fixtures = [
        "branch_simplification_test.wat",
        "dce_test.wat",
        "vacuum_test.wat",
        "simplify_locals_test.wat",
        "fibonacci.wat",
        "quicksort.wat",
    ];

    for fixture in fixtures {
        let wasm = load_fixture(fixture);
        let result = analyze_dead_code(&wasm);
        assert!(
            result.is_ok(),
            "Failed to analyze {}: {:?}",
            fixture,
            result.err()
        );
    }
}
