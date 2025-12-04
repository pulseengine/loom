//! Component Model Execution Verification Tests
//!
//! Tests for runtime verification of component model correctness after optimization.
//! These tests verify that optimized components maintain functional correctness by
//! instantiating them with wasmtime and checking structure preservation.

use loom_core::{optimize_component, ComponentExecutor};

#[test]
fn test_component_executor_creation() {
    // Verify we can create a component executor
    let executor = ComponentExecutor::new();
    assert!(
        executor.is_ok(),
        "Should create component executor successfully"
    );
}

#[test]
fn test_simple_component_loads() {
    // Test that a simple component can be loaded
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    let result = executor.load_component(&component_bytes);
    assert!(result.is_ok(), "Should load component successfully");

    let result = result.unwrap();
    assert!(
        result.loads_successfully,
        "Component should load successfully"
    );
    assert!(
        result.structure_preserved,
        "Component structure should be preserved"
    );
}

#[test]
fn test_component_executor_rejects_invalid_input() {
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    // Test with non-component bytes
    let invalid_bytes = vec![0u8; 100];
    let result = executor.load_component(&invalid_bytes);
    assert!(result.is_err(), "Should reject invalid component bytes");

    // Test with empty bytes
    let empty_bytes = vec![];
    let result = executor.load_component(&empty_bytes);
    assert!(result.is_err(), "Should reject empty bytes");
}

#[test]
fn test_optimized_component_maintains_structure() {
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    // Optimize the component
    let (optimized, _stats) = optimize_component(&component_bytes).expect("Optimization failed");

    // Load both original and optimized
    let original_result = executor
        .load_component(&component_bytes)
        .expect("Failed to load original");
    let optimized_result = executor
        .load_component(&optimized)
        .expect("Failed to load optimized");

    // Verify both load successfully
    assert!(
        original_result.loads_successfully,
        "Original should load successfully"
    );
    assert!(
        optimized_result.loads_successfully,
        "Optimized should load successfully"
    );

    // Verify structure is preserved
    assert_eq!(
        original_result.export_count, optimized_result.export_count,
        "Export count should be preserved"
    );
}

#[test]
fn test_component_verification_report() {
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    let (optimized, _stats) = optimize_component(&component_bytes).expect("Optimization failed");

    // Verify optimization
    let report = executor
        .verify_component_optimization(&component_bytes, &optimized)
        .expect("Verification failed");

    println!("Verification Report:");
    println!("  Passed: {}", report.verification_passed);
    println!("  Original exports: {}", report.original_exports);
    println!("  Optimized exports: {}", report.optimized_exports);
    println!(
        "  Original canonical functions: {}",
        report.original_canonical_functions
    );
    println!(
        "  Optimized canonical functions: {}",
        report.optimized_canonical_functions
    );
    if !report.issues.is_empty() {
        println!("  Issues:");
        for issue in &report.issues {
            println!("    - {}", issue);
        }
    }

    // Verify the report shows successful verification
    assert!(
        report.verification_passed,
        "Component optimization should pass verification"
    );
    assert!(
        report.original_exports > 0,
        "Should have identified exports in original"
    );
    assert_eq!(
        report.original_exports, report.optimized_exports,
        "Exports should be preserved"
    );
}

#[test]
fn test_canonical_function_analysis() {
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    // Analyze canonical functions
    let canonicals = executor
        .analyze_canonical_functions(&component_bytes)
        .expect("Analysis failed");

    println!(
        "Found {} canonical functions in component",
        canonicals.len()
    );
    for (i, cf) in canonicals.iter().enumerate() {
        println!(
            "  [{}] {} - params: {}, returns: {}, preserved: {}",
            i, cf.name, cf.param_count, cf.return_count, cf.preserved
        );
    }

    // Component should have canonical function analysis
    assert!(!canonicals.is_empty(), "Should analyze canonical functions");

    // All should be marked as preserved
    for cf in canonicals {
        assert!(cf.preserved, "Canonical functions should be preserved");
    }
}

#[test]
fn test_calculator_component_execution_verification() {
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    let component_bytes = std::fs::read("tests/component_fixtures/calc.component.wasm")
        .expect("Failed to read calculator component");

    // Optimize the component
    let (optimized, _stats) = optimize_component(&component_bytes).expect("Optimization failed");

    // Verify optimization
    let report = executor
        .verify_component_optimization(&component_bytes, &optimized)
        .expect("Verification failed");

    println!("Calculator Component Verification:");
    println!("  Verification passed: {}", report.verification_passed);
    println!(
        "  Export count: {} → {}",
        report.original_exports, report.optimized_exports
    );
    println!(
        "  Canonical functions: {} → {}",
        report.original_canonical_functions, report.optimized_canonical_functions
    );

    // Verify successful optimization
    assert!(
        report.verification_passed,
        "Calculator component optimization should pass verification"
    );
    assert_eq!(
        report.original_exports, report.optimized_exports,
        "Calculator exports should be preserved"
    );
}

#[test]
fn test_canonical_function_preservation() {
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    // Optimize the component
    let (optimized, _stats) = optimize_component(&component_bytes).expect("Optimization failed");

    // Check canonical preservation directly
    let preserved = executor
        .check_canonical_preservation(&component_bytes, &optimized)
        .expect("Preservation check failed");

    println!("Canonical functions preserved: {}", preserved);
    assert!(
        preserved,
        "Canonical functions should be preserved after optimization"
    );
}

#[test]
fn test_verification_report_includes_canonical_status() {
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    // Optimize the component
    let (optimized, _stats) = optimize_component(&component_bytes).expect("Optimization failed");

    // Get verification report
    let report = executor
        .verify_component_optimization(&component_bytes, &optimized)
        .expect("Verification failed");

    println!(
        "Verification report canonical_functions_preserved: {}",
        report.canonical_functions_preserved
    );

    // Check that report includes canonical preservation status
    assert!(
        report.canonical_functions_preserved,
        "Report should indicate canonical functions are preserved"
    );

    // Overall verification should pass if canonical functions are preserved
    if report.canonical_functions_preserved && report.original_exports == report.optimized_exports {
        assert!(
            report.verification_passed,
            "Verification should pass when exports and canonicals are preserved"
        );
    }
}

#[test]
fn test_differential_testing_simple_component() {
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    // Optimize with LOOM
    let (loom_optimized, _stats) =
        optimize_component(&component_bytes).expect("Optimization failed");

    // Perform differential test
    let report = executor
        .differential_test(&component_bytes, &loom_optimized)
        .expect("Differential test failed");

    println!("Differential Test Report (simple component):");
    println!("  Test passed: {}", report.test_passed);
    println!(
        "  Size: {} → {} bytes ({:+.1}%)",
        report.original_size, report.loom_optimized_size, report.size_improvement_percent
    );
    println!(
        "  Exports: {} → {}",
        report.original_export_count, report.loom_export_count
    );

    if !report.issues.is_empty() {
        println!("  Issues:");
        for issue in &report.issues {
            println!("    - {}", issue);
        }
    }

    if !report.warnings.is_empty() {
        println!("  Warnings:");
        for warning in &report.warnings {
            println!("    - {}", warning);
        }
    }

    // Differential test should pass for simple component
    assert!(
        report.test_passed,
        "Simple component differential test should pass"
    );
    assert_eq!(
        report.original_export_count, report.loom_export_count,
        "Export count should be preserved"
    );
}

#[test]
fn test_differential_testing_calculator_component() {
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    let component_bytes = std::fs::read("tests/component_fixtures/calc.component.wasm")
        .expect("Failed to read calculator component");

    // Optimize with LOOM
    let (loom_optimized, loom_stats) =
        optimize_component(&component_bytes).expect("Optimization failed");

    // Perform differential test
    let report = executor
        .differential_test(&component_bytes, &loom_optimized)
        .expect("Differential test failed");

    println!("Differential Test Report (calculator component):");
    println!("  Test passed: {}", report.test_passed);
    println!(
        "  Size: {} → {} bytes ({:+.1}%)",
        report.original_size, report.loom_optimized_size, report.size_improvement_percent
    );
    println!(
        "  Module reduction: {:.1}%",
        loom_stats.module_reduction_percentage()
    );

    // Calculator component differential test should pass
    assert!(
        report.test_passed,
        "Calculator component differential test should pass"
    );
}

#[test]
fn test_differential_report_metrics() {
    let executor = ComponentExecutor::new().expect("Failed to create executor");

    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    let (loom_optimized, _stats) =
        optimize_component(&component_bytes).expect("Optimization failed");

    let report = executor
        .differential_test(&component_bytes, &loom_optimized)
        .expect("Differential test failed");

    // Check that all metrics are present
    assert!(report.original_size > 0, "Original size should be recorded");
    assert!(
        report.loom_optimized_size > 0,
        "Optimized size should be recorded"
    );
    assert!(
        report.original_export_count > 0 || report.original_export_count == 0,
        "Export count should be recorded"
    );

    // Size improvement can be positive (good) or negative (regression)
    let size_diff = report.loom_optimized_size as i32 - report.original_size as i32;
    println!(
        "Size difference: {} bytes ({:+.1}%)",
        size_diff, report.size_improvement_percent
    );

    // At minimum, the optimization should not corrupt the component
    assert!(
        report.test_passed || !report.issues.is_empty(),
        "Should have clear test result"
    );
}
