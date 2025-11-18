//! Component Model Optimization Tests
//!
//! Tests for WebAssembly Component Model support.
//! LOOM is the first optimizer to support the Component Model.

use loom_core::{analyze_component_structure, optimize_component, ComponentStats};

#[test]
fn test_simple_component_optimization() {
    // Load test component
    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    // Optimize
    let (optimized, stats) = optimize_component(&component_bytes).expect("Optimization failed");

    // Verify stats
    assert_eq!(stats.module_count, 1, "Should have 1 core module");
    assert_eq!(stats.modules_optimized, 1, "Should have optimized 1 module");

    // Verify size reduction
    assert!(
        optimized.len() < component_bytes.len(),
        "Optimized component should be smaller"
    );
    assert!(
        stats.reduction_percentage() > 0.0,
        "Should have some reduction"
    );

    println!(
        "Component: {} → {} bytes",
        stats.original_size, stats.optimized_size
    );
    println!("Reduction: {:.1}%", stats.reduction_percentage());
    println!(
        "Module reduction: {:.1}%",
        stats.module_reduction_percentage()
    );

    // Validate the optimized component
    wasmparser::validate(&optimized).expect("Optimized component should be valid");
}

#[test]
fn test_component_preserves_interface() {
    // Load test component
    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    // Optimize
    let (optimized, _stats) = optimize_component(&component_bytes).expect("Optimization failed");

    // Both should parse as components
    assert!(
        is_component(&component_bytes),
        "Input should be a component"
    );
    assert!(
        is_component(&optimized),
        "Output should still be a component"
    );

    // Validate both
    wasmparser::validate(&component_bytes).expect("Original should be valid");
    wasmparser::validate(&optimized).expect("Optimized should be valid");
}

#[test]
fn test_component_stats_calculations() {
    let stats = ComponentStats {
        original_size: 1000,
        optimized_size: 200,
        module_count: 2,
        modules_optimized: 2,
        original_module_size: 800,
        optimized_module_size: 100,
        message: "Test".to_string(),
    };

    // Test reduction percentage
    assert_eq!(stats.reduction_percentage(), 80.0);
    assert_eq!(stats.module_reduction_percentage(), 87.5);
}

#[test]
fn test_component_error_handling() {
    // Not a component (just raw bytes)
    let not_component = vec![0u8; 100];
    let result = optimize_component(&not_component);
    assert!(result.is_err(), "Should fail on non-component input");

    // Empty component (just header, no modules)
    let empty_component = vec![
        0x00, 0x61, 0x73, 0x6D, // magic
        0x0D, 0x00, // version
        0x01, 0x00, // layer (component)
    ];
    let result = optimize_component(&empty_component);
    assert!(
        result.is_err(),
        "Should fail on component with no core modules"
    );
}

/// Helper: Check if bytes represent a component
fn is_component(bytes: &[u8]) -> bool {
    if bytes.len() < 8 {
        return false;
    }
    // Magic number + version + layer check
    &bytes[0..4] == b"\0asm" && bytes[4] == 0x0d && bytes[6] == 0x01
}

#[test]
fn test_component_analysis() {
    // Phase 2: Test component structure analysis
    let component_bytes = std::fs::read("tests/component_fixtures/simple.component.wasm")
        .expect("Failed to read test component");

    // Analyze component structure
    let analysis = analyze_component_structure(&component_bytes).expect("Analysis failed");

    println!("Component Analysis:");
    println!("  Core modules: {}", analysis.core_module_count);
    println!("  Component types: {}", analysis.component_type_count);
    println!("  Imports: {}", analysis.import_count);
    println!("  Exports: {}", analysis.export_count);
    println!(
        "  Canonical functions: {}",
        analysis.canonical_function_count
    );
    println!("  Instances: {}", analysis.instance_count);
    println!("  Aliases: {}", analysis.alias_count);
    println!("  Nested components: {}", analysis.nested_component_count);

    // Verify expected structure for simple component
    assert_eq!(analysis.core_module_count, 1, "Should have 1 core module");
    assert!(analysis.export_count > 0, "Should have exports");
    assert!(
        analysis.canonical_function_count > 0,
        "Should have canonical functions"
    );
}

#[test]
fn test_calculator_component_optimization() {
    // Test with a more complex component (4 exported functions)
    let component_bytes = std::fs::read("tests/component_fixtures/calc.component.wasm")
        .expect("Failed to read calculator component");

    println!("Original component size: {} bytes", component_bytes.len());

    // Optimize
    let (optimized, stats) = optimize_component(&component_bytes).expect("Optimization failed");

    println!("Optimized component size: {} bytes", optimized.len());
    println!("Overall reduction: {:.1}%", stats.reduction_percentage());
    println!(
        "Module reduction: {:.1}%",
        stats.module_reduction_percentage()
    );

    // Verify optimization occurred
    assert!(
        optimized.len() < component_bytes.len(),
        "Should optimize calculator component"
    );
    assert!(
        stats.reduction_percentage() > 0.0,
        "Should have measurable reduction"
    );

    // Verify it's still a valid component
    wasmparser::validate(&optimized).expect("Optimized component should be valid");
    assert!(
        is_component(&optimized),
        "Output should still be a component"
    );

    // Analyze the structure
    let analysis = analyze_component_structure(&optimized)
        .expect("Should be able to analyze optimized component");

    // Should preserve structure
    assert_eq!(
        analysis.core_module_count, 1,
        "Should still have 1 core module"
    );
    assert!(analysis.export_count > 0, "Should have exports");
    assert!(
        analysis.canonical_function_count > 0,
        "Should have canonical functions"
    );

    println!(
        "Calculator component structure: {} modules, {} exports, {} canonical functions",
        analysis.core_module_count, analysis.export_count, analysis.canonical_function_count
    );
}
