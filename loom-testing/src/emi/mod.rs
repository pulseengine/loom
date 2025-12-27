//! EMI (Equivalence Modulo Inputs) Testing for WebAssembly
//!
//! This module implements EMI testing for LOOM's optimizer.
//! EMI testing finds miscompilation bugs by mutating dead code
//! and verifying that outputs remain unchanged.
//!
//! # Overview
//!
//! 1. Parse a WebAssembly module
//! 2. Statically analyze for dead code regions (constant branches, unreachable code)
//! 3. Generate variants by mutating dead regions
//! 4. Optimize each variant with LOOM
//! 5. Execute and compare outputs - any difference is a bug
//!
//! # Example
//!
//! ```ignore
//! use loom_testing::emi::{emi_test, EmiConfig};
//!
//! let wasm = wat::parse_str(r#"
//!     (module
//!       (func (export "test") (result i32)
//!         (if (result i32) (i32.const 1)
//!           (then (i32.const 42))
//!           (else (i32.const 99))))  ;; dead branch
//!     )
//! "#)?;
//!
//! let result = emi_test(&wasm, EmiConfig::default())?;
//! assert!(result.bugs_found.is_empty());
//! ```

mod analysis;
mod mutation;
mod types;

pub use analysis::*;
pub use mutation::*;
pub use types::*;

use anyhow::{Context, Result};
use rand::prelude::*;

#[cfg(feature = "runtime")]
use std::process::Command;
#[cfg(feature = "runtime")]
use tempfile::NamedTempFile;
#[cfg(feature = "runtime")]
use wasmtime::{Engine, Instance, Module, Store, Val};

/// Configuration for EMI testing
#[derive(Debug, Clone)]
pub struct EmiConfig {
    /// Number of variants to generate and test
    pub iterations: usize,
    /// Mutation strategies to use
    pub strategies: Vec<MutationStrategy>,
    /// Whether to stop on first bug found
    pub stop_on_first_bug: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Path to LOOM binary (if None, uses library directly)
    pub loom_binary: Option<String>,
}

impl Default for EmiConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            strategies: vec![
                MutationStrategy::ModifyConstants,
                MutationStrategy::ReplaceWithUnreachable,
                MutationStrategy::InsertDeadCode,
            ],
            stop_on_first_bug: false,
            seed: None,
            loom_binary: None,
        }
    }
}

impl EmiConfig {
    /// Create a conservative config (fewer mutations, less likely to cause issues)
    pub fn conservative() -> Self {
        Self {
            iterations: 50,
            strategies: vec![MutationStrategy::ModifyConstants],
            stop_on_first_bug: false,
            seed: None,
            loom_binary: None,
        }
    }

    /// Create an aggressive config (more mutations, more thorough)
    pub fn aggressive() -> Self {
        Self {
            iterations: 200,
            strategies: vec![
                MutationStrategy::ModifyConstants,
                MutationStrategy::ReplaceWithUnreachable,
                MutationStrategy::ReplaceWithNop,
                MutationStrategy::InsertDeadCode,
            ],
            stop_on_first_bug: false,
            seed: None,
            loom_binary: None,
        }
    }
}

/// Run EMI testing on a WebAssembly module (requires "runtime" feature)
///
/// # Arguments
/// * `wasm_bytes` - The WebAssembly binary to test
/// * `config` - EMI testing configuration
///
/// # Returns
/// * `EmiTestResult` containing analysis results and any bugs found
#[cfg(feature = "runtime")]
pub fn emi_test(wasm_bytes: &[u8], config: EmiConfig) -> Result<EmiTestResult> {
    let engine = Engine::default();

    // Set up RNG
    let mut rng: Box<dyn RngCore> = match config.seed {
        Some(seed) => Box::new(rand::rngs::StdRng::seed_from_u64(seed)),
        None => Box::new(rand::thread_rng()),
    };

    // 1. Analyze for dead code regions
    let dead_regions = analyze_dead_code(wasm_bytes)?;

    if dead_regions.is_empty() {
        return Ok(EmiTestResult {
            dead_regions_found: 0,
            variants_tested: 0,
            bugs_found: vec![],
            analysis_details: vec!["No dead code regions found".to_string()],
        });
    }

    // 2. Get original module's exported functions and their expected outputs
    let original_outputs = collect_function_outputs(&engine, wasm_bytes)?;

    // 3. Generate and test variants
    let mut bugs_found = Vec::new();
    let mut variants_tested = 0;

    for i in 0..config.iterations {
        // Pick random dead region and mutation strategy
        let region_idx = rng.gen_range(0..dead_regions.len());
        let region = &dead_regions[region_idx];

        let strategy_idx = rng.gen_range(0..config.strategies.len());
        let strategy = config.strategies[strategy_idx];

        // Apply mutation to create variant
        let variant_bytes = match apply_mutation(wasm_bytes, region, strategy) {
            Ok(bytes) => bytes,
            Err(_e) => {
                // Mutation failed - skip this variant
                continue;
            }
        };

        // Validate variant is still valid Wasm
        if wasmparser::validate(&variant_bytes).is_err() {
            // Invalid variant - mutation broke the module structure
            // This might indicate a bug in our mutation logic
            continue;
        }

        // 4. Optimize the variant with LOOM
        let optimized_bytes = match optimize_with_loom(&variant_bytes, &config.loom_binary) {
            Ok(bytes) => bytes,
            Err(e) => {
                // Optimization failed - this could be a bug!
                bugs_found.push(EmiBug {
                    variant_id: i,
                    mutation_strategy: strategy,
                    dead_region: region.clone(),
                    bug_type: EmiBugType::OptimizationCrash(e.to_string()),
                    expected: None,
                    actual: None,
                });

                if config.stop_on_first_bug {
                    break;
                }
                continue;
            }
        };

        // Validate optimized output is valid Wasm
        if let Err(e) = wasmparser::validate(&optimized_bytes) {
            bugs_found.push(EmiBug {
                variant_id: i,
                mutation_strategy: strategy,
                dead_region: region.clone(),
                bug_type: EmiBugType::InvalidOutput(e.to_string()),
                expected: None,
                actual: None,
            });

            if config.stop_on_first_bug {
                break;
            }
            continue;
        }

        // 5. Execute optimized variant and compare outputs
        let variant_outputs = match collect_function_outputs(&engine, &optimized_bytes) {
            Ok(outputs) => outputs,
            Err(e) => {
                bugs_found.push(EmiBug {
                    variant_id: i,
                    mutation_strategy: strategy,
                    dead_region: region.clone(),
                    bug_type: EmiBugType::ExecutionError(e.to_string()),
                    expected: None,
                    actual: None,
                });

                if config.stop_on_first_bug {
                    break;
                }
                continue;
            }
        };

        // Compare outputs
        for (func_name, expected) in &original_outputs {
            if let Some(actual) = variant_outputs.get(func_name) {
                if !values_equal(expected, actual) {
                    bugs_found.push(EmiBug {
                        variant_id: i,
                        mutation_strategy: strategy,
                        dead_region: region.clone(),
                        bug_type: EmiBugType::OutputMismatch {
                            function: func_name.clone(),
                        },
                        expected: Some(format!("{:?}", expected)),
                        actual: Some(format!("{:?}", actual)),
                    });

                    if config.stop_on_first_bug {
                        break;
                    }
                }
            }
        }

        variants_tested += 1;

        if config.stop_on_first_bug && !bugs_found.is_empty() {
            break;
        }
    }

    Ok(EmiTestResult {
        dead_regions_found: dead_regions.len(),
        variants_tested,
        bugs_found,
        analysis_details: dead_regions.iter().map(|r| format!("{:?}", r)).collect(),
    })
}

/// Run EMI testing without runtime execution (validation only)
///
/// This version works without wasmtime and only validates that
/// optimized modules are valid WebAssembly. It cannot detect
/// semantic bugs (wrong output values).
#[cfg(not(feature = "runtime"))]
pub fn emi_test(wasm_bytes: &[u8], config: EmiConfig) -> Result<EmiTestResult> {
    // Set up RNG
    let mut rng: Box<dyn RngCore> = match config.seed {
        Some(seed) => Box::new(rand::rngs::StdRng::seed_from_u64(seed)),
        None => Box::new(rand::thread_rng()),
    };

    // 1. Analyze for dead code regions
    let dead_regions = analyze_dead_code(wasm_bytes)?;

    if dead_regions.is_empty() {
        return Ok(EmiTestResult {
            dead_regions_found: 0,
            variants_tested: 0,
            bugs_found: vec![],
            analysis_details: vec!["No dead code regions found".to_string()],
        });
    }

    // 3. Generate and test variants (validation only - no execution)
    let mut bugs_found = Vec::new();
    let mut variants_tested = 0;

    for i in 0..config.iterations {
        // Pick random dead region and mutation strategy
        let region_idx = rng.gen_range(0..dead_regions.len());
        let region = &dead_regions[region_idx];

        let strategy_idx = rng.gen_range(0..config.strategies.len());
        let strategy = config.strategies[strategy_idx];

        // Apply mutation to create variant
        let variant_bytes = match apply_mutation(wasm_bytes, region, strategy) {
            Ok(bytes) => bytes,
            Err(_e) => {
                // Mutation failed - skip this variant
                continue;
            }
        };

        // Validate variant is still valid Wasm
        if wasmparser::validate(&variant_bytes).is_err() {
            continue;
        }

        // 4. Optimize the variant with LOOM (library only, no external binary support)
        let optimized_bytes = match optimize_with_loom_library(&variant_bytes) {
            Ok(bytes) => bytes,
            Err(e) => {
                bugs_found.push(EmiBug {
                    variant_id: i,
                    mutation_strategy: strategy,
                    dead_region: region.clone(),
                    bug_type: EmiBugType::OptimizationCrash(e.to_string()),
                    expected: None,
                    actual: None,
                });

                if config.stop_on_first_bug {
                    break;
                }
                continue;
            }
        };

        // Validate optimized output is valid Wasm
        if let Err(e) = wasmparser::validate(&optimized_bytes) {
            bugs_found.push(EmiBug {
                variant_id: i,
                mutation_strategy: strategy,
                dead_region: region.clone(),
                bug_type: EmiBugType::InvalidOutput(e.to_string()),
                expected: None,
                actual: None,
            });

            if config.stop_on_first_bug {
                break;
            }
            continue;
        }

        variants_tested += 1;

        if config.stop_on_first_bug && !bugs_found.is_empty() {
            break;
        }
    }

    Ok(EmiTestResult {
        dead_regions_found: dead_regions.len(),
        variants_tested,
        bugs_found,
        analysis_details: dead_regions.iter().map(|r| format!("{:?}", r)).collect(),
    })
}

/// Collect outputs from all exported functions that take no arguments
#[cfg(feature = "runtime")]
fn collect_function_outputs(
    engine: &Engine,
    wasm_bytes: &[u8],
) -> Result<std::collections::HashMap<String, Vec<Val>>> {
    let module = Module::new(engine, wasm_bytes).context("Failed to compile module")?;
    let mut store = Store::new(engine, ());
    let instance = Instance::new(&mut store, &module, &[]).context("Failed to instantiate")?;

    let mut outputs = std::collections::HashMap::new();

    for export in module.exports() {
        if let wasmtime::ExternType::Func(func_type) = export.ty() {
            // Only test functions with no parameters (for simplicity)
            if func_type.params().len() == 0 {
                let func = instance
                    .get_func(&mut store, export.name())
                    .context("Failed to get function")?;

                let result_count = func_type.results().len();
                let mut results = vec![Val::I32(0); result_count];

                match func.call(&mut store, &[], &mut results) {
                    Ok(()) => {
                        outputs.insert(export.name().to_string(), results);
                    }
                    Err(_) => {
                        // Function trapped - skip it
                    }
                }
            }
        }
    }

    Ok(outputs)
}

/// Compare two value vectors for equality
#[cfg(feature = "runtime")]
fn values_equal(a: &[Val], b: &[Val]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    a.iter().zip(b.iter()).all(|(va, vb)| match (va, vb) {
        (Val::I32(a), Val::I32(b)) => a == b,
        (Val::I64(a), Val::I64(b)) => a == b,
        (Val::F32(a), Val::F32(b)) => a == b,
        (Val::F64(a), Val::F64(b)) => a == b,
        _ => false,
    })
}

/// Optimize Wasm bytes using LOOM (with optional external binary)
#[cfg(feature = "runtime")]
fn optimize_with_loom(wasm_bytes: &[u8], loom_binary: &Option<String>) -> Result<Vec<u8>> {
    match loom_binary {
        Some(binary_path) => {
            // Use external binary
            let temp_in = NamedTempFile::new()?;
            std::fs::write(temp_in.path(), wasm_bytes)?;

            let temp_out = NamedTempFile::new()?;

            let output = Command::new(binary_path)
                .arg("optimize")
                .arg(temp_in.path())
                .arg("-o")
                .arg(temp_out.path())
                .output()
                .context("Failed to execute LOOM")?;

            if !output.status.success() {
                anyhow::bail!("LOOM failed: {}", String::from_utf8_lossy(&output.stderr));
            }

            Ok(std::fs::read(temp_out.path())?)
        }
        None => optimize_with_loom_library(wasm_bytes),
    }
}

/// Optimize Wasm bytes using LOOM library directly
fn optimize_with_loom_library(wasm_bytes: &[u8]) -> Result<Vec<u8>> {
    let mut module = loom_core::parse::parse_wasm(wasm_bytes)
        .context("Failed to parse WASM for LOOM optimization")?;
    loom_core::optimize::optimize_module(&mut module).context("LOOM optimization failed")?;
    loom_core::encode::encode_wasm(&module).context("Failed to encode LOOM optimized module")
}

#[cfg(all(test, feature = "runtime"))]
mod tests {
    use super::*;

    #[test]
    fn test_emi_simple_module() {
        let wat = r#"
            (module
              (func (export "always_42") (result i32)
                (if (result i32) (i32.const 1)
                  (then (i32.const 42))
                  (else (i32.const 99))))
            )
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let config = EmiConfig::conservative();

        let result = emi_test(&wasm, config).expect("EMI test failed");

        println!("Dead regions found: {}", result.dead_regions_found);
        println!("Variants tested: {}", result.variants_tested);
        println!("Bugs found: {}", result.bugs_found.len());

        // Should find at least one dead region (the else branch)
        assert!(result.dead_regions_found > 0, "Should find dead regions");
    }
}
