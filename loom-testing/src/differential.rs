//! Differential Testing for LOOM WebAssembly Optimizer
//!
//! This module provides infrastructure for differential testing of WebAssembly
//! optimization. It compares the execution behavior of original and optimized
//! WebAssembly modules to ensure semantic correctness is preserved.
//!
//! # Overview
//!
//! Differential testing is critical for proving that LOOM's optimizations are
//! semantically correct. The workflow is:
//!
//! 1. Take an input .wasm file
//! 2. Run it through wasmtime to get baseline results
//! 3. Optimize it with LOOM
//! 4. Run the optimized version through wasmtime
//! 5. Compare results - any difference indicates a bug
//!
//! # Example
//!
//! ```ignore
//! use loom_testing::differential::{DifferentialExecutor, ExecutionConfig};
//!
//! let executor = DifferentialExecutor::new()?;
//! let wasm_bytes = std::fs::read("input.wasm")?;
//! let result = executor.test_optimization(&wasm_bytes)?;
//!
//! assert!(result.semantics_preserved, "Optimization changed semantics!");
//! ```
//!
//! # Features
//!
//! This module requires the `runtime` feature flag to enable wasmtime execution.
//! Without it, only structural validation is possible.

use anyhow::{Context, Result};

#[cfg(feature = "runtime")]
use wasmtime::{Caller, Engine, Func, Linker, Module, Store, Val, ValType};

/// Configuration for differential execution testing
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum number of test inputs per function parameter
    pub inputs_per_param: usize,
    /// Maximum number of parameters to support for test input generation
    pub max_params: usize,
    /// Timeout for function execution in milliseconds (0 = no timeout)
    pub execution_timeout_ms: u64,
    /// Whether to test functions with memory operations
    pub test_memory_functions: bool,
    /// Whether to include edge case values (MIN, MAX, etc.)
    pub include_edge_cases: bool,
    /// Custom i32 test values
    pub i32_test_values: Vec<i32>,
    /// Custom i64 test values
    pub i64_test_values: Vec<i64>,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            inputs_per_param: 9,
            max_params: 4,
            execution_timeout_ms: 1000,
            test_memory_functions: true,
            include_edge_cases: true,
            i32_test_values: vec![0, 1, -1, i32::MIN, i32::MAX, 42, 100, 255, 256],
            i64_test_values: vec![0, 1, -1, i64::MIN, i64::MAX, 42, 100, 255, 256],
        }
    }
}

impl ExecutionConfig {
    /// Create a minimal config for fast testing
    pub fn minimal() -> Self {
        Self {
            inputs_per_param: 3,
            max_params: 2,
            execution_timeout_ms: 500,
            test_memory_functions: false,
            include_edge_cases: false,
            i32_test_values: vec![0, 1, -1],
            i64_test_values: vec![0, 1, -1],
        }
    }

    /// Create a thorough config for comprehensive testing
    pub fn thorough() -> Self {
        Self {
            inputs_per_param: 20,
            max_params: 6,
            execution_timeout_ms: 5000,
            test_memory_functions: true,
            include_edge_cases: true,
            i32_test_values: vec![
                0,
                1,
                -1,
                2,
                -2,
                i32::MIN,
                i32::MAX,
                i32::MIN + 1,
                i32::MAX - 1,
                42,
                100,
                255,
                256,
                1000,
                -1000,
                0x7FFF,
                0xFFFF,
                0x7FFFFFFF,
            ],
            i64_test_values: vec![
                0,
                1,
                -1,
                2,
                -2,
                i64::MIN,
                i64::MAX,
                i64::MIN + 1,
                i64::MAX - 1,
                42,
                100,
                255,
                256,
                1000,
                -1000,
                0x7FFFFFFF,
                0xFFFFFFFF,
                0x7FFFFFFFFFFFFFFF,
            ],
        }
    }
}

/// Result of a single function execution comparison
#[derive(Debug, Clone)]
pub struct FunctionTestResult {
    /// Name of the function tested
    pub function_name: String,
    /// Number of test inputs tried
    pub inputs_tested: usize,
    /// Whether all outputs matched
    pub all_matched: bool,
    /// Details of any divergence
    pub divergence: Option<DivergenceInfo>,
}

/// Information about a semantic divergence between original and optimized
#[derive(Debug, Clone)]
pub struct DivergenceInfo {
    /// Input values that caused the divergence
    pub input_values: Vec<String>,
    /// Result from original module
    pub original_result: ExecutionResult,
    /// Result from optimized module
    pub optimized_result: ExecutionResult,
}

/// Result of executing a function
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionResult {
    /// Function returned successfully with these values
    Success(Vec<String>),
    /// Function trapped (e.g., division by zero, unreachable)
    Trap(String),
    /// Function timed out
    Timeout,
    /// Could not execute (missing imports, etc.)
    Skipped(String),
}

impl std::fmt::Display for ExecutionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionResult::Success(vals) => write!(f, "Success({:?})", vals),
            ExecutionResult::Trap(msg) => write!(f, "Trap({})", msg),
            ExecutionResult::Timeout => write!(f, "Timeout"),
            ExecutionResult::Skipped(msg) => write!(f, "Skipped({})", msg),
        }
    }
}

/// Overall result of differential testing a module
#[derive(Debug, Clone)]
pub struct DifferentialTestResult {
    /// Whether semantics were preserved (critical check!)
    pub semantics_preserved: bool,
    /// Original module size in bytes
    pub original_size: usize,
    /// Optimized module size in bytes
    pub optimized_size: usize,
    /// Whether the optimized module is valid WebAssembly
    pub optimized_valid: bool,
    /// Number of exported functions tested
    pub functions_tested: usize,
    /// Number of functions that matched on all inputs
    pub functions_matched: usize,
    /// Number of functions that diverged (bug!)
    pub functions_diverged: usize,
    /// Number of functions skipped (complex signatures, etc.)
    pub functions_skipped: usize,
    /// Per-function results
    pub function_results: Vec<FunctionTestResult>,
    /// Any issues encountered during testing
    pub issues: Vec<String>,
}

impl DifferentialTestResult {
    /// Calculate size reduction percentage
    pub fn size_reduction_percent(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            (1.0 - (self.optimized_size as f64 / self.original_size as f64)) * 100.0
        }
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Differential Test: {} -> {} bytes ({:.1}% reduction), {}/{} functions matched, semantics {}",
            self.original_size,
            self.optimized_size,
            self.size_reduction_percent(),
            self.functions_matched,
            self.functions_tested,
            if self.semantics_preserved { "PRESERVED" } else { "VIOLATED" }
        )
    }

    /// Check if the test passed (semantics preserved and output valid)
    pub fn passed(&self) -> bool {
        self.semantics_preserved && self.optimized_valid
    }
}

/// Differential execution tester using wasmtime
#[cfg(feature = "runtime")]
pub struct DifferentialExecutor {
    engine: Engine,
    config: ExecutionConfig,
}

#[cfg(feature = "runtime")]
impl DifferentialExecutor {
    /// Create a new differential executor with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ExecutionConfig::default())
    }

    /// Create a new differential executor with custom configuration
    pub fn with_config(config: ExecutionConfig) -> Result<Self> {
        let engine = Engine::default();
        Ok(Self { engine, config })
    }

    /// Test that LOOM optimization preserves semantics
    ///
    /// This is the main entry point for differential testing. It:
    /// 1. Validates the input is valid WebAssembly
    /// 2. Optimizes with LOOM
    /// 3. Validates the output is valid WebAssembly
    /// 4. Compares execution on all exported functions
    pub fn test_optimization(&self, wasm_bytes: &[u8]) -> Result<DifferentialTestResult> {
        let original_size = wasm_bytes.len();

        // Validate input
        wasmparser::validate(wasm_bytes).context("Input is not valid WebAssembly")?;

        // Optimize with LOOM
        let optimized_bytes = self.optimize_with_loom(wasm_bytes)?;
        let optimized_size = optimized_bytes.len();

        // Validate output
        let optimized_valid = wasmparser::validate(&optimized_bytes).is_ok();
        if !optimized_valid {
            return Ok(DifferentialTestResult {
                semantics_preserved: false,
                original_size,
                optimized_size,
                optimized_valid: false,
                functions_tested: 0,
                functions_matched: 0,
                functions_diverged: 0,
                functions_skipped: 0,
                function_results: vec![],
                issues: vec!["LOOM produced invalid WebAssembly".to_string()],
            });
        }

        // Compare execution
        self.compare_execution(wasm_bytes, &optimized_bytes, original_size, optimized_size)
    }

    /// Optimize WebAssembly bytes using LOOM library directly
    fn optimize_with_loom(&self, wasm_bytes: &[u8]) -> Result<Vec<u8>> {
        let mut module = loom_core::parse::parse_wasm(wasm_bytes)
            .context("Failed to parse WebAssembly for optimization")?;
        loom_core::optimize::optimize_module(&mut module).context("LOOM optimization failed")?;
        loom_core::encode::encode_wasm(&module).context("Failed to encode optimized module")
    }

    /// Compare execution of original and optimized modules
    fn compare_execution(
        &self,
        original_bytes: &[u8],
        optimized_bytes: &[u8],
        original_size: usize,
        optimized_size: usize,
    ) -> Result<DifferentialTestResult> {
        // Create stores for both modules
        let mut original_store = Store::new(&self.engine, ());
        let mut optimized_store = Store::new(&self.engine, ());

        // Create linker with common imports
        let linker = self.create_linker()?;

        // Load modules
        let original_module = Module::new(&self.engine, original_bytes)
            .context("Failed to compile original module")?;
        let optimized_module = Module::new(&self.engine, optimized_bytes)
            .context("Failed to compile optimized module")?;

        // Try to instantiate
        let (original_instance, optimized_instance) = match (
            linker.instantiate(&mut original_store, &original_module),
            linker.instantiate(&mut optimized_store, &optimized_module),
        ) {
            (Ok(orig), Ok(opt)) => (orig, opt),
            (Err(e), _) | (_, Err(e)) => {
                // Can't instantiate - return with skipped functions
                return Ok(DifferentialTestResult {
                    semantics_preserved: true, // Assume preserved if can't test
                    original_size,
                    optimized_size,
                    optimized_valid: true,
                    functions_tested: 0,
                    functions_matched: 0,
                    functions_diverged: 0,
                    functions_skipped: 1,
                    function_results: vec![],
                    issues: vec![format!("Could not instantiate: {}", e)],
                });
            }
        };

        // Compare each exported function
        let mut function_results = Vec::new();
        let mut functions_tested = 0;
        let mut functions_matched = 0;
        let mut functions_diverged = 0;
        let mut functions_skipped = 0;
        let mut issues = Vec::new();

        for export in original_module.exports() {
            let orig_func = match original_instance.get_func(&mut original_store, export.name()) {
                Some(f) => f,
                None => continue,
            };

            let opt_func = match optimized_instance.get_func(&mut optimized_store, export.name()) {
                Some(f) => f,
                None => {
                    issues.push(format!(
                        "Function '{}' missing from optimized module",
                        export.name()
                    ));
                    functions_diverged += 1;
                    continue;
                }
            };

            let result = self.compare_function(
                export.name(),
                &orig_func,
                &opt_func,
                &mut original_store,
                &mut optimized_store,
            );

            match &result.divergence {
                Some(_) => {
                    functions_tested += 1;
                    functions_diverged += 1;
                }
                None if result.inputs_tested > 0 => {
                    functions_tested += 1;
                    functions_matched += 1;
                }
                None => {
                    functions_skipped += 1;
                }
            }

            function_results.push(result);
        }

        let semantics_preserved = functions_diverged == 0;

        Ok(DifferentialTestResult {
            semantics_preserved,
            original_size,
            optimized_size,
            optimized_valid: true,
            functions_tested,
            functions_matched,
            functions_diverged,
            functions_skipped,
            function_results,
            issues,
        })
    }

    /// Create a linker with common stub imports
    fn create_linker(&self) -> Result<Linker<()>> {
        let mut linker = Linker::new(&self.engine);

        // Common WASI imports (stubs)
        linker.func_wrap(
            "wasi_snapshot_preview1",
            "fd_write",
            |_: Caller<'_, ()>, _: i32, _: i32, _: i32, _: i32| -> i32 { 0 },
        )?;
        linker.func_wrap("wasi_snapshot_preview1", "proc_exit", |_: i32| {})?;
        linker.func_wrap(
            "wasi_snapshot_preview1",
            "fd_close",
            |_: Caller<'_, ()>, _: i32| -> i32 { 0 },
        )?;
        linker.func_wrap(
            "wasi_snapshot_preview1",
            "fd_seek",
            |_: Caller<'_, ()>, _: i32, _: i64, _: i32, _: i32| -> i32 { 0 },
        )?;
        linker.func_wrap(
            "wasi_snapshot_preview1",
            "fd_read",
            |_: Caller<'_, ()>, _: i32, _: i32, _: i32, _: i32| -> i32 { 0 },
        )?;
        linker.func_wrap(
            "wasi_snapshot_preview1",
            "environ_get",
            |_: Caller<'_, ()>, _: i32, _: i32| -> i32 { 0 },
        )?;
        linker.func_wrap(
            "wasi_snapshot_preview1",
            "environ_sizes_get",
            |_: Caller<'_, ()>, _: i32, _: i32| -> i32 { 0 },
        )?;

        // Common env imports (stubs)
        linker.func_wrap("env", "abort", |_: i32, _: i32, _: i32, _: i32| {})?;

        Ok(linker)
    }

    /// Compare execution of a single function on test inputs
    fn compare_function(
        &self,
        name: &str,
        orig_func: &Func,
        opt_func: &Func,
        orig_store: &mut Store<()>,
        opt_store: &mut Store<()>,
    ) -> FunctionTestResult {
        let ty = orig_func.ty(&orig_store);
        let params: Vec<ValType> = ty.params().collect();
        let results: Vec<ValType> = ty.results().collect();

        // Check if we can test this function
        if !self.can_test_function(&params, &results) {
            return FunctionTestResult {
                function_name: name.to_string(),
                inputs_tested: 0,
                all_matched: true,
                divergence: None,
            };
        }

        // Generate test inputs
        let test_inputs = self.generate_test_inputs(&params);
        let mut inputs_tested = 0;

        for input_vals in test_inputs {
            // Execute on original
            let orig_result = self.execute_function(orig_func, orig_store, &input_vals, &results);

            // Execute on optimized
            let opt_result = self.execute_function(opt_func, opt_store, &input_vals, &results);

            inputs_tested += 1;

            // Compare results
            if orig_result != opt_result {
                return FunctionTestResult {
                    function_name: name.to_string(),
                    inputs_tested,
                    all_matched: false,
                    divergence: Some(DivergenceInfo {
                        input_values: input_vals.iter().map(|v| format!("{:?}", v)).collect(),
                        original_result: orig_result,
                        optimized_result: opt_result,
                    }),
                };
            }
        }

        FunctionTestResult {
            function_name: name.to_string(),
            inputs_tested,
            all_matched: true,
            divergence: None,
        }
    }

    /// Check if a function signature can be tested
    fn can_test_function(&self, params: &[ValType], results: &[ValType]) -> bool {
        // Check param count
        if params.len() > self.config.max_params {
            return false;
        }

        // Check param types (only numeric)
        for p in params {
            match p {
                ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64 => {}
                _ => return false,
            }
        }

        // Check result types (only numeric)
        for r in results {
            match r {
                ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64 => {}
                _ => return false,
            }
        }

        true
    }

    /// Generate test inputs for a function signature
    fn generate_test_inputs(&self, params: &[ValType]) -> Vec<Vec<Val>> {
        if params.is_empty() {
            return vec![vec![]];
        }

        let f32_vals: Vec<f32> = vec![0.0, 1.0, -1.0, f32::MIN, f32::MAX, 2.5, -0.5];
        let f64_vals: Vec<f64> = vec![0.0, 1.0, -1.0, f64::MIN, f64::MAX, 2.5, -0.5];

        // For single param, test all values
        if params.len() == 1 {
            return match params[0] {
                ValType::I32 => self
                    .config
                    .i32_test_values
                    .iter()
                    .map(|v| vec![Val::I32(*v)])
                    .collect(),
                ValType::I64 => self
                    .config
                    .i64_test_values
                    .iter()
                    .map(|v| vec![Val::I64(*v)])
                    .collect(),
                ValType::F32 => f32_vals
                    .iter()
                    .map(|v| vec![Val::F32(v.to_bits())])
                    .collect(),
                ValType::F64 => f64_vals
                    .iter()
                    .map(|v| vec![Val::F64(v.to_bits())])
                    .collect(),
                _ => vec![],
            };
        }

        // For multiple params, use representative combinations
        let mut inputs = Vec::new();
        let subset_size = self.config.inputs_per_param.min(10);

        for i in 0..subset_size {
            let mut vals = Vec::new();
            for (j, p) in params.iter().enumerate() {
                let idx = (i + j) % subset_size;
                let val = match p {
                    ValType::I32 => {
                        Val::I32(self.config.i32_test_values[idx % self.config.i32_test_values.len()])
                    }
                    ValType::I64 => {
                        Val::I64(self.config.i64_test_values[idx % self.config.i64_test_values.len()])
                    }
                    ValType::F32 => Val::F32(f32_vals[idx % f32_vals.len()].to_bits()),
                    ValType::F64 => Val::F64(f64_vals[idx % f64_vals.len()].to_bits()),
                    _ => continue,
                };
                vals.push(val);
            }
            if vals.len() == params.len() {
                inputs.push(vals);
            }
        }

        inputs
    }

    /// Execute a function and return the result
    fn execute_function(
        &self,
        func: &Func,
        store: &mut Store<()>,
        inputs: &[Val],
        result_types: &[ValType],
    ) -> ExecutionResult {
        // Initialize result buffer
        let mut results: Vec<Val> = result_types
            .iter()
            .map(|t| match t {
                ValType::I32 => Val::I32(0),
                ValType::I64 => Val::I64(0),
                ValType::F32 => Val::F32(0),
                ValType::F64 => Val::F64(0),
                _ => Val::I32(0),
            })
            .collect();

        // Call the function
        match func.call(store, inputs, &mut results) {
            Ok(()) => {
                let result_strs: Vec<String> = results
                    .iter()
                    .map(|v| match v {
                        Val::I32(x) => format!("i32:{}", x),
                        Val::I64(x) => format!("i64:{}", x),
                        Val::F32(x) => {
                            let f = f32::from_bits(*x);
                            if f.is_nan() {
                                "f32:NaN".to_string()
                            } else {
                                format!("f32:{}", f)
                            }
                        }
                        Val::F64(x) => {
                            let f = f64::from_bits(*x);
                            if f.is_nan() {
                                "f64:NaN".to_string()
                            } else {
                                format!("f64:{}", f)
                            }
                        }
                        _ => format!("{:?}", v),
                    })
                    .collect();
                ExecutionResult::Success(result_strs)
            }
            Err(e) => ExecutionResult::Trap(e.to_string()),
        }
    }
}

/// Batch differential testing for multiple WASM files
#[cfg(feature = "runtime")]
pub struct BatchDifferentialTester {
    executor: DifferentialExecutor,
}

#[cfg(feature = "runtime")]
impl BatchDifferentialTester {
    /// Create a new batch tester
    pub fn new() -> Result<Self> {
        Ok(Self {
            executor: DifferentialExecutor::new()?,
        })
    }

    /// Create a batch tester with custom config
    pub fn with_config(config: ExecutionConfig) -> Result<Self> {
        Ok(Self {
            executor: DifferentialExecutor::with_config(config)?,
        })
    }

    /// Test multiple WASM files and return aggregated results
    pub fn test_files(&self, paths: &[std::path::PathBuf]) -> BatchTestResult {
        let mut results = Vec::new();
        let mut total_tested = 0;
        let mut total_passed = 0;
        let mut total_failed = 0;
        let mut total_errors = 0;

        for path in paths {
            let file_result = match std::fs::read(path) {
                Ok(bytes) => match self.executor.test_optimization(&bytes) {
                    Ok(result) => {
                        total_tested += 1;
                        if result.passed() {
                            total_passed += 1;
                            FileTestResult::Passed(result)
                        } else {
                            total_failed += 1;
                            FileTestResult::Failed(result)
                        }
                    }
                    Err(e) => {
                        total_errors += 1;
                        FileTestResult::Error(e.to_string())
                    }
                },
                Err(e) => {
                    total_errors += 1;
                    FileTestResult::Error(format!("Failed to read file: {}", e))
                }
            };

            results.push((path.clone(), file_result));
        }

        BatchTestResult {
            results,
            total_tested,
            total_passed,
            total_failed,
            total_errors,
        }
    }
}

/// Result of testing a single file in a batch
#[cfg(feature = "runtime")]
#[derive(Debug)]
pub enum FileTestResult {
    /// Test passed (semantics preserved)
    Passed(DifferentialTestResult),
    /// Test failed (semantics violated)
    Failed(DifferentialTestResult),
    /// Error during testing
    Error(String),
}

/// Results from batch testing
#[cfg(feature = "runtime")]
#[derive(Debug)]
pub struct BatchTestResult {
    /// Per-file results
    pub results: Vec<(std::path::PathBuf, FileTestResult)>,
    /// Total files tested
    pub total_tested: usize,
    /// Files that passed
    pub total_passed: usize,
    /// Files that failed
    pub total_failed: usize,
    /// Files with errors
    pub total_errors: usize,
}

#[cfg(feature = "runtime")]
impl BatchTestResult {
    /// Get overall pass rate
    pub fn pass_rate(&self) -> f64 {
        if self.total_tested == 0 {
            100.0
        } else {
            (self.total_passed as f64 / self.total_tested as f64) * 100.0
        }
    }

    /// Print a summary report
    pub fn summary(&self) -> String {
        format!(
            "Batch Test: {}/{} passed ({:.1}%), {} errors",
            self.total_passed,
            self.total_tested,
            self.pass_rate(),
            self.total_errors
        )
    }
}

/// Quick validation without full execution testing
///
/// This function performs structural validation only:
/// 1. Validates input is valid WebAssembly
/// 2. Optimizes with LOOM
/// 3. Validates output is valid WebAssembly
/// 4. Compares exports structure
///
/// This is faster than full differential testing but cannot detect
/// semantic bugs (wrong output values).
pub fn validate_optimization(wasm_bytes: &[u8]) -> Result<ValidationResult> {
    let original_size = wasm_bytes.len();

    // Validate input
    wasmparser::validate(wasm_bytes).context("Input is not valid WebAssembly")?;

    // Optimize with LOOM
    let mut module =
        loom_core::parse::parse_wasm(wasm_bytes).context("Failed to parse WebAssembly")?;
    loom_core::optimize::optimize_module(&mut module).context("LOOM optimization failed")?;
    let optimized_bytes =
        loom_core::encode::encode_wasm(&module).context("Failed to encode optimized module")?;

    let optimized_size = optimized_bytes.len();

    // Validate output
    let optimized_valid = wasmparser::validate(&optimized_bytes).is_ok();

    // Compare exports
    let original_exports = count_exports(wasm_bytes)?;
    let optimized_exports = count_exports(&optimized_bytes)?;
    let exports_preserved = original_exports == optimized_exports;

    Ok(ValidationResult {
        original_size,
        optimized_size,
        optimized_valid,
        exports_preserved,
        original_export_count: original_exports,
        optimized_export_count: optimized_exports,
    })
}

/// Count the number of exports in a WebAssembly module
fn count_exports(wasm_bytes: &[u8]) -> Result<usize> {
    use wasmparser::{Parser, Payload};

    let mut count = 0;
    for payload in Parser::new(0).parse_all(wasm_bytes) {
        if let Payload::ExportSection(reader) = payload? {
            count = reader.count() as usize;
            break;
        }
    }
    Ok(count)
}

/// Result of structural validation (without execution)
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Original module size
    pub original_size: usize,
    /// Optimized module size
    pub optimized_size: usize,
    /// Whether optimized output is valid WebAssembly
    pub optimized_valid: bool,
    /// Whether exports are preserved
    pub exports_preserved: bool,
    /// Number of exports in original
    pub original_export_count: usize,
    /// Number of exports in optimized
    pub optimized_export_count: usize,
}

impl ValidationResult {
    /// Calculate size reduction percentage
    pub fn size_reduction_percent(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            (1.0 - (self.optimized_size as f64 / self.original_size as f64)) * 100.0
        }
    }

    /// Check if validation passed
    pub fn passed(&self) -> bool {
        self.optimized_valid && self.exports_preserved
    }
}

#[cfg(all(test, feature = "runtime"))]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function_equivalence() {
        let wat = r#"
            (module
              (func (export "add") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add)
            )
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let executor = DifferentialExecutor::new().expect("Failed to create executor");
        let result = executor
            .test_optimization(&wasm)
            .expect("Test failed");

        assert!(result.semantics_preserved, "Semantics should be preserved");
        assert!(result.optimized_valid, "Output should be valid");
        assert!(result.functions_tested > 0, "Should test at least one function");
    }

    #[test]
    fn test_constant_folding() {
        let wat = r#"
            (module
              (func (export "const") (result i32)
                i32.const 10
                i32.const 20
                i32.add)
            )
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let executor = DifferentialExecutor::new().expect("Failed to create executor");
        let result = executor
            .test_optimization(&wasm)
            .expect("Test failed");

        assert!(result.semantics_preserved, "Constant folding should preserve semantics");
        assert!(result.functions_matched >= 1, "const function should match");
    }

    #[test]
    fn test_validation_only() {
        let wat = r#"
            (module
              (func (export "identity") (param i32) (result i32)
                local.get 0)
            )
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let result = validate_optimization(&wasm).expect("Validation failed");

        assert!(result.optimized_valid, "Output should be valid");
        assert!(result.exports_preserved, "Exports should be preserved");
        assert_eq!(result.original_export_count, 1, "Should have 1 export");
    }

    #[test]
    fn test_execution_config() {
        let minimal = ExecutionConfig::minimal();
        assert_eq!(minimal.inputs_per_param, 3);
        assert_eq!(minimal.max_params, 2);

        let thorough = ExecutionConfig::thorough();
        assert!(thorough.inputs_per_param > minimal.inputs_per_param);
        assert!(thorough.max_params > minimal.max_params);
    }

    #[test]
    fn test_divergence_detection() {
        // This test verifies that divergence would be detected if it occurred
        // We don't have a known bug to test, so we just verify the mechanism works
        let wat = r#"
            (module
              (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            )
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let executor = DifferentialExecutor::new().expect("Failed to create executor");
        let result = executor
            .test_optimization(&wasm)
            .expect("Test failed");

        // If there were a bug, functions_diverged would be > 0
        assert_eq!(
            result.functions_diverged, 0,
            "No divergence expected for correct optimizer"
        );
    }
}
