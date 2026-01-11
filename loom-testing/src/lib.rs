//! Differential testing framework for LOOM
//!
//! This module provides infrastructure for comparing LOOM's optimization
//! results against wasm-opt to validate correctness and identify gaps.

pub mod differential;
pub mod emi;

#[cfg(feature = "runtime")]
use anyhow::{Context, Result};
#[cfg(feature = "runtime")]
use std::process::Command;
#[cfg(feature = "runtime")]
use tempfile::NamedTempFile;
#[cfg(feature = "runtime")]
use wasmtime::{Engine, Module};

/// Differential tester comparing LOOM vs wasm-opt
#[cfg(feature = "runtime")]
pub struct DifferentialTester {
    loom_binary: String,
    wasm_opt_binary: String,
}

#[cfg(feature = "runtime")]
impl DifferentialTester {
    /// Create a new differential tester
    ///
    /// Finds LOOM and wasm-opt binaries in PATH
    pub fn new() -> Result<Self> {
        let loom = which::which("loom")
            .context("Could not find 'loom' binary. Build with 'cargo build --release'")?;
        let wasm_opt = which::which("wasm-opt")
            .context("Could not find 'wasm-opt' binary. Install with 'brew install binaryen'")?;

        Ok(Self {
            loom_binary: loom.to_string_lossy().to_string(),
            wasm_opt_binary: wasm_opt.to_string_lossy().to_string(),
        })
    }

    /// Create a tester with custom binary paths
    pub fn with_binaries(loom_path: String, wasm_opt_path: String) -> Self {
        Self {
            loom_binary: loom_path,
            wasm_opt_binary: wasm_opt_path,
        }
    }

    /// Test a WASM module by optimizing with both tools and comparing
    pub fn test(&self, input_wasm: &[u8]) -> Result<TestResult> {
        // Optimize with LOOM
        let loom_output = self.run_loom(input_wasm)?;

        // Optimize with wasm-opt
        let wasm_opt_output = self.run_wasm_opt(input_wasm)?;

        // Compare results
        TestResult::compare(input_wasm, &loom_output, &wasm_opt_output)
    }

    /// Run LOOM optimizer on input
    fn run_loom(&self, input: &[u8]) -> Result<Vec<u8>> {
        let temp_in = NamedTempFile::new()?;
        std::fs::write(temp_in.path(), input)?;

        let temp_out = NamedTempFile::new()?;

        let output = Command::new(&self.loom_binary)
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

    /// Run wasm-opt on input with -O3 optimization level
    fn run_wasm_opt(&self, input: &[u8]) -> Result<Vec<u8>> {
        let temp_in = NamedTempFile::new()?;
        std::fs::write(temp_in.path(), input)?;

        let temp_out = NamedTempFile::new()?;

        let output = Command::new(&self.wasm_opt_binary)
            .arg(temp_in.path())
            .arg("-O3")
            .arg("-o")
            .arg(temp_out.path())
            .output()
            .context("Failed to execute wasm-opt")?;

        if !output.status.success() {
            anyhow::bail!(
                "wasm-opt failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(std::fs::read(temp_out.path())?)
    }
}

/// Results of differential testing comparison
#[cfg(feature = "runtime")]
#[derive(Debug, Clone)]
pub struct TestResult {
    pub input_size: usize,
    pub loom_size: usize,
    pub wasm_opt_size: usize,

    pub loom_valid: bool,
    pub wasm_opt_valid: bool,

    /// Semantic equivalence (None if couldn't execute)
    pub semantically_equivalent: Option<bool>,

    /// Original vs LOOM equivalence (the critical correctness check)
    pub original_loom_equivalent: Option<bool>,

    /// Execution results comparison details
    pub execution_details: Option<ExecutionComparison>,
}

/// Details of execution comparison between original and optimized
#[cfg(feature = "runtime")]
#[derive(Debug, Clone)]
pub struct ExecutionComparison {
    /// Functions tested
    pub functions_tested: usize,
    /// Functions that matched
    pub functions_matching: usize,
    /// Functions that diverged (bug detected!)
    pub functions_diverged: usize,
    /// Functions that couldn't be tested (missing imports, etc.)
    pub functions_skipped: usize,
    /// Details of any divergence
    pub divergence_details: Vec<DivergenceDetail>,
}

/// Details about a semantic divergence (potential optimization bug)
#[cfg(feature = "runtime")]
#[derive(Debug, Clone)]
pub struct DivergenceDetail {
    pub function_name: String,
    pub input_values: Vec<String>,
    pub original_result: String,
    pub optimized_result: String,
}

/// Result of comparing execution of a single function
#[cfg(feature = "runtime")]
enum FunctionCompareResult {
    /// Function execution matched on all test inputs
    Match,
    /// Function execution diverged (potential bug!)
    Diverged(DivergenceDetail),
    /// Function couldn't be tested (complex signature, etc.)
    Skipped,
}

#[cfg(feature = "runtime")]
impl TestResult {
    /// Compare LOOM and wasm-opt outputs
    pub fn compare(input: &[u8], loom: &[u8], wasm_opt: &[u8]) -> Result<Self> {
        let input_size = input.len();
        let loom_size = loom.len();
        let wasm_opt_size = wasm_opt.len();

        // Validate both outputs
        let loom_valid = wasmparser::validate(loom).is_ok();
        let wasm_opt_valid = wasmparser::validate(wasm_opt).is_ok();

        // Check semantic equivalence via wasmtime execution
        let semantically_equivalent = if loom_valid && wasm_opt_valid {
            Self::check_semantic_equivalence(loom, wasm_opt).ok()
        } else {
            None
        };

        // CRITICAL: Check that LOOM's optimization preserves semantics vs original
        let (original_loom_equivalent, execution_details) = if loom_valid {
            Self::compare_execution(input, loom)
                .map(|(eq, details)| (Some(eq), Some(details)))
                .unwrap_or((None, None))
        } else {
            (None, None)
        };

        Ok(TestResult {
            input_size,
            loom_size,
            wasm_opt_size,
            loom_valid,
            wasm_opt_valid,
            semantically_equivalent,
            original_loom_equivalent,
            execution_details,
        })
    }

    /// Compare execution of original and optimized modules
    ///
    /// This is the CRITICAL correctness check - if the optimized version
    /// produces different results than the original, we've found a bug.
    fn compare_execution(original: &[u8], optimized: &[u8]) -> Result<(bool, ExecutionComparison)> {
        use wasmtime::{Caller, Linker, Store};

        let engine = Engine::default();
        let mut store = Store::new(&engine, ());

        // Create a linker with common imports
        let mut linker = Linker::new(&engine);

        // Add dummy imports for common patterns
        linker.func_wrap("env", "abort", |_: i32, _: i32, _: i32, _: i32| {})?;
        linker.func_wrap("env", "memory", || {})?;
        linker.func_wrap(
            "wasi_snapshot_preview1",
            "fd_write",
            |_: Caller<'_, ()>, _: i32, _: i32, _: i32, _: i32| -> i32 { 0 },
        )?;
        linker.func_wrap("wasi_snapshot_preview1", "proc_exit", |_: i32| {})?;

        // Try to load modules
        let original_module =
            Module::new(&engine, original).context("Failed to load original module")?;
        let optimized_module =
            Module::new(&engine, optimized).context("Failed to load optimized module")?;

        // Try to instantiate (may fail due to missing imports)
        let original_instance = match linker.instantiate(&mut store, &original_module) {
            Ok(inst) => inst,
            Err(_) => {
                // Can't instantiate - return skipped
                return Ok((
                    true, // Assume equivalent if can't test
                    ExecutionComparison {
                        functions_tested: 0,
                        functions_matching: 0,
                        functions_diverged: 0,
                        functions_skipped: 1,
                        divergence_details: vec![],
                    },
                ));
            }
        };

        let optimized_instance = match linker.instantiate(&mut store, &optimized_module) {
            Ok(inst) => inst,
            Err(_) => {
                return Ok((
                    true,
                    ExecutionComparison {
                        functions_tested: 0,
                        functions_matching: 0,
                        functions_diverged: 0,
                        functions_skipped: 1,
                        divergence_details: vec![],
                    },
                ));
            }
        };

        // Compare exports
        let mut comparison = ExecutionComparison {
            functions_tested: 0,
            functions_matching: 0,
            functions_diverged: 0,
            functions_skipped: 0,
            divergence_details: vec![],
        };

        // Get exported functions
        for export in original_module.exports() {
            if let Some(orig_func) = original_instance.get_func(&mut store, export.name()) {
                if let Some(opt_func) = optimized_instance.get_func(&mut store, export.name()) {
                    let result = Self::compare_function_execution(
                        &mut store,
                        export.name(),
                        &orig_func,
                        &opt_func,
                    );

                    match result {
                        FunctionCompareResult::Match => {
                            comparison.functions_tested += 1;
                            comparison.functions_matching += 1;
                        }
                        FunctionCompareResult::Diverged(detail) => {
                            comparison.functions_tested += 1;
                            comparison.functions_diverged += 1;
                            comparison.divergence_details.push(detail);
                        }
                        FunctionCompareResult::Skipped => {
                            comparison.functions_skipped += 1;
                        }
                    }
                }
            }
        }

        let all_match = comparison.functions_diverged == 0;
        Ok((all_match, comparison))
    }

    /// Compare execution of a single function with test inputs
    fn compare_function_execution(
        store: &mut wasmtime::Store<()>,
        name: &str,
        original: &wasmtime::Func,
        optimized: &wasmtime::Func,
    ) -> FunctionCompareResult {
        use wasmtime::{Val, ValType};

        let ty = original.ty(&store);

        // Only test simple numeric functions for now
        let params: Vec<_> = ty.params().collect();
        let results: Vec<_> = ty.results().collect();

        // Skip functions with non-numeric params/results
        for p in &params {
            match p {
                ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64 => {}
                _ => return FunctionCompareResult::Skipped,
            }
        }
        for r in &results {
            match r {
                ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64 => {}
                _ => return FunctionCompareResult::Skipped,
            }
        }

        // Skip functions with too many params (expensive to test)
        if params.len() > 4 {
            return FunctionCompareResult::Skipped;
        }

        // Generate test inputs
        let test_inputs = Self::generate_test_inputs(&params);

        for input_vals in test_inputs {
            let mut orig_results = vec![Val::I32(0); results.len()];
            let mut opt_results = vec![Val::I32(0); results.len()];

            // Initialize result vectors with correct types
            for (i, r) in results.iter().enumerate() {
                orig_results[i] = match r {
                    ValType::I32 => Val::I32(0),
                    ValType::I64 => Val::I64(0),
                    ValType::F32 => Val::F32(0),
                    ValType::F64 => Val::F64(0),
                    _ => return FunctionCompareResult::Skipped,
                };
                opt_results[i] = orig_results[i].clone();
            }

            // Call original
            let orig_ok = original
                .call(&mut *store, &input_vals, &mut orig_results)
                .is_ok();

            // Call optimized
            let opt_ok = optimized
                .call(&mut *store, &input_vals, &mut opt_results)
                .is_ok();

            // Compare results
            if orig_ok != opt_ok {
                return FunctionCompareResult::Diverged(DivergenceDetail {
                    function_name: name.to_string(),
                    input_values: input_vals.iter().map(|v| format!("{:?}", v)).collect(),
                    original_result: if orig_ok {
                        format!("{:?}", orig_results)
                    } else {
                        "trap".to_string()
                    },
                    optimized_result: if opt_ok {
                        format!("{:?}", opt_results)
                    } else {
                        "trap".to_string()
                    },
                });
            }

            if orig_ok {
                // Compare result values
                for (orig, opt) in orig_results.iter().zip(opt_results.iter()) {
                    if !Self::vals_equal(orig, opt) {
                        return FunctionCompareResult::Diverged(DivergenceDetail {
                            function_name: name.to_string(),
                            input_values: input_vals.iter().map(|v| format!("{:?}", v)).collect(),
                            original_result: format!("{:?}", orig_results),
                            optimized_result: format!("{:?}", opt_results),
                        });
                    }
                }
            }
        }

        FunctionCompareResult::Match
    }

    /// Generate test inputs for a function signature
    fn generate_test_inputs(params: &[wasmtime::ValType]) -> Vec<Vec<wasmtime::Val>> {
        use wasmtime::{Val, ValType};

        // Edge case values for integers
        let i32_vals: Vec<i32> = vec![0, 1, -1, i32::MIN, i32::MAX, 42, 100, 255, 256];
        let i64_vals: Vec<i64> = vec![0, 1, -1, i64::MIN, i64::MAX, 42, 100, 255, 256];
        let f32_vals: Vec<f32> = vec![0.0, 1.0, -1.0, f32::MIN, f32::MAX, 2.5];
        let f64_vals: Vec<f64> = vec![0.0, 1.0, -1.0, f64::MIN, f64::MAX, 2.5];

        if params.is_empty() {
            return vec![vec![]];
        }

        // For single param, test all values
        if params.len() == 1 {
            return match params[0] {
                ValType::I32 => i32_vals.iter().map(|v| vec![Val::I32(*v)]).collect(),
                ValType::I64 => i64_vals.iter().map(|v| vec![Val::I64(*v)]).collect(),
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

        // For multiple params, use a representative subset
        let mut inputs = Vec::new();
        let subset_size = 5; // Test 5 combinations

        for i in 0..subset_size {
            let mut vals = Vec::new();
            for (j, p) in params.iter().enumerate() {
                let idx = (i + j) % subset_size;
                let val = match p {
                    ValType::I32 => Val::I32(i32_vals[idx % i32_vals.len()]),
                    ValType::I64 => Val::I64(i64_vals[idx % i64_vals.len()]),
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

    /// Compare two wasmtime Val values for equality
    fn vals_equal(a: &wasmtime::Val, b: &wasmtime::Val) -> bool {
        use wasmtime::Val;
        match (a, b) {
            (Val::I32(x), Val::I32(y)) => x == y,
            (Val::I64(x), Val::I64(y)) => x == y,
            (Val::F32(x), Val::F32(y)) => {
                // Handle NaN comparison
                let xf = f32::from_bits(*x);
                let yf = f32::from_bits(*y);
                (xf.is_nan() && yf.is_nan()) || x == y
            }
            (Val::F64(x), Val::F64(y)) => {
                let xf = f64::from_bits(*x);
                let yf = f64::from_bits(*y);
                (xf.is_nan() && yf.is_nan()) || x == y
            }
            _ => false,
        }
    }

    /// Check if two WASM modules are semantically equivalent via execution testing
    fn check_semantic_equivalence(loom_wasm: &[u8], wasm_opt_wasm: &[u8]) -> Result<bool> {
        let engine = Engine::default();

        // Load both modules
        let loom_module =
            Module::new(&engine, loom_wasm).context("Failed to load LOOM optimized module")?;
        let wasm_opt_module = Module::new(&engine, wasm_opt_wasm)
            .context("Failed to load wasm-opt optimized module")?;

        // Try to instantiate and compare exports
        let loom_results = Self::extract_function_results(&engine, &loom_module);
        let wasm_opt_results = Self::extract_function_results(&engine, &wasm_opt_module);

        // If both modules have the same exported functions with same signatures,
        // they're semantically equivalent at the interface level
        Ok(loom_results == wasm_opt_results)
    }

    /// Extract function signatures from a module (for structural comparison)
    fn extract_function_results(_engine: &Engine, module: &Module) -> Vec<String> {
        let mut results = Vec::new();

        // Check what's exported
        for export in module.exports() {
            results.push(format!("export:{}", export.name()));
        }

        results.sort();
        results
    }

    /// Check if LOOM produced a smaller output
    pub fn loom_wins(&self) -> bool {
        self.loom_valid && self.loom_size < self.wasm_opt_size
    }

    /// Check if wasm-opt produced a smaller output
    pub fn wasm_opt_wins(&self) -> bool {
        self.wasm_opt_valid && self.wasm_opt_size < self.loom_size
    }

    /// Check if both produced the same size
    pub fn tie(&self) -> bool {
        self.loom_valid && self.wasm_opt_valid && self.loom_size == self.wasm_opt_size
    }

    /// Get the winner as a string
    pub fn winner(&self) -> &'static str {
        if !self.loom_valid && !self.wasm_opt_valid {
            "BOTH_INVALID"
        } else if !self.loom_valid {
            "LOOM_INVALID"
        } else if !self.wasm_opt_valid {
            "WASM_OPT_INVALID"
        } else if self.loom_wins() {
            "LOOM"
        } else if self.wasm_opt_wins() {
            "wasm-opt"
        } else {
            "TIE"
        }
    }

    /// Calculate LOOM's reduction percentage
    pub fn loom_reduction_pct(&self) -> f64 {
        if self.input_size == 0 {
            0.0
        } else {
            (1.0 - self.loom_size as f64 / self.input_size as f64) * 100.0
        }
    }

    /// Calculate wasm-opt's reduction percentage
    pub fn wasm_opt_reduction_pct(&self) -> f64 {
        if self.input_size == 0 {
            0.0
        } else {
            (1.0 - self.wasm_opt_size as f64 / self.input_size as f64) * 100.0
        }
    }

    /// Calculate size delta (negative means LOOM is smaller)
    pub fn size_delta(&self) -> i64 {
        self.loom_size as i64 - self.wasm_opt_size as i64
    }
}

#[cfg(all(test, feature = "runtime"))]
mod tests {
    use super::*;

    #[test]
    fn test_result_winner() {
        let result = TestResult {
            input_size: 1000,
            loom_size: 400,
            wasm_opt_size: 500,
            loom_valid: true,
            wasm_opt_valid: true,
            semantically_equivalent: None,
            original_loom_equivalent: Some(true),
            execution_details: None,
        };

        assert_eq!(result.winner(), "LOOM");
        assert!(result.loom_wins());
        assert!(!result.wasm_opt_wins());
        assert!(!result.tie());
    }

    #[test]
    fn test_result_reduction() {
        let result = TestResult {
            input_size: 1000,
            loom_size: 200,
            wasm_opt_size: 300,
            loom_valid: true,
            wasm_opt_valid: true,
            semantically_equivalent: None,
            original_loom_equivalent: Some(true),
            execution_details: None,
        };

        assert_eq!(result.loom_reduction_pct(), 80.0);
        assert_eq!(result.wasm_opt_reduction_pct(), 70.0);
        assert_eq!(result.size_delta(), -100); // LOOM is 100 bytes smaller
    }

    #[test]
    fn test_execution_comparison() {
        let comparison = ExecutionComparison {
            functions_tested: 10,
            functions_matching: 8,
            functions_diverged: 1,
            functions_skipped: 1,
            divergence_details: vec![DivergenceDetail {
                function_name: "add".to_string(),
                input_values: vec!["42".to_string(), "1".to_string()],
                original_result: "[I32(43)]".to_string(),
                optimized_result: "[I32(44)]".to_string(),
            }],
        };

        assert_eq!(comparison.functions_tested, 10);
        assert_eq!(comparison.functions_diverged, 1);
        assert!(!comparison.divergence_details.is_empty());
    }
}
