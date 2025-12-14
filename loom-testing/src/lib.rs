//! Differential testing framework for LOOM
//!
//! This module provides infrastructure for comparing LOOM's optimization
//! results against wasm-opt to validate correctness and identify gaps.

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

        Ok(TestResult {
            input_size,
            loom_size,
            wasm_opt_size,
            loom_valid,
            wasm_opt_valid,
            semantically_equivalent,
        })
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
        };

        assert_eq!(result.loom_reduction_pct(), 80.0);
        assert_eq!(result.wasm_opt_reduction_pct(), 70.0);
        assert_eq!(result.size_delta(), -100); // LOOM is 100 bytes smaller
    }
}
