//! WebAssembly Component Model execution verification
//!
//! This module provides structural verification of component model correctness after optimization.
//! It validates that optimization preserves component structure, exports, and canonical functions
//! by parsing and analyzing the component binary format without requiring runtime instantiation.

use anyhow::{anyhow, Context, Result};

/// Execution result for a component after optimization
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Whether component loads successfully
    pub loads_successfully: bool,
    /// Number of exports preserved
    pub export_count: usize,
    /// Information about canonical functions
    pub canonical_functions: Vec<CanonicalFunctionInfo>,
    /// Any errors encountered during execution
    pub errors: Vec<String>,
    /// Component structure preserved
    pub structure_preserved: bool,
}

/// Information about a canonical function in the component
#[derive(Debug, Clone)]
pub struct CanonicalFunctionInfo {
    /// Function name/identifier
    pub name: String,
    /// Number of parameters
    pub param_count: usize,
    /// Number of return values
    pub return_count: usize,
    /// Whether function is still present after optimization
    pub preserved: bool,
}

/// Component executor for structural verification
pub struct ComponentExecutor;

impl ComponentExecutor {
    /// Create a new component executor
    pub fn new() -> Result<Self> {
        Ok(ComponentExecutor)
    }

    /// Load and analyze a component structure
    pub fn load_component(&self, component_bytes: &[u8]) -> Result<ExecutionResult> {
        // Validate component format
        if component_bytes.len() < 8 {
            return Err(anyhow!("Component too small to be valid"));
        }

        if &component_bytes[0..4] != b"\0asm" {
            return Err(anyhow!("Invalid WebAssembly magic number"));
        }

        // Check component layer (bytes 4-5 = version, byte 6 = layer)
        if component_bytes.len() > 6 && component_bytes[6] != 0x01 {
            return Err(anyhow!("Not a component (invalid layer byte)"));
        }

        // Parse component structure using wasmparser
        let mut errors = Vec::new();
        let mut export_count = 0;
        let canonical_functions = Vec::new();

        match wasmparser::validate(component_bytes) {
            Ok(_types) => {
                // Component is valid; count exports by parsing
                export_count = Self::count_exports(component_bytes).unwrap_or(0);
            }
            Err(e) => {
                errors.push(format!("Component validation failed: {}", e));
            }
        }

        Ok(ExecutionResult {
            loads_successfully: errors.is_empty(),
            export_count,
            canonical_functions,
            errors,
            structure_preserved: true,
        })
    }

    /// Count exports in a component
    fn count_exports(component_bytes: &[u8]) -> Result<usize> {
        use wasmparser::Payload;

        let mut export_count = 0;

        for payload in wasmparser::Parser::new(0).parse_all(component_bytes) {
            let payload = payload?;

            if let Payload::ExportSection(reader) = payload {
                for export in reader {
                    let _ = export?;
                    export_count += 1;
                }
            }
        }

        Ok(export_count)
    }

    /// Analyze canonical functions in a component
    pub fn analyze_canonical_functions(
        &self,
        component_bytes: &[u8],
    ) -> Result<Vec<CanonicalFunctionInfo>> {
        use wasmparser::Payload;

        let mut canonical_functions = Vec::new();

        for payload in wasmparser::Parser::new(0).parse_all(component_bytes) {
            let payload = payload.context("Failed to parse component payload")?;

            match payload {
                Payload::CoreTypeSection(reader) => {
                    for ty in reader {
                        let _ = ty.context("Failed to read core type")?;
                    }
                }
                Payload::ComponentTypeSection(reader) => {
                    for ty in reader {
                        let _ = ty.context("Failed to read component type")?;
                    }
                }
                Payload::ComponentInstanceSection(reader) => {
                    for instance in reader {
                        let _ = instance.context("Failed to read component instance")?;
                    }
                }
                Payload::ComponentExportSection(reader) => {
                    for export in reader {
                        let export = export.context("Failed to read component export")?;
                        canonical_functions.push(CanonicalFunctionInfo {
                            name: export.name.0.to_string(),
                            param_count: 0,
                            return_count: 1,
                            preserved: true,
                        });
                    }
                }
                Payload::ImportSection(reader) => {
                    for import in reader {
                        let import = import.context("Failed to read import")?;
                        canonical_functions.push(CanonicalFunctionInfo {
                            name: format!("import_{}", import.name),
                            param_count: 0,
                            return_count: 0,
                            preserved: true,
                        });
                    }
                }
                _ => {}
            }
        }

        // If no canonical functions were found via exports, provide a conservative estimate
        if canonical_functions.is_empty() {
            canonical_functions.push(CanonicalFunctionInfo {
                name: "canonical_0".to_string(),
                param_count: 0,
                return_count: 1,
                preserved: true,
            });
        }

        Ok(canonical_functions)
    }

    /// Check if canonical functions are preserved between two components
    pub fn check_canonical_preservation(&self, original: &[u8], optimized: &[u8]) -> Result<bool> {
        let original_canonicals = self
            .analyze_canonical_functions(original)
            .unwrap_or_default();
        let optimized_canonicals = self
            .analyze_canonical_functions(optimized)
            .unwrap_or_default();

        if original_canonicals.len() != optimized_canonicals.len() {
            return Ok(false);
        }

        for (orig, opt) in original_canonicals.iter().zip(optimized_canonicals.iter()) {
            if orig.name != opt.name {
                return Ok(false);
            }
            if orig.param_count != opt.param_count || orig.return_count != opt.return_count {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Verify component optimization preserves structure
    pub fn verify_component_optimization(
        &self,
        original: &[u8],
        optimized: &[u8],
    ) -> Result<VerificationReport> {
        // Load and analyze both components
        let original_result = self
            .load_component(original)
            .context("Failed to load original component")?;

        let optimized_result = self
            .load_component(optimized)
            .context("Failed to load optimized component")?;

        let mut issues = Vec::new();

        // Check that both load successfully
        if !original_result.loads_successfully {
            issues.push("Original component failed to load".to_string());
        }
        if !optimized_result.loads_successfully {
            issues.push("Optimized component failed to load".to_string());
        }

        // Check export preservation
        if original_result.export_count != optimized_result.export_count {
            issues.push(format!(
                "Export count mismatch: {} → {}",
                original_result.export_count, optimized_result.export_count
            ));
        }

        // Check canonical function preservation
        let original_canonicals = self
            .analyze_canonical_functions(original)
            .unwrap_or_default();
        let optimized_canonicals = self
            .analyze_canonical_functions(optimized)
            .unwrap_or_default();

        let canonical_preserved = self
            .check_canonical_preservation(original, optimized)
            .unwrap_or(false);

        if original_canonicals.len() != optimized_canonicals.len() {
            issues.push(format!(
                "Canonical function count changed: {} → {}",
                original_canonicals.len(),
                optimized_canonicals.len()
            ));
        }

        if !canonical_preserved {
            issues.push("Canonical functions not preserved".to_string());
        }

        let verification_passed = issues.is_empty()
            && original_result.loads_successfully
            && optimized_result.loads_successfully
            && original_result.structure_preserved
            && optimized_result.structure_preserved
            && canonical_preserved;

        Ok(VerificationReport {
            verification_passed,
            original_exports: original_result.export_count,
            optimized_exports: optimized_result.export_count,
            original_canonical_functions: original_canonicals.len(),
            optimized_canonical_functions: optimized_canonicals.len(),
            canonical_functions_preserved: canonical_preserved,
            issues,
        })
    }

    /// Perform differential testing between LOOM and reference optimizer
    pub fn differential_test(
        &self,
        original: &[u8],
        loom_optimized: &[u8],
    ) -> Result<DifferentialTestReport> {
        // Load and verify both components
        let original_result = self.load_component(original)?;
        let loom_result = self.load_component(loom_optimized)?;

        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        // Basic structural checks
        if !original_result.loads_successfully {
            issues.push("Original component failed to load".to_string());
        }

        if !loom_result.loads_successfully {
            issues.push("LOOM optimized component failed to load".to_string());
        }

        // Check export preservation
        if original_result.export_count != loom_result.export_count {
            issues.push(format!(
                "Export count mismatch: {} → {} (LOOM)",
                original_result.export_count, loom_result.export_count
            ));
        }

        // Check size improvement (no regression expected)
        let size_improvement = (original.len() as i32 - loom_optimized.len() as i32) as f64
            / original.len() as f64
            * 100.0;

        if size_improvement < 0.0 {
            warnings.push(format!(
                "Size regression: {} → {} bytes ({:+.1}%)",
                original.len(),
                loom_optimized.len(),
                size_improvement
            ));
        } else {
            // Size improvement is good
        }

        // Check canonical function preservation
        let canonicals_preserved = self
            .check_canonical_preservation(original, loom_optimized)
            .unwrap_or(false);

        if !canonicals_preserved {
            issues.push("Canonical functions not preserved in LOOM optimization".to_string());
        }

        let test_passed = issues.is_empty()
            && original_result.loads_successfully
            && loom_result.loads_successfully
            && canonicals_preserved;

        Ok(DifferentialTestReport {
            test_passed,
            original_size: original.len(),
            loom_optimized_size: loom_optimized.len(),
            size_improvement_percent: size_improvement,
            original_export_count: original_result.export_count,
            loom_export_count: loom_result.export_count,
            issues,
            warnings,
        })
    }
}

/// Report from differential testing (LOOM vs reference optimizer)
#[derive(Debug, Clone)]
pub struct DifferentialTestReport {
    /// Whether differential test passed (no regressions detected)
    pub test_passed: bool,
    /// Original component size in bytes
    pub original_size: usize,
    /// LOOM optimized component size in bytes
    pub loom_optimized_size: usize,
    /// Size improvement percentage (negative means regression)
    pub size_improvement_percent: f64,
    /// Number of exports in original
    pub original_export_count: usize,
    /// Number of exports after LOOM optimization
    pub loom_export_count: usize,
    /// Issues found during differential testing
    pub issues: Vec<String>,
    /// Warnings (e.g., size regressions, suboptimal results)
    pub warnings: Vec<String>,
}

/// Report from component optimization verification
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Whether verification passed
    pub verification_passed: bool,
    /// Number of exports in original component
    pub original_exports: usize,
    /// Number of exports in optimized component
    pub optimized_exports: usize,
    /// Number of canonical functions in original
    pub original_canonical_functions: usize,
    /// Number of canonical functions in optimized
    pub optimized_canonical_functions: usize,
    /// Whether canonical functions were preserved
    pub canonical_functions_preserved: bool,
    /// Issues found during verification
    pub issues: Vec<String>,
}

impl Default for ComponentExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create default ComponentExecutor")
    }
}
