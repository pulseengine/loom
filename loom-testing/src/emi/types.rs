//! Type definitions for EMI testing

/// A region of code identified as dead (not executed for known inputs)
#[derive(Debug, Clone)]
pub struct DeadRegion {
    /// Index of the function containing this dead region
    pub func_idx: u32,
    /// Byte offset where dead region starts in the function body
    pub start_offset: usize,
    /// Byte offset where dead region ends in the function body
    pub end_offset: usize,
    /// Type of dead region
    pub region_type: DeadRegionType,
    /// Human-readable description
    pub description: String,
}

/// Classification of dead code regions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadRegionType {
    /// Dead branch of an if/else due to constant condition
    /// `is_then` = true means the "then" branch is dead (condition is always false)
    /// `is_then` = false means the "else" branch is dead (condition is always true)
    ConstantBranchIf { is_then_dead: bool },

    /// Code after an unconditional branch (br)
    AfterUnconditionalBranch,

    /// Code after a return instruction
    AfterReturn,

    /// Code after an unreachable instruction
    AfterUnreachable,

    /// br_if with constant condition that's always taken
    ConstantBrIf { always_taken: bool },
}

/// Mutation strategies for EMI testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationStrategy {
    /// Modify constant values in dead code (safest)
    ModifyConstants,

    /// Replace dead code body with unreachable instruction
    ReplaceWithUnreachable,

    /// Replace dead code with nop instructions
    ReplaceWithNop,

    /// Insert additional harmless dead code
    InsertDeadCode,

    /// Delete dead code entirely (most aggressive)
    Delete,
}

impl MutationStrategy {
    /// Get a human-readable name for this strategy
    pub fn name(&self) -> &'static str {
        match self {
            Self::ModifyConstants => "modify_constants",
            Self::ReplaceWithUnreachable => "replace_unreachable",
            Self::ReplaceWithNop => "replace_nop",
            Self::InsertDeadCode => "insert_dead",
            Self::Delete => "delete",
        }
    }
}

/// Result of EMI testing
#[derive(Debug)]
pub struct EmiTestResult {
    /// Number of dead code regions found in the module
    pub dead_regions_found: usize,
    /// Number of variants successfully tested
    pub variants_tested: usize,
    /// Bugs found during testing
    pub bugs_found: Vec<EmiBug>,
    /// Detailed analysis information
    pub analysis_details: Vec<String>,
}

impl EmiTestResult {
    /// Check if any bugs were found
    pub fn has_bugs(&self) -> bool {
        !self.bugs_found.is_empty()
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "EMI Test: {} dead regions, {} variants tested, {} bugs found",
            self.dead_regions_found,
            self.variants_tested,
            self.bugs_found.len()
        )
    }
}

/// A bug found during EMI testing
#[derive(Debug)]
pub struct EmiBug {
    /// Variant ID that triggered this bug
    pub variant_id: usize,
    /// Mutation strategy that was applied
    pub mutation_strategy: MutationStrategy,
    /// Dead region that was mutated
    pub dead_region: DeadRegion,
    /// Type of bug
    pub bug_type: EmiBugType,
    /// Expected value (if applicable)
    pub expected: Option<String>,
    /// Actual value (if applicable)
    pub actual: Option<String>,
}

/// Classification of EMI bugs
#[derive(Debug, Clone)]
pub enum EmiBugType {
    /// Optimizer crashed or returned an error
    OptimizationCrash(String),

    /// Optimizer produced invalid WebAssembly
    InvalidOutput(String),

    /// Optimized code produced different output than original
    OutputMismatch { function: String },

    /// Execution failed (trap, etc.)
    ExecutionError(String),
}

impl EmiBugType {
    /// Get a short description of the bug type
    pub fn short_description(&self) -> &str {
        match self {
            Self::OptimizationCrash(_) => "optimization_crash",
            Self::InvalidOutput(_) => "invalid_output",
            Self::OutputMismatch { .. } => "output_mismatch",
            Self::ExecutionError(_) => "execution_error",
        }
    }
}
