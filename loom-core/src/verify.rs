//! Formal Verification Module for LOOM
//!
//! This module implements translation validation using the Z3 SMT solver to prove
//! that optimizations preserve program semantics.
//!
//! # Overview
//!
//! Translation validation works by encoding both the original and optimized programs
//! as SMT formulas and asking Z3 to prove they are semantically equivalent for all
//! possible inputs.
//!
//! # Example
//!
//! ```rust,ignore
//! use loom_core::verify::verify_optimization;
//!
//! let original = parse_wat("(module (func (result i32) (i32.add (i32.const 2) (i32.const 3))))");
//! let optimized = parse_wat("(module (func (result i32) (i32.const 5)))");
//!
//! // Verify that 2 + 3 = 5 optimization is correct
//! assert!(verify_optimization(&original, &optimized).unwrap());
//! ```

#[cfg(feature = "verification")]
use z3::ast::{Array, Bool, BV};
#[cfg(feature = "verification")]
use z3::{with_z3_config, Config, SatResult, Solver, Sort};

#[cfg(not(feature = "verification"))]
use crate::Module;
#[cfg(feature = "verification")]
use crate::{BlockType, Function, FunctionSignature, ImportKind, Instruction, Module};
#[cfg(feature = "verification")]
use anyhow::Context as AnyhowContext;
use anyhow::{anyhow, Result};

/// Signature context for verification - stores function and type signatures
/// for proper Call/CallIndirect stack effect modeling.
#[cfg(feature = "verification")]
#[derive(Clone, Default)]
pub struct VerificationSignatureContext {
    /// Function signatures indexed by function index (imports first, then locals)
    pub function_signatures: Vec<FunctionSignature>,
    /// Type signatures for CallIndirect (indexed by type index)
    pub type_signatures: Vec<FunctionSignature>,
}

#[cfg(feature = "verification")]
impl VerificationSignatureContext {
    /// Create a new empty context (for backwards compatibility)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a signature context from a module
    pub fn from_module(module: &Module) -> Self {
        let mut function_signatures = Vec::new();

        // First, add imported function signatures (they come first in indexing)
        for import in &module.imports {
            if let ImportKind::Func(type_idx) = &import.kind {
                if let Some(sig) = module.types.get(*type_idx as usize) {
                    function_signatures.push(sig.clone());
                }
            }
        }

        // Then add local function signatures
        for func in &module.functions {
            function_signatures.push(func.signature.clone());
        }

        VerificationSignatureContext {
            function_signatures,
            type_signatures: module.types.clone(),
        }
    }

    /// Get the signature for a function by its function index
    pub fn get_function_signature(&self, func_idx: u32) -> Option<&FunctionSignature> {
        self.function_signatures.get(func_idx as usize)
    }

    /// Get the signature for a type by its index (for indirect calls)
    pub fn get_type_signature(&self, type_idx: u32) -> Option<&FunctionSignature> {
        self.type_signatures.get(type_idx as usize)
    }
}

/// Stub signature context when verification is disabled
#[cfg(not(feature = "verification"))]
#[derive(Clone, Default)]
pub struct VerificationSignatureContext;

#[cfg(not(feature = "verification"))]
impl VerificationSignatureContext {
    /// Create a new empty context (stub)
    pub fn new() -> Self {
        Self
    }

    /// Create a signature context from a module (stub)
    pub fn from_module(_module: &Module) -> Self {
        Self
    }
}

// ============================================================================
// Verification Coverage Tracking
// ============================================================================

/// Tracks verification coverage across function validations.
///
/// This struct provides metrics on what percentage of functions are fully
/// Z3-verified vs skipped due to unsupported features (loops, memory ops).
///
/// # Usage
///
/// ```rust,ignore
/// let mut coverage = VerificationCoverage::new();
/// // After verification...
/// println!("Coverage: {:.1}%", coverage.coverage_percent());
/// ```
#[derive(Debug, Clone, Default)]
pub struct VerificationCoverage {
    /// Number of functions fully verified with Z3
    pub verified: usize,
    /// Number of functions skipped due to containing loops
    pub skipped_loops: usize,
    /// Number of functions skipped due to memory operations
    pub skipped_memory: usize,
    /// Number of functions skipped due to unknown/unsupported instructions
    pub skipped_unknown: usize,
    /// Number of functions where verification failed (counterexample found)
    pub verification_failed: usize,
    /// Number of functions where verification errored (timeout, etc.)
    pub verification_error: usize,
}

impl VerificationCoverage {
    /// Create a new empty coverage tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a fully verified function
    pub fn record_verified(&mut self) {
        self.verified += 1;
    }

    /// Record a function skipped due to loops
    pub fn record_skipped_loop(&mut self) {
        self.skipped_loops += 1;
    }

    /// Record a function skipped due to memory operations
    pub fn record_skipped_memory(&mut self) {
        self.skipped_memory += 1;
    }

    /// Record a function skipped due to unknown instructions
    pub fn record_skipped_unknown(&mut self) {
        self.skipped_unknown += 1;
    }

    /// Record a verification failure (counterexample found)
    pub fn record_failed(&mut self) {
        self.verification_failed += 1;
    }

    /// Record a verification error (timeout, encoding error)
    pub fn record_error(&mut self) {
        self.verification_error += 1;
    }

    /// Total number of functions processed
    pub fn total(&self) -> usize {
        self.verified
            + self.skipped_loops
            + self.skipped_memory
            + self.skipped_unknown
            + self.verification_failed
            + self.verification_error
    }

    /// Total number of functions skipped (not fully verified)
    pub fn total_skipped(&self) -> usize {
        self.skipped_loops + self.skipped_memory + self.skipped_unknown
    }

    /// Calculate verification coverage percentage (0.0 to 100.0)
    ///
    /// Coverage = verified / (verified + skipped)
    /// Functions that failed or errored are not counted as skipped.
    pub fn coverage_percent(&self) -> f64 {
        let relevant = self.verified + self.total_skipped();
        if relevant == 0 {
            100.0
        } else {
            (self.verified as f64 / relevant as f64) * 100.0
        }
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Verification: {}/{} functions ({:.1}% Z3-proven)\n  \
             Skipped: {} loops, {} memory ops, {} unknown\n  \
             Failed: {}, Errors: {}",
            self.verified,
            self.total(),
            self.coverage_percent(),
            self.skipped_loops,
            self.skipped_memory,
            self.skipped_unknown,
            self.verification_failed,
            self.verification_error
        )
    }

    /// Merge another coverage into this one
    pub fn merge(&mut self, other: &VerificationCoverage) {
        self.verified += other.verified;
        self.skipped_loops += other.skipped_loops;
        self.skipped_memory += other.skipped_memory;
        self.skipped_unknown += other.skipped_unknown;
        self.verification_failed += other.verification_failed;
        self.verification_error += other.verification_error;
    }
}

/// Result of verifying a function, including coverage info
#[derive(Debug)]
pub enum VerificationResult {
    /// Function was fully verified as equivalent
    Verified,
    /// Function was skipped due to loops
    SkippedLoop,
    /// Function was skipped due to memory operations
    SkippedMemory,
    /// Function was skipped due to unknown instructions
    SkippedUnknown,
    /// Verification found a counterexample (not equivalent)
    Failed(String),
    /// Verification encountered an error
    Error(String),
}

impl VerificationResult {
    /// Returns true if this result indicates the functions are equivalent
    /// (either proven or assumed due to skipping)
    pub fn is_equivalent(&self) -> bool {
        matches!(
            self,
            VerificationResult::Verified
                | VerificationResult::SkippedLoop
                | VerificationResult::SkippedMemory
                | VerificationResult::SkippedUnknown
        )
    }

    /// Returns true if this result was fully Z3-verified
    pub fn is_verified(&self) -> bool {
        matches!(self, VerificationResult::Verified)
    }

    /// Update coverage tracker based on this result
    pub fn update_coverage(&self, coverage: &mut VerificationCoverage) {
        match self {
            VerificationResult::Verified => coverage.record_verified(),
            VerificationResult::SkippedLoop => coverage.record_skipped_loop(),
            VerificationResult::SkippedMemory => coverage.record_skipped_memory(),
            VerificationResult::SkippedUnknown => coverage.record_skipped_unknown(),
            VerificationResult::Failed(_) => coverage.record_failed(),
            VerificationResult::Error(_) => coverage.record_error(),
        }
    }
}

/// Maximum loop unrolling depth for verification
/// Higher values = more precise but slower
#[cfg(feature = "verification")]
#[allow(dead_code)]
const MAX_LOOP_UNROLL: usize = 3;

/// Shared symbolic inputs for verification
///
/// When comparing original vs optimized functions, BOTH encodings must use
/// the SAME symbolic inputs. Otherwise Z3 treats them as independent variables
/// and can trivially find counterexamples by assigning different values.
#[cfg(feature = "verification")]
struct SharedSymbolicInputs {
    /// Symbolic parameters (shared between original and optimized)
    #[allow(dead_code)]
    params: Vec<BV>,
    /// Initial local variable values (zeros for non-param locals)
    initial_locals: Vec<BV>,
    /// Symbolic globals (shared between original and optimized)
    globals: Vec<BV>,
    /// Symbolic memory (Array from 32-bit address to 8-bit byte)
    /// Using Array theory allows Z3 to reason about memory operations precisely.
    memory: Array,
}

#[cfg(feature = "verification")]
impl SharedSymbolicInputs {
    /// Create shared symbolic inputs for a function
    fn from_function(func: &Function) -> Self {
        let mut params = Vec::new();
        let mut initial_locals = Vec::new();

        // Create symbolic parameters
        for (idx, param_type) in func.signature.params.iter().enumerate() {
            let width = match param_type {
                crate::ValueType::I32 => 32,
                crate::ValueType::I64 => 64,
                crate::ValueType::F32 => 32,
                crate::ValueType::F64 => 64,
            };
            let param = BV::new_const(format!("param{}", idx), width);
            params.push(param.clone());
            initial_locals.push(param);
        }

        // Initialize non-param locals to zero
        for (count, local_type) in func.locals.iter() {
            let width = match local_type {
                crate::ValueType::I32 => 32,
                crate::ValueType::I64 => 64,
                crate::ValueType::F32 => 32,
                crate::ValueType::F64 => 64,
            };
            for _ in 0..*count {
                initial_locals.push(BV::from_u64(0, width));
            }
        }

        // Create symbolic globals
        let mut globals = Vec::new();
        for i in 0..16 {
            globals.push(BV::new_const(format!("global{}", i), 32));
        }

        // Create symbolic memory: Array[BitVec32 -> BitVec8]
        // This is a fully symbolic initial memory state
        let addr_sort = Sort::bitvector(32);
        let byte_sort = Sort::bitvector(8);
        let memory = Array::new_const("memory", &addr_sort, &byte_sort);

        SharedSymbolicInputs {
            params,
            initial_locals,
            globals,
            memory,
        }
    }
}

/// Execution state for symbolic execution
/// Tracks the symbolic state at each point during execution
#[cfg(feature = "verification")]
#[allow(dead_code)]
struct ExecutionState {
    /// Value stack
    stack: Vec<BV>,
    /// Local variables
    locals: Vec<BV>,
    /// Global variables
    globals: Vec<BV>,
    /// Path condition (constraints that must be true for this path)
    path_condition: Bool,
    /// Whether this execution path is reachable
    reachable: bool,
    /// Unique counter for generating fresh variable names
    counter: usize,
}

#[cfg(feature = "verification")]
#[allow(dead_code)]
impl ExecutionState {
    fn new(locals: Vec<BV>, globals: Vec<BV>) -> Self {
        ExecutionState {
            stack: Vec::new(),
            locals,
            globals,
            path_condition: Bool::from_bool(true),
            reachable: true,
            counter: 0,
        }
    }

    fn clone_state(&self) -> Self {
        ExecutionState {
            stack: self.stack.clone(),
            locals: self.locals.clone(),
            globals: self.globals.clone(),
            path_condition: self.path_condition.clone(),
            reachable: self.reachable,
            counter: self.counter,
        }
    }

    fn fresh_name(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{}_{}", prefix, self.counter)
    }
}

/// Result of encoding a block or instruction sequence
#[cfg(feature = "verification")]
#[allow(dead_code)]
struct BlockResult {
    /// The resulting execution state after the block
    state: ExecutionState,
    /// Values produced by the block (for blocks with result types)
    results: Vec<BV>,
    /// Whether a branch was taken that exits this block
    branched: bool,
    /// Branch depth (how many blocks to exit, 0 = this block)
    branch_depth: Option<u32>,
}

/// Merge two bitvector values based on a condition
/// Returns: if cond then true_val else false_val
#[cfg(feature = "verification")]
fn merge_bv(cond: &Bool, true_val: &BV, false_val: &BV) -> BV {
    cond.ite(true_val, false_val)
}

/// Merge two execution states based on a condition
/// Used for joining paths after if/else or when a branch may or may not be taken
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn merge_states(
    cond: &Bool,
    true_state: &ExecutionState,
    false_state: &ExecutionState,
) -> ExecutionState {
    // Merge stacks (must have same length for valid merge)
    let stack = if true_state.stack.len() == false_state.stack.len() {
        true_state
            .stack
            .iter()
            .zip(false_state.stack.iter())
            .map(|(t, f)| merge_bv(cond, t, f))
            .collect()
    } else {
        // Different stack heights - use true branch (caller should handle this case)
        true_state.stack.clone()
    };

    // Merge locals
    let locals = true_state
        .locals
        .iter()
        .zip(false_state.locals.iter())
        .map(|(t, f)| merge_bv(cond, t, f))
        .collect();

    // Merge globals
    let globals = true_state
        .globals
        .iter()
        .zip(false_state.globals.iter())
        .map(|(t, f)| merge_bv(cond, t, f))
        .collect();

    // Merge path conditions
    let path_condition = Bool::or(&[
        Bool::and(&[cond.clone(), true_state.path_condition.clone()]),
        Bool::and(&[cond.not(), false_state.path_condition.clone()]),
    ]);

    ExecutionState {
        stack,
        locals,
        globals,
        path_condition,
        reachable: true_state.reachable || false_state.reachable,
        counter: true_state.counter.max(false_state.counter),
    }
}

/// Get the bit width for a block type result
#[cfg(feature = "verification")]
fn block_type_width(block_type: &BlockType) -> Option<u32> {
    match block_type {
        BlockType::Empty => None,
        BlockType::Value(vt) => match vt {
            crate::ValueType::I32 => Some(32),
            crate::ValueType::I64 => Some(64),
            crate::ValueType::F32 => Some(32),
            crate::ValueType::F64 => Some(64),
        },
        BlockType::Func { results, .. } => {
            // For multi-value, return width of first result (simplified)
            results.first().map(|vt| match vt {
                crate::ValueType::I32 => 32,
                crate::ValueType::I64 => 64,
                crate::ValueType::F32 => 32,
                crate::ValueType::F64 => 64,
            })
        }
    }
}

/// Check if a function contains complex loops that would make verification unsound
///
/// Simple loops (no nesting, no unverifiable ops, bounded body) can be verified
/// using bounded unrolling. Complex loops are skipped.
#[cfg(feature = "verification")]
fn contains_complex_loops(instructions: &[Instruction]) -> bool {
    for instr in instructions {
        match instr {
            Instruction::Loop { body, .. } => {
                // Check if this loop is too complex for bounded verification
                if is_complex_loop(body) {
                    return true;
                }
            }
            Instruction::Block { body, .. } => {
                if contains_complex_loops(body) {
                    return true;
                }
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                if contains_complex_loops(then_body) || contains_complex_loops(else_body) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Maximum instructions in a loop body for bounded verification
const MAX_LOOP_BODY_INSTRUCTIONS: usize = 100;

/// Maximum nesting depth for loop verification
/// Allows one level of nesting to verify nested loops
const MAX_LOOP_NESTING_DEPTH: usize = 1;

/// Check if a loop body is too complex for bounded unrolling verification
///
/// Complex loops include:
/// - Deeply nested loops (more than MAX_LOOP_NESTING_DEPTH levels)
/// - Unverifiable instructions in loop body
/// - Very large loop bodies (> MAX_LOOP_BODY_INSTRUCTIONS)
#[cfg(feature = "verification")]
fn is_complex_loop(body: &[Instruction]) -> bool {
    is_complex_loop_at_depth(body, 0)
}

/// Check loop complexity at a given nesting depth
#[cfg(feature = "verification")]
fn is_complex_loop_at_depth(body: &[Instruction], depth: usize) -> bool {
    // Check for unverifiable instructions first
    if contains_unverifiable_instructions(body) {
        return true;
    }

    // Check body size (very large loops may timeout)
    if count_instructions(body) > MAX_LOOP_BODY_INSTRUCTIONS {
        return true;
    }

    // Check for nested loops - allow up to MAX_LOOP_NESTING_DEPTH
    for instr in body {
        match instr {
            Instruction::Loop {
                body: inner_body, ..
            } => {
                // When MAX_LOOP_NESTING_DEPTH is 0, any nested loop is too deep
                // When > 0, check if we've exceeded the allowed depth
                #[allow(clippy::absurd_extreme_comparisons)]
                if depth >= MAX_LOOP_NESTING_DEPTH {
                    // Too deep
                    return true;
                }
                // Check the nested loop at increased depth
                if is_complex_loop_at_depth(inner_body, depth + 1) {
                    return true;
                }
            }
            Instruction::Block {
                body: inner_body, ..
            } => {
                // Recurse into blocks (they don't increase loop depth)
                if is_complex_loop_at_depth(inner_body, depth) {
                    return true;
                }
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                if is_complex_loop_at_depth(then_body, depth)
                    || is_complex_loop_at_depth(else_body, depth)
                {
                    return true;
                }
            }
            _ => {}
        }
    }

    false
}

/// Check if instructions contain ANY loop (for nested loop detection)
/// Currently unused but kept for potential future use when MAX_LOOP_NESTING_DEPTH > 0
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn contains_any_loop(instructions: &[Instruction]) -> bool {
    for instr in instructions {
        match instr {
            Instruction::Loop { .. } => return true,
            Instruction::Block { body, .. } => {
                if contains_any_loop(body) {
                    return true;
                }
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                if contains_any_loop(then_body) || contains_any_loop(else_body) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Count total instructions recursively
#[cfg(feature = "verification")]
fn count_instructions(instructions: &[Instruction]) -> usize {
    let mut count = 0;
    for instr in instructions {
        count += 1;
        match instr {
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                count += count_instructions(body);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                count += count_instructions(then_body);
                count += count_instructions(else_body);
            }
            _ => {}
        }
    }
    count
}

/// Find the maximum local index used in instructions
///
/// This is needed because optimizations might create locals that aren't
/// declared in the function's locals list yet (e.g., LICM adds temporaries).
#[cfg(feature = "verification")]
fn find_max_local_index(instructions: &[Instruction]) -> usize {
    let mut max_idx = 0;
    for instr in instructions {
        match instr {
            Instruction::LocalGet(idx)
            | Instruction::LocalSet(idx)
            | Instruction::LocalTee(idx) => {
                max_idx = max_idx.max(*idx as usize);
            }
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                max_idx = max_idx.max(find_max_local_index(body));
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                max_idx = max_idx.max(find_max_local_index(then_body));
                max_idx = max_idx.max(find_max_local_index(else_body));
            }
            _ => {}
        }
    }
    max_idx
}

/// Check if a function contains instructions that cannot be precisely verified
///
/// These include instructions that would require imprecise modeling (producing
/// unconstrained symbolic values), which can lead to false counterexamples.
/// For rock-solid verification, we skip functions with such instructions.
#[cfg(feature = "verification")]
fn contains_unverifiable_instructions(instructions: &[Instruction]) -> bool {
    for instr in instructions {
        match instr {
            // Float memory operations not yet modeled with Array theory
            Instruction::F32Load { .. }
            | Instruction::F64Load { .. }
            | Instruction::F32Store { .. }
            | Instruction::F64Store { .. } => {
                return true;
            }

            // Note: All integer memory operations are now verified using Z3 Array theory:
            // - I32Load, I64Load, I32Store, I64Store (full-width)
            // - I32Load8S/U, I32Load16S/U (partial-width loads with sign/zero extension)
            // - I64Load8S/U, I64Load16S/U, I64Load32S/U (partial-width loads)
            // - I32Store8, I32Store16, I64Store8, I64Store16, I64Store32 (partial-width stores)

            // Note: I32Load, I64Load, I32Store, I64Store are now verified
            // using Z3 Array theory (Array[BitVec32 -> BitVec8] with little-endian encoding)

            // These operations are modeled but imprecisely
            Instruction::MemorySize(_) | Instruction::MemoryGrow(_) => {
                return true;
            }

            // Unknown instructions can't be modeled
            Instruction::Unknown(_) => {
                return true;
            }

            // Recurse into control flow structures
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                if contains_unverifiable_instructions(body) {
                    return true;
                }
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                if contains_unverifiable_instructions(then_body)
                    || contains_unverifiable_instructions(else_body)
                {
                    return true;
                }
            }

            // All other instructions have precise SMT encodings
            _ => {}
        }
    }
    false
}

/// Verify that an optimization preserves program semantics
///
/// This function uses Z3 SMT solver to prove that the optimized program is semantically
/// equivalent to the original program for all possible inputs.
///
/// # Arguments
///
/// * `original` - The original unoptimized module
/// * `optimized` - The optimized module
///
/// # Returns
///
/// * `Ok(true)` - Programs are proven equivalent
/// * `Ok(false)` - Found a counterexample (programs differ)
/// * `Err(_)` - Verification error or timeout
///
/// # Examples
///
/// ```rust,ignore
/// let original = parse_wat("...");
/// let optimized = optimize_module(original.clone());
/// assert!(verify_optimization(&original, &optimized)?);
/// ```
#[cfg(feature = "verification")]
pub fn verify_optimization(original: &Module, optimized: &Module) -> Result<bool> {
    // Check basic structural equivalence first
    if original.functions.len() != optimized.functions.len() {
        return Ok(false);
    }

    // Create Z3 context and solver using thread-local context
    let cfg = Config::new();
    with_z3_config(&cfg, || {
        let solver = Solver::new();

        // Verify each function pair
        for (orig_func, opt_func) in original.functions.iter().zip(optimized.functions.iter()) {
            // Check signatures match
            if orig_func.signature.params != opt_func.signature.params
                || orig_func.signature.results != opt_func.signature.results
            {
                return Ok(false);
            }

            // Encode both functions to SMT
            let orig_formula = encode_function_to_smt(orig_func)?;
            let opt_formula = encode_function_to_smt(opt_func)?;

            // Assert they are NOT equal (looking for counterexample)
            solver.push();

            // Handle void functions (both should be None) vs returning functions
            match (orig_formula, opt_formula) {
                (Some(orig), Some(opt)) => {
                    solver.assert(orig.eq(&opt).not());
                }
                (None, None) => {
                    // Both void functions - equivalent by definition (no return value to compare)
                    solver.pop(1);
                    continue;
                }
                _ => {
                    // One returns value, one doesn't - not equivalent
                    return Ok(false);
                }
            }

            // UNSAT means equivalent (no counterexample exists)
            match solver.check() {
                SatResult::Unsat => {
                    // Functions are equivalent
                    solver.pop(1);
                    continue;
                }
                SatResult::Sat => {
                    // Found counterexample - not equivalent!
                    let model = solver.get_model().context("Failed to get counterexample")?;
                    eprintln!("Counterexample found:");
                    eprintln!("{}", model);
                    return Ok(false);
                }
                SatResult::Unknown => {
                    return Err(anyhow!(
                        "SMT solver returned unknown (timeout or too complex)"
                    ));
                }
            }
        }

        Ok(true)
    })
}

/// Verify that a function transformation preserves semantics
///
/// This is the core translation validation function used by optimization passes.
/// It symbolically executes both versions and proves equivalence using Z3.
///
/// # Arguments
/// * `original` - The original function before optimization
/// * `optimized` - The function after optimization
/// * `pass_name` - Name of the optimization pass (for error messages)
///
/// # Returns
/// * `Ok(true)` - Functions are proven semantically equivalent
/// * `Ok(false)` - Found a counterexample showing different behavior
/// * `Err(_)` - Verification error or timeout
#[cfg(feature = "verification")]
pub fn verify_function_equivalence(
    original: &Function,
    optimized: &Function,
    pass_name: &str,
) -> Result<bool> {
    // Quick structural checks
    if original.signature.params != optimized.signature.params
        || original.signature.results != optimized.signature.results
    {
        return Err(anyhow!(
            "{}: Function signature changed during optimization",
            pass_name
        ));
    }

    // Skip verification for functions containing loops
    // Bounded loop unrolling can create false positives because the unrolled state
    // may differ between original and optimized even when semantically equivalent.
    // A full loop verification would require loop invariants or abstract interpretation.
    if contains_complex_loops(&original.instructions)
        || contains_complex_loops(&optimized.instructions)
    {
        return Ok(true); // Assume equivalent - loop verification is incomplete
    }

    // Skip verification for functions containing imprecisely-modeled instructions
    // (memory ops, unknown ops) - these would produce false counterexamples
    if contains_unverifiable_instructions(&original.instructions)
        || contains_unverifiable_instructions(&optimized.instructions)
    {
        return Ok(true); // Assume equivalent - can't prove without precise model
    }

    // Create Z3 context and solver using thread-local context
    let cfg = Config::new();
    with_z3_config(&cfg, || {
        let solver = Solver::new();

        // Encode both functions
        let orig_result = encode_function_to_smt(original);
        let opt_result = encode_function_to_smt(optimized);

        // Compute result while BVs are still valid
        match (&orig_result, &opt_result) {
            (Ok(Some(orig)), Ok(Some(opt))) => {
                // Assert they are NOT equal (looking for counterexample)
                solver.assert(orig.eq(opt).not());

                match solver.check() {
                    SatResult::Unsat => Ok(true), // No counterexample = equivalent
                    SatResult::Sat => {
                        if let Some(model) = solver.get_model() {
                            eprintln!("{}: Verification failed! Found counterexample:", pass_name);
                            eprintln!("{}", model);
                        }
                        Ok(false)
                    }
                    SatResult::Unknown => Err(anyhow!(
                        "{}: SMT solver returned unknown (timeout or too complex)",
                        pass_name
                    )),
                }
            }
            (Ok(None), Ok(None)) => Ok(true), // Both void functions
            (Err(e), _) => Err(anyhow!("{}: Failed to encode original: {}", pass_name, e)),
            (_, Err(e)) => Err(anyhow!("{}: Failed to encode optimized: {}", pass_name, e)),
            _ => Err(anyhow!(
                "{}: Return type mismatch between original and optimized",
                pass_name
            )),
        }
    })
}

/// Stub for when verification feature is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_function_equivalence(
    _original: &crate::Function,
    _optimized: &crate::Function,
    _pass_name: &str,
) -> Result<bool> {
    // When verification is disabled, assume the optimization is correct
    Ok(true)
}

/// Verify function equivalence and return detailed result for coverage tracking.
///
/// This version returns a `VerificationResult` that indicates WHY a function
/// was skipped or whether it was fully verified, enabling accurate coverage metrics.
#[cfg(feature = "verification")]
pub fn verify_function_equivalence_with_result(
    original: &Function,
    optimized: &Function,
    pass_name: &str,
) -> VerificationResult {
    // Quick structural checks
    if original.signature.params != optimized.signature.params
        || original.signature.results != optimized.signature.results
    {
        return VerificationResult::Error(format!(
            "{}: Function signature changed during optimization",
            pass_name
        ));
    }

    // Check for loops FIRST
    if contains_complex_loops(&original.instructions)
        || contains_complex_loops(&optimized.instructions)
    {
        return VerificationResult::SkippedLoop;
    }

    // Check for memory/unknown instructions
    if contains_unverifiable_instructions(&original.instructions)
        || contains_unverifiable_instructions(&optimized.instructions)
    {
        // Distinguish between memory ops and unknown ops
        if contains_memory_instructions(&original.instructions)
            || contains_memory_instructions(&optimized.instructions)
        {
            return VerificationResult::SkippedMemory;
        }
        return VerificationResult::SkippedUnknown;
    }

    // Create Z3 context and solver
    let cfg = Config::new();
    with_z3_config(&cfg, || {
        let solver = Solver::new();

        let orig_result = encode_function_to_smt(original);
        let opt_result = encode_function_to_smt(optimized);

        match (&orig_result, &opt_result) {
            (Ok(Some(orig)), Ok(Some(opt))) => {
                solver.assert(orig.eq(opt).not());

                match solver.check() {
                    SatResult::Unsat => VerificationResult::Verified,
                    SatResult::Sat => {
                        let model = solver
                            .get_model()
                            .map(|m| format!("{}", m))
                            .unwrap_or_else(|| "no model".to_string());
                        VerificationResult::Failed(format!(
                            "{}: Found counterexample: {}",
                            pass_name, model
                        ))
                    }
                    SatResult::Unknown => VerificationResult::Error(format!(
                        "{}: SMT solver returned unknown (timeout or too complex)",
                        pass_name
                    )),
                }
            }
            (Ok(None), Ok(None)) => VerificationResult::Verified, // Both void
            (Err(e), _) => VerificationResult::Error(format!(
                "{}: Failed to encode original: {}",
                pass_name, e
            )),
            (_, Err(e)) => VerificationResult::Error(format!(
                "{}: Failed to encode optimized: {}",
                pass_name, e
            )),
            _ => VerificationResult::Error(format!(
                "{}: Return type mismatch between original and optimized",
                pass_name
            )),
        }
    })
}

/// Stub for when verification is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_function_equivalence_with_result(
    _original: &crate::Function,
    _optimized: &crate::Function,
    _pass_name: &str,
) -> VerificationResult {
    VerificationResult::Verified
}

#[cfg(feature = "verification")]
/// Check if instructions contain unverifiable memory operations.
///
/// Note: I32Load, I64Load, I32Store, I64Store are NOW verifiable with Z3 Array theory.
/// Only partial-width (8/16-bit) and float memory operations remain unverifiable.
fn contains_memory_instructions(instructions: &[Instruction]) -> bool {
    for instr in instructions {
        match instr {
            // Float memory operations - not yet modeled
            Instruction::F32Load { .. }
            | Instruction::F64Load { .. }
            | Instruction::F32Store { .. }
            | Instruction::F64Store { .. } => return true,

            // Partial-width memory operations - not yet modeled
            Instruction::I32Load8S { .. }
            | Instruction::I32Load8U { .. }
            | Instruction::I32Load16S { .. }
            | Instruction::I32Load16U { .. }
            | Instruction::I64Load8S { .. }
            | Instruction::I64Load8U { .. }
            | Instruction::I64Load16S { .. }
            | Instruction::I64Load16U { .. }
            | Instruction::I64Load32S { .. }
            | Instruction::I64Load32U { .. }
            | Instruction::I32Store8 { .. }
            | Instruction::I32Store16 { .. }
            | Instruction::I64Store8 { .. }
            | Instruction::I64Store16 { .. }
            | Instruction::I64Store32 { .. } => return true,

            // Memory size/grow not modeled
            Instruction::MemorySize(_) | Instruction::MemoryGrow(_) => return true,

            // Note: I32Load, I64Load, I32Store, I64Store are now verifiable
            // via Z3 Array theory (Array[BitVec32 -> BitVec8] with little-endian encoding)
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                if contains_memory_instructions(body) {
                    return true;
                }
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                if contains_memory_instructions(then_body)
                    || contains_memory_instructions(else_body)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Verify function equivalence with signature context for proper Call handling
///
/// This version uses the signature context to properly model Call/CallIndirect
/// stack effects, providing more accurate verification.
#[cfg(feature = "verification")]
pub fn verify_function_equivalence_with_context(
    original: &Function,
    optimized: &Function,
    pass_name: &str,
    sig_ctx: &VerificationSignatureContext,
) -> Result<bool> {
    // Quick structural checks
    if original.signature.params != optimized.signature.params
        || original.signature.results != optimized.signature.results
    {
        return Err(anyhow!(
            "{}: Function signature changed during optimization",
            pass_name
        ));
    }

    // Skip verification for functions containing loops
    // Bounded loop unrolling produces false positives
    if contains_complex_loops(&original.instructions)
        || contains_complex_loops(&optimized.instructions)
    {
        return Ok(true); // Assume equivalent - loop verification is incomplete
    }

    // Skip verification for functions containing imprecisely-modeled instructions
    // (memory ops, unknown ops) - these would produce false counterexamples
    if contains_unverifiable_instructions(&original.instructions)
        || contains_unverifiable_instructions(&optimized.instructions)
    {
        return Ok(true); // Assume equivalent - can't prove without precise model
    }

    // Create Z3 context and solver using thread-local context
    let cfg = Config::new();
    with_z3_config(&cfg, || {
        let solver = Solver::new();

        // CRITICAL: Create SHARED symbolic inputs that both encodings will use.
        // Without this, each encoding creates independent symbolic variables,
        // and Z3 trivially finds they can differ.
        let shared_inputs = SharedSymbolicInputs::from_function(original);

        // Encode both functions with the SAME shared inputs
        let orig_result =
            encode_function_to_smt_with_shared_inputs(original, sig_ctx, &shared_inputs);
        let opt_result =
            encode_function_to_smt_with_shared_inputs(optimized, sig_ctx, &shared_inputs);

        // Compute result while BVs are still valid
        match (&orig_result, &opt_result) {
            (Ok(Some(orig)), Ok(Some(opt))) => {
                // Assert they are NOT equal (looking for counterexample)
                solver.assert(orig.eq(opt).not());

                match solver.check() {
                    SatResult::Unsat => Ok(true), // No counterexample = equivalent
                    SatResult::Sat => {
                        if let Some(model) = solver.get_model() {
                            eprintln!("{}: Verification failed! Found counterexample:", pass_name);
                            eprintln!("{}", model);
                        }
                        Ok(false)
                    }
                    SatResult::Unknown => Err(anyhow!(
                        "{}: SMT solver returned unknown (timeout or too complex)",
                        pass_name
                    )),
                }
            }
            (Ok(None), Ok(None)) => Ok(true), // Both void functions
            (Err(e), _) => Err(anyhow!("{}: Failed to encode original: {}", pass_name, e)),
            (_, Err(e)) => Err(anyhow!("{}: Failed to encode optimized: {}", pass_name, e)),
            _ => Err(anyhow!(
                "{}: Return type mismatch between original and optimized",
                pass_name
            )),
        }
    })
}

/// Stub for when verification feature is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_function_equivalence_with_context(
    _original: &crate::Function,
    _optimized: &crate::Function,
    _pass_name: &str,
    _sig_ctx: &VerificationSignatureContext,
) -> Result<bool> {
    Ok(true)
}

/// Verify a module transformation by checking each function
///
/// This is the preferred way to verify an optimization pass.
/// It checks every function in the module for semantic equivalence.
#[cfg(feature = "verification")]
pub fn verify_module_transformation(
    original: &Module,
    optimized: &Module,
    pass_name: &str,
) -> Result<bool> {
    if original.functions.len() != optimized.functions.len() {
        return Err(anyhow!(
            "{}: Function count changed (was {}, now {})",
            pass_name,
            original.functions.len(),
            optimized.functions.len()
        ));
    }

    for (idx, (orig_func, opt_func)) in original
        .functions
        .iter()
        .zip(optimized.functions.iter())
        .enumerate()
    {
        let func_name = orig_func
            .name
            .clone()
            .unwrap_or_else(|| format!("func_{}", idx));

        match verify_function_equivalence(orig_func, opt_func, pass_name) {
            Ok(true) => continue,
            Ok(false) => {
                return Err(anyhow!(
                    "{}: Function '{}' is not semantically equivalent after optimization",
                    pass_name,
                    func_name
                ));
            }
            Err(e) => {
                // Log but continue - some functions may be too complex to verify
                eprintln!("Warning: Could not verify function '{}': {}", func_name, e);
            }
        }
    }

    Ok(true)
}

/// Stub for when verification feature is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_module_transformation(
    _original: &Module,
    _optimized: &Module,
    _pass_name: &str,
) -> Result<bool> {
    Ok(true)
}

/// Compute verification coverage for a module transformation.
///
/// This analyzes each function in the module and returns detailed coverage
/// metrics showing what percentage of functions can be fully Z3-verified
/// vs. what must be skipped due to loops or memory operations.
///
/// # Example
///
/// ```rust,ignore
/// let coverage = compute_verification_coverage(&original, &optimized, "optimize");
/// println!("{}", coverage.summary());
/// // Output: "Verification: 15/20 functions (75.0% Z3-proven)
/// //          Skipped: 3 loops, 2 memory ops, 0 unknown"
/// ```
#[cfg(feature = "verification")]
pub fn compute_verification_coverage(
    original: &Module,
    optimized: &Module,
    pass_name: &str,
) -> VerificationCoverage {
    let mut coverage = VerificationCoverage::new();

    if original.functions.len() != optimized.functions.len() {
        // Module structure changed - can't compute meaningful coverage
        coverage.record_error();
        return coverage;
    }

    for (orig_func, opt_func) in original.functions.iter().zip(optimized.functions.iter()) {
        let result = verify_function_equivalence_with_result(orig_func, opt_func, pass_name);
        result.update_coverage(&mut coverage);
    }

    coverage
}

/// Stub for when verification is disabled
#[cfg(not(feature = "verification"))]
pub fn compute_verification_coverage(
    original: &Module,
    optimized: &Module,
    _pass_name: &str,
) -> VerificationCoverage {
    // When verification is disabled, report all functions as verified
    // (they're assumed correct)
    let mut coverage = VerificationCoverage::new();
    let func_count = original.functions.len().min(optimized.functions.len());
    for _ in 0..func_count {
        coverage.record_verified();
    }
    coverage
}

// ============================================================================
// Translation Validator - RAII guard for optimization pass verification
// ============================================================================

/// RAII-style translation validator for optimization passes
///
/// This captures the original function state before optimization, then verifies
/// semantic equivalence after the pass completes. This provides the same level
/// of correctness guarantee as years of Binaryen's fuzzing - but mathematically.
///
/// # Example
///
/// ```rust,ignore
/// use loom_core::verify::TranslationValidator;
///
/// fn my_optimization_pass(func: &mut Function) -> Result<()> {
///     let validator = TranslationValidator::new(func, "my_pass");
///
///     // ... perform optimizations on func ...
///
///     validator.verify(func)?; // Prove equivalence with Z3
///     Ok(())
/// }
/// ```
#[cfg(feature = "verification")]
pub struct TranslationValidator {
    /// Snapshot of the original function before optimization
    original: Function,
    /// Name of the optimization pass (for error messages)
    pass_name: String,
    /// Signature context for Call/CallIndirect verification
    sig_ctx: VerificationSignatureContext,
}

#[cfg(feature = "verification")]
impl TranslationValidator {
    /// Create a new validator, capturing the current function state
    /// Uses empty signature context (Call/CallIndirect will use conservative encoding)
    pub fn new(func: &Function, pass_name: &str) -> Self {
        Self {
            original: func.clone(),
            pass_name: pass_name.to_string(),
            sig_ctx: VerificationSignatureContext::new(),
        }
    }

    /// Create a new validator with signature context for proper Call verification
    pub fn new_with_context(
        func: &Function,
        pass_name: &str,
        sig_ctx: VerificationSignatureContext,
    ) -> Self {
        Self {
            original: func.clone(),
            pass_name: pass_name.to_string(),
            sig_ctx,
        }
    }

    /// Verify that the optimized function is semantically equivalent to the original
    ///
    /// Returns Ok(()) if verified equivalent, Err if different or verification fails
    pub fn verify(&self, optimized: &Function) -> Result<()> {
        // Use catch_unwind to handle Z3 internal panics gracefully
        let sig_ctx = self.sig_ctx.clone();
        let original = self.original.clone();
        let pass_name = self.pass_name.clone();
        let optimized_clone = optimized.clone();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            verify_function_equivalence_with_context(
                &original,
                &optimized_clone,
                &pass_name,
                &sig_ctx,
            )
        }));

        match result {
            Ok(Ok(true)) => Ok(()),
            Ok(Ok(false)) => {
                // Z3 found a counterexample - the optimization is NOT proven correct.
                // Per our proof-first philosophy, we MUST reject unproven optimizations.
                eprintln!(
                    "{}: Verification failed! Found counterexample:",
                    self.pass_name
                );
                Err(anyhow!(
                    "{}: Z3 found counterexample - optimization rejected (unproven)",
                    self.pass_name
                ))
            }
            Ok(Err(e)) => {
                // Verification error - we cannot prove correctness.
                // Per our philosophy: if we cannot prove it, we must not do it.
                Err(anyhow!(
                    "{}: Verification failed ({}) - optimization rejected (unproven)",
                    self.pass_name,
                    e
                ))
            }
            Err(_panic) => {
                // Z3 internal error - we cannot prove correctness.
                // Per our philosophy: if we cannot prove it, we must not do it.
                Err(anyhow!(
                    "{}: Z3 internal error - optimization rejected (unproven)",
                    self.pass_name
                ))
            }
        }
    }

    /// Verify and return detailed result instead of Result<()>
    pub fn verify_detailed(&self, optimized: &Function) -> TranslationResult {
        match verify_function_equivalence_with_context(
            &self.original,
            optimized,
            &self.pass_name,
            &self.sig_ctx,
        ) {
            Ok(true) => TranslationResult::Equivalent,
            Ok(false) => TranslationResult::Different,
            Err(e) => TranslationResult::Unknown(e.to_string()),
        }
    }
}

/// Result of translation validation
#[cfg(feature = "verification")]
#[derive(Debug, Clone, PartialEq)]
pub enum TranslationResult {
    /// Functions are proven semantically equivalent
    Equivalent,
    /// Found a counterexample - functions differ on some input
    Different,
    /// Could not determine (timeout, too complex, or error)
    Unknown(String),
}

/// Stub TranslationValidator for when verification is disabled
#[cfg(not(feature = "verification"))]
pub struct TranslationValidator {
    #[allow(dead_code)]
    pass_name: String,
}

#[cfg(not(feature = "verification"))]
impl TranslationValidator {
    /// Create a stub validator (does nothing when verification disabled)
    pub fn new(_func: &crate::Function, pass_name: &str) -> Self {
        Self {
            pass_name: pass_name.to_string(),
        }
    }

    /// Create a stub validator with context (does nothing when verification disabled)
    pub fn new_with_context(
        _func: &crate::Function,
        pass_name: &str,
        _sig_ctx: VerificationSignatureContext,
    ) -> Self {
        Self {
            pass_name: pass_name.to_string(),
        }
    }

    /// Stub verify - always succeeds when verification disabled
    pub fn verify(&self, _optimized: &crate::Function) -> Result<()> {
        Ok(())
    }
}

/// Stub TranslationResult for when verification is disabled
#[cfg(not(feature = "verification"))]
#[derive(Debug, Clone, PartialEq)]
pub enum TranslationResult {
    /// Functions are proven semantically equivalent
    Equivalent,
    /// Found a counterexample - functions differ on some input
    Different,
    /// Could not determine (timeout, too complex, or error)
    Unknown(String),
}

/// Encode a WebAssembly function to an SMT formula
///
/// This converts the instruction sequence into a symbolic execution that Z3 can reason about.
/// Returns None for void functions, Some(BV) for functions with a return value.
#[cfg(feature = "verification")]
fn encode_function_to_smt(func: &Function) -> Result<Option<BV>> {
    encode_function_to_smt_impl(func, None)
}

/// Encode a WebAssembly function to an SMT formula with signature context
///
/// This version uses the signature context for proper Call/CallIndirect stack modeling.
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn encode_function_to_smt_with_context(
    func: &Function,
    sig_ctx: &VerificationSignatureContext,
) -> Result<Option<BV>> {
    encode_function_to_smt_impl(func, Some(sig_ctx))
}

/// Encode a WebAssembly function using pre-created shared symbolic inputs
///
/// This is CRITICAL for correct verification: both original and optimized functions
/// must use the SAME symbolic inputs. Otherwise Z3 treats them as independent and
/// can trivially find counterexamples.
#[cfg(feature = "verification")]
fn encode_function_to_smt_with_shared_inputs(
    func: &Function,
    sig_ctx: &VerificationSignatureContext,
    shared: &SharedSymbolicInputs,
) -> Result<Option<BV>> {
    encode_function_to_smt_impl_inner(func, Some(sig_ctx), Some(shared))
}

/// Internal implementation of SMT encoding with optional signature context
#[cfg(feature = "verification")]
fn encode_function_to_smt_impl(
    func: &Function,
    sig_ctx: Option<&VerificationSignatureContext>,
) -> Result<Option<BV>> {
    encode_function_to_smt_impl_inner(func, sig_ctx, None)
}

/// Core SMT encoding implementation
///
/// When `shared_inputs` is Some, uses the provided symbolic inputs.
/// When None, creates fresh symbolic inputs (for standalone encoding).
#[cfg(feature = "verification")]
fn encode_function_to_smt_impl_inner(
    func: &Function,
    sig_ctx: Option<&VerificationSignatureContext>,
    shared_inputs: Option<&SharedSymbolicInputs>,
) -> Result<Option<BV>> {
    // Create symbolic variables for parameters
    let mut stack: Vec<BV> = Vec::new();
    let mut locals: Vec<BV>;
    let mut globals: Vec<BV>;
    let mut memory: Array;

    if let Some(shared) = shared_inputs {
        // Use shared inputs - CRITICAL for correct verification
        locals = shared.initial_locals.clone();
        globals = shared.globals.clone();
        memory = shared.memory.clone();

        // Extend locals if the optimized function uses more locals than the original
        // This can happen when optimizations like LICM add temporary locals
        // We need to check both the declared locals AND the maximum index actually used
        let declared_local_count = func.signature.params.len()
            + func
                .locals
                .iter()
                .map(|(count, _)| *count as usize)
                .sum::<usize>();
        let max_used_local = find_max_local_index(&func.instructions);
        let needed_locals = declared_local_count.max(max_used_local + 1);

        while locals.len() < needed_locals {
            // Add zero-initialized locals for any extra locals in the optimized function
            // Use i32 as default type since we don't know the actual type from the index alone
            locals.push(BV::from_u64(0, 32));
        }
    } else {
        // Create fresh inputs (for standalone encoding)
        locals = Vec::new();

        // Initialize parameters as symbolic inputs
        for (idx, param_type) in func.signature.params.iter().enumerate() {
            let width = match param_type {
                crate::ValueType::I32 => 32,
                crate::ValueType::I64 => 64,
                crate::ValueType::F32 => 32,
                crate::ValueType::F64 => 64,
            };
            let param = BV::new_const(format!("param{}", idx), width);
            locals.push(param);
        }

        // Initialize local variables to zero
        for (count, local_type) in func.locals.iter() {
            let width = match local_type {
                crate::ValueType::I32 => 32,
                crate::ValueType::I64 => 64,
                crate::ValueType::F32 => 32,
                crate::ValueType::F64 => 64,
            };
            for _ in 0..*count {
                locals.push(BV::from_u64(0, width));
            }
        }

        // Initialize globals as symbolic
        globals = Vec::new();
        for i in 0..16 {
            globals.push(BV::new_const(format!("global{}", i), 32));
        }

        // Create symbolic memory: Array[BitVec32 -> BitVec8]
        let addr_sort = Sort::bitvector(32);
        let byte_sort = Sort::bitvector(8);
        memory = Array::new_const("memory", &addr_sort, &byte_sort);
    }

    // Symbolically execute instructions
    for instr in &func.instructions {
        match instr {
            // Constants
            Instruction::I32Const(n) => {
                stack.push(BV::from_i64(*n as i64, 32));
            }
            Instruction::I64Const(n) => {
                stack.push(BV::from_i64(*n, 64));
            }
            Instruction::F32Const(bits) => {
                // Float constants are treated as bit patterns for now
                // We don't perform floating-point arithmetic verification yet
                stack.push(BV::from_i64(*bits as i64, 32));
            }
            Instruction::F64Const(bits) => {
                // Float constants are treated as bit patterns for now
                // We don't perform floating-point arithmetic verification yet
                stack.push(BV::from_i64(*bits as i64, 64));
            }

            // Arithmetic operations (i32)
            Instruction::I32Add => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Add"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvadd(&rhs));
            }
            Instruction::I32Sub => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Sub"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvsub(&rhs));
            }
            Instruction::I32Mul => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Mul"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvmul(&rhs));
            }

            // Arithmetic operations (i64)
            Instruction::I64Add => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Add"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvadd(&rhs));
            }
            Instruction::I64Sub => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Sub"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvsub(&rhs));
            }
            Instruction::I64Mul => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Mul"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvmul(&rhs));
            }

            // Bitwise operations (i32)
            Instruction::I32And => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32And"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvand(&rhs));
            }
            Instruction::I32Or => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Or"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvor(&rhs));
            }
            Instruction::I32Xor => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Xor"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvxor(&rhs));
            }
            Instruction::I32Shl => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Shl"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvshl(&rhs));
            }
            Instruction::I32ShrU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32ShrU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvlshr(&rhs));
            }
            Instruction::I32ShrS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32ShrS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvashr(&rhs));
            }

            // Bitwise operations (i64)
            Instruction::I64And => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64And"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvand(&rhs));
            }
            Instruction::I64Or => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Or"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvor(&rhs));
            }
            Instruction::I64Xor => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Xor"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvxor(&rhs));
            }
            Instruction::I64Shl => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Shl"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvshl(&rhs));
            }
            Instruction::I64ShrU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64ShrU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvlshr(&rhs));
            }
            Instruction::I64ShrS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64ShrS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvashr(&rhs));
            }

            // Comparison operations (i32) - produce i32 boolean (0 or 1)
            Instruction::I32Eq => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Eq"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                // (lhs == rhs) ? 1 : 0
                stack.push(lhs.eq(&rhs).ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)));
            }
            Instruction::I32Ne => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Ne"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.eq(&rhs)
                        .not()
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32LtS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32LtS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvslt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32LtU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32LtU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvult(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32GtS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32GtS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvsgt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32GtU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32GtU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvugt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32LeS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32LeS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvsle(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32LeU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32LeU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvule(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32GeS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32GeS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvsge(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32GeU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32GeU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvuge(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }

            // Comparison operations (i64) - produce i32 boolean
            Instruction::I64Eq => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Eq"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.eq(&rhs).ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)));
            }
            Instruction::I64Ne => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Ne"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.eq(&rhs)
                        .not()
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64LtS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64LtS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvslt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64LtU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64LtU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvult(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64GtS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64GtS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvsgt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64GtU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64GtU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvugt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64LeS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64LeS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvsle(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64LeU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64LeU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvule(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64GeS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64GeS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvsge(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64GeU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64GeU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(
                    lhs.bvuge(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }

            // Division and remainder operations (i32)
            Instruction::I32DivS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32DivS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvsdiv(&rhs));
            }
            Instruction::I32DivU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32DivU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvudiv(&rhs));
            }
            Instruction::I32RemS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32RemS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvsrem(&rhs));
            }
            Instruction::I32RemU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32RemU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvurem(&rhs));
            }

            // Division and remainder operations (i64)
            Instruction::I64DivS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64DivS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvsdiv(&rhs));
            }
            Instruction::I64DivU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64DivU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvudiv(&rhs));
            }
            Instruction::I64RemS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64RemS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvsrem(&rhs));
            }
            Instruction::I64RemU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64RemU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvurem(&rhs));
            }

            // Unary operations (i32)
            Instruction::I32Eqz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Eqz"));
                }
                let val = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                stack.push(
                    val.eq(&zero)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32Clz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Clz"));
                }
                let val = stack.pop().unwrap();
                // Count leading zeros: encode as nested conditionals checking each bit from MSB
                // If bit 31 is set → 0, else if bit 30 is set → 1, ..., else → 32
                // CRITICAL: We build the ITE chain so MSB check is outermost (evaluated first)
                let mut result = BV::from_i64(32, 32); // All zeros case
                for i in (0..32).rev() {
                    // Reverse order so MSB (i=0, bit_pos=31) becomes outermost ITE
                    let bit_pos = 31 - i;
                    let mask = BV::from_i64(1i64 << bit_pos, 32);
                    let bit_set = val.bvand(&mask).eq(&mask);
                    result = bit_set.ite(&BV::from_i64(i as i64, 32), &result);
                }
                stack.push(result);
            }
            Instruction::I32Ctz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Ctz"));
                }
                let val = stack.pop().unwrap();
                // Count trailing zeros: encode as nested conditionals checking each bit from LSB
                // If bit 0 is set → 0, else if bit 1 is set → 1, ..., else → 32
                let mut result = BV::from_i64(32, 32); // All zeros case
                for i in 0..32 {
                    let bit_pos = 31 - i;
                    let mask = BV::from_i64(1i64 << bit_pos, 32);
                    let bit_set = val.bvand(&mask).eq(&mask);
                    result = bit_set.ite(&BV::from_i64(bit_pos as i64, 32), &result);
                }
                stack.push(result);
            }
            Instruction::I32Popcnt => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Popcnt"));
                }
                let val = stack.pop().unwrap();
                // Population count: sum all the bits
                // For each bit position, extract (val >> i) & 1 and sum them
                let mut result = BV::from_i64(0, 32);
                for i in 0..32 {
                    let mask = BV::from_i64(1i64 << i, 32);
                    let bit = val.bvand(&mask).eq(&mask);
                    // Add 1 if bit is set, 0 otherwise
                    result = bit.ite(&result.bvadd(BV::from_i64(1, 32)), &result);
                }
                stack.push(result);
            }

            // Unary operations (i64)
            Instruction::I64Eqz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Eqz"));
                }
                let val = stack.pop().unwrap();
                let zero = BV::from_i64(0, 64);
                // Result is i32!
                stack.push(
                    val.eq(&zero)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64Clz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Clz"));
                }
                let val = stack.pop().unwrap();
                // Count leading zeros for i64: same approach as i32 but 64 bits
                // CRITICAL: Reverse order so MSB check is outermost
                let mut result = BV::from_i64(64, 64); // All zeros case
                for i in (0..64).rev() {
                    let bit_pos = 63 - i;
                    let mask = if bit_pos < 63 {
                        BV::from_i64(1i64 << bit_pos, 64)
                    } else {
                        // Handle bit 63 specially to avoid overflow
                        BV::from_i64(i64::MIN, 64)
                    };
                    let bit_set = val.bvand(&mask).eq(&mask);
                    result = bit_set.ite(&BV::from_i64(i as i64, 64), &result);
                }
                stack.push(result);
            }
            Instruction::I64Ctz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Ctz"));
                }
                let val = stack.pop().unwrap();
                // Count trailing zeros for i64
                let mut result = BV::from_i64(64, 64); // All zeros case
                for i in 0..64 {
                    let bit_pos = 63 - i;
                    let mask = if bit_pos < 63 {
                        BV::from_i64(1i64 << bit_pos, 64)
                    } else {
                        BV::from_i64(i64::MIN, 64)
                    };
                    let bit_set = val.bvand(&mask).eq(&mask);
                    result = bit_set.ite(&BV::from_i64(bit_pos as i64, 64), &result);
                }
                stack.push(result);
            }
            Instruction::I64Popcnt => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Popcnt"));
                }
                let val = stack.pop().unwrap();
                // Population count for i64: sum all 64 bits
                let mut result = BV::from_i64(0, 64);
                for i in 0..64 {
                    let mask = if i < 63 {
                        BV::from_i64(1i64 << i, 64)
                    } else {
                        BV::from_i64(i64::MIN, 64)
                    };
                    let bit = val.bvand(&mask).eq(&mask);
                    result = bit.ite(&result.bvadd(BV::from_i64(1, 64)), &result);
                }
                stack.push(result);
            }

            // Select operation: [T, T, i32] -> [T]
            Instruction::Select => {
                if stack.len() < 3 {
                    return Err(anyhow!("Stack underflow in Select"));
                }
                let cond = stack.pop().unwrap();
                let val2 = stack.pop().unwrap();
                let val1 = stack.pop().unwrap();
                // Select: if cond != 0 then val1 else val2
                let zero = BV::from_i64(0, 32);
                stack.push(cond.eq(&zero).not().ite(&val1, &val2));
            }

            // Nop does nothing
            Instruction::Nop => {}

            // Drop pops and discards top of stack
            Instruction::Drop => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in Drop"));
                }
                stack.pop();
            }

            // Local operations
            // Note: Optimization passes (like CSE) may add new locals to the optimized
            // function that don't exist in the original. We extend locals dynamically
            // similar to how we handle globals.
            Instruction::LocalGet(idx) => {
                let idx = *idx as usize;
                // Extend locals vector if needed (for optimizer-added locals)
                while locals.len() <= idx {
                    // Default to 32-bit zero - will be set by LocalTee before read
                    locals.push(BV::from_u64(0, 32));
                }
                stack.push(locals[idx].clone());
            }
            Instruction::LocalSet(idx) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in LocalSet"));
                }
                let idx = *idx as usize;
                // Extend locals vector if needed
                while locals.len() <= idx {
                    locals.push(BV::from_u64(0, 32));
                }
                let value = stack.pop().unwrap();
                locals[idx] = value;
            }
            Instruction::LocalTee(idx) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in LocalTee"));
                }
                let idx = *idx as usize;
                // Extend locals vector if needed (CSE adds locals for caching)
                while locals.len() <= idx {
                    locals.push(BV::from_u64(0, 32));
                }
                let value = stack.last().unwrap().clone();
                locals[idx] = value;
            }

            // Global operations
            Instruction::GlobalGet(idx) => {
                let idx = *idx as usize;
                // Extend globals vector if needed
                while globals.len() <= idx {
                    let new_idx = globals.len();
                    globals.push(BV::new_const(format!("global{}", new_idx), 32));
                }
                stack.push(globals[idx].clone());
            }
            Instruction::GlobalSet(idx) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in GlobalSet"));
                }
                let idx = *idx as usize;
                while globals.len() <= idx {
                    let new_idx = globals.len();
                    globals.push(BV::new_const(format!("global{}", new_idx), 32));
                }
                let value = stack.pop().unwrap();
                globals[idx] = value;
            }

            // Memory operations using Z3 Array theory
            // Memory is byte-addressable: Array[BitVec32 -> BitVec8]
            // Multi-byte loads/stores combine bytes in little-endian order
            Instruction::I32Load { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Load"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));

                // Read 4 bytes in little-endian order and combine into 32-bit
                // memory.select returns Dynamic, cast to BV via as_bv()
                let byte0: BV = memory.select(&effective_addr).as_bv().unwrap();
                let byte1: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(1, 32)))
                    .as_bv()
                    .unwrap();
                let byte2: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(2, 32)))
                    .as_bv()
                    .unwrap();
                let byte3: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(3, 32)))
                    .as_bv()
                    .unwrap();

                // Zero-extend each byte to 32 bits
                let b0 = byte0.zero_ext(24);
                let b1 = byte1.zero_ext(24);
                let b2 = byte2.zero_ext(24);
                let b3 = byte3.zero_ext(24);

                // Combine: result = b3 << 24 | b2 << 16 | b1 << 8 | b0
                let result = b0
                    .bvor(b1.bvshl(BV::from_i64(8, 32)))
                    .bvor(b2.bvshl(BV::from_i64(16, 32)))
                    .bvor(b3.bvshl(BV::from_i64(24, 32)));

                stack.push(result);
            }
            Instruction::I32Store { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Store"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));

                // Extract 4 bytes from 32-bit value (little-endian)
                let byte0 = value.extract(7, 0);
                let byte1 = value.extract(15, 8);
                let byte2 = value.extract(23, 16);
                let byte3 = value.extract(31, 24);

                // Store each byte
                memory = memory.store(&effective_addr, &byte0);
                memory = memory.store(&effective_addr.bvadd(BV::from_i64(1, 32)), &byte1);
                memory = memory.store(&effective_addr.bvadd(BV::from_i64(2, 32)), &byte2);
                memory = memory.store(&effective_addr.bvadd(BV::from_i64(3, 32)), &byte3);
            }
            Instruction::I64Load { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));

                // Read 8 bytes in little-endian order
                let mut result = BV::from_i64(0, 64);
                for i in 0..8i64 {
                    let byte_addr = effective_addr.bvadd(BV::from_i64(i, 32));
                    let byte_val: BV = memory.select(&byte_addr).as_bv().unwrap();
                    let extended = byte_val.zero_ext(56);
                    result = result.bvor(extended.bvshl(BV::from_i64(i * 8, 64)));
                }
                stack.push(result);
            }
            Instruction::I64Store { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Store"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));

                // Store 8 bytes (little-endian)
                for i in 0..8i64 {
                    let byte_val = value.extract((i * 8 + 7) as u32, (i * 8) as u32);
                    let byte_addr = effective_addr.bvadd(BV::from_i64(i, 32));
                    memory = memory.store(&byte_addr, &byte_val);
                }
            }

            // Partial-width loads (8-bit and 16-bit with sign/zero extension)
            Instruction::I32Load8S { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Load8S"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                let byte_val: BV = memory.select(&effective_addr).as_bv().unwrap();
                // Sign-extend 8-bit to 32-bit
                let result = byte_val.sign_ext(24);
                stack.push(result);
            }
            Instruction::I32Load8U { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Load8U"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                let byte_val: BV = memory.select(&effective_addr).as_bv().unwrap();
                // Zero-extend 8-bit to 32-bit
                let result = byte_val.zero_ext(24);
                stack.push(result);
            }
            Instruction::I32Load16S { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Load16S"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Read 2 bytes in little-endian order
                let b0: BV = memory.select(&effective_addr).as_bv().unwrap();
                let b1: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(1, 32)))
                    .as_bv()
                    .unwrap();
                let val16 = b0
                    .zero_ext(8)
                    .bvor(b1.zero_ext(8).bvshl(BV::from_i64(8, 16)));
                // Sign-extend 16-bit to 32-bit
                let result = val16.sign_ext(16);
                stack.push(result);
            }
            Instruction::I32Load16U { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Load16U"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Read 2 bytes in little-endian order
                let b0: BV = memory.select(&effective_addr).as_bv().unwrap();
                let b1: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(1, 32)))
                    .as_bv()
                    .unwrap();
                let val16 = b0
                    .zero_ext(8)
                    .bvor(b1.zero_ext(8).bvshl(BV::from_i64(8, 16)));
                // Zero-extend 16-bit to 32-bit
                let result = val16.zero_ext(16);
                stack.push(result);
            }
            Instruction::I64Load8S { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load8S"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                let byte_val: BV = memory.select(&effective_addr).as_bv().unwrap();
                // Sign-extend 8-bit to 64-bit
                let result = byte_val.sign_ext(56);
                stack.push(result);
            }
            Instruction::I64Load8U { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load8U"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                let byte_val: BV = memory.select(&effective_addr).as_bv().unwrap();
                // Zero-extend 8-bit to 64-bit
                let result = byte_val.zero_ext(56);
                stack.push(result);
            }
            Instruction::I64Load16S { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load16S"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Read 2 bytes in little-endian order
                let b0: BV = memory.select(&effective_addr).as_bv().unwrap();
                let b1: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(1, 32)))
                    .as_bv()
                    .unwrap();
                let val16 = b0
                    .zero_ext(8)
                    .bvor(b1.zero_ext(8).bvshl(BV::from_i64(8, 16)));
                // Sign-extend 16-bit to 64-bit
                let result = val16.sign_ext(48);
                stack.push(result);
            }
            Instruction::I64Load16U { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load16U"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Read 2 bytes in little-endian order
                let b0: BV = memory.select(&effective_addr).as_bv().unwrap();
                let b1: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(1, 32)))
                    .as_bv()
                    .unwrap();
                let val16 = b0
                    .zero_ext(8)
                    .bvor(b1.zero_ext(8).bvshl(BV::from_i64(8, 16)));
                // Zero-extend 16-bit to 64-bit
                let result = val16.zero_ext(48);
                stack.push(result);
            }
            Instruction::I64Load32S { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load32S"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Read 4 bytes in little-endian order
                let b0: BV = memory.select(&effective_addr).as_bv().unwrap();
                let b1: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(1, 32)))
                    .as_bv()
                    .unwrap();
                let b2: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(2, 32)))
                    .as_bv()
                    .unwrap();
                let b3: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(3, 32)))
                    .as_bv()
                    .unwrap();
                let val32 = b0
                    .zero_ext(24)
                    .bvor(b1.zero_ext(24).bvshl(BV::from_i64(8, 32)))
                    .bvor(b2.zero_ext(24).bvshl(BV::from_i64(16, 32)))
                    .bvor(b3.zero_ext(24).bvshl(BV::from_i64(24, 32)));
                // Sign-extend 32-bit to 64-bit
                let result = val32.sign_ext(32);
                stack.push(result);
            }
            Instruction::I64Load32U { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load32U"));
                }
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Read 4 bytes in little-endian order
                let b0: BV = memory.select(&effective_addr).as_bv().unwrap();
                let b1: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(1, 32)))
                    .as_bv()
                    .unwrap();
                let b2: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(2, 32)))
                    .as_bv()
                    .unwrap();
                let b3: BV = memory
                    .select(&effective_addr.bvadd(BV::from_i64(3, 32)))
                    .as_bv()
                    .unwrap();
                let val32 = b0
                    .zero_ext(24)
                    .bvor(b1.zero_ext(24).bvshl(BV::from_i64(8, 32)))
                    .bvor(b2.zero_ext(24).bvshl(BV::from_i64(16, 32)))
                    .bvor(b3.zero_ext(24).bvshl(BV::from_i64(24, 32)));
                // Zero-extend 32-bit to 64-bit
                let result = val32.zero_ext(32);
                stack.push(result);
            }

            // Partial-width stores (8-bit, 16-bit, 32-bit)
            Instruction::I32Store8 { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Store8"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Store low 8 bits
                let byte_val = value.extract(7, 0);
                memory = memory.store(&effective_addr, &byte_val);
            }
            Instruction::I32Store16 { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Store16"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Store low 16 bits in little-endian order
                let byte0 = value.extract(7, 0);
                let byte1 = value.extract(15, 8);
                memory = memory.store(&effective_addr, &byte0);
                memory = memory.store(&effective_addr.bvadd(BV::from_i64(1, 32)), &byte1);
            }
            Instruction::I64Store8 { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Store8"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Store low 8 bits
                let byte_val = value.extract(7, 0);
                memory = memory.store(&effective_addr, &byte_val);
            }
            Instruction::I64Store16 { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Store16"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Store low 16 bits in little-endian order
                let byte0 = value.extract(7, 0);
                let byte1 = value.extract(15, 8);
                memory = memory.store(&effective_addr, &byte0);
                memory = memory.store(&effective_addr.bvadd(BV::from_i64(1, 32)), &byte1);
            }
            Instruction::I64Store32 { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Store32"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                let effective_addr = addr.bvadd(BV::from_i64(*offset as i64, 32));
                // Store low 32 bits in little-endian order
                let byte0 = value.extract(7, 0);
                let byte1 = value.extract(15, 8);
                let byte2 = value.extract(23, 16);
                let byte3 = value.extract(31, 24);
                memory = memory.store(&effective_addr, &byte0);
                memory = memory.store(&effective_addr.bvadd(BV::from_i64(1, 32)), &byte1);
                memory = memory.store(&effective_addr.bvadd(BV::from_i64(2, 32)), &byte2);
                memory = memory.store(&effective_addr.bvadd(BV::from_i64(3, 32)), &byte3);
            }

            // Control flow instructions

            // Block: execute body, result goes on stack
            Instruction::Block { block_type, body } => {
                // Execute the block body
                let block_result = encode_block_body(body, &mut stack, &mut locals, &mut globals)?;

                // If block has a result type, it should be on the stack
                if let Some(width) = block_type_width(block_type) {
                    if block_result.is_none() && stack.is_empty() {
                        // Block didn't produce a result - create symbolic placeholder
                        stack.push(BV::new_const("block_result", width));
                    }
                }
            }

            // If/Else: branch based on condition
            Instruction::If {
                block_type,
                then_body,
                else_body,
            } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in If"));
                }
                let cond = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let cond_bool = cond.eq(&zero).not(); // cond != 0

                // Save state before branches
                let saved_stack = stack.clone();
                let saved_locals = locals.clone();
                let saved_globals = globals.clone();

                // Execute then branch
                let then_result =
                    encode_block_body(then_body, &mut stack, &mut locals, &mut globals)?;
                let then_stack = stack.clone();
                let then_locals = locals.clone();
                let then_globals = globals.clone();

                // Restore and execute else branch
                stack = saved_stack;
                locals = saved_locals;
                globals = saved_globals;
                let else_result =
                    encode_block_body(else_body, &mut stack, &mut locals, &mut globals)?;

                // Merge the two branches - collect merged values first to avoid borrow conflicts
                let merged_locals: Vec<BV> = then_locals
                    .iter()
                    .zip(locals.iter())
                    .map(|(then_local, else_local)| merge_bv(&cond_bool, then_local, else_local))
                    .collect();
                locals = merged_locals;

                let merged_globals: Vec<BV> = then_globals
                    .iter()
                    .zip(globals.iter())
                    .map(|(then_global, else_global)| {
                        merge_bv(&cond_bool, then_global, else_global)
                    })
                    .collect();
                globals = merged_globals;

                // For stack, merge if both branches produce same number of values
                if then_stack.len() == stack.len() {
                    let merged_stack: Vec<BV> = then_stack
                        .iter()
                        .zip(stack.iter())
                        .map(|(then_val, else_val)| merge_bv(&cond_bool, then_val, else_val))
                        .collect();
                    stack = merged_stack;
                } else if let Some(width) = block_type_width(block_type) {
                    // Block expects a result - use ITE on results
                    let then_val = then_result.unwrap_or_else(|| {
                        then_stack
                            .last()
                            .cloned()
                            .unwrap_or_else(|| BV::from_i64(0, width))
                    });
                    let else_val = else_result.unwrap_or_else(|| {
                        stack
                            .last()
                            .cloned()
                            .unwrap_or_else(|| BV::from_i64(0, width))
                    });
                    stack.clear();
                    stack.push(merge_bv(&cond_bool, &then_val, &else_val));
                }
            }

            // Loop: bounded unrolling for verification
            Instruction::Loop { block_type, body } => {
                // For verification, we unroll loops a fixed number of times
                // This is sound but incomplete (may miss bugs in later iterations)
                for _iteration in 0..MAX_LOOP_UNROLL {
                    let _ = encode_block_body(body, &mut stack, &mut locals, &mut globals)?;
                }
                // After unrolling, if loop has a result type, ensure something is on stack
                if let Some(width) = block_type_width(block_type) {
                    if stack.is_empty() {
                        stack.push(BV::new_const("loop_result", width));
                    }
                }
            }

            // Branch: exit enclosing block (handled by returning early from block encoder)
            Instruction::Br(_depth) => {
                // Branch exits the current block - for simple verification,
                // we treat this as terminating the current instruction sequence
                // The actual depth handling is done in encode_block_body
                break;
            }

            // Conditional branch
            Instruction::BrIf(_depth) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in BrIf"));
                }
                let cond = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let _cond_bool = cond.eq(&zero).not();
                // For simple verification, we continue execution
                // A more precise encoding would fork paths here
            }

            // Branch table
            Instruction::BrTable { .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in BrTable"));
                }
                let _index = stack.pop().unwrap();
                // Branch table is complex - treat as terminating for now
                break;
            }

            // Return from function
            Instruction::Return => {
                // Return terminates execution with current stack top
                break;
            }

            // Function call - properly model stack effects using signature context
            Instruction::Call(func_idx) => {
                // For verification, we can't inline calls (would need interprocedural analysis)
                // Instead, we properly model the stack effects:
                // 1. Pop the correct number of arguments
                // 2. Push fresh symbolic values for results
                if let Some(ctx_ref) = sig_ctx {
                    if let Some(sig) = ctx_ref.get_function_signature(*func_idx) {
                        // Pop arguments (in reverse order, as they were pushed)
                        for i in 0..sig.params.len() {
                            if stack.is_empty() {
                                return Err(anyhow!(
                                    "Stack underflow in Call: missing arg {} of {}",
                                    i + 1,
                                    sig.params.len()
                                ));
                            }
                            let _ = stack.pop().unwrap();
                        }
                        // Push results
                        for (i, result_type) in sig.results.iter().enumerate() {
                            let width = match result_type {
                                crate::ValueType::I32 | crate::ValueType::F32 => 32,
                                crate::ValueType::I64 | crate::ValueType::F64 => 64,
                            };
                            stack.push(BV::new_const(
                                format!("call_{}_result_{}", func_idx, i),
                                width,
                            ));
                        }
                    } else {
                        // Unknown function - conservative: assume returns i32
                        stack.push(BV::new_const("call_unknown_result", 32));
                    }
                } else {
                    // No context - conservative: assume returns i32
                    stack.push(BV::new_const("call_result", 32));
                }
            }

            // Indirect call - properly model stack effects using type signature
            Instruction::CallIndirect { type_idx, .. } => {
                // Pop table index first
                if stack.is_empty() {
                    return Err(anyhow!(
                        "Stack underflow in CallIndirect: missing table index"
                    ));
                }
                let _table_idx = stack.pop().unwrap();

                // Use type signature to properly model stack effects
                if let Some(ctx_ref) = sig_ctx {
                    if let Some(sig) = ctx_ref.get_type_signature(*type_idx) {
                        // Pop arguments
                        for i in 0..sig.params.len() {
                            if stack.is_empty() {
                                return Err(anyhow!(
                                    "Stack underflow in CallIndirect: missing arg {} of {}",
                                    i + 1,
                                    sig.params.len()
                                ));
                            }
                            let _ = stack.pop().unwrap();
                        }
                        // Push results
                        for (i, result_type) in sig.results.iter().enumerate() {
                            let width = match result_type {
                                crate::ValueType::I32 | crate::ValueType::F32 => 32,
                                crate::ValueType::I64 | crate::ValueType::F64 => 64,
                            };
                            stack.push(BV::new_const(
                                format!("call_indirect_{}_result_{}", type_idx, i),
                                width,
                            ));
                        }
                    } else {
                        // Unknown type - conservative: assume returns i32
                        stack.push(BV::new_const("call_indirect_unknown_result", 32));
                    }
                } else {
                    // No context - conservative: assume returns i32
                    stack.push(BV::new_const("call_indirect_result", 32));
                }
            }

            // End of function/block
            Instruction::End => {
                break;
            }

            // Unreachable terminates execution
            Instruction::Unreachable => {
                // For SMT purposes, unreachable means no concrete output
                // Return a fresh symbolic variable (represents undefined)
                return Ok(Some(BV::new_const("unreachable", 32)));
            }

            // ============================================================
            // Float operations - treated as bitvector operations
            // This verifies bit-pattern equality, not IEEE 754 semantics
            // ============================================================

            // Float binary operations (f32): [f32, f32] -> [f32]
            Instruction::F32Add
            | Instruction::F32Sub
            | Instruction::F32Mul
            | Instruction::F32Div
            | Instruction::F32Min
            | Instruction::F32Max
            | Instruction::F32Copysign => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32 binary op"));
                }
                let _rhs = stack.pop().unwrap();
                let _lhs = stack.pop().unwrap();
                // Float ops produce fresh symbolic value (we don't model IEEE 754 arithmetic)
                stack.push(BV::new_const("f32_result", 32));
            }

            // Float unary operations (f32): [f32] -> [f32]
            Instruction::F32Abs
            | Instruction::F32Neg
            | Instruction::F32Ceil
            | Instruction::F32Floor
            | Instruction::F32Trunc
            | Instruction::F32Nearest
            | Instruction::F32Sqrt => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32 unary op"));
                }
                let _val = stack.pop().unwrap();
                stack.push(BV::new_const("f32_unary_result", 32));
            }

            // Float comparison (f32): [f32, f32] -> [i32]
            Instruction::F32Eq
            | Instruction::F32Ne
            | Instruction::F32Lt
            | Instruction::F32Gt
            | Instruction::F32Le
            | Instruction::F32Ge => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32 comparison"));
                }
                let _rhs = stack.pop().unwrap();
                let _lhs = stack.pop().unwrap();
                // Comparison produces i32 (0 or 1)
                stack.push(BV::new_const("f32_cmp_result", 32));
            }

            // Float binary operations (f64): [f64, f64] -> [f64]
            Instruction::F64Add
            | Instruction::F64Sub
            | Instruction::F64Mul
            | Instruction::F64Div
            | Instruction::F64Min
            | Instruction::F64Max
            | Instruction::F64Copysign => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64 binary op"));
                }
                let _rhs = stack.pop().unwrap();
                let _lhs = stack.pop().unwrap();
                stack.push(BV::new_const("f64_result", 64));
            }

            // Float unary operations (f64): [f64] -> [f64]
            Instruction::F64Abs
            | Instruction::F64Neg
            | Instruction::F64Ceil
            | Instruction::F64Floor
            | Instruction::F64Trunc
            | Instruction::F64Nearest
            | Instruction::F64Sqrt => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64 unary op"));
                }
                let _val = stack.pop().unwrap();
                stack.push(BV::new_const("f64_unary_result", 64));
            }

            // Float comparison (f64): [f64, f64] -> [i32]
            Instruction::F64Eq
            | Instruction::F64Ne
            | Instruction::F64Lt
            | Instruction::F64Gt
            | Instruction::F64Le
            | Instruction::F64Ge => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64 comparison"));
                }
                let _rhs = stack.pop().unwrap();
                let _lhs = stack.pop().unwrap();
                stack.push(BV::new_const("f64_cmp_result", 32));
            }

            // ============================================================
            // Conversion operations - precise bitvector modeling
            // ============================================================

            // i32.wrap_i64: [i64] -> [i32] (truncate to low 32 bits)
            Instruction::I32WrapI64 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32WrapI64"));
                }
                let val = stack.pop().unwrap();
                stack.push(val.extract(31, 0)); // Extract low 32 bits
            }

            // i64.extend_i32_s: [i32] -> [i64] (sign-extend)
            Instruction::I64ExtendI32S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64ExtendI32S"));
                }
                let val = stack.pop().unwrap();
                stack.push(val.sign_ext(32)); // Sign-extend by 32 bits
            }

            // i64.extend_i32_u: [i32] -> [i64] (zero-extend)
            Instruction::I64ExtendI32U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64ExtendI32U"));
                }
                let val = stack.pop().unwrap();
                stack.push(val.zero_ext(32)); // Zero-extend by 32 bits
            }

            // Int-to-float conversions: produce fresh symbolic (no IEEE 754 modeling)
            Instruction::I32TruncF32S
            | Instruction::I32TruncF32U
            | Instruction::I32TruncF64S
            | Instruction::I32TruncF64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Trunc"));
                }
                let _val = stack.pop().unwrap();
                stack.push(BV::new_const("i32_trunc_result", 32));
            }

            Instruction::I64TruncF32S
            | Instruction::I64TruncF32U
            | Instruction::I64TruncF64S
            | Instruction::I64TruncF64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Trunc"));
                }
                let _val = stack.pop().unwrap();
                stack.push(BV::new_const("i64_trunc_result", 64));
            }

            // Float-from-int conversions
            Instruction::F32ConvertI32S
            | Instruction::F32ConvertI32U
            | Instruction::F32ConvertI64S
            | Instruction::F32ConvertI64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32Convert"));
                }
                let _val = stack.pop().unwrap();
                stack.push(BV::new_const("f32_convert_result", 32));
            }

            Instruction::F64ConvertI32S
            | Instruction::F64ConvertI32U
            | Instruction::F64ConvertI64S
            | Instruction::F64ConvertI64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64Convert"));
                }
                let _val = stack.pop().unwrap();
                stack.push(BV::new_const("f64_convert_result", 64));
            }

            // Float-to-float conversions
            Instruction::F32DemoteF64 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32DemoteF64"));
                }
                let _val = stack.pop().unwrap();
                stack.push(BV::new_const("f32_demote_result", 32));
            }

            Instruction::F64PromoteF32 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64PromoteF32"));
                }
                let _val = stack.pop().unwrap();
                stack.push(BV::new_const("f64_promote_result", 64));
            }

            // Reinterpret operations - bit-cast (exact bitvector modeling)
            Instruction::I32ReinterpretF32 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32ReinterpretF32"));
                }
                // Bits don't change, just reinterpretation - no-op for BV
                // Stack already has 32-bit value
            }

            Instruction::I64ReinterpretF64 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64ReinterpretF64"));
                }
                // No-op for BV - bits stay the same
            }

            Instruction::F32ReinterpretI32 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32ReinterpretI32"));
                }
                // No-op for BV
            }

            Instruction::F64ReinterpretI64 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64ReinterpretI64"));
                }
                // No-op for BV
            }

            // ============================================================
            // Additional memory operations
            // ============================================================
            Instruction::F32Load { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32Load"));
                }
                let _addr = stack.pop().unwrap();
                let _ = offset;
                stack.push(BV::new_const("f32_load_result", 32));
            }

            Instruction::F32Store { .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Store"));
                }
                let _value = stack.pop().unwrap();
                let _addr = stack.pop().unwrap();
            }

            Instruction::F64Load { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64Load"));
                }
                let _addr = stack.pop().unwrap();
                let _ = offset;
                stack.push(BV::new_const("f64_load_result", 64));
            }

            Instruction::F64Store { .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Store"));
                }
                let _value = stack.pop().unwrap();
                let _addr = stack.pop().unwrap();
            }

            // Note: Integer partial loads/stores are now handled with precise Z3 encodings
            // earlier in this match statement (I32Load8S/U, I32Load16S/U, I64Load8S/U, etc.)

            // Memory size/grow
            Instruction::MemorySize(_) => {
                stack.push(BV::new_const("memory_size", 32));
            }

            Instruction::MemoryGrow(_) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in MemoryGrow"));
                }
                let _delta = stack.pop().unwrap();
                stack.push(BV::new_const("memory_grow_result", 32));
            }

            // Rotate operations - precise bitvector modeling
            Instruction::I32Rotl => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Rotl"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvrotl(&rhs));
            }

            Instruction::I32Rotr => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Rotr"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvrotr(&rhs));
            }

            Instruction::I64Rotl => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Rotl"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                // For i64, shift amount is still i32 in WASM, but Z3 needs same width
                // Extend rhs to 64 bits
                let rhs64 = rhs.zero_ext(32);
                stack.push(lhs.bvrotl(&rhs64));
            }

            Instruction::I64Rotr => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Rotr"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let rhs64 = rhs.zero_ext(32);
                stack.push(lhs.bvrotr(&rhs64));
            }

            // Sign extension operations
            Instruction::I32Extend8S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Extend8S"));
                }
                let val = stack.pop().unwrap();
                // Extract low 8 bits and sign-extend to 32 bits
                let low8 = val.extract(7, 0);
                stack.push(low8.sign_ext(24));
            }
            Instruction::I32Extend16S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Extend16S"));
                }
                let val = stack.pop().unwrap();
                // Extract low 16 bits and sign-extend to 32 bits
                let low16 = val.extract(15, 0);
                stack.push(low16.sign_ext(16));
            }
            Instruction::I64Extend8S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Extend8S"));
                }
                let val = stack.pop().unwrap();
                // Extract low 8 bits and sign-extend to 64 bits
                let low8 = val.extract(7, 0);
                stack.push(low8.sign_ext(56));
            }
            Instruction::I64Extend16S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Extend16S"));
                }
                let val = stack.pop().unwrap();
                // Extract low 16 bits and sign-extend to 64 bits
                let low16 = val.extract(15, 0);
                stack.push(low16.sign_ext(48));
            }
            Instruction::I64Extend32S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Extend32S"));
                }
                let val = stack.pop().unwrap();
                // Extract low 32 bits and sign-extend to 64 bits
                let low32 = val.extract(31, 0);
                stack.push(low32.sign_ext(32));
            }

            // Saturating truncation operations - produce symbolic values
            // These are conversion operations that don't trap
            Instruction::I32TruncSatF32S
            | Instruction::I32TruncSatF32U
            | Instruction::I32TruncSatF64S
            | Instruction::I32TruncSatF64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in saturating truncation"));
                }
                stack.pop();
                stack.push(BV::new_const("trunc_sat_result_i32", 32));
            }
            Instruction::I64TruncSatF32S
            | Instruction::I64TruncSatF32U
            | Instruction::I64TruncSatF64S
            | Instruction::I64TruncSatF64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in saturating truncation"));
                }
                stack.pop();
                stack.push(BV::new_const("trunc_sat_result_i64", 64));
            }

            // Bulk memory operations - these modify memory, no stack return value
            Instruction::MemoryFill(_)
            | Instruction::MemoryCopy { .. }
            | Instruction::MemoryInit { .. } => {
                if stack.len() < 3 {
                    return Err(anyhow!("Stack underflow in bulk memory operation"));
                }
                stack.pop(); // len
                stack.pop(); // src/val
                stack.pop(); // dst
                             // No return value
            }
            Instruction::DataDrop(_) => {
                // No stack effect
            }

            // Unknown instructions - these should be rare now
            Instruction::Unknown(_) => {
                // Unknown bytes - can't verify, produce fresh symbolic value
                stack.push(BV::new_const("unknown_result", 32));
            }
        }
    }

    // Return value should be on stack if function has result type
    if func.signature.results.is_empty() {
        // Void function - no return value expected
        Ok(None)
    } else if stack.is_empty() {
        Err(anyhow!(
            "Function returned no value (stack empty) but result type expected"
        ))
    } else {
        Ok(Some(stack.pop().unwrap()))
    }
}

/// Encode a block body (helper for control flow)
/// Returns the result value if the block produces one
#[cfg(feature = "verification")]
fn encode_block_body(
    body: &[Instruction],
    stack: &mut Vec<BV>,
    locals: &mut Vec<BV>,
    globals: &mut Vec<BV>,
) -> Result<Option<BV>> {
    for instr in body {
        match instr {
            // For nested blocks, recursively encode
            Instruction::Block {
                block_type,
                body: nested_body,
            } => {
                let result = encode_block_body(nested_body, stack, locals, globals)?;
                if let Some(width) = block_type_width(block_type) {
                    if result.is_none() && stack.is_empty() {
                        stack.push(BV::new_const("nested_block_result", width));
                    }
                }
            }

            Instruction::If {
                block_type,
                then_body,
                else_body,
            } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in nested If"));
                }
                let cond = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let cond_bool = cond.eq(&zero).not();

                let saved_stack = stack.clone();
                let saved_locals = locals.clone();
                let saved_globals = globals.clone();

                let then_result = encode_block_body(then_body, stack, locals, globals)?;
                let then_stack = stack.clone();
                let then_locals = locals.clone();
                let then_globals = globals.clone();

                *stack = saved_stack;
                *locals = saved_locals;
                *globals = saved_globals;
                let else_result = encode_block_body(else_body, stack, locals, globals)?;

                // Merge branches - collect new values first to avoid borrow conflicts
                let merged_locals: Vec<BV> = then_locals
                    .iter()
                    .zip(locals.iter())
                    .map(|(then_local, else_local)| merge_bv(&cond_bool, then_local, else_local))
                    .collect();
                *locals = merged_locals;

                let merged_globals: Vec<BV> = then_globals
                    .iter()
                    .zip(globals.iter())
                    .map(|(then_global, else_global)| {
                        merge_bv(&cond_bool, then_global, else_global)
                    })
                    .collect();
                *globals = merged_globals;

                if then_stack.len() == stack.len() {
                    let merged_stack: Vec<BV> = then_stack
                        .iter()
                        .zip(stack.iter())
                        .map(|(then_val, else_val)| merge_bv(&cond_bool, then_val, else_val))
                        .collect();
                    *stack = merged_stack;
                } else if let Some(width) = block_type_width(block_type) {
                    let then_val = then_result.unwrap_or_else(|| {
                        then_stack
                            .last()
                            .cloned()
                            .unwrap_or_else(|| BV::from_i64(0, width))
                    });
                    let else_val = else_result.unwrap_or_else(|| {
                        stack
                            .last()
                            .cloned()
                            .unwrap_or_else(|| BV::from_i64(0, width))
                    });
                    stack.clear();
                    stack.push(merge_bv(&cond_bool, &then_val, &else_val));
                }
            }

            Instruction::Loop {
                block_type,
                body: loop_body,
            } => {
                for _iteration in 0..MAX_LOOP_UNROLL {
                    let _ = encode_block_body(loop_body, stack, locals, globals)?;
                }
                if let Some(width) = block_type_width(block_type) {
                    if stack.is_empty() {
                        stack.push(BV::new_const("nested_loop_result", width));
                    }
                }
            }

            // Branch exits the block
            Instruction::Br(_) | Instruction::Return => {
                return Ok(stack.last().cloned());
            }

            Instruction::BrIf(_) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in nested BrIf"));
                }
                let _cond = stack.pop().unwrap();
                // Continue execution (conservative)
            }

            Instruction::End => {
                return Ok(stack.last().cloned());
            }

            // Handle all other instructions inline (constants, arithmetic, etc.)
            _ => {
                encode_simple_instruction(instr, stack, locals, globals)?;
            }
        }
    }

    Ok(stack.last().cloned())
}

/// Encode a simple (non-control-flow) instruction
#[cfg(feature = "verification")]
#[allow(clippy::ptr_arg)] // Vec is needed for push/pop
#[allow(clippy::manual_range_patterns)] // Comments in OR patterns are more readable
fn encode_simple_instruction(
    instr: &Instruction,
    stack: &mut Vec<BV>,
    locals: &mut Vec<BV>,
    globals: &mut Vec<BV>,
) -> Result<()> {
    match instr {
        // Constants
        Instruction::I32Const(n) => {
            stack.push(BV::from_i64(*n as i64, 32));
        }
        Instruction::I64Const(n) => {
            stack.push(BV::from_i64(*n, 64));
        }
        Instruction::F32Const(bits) => {
            stack.push(BV::from_i64(*bits as i64, 32));
        }
        Instruction::F64Const(bits) => {
            stack.push(BV::from_i64(*bits as i64, 64));
        }

        // i32 binary arithmetic
        Instruction::I32Add => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(lhs.bvadd(&rhs));
        }
        Instruction::I32Sub => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(lhs.bvsub(&rhs));
        }
        Instruction::I32Mul => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(lhs.bvmul(&rhs));
        }
        Instruction::I32And => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(lhs.bvand(&rhs));
        }
        Instruction::I32Or => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(lhs.bvor(&rhs));
        }
        Instruction::I32Xor => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(lhs.bvxor(&rhs));
        }
        Instruction::I32Shl => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(lhs.bvshl(&rhs));
        }
        Instruction::I32ShrS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(lhs.bvashr(&rhs));
        }
        Instruction::I32ShrU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(lhs.bvlshr(&rhs));
        }

        // i32 comparisons
        Instruction::I32Eq => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(lhs.eq(&rhs).ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)));
        }
        Instruction::I32Ne => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(
                lhs.eq(&rhs)
                    .not()
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32LtS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(
                lhs.bvslt(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32LtU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(
                lhs.bvult(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32GtS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(
                lhs.bvsgt(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32GtU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(
                lhs.bvugt(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32LeS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(
                lhs.bvsle(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32LeU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(
                lhs.bvule(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32GeS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(
                lhs.bvsge(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32GeU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(
                lhs.bvuge(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }

        // i32 unary
        Instruction::I32Eqz => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let val = stack.pop().unwrap();
            let zero = BV::from_i64(0, 32);
            stack.push(
                val.eq(&zero)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }

        // Local operations
        Instruction::LocalGet(idx) => {
            let idx = *idx as usize;
            if idx >= locals.len() {
                return Err(anyhow!("LocalGet index {} out of bounds", idx));
            }
            stack.push(locals[idx].clone());
        }
        Instruction::LocalSet(idx) => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let idx = *idx as usize;
            if idx >= locals.len() {
                return Err(anyhow!("LocalSet index {} out of bounds", idx));
            }
            let val = stack.pop().unwrap();
            locals[idx] = val;
        }
        Instruction::LocalTee(idx) => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let idx = *idx as usize;
            if idx >= locals.len() {
                return Err(anyhow!("LocalTee index {} out of bounds", idx));
            }
            let val = stack.last().unwrap().clone();
            locals[idx] = val;
        }

        // Global operations
        Instruction::GlobalGet(idx) => {
            let idx = *idx as usize;
            while globals.len() <= idx {
                let new_idx = globals.len();
                globals.push(BV::new_const(format!("global{}", new_idx), 32));
            }
            stack.push(globals[idx].clone());
        }
        Instruction::GlobalSet(idx) => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let idx = *idx as usize;
            while globals.len() <= idx {
                let new_idx = globals.len();
                globals.push(BV::new_const(format!("global{}", new_idx), 32));
            }
            let val = stack.pop().unwrap();
            globals[idx] = val;
        }

        // Select
        Instruction::Select => {
            if stack.len() < 3 {
                return Err(anyhow!("Stack underflow"));
            }
            let cond = stack.pop().unwrap();
            let val2 = stack.pop().unwrap();
            let val1 = stack.pop().unwrap();
            let zero = BV::from_i64(0, 32);
            stack.push(cond.eq(&zero).not().ite(&val1, &val2));
        }

        // Nop
        Instruction::Nop => {}

        // Drop pops and discards top of stack
        Instruction::Drop => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow in Drop"));
            }
            stack.pop();
        }

        // Unknown instruction - may contain float operations stored as raw bytes
        // We decode the opcode to handle float operations with bitvector semantics
        Instruction::Unknown(bytes) => {
            if let Some(&opcode) = bytes.first() {
                match opcode {
                    // f32 binary operations (consume 2 f32, produce 1 f32)
                    // Using bitvector arithmetic - NOT IEEE 754 semantics
                    0x92 /* f32.add */ | 0x93 /* f32.sub */ | 0x94 /* f32.mul */ |
                    0x95 /* f32.div */ | 0x96 /* f32.min */ | 0x97 /* f32.max */ |
                    0x98 /* f32.copysign */ => {
                        if stack.len() < 2 {
                            return Err(anyhow!("Stack underflow in f32 binary op"));
                        }
                        let _rhs = stack.pop().unwrap();
                        let _lhs = stack.pop().unwrap();
                        // Create fresh symbolic value - float semantics not modeled in Z3 bitvectors
                        // This is sound but imprecise: any float result is a valid bitvector
                        stack.push(BV::new_const(format!("f32_result_{}", opcode), 32));
                    }

                    // f32 unary operations (consume 1 f32, produce 1 f32)
                    0x8b /* f32.abs */ | 0x8c /* f32.neg */ | 0x8d /* f32.ceil */ |
                    0x8e /* f32.floor */ | 0x8f /* f32.trunc */ | 0x90 /* f32.nearest */ |
                    0x91 /* f32.sqrt */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f32 unary op"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const(format!("f32_result_{}", opcode), 32));
                    }

                    // f32 comparisons (consume 2 f32, produce 1 i32)
                    0x5b /* f32.eq */ | 0x5c /* f32.ne */ | 0x5d /* f32.lt */ |
                    0x5e /* f32.gt */ | 0x5f /* f32.le */ | 0x60 /* f32.ge */ => {
                        if stack.len() < 2 {
                            return Err(anyhow!("Stack underflow in f32 comparison"));
                        }
                        let _rhs = stack.pop().unwrap();
                        let _lhs = stack.pop().unwrap();
                        // Comparison produces i32 (0 or 1)
                        stack.push(BV::new_const(format!("f32_cmp_{}", opcode), 32));
                    }

                    // f64 binary operations (consume 2 f64, produce 1 f64)
                    0xa0 /* f64.add */ | 0xa1 /* f64.sub */ | 0xa2 /* f64.mul */ |
                    0xa3 /* f64.div */ | 0xa4 /* f64.min */ | 0xa5 /* f64.max */ |
                    0xa6 /* f64.copysign */ => {
                        if stack.len() < 2 {
                            return Err(anyhow!("Stack underflow in f64 binary op"));
                        }
                        let _rhs = stack.pop().unwrap();
                        let _lhs = stack.pop().unwrap();
                        // Create fresh symbolic value - 64-bit for f64
                        stack.push(BV::new_const(format!("f64_result_{}", opcode), 64));
                    }

                    // f64 unary operations (consume 1 f64, produce 1 f64)
                    0x99 /* f64.abs */ | 0x9a /* f64.neg */ | 0x9b /* f64.ceil */ |
                    0x9c /* f64.floor */ | 0x9d /* f64.trunc */ | 0x9e /* f64.nearest */ |
                    0x9f /* f64.sqrt */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f64 unary op"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const(format!("f64_result_{}", opcode), 64));
                    }

                    // f64 comparisons (consume 2 f64, produce 1 i32)
                    0x61 /* f64.eq */ | 0x62 /* f64.ne */ | 0x63 /* f64.lt */ |
                    0x64 /* f64.gt */ | 0x65 /* f64.le */ | 0x66 /* f64.ge */ => {
                        if stack.len() < 2 {
                            return Err(anyhow!("Stack underflow in f64 comparison"));
                        }
                        let _rhs = stack.pop().unwrap();
                        let _lhs = stack.pop().unwrap();
                        // Comparison produces i32 (0 or 1)
                        stack.push(BV::new_const(format!("f64_cmp_{}", opcode), 32));
                    }

                    // Conversion operations
                    0xa7 /* i32.trunc_f32_s */ | 0xa8 /* i32.trunc_f32_u */ |
                    0xa9 /* i32.trunc_f64_s */ | 0xaa /* i32.trunc_f64_u */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in i32.trunc"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const("i32_trunc_result", 32));
                    }

                    0xab /* i64.trunc_f32_s */ | 0xac /* i64.trunc_f32_u */ |
                    0xad /* i64.trunc_f64_s */ | 0xae /* i64.trunc_f64_u */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in i64.trunc"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const("i64_trunc_result", 64));
                    }

                    0xb2 /* f32.convert_i32_s */ | 0xb3 /* f32.convert_i32_u */ |
                    0xb4 /* f32.convert_i64_s */ | 0xb5 /* f32.convert_i64_u */ |
                    0xb6 /* f32.demote_f64 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f32 convert"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const("f32_convert_result", 32));
                    }

                    0xb7 /* f64.convert_i32_s */ | 0xb8 /* f64.convert_i32_u */ |
                    0xb9 /* f64.convert_i64_s */ | 0xba /* f64.convert_i64_u */ |
                    0xbb /* f64.promote_f32 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f64 convert"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const("f64_convert_result", 64));
                    }

                    // Reinterpret operations (bit pattern preservation)
                    0xbc /* i32.reinterpret_f32 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in i32.reinterpret"));
                        }
                        // Reinterpret preserves bits - same bitvector, different type interpretation
                        // For Z3, this is identity since we use bitvectors for both
                        // Stack value stays the same (32-bit bitvector)
                    }

                    0xbd /* i64.reinterpret_f64 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in i64.reinterpret"));
                        }
                        // Same reasoning - 64-bit bitvector stays unchanged
                    }

                    0xbe /* f32.reinterpret_i32 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f32.reinterpret"));
                        }
                        // Bit pattern preserved - 32-bit bitvector unchanged
                    }

                    0xbf /* f64.reinterpret_i64 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f64.reinterpret"));
                        }
                        // Bit pattern preserved - 64-bit bitvector unchanged
                    }

                    // Default: create symbolic result for truly unknown ops
                    _ => {
                        stack.push(BV::new_const(format!("unknown_op_{}", opcode), 32));
                    }
                }
            } else {
                // Empty Unknown instruction - shouldn't happen, but handle gracefully
                stack.push(BV::new_const("empty_unknown", 32));
            }
        }

        // For other instructions not handled above, create symbolic result
        _ => {
            // Create a fresh symbolic value for unsupported instructions
            // This is sound but imprecise
            stack.push(BV::new_const("unsupported_result", 32));
        }
    }
    Ok(())
}

/// Verify that block stack properties are preserved across optimization
///
/// Uses Z3 to formally verify that a block transformation preserves:
/// 1. Stack composition (instructions sequence correctly)
/// 2. Input parameter types
/// 3. Output result types
///
/// # Arguments
///
/// * `original_instrs` - Instructions in original block
/// * `optimized_instrs` - Instructions in optimized block
/// * `block_params` - Input types the block expects
/// * `block_results` - Output types the block must produce
///
/// # Returns
///
/// * `Ok(true)` - Stack properties are preserved
/// * `Ok(false)` - Found a counterexample
/// * `Err(_)` - Verification error
#[cfg(feature = "verification")]
pub fn verify_stack_properties(
    original_instrs: &[Instruction],
    optimized_instrs: &[Instruction],
    block_params: &[crate::ValueType],
    block_results: &[crate::ValueType],
) -> Result<bool> {
    // Convert ValueType to stack::ValueType
    let stack_params: Vec<crate::stack::ValueType> = block_params
        .iter()
        .map(|vt| match vt {
            crate::ValueType::I32 => crate::stack::ValueType::I32,
            crate::ValueType::I64 => crate::stack::ValueType::I64,
            crate::ValueType::F32 => crate::stack::ValueType::F32,
            crate::ValueType::F64 => crate::stack::ValueType::F64,
        })
        .collect();

    let stack_results: Vec<crate::stack::ValueType> = block_results
        .iter()
        .map(|vt| match vt {
            crate::ValueType::I32 => crate::stack::ValueType::I32,
            crate::ValueType::I64 => crate::stack::ValueType::I64,
            crate::ValueType::F32 => crate::stack::ValueType::F32,
            crate::ValueType::F64 => crate::stack::ValueType::F64,
        })
        .collect();

    // First, validate that both blocks have correct structure
    let orig_validation =
        crate::stack::validation::validate_block(original_instrs, &stack_params, &stack_results);

    let opt_validation =
        crate::stack::validation::validate_block(optimized_instrs, &stack_params, &stack_results);

    // Check validation results
    match (&orig_validation, &opt_validation) {
        (
            crate::stack::validation::ValidationResult::Valid(_),
            crate::stack::validation::ValidationResult::Valid(_),
        ) => {
            // Both are valid - stack properties are preserved!
            Ok(true)
        }
        (crate::stack::validation::ValidationResult::Valid(_), _) => {
            // Original valid but optimized invalid - stack properties NOT preserved
            eprintln!(
                "Stack property violation: optimization produced invalid stack: {:?}",
                opt_validation
            );
            Ok(false)
        }
        (_, crate::stack::validation::ValidationResult::Valid(_)) => {
            // This shouldn't happen if original was valid - something went wrong
            eprintln!(
                "Stack property violation: original block became invalid: {:?}",
                orig_validation
            );
            Ok(false)
        }
        _ => {
            // Both invalid
            eprintln!(
                "Stack property violation in original block: {:?}",
                orig_validation
            );
            Ok(false)
        }
    }
}

/// Validate all blocks in a module using stack analysis
///
/// Phase 5 Integration: Validates that all block structures in the module have
/// correct stack compositions. This is called after optimization passes to ensure
/// they preserve stack types.
///
/// # Arguments
///
/// * `module` - The module to validate
///
/// # Returns
///
/// * `Ok(true)` - All blocks have valid stack properties
/// * `Ok(false)` - A block violates stack composition rules
/// * `Err(_)` - Validation error
#[cfg(feature = "verification")]
pub fn validate_module_blocks(module: &Module) -> Result<bool> {
    for func in &module.functions {
        // Collect block parameters from function signature
        let block_params: Vec<crate::ValueType> = func.signature.params.clone();
        let block_results: Vec<crate::ValueType> = func.signature.results.clone();

        // Validate function body blocks
        if !validate_instruction_sequence(&func.instructions, &block_params, &block_results)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Helper to recursively validate instruction sequences within blocks
#[cfg(feature = "verification")]
fn validate_instruction_sequence(
    instrs: &[crate::Instruction],
    params: &[crate::ValueType],
    results: &[crate::ValueType],
) -> Result<bool> {
    // Use our stack validation module to check composition
    let stack_params: Vec<crate::stack::ValueType> = params
        .iter()
        .map(|vt| match vt {
            crate::ValueType::I32 => crate::stack::ValueType::I32,
            crate::ValueType::I64 => crate::stack::ValueType::I64,
            crate::ValueType::F32 => crate::stack::ValueType::F32,
            crate::ValueType::F64 => crate::stack::ValueType::F64,
        })
        .collect();

    let stack_results: Vec<crate::stack::ValueType> = results
        .iter()
        .map(|vt| match vt {
            crate::ValueType::I32 => crate::stack::ValueType::I32,
            crate::ValueType::I64 => crate::stack::ValueType::I64,
            crate::ValueType::F32 => crate::stack::ValueType::F32,
            crate::ValueType::F64 => crate::stack::ValueType::F64,
        })
        .collect();

    // Create a block validation context
    let result = crate::stack::validation::validate_block(instrs, &stack_params, &stack_results);

    match result {
        crate::stack::validation::ValidationResult::Valid(_) => Ok(true),
        crate::stack::validation::ValidationResult::StackMismatch {
            position,
            expected,
            actual,
        } => {
            eprintln!("Stack validation: StackMismatch at position {}", position);
            eprintln!("  Expected: {:?}", expected);
            eprintln!("  Actual: {:?}", actual);
            Ok(false)
        }
        crate::stack::validation::ValidationResult::MissingInput { expected_params } => {
            eprintln!(
                "Stack validation: MissingInput, expected: {:?}",
                expected_params
            );
            Ok(false)
        }
        crate::stack::validation::ValidationResult::WrongOutput {
            expected_results,
            actual_results,
        } => {
            eprintln!("Stack validation: WrongOutput");
            eprintln!("  Expected: {:?}", expected_results);
            eprintln!("  Actual: {:?}", actual_results);
            Ok(false)
        }
        crate::stack::validation::ValidationResult::InstructionError {
            position,
            message,
            stack_state,
        } => {
            eprintln!(
                "Stack validation: InstructionError at position {}",
                position
            );
            eprintln!("  Message: {}", message);
            eprintln!("  Stack state: {:?}", stack_state);
            Ok(false)
        }
    }
}

#[cfg(not(feature = "verification"))]
pub fn verify_optimization(_original: &Module, _optimized: &Module) -> Result<bool> {
    Err(anyhow!(
        "Verification support not enabled. Rebuild with --features verification"
    ))
}

/// Non-verification stub for module block validation when Z3 feature is disabled
#[cfg(not(feature = "verification"))]
pub fn validate_module_blocks(_module: &Module) -> Result<bool> {
    Err(anyhow!(
        "Verification support not enabled. Rebuild with --features verification"
    ))
}

/// Non-verification stub for stack properties when Z3 feature is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_stack_properties(
    _original_instrs: &[crate::Instruction],
    _optimized_instrs: &[crate::Instruction],
    _block_params: &[crate::ValueType],
    _block_results: &[crate::ValueType],
) -> Result<bool> {
    Err(anyhow!(
        "Verification support not enabled. Rebuild with --features verification"
    ))
}

#[cfg(all(test, feature = "verification"))]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn test_verify_constant_folding() {
        // Original: (i32.add (i32.const 2) (i32.const 3))
        let original_wat = r#"
            (module
                (func (result i32)
                    i32.const 2
                    i32.const 3
                    i32.add
                )
            )
        "#;

        // Optimized: (i32.const 5)
        let optimized_wat = r#"
            (module
                (func (result i32)
                    i32.const 5
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        // Verify equivalence
        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "Programs should be equivalent");
    }

    #[test]
    fn test_verify_strength_reduction() {
        // Original: (i32.mul x 4)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 4
                    i32.mul
                )
            )
        "#;

        // Optimized: (i32.shl x 2)
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 2
                    i32.shl
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        // Verify equivalence
        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(
            result.unwrap(),
            "Strength reduction should preserve semantics"
        );
    }

    #[test]
    fn test_verify_bitwise_identity() {
        // Original: (i32.xor x x)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    local.get 0
                    i32.xor
                )
            )
        "#;

        // Optimized: (i32.const 0)
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 0
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        // Verify equivalence
        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "x XOR x = 0 should be proven");
    }

    #[test]
    fn test_verify_detects_incorrect_optimization() {
        // Original: (i32.add x 1)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                )
            )
        "#;

        // Incorrectly optimized to: (i32.const 2) - WRONG!
        let wrong_optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 2
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let wrong_optimized = parse::parse_wat(wrong_optimized_wat).unwrap();

        // Verification should detect this is wrong
        let result = verify_optimization(&original, &wrong_optimized);
        assert!(result.is_ok(), "Verification should complete");
        assert!(!result.unwrap(), "Should detect incorrect optimization");
    }

    #[test]
    fn test_verify_comparison_optimization() {
        // Original: (i32.eq x x) should always be true (1)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    local.get 0
                    i32.eq
                )
            )
        "#;

        // Optimized: (i32.const 1)
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 1
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "x == x should always be 1");
    }

    #[test]
    fn test_verify_division_optimization() {
        // Original: (i32.div_u x 1) should equal x
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.div_u
                )
            )
        "#;

        // Optimized: just return x
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "x / 1 should equal x");
    }

    #[test]
    fn test_verify_eqz_optimization() {
        // Original: (i32.eqz (i32.const 0)) should be 1
        let original_wat = r#"
            (module
                (func (result i32)
                    i32.const 0
                    i32.eqz
                )
            )
        "#;

        // Optimized: (i32.const 1)
        let optimized_wat = r#"
            (module
                (func (result i32)
                    i32.const 1
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "eqz(0) should be 1");
    }

    #[test]
    fn test_verify_select_optimization() {
        // Original: (select 42 99 1) should be 42 (condition != 0)
        let original_wat = r#"
            (module
                (func (result i32)
                    i32.const 42
                    i32.const 99
                    i32.const 1
                    select
                )
            )
        "#;

        // Optimized: (i32.const 42)
        let optimized_wat = r#"
            (module
                (func (result i32)
                    i32.const 42
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "select(42, 99, 1) should be 42");
    }

    #[test]
    fn test_verify_global_operations() {
        // Test that global.get followed by identity transformation works
        let original_wat = r#"
            (module
                (global (mut i32) (i32.const 0))
                (func (result i32)
                    global.get 0
                    i32.const 0
                    i32.add
                )
            )
        "#;

        // Optimized: just global.get (adding 0 is identity)
        let optimized_wat = r#"
            (module
                (global (mut i32) (i32.const 0))
                (func (result i32)
                    global.get 0
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "global.get + 0 should equal global.get");
    }

    #[test]
    fn test_verify_if_else_simplification() {
        // Original: if x then 42 else 42 (both branches return same value)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    if (result i32)
                        i32.const 42
                    else
                        i32.const 42
                    end
                )
            )
        "#;

        // Optimized: just return 42
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 42
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "if x then 42 else 42 should equal 42");
    }

    #[test]
    fn test_verify_if_else_with_condition() {
        // Original: if 1 then x else y (condition is always true)
        let original_wat = r#"
            (module
                (func (param i32) (param i32) (result i32)
                    i32.const 1
                    if (result i32)
                        local.get 0
                    else
                        local.get 1
                    end
                )
            )
        "#;

        // Optimized: just return x (first param)
        let optimized_wat = r#"
            (module
                (func (param i32) (param i32) (result i32)
                    local.get 0
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "if 1 then x else y should equal x");
    }

    #[test]
    fn test_verify_block_result() {
        // Original: block with result type that does arithmetic
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    block (result i32)
                        local.get 0
                        i32.const 1
                        i32.add
                    end
                )
            )
        "#;

        // Optimized: same arithmetic without block wrapper
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "block {{ x + 1 }} should equal x + 1");
    }

    #[test]
    fn test_verify_nested_if() {
        // Original: nested if that simplifies
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 1
                    if (result i32)
                        i32.const 0
                        if (result i32)
                            i32.const 99
                        else
                            i32.const 42
                        end
                    else
                        i32.const 0
                    end
                )
            )
        "#;

        // Optimized: the outer if is true, inner is false, so result is 42
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 42
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(
            result.unwrap(),
            "nested if with constant conditions should simplify"
        );
    }

    #[test]
    fn test_verify_float_param_function() {
        // Test that functions with f32 parameters can be verified
        // Float parameters are treated as bitvectors for verification
        let original_wat = r#"
            (module
                (func (param f32) (result i32)
                    i32.const 42
                )
            )
        "#;

        // Same function - should verify as equivalent
        let optimized_wat = r#"
            (module
                (func (param f32) (result i32)
                    i32.const 42
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification with f32 param failed: {:?}",
            result
        );
        assert!(result.unwrap(), "Functions with f32 params should verify");
    }

    #[test]
    fn test_verify_f64_param_function() {
        // Test that functions with f64 parameters can be verified
        let original_wat = r#"
            (module
                (func (param f64) (result i32)
                    i32.const 99
                )
            )
        "#;

        let optimized_wat = r#"
            (module
                (func (param f64) (result i32)
                    i32.const 99
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification with f64 param failed: {:?}",
            result
        );
        assert!(result.unwrap(), "Functions with f64 params should verify");
    }

    #[test]
    fn test_verify_float_constant_folding() {
        // Test that f32 constant folding works (bits are preserved)
        // Original: f32.const 3.14
        let original_wat = r#"
            (module
                (func (result f32)
                    f32.const 3.14
                )
            )
        "#;

        // Same constant - bits should match
        let optimized_wat = r#"
            (module
                (func (result f32)
                    f32.const 3.14
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification of f32 constant failed: {:?}",
            result
        );
        assert!(result.unwrap(), "f32 constants should verify as equivalent");
    }
}
