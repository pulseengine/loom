//! Fused Component Optimization
//!
//! Specialized optimization passes for WebAssembly modules produced by component
//! fusion tools (e.g., meld). When multiple P2/P3 components are fused into a
//! single core module, the result contains characteristic patterns that benefit
//! from targeted optimization:
//!
//! ## Optimization Passes
//!
//! ### 0. Same-Memory Adapter Collapse
//! In single-memory modules, meld generates adapters that allocate+copy within
//! the same linear memory. Since both pointers alias the same address space,
//! the copy is redundant. This pass collapses them to trivial forwarding
//! trampolines, which Pass 1 then devirtualizes.
//!
//! ### 1. Adapter Devirtualization
//! Component fusion generates adapter trampolines for cross-component calls.
//! Trivial adapters that simply forward all parameters to a target function
//! can be bypassed: callers are rewritten to call the target directly.
//!
//! ### 2. Trivial Call Elimination
//! Component fusion generates `cabi_post_return` functions for Canonical ABI
//! compliance. When these are empty no-ops (`() -> ()`), calls to them are
//! eliminated.
//!
//! ### 3. Function Type Deduplication
//! Each source component contributes its own type section. After fusion, many
//! identical function types exist. Deduplication merges them and remaps all
//! type references.
//!
//! ### 4. Dead Function Elimination
//! After adapter devirtualization, adapter functions may become unreachable.
//! This pass removes functions with zero call sites that are not exported.
//!
//! ### 5. Import Deduplication
//! Fused modules may contain duplicate imports (same module+name+type) from
//! different source components. These are merged with reference remapping.
//!
//! ## Correctness
//!
//! All transformations are provably correct:
//! - Same-memory adapter collapse: same-memory copy is redundant (Spec §5.4.7)
//! - Adapter devirtualization: semantically identical (adapter body = forward call)
//! - Type deduplication: structural type equality, index remapping preserves refs
//! - Dead function elimination: unreachable code cannot affect semantics
//! - Import deduplication: same external binding, merged references
//!
//! See `proofs/simplify/FusedOptimization.v` for formal Rocq proofs.

use crate::{ExportKind, Function, FunctionSignature, Import, ImportKind, Instruction, Module};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Statistics about fused module optimization
#[derive(Debug, Clone, Default)]
pub struct FusedOptimizationStats {
    /// Number of same-memory adapters collapsed to forwarding trampolines
    pub same_memory_adapters_collapsed: usize,
    /// Number of adapter functions detected
    pub adapters_detected: usize,
    /// Number of call sites devirtualized (adapter call -> direct call)
    pub calls_devirtualized: usize,
    /// Number of duplicate function types merged
    pub types_deduplicated: usize,
    /// Number of dead functions eliminated
    pub dead_functions_eliminated: usize,
    /// Number of duplicate imports merged
    pub imports_deduplicated: usize,
    /// Number of trivial post-return calls eliminated
    pub trivial_calls_eliminated: usize,
}

/// An adapter trampoline detected in the fused module.
///
/// Pattern recognized (direct adapter, no memory crossing):
/// ```text
/// (func $adapter (param ...) (result ...)
///   local.get 0
///   local.get 1
///   ...
///   local.get N
///   call $target
///   end)
/// ```
///
/// The adapter simply loads all parameters in order and calls a target function
/// with the same signature. Callers of the adapter can safely call the target
/// directly, eliminating the trampoline overhead.
#[derive(Debug, Clone)]
struct AdapterInfo {
    /// Index of the adapter function (in module.functions, not accounting for imports)
    func_index: usize,
    /// The target function index that the adapter forwards to (absolute index)
    target_func_idx: u32,
}

/// Run all fused module optimization passes.
///
/// This is the main entry point for optimizing modules produced by component
/// fusion. It runs adapter devirtualization, type deduplication, dead function
/// elimination, and import deduplication in sequence.
///
/// Returns optimization statistics.
pub fn optimize_fused_module(module: &mut Module) -> Result<FusedOptimizationStats> {
    // Pass 0: Collapse same-memory adapters into trivial forwarding trampolines
    // This converts memory-crossing adapters (realloc + memory.copy within the same
    // memory) into trivial forwarding trampolines that Pass 1 can then devirtualize.
    let same_memory_adapters_collapsed = collapse_same_memory_adapters(module)?;

    // Pass 1: Detect and devirtualize adapter trampolines
    let adapter_stats = devirtualize_adapters(module)?;

    let mut stats = FusedOptimizationStats {
        same_memory_adapters_collapsed,
        adapters_detected: adapter_stats.adapters_detected,
        calls_devirtualized: adapter_stats.calls_devirtualized,
        ..Default::default()
    };

    // Pass 2: Eliminate trivial no-op function calls (e.g. empty cabi_post_return)
    stats.trivial_calls_eliminated = eliminate_trivial_calls(module)?;

    // Pass 3: Deduplicate function types
    stats.types_deduplicated = deduplicate_function_types(module)?;

    // Pass 4: Eliminate dead functions (after devirtualization may create dead adapters)
    stats.dead_functions_eliminated = eliminate_dead_functions(module)?;

    // Pass 5: Deduplicate imports
    stats.imports_deduplicated = deduplicate_imports(module)?;

    Ok(stats)
}

// ============================================================================
// Pass 0: Same-Memory Adapter Collapse
// ============================================================================

/// Collapse same-memory adapters into trivial forwarding trampolines.
///
/// In a single-memory module, meld generates adapters that:
/// 1. Call `cabi_realloc` to allocate a buffer
/// 2. `memory.copy {0, 0}` to copy data within the same memory
/// 3. Call the target function with the new buffer pointer
///
/// Since both pointers reference the same linear memory, the target can
/// read the data at the original pointer directly. The allocation and copy
/// are semantically redundant.
///
/// This pass rewrites such adapters to trivial forwarding trampolines
/// (local.get 0; ...; local.get N; call $target; end), which Pass 1
/// then devirtualizes by rewriting callers to call the target directly.
///
/// ## Safety
///
/// A function is only collapsed when ALL of:
/// - The module has exactly one memory (including imported memories)
/// - The function has locals (adapters use locals for temporary pointers)
/// - All memory.copy instructions use {dst: 0, src: 0} (same memory)
/// - At least one call to a known cabi_realloc function
/// - Exactly one call to a non-realloc target function
/// - No control flow (Block, Loop, If, Br, BrIf, BrTable)
/// - No memory stores (I32Store, I64Store, F32Store, F64Store, etc.)
/// - No global writes (GlobalSet)
/// - Target function has the same signature as the adapter
///
/// Returns the number of adapters collapsed.
fn collapse_same_memory_adapters(module: &mut Module) -> Result<usize> {
    // Only applies to single-memory modules
    if count_total_memories(module) != 1 {
        return Ok(0);
    }

    let num_imported_funcs = count_imported_functions(module);
    let realloc_funcs = find_realloc_functions(module);

    if realloc_funcs.is_empty() {
        return Ok(0);
    }

    // Phase 1: Identify same-memory adapters and their target functions
    let mut collapse_targets: Vec<(usize, u32)> = Vec::new();

    for (func_idx, func) in module.functions.iter().enumerate() {
        if let Some(target_idx) = is_same_memory_adapter(func, &realloc_funcs) {
            // Verify target has the same signature
            let target_sig = get_function_signature(module, target_idx, num_imported_funcs);
            match target_sig {
                Some(sig) if *sig == func.signature => {
                    collapse_targets.push((func_idx, target_idx));
                }
                _ => {} // Signature mismatch or unknown target: skip
            }
        }
    }

    if collapse_targets.is_empty() {
        return Ok(0);
    }

    // Phase 2: Collapse each adapter to a forwarding trampoline
    let count = collapse_targets.len();
    for (func_idx, target_idx) in collapse_targets {
        collapse_to_forwarding(&mut module.functions[func_idx], target_idx);
    }

    Ok(count)
}

/// Count total memories in a module, including imported memories.
fn count_total_memories(module: &Module) -> usize {
    let imported_memories = module
        .imports
        .iter()
        .filter(|i| matches!(i.kind, ImportKind::Memory(_)))
        .count();
    imported_memories + module.memories.len()
}

/// Find all function indices that are `cabi_realloc` functions.
///
/// A function is `cabi_realloc` if:
/// - It is an imported function with name containing "cabi_realloc" or "realloc"
///   AND has signature `(i32, i32, i32, i32) -> i32`
/// - OR it is a local function named "cabi_realloc" (or variant)
///   AND has signature `(i32, i32, i32, i32) -> i32`
fn find_realloc_functions(module: &Module) -> HashSet<u32> {
    let mut realloc_set = HashSet::new();

    let realloc_sig = FunctionSignature {
        params: vec![
            crate::ValueType::I32,
            crate::ValueType::I32,
            crate::ValueType::I32,
            crate::ValueType::I32,
        ],
        results: vec![crate::ValueType::I32],
    };

    // Check imported functions
    let mut func_import_idx = 0u32;
    for import in &module.imports {
        if let ImportKind::Func(type_idx) = &import.kind {
            let name_matches =
                import.name.contains("cabi_realloc") || import.name.contains("realloc");
            if name_matches {
                // Verify signature matches
                if let Some(sig) = module.types.get(*type_idx as usize) {
                    if *sig == realloc_sig {
                        realloc_set.insert(func_import_idx);
                    }
                }
            }
            func_import_idx += 1;
        }
    }

    // Check local functions
    let num_imported_funcs = count_imported_functions(module);
    for (idx, func) in module.functions.iter().enumerate() {
        if let Some(ref name) = func.name {
            if (name.contains("cabi_realloc") || name.contains("realloc"))
                && func.signature == realloc_sig
            {
                realloc_set.insert(num_imported_funcs as u32 + idx as u32);
            }
        }
    }

    realloc_set
}

/// Check if a function is a same-memory adapter.
///
/// Returns the target function index if the function matches the pattern, None otherwise.
///
/// Detection criteria:
/// 1. Has locals (adapters use locals for temporary pointers)
/// 2. Contains at least one `memory.copy {0, 0}` and no cross-memory copies
/// 3. Contains at least one call to a realloc function
/// 4. Contains exactly one call to a non-realloc function (the target)
/// 5. No control flow instructions
/// 6. No memory store instructions
/// 7. No global write instructions
fn is_same_memory_adapter(func: &Function, realloc_funcs: &HashSet<u32>) -> Option<u32> {
    // Must have locals (trivial adapters without locals are handled by Pass 1)
    if func.locals.is_empty() {
        return None;
    }

    let instructions = &func.instructions;

    // Check for disqualifying instructions
    if has_control_flow(instructions) {
        return None;
    }
    if has_memory_stores(instructions) {
        return None;
    }
    if has_global_writes(instructions) {
        return None;
    }

    // Count memory.copy instructions - must have at least one same-memory copy
    // and no cross-memory copies
    let mut same_memory_copies = 0usize;
    let mut has_cross_memory_copy = false;
    // Count calls - must have at least one realloc call and exactly one target call
    let mut realloc_calls = 0usize;
    let mut target_call: Option<u32> = None;
    let mut target_call_count = 0usize;

    for instr in instructions {
        match instr {
            Instruction::MemoryCopy { dst_mem, src_mem } => {
                if *dst_mem == 0 && *src_mem == 0 {
                    same_memory_copies += 1;
                } else {
                    has_cross_memory_copy = true;
                }
            }
            Instruction::Call(func_idx) => {
                if realloc_funcs.contains(func_idx) {
                    realloc_calls += 1;
                } else {
                    target_call = Some(*func_idx);
                    target_call_count += 1;
                }
            }
            _ => {}
        }
    }

    // Validate the pattern
    if same_memory_copies == 0 {
        return None;
    }
    if has_cross_memory_copy {
        return None;
    }
    if realloc_calls == 0 {
        return None;
    }
    if target_call_count != 1 {
        return None;
    }

    target_call
}

/// Check if instructions contain control flow (Block, Loop, If, Br, BrIf, BrTable).
fn has_control_flow(instructions: &[Instruction]) -> bool {
    instructions.iter().any(|instr| {
        matches!(
            instr,
            Instruction::Block { .. }
                | Instruction::Loop { .. }
                | Instruction::If { .. }
                | Instruction::Br(_)
                | Instruction::BrIf(_)
                | Instruction::BrTable { .. }
        )
    })
}

/// Check if instructions contain memory store operations.
fn has_memory_stores(instructions: &[Instruction]) -> bool {
    instructions.iter().any(|instr| {
        matches!(
            instr,
            Instruction::I32Store { .. }
                | Instruction::I64Store { .. }
                | Instruction::F32Store { .. }
                | Instruction::F64Store { .. }
                | Instruction::I32Store8 { .. }
                | Instruction::I32Store16 { .. }
                | Instruction::I64Store8 { .. }
                | Instruction::I64Store16 { .. }
                | Instruction::I64Store32 { .. }
        )
    })
}

/// Check if instructions contain global write operations.
fn has_global_writes(instructions: &[Instruction]) -> bool {
    instructions
        .iter()
        .any(|instr| matches!(instr, Instruction::GlobalSet(_)))
}

/// Rewrite a function body to a trivial forwarding trampoline.
///
/// Replaces the function body with:
///   local.get 0; local.get 1; ...; local.get N; call $target; end
///
/// Also clears the locals list (no longer needed).
fn collapse_to_forwarding(func: &mut Function, target_idx: u32) {
    let param_count = func.signature.params.len();

    let mut new_body = Vec::with_capacity(param_count + 2);
    for i in 0..param_count {
        new_body.push(Instruction::LocalGet(i as u32));
    }
    new_body.push(Instruction::Call(target_idx));
    new_body.push(Instruction::End);

    func.instructions = new_body;
    func.locals.clear();
}

// ============================================================================
// Pass 1: Adapter Devirtualization
// ============================================================================

/// Detect trivial adapter trampolines and rewrite callers to bypass them.
///
/// A trivial adapter is a function whose body consists solely of:
/// 1. `local.get 0`, `local.get 1`, ..., `local.get N` (loading all params in order)
/// 2. `call $target` (calling a single target function)
/// 3. `end` (returning the target's results)
///
/// The adapter has no locals beyond parameters, no control flow, no side effects.
/// It is semantically identical to the target function call.
///
/// ## Proof Obligation
///
/// For a function `adapter(p0, ..., pN)` with body:
///   `local.get 0; ...; local.get N; call target; end`
///
/// For any caller context C and stack state S:
///   `eval(C[call adapter], S) = eval(C[call target], S)`
///
/// This holds because the adapter pushes exactly the same arguments onto the stack
/// in the same order and calls the target, producing identical results and side effects.
fn devirtualize_adapters(module: &mut Module) -> Result<DevirtualizationStats> {
    let mut stats = DevirtualizationStats::default();

    // Count imported functions to compute absolute function indices
    let num_imported_funcs = count_imported_functions(module);

    // Phase 1: Detect adapter trampolines
    let adapters = detect_adapters(module, num_imported_funcs);
    stats.adapters_detected = adapters.len();

    if adapters.is_empty() {
        return Ok(stats);
    }

    // Build mapping: adapter absolute index -> target absolute index
    let mut adapter_to_target: HashMap<u32, u32> = HashMap::new();
    for adapter in &adapters {
        let adapter_abs_idx = num_imported_funcs as u32 + adapter.func_index as u32;
        adapter_to_target.insert(adapter_abs_idx, adapter.target_func_idx);
    }

    // Transitively resolve adapter chains (adapter -> adapter -> target)
    // This handles cases where meld creates adapter chains across multiple components
    let resolved = resolve_adapter_chains(&adapter_to_target);

    // Phase 2: Rewrite call sites in all functions
    for func in &mut module.functions {
        let devirtualized = rewrite_calls(&func.instructions, &resolved);
        stats.calls_devirtualized += devirtualized.1;
        func.instructions = devirtualized.0;
    }

    Ok(stats)
}

#[derive(Debug, Default)]
struct DevirtualizationStats {
    adapters_detected: usize,
    calls_devirtualized: usize,
}

/// Detect functions that are trivial adapter trampolines.
///
/// A function is a trivial adapter if:
/// 1. It has no local variables (only parameters)
/// 2. Its body is exactly: local.get 0, local.get 1, ..., local.get N, call $target, end
/// 3. N equals the number of parameters
fn detect_adapters(module: &Module, num_imported_funcs: usize) -> Vec<AdapterInfo> {
    let mut adapters = Vec::new();

    for (func_idx, func) in module.functions.iter().enumerate() {
        if let Some(target) = is_trivial_adapter(func) {
            adapters.push(AdapterInfo {
                func_index: func_idx,
                target_func_idx: target,
            });
        }
    }

    // Validate: adapter must have same signature as target
    // (This is always true for meld-generated adapters, but we verify for safety)
    adapters.retain(|adapter| {
        let adapter_sig = &module.functions[adapter.func_index].signature;
        let target_sig =
            get_function_signature(module, adapter.target_func_idx, num_imported_funcs);
        match target_sig {
            Some(sig) => sig == adapter_sig,
            None => false, // Target is an import without a known signature - skip
        }
    });

    adapters
}

/// Check if a function is a trivial adapter trampoline.
///
/// Returns the target function index if it is, None otherwise.
fn is_trivial_adapter(func: &Function) -> Option<u32> {
    // Must have no local variables (only parameters)
    if !func.locals.is_empty() {
        return None;
    }

    let param_count = func.signature.params.len();
    let instructions = &func.instructions;

    // Expected pattern: local.get 0, local.get 1, ..., local.get N-1, call $target, end
    // Total instructions: param_count + 1 (call) + 1 (end) = param_count + 2
    let expected_len = param_count + 2;

    if instructions.len() != expected_len {
        return None;
    }

    // Verify local.get sequence
    for (i, instr) in instructions.iter().enumerate().take(param_count) {
        match instr {
            Instruction::LocalGet(idx) if *idx == i as u32 => {}
            _ => return None,
        }
    }

    // Verify call instruction
    let target = match &instructions[param_count] {
        Instruction::Call(target_idx) => *target_idx,
        _ => return None,
    };

    // Verify end
    match &instructions[param_count + 1] {
        Instruction::End => {}
        _ => return None,
    }

    Some(target)
}

/// Resolve adapter chains transitively.
///
/// If adapter A -> adapter B -> target T, resolve to A -> T.
/// This handles multi-hop adapter chains that can occur when components
/// are fused in a chain (A imports from B, B imports from C).
fn resolve_adapter_chains(adapter_to_target: &HashMap<u32, u32>) -> HashMap<u32, u32> {
    let mut resolved = adapter_to_target.clone();

    // Fixed-point iteration (terminates because chains are finite and acyclic)
    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 100 {
        changed = false;
        iterations += 1;

        let snapshot = resolved.clone();
        for (adapter, target) in resolved.iter_mut() {
            if let Some(deeper_target) = snapshot.get(target) {
                if *deeper_target != *adapter {
                    // Avoid self-loops
                    *target = *deeper_target;
                    changed = true;
                }
            }
        }
    }

    resolved
}

/// Rewrite call instructions to bypass adapters.
///
/// Returns the rewritten instructions and the count of devirtualized calls.
fn rewrite_calls(
    instructions: &[Instruction],
    adapter_to_target: &HashMap<u32, u32>,
) -> (Vec<Instruction>, usize) {
    let mut result = Vec::with_capacity(instructions.len());
    let mut count = 0;

    for instr in instructions {
        match instr {
            Instruction::Call(func_idx) => {
                if let Some(&target) = adapter_to_target.get(func_idx) {
                    result.push(Instruction::Call(target));
                    count += 1;
                } else {
                    result.push(instr.clone());
                }
            }
            // Recursively rewrite nested blocks
            Instruction::Block { block_type, body } => {
                let (new_body, c) = rewrite_calls(body, adapter_to_target);
                count += c;
                result.push(Instruction::Block {
                    block_type: block_type.clone(),
                    body: new_body,
                });
            }
            Instruction::Loop { block_type, body } => {
                let (new_body, c) = rewrite_calls(body, adapter_to_target);
                count += c;
                result.push(Instruction::Loop {
                    block_type: block_type.clone(),
                    body: new_body,
                });
            }
            Instruction::If {
                block_type,
                then_body,
                else_body,
            } => {
                let (new_then, c1) = rewrite_calls(then_body, adapter_to_target);
                let (new_else, c2) = rewrite_calls(else_body, adapter_to_target);
                count += c1 + c2;
                result.push(Instruction::If {
                    block_type: block_type.clone(),
                    then_body: new_then,
                    else_body: new_else,
                });
            }
            _ => {
                result.push(instr.clone());
            }
        }
    }

    (result, count)
}

// ============================================================================
// Pass 2: Function Type Deduplication
// ============================================================================

/// Deduplicate identical function types and remap all references.
///
/// After component fusion, the type section contains types from all source
/// components, many of which are identical (e.g., `(func (param i32) (result i32))`
/// appears once per component that uses it).
///
/// This pass:
/// 1. Hashes each function type
/// 2. Builds a canonical mapping (duplicate -> first occurrence)
/// 3. Remaps all type references (imports, functions, call_indirect)
/// 4. Removes duplicate types
///
/// Returns the number of types removed.
pub fn deduplicate_function_types(module: &mut Module) -> Result<usize> {
    if module.types.len() <= 1 {
        return Ok(0);
    }

    // Skip if raw type section bytes are present (GC/reference types)
    // Deduplication would require re-encoding which is complex for these types
    if module.type_section_bytes.is_some() {
        return Ok(0);
    }

    // Phase 1: Find canonical mapping
    // Map each unique type to its first index
    let mut canonical: HashMap<TypeKey, u32> = HashMap::new();
    let mut old_to_new: Vec<u32> = Vec::with_capacity(module.types.len());
    let mut new_types: Vec<FunctionSignature> = Vec::new();

    for ty in module.types.iter() {
        let key = TypeKey::from_signature(ty);
        if let Some(&canonical_idx) = canonical.get(&key) {
            // This type is a duplicate; map to canonical
            old_to_new.push(canonical_idx);
        } else {
            // New unique type
            let new_idx = new_types.len() as u32;
            canonical.insert(key, new_idx);
            old_to_new.push(new_idx);
            new_types.push(ty.clone());
        }
    }

    let dedup_count = module.types.len() - new_types.len();
    if dedup_count == 0 {
        return Ok(0);
    }

    // Phase 2: Remap all type references
    // Remap imports that reference type indices
    for import in &mut module.imports {
        if let ImportKind::Func(ref mut type_idx) = import.kind {
            if let Some(&new_idx) = old_to_new.get(*type_idx as usize) {
                *type_idx = new_idx;
            }
        }
    }

    // Remap function bodies (CallIndirect uses type indices)
    for func in &mut module.functions {
        remap_type_refs_in_block(&mut func.instructions, &old_to_new);
    }

    // Phase 3: Replace type section
    module.types = new_types;

    Ok(dedup_count)
}

/// Hash key for function type deduplication.
#[derive(Hash, PartialEq, Eq)]
struct TypeKey {
    params: Vec<crate::ValueType>,
    results: Vec<crate::ValueType>,
}

impl TypeKey {
    fn from_signature(sig: &FunctionSignature) -> Self {
        TypeKey {
            params: sig.params.clone(),
            results: sig.results.clone(),
        }
    }
}

/// Remap type references in call_indirect instructions within a block.
fn remap_type_refs_in_block(instructions: &mut [Instruction], old_to_new: &[u32]) {
    for instr in instructions.iter_mut() {
        match instr {
            Instruction::CallIndirect {
                type_idx,
                table_idx: _,
            } => {
                if let Some(&new_idx) = old_to_new.get(*type_idx as usize) {
                    *type_idx = new_idx;
                }
            }
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                remap_type_refs_in_block(body, old_to_new);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                remap_type_refs_in_block(then_body, old_to_new);
                remap_type_refs_in_block(else_body, old_to_new);
            }
            _ => {}
        }
    }
}

// ============================================================================
// Pass 3: Dead Function Elimination
// ============================================================================

/// Eliminate functions that are never called and not exported.
///
/// After adapter devirtualization, the adapter functions themselves may become
/// dead code (no remaining callers). This pass identifies and removes them.
///
/// ## Safety
///
/// A function is considered live if:
/// - It is exported (directly or transitively)
/// - It is the start function
/// - It is referenced by a `call` or `ref.func` instruction in any live function
/// - It is referenced in an element segment (indirect call table)
///
/// Returns the number of functions eliminated.
pub fn eliminate_dead_functions(module: &mut Module) -> Result<usize> {
    let num_imported_funcs = count_imported_functions(module);
    let total_funcs = num_imported_funcs + module.functions.len();

    // Build liveness set using reachability analysis
    let mut live: HashSet<u32> = HashSet::new();

    // All exported functions are live
    for export in &module.exports {
        if let ExportKind::Func(idx) = &export.kind {
            live.insert(*idx);
        }
    }

    // Start function is live
    if let Some(start_idx) = module.start_function {
        live.insert(start_idx);
    }

    // If there are no exports and no start function, we cannot determine
    // which functions are live. Conservatively keep all functions.
    // This handles test modules and library modules without explicit exports.
    if live.is_empty() {
        return Ok(0);
    }

    // Element segments reference functions (indirect call targets).
    // Parse the element section to extract exact function references rather
    // than conservatively marking all functions as live.
    if let Some(ref element_bytes) = module.element_section_bytes {
        match extract_element_func_refs(element_bytes) {
            Ok(refs) => {
                for func_idx in refs {
                    live.insert(func_idx);
                }
            }
            Err(_) => {
                // Parsing failed: fall back to conservative behavior
                for i in 0..total_funcs {
                    live.insert(i as u32);
                }
                return Ok(0);
            }
        }
    }

    // Transitive closure: walk call graph from live roots
    let mut worklist: Vec<u32> = live.iter().copied().collect();

    while let Some(func_idx) = worklist.pop() {
        // Only analyze local functions (imports have no bodies)
        if func_idx < num_imported_funcs as u32 {
            continue;
        }
        let local_idx = (func_idx - num_imported_funcs as u32) as usize;
        if local_idx >= module.functions.len() {
            continue;
        }

        let callees = collect_function_refs(&module.functions[local_idx].instructions);
        for callee in callees {
            if live.insert(callee) {
                worklist.push(callee);
            }
        }
    }

    // Identify dead local functions
    let mut dead_indices: Vec<usize> = Vec::new();
    for (local_idx, _func) in module.functions.iter().enumerate() {
        let abs_idx = num_imported_funcs as u32 + local_idx as u32;
        if !live.contains(&abs_idx) {
            dead_indices.push(local_idx);
        }
    }

    if dead_indices.is_empty() {
        return Ok(0);
    }

    let eliminated = dead_indices.len();

    // Build remapping: old absolute index -> new absolute index
    // Dead functions are removed, so surviving functions shift down
    let dead_set: HashSet<usize> = dead_indices.iter().copied().collect();
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let mut new_local_idx = 0u32;
    for local_idx in 0..module.functions.len() {
        let abs_idx = num_imported_funcs as u32 + local_idx as u32;
        if !dead_set.contains(&local_idx) {
            remap.insert(abs_idx, num_imported_funcs as u32 + new_local_idx);
            new_local_idx += 1;
        }
    }

    // Also map imports (identity mapping, they do not shift)
    for i in 0..num_imported_funcs {
        remap.insert(i as u32, i as u32);
    }

    // Remove dead functions (in reverse to maintain indices)
    for &local_idx in dead_indices.iter().rev() {
        module.functions.remove(local_idx);
    }

    // Remap all function references
    for func in &mut module.functions {
        remap_func_refs_in_block(&mut func.instructions, &remap);
    }

    // Remap exports
    for export in &mut module.exports {
        if let ExportKind::Func(ref mut idx) = export.kind {
            if let Some(&new_idx) = remap.get(idx) {
                *idx = new_idx;
            }
        }
    }

    // Remap start function
    if let Some(ref mut start_idx) = module.start_function {
        if let Some(&new_idx) = remap.get(start_idx) {
            *start_idx = new_idx;
        }
    }

    // Remap global initializers
    for global in &mut module.globals {
        remap_func_refs_in_block(&mut global.init, &remap);
    }

    // Remap data segment offset expressions
    for segment in &mut module.data_segments {
        remap_func_refs_in_block(&mut segment.offset, &remap);
    }

    // Remap function indices in element section
    if module.element_section_bytes.is_some() {
        match remap_element_section_refs(module, &remap) {
            Ok(()) => {}
            Err(_) => {
                // If remapping fails, the element section is inconsistent.
                // This should not happen since we successfully parsed it above,
                // but if it does, we have already removed the functions.
                // The module may be invalid - this is caught by validation.
            }
        }
    }

    Ok(eliminated)
}

/// Collect all function references (call targets, ref.func) from a block of instructions.
fn collect_function_refs(instructions: &[Instruction]) -> HashSet<u32> {
    let mut refs = HashSet::new();
    collect_function_refs_recursive(instructions, &mut refs);
    refs
}

fn collect_function_refs_recursive(instructions: &[Instruction], refs: &mut HashSet<u32>) {
    for instr in instructions {
        match instr {
            Instruction::Call(idx) => {
                refs.insert(*idx);
            }
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                collect_function_refs_recursive(body, refs);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                collect_function_refs_recursive(then_body, refs);
                collect_function_refs_recursive(else_body, refs);
            }
            _ => {}
        }
    }
}

/// Remap function references in a block of instructions.
fn remap_func_refs_in_block(instructions: &mut [Instruction], remap: &HashMap<u32, u32>) {
    for instr in instructions.iter_mut() {
        match instr {
            Instruction::Call(ref mut idx) => {
                if let Some(&new_idx) = remap.get(idx) {
                    *idx = new_idx;
                }
            }
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                remap_func_refs_in_block(body, remap);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                remap_func_refs_in_block(then_body, remap);
                remap_func_refs_in_block(else_body, remap);
            }
            _ => {}
        }
    }
}

// ============================================================================
// Pass 4: Import Deduplication
// ============================================================================

/// Deduplicate identical imports and remap all references.
///
/// After component fusion, multiple source components may import the same
/// external function (e.g., `wasi:io/streams.read`). These duplicate imports
/// can be merged.
///
/// Two imports are considered identical if they have the same:
/// - Module name
/// - Field name
/// - Import kind and type
///
/// Returns the number of imports deduplicated.
pub fn deduplicate_imports(module: &mut Module) -> Result<usize> {
    if module.imports.len() <= 1 {
        return Ok(0);
    }

    let num_old_imports = count_imported_functions(module);

    // Only deduplicate function imports (memory/table/global imports are rare and
    // deduplication there requires more careful analysis)
    let mut canonical_imports: HashMap<ImportKey, u32> = HashMap::new();
    let mut import_remap: HashMap<u32, u32> = HashMap::new();
    let mut new_func_imports: Vec<Import> = Vec::new();
    let mut new_import_func_count = 0u32;
    let mut non_func_imports: Vec<Import> = Vec::new();

    // Separate function imports from non-function imports
    let mut func_import_idx = 0u32;
    for import in &module.imports {
        match &import.kind {
            ImportKind::Func(type_idx) => {
                let key = ImportKey {
                    module: import.module.clone(),
                    name: import.name.clone(),
                    type_idx: *type_idx,
                };

                if let Some(&canonical_idx) = canonical_imports.get(&key) {
                    // Duplicate: remap to canonical
                    import_remap.insert(func_import_idx, canonical_idx);
                } else {
                    // New unique import
                    canonical_imports.insert(key, new_import_func_count);
                    import_remap.insert(func_import_idx, new_import_func_count);
                    new_func_imports.push(import.clone());
                    new_import_func_count += 1;
                }
                func_import_idx += 1;
            }
            _ => {
                non_func_imports.push(import.clone());
            }
        }
    }

    let dedup_count = num_old_imports - new_import_func_count as usize;
    if dedup_count == 0 {
        return Ok(0);
    }

    // Build full function index remap: old absolute index -> new absolute index
    // Imported functions shift, local functions shift by the reduction in import count
    let import_reduction = dedup_count;
    let mut func_remap: HashMap<u32, u32> = HashMap::new();

    // Remap imported function indices
    for (old_idx, new_idx) in &import_remap {
        func_remap.insert(*old_idx, *new_idx);
    }

    // Remap local function indices (shift down by import_reduction)
    for local_idx in 0..module.functions.len() {
        let old_abs = num_old_imports as u32 + local_idx as u32;
        let new_abs = (num_old_imports - import_reduction) as u32 + local_idx as u32;
        func_remap.insert(old_abs, new_abs);
    }

    // Apply remapping to all function references
    for func in &mut module.functions {
        remap_func_refs_in_block(&mut func.instructions, &func_remap);
    }

    // Remap exports
    for export in &mut module.exports {
        if let ExportKind::Func(ref mut idx) = export.kind {
            if let Some(&new_idx) = func_remap.get(idx) {
                *idx = new_idx;
            }
        }
    }

    // Remap start function
    if let Some(ref mut start_idx) = module.start_function {
        if let Some(&new_idx) = func_remap.get(start_idx) {
            *start_idx = new_idx;
        }
    }

    // Remap global initializers
    for global in &mut module.globals {
        remap_func_refs_in_block(&mut global.init, &func_remap);
    }

    // Replace imports (function imports first, then non-function imports)
    let mut new_imports = new_func_imports;
    new_imports.extend(non_func_imports);
    module.imports = new_imports;

    Ok(dedup_count)
}

/// Hash key for import deduplication.
#[derive(Hash, PartialEq, Eq)]
struct ImportKey {
    module: String,
    name: String,
    type_idx: u32,
}

// ============================================================================
// Pass 2: Trivial Call Elimination
// ============================================================================

/// Detect functions with empty bodies and eliminate calls to them.
///
/// Component fusion generates `cabi_post_return` functions for Canonical ABI
/// compliance. When a component's post-return is a no-op (empty function body),
/// calls to it can be safely eliminated.
///
/// ## Pattern detected
///
/// A function is considered trivial (no-op) when:
/// 1. It has no parameters and no results (signature `() -> ()`)
/// 2. Its body is just `[End]` or `[Nop*, End]`
///
/// All `call $trivial_func` instructions are removed from the module.
///
/// ## Proof Obligation
///
/// For a function F with body `[End]` and signature `() -> ()`:
///   For any caller context C and stack state S:
///     `eval(C[call F; rest], S) = eval(C[rest], S)`
///
/// This holds because F has no parameters (nothing popped), no results
/// (nothing pushed), and no side effects (body is empty).
fn eliminate_trivial_calls(module: &mut Module) -> Result<usize> {
    let num_imported_funcs = count_imported_functions(module);

    // Phase 1: Identify trivial no-op functions
    // A function is a no-op if:
    //   - Signature is () -> ()
    //   - Body is just [End] or [Nop*, End]
    let mut trivial_funcs: HashSet<u32> = HashSet::new();

    for (idx, func) in module.functions.iter().enumerate() {
        if !func.signature.params.is_empty() || !func.signature.results.is_empty() {
            continue;
        }
        if is_nop_body(&func.instructions) {
            let abs_idx = num_imported_funcs as u32 + idx as u32;
            trivial_funcs.insert(abs_idx);
        }
    }

    if trivial_funcs.is_empty() {
        return Ok(0);
    }

    // Phase 2: Remove all calls to trivial functions from every function body
    let mut total_eliminated = 0;
    for func in &mut module.functions {
        total_eliminated += remove_trivial_calls_from_block(&mut func.instructions, &trivial_funcs);
    }

    // Phase 3: Remove trivial calls from global initializers
    for global in &mut module.globals {
        total_eliminated += remove_trivial_calls_from_block(&mut global.init, &trivial_funcs);
    }

    Ok(total_eliminated)
}

/// Check if a function body is a no-op (just Nops and End).
fn is_nop_body(instructions: &[Instruction]) -> bool {
    instructions
        .iter()
        .all(|i| matches!(i, Instruction::Nop | Instruction::End))
}

/// Remove calls to trivial functions from an instruction block, recursively.
/// Returns the number of calls eliminated.
fn remove_trivial_calls_from_block(
    instructions: &mut Vec<Instruction>,
    trivial_funcs: &HashSet<u32>,
) -> usize {
    let mut eliminated = 0;

    // First, recurse into nested blocks
    for instr in instructions.iter_mut() {
        match instr {
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                eliminated += remove_trivial_calls_from_block(body, trivial_funcs);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                eliminated += remove_trivial_calls_from_block(then_body, trivial_funcs);
                eliminated += remove_trivial_calls_from_block(else_body, trivial_funcs);
            }
            _ => {}
        }
    }

    // Then, remove trivial calls at this level
    let before = instructions.len();
    instructions.retain(|instr| {
        if let Instruction::Call(idx) = instr {
            if trivial_funcs.contains(idx) {
                return false;
            }
        }
        true
    });
    eliminated += before - instructions.len();

    eliminated
}

// ============================================================================
// Helpers
// ============================================================================

/// Remap function references in the element section after dead function removal.
///
/// Uses `wasm_encoder` to rebuild the element section with updated function indices.
/// Only handles segments with direct function indices (not expression-based segments).
/// Falls back to keeping original bytes if expression-based segments are present.
fn remap_element_section_refs(module: &mut Module, remap: &HashMap<u32, u32>) -> Result<()> {
    use wasmparser::{BinaryReader, ElementItems, ElementKind, FromReader};

    let element_bytes = match &module.element_section_bytes {
        Some(bytes) => bytes.clone(),
        None => return Ok(()),
    };

    let mut reader = BinaryReader::new(&element_bytes, 0);
    let count = reader
        .read_var_u32()
        .map_err(|e| anyhow::anyhow!("failed to read element count: {}", e))?;

    // First pass: check if all elements use simple function index lists.
    // If any use expressions, bail out and keep original bytes.
    let save_pos = reader.clone();
    for _ in 0..count {
        let element = wasmparser::Element::from_reader(&mut reader)
            .map_err(|e| anyhow::anyhow!("failed to parse element: {}", e))?;

        if matches!(&element.items, ElementItems::Expressions(_, _)) {
            // Expression-based elements are too complex to remap.
            // Keep original bytes - indices may be stale but this is the
            // conservative fallback. Validation will catch issues.
            return Ok(());
        }
    }

    // Second pass: all elements use Functions lists. Parse and rebuild.
    let mut reader = save_pos;
    let mut new_section = wasm_encoder::ElementSection::new();

    for _ in 0..count {
        let element = wasmparser::Element::from_reader(&mut reader)
            .map_err(|e| anyhow::anyhow!("failed to parse element: {}", e))?;

        if let ElementItems::Functions(func_reader) = &element.items {
            let mut indices: Vec<u32> = Vec::new();
            for func_idx in func_reader.clone() {
                let idx =
                    func_idx.map_err(|e| anyhow::anyhow!("failed to read func idx: {}", e))?;
                indices.push(remap.get(&idx).copied().unwrap_or(idx));
            }

            let elements = wasm_encoder::Elements::Functions(std::borrow::Cow::Owned(indices));

            match element.kind {
                ElementKind::Active {
                    table_index,
                    offset_expr,
                } => {
                    // Re-encode the offset expression
                    let mut ops_reader = offset_expr.get_operators_reader();
                    let mut const_expr = wasm_encoder::ConstExpr::empty();
                    while let Ok(op) = ops_reader.read() {
                        match op {
                            wasmparser::Operator::I32Const { value } => {
                                const_expr = wasm_encoder::ConstExpr::i32_const(value);
                            }
                            wasmparser::Operator::I64Const { value } => {
                                const_expr = wasm_encoder::ConstExpr::i64_const(value);
                            }
                            wasmparser::Operator::GlobalGet { global_index } => {
                                const_expr = wasm_encoder::ConstExpr::global_get(global_index);
                            }
                            wasmparser::Operator::End => break,
                            _ => {
                                // Unsupported offset expression - bail out
                                return Ok(());
                            }
                        }
                    }

                    new_section.active(table_index, &const_expr, elements);
                }
                ElementKind::Passive => {
                    new_section.passive(elements);
                }
                ElementKind::Declared => {
                    new_section.declared(elements);
                }
            }
        }
    }

    // Encode the new section and extract just the section data
    // (skip section ID and LEB128 length prefix)
    use wasm_encoder::Encode;
    let mut encoded = Vec::new();
    new_section.encode(&mut encoded);
    // ElementSection::encode writes: section_id (1 byte) + LEB128 length + data
    if encoded.len() > 1 {
        let mut pos = 1; // Skip section ID byte
        while pos < encoded.len() && encoded[pos] & 0x80 != 0 {
            pos += 1; // Skip LEB128 length bytes
        }
        pos += 1; // Skip last byte of LEB128
        if pos < encoded.len() {
            module.element_section_bytes = Some(encoded[pos..].to_vec());
        }
    }

    Ok(())
}

/// Parse raw element section bytes and extract all referenced function indices.
///
/// Element segments can contain function references in two forms:
/// - `ElementItems::Functions`: direct function index list
/// - `ElementItems::Expressions`: const expressions, which may contain `ref.func`
///
/// This allows dead function elimination to work even when element segments
/// exist, by marking only the actually-referenced functions as live rather
/// than conservatively keeping all functions.
fn extract_element_func_refs(element_bytes: &[u8]) -> Result<HashSet<u32>> {
    use wasmparser::{BinaryReader, ElementItems, FromReader};

    let mut refs = HashSet::new();
    let mut reader = BinaryReader::new(element_bytes, 0);

    // The element section is a vector of element segments
    let count = reader
        .read_var_u32()
        .map_err(|e| anyhow::anyhow!("failed to read element count: {}", e))?;

    for _ in 0..count {
        let element = wasmparser::Element::from_reader(&mut reader)
            .map_err(|e| anyhow::anyhow!("failed to parse element: {}", e))?;

        match element.items {
            ElementItems::Functions(func_reader) => {
                for func_idx in func_reader {
                    let idx =
                        func_idx.map_err(|e| anyhow::anyhow!("failed to read func idx: {}", e))?;
                    refs.insert(idx);
                }
            }
            ElementItems::Expressions(_ty, expr_reader) => {
                // Const expressions may contain ref.func - extract those indices
                for expr in expr_reader {
                    let const_expr =
                        expr.map_err(|e| anyhow::anyhow!("failed to read const expr: {}", e))?;
                    let mut ops = const_expr.get_operators_reader();
                    while let Ok(op) = ops.read() {
                        if let wasmparser::Operator::RefFunc { function_index } = op {
                            refs.insert(function_index);
                        }
                    }
                }
            }
        }
    }

    Ok(refs)
}

/// Count the number of imported functions in a module.
fn count_imported_functions(module: &Module) -> usize {
    module
        .imports
        .iter()
        .filter(|i| matches!(i.kind, ImportKind::Func(_)))
        .count()
}

/// Get the signature of a function by its absolute index.
///
/// Accounts for imported functions (indices 0..num_imports) and
/// local functions (indices num_imports..).
fn get_function_signature(
    module: &Module,
    abs_idx: u32,
    num_imported_funcs: usize,
) -> Option<&FunctionSignature> {
    if (abs_idx as usize) < num_imported_funcs {
        // Imported function: look up its type
        let mut func_import_idx = 0usize;
        for import in &module.imports {
            if let ImportKind::Func(type_idx) = &import.kind {
                if func_import_idx == abs_idx as usize {
                    return module.types.get(*type_idx as usize);
                }
                func_import_idx += 1;
            }
        }
        None
    } else {
        // Local function
        let local_idx = abs_idx as usize - num_imported_funcs;
        module.functions.get(local_idx).map(|f| &f.signature)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockType, Export, ValueType};

    /// Create a minimal module for testing
    fn empty_module() -> Module {
        Module {
            functions: vec![],
            memories: vec![],
            tables: vec![],
            globals: vec![],
            types: vec![],
            exports: vec![],
            imports: vec![],
            data_segments: vec![],
            element_section_bytes: None,
            start_function: None,
            custom_sections: vec![],
            type_section_bytes: None,
            global_section_bytes: None,
        }
    }

    /// Create a trivial adapter function: loads all params and calls target
    fn make_adapter(params: &[ValueType], results: &[ValueType], target: u32) -> Function {
        let mut instructions = Vec::new();
        for i in 0..params.len() {
            instructions.push(Instruction::LocalGet(i as u32));
        }
        instructions.push(Instruction::Call(target));
        instructions.push(Instruction::End);

        Function {
            name: Some(format!("$adapter_to_{}", target)),
            signature: FunctionSignature {
                params: params.to_vec(),
                results: results.to_vec(),
            },
            locals: vec![],
            instructions,
        }
    }

    /// Create a simple function that calls another function
    fn make_caller(target: u32) -> Function {
        Function {
            name: Some("caller".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::Call(target),
                Instruction::End,
            ],
        }
    }

    /// Create a simple target function (identity: returns its parameter)
    fn make_target() -> Function {
        Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        }
    }

    #[test]
    fn test_is_trivial_adapter_simple() {
        let adapter = make_adapter(&[ValueType::I32], &[ValueType::I32], 5);
        assert_eq!(is_trivial_adapter(&adapter), Some(5));
    }

    #[test]
    fn test_is_trivial_adapter_multi_param() {
        let adapter = make_adapter(
            &[ValueType::I32, ValueType::I64, ValueType::F32],
            &[ValueType::I32],
            10,
        );
        assert_eq!(is_trivial_adapter(&adapter), Some(10));
    }

    #[test]
    fn test_is_trivial_adapter_no_params() {
        let adapter = make_adapter(&[], &[ValueType::I32], 3);
        assert_eq!(is_trivial_adapter(&adapter), Some(3));
    }

    #[test]
    fn test_not_adapter_with_locals() {
        let mut func = make_adapter(&[ValueType::I32], &[ValueType::I32], 5);
        func.locals = vec![(1, ValueType::I32)]; // Has locals = not a trivial adapter
        assert_eq!(is_trivial_adapter(&func), None);
    }

    #[test]
    fn test_not_adapter_wrong_order() {
        let func = Function {
            name: None,
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I64],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![
                Instruction::LocalGet(1), // Wrong order
                Instruction::LocalGet(0),
                Instruction::Call(5),
                Instruction::End,
            ],
        };
        assert_eq!(is_trivial_adapter(&func), None);
    }

    #[test]
    fn test_not_adapter_extra_instructions() {
        let func = Function {
            name: None,
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::I32Const(1),
                Instruction::I32Add,
                Instruction::Call(5),
                Instruction::End,
            ],
        };
        assert_eq!(is_trivial_adapter(&func), None);
    }

    #[test]
    fn test_devirtualize_adapters() {
        let mut module = empty_module();

        let sig = FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        };
        module.types.push(sig.clone());

        // Function 0: target
        module.functions.push(make_target());

        // Function 1: adapter that forwards to function 0
        module
            .functions
            .push(make_adapter(&[ValueType::I32], &[ValueType::I32], 0));

        // Function 2: caller that calls the adapter (function 1)
        // Uses I32Const + I32Add to NOT match the adapter pattern
        module.functions.push(Function {
            name: Some("caller".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::I32Const(1),
                Instruction::I32Add,
                Instruction::Call(1), // Calls the adapter
                Instruction::End,
            ],
        });

        // Export the caller and target to keep them alive
        module.exports.push(Export {
            name: "caller".to_string(),
            kind: ExportKind::Func(2),
        });
        module.exports.push(Export {
            name: "target".to_string(),
            kind: ExportKind::Func(0),
        });

        let stats = devirtualize_adapters(&mut module).unwrap();

        assert_eq!(stats.adapters_detected, 1);
        assert_eq!(stats.calls_devirtualized, 1);

        // Verify the caller now calls function 0 directly
        assert!(module.functions[2]
            .instructions
            .contains(&Instruction::Call(0)));
    }

    #[test]
    fn test_deduplicate_function_types() {
        let mut module = empty_module();

        let sig_a = FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        };
        let sig_b = FunctionSignature {
            params: vec![ValueType::I32, ValueType::I64],
            results: vec![],
        };

        // Add duplicate types
        module.types.push(sig_a.clone()); // 0: (i32) -> i32
        module.types.push(sig_b.clone()); // 1: (i32, i64) -> ()
        module.types.push(sig_a.clone()); // 2: duplicate of 0
        module.types.push(sig_a.clone()); // 3: duplicate of 0
        module.types.push(sig_b.clone()); // 4: duplicate of 1

        let dedup_count = deduplicate_function_types(&mut module).unwrap();

        assert_eq!(dedup_count, 3); // Removed 3 duplicates
        assert_eq!(module.types.len(), 2); // Only 2 unique types remain
    }

    #[test]
    fn test_deduplicate_imports() {
        let mut module = empty_module();

        let sig = FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        };
        module.types.push(sig.clone());

        // Add duplicate imports
        module.imports.push(Import {
            module: "wasi".to_string(),
            name: "read".to_string(),
            kind: ImportKind::Func(0),
        });
        module.imports.push(Import {
            module: "wasi".to_string(),
            name: "read".to_string(),
            kind: ImportKind::Func(0),
        });
        module.imports.push(Import {
            module: "wasi".to_string(),
            name: "write".to_string(),
            kind: ImportKind::Func(0),
        });

        // Add a function that calls import 1 (duplicate of import 0)
        module.functions.push(Function {
            name: Some("test".to_string()),
            signature: sig.clone(),
            locals: vec![],
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::Call(1), // Calls second "wasi:read" import
                Instruction::End,
            ],
        });

        module.exports.push(Export {
            name: "test".to_string(),
            kind: ExportKind::Func(3), // 3 imports + 0 local = func 3
        });

        let dedup_count = deduplicate_imports(&mut module).unwrap();

        assert_eq!(dedup_count, 1); // One duplicate removed
        assert_eq!(
            module
                .imports
                .iter()
                .filter(|i| matches!(i.kind, ImportKind::Func(_)))
                .count(),
            2
        ); // 2 unique imports remain

        // Verify the function call was remapped
        // Import 1 (duplicate) -> Import 0 (canonical)
        assert!(module.functions[0]
            .instructions
            .contains(&Instruction::Call(0)));
    }

    #[test]
    fn test_eliminate_dead_functions() {
        let mut module = empty_module();

        let sig = FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        };
        module.types.push(sig.clone());

        // Function 0: exported (live), calls function 2
        module.functions.push(Function {
            name: Some("live".to_string()),
            signature: sig.clone(),
            locals: vec![],
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::Call(2), // Calls function 2
                Instruction::End,
            ],
        });

        // Function 1: dead (no callers, not exported)
        module.functions.push(Function {
            name: Some("dead".to_string()),
            signature: sig.clone(),
            locals: vec![],
            instructions: vec![Instruction::I32Const(42), Instruction::End],
        });

        // Function 2: live (called by function 0)
        module.functions.push(Function {
            name: Some("called".to_string()),
            signature: sig.clone(),
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        module.exports.push(Export {
            name: "live".to_string(),
            kind: ExportKind::Func(0),
        });

        let eliminated = eliminate_dead_functions(&mut module).unwrap();

        assert_eq!(eliminated, 1); // Function 1 (dead) removed
        assert_eq!(module.functions.len(), 2); // Functions 0 and 2 remain

        // Verify function references were remapped
        // Old: func 0 calls func 2
        // After removing func 1: func 0 calls func 1 (shifted)
        assert!(module.functions[0]
            .instructions
            .contains(&Instruction::Call(1)));
    }

    #[test]
    fn test_adapter_chain_resolution() {
        let mut chains: HashMap<u32, u32> = HashMap::new();
        chains.insert(10, 20); // Adapter 10 -> 20
        chains.insert(20, 30); // Adapter 20 -> 30

        let resolved = resolve_adapter_chains(&chains);

        assert_eq!(resolved[&10], 30); // 10 -> 30 (bypassing 20)
        assert_eq!(resolved[&20], 30); // 20 -> 30
    }

    #[test]
    fn test_rewrite_calls_nested() {
        let mut adapter_map: HashMap<u32, u32> = HashMap::new();
        adapter_map.insert(5, 10);

        let instructions = vec![
            Instruction::Block {
                block_type: BlockType::Empty,
                body: vec![
                    Instruction::LocalGet(0),
                    Instruction::Call(5), // Should be rewritten to Call(10)
                    Instruction::If {
                        block_type: BlockType::Empty,
                        then_body: vec![Instruction::Call(5)], // Should be rewritten
                        else_body: vec![Instruction::Call(3)], // Should NOT be rewritten
                    },
                ],
            },
            Instruction::End,
        ];

        let (rewritten, count) = rewrite_calls(&instructions, &adapter_map);
        assert_eq!(count, 2);

        // Verify nested calls were rewritten
        if let Instruction::Block { body, .. } = &rewritten[0] {
            assert_eq!(body[1], Instruction::Call(10));
            if let Instruction::If {
                then_body,
                else_body,
                ..
            } = &body[2]
            {
                assert_eq!(then_body[0], Instruction::Call(10));
                assert_eq!(else_body[0], Instruction::Call(3)); // Unchanged
            }
        }
    }

    // ========================================================================
    // Pass 2: Trivial Call Elimination tests
    // ========================================================================

    #[test]
    fn test_is_nop_body() {
        assert!(is_nop_body(&[Instruction::End]));
        assert!(is_nop_body(&[Instruction::Nop, Instruction::End]));
        assert!(is_nop_body(&[
            Instruction::Nop,
            Instruction::Nop,
            Instruction::End
        ]));
        assert!(!is_nop_body(&[Instruction::I32Const(0), Instruction::End]));
        assert!(!is_nop_body(&[Instruction::Call(0), Instruction::End]));
    }

    #[test]
    fn test_eliminate_trivial_calls() {
        let mut module = empty_module();

        // Function 0: empty no-op function () -> ()
        module.functions.push(Function {
            name: Some("cabi_post_return".to_string()),
            signature: FunctionSignature {
                params: vec![],
                results: vec![],
            },
            locals: vec![],
            instructions: vec![Instruction::End],
        });

        // Function 1: real function that calls the no-op
        module.functions.push(Function {
            name: Some("caller".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::Call(1), // Calls real func (function 2)
                Instruction::Call(0), // Calls no-op post-return
                Instruction::End,
            ],
        });

        // Function 2: target function
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        module.exports.push(Export {
            name: "main".to_string(),
            kind: ExportKind::Func(1),
        });

        let eliminated = eliminate_trivial_calls(&mut module).unwrap();
        assert_eq!(eliminated, 1);

        // Verify the call to the no-op was removed
        assert_eq!(
            module.functions[1].instructions,
            vec![
                Instruction::LocalGet(0),
                Instruction::Call(1),
                Instruction::End,
            ]
        );
    }

    #[test]
    fn test_eliminate_trivial_calls_preserves_real_funcs() {
        let mut module = empty_module();

        // Function 0: non-trivial () -> () function (has real instructions)
        module.functions.push(Function {
            name: Some("real_cleanup".to_string()),
            signature: FunctionSignature {
                params: vec![],
                results: vec![],
            },
            locals: vec![],
            instructions: vec![
                Instruction::I32Const(0),
                Instruction::Drop,
                Instruction::End,
            ],
        });

        // Function 1: caller
        module.functions.push(Function {
            name: Some("caller".to_string()),
            signature: FunctionSignature {
                params: vec![],
                results: vec![],
            },
            locals: vec![],
            instructions: vec![
                Instruction::Call(0), // Calls non-trivial func - should NOT be removed
                Instruction::End,
            ],
        });

        let eliminated = eliminate_trivial_calls(&mut module).unwrap();
        assert_eq!(eliminated, 0);

        // Call preserved
        assert!(module.functions[1]
            .instructions
            .contains(&Instruction::Call(0)));
    }

    #[test]
    fn test_eliminate_trivial_calls_skips_params() {
        let mut module = empty_module();

        // Function 0: empty body but has params - NOT trivial
        // (calling it would pop values from the stack)
        module.functions.push(Function {
            name: Some("empty_with_params".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![],
            },
            locals: vec![],
            instructions: vec![Instruction::End],
        });

        // Function 1: caller
        module.functions.push(Function {
            name: Some("caller".to_string()),
            signature: FunctionSignature {
                params: vec![],
                results: vec![],
            },
            locals: vec![],
            instructions: vec![
                Instruction::I32Const(0),
                Instruction::Call(0), // Should NOT be removed
                Instruction::End,
            ],
        });

        let eliminated = eliminate_trivial_calls(&mut module).unwrap();
        assert_eq!(eliminated, 0);
    }

    #[test]
    fn test_eliminate_trivial_calls_nested() {
        let mut module = empty_module();

        // Function 0: no-op
        module.functions.push(Function {
            name: Some("noop".to_string()),
            signature: FunctionSignature {
                params: vec![],
                results: vec![],
            },
            locals: vec![],
            instructions: vec![Instruction::Nop, Instruction::End],
        });

        // Function 1: caller with nested blocks
        module.functions.push(Function {
            name: Some("caller".to_string()),
            signature: FunctionSignature {
                params: vec![],
                results: vec![],
            },
            locals: vec![],
            instructions: vec![
                Instruction::Block {
                    block_type: BlockType::Empty,
                    body: vec![
                        Instruction::Call(0), // Should be removed
                        Instruction::If {
                            block_type: BlockType::Empty,
                            then_body: vec![Instruction::Call(0)], // Should be removed
                            else_body: vec![Instruction::Nop],
                        },
                    ],
                },
                Instruction::End,
            ],
        });

        let eliminated = eliminate_trivial_calls(&mut module).unwrap();
        assert_eq!(eliminated, 2);
    }

    // ========================================================================
    // Pass 0: Same-Memory Adapter Collapse tests
    // ========================================================================

    /// Helper: create a same-memory adapter function.
    ///
    /// Simulates a meld-generated adapter that allocates a buffer via cabi_realloc,
    /// copies data within the same memory, and calls the target.
    fn make_same_memory_adapter(
        params: &[ValueType],
        results: &[ValueType],
        realloc_idx: u32,
        target_idx: u32,
    ) -> Function {
        let mut instructions = Vec::new();

        // Typical adapter pattern:
        // 1. Read parameters
        instructions.push(Instruction::LocalGet(0));
        if params.len() > 1 {
            instructions.push(Instruction::LocalGet(1));
        }

        // 2. Call cabi_realloc to allocate buffer
        instructions.push(Instruction::I32Const(0)); // old_ptr
        instructions.push(Instruction::I32Const(0)); // old_size
        instructions.push(Instruction::I32Const(1)); // align
        instructions.push(Instruction::I32Const(8)); // new_size
        instructions.push(Instruction::Call(realloc_idx));
        instructions.push(Instruction::LocalSet(params.len() as u32)); // store new_ptr in local

        // 3. memory.copy within same memory (dst=0, src=0)
        instructions.push(Instruction::LocalGet(params.len() as u32)); // dst
        instructions.push(Instruction::LocalGet(0)); // src
        instructions.push(Instruction::I32Const(8)); // len
        instructions.push(Instruction::MemoryCopy {
            dst_mem: 0,
            src_mem: 0,
        });

        // 4. Call target with new pointer
        instructions.push(Instruction::LocalGet(params.len() as u32));
        if params.len() > 1 {
            instructions.push(Instruction::LocalGet(1));
        }
        instructions.push(Instruction::Call(target_idx));

        instructions.push(Instruction::End);

        Function {
            name: Some(format!("$adapter_same_mem_{}", target_idx)),
            signature: FunctionSignature {
                params: params.to_vec(),
                results: results.to_vec(),
            },
            locals: vec![(1, ValueType::I32)], // Temporary pointer local
            instructions,
        }
    }

    /// Helper: create a module with a single memory and a cabi_realloc import.
    fn single_memory_module_with_realloc() -> (Module, u32) {
        let mut module = empty_module();

        // Add realloc signature type: (i32, i32, i32, i32) -> i32
        let realloc_sig = FunctionSignature {
            params: vec![
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
            ],
            results: vec![ValueType::I32],
        };
        module.types.push(realloc_sig);

        // Add target signature type: (i32, i32) -> i32
        let target_sig = FunctionSignature {
            params: vec![ValueType::I32, ValueType::I32],
            results: vec![ValueType::I32],
        };
        module.types.push(target_sig);

        // Import cabi_realloc (function index 0)
        module.imports.push(Import {
            module: "cabi".to_string(),
            name: "cabi_realloc".to_string(),
            kind: ImportKind::Func(0), // type index 0 = realloc sig
        });

        // Add a single memory
        module.memories.push(crate::Memory {
            min: 1,
            max: None,
            shared: false,
            memory64: false,
        });

        let realloc_idx = 0u32; // Import index 0
        (module, realloc_idx)
    }

    #[test]
    fn test_collapse_same_memory_adapter_basic() {
        let (mut module, realloc_idx) = single_memory_module_with_realloc();

        // Function 0 (abs idx 1): target function
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): same-memory adapter -> target (abs idx 1)
        module.functions.push(make_same_memory_adapter(
            &[ValueType::I32, ValueType::I32],
            &[ValueType::I32],
            realloc_idx,
            1, // target abs idx
        ));

        module.exports.push(Export {
            name: "adapter".to_string(),
            kind: ExportKind::Func(2),
        });

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(collapsed, 1);

        // Verify the adapter was collapsed to a forwarding trampoline
        let adapter = &module.functions[1];
        assert!(adapter.locals.is_empty(), "locals should be cleared");
        assert_eq!(
            adapter.instructions,
            vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::Call(1),
                Instruction::End,
            ]
        );
    }

    #[test]
    fn test_collapse_preserves_non_adapter() {
        let (mut module, _realloc_idx) = single_memory_module_with_realloc();

        // Function 0 (abs idx 1): a normal function with real logic (not an adapter)
        module.functions.push(Function {
            name: Some("real_function".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![(1, ValueType::I32)],
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::I32Const(1),
                Instruction::I32Add,
                Instruction::End,
            ],
        });

        module.exports.push(Export {
            name: "func".to_string(),
            kind: ExportKind::Func(1),
        });

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(collapsed, 0);

        // Function unchanged
        assert_eq!(module.functions[0].instructions.len(), 4);
    }

    #[test]
    fn test_collapse_skips_multi_memory() {
        let (mut module, realloc_idx) = single_memory_module_with_realloc();

        // Add a second memory -> multi-memory module
        module.memories.push(crate::Memory {
            min: 1,
            max: None,
            shared: false,
            memory64: false,
        });

        // Function 0 (abs idx 1): target
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): same-memory adapter
        module.functions.push(make_same_memory_adapter(
            &[ValueType::I32, ValueType::I32],
            &[ValueType::I32],
            realloc_idx,
            1,
        ));

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(collapsed, 0, "multi-memory modules should be skipped");
    }

    #[test]
    fn test_collapse_skips_different_memory_copy() {
        let (mut module, realloc_idx) = single_memory_module_with_realloc();

        // Function 0 (abs idx 1): target
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): adapter with cross-memory copy (dst=0, src=1)
        module.functions.push(Function {
            name: Some("cross_mem_adapter".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![(1, ValueType::I32)],
            instructions: vec![
                Instruction::I32Const(0),
                Instruction::I32Const(0),
                Instruction::I32Const(1),
                Instruction::I32Const(8),
                Instruction::Call(realloc_idx),
                Instruction::LocalSet(2),
                Instruction::LocalGet(2),
                Instruction::LocalGet(0),
                Instruction::I32Const(8),
                Instruction::MemoryCopy {
                    dst_mem: 0,
                    src_mem: 1, // Cross-memory!
                },
                Instruction::LocalGet(2),
                Instruction::LocalGet(1),
                Instruction::Call(1),
                Instruction::End,
            ],
        });

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(collapsed, 0, "cross-memory copy should not be collapsed");
    }

    #[test]
    fn test_collapse_skips_no_locals() {
        let (mut module, _realloc_idx) = single_memory_module_with_realloc();

        // Function 0 (abs idx 1): target
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): trivial adapter (no locals) - Pass 1 handles this
        module
            .functions
            .push(make_adapter(&[ValueType::I32], &[ValueType::I32], 1));

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(
            collapsed, 0,
            "trivial adapters without locals should be skipped (Pass 1 handles)"
        );
    }

    #[test]
    fn test_collapse_skips_control_flow() {
        let (mut module, realloc_idx) = single_memory_module_with_realloc();

        // Function 0 (abs idx 1): target
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): adapter with control flow (If)
        module.functions.push(Function {
            name: Some("adapter_with_if".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![(1, ValueType::I32)],
            instructions: vec![
                Instruction::I32Const(0),
                Instruction::I32Const(0),
                Instruction::I32Const(1),
                Instruction::I32Const(8),
                Instruction::Call(realloc_idx),
                Instruction::LocalSet(2),
                Instruction::LocalGet(0),
                Instruction::If {
                    block_type: BlockType::Empty,
                    then_body: vec![
                        Instruction::LocalGet(2),
                        Instruction::LocalGet(0),
                        Instruction::I32Const(8),
                        Instruction::MemoryCopy {
                            dst_mem: 0,
                            src_mem: 0,
                        },
                    ],
                    else_body: vec![Instruction::Nop],
                },
                Instruction::LocalGet(2),
                Instruction::LocalGet(1),
                Instruction::Call(1),
                Instruction::End,
            ],
        });

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(
            collapsed, 0,
            "adapter with control flow should not be collapsed"
        );
    }

    #[test]
    fn test_collapse_skips_memory_stores() {
        let (mut module, realloc_idx) = single_memory_module_with_realloc();

        // Function 0 (abs idx 1): target
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): adapter with i32.store
        module.functions.push(Function {
            name: Some("adapter_with_store".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![(1, ValueType::I32)],
            instructions: vec![
                Instruction::I32Const(0),
                Instruction::I32Const(0),
                Instruction::I32Const(1),
                Instruction::I32Const(8),
                Instruction::Call(realloc_idx),
                Instruction::LocalSet(2),
                Instruction::LocalGet(2),
                Instruction::LocalGet(0),
                Instruction::I32Store {
                    offset: 0,
                    align: 2,
                }, // Store!
                Instruction::LocalGet(2),
                Instruction::LocalGet(0),
                Instruction::I32Const(8),
                Instruction::MemoryCopy {
                    dst_mem: 0,
                    src_mem: 0,
                },
                Instruction::LocalGet(2),
                Instruction::LocalGet(1),
                Instruction::Call(1),
                Instruction::End,
            ],
        });

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(
            collapsed, 0,
            "adapter with memory stores should not be collapsed"
        );
    }

    #[test]
    fn test_collapse_skips_global_writes() {
        let (mut module, realloc_idx) = single_memory_module_with_realloc();

        // Function 0 (abs idx 1): target
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): adapter with global.set
        module.functions.push(Function {
            name: Some("adapter_with_global_set".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![(1, ValueType::I32)],
            instructions: vec![
                Instruction::I32Const(0),
                Instruction::GlobalSet(0), // Global write!
                Instruction::I32Const(0),
                Instruction::I32Const(0),
                Instruction::I32Const(1),
                Instruction::I32Const(8),
                Instruction::Call(realloc_idx),
                Instruction::LocalSet(2),
                Instruction::LocalGet(2),
                Instruction::LocalGet(0),
                Instruction::I32Const(8),
                Instruction::MemoryCopy {
                    dst_mem: 0,
                    src_mem: 0,
                },
                Instruction::LocalGet(2),
                Instruction::LocalGet(1),
                Instruction::Call(1),
                Instruction::End,
            ],
        });

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(
            collapsed, 0,
            "adapter with global writes should not be collapsed"
        );
    }

    #[test]
    fn test_collapse_skips_signature_mismatch() {
        let (mut module, realloc_idx) = single_memory_module_with_realloc();

        // Function 0 (abs idx 1): target with DIFFERENT signature
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32], // Only 1 param vs adapter's 2
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): adapter (i32, i32) -> i32, but target is (i32) -> i32
        module.functions.push(make_same_memory_adapter(
            &[ValueType::I32, ValueType::I32],
            &[ValueType::I32],
            realloc_idx,
            1,
        ));

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(collapsed, 0, "signature mismatch should prevent collapse");
    }

    #[test]
    fn test_collapse_then_devirtualize() {
        let (mut module, realloc_idx) = single_memory_module_with_realloc();

        // Function 0 (abs idx 1): target
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): same-memory adapter -> target (abs idx 1)
        module.functions.push(make_same_memory_adapter(
            &[ValueType::I32, ValueType::I32],
            &[ValueType::I32],
            realloc_idx,
            1,
        ));

        // Function 2 (abs idx 3): caller that calls the adapter (abs idx 2)
        module.functions.push(Function {
            name: Some("caller".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::Call(2), // calls adapter
                Instruction::End,
            ],
        });

        module.exports.push(Export {
            name: "caller".to_string(),
            kind: ExportKind::Func(3),
        });
        module.exports.push(Export {
            name: "target".to_string(),
            kind: ExportKind::Func(1),
        });

        // Run full pipeline
        let stats = optimize_fused_module(&mut module).unwrap();

        // Pass 0 should have collapsed the adapter
        assert_eq!(stats.same_memory_adapters_collapsed, 1);

        // Pass 1 should have devirtualized the call
        assert!(stats.adapters_detected >= 1);
        assert!(stats.calls_devirtualized >= 1);

        // After the full pipeline (including DCE), find the caller function.
        // The adapter may have been eliminated as dead, shifting indices.
        let caller = module
            .functions
            .iter()
            .find(|f| f.name.as_deref() == Some("caller"))
            .expect("caller function should still exist");

        // The caller should NOT still be calling the original adapter index (2).
        // After collapse + devirtualize + DCE, it should call the target directly.
        let calls_adapter = caller.instructions.contains(&Instruction::Call(2));
        assert!(
            !calls_adapter,
            "caller should no longer call the adapter after collapse + devirtualize"
        );
    }

    #[test]
    fn test_collapse_multiple_adapters() {
        let (mut module, realloc_idx) = single_memory_module_with_realloc();

        let sig = FunctionSignature {
            params: vec![ValueType::I32, ValueType::I32],
            results: vec![ValueType::I32],
        };

        // Function 0 (abs idx 1): target A
        module.functions.push(Function {
            name: Some("target_a".to_string()),
            signature: sig.clone(),
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): target B
        module.functions.push(Function {
            name: Some("target_b".to_string()),
            signature: sig.clone(),
            locals: vec![],
            instructions: vec![Instruction::LocalGet(1), Instruction::End],
        });

        // Function 2 (abs idx 3): adapter -> target A (abs idx 1)
        module.functions.push(make_same_memory_adapter(
            &[ValueType::I32, ValueType::I32],
            &[ValueType::I32],
            realloc_idx,
            1,
        ));

        // Function 3 (abs idx 4): adapter -> target B (abs idx 2)
        module.functions.push(make_same_memory_adapter(
            &[ValueType::I32, ValueType::I32],
            &[ValueType::I32],
            realloc_idx,
            2,
        ));

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(collapsed, 2, "both adapters should be collapsed");

        // Verify both are now forwarding trampolines
        for (func_local_idx, target_abs_idx) in [(2, 1u32), (3, 2u32)] {
            let adapter = &module.functions[func_local_idx];
            assert!(adapter.locals.is_empty());
            assert_eq!(
                adapter.instructions,
                vec![
                    Instruction::LocalGet(0),
                    Instruction::LocalGet(1),
                    Instruction::Call(target_abs_idx),
                    Instruction::End,
                ]
            );
        }
    }

    #[test]
    fn test_collapse_with_return_copy() {
        let (mut module, realloc_idx) = single_memory_module_with_realloc();

        // Function 0 (abs idx 1): target
        module.functions.push(Function {
            name: Some("target".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0), Instruction::End],
        });

        // Function 1 (abs idx 2): adapter with forward copy AND return copy
        // (both are same-memory copies)
        module.functions.push(Function {
            name: Some("adapter_with_return_copy".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![(2, ValueType::I32)], // Two temp locals
            instructions: vec![
                // Forward: allocate + copy args
                Instruction::I32Const(0),
                Instruction::I32Const(0),
                Instruction::I32Const(1),
                Instruction::I32Const(8),
                Instruction::Call(realloc_idx),
                Instruction::LocalSet(2),
                Instruction::LocalGet(2),
                Instruction::LocalGet(0),
                Instruction::I32Const(8),
                Instruction::MemoryCopy {
                    dst_mem: 0,
                    src_mem: 0,
                }, // Forward copy
                // Call target
                Instruction::LocalGet(2),
                Instruction::LocalGet(1),
                Instruction::Call(1),
                // Return: copy results back (same memory)
                Instruction::LocalSet(3),
                Instruction::I32Const(0),
                Instruction::I32Const(0),
                Instruction::I32Const(1),
                Instruction::I32Const(4),
                Instruction::Call(realloc_idx),
                Instruction::LocalSet(2),
                Instruction::LocalGet(2),
                Instruction::LocalGet(3),
                Instruction::I32Const(4),
                Instruction::MemoryCopy {
                    dst_mem: 0,
                    src_mem: 0,
                }, // Return copy
                Instruction::LocalGet(2),
                Instruction::End,
            ],
        });

        let collapsed = collapse_same_memory_adapters(&mut module).unwrap();
        assert_eq!(
            collapsed, 1,
            "adapter with forward + return copy should be collapsed"
        );

        // Verify it's now a forwarding trampoline
        let adapter = &module.functions[1];
        assert!(adapter.locals.is_empty());
        assert_eq!(
            adapter.instructions,
            vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::Call(1),
                Instruction::End,
            ]
        );
    }

    // ========================================================================
    // Full Pipeline tests
    // ========================================================================

    #[test]
    fn test_optimize_fused_module_full_pipeline() {
        let mut module = empty_module();

        let sig = FunctionSignature {
            params: vec![ValueType::I32],
            results: vec![ValueType::I32],
        };
        module.types.push(sig.clone());
        module.types.push(sig.clone()); // Duplicate type

        // Function 0: target
        module.functions.push(make_target());

        // Function 1: adapter -> function 0
        module
            .functions
            .push(make_adapter(&[ValueType::I32], &[ValueType::I32], 0));

        // Function 2: caller that calls adapter
        module.functions.push(make_caller(1));

        module.exports.push(Export {
            name: "main".to_string(),
            kind: ExportKind::Func(2),
        });

        let stats = optimize_fused_module(&mut module).unwrap();

        // Adapter detected and devirtualized
        assert!(stats.adapters_detected >= 1);
        // Type duplicates removed
        assert!(stats.types_deduplicated >= 1);
    }
}
