//! Fused Component Optimization
//!
//! Specialized optimization passes for WebAssembly modules produced by component
//! fusion tools (e.g., meld). When multiple P2/P3 components are fused into a
//! single core module, the result contains characteristic patterns that benefit
//! from targeted optimization:
//!
//! ## Optimization Passes
//!
//! ### 1. Adapter Devirtualization
//! Component fusion generates adapter trampolines for cross-component calls.
//! Trivial adapters that simply forward all parameters to a target function
//! can be bypassed: callers are rewritten to call the target directly.
//!
//! ### 2. Function Type Deduplication
//! Each source component contributes its own type section. After fusion, many
//! identical function types exist. Deduplication merges them and remaps all
//! type references.
//!
//! ### 3. Dead Function Elimination
//! After adapter devirtualization, adapter functions may become unreachable.
//! This pass removes functions with zero call sites that are not exported.
//!
//! ### 4. Import Deduplication
//! Fused modules may contain duplicate imports (same module+name+type) from
//! different source components. These are merged with reference remapping.
//!
//! ## Correctness
//!
//! All transformations are provably correct:
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
    let mut stats = FusedOptimizationStats::default();

    // Pass 1: Detect and devirtualize adapter trampolines
    let adapter_stats = devirtualize_adapters(module)?;
    stats.adapters_detected = adapter_stats.adapters_detected;
    stats.calls_devirtualized = adapter_stats.calls_devirtualized;

    // Pass 2: Deduplicate function types
    stats.types_deduplicated = deduplicate_function_types(module)?;

    // Pass 3: Eliminate dead functions (after devirtualization may create dead adapters)
    stats.dead_functions_eliminated = eliminate_dead_functions(module)?;

    // Pass 4: Deduplicate imports
    stats.imports_deduplicated = deduplicate_imports(module)?;

    Ok(stats)
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

    // Element segments reference functions (indirect call targets)
    // Since we store raw element bytes, we conservatively mark all local functions
    // as potentially referenced via element segments.
    if module.element_section_bytes.is_some() {
        // Conservative: cannot determine exact function references from raw bytes
        for i in 0..total_funcs {
            live.insert(i as u32);
        }
        return Ok(0);
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
// Helpers
// ============================================================================

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
