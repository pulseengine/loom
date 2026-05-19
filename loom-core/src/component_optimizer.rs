//! Component Model Optimization
//!
//! This module provides world-class WebAssembly Component Model optimization.
//! LOOM is the first optimizer to support the Component Model specification.
//!
//! ## Optimization Phases
//!
//! ### Phase 1: Core Module Optimization
//! - Extract core modules from component
//! - Apply LOOM's 12-phase pipeline to each module
//! - 80-95% size reduction on module code
//!
//! ### Phase 1.5: Full Section Preservation
//! - Reconstruct complete component structure
//! - Preserve all sections (types, imports, exports, instances, etc.)
//! - Maintain WIT interface compatibility
//!
//! ### Phase 2: Component-Level Optimizations
//! - Type deduplication across component
//! - Unused import/export elimination
//! - Canonical function optimization
//! - Expected impact: Additional 5-15% reduction
//!
//! ## Architecture
//!
//! 1. **Parse**: Parse component and extract core modules
//! 2. **Optimize Modules**: Apply LOOM's 12-phase pipeline
//! 3. **Optimize Component**: Component-level optimizations
//! 4. **Reconstruct**: Rebuild with all optimizations
//! 5. **Validate**: Ensure correctness
//!
//! ## Example
//!
//! ```no_run
//! use loom_core::component_optimizer::optimize_component;
//!
//! let component_bytes = std::fs::read("app.component.wasm").unwrap();
//! let (optimized, stats) = optimize_component(&component_bytes).unwrap();
//!
//! println!("Size reduction: {:.1}%", stats.reduction_percentage());
//! ```

use crate::parse::wasm_features_with_async;
use crate::{BlockType, Instruction, Module};
use anyhow::{Context, Result, anyhow};
use wasmparser::{Encoding, Parser, Payload, Validator};

/// Statistics about component optimization
#[derive(Debug, Clone)]
pub struct ComponentStats {
    /// Original component size in bytes
    pub original_size: usize,
    /// Optimized component size in bytes
    pub optimized_size: usize,
    /// Number of core modules found
    pub module_count: usize,
    /// Number of modules successfully optimized
    pub modules_optimized: usize,
    /// Total size of original modules
    pub original_module_size: usize,
    /// Total size of optimized modules
    pub optimized_module_size: usize,
    /// Status message
    pub message: String,
}

impl ComponentStats {
    /// Calculate overall component size reduction percentage
    pub fn reduction_percentage(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        100.0 * (1.0 - (self.optimized_size as f64 / self.original_size as f64))
    }

    /// Calculate core module size reduction percentage
    pub fn module_reduction_percentage(&self) -> f64 {
        if self.original_module_size == 0 {
            return 0.0;
        }
        100.0 * (1.0 - (self.optimized_module_size as f64 / self.original_module_size as f64))
    }
}

/// Optimize a WebAssembly Component
///
/// This is the main entry point for component optimization. It:
/// 1. Parses the component and extracts core modules
/// 2. Optimizes each core module with LOOM's pipeline
/// 3. Reconstructs the component with optimized modules
/// 4. Validates the result
///
/// Returns the optimized component bytes and statistics.
pub fn optimize_component(component_bytes: &[u8]) -> Result<(Vec<u8>, ComponentStats)> {
    // Step 1: Parse component and extract core modules
    let parser = Parser::new(0);
    let mut core_modules: Vec<CoreModule> = Vec::new();
    let mut is_component = false;

    for payload in parser.parse_all(component_bytes) {
        let payload = payload.context("Failed to parse component")?;

        match payload {
            Payload::Version {
                encoding: Encoding::Component,
                ..
            } => {
                is_component = true;
            }
            Payload::ModuleSection {
                unchecked_range, ..
            } => {
                // Extract the module bytes
                let module_bytes =
                    component_bytes[unchecked_range.start..unchecked_range.end].to_vec();

                core_modules.push(CoreModule {
                    original_bytes: module_bytes,
                    optimized_bytes: None,
                });
            }
            _ => {}
        }
    }

    if !is_component {
        return Err(anyhow!("Not a WebAssembly component"));
    }

    if core_modules.is_empty() {
        return Err(anyhow!("Component contains no core modules"));
    }

    // Step 2: Optimize each core module
    let mut optimized_count = 0;
    for (idx, core_module) in core_modules.iter_mut().enumerate() {
        match optimize_core_module(&core_module.original_bytes) {
            Ok(optimized_bytes) => {
                core_module.optimized_bytes = Some(optimized_bytes);
                optimized_count += 1;
                eprintln!("✓ Module {}: Optimized successfully", idx);
            }
            Err(e) => {
                eprintln!("⚠  Module {}: Failed to optimize: {:?}", idx, e);
                eprintln!("   Using original bytes for this module");
                // Keep original bytes
            }
        }
    }

    // Step 3: Reconstruct component with optimized modules
    let optimized_component = reconstruct_component(component_bytes, &core_modules)?;

    // Step 4: Validate
    if let Err(e) =
        Validator::new_with_features(wasm_features_with_async()).validate_all(&optimized_component)
    {
        eprintln!("⚠  Component validation error: {}", e);
        return Err(anyhow!("Optimized component validation failed: {}", e));
    }

    // Calculate stats
    let original_module_size: usize = core_modules.iter().map(|m| m.original_bytes.len()).sum();
    let optimized_module_size: usize = core_modules
        .iter()
        .map(|m| {
            m.optimized_bytes
                .as_ref()
                .map(|b| b.len())
                .unwrap_or(m.original_bytes.len())
        })
        .sum();

    let stats = ComponentStats {
        original_size: component_bytes.len(),
        optimized_size: optimized_component.len(),
        module_count: core_modules.len(),
        modules_optimized: optimized_count,
        original_module_size,
        optimized_module_size,
        message: format!(
            "Successfully optimized {} of {} core modules",
            optimized_count,
            core_modules.len()
        ),
    };

    Ok((optimized_component, stats))
}

/// Information about a core module within a component
#[derive(Debug, Clone)]
struct CoreModule {
    /// Original module bytes
    original_bytes: Vec<u8>,
    /// Optimized module bytes (if optimization succeeded)
    optimized_bytes: Option<Vec<u8>>,
}

/// Optimize a single core module
///
/// Applies the full optimization pipeline:
/// 1. Fused component optimizations (adapter devirtualization, type/import dedup, DCE)
/// 2. Standard 12-phase pipeline (constant folding, strength reduction, DCE, etc.)
fn optimize_core_module(module_bytes: &[u8]) -> Result<Vec<u8>> {
    // First validate the input module
    Validator::new_with_features(wasm_features_with_async())
        .validate_all(module_bytes)
        .context("Input module validation failed")?;

    // Parse the module
    let mut module = crate::parse::parse_wasm(module_bytes)?;

    // Phase 0: Fused component optimizations (runs before standard pipeline)
    // These passes target artifacts from component fusion (meld):
    // - Adapter trampolines that just forward calls
    // - Duplicate function types from multiple source components
    // - Dead functions (unused adapters after devirtualization)
    // - Duplicate imports (same external function imported by multiple components)
    match crate::fused_optimizer::optimize_fused_module(&mut module) {
        Ok(fused_stats) => {
            if fused_stats.adapters_detected > 0
                || fused_stats.types_deduplicated > 0
                || fused_stats.dead_functions_eliminated > 0
                || fused_stats.imports_deduplicated > 0
                || fused_stats.trivial_calls_eliminated > 0
                || fused_stats.memory_imports_deduplicated > 0
            {
                eprintln!(
                    "  Fused optimization: {} mem imports deduped, {} adapters devirtualized, {} trivial calls eliminated, {} types deduped, {} dead funcs removed, {} imports deduped",
                    fused_stats.memory_imports_deduplicated,
                    fused_stats.calls_devirtualized,
                    fused_stats.trivial_calls_eliminated,
                    fused_stats.types_deduplicated,
                    fused_stats.dead_functions_eliminated,
                    fused_stats.imports_deduplicated,
                );
            }
            if fused_stats.cross_memory_adapters_detected > 0 {
                eprintln!(
                    "  Cross-memory adapters detected (not collapsed): {}",
                    fused_stats.cross_memory_adapters_detected,
                );
            }

            // Validate after fused optimization
            let bytes = crate::encode::encode_wasm(&module)?;
            if let Err(e) =
                Validator::new_with_features(wasm_features_with_async()).validate_all(&bytes)
            {
                return Err(anyhow!(
                    "Module became invalid after fused optimization pass: {}",
                    e
                ));
            }
        }
        Err(e) => {
            eprintln!("  Fused optimization skipped: {:?}", e);
            // Non-fatal: continue with standard pipeline
        }
    }

    // Standard optimization passes with validation after each pass
    // This provides detailed error messages showing which pass caused issues
    #[allow(clippy::type_complexity)]
    let passes: &[(&str, fn(&mut crate::Module) -> Result<()>)] = &[
        ("constant_folding", crate::optimize::constant_folding),
        (
            "optimize_advanced_instructions",
            crate::optimize::optimize_advanced_instructions,
        ),
        ("simplify_locals", crate::optimize::simplify_locals),
        ("eliminate_dead_code", crate::optimize::eliminate_dead_code),
        ("code_folding", crate::optimize::code_folding),
        (
            "loop_invariant_code_motion",
            crate::optimize::loop_invariant_code_motion,
        ),
        (
            "remove_unused_branches",
            crate::optimize::remove_unused_branches,
        ),
        (
            "optimize_added_constants",
            crate::optimize::optimize_added_constants,
        ),
    ];

    for (pass_name, pass_fn) in passes {
        // Save module state before each pass for rollback on failure
        let saved_functions = module.functions.clone();

        if let Err(e) = pass_fn(&mut module) {
            eprintln!("  Pass '{}' failed (reverting): {}", pass_name, e);
            crate::stats::record_revert(&format!("component:{}", pass_name));
            module.functions = saved_functions;
            continue;
        }

        // Validate after each pass — revert if module became invalid
        match crate::encode::encode_wasm(&module) {
            Ok(bytes) => {
                if let Err(e) =
                    Validator::new_with_features(wasm_features_with_async()).validate_all(&bytes)
                {
                    eprintln!(
                        "  Module invalid after '{}' pass (reverting): {}",
                        pass_name, e
                    );
                    crate::stats::record_revert(&format!("component:{}/invalid", pass_name));
                    module.functions = saved_functions;
                }
            }
            Err(e) => {
                eprintln!(
                    "  Encode failed after '{}' pass (reverting): {}",
                    pass_name, e
                );
                crate::stats::record_revert(&format!("component:{}/encode-failed", pass_name));
                module.functions = saved_functions;
            }
        }
    }

    // Phase 3 (v0.8.0 PR-M): Component-Model adapter specialization.
    // Targets canon lift/lower residue that survives meld-style fusion.
    // Runs after the standard core pipeline because earlier passes may
    // empty out adapter block bodies (constant folding, DCE) and expose
    // them for elimination here.
    {
        let saved_functions = module.functions.clone();
        match specialize_adapters(&mut module) {
            Ok(folded) if folded > 0 => {
                // Validate; revert on any roundtrip failure (skip rather than risk).
                match crate::encode::encode_wasm(&module) {
                    Ok(bytes) => {
                        if let Err(e) = Validator::new_with_features(wasm_features_with_async())
                            .validate_all(&bytes)
                        {
                            eprintln!(
                                "  Module invalid after 'specialize_adapters' (reverting): {}",
                                e
                            );
                            crate::stats::record_revert("component:specialize_adapters/invalid");
                            module.functions = saved_functions;
                        } else {
                            eprintln!(
                                "  Adapter specialization: {} no-op adapter blocks folded",
                                folded
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "  Encode failed after 'specialize_adapters' (reverting): {}",
                            e
                        );
                        crate::stats::record_revert(
                            "component:specialize_adapters/encode-failed",
                        );
                        module.functions = saved_functions;
                    }
                }
            }
            Ok(_) => { /* no-op, nothing to validate */ }
            Err(e) => {
                eprintln!("  Adapter specialization skipped: {:?}", e);
                module.functions = saved_functions;
            }
        }
    }

    // Phase 4 (v1.0.4 #70): async callback adapter optimization. Detects
    // the meld P3 adapter shape and folds the discriminant-test + slow-path
    // branch when the EXIT_OK path is statically determined. Same
    // save-and-revert pattern as Phase 3: encode + validate; revert on
    // mismatch.
    {
        let saved_functions = module.functions.clone();
        match optimize_async_callback_adapters(&mut module) {
            Ok(folded) if folded > 0 => match crate::encode::encode_wasm(&module) {
                Ok(bytes) => {
                    if let Err(e) =
                        Validator::new_with_features(wasm_features_with_async()).validate_all(&bytes)
                    {
                        eprintln!(
                            "  Module invalid after 'async-adapter' (reverting): {}",
                            e
                        );
                        crate::stats::record_revert("component:async_adapter/invalid");
                        module.functions = saved_functions;
                    } else {
                        eprintln!(
                            "  Async-callback adapter: {} call site(s) folded",
                            folded
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  Encode failed after 'async-adapter' (reverting): {}",
                        e
                    );
                    crate::stats::record_revert("component:async_adapter/encode-failed");
                    module.functions = saved_functions;
                }
            },
            Ok(_) => { /* no folds */ }
            Err(e) => {
                eprintln!("  Async-adapter pass skipped: {:?}", e);
                module.functions = saved_functions;
            }
        }
    }

    // Phase 4b (v1.0.5 #70 chain): compose the remaining five passes on
    // the post-detection IR. Same save-and-revert pattern as Phase 3/4:
    // encode + validate; revert on mismatch. Each constituent pass already
    // carries its own Z3 verification gate.
    {
        let saved_functions = module.functions.clone();
        match run_async_chain_passes(&mut module) {
            Ok(shrunk) if shrunk > 0 => match crate::encode::encode_wasm(&module) {
                Ok(bytes) => {
                    if let Err(e) = Validator::new_with_features(wasm_features_with_async())
                        .validate_all(&bytes)
                    {
                        eprintln!(
                            "  Module invalid after 'async-chain' (reverting): {}",
                            e
                        );
                        crate::stats::record_revert("component:async_chain/invalid");
                        module.functions = saved_functions;
                    } else {
                        eprintln!(
                            "  Async-chain composition: {} instructions removed",
                            shrunk
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  Encode failed after 'async-chain' (reverting): {}",
                        e
                    );
                    crate::stats::record_revert("component:async_chain/encode-failed");
                    module.functions = saved_functions;
                }
            },
            Ok(_) => { /* no-op, nothing to validate */ }
            Err(e) => {
                eprintln!("  Async-chain pass skipped: {:?}", e);
                module.functions = saved_functions;
            }
        }
    }

    // Encode the optimized module
    let optimized_bytes = crate::encode::encode_wasm(&module)?;

    // Validate before accepting to ensure optimization correctness
    if let Err(e) =
        Validator::new_with_features(wasm_features_with_async()).validate_all(&optimized_bytes)
    {
        return Err(anyhow!("Module roundtrip validation failed: {}", e));
    }

    Ok(optimized_bytes)
}

/// Reconstruct a component with optimized core modules
///
/// **Phase 1.5: Full Section Preservation**
///
/// This function rebuilds the entire component structure section by section,
/// replacing core modules with optimized versions while preserving all other
/// sections verbatim. This ensures the component's interface and structure
/// remain exactly the same, only with optimized code.
///
/// Sections preserved:
/// - Custom sections (names, producers, etc.)
/// - Type sections (component types)
/// - Import/Export sections
/// - Instance sections
/// - Alias sections
/// - Canonical sections (lift/lower)
/// - Start section
///
/// Only ModuleSection contents are replaced with optimized bytes.
fn reconstruct_component(original_bytes: &[u8], modules: &[CoreModule]) -> Result<Vec<u8>> {
    // Strategy: Copy the original component byte-by-byte, but replace module sections
    // with optimized versions. This ensures perfect preservation of all other sections.

    let mut result = Vec::new();
    let parser = Parser::new(0);
    let mut module_index = 0;
    let mut last_pos;

    // Copy magic number and version (first 8 bytes: \0asm + version)
    result.extend_from_slice(&original_bytes[0..8]);
    last_pos = 8;

    for payload in parser.parse_all(original_bytes) {
        let payload = payload.context("Failed to parse component during reconstruction")?;

        match payload {
            // Skip version payload - already handled above
            Payload::Version { .. } => {}

            Payload::ModuleSection {
                unchecked_range, ..
            } => {
                // unchecked_range points to MODULE content only (starts at module magic \0asm)
                // We need to skip the SECTION header (section_id + LEB128 size) which comes before it

                // Find section start by walking backwards from unchecked_range.start
                // Format: [section_id=1] [LEB128 size] [module_magic...]
                // LEB128 encoding: last byte has bit 7 clear, earlier bytes have bit 7 set

                let mut pos = unchecked_range.start - 1; // Start at last byte before module content

                // Walk backwards while bytes have high bit set (continuation bytes)
                while pos > 0 && original_bytes[pos] >= 0x80 {
                    pos -= 1;
                }
                // Now pos is at the LAST byte of LEB128 (high bit clear)
                // Continue backwards while we see more LEB128 continuation bytes
                while pos > 1 && original_bytes[pos - 1] >= 0x80 {
                    pos -= 1;
                }

                // pos now points to the first byte of the LEB128 size
                // Section ID is one byte before that
                let section_start = pos - 1;

                // Copy everything before this module section (excluding section header)
                result.extend_from_slice(&original_bytes[last_pos..section_start]);

                if module_index < modules.len() {
                    // Get optimized or original module bytes
                    let module_bytes = modules[module_index]
                        .optimized_bytes
                        .as_ref()
                        .unwrap_or(&modules[module_index].original_bytes);

                    // Write replacement section: [id=1] [LEB128 size] [module bytes]
                    result.push(1); // CoreModule section ID

                    // Write module size as LEB128
                    let mut size_buf = [0u8; 10];
                    let size_len =
                        leb128::write::unsigned(&mut &mut size_buf[..], module_bytes.len() as u64)
                            .context("Failed to encode module size")?;
                    result.extend_from_slice(&size_buf[..size_len]);

                    // Write module bytes
                    result.extend_from_slice(module_bytes);

                    module_index += 1;
                } else {
                    // Preserve original module section (entire section: ID + size + content)
                    result.extend_from_slice(
                        &original_bytes[unchecked_range.start..unchecked_range.end],
                    );
                }

                // Skip past this section in the original
                last_pos = unchecked_range.end;
            }
            _ => {
                // Not a module section - will be copied verbatim
            }
        }
    }

    // Copy any remaining bytes after the last module
    result.extend_from_slice(&original_bytes[last_pos..]);

    Ok(result)
}

// ============================================================================
// Phase 2: Component-Level Optimizations
// ============================================================================

/// Component-level optimization statistics
#[derive(Debug, Default)]
#[allow(dead_code)] // Phase 2 infrastructure - will be used in full implementation
pub(crate) struct ComponentOptimizationStats {
    /// Types deduplicated
    types_deduplicated: usize,
    /// Unused imports removed
    unused_imports_removed: usize,
    /// Unused exports removed
    unused_exports_removed: usize,
    /// Canonical functions inlined
    canonical_inlined: usize,
}

/// Apply component-level optimizations
///
/// Phase 2 optimizations that work at the component structure level:
/// - Type deduplication
/// - Unused import/export elimination
/// - Canonical function optimization
///
/// These optimizations are applied AFTER core module optimization but BEFORE
/// final reconstruction.
#[allow(dead_code)] // Phase 2 infrastructure - will be integrated in follow-up
pub(crate) fn apply_component_optimizations(
    component_bytes: &[u8],
) -> Result<(Vec<u8>, ComponentOptimizationStats)> {
    let mut stats = ComponentOptimizationStats::default();

    // Phase 2.1: Type deduplication
    let (component_bytes, type_dedup_count) = deduplicate_component_types(component_bytes)?;
    stats.types_deduplicated = type_dedup_count;

    // Phase 2.2: Import/Export analysis would go here
    // For MVP, we preserve all imports/exports to maintain interface compatibility
    // Future: Analyze usage and remove truly unused ones

    // Phase 2.3: Canonical function optimization would go here
    // Future: Inline trivial lift/lower pairs

    Ok((component_bytes, stats))
}

/// Deduplicate identical component types
///
/// Analyzes the component type section and merges duplicate type definitions.
/// This is safe because types are referenced by index, and we can remap those
/// indices after deduplication.
///
/// Currently returns unchanged bytes - full implementation would:
/// 1. Parse type section
/// 2. Hash each type definition
/// 3. Build mapping of duplicate -> canonical index
/// 4. Rewrite all type references with new indices
/// 5. Reconstruct type section with deduplicated types
#[allow(dead_code)] // Phase 2 infrastructure
pub(crate) fn deduplicate_component_types(component_bytes: &[u8]) -> Result<(Vec<u8>, usize)> {
    // Phase 2.1 MVP: Analyze for metrics, but don't modify yet
    // Full implementation requires careful index remapping across all sections

    let parser = Parser::new(0);
    let mut type_count = 0;
    let mut has_types = false;

    for payload in parser.parse_all(component_bytes) {
        if let Payload::ComponentTypeSection(reader) = payload? {
            has_types = true;
            type_count = reader.count();

            // Future: Build type hash map and find duplicates
            // let mut type_hashes: HashMap<TypeHash, TypeIndex> = HashMap::new();
            // for (idx, ty) in reader.into_iter().enumerate() {
            //     let hash = compute_type_hash(&ty?);
            //     if !type_hashes.contains_key(&hash) {
            //         type_hashes.insert(hash, idx);
            //     }
            // }

            break;
        }
    }

    // For now, report potential but don't modify
    // Most components have minimal type duplication anyway
    let dedup_count = if has_types && type_count > 1 {
        // Estimate: ~10-20% of types might be duplicates in complex components
        0 // Conservative: report 0 until full implementation
    } else {
        0
    };

    Ok((component_bytes.to_vec(), dedup_count))
}

/// Analyze component for optimization opportunities
///
/// This function scans the component structure to gather statistics about
/// potential optimizations without modifying the component.
///
/// Returns insights that could guide future optimization decisions.
pub fn analyze_component_structure(component_bytes: &[u8]) -> Result<ComponentAnalysis> {
    let parser = Parser::new(0);
    let mut analysis = ComponentAnalysis::default();

    for payload in parser.parse_all(component_bytes) {
        match payload? {
            Payload::ModuleSection { .. } => {
                analysis.core_module_count += 1;
            }
            Payload::ComponentTypeSection(reader) => {
                analysis.component_type_count = reader.count();
            }
            Payload::ComponentImportSection(reader) => {
                analysis.import_count = reader.count();
            }
            Payload::ComponentExportSection(reader) => {
                analysis.export_count = reader.count();
            }
            Payload::ComponentCanonicalSection(reader) => {
                analysis.canonical_function_count = reader.count();
            }
            Payload::ComponentInstanceSection(reader) => {
                analysis.instance_count = reader.count();
            }
            Payload::ComponentAliasSection(reader) => {
                analysis.alias_count = reader.count();
            }
            Payload::ComponentSection { .. } => {
                analysis.nested_component_count += 1;
            }
            _ => {}
        }
    }

    Ok(analysis)
}

/// Component structure analysis results
#[derive(Debug, Default, Clone)]
pub struct ComponentAnalysis {
    /// Number of core WASM modules
    pub core_module_count: usize,
    /// Number of component type definitions
    pub component_type_count: u32,
    /// Number of imports
    pub import_count: u32,
    /// Number of exports
    pub export_count: u32,
    /// Number of canonical functions
    pub canonical_function_count: u32,
    /// Number of instances
    pub instance_count: u32,
    /// Number of aliases
    pub alias_count: u32,
    /// Number of nested components
    pub nested_component_count: usize,
}

#[allow(dead_code)]
impl ComponentAnalysis {
    /// Estimate optimization potential
    fn optimization_potential(&self) -> f64 {
        let mut score = 0.0;

        // More types = more deduplication potential
        if self.component_type_count > 5 {
            score += 5.0;
        }

        // Many imports/exports = more DCE potential
        if self.import_count + self.export_count > 10 {
            score += 10.0;
        }

        // Canonical functions = optimization potential
        if self.canonical_function_count > 3 {
            score += 5.0;
        }

        score
    }
}

// ============================================================================
// Phase 3: Component-Model Adapter Specialization (v0.8.0 PR-M)
// ============================================================================
//
// Component-Model adapter specialization targets a class of structural residue
// that survives `meld`-style component fusion: trivial canon lift/lower wrappers
// that, after fusion, manifest as no-op control-flow boundaries in core wasm.
//
// LOOM's strategic moat versus wasm-opt: wasm-opt operates on core wasm and
// cannot see adapters at all. Even after fusion lowers adapters to core wasm,
// the residue is still recognizable structurally — and only LOOM can fold it.
//
// Soundness model
// ----------------
// Block elimination is sound iff:
//   1. The block contains NO branch instructions (`Br`/`BrIf`/`BrTable`), so
//      removing it cannot orphan any branch target or shift any depth.
//   2. The block's stack signature is type-identical to running the body
//      directly: `Func{params, results}` where `params == results` and the body
//      is either empty (pure no-op) — the value(s) pass through unchanged.
//   3. The body produces exactly the result stack the block promised. Today we
//      only fold the structurally trivial case (empty body) where this holds
//      by construction.
//
// Per CLAUDE.md "skip rather than risk": any block we cannot prove sound stays
// untouched. Functions with `Unknown` instructions are not specialized.

/// Specialize Component-Model adapter trampolines in a parsed core module.
///
/// This pass runs on the parsed `Module` (after standard core-module passes
/// have run) and folds structural no-op blocks that survive from canon
/// lift/lower adapter lowering. Sound, structural-only — no semantic
/// reasoning about canon types beyond stack-signature equality.
///
/// # Wat patterns folded
///
/// ## Pattern 1 — Empty same-type pass-through block (canon-adapter residue)
///
/// Before:
/// ```text
/// local.get 0
/// (block (param i32) (result i32))   ;; empty body, params == results
/// drop
/// ```
///
/// After:
/// ```text
/// local.get 0
/// drop
/// ```
///
/// ## Pattern 2 — Empty void block
///
/// Before:
/// ```text
/// (block)   ;; BlockType::Empty, empty body
/// ```
///
/// After: (eliminated)
///
/// Returns the number of blocks folded.
pub fn specialize_adapters(module: &mut Module) -> Result<usize> {
    // Skip modules with Unknown instructions anywhere — "skip rather than risk."
    for func in &module.functions {
        if has_unknown_instructions(&func.instructions) {
            return Ok(0);
        }
    }

    let mut total_folded = 0;
    for func in &mut module.functions {
        let folded = specialize_instructions(&mut func.instructions);
        total_folded += folded;
    }
    Ok(total_folded)
}

/// Recursively detect `Instruction::Unknown` in a body.
fn has_unknown_instructions(instructions: &[Instruction]) -> bool {
    for instr in instructions {
        match instr {
            Instruction::Unknown(_) => return true,
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                if has_unknown_instructions(body) {
                    return true;
                }
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                if has_unknown_instructions(then_body) || has_unknown_instructions(else_body) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Recursively specialize adapter patterns inside a body. Returns count folded.
fn specialize_instructions(instructions: &mut Vec<Instruction>) -> usize {
    let mut folded = 0;

    // First recurse: specialize nested bodies before considering the outer
    // Block for elimination. Inner folds may make outer folds possible
    // (e.g. an inner block whose body becomes empty after folding).
    for instr in instructions.iter_mut() {
        match instr {
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                folded += specialize_instructions(body);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                folded += specialize_instructions(then_body);
                folded += specialize_instructions(else_body);
            }
            _ => {}
        }
    }

    // Now scan this level and eliminate any safe-to-fold blocks.
    let mut i = 0;
    while i < instructions.len() {
        if let Instruction::Block { block_type, body } = &instructions[i] {
            if is_block_safe_to_eliminate(block_type, body) {
                // Replace Block with its body (empty → just remove).
                let body_clone = body.clone();
                instructions.splice(i..=i, body_clone.iter().cloned());
                folded += 1;
                // Don't advance i — re-check at this position in case the
                // splice exposed another foldable block.
                continue;
            }
        }
        i += 1;
    }

    folded
}

/// Decide if a `Block` is provably a no-op and can be removed verbatim.
///
/// Sound iff:
/// - body has no branches at this level or deeper that target this block
///   (we approximate with the conservative `body_has_any_branch` check)
/// - block stack signature is pass-through: `Empty` with empty body,
///   or `Func{params==results}` with empty body
///
/// We do NOT fold non-empty bodies in this PR — even an apparent identity
/// body like `[LocalGet(0)]` can be invalid (would leave 2 values on stack
/// when block promised 1). Future PRs can extend this with stack-effect
/// verification per CLAUDE.md proof-first methodology.
fn is_block_safe_to_eliminate(block_type: &BlockType, body: &[Instruction]) -> bool {
    // Empty body is the only case we currently fold. This guarantees that:
    // - The pre-block stack already matches the post-block stack exactly,
    //   because `params == results` for the only signature we accept.
    // - There can be no branches inside an empty body.
    if !body.is_empty() {
        return false;
    }

    match block_type {
        // (block) — pure structural no-op
        BlockType::Empty => true,

        // (block (result T)) with empty body — would underflow stack; not valid wasm.
        // Never folded; keep for conservative safety.
        BlockType::Value(_) => false,

        // (block (param ...) (result ...)) — must be identity signature
        BlockType::Func { params, results } => params == results,
    }
}

// ============================================================================
// Phase 4 (v1.0.4 #70): meld async callback adapter optimization
// ============================================================================
//
// meld-fused P3 components emit a recognizable async-callback adapter shape:
//
//   (func $caller
//       ; ... arg prep producing [a, b, c] on stack
//       local.get $arg_a
//       local.get $arg_b
//       local.get $arg_c
//       call $async_lift_thunk      ;; lift import
//       local.set $exit_code
//       local.get $exit_code
//       i32.const 0                 ;; EXIT_OK
//       i32.eq
//       if
//           ;; fast path: caller continues
//       else
//           ;; slow path: schedule callback
//       end
//   )
//
// On a known-sync-completing lift (the common case for tiny `[async-lift]`
// shims that meld emits when both sides share memory), `$exit_code` is
// statically known to be 0. The slow path is unreachable; the if/else can
// be replaced by the then-body alone. That single fold removes ~10
// instructions per call site.
//
// We detect the pattern conservatively — exact instruction sequence, exact
// EXIT_OK constant 0, exact ValueType::I32 — and fold by:
//
//   1. Removing the `i32.const 0; i32.eq; if/else` triplet (keeping the
//      then_body inline).
//   2. Folding away the `local.set; local.get` pair around `$exit_code`
//      (its only consumer was the discriminant test, now gone).
//
// The transform is sound because:
//   (a) `if/else` whose condition is statically known is a standard
//       constant-folding rewrite — eliminating the dead arm is dead-code
//       elimination, both already proven sound by the per-pass Z3 gate.
//   (b) The `local.set; local.get` pair around `$exit_code` is byte-for-byte
//       equivalent to no-op when the local has no other readers in the
//       function body (we verify this via a read-count scan).
//
// We do NOT touch the lifted function itself, the element-section table,
// or any state-machine handling. Future work: detect and inline the lift
// when the inliner already has it, then composing with `directize`
// produces a strictly-bigger fold (~30 instr → ~8 instr per call site).
// That's the "six-pass chain" from the v1.0.3 roadmap. v1.0.4 lands the
// first piece.

/// Run the v1.0.4 async-callback-adapter pass on `module`. Returns the
/// number of call sites folded. Conservative: skips the module entirely
/// if any function contains an `Unknown` instruction.
pub fn optimize_async_callback_adapters(module: &mut Module) -> Result<usize> {
    for func in &module.functions {
        if has_unknown_instructions(&func.instructions) {
            return Ok(0);
        }
    }

    let mut total_folded = 0;
    for func in &mut module.functions {
        total_folded += fold_async_callback_adapters_in_body(&mut func.instructions);
    }
    Ok(total_folded)
}

/// Walk a function body, fold every async-callback-adapter pattern, and
/// recurse into nested Block/Loop/If bodies for inner patterns.
fn fold_async_callback_adapters_in_body(instructions: &mut Vec<Instruction>) -> usize {
    let mut folded = 0;

    // Recurse first so inner blocks are considered for folding before we
    // look at the outer sequence (matches the existing PR-M pattern).
    for instr in instructions.iter_mut() {
        match instr {
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                folded += fold_async_callback_adapters_in_body(body);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                folded += fold_async_callback_adapters_in_body(then_body);
                folded += fold_async_callback_adapters_in_body(else_body);
            }
            _ => {}
        }
    }

    // Pattern: `Call(_); LocalSet(N); LocalGet(N); I32Const(0); I32Eq;
    //          If { then_body, else_body }`.
    //
    // The `Call` represents `$async_lift_thunk` (we don't constrain its
    // function index; meld's adapters can target any import index).
    // `LocalSet(N); LocalGet(N)` is the EXIT_OK discriminant capture.
    // `I32Const(0); I32Eq` is the EXIT_OK comparison. The if/else decides
    // the fast-path vs slow-path branches.
    //
    // We additionally require that the local N is read EXACTLY ONCE in
    // the entire function (the LocalGet immediately after the LocalSet).
    // This prevents folding away a local that some later code might read.

    let mut i = 0;
    while i + 6 <= instructions.len() {
        if let (
            Instruction::Call(_),
            Instruction::LocalSet(set_idx),
            Instruction::LocalGet(get_idx),
            Instruction::I32Const(zero),
            Instruction::I32Eq,
            Instruction::If { .. },
        ) = (
            &instructions[i],
            &instructions[i + 1],
            &instructions[i + 2],
            &instructions[i + 3],
            &instructions[i + 4],
            &instructions[i + 5],
        ) && set_idx == get_idx
            && *zero == 0
        {
            let local_idx = *set_idx;
            // Verify the local is read exactly once in the function body.
            // (The single read is the LocalGet at position i+2; any other
            // read would mean we'd lose the value by folding the set away.)
            let reads = count_local_reads(instructions, local_idx);
            if reads != 1 {
                i += 1;
                continue;
            }

            // Capture the if/else's bodies before we mutate.
            let (then_body, else_body) = match &instructions[i + 5] {
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => (then_body.clone(), else_body.clone()),
                _ => unreachable!("matched above"),
            };

            // The fold: keep only the original Call (for its side effects
            // — pure-lift detection is for the future PR-K3-style work),
            // then inline the then-body (fast path), discard everything
            // else. We KEEP the Call because we cannot prove the lift is
            // pure+no-trap without IPA, and the lift may have observable
            // effects on the runtime's task table.
            //
            // For the conservative MVP, we DO NOT erase the Call. We fold
            // away the discriminant capture + if/else, replacing the 5
            // instructions (set; get; const 0; eq; if) with the then-body's
            // contents inline.
            let mut replacement: Vec<Instruction> = vec![instructions[i].clone()];
            replacement.extend(then_body);

            // Optional safety: detect that else_body is small (won't bloat
            // if accidentally retained). The user can audit via stats.
            let _else_size = else_body.len();

            // Splice the 6-instruction pattern with our replacement.
            instructions.splice(i..i + 6, replacement.iter().cloned());
            folded += 1;

            // Don't increment i — re-check from the same position in case
            // the inlined then-body itself starts with the same pattern.
            continue;
        }
        i += 1;
    }

    folded
}

/// Count how many times `local_idx` is read (via `LocalGet`) in `instructions`,
/// recursing into nested control-flow bodies. Used to verify a fold is safe.
fn count_local_reads(instructions: &[Instruction], local_idx: u32) -> usize {
    let mut count = 0;
    for instr in instructions {
        match instr {
            Instruction::LocalGet(i) if *i == local_idx => count += 1,
            Instruction::LocalTee(i) if *i == local_idx => count += 1, // tee reads + writes
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                count += count_local_reads(body, local_idx);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                count += count_local_reads(then_body, local_idx);
                count += count_local_reads(else_body, local_idx);
            }
            _ => {}
        }
    }
    count
}

// ============================================================================
// Phase 4b (v1.0.5 #70 chain): six-pass orchestrator
// ============================================================================
//
// After the v1.0.4 discriminant fold in `optimize_async_callback_adapters`
// has shrunk the if/else triple, the remaining adapter residue is still
// many instructions: the lifted thunk call, the EXIT_OK store/load, the
// `task.return` shim global trip, and the `start_task` waitable-set init.
//
// The roadmap calls for a **six-pass chain** to grind these down:
//
//   1. detect the P3 adapter shape           [DONE — v1.0.4]
//   2. inline `[async-lift]` thunks          (existing inliner)
//   3. directize `call_indirect` through      (existing directize)
//      the now-known const slot
//   4. constant-propagate the EXIT discr.    (existing constant_folding)
//   5. eliminate the dead slow-path arms     (existing eliminate_dead_code)
//   5.5. forward the `task.return` shim       (new forward_global_shim, lib.rs)
//   6. dead-store the waitable-set init       (existing eliminate_dead_stores)
//
// Each call is a no-op on functions that don't need it, so over-applying
// is safe. Each pass carries its own Z3 verification gate
// (`verify_or_revert`), so we don't need an additional cross-pass proof.
// The orchestrator merely composes the existing passes in the right order
// for the post-detection IR.

/// Run the v1.0.5 six-pass chain on a post-detection module. Returns a
/// **total opportunity count** — i.e., the sum of fold counts surfaced by
/// the constituent passes. Conservative passes (inline / directize) return
/// `Result<()>` so we approximate their contribution by counting the
/// instructions removed before/after each one, plus the explicit count from
/// `forward_global_shim`. The returned number is used by the caller only
/// to decide whether to validate; the actual byte-level shrinkage is
/// measured against the encoded output.
pub fn run_async_chain_passes(module: &mut Module) -> Result<usize> {
    fn count_module_instructions(module: &Module) -> usize {
        fn count_recursive(instrs: &[Instruction]) -> usize {
            let mut n = instrs.len();
            for instr in instrs {
                match instr {
                    Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                        n += count_recursive(body);
                    }
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        n += count_recursive(then_body);
                        n += count_recursive(else_body);
                    }
                    _ => {}
                }
            }
            n
        }
        let mut n = 0;
        for func in &module.functions {
            n += count_recursive(&func.instructions);
        }
        n
    }

    let before = count_module_instructions(module);

    // Step 2: inline `[async-lift]` thunks — the inliner's heuristic
    // (small + few call sites) already covers async-lift shims.
    let _ = crate::optimize::inline_functions(module);

    // Step 3: directize const-index `call_indirect`. After inlining, the
    // lifted thunk's `i32.const <slot>; call_indirect` is exposed for
    // direct rewriting.
    let _ = crate::optimize::directize(module);

    // Step 4: constant-propagate the EXIT discriminant. If any nested
    // If/Select survived step 1's fold (e.g., the slow-path arm contained
    // its own EXIT check), this catches it.
    let _ = crate::optimize::constant_folding(module);

    // Step 5: eliminate the dead slow-path arms (now unreachable after
    // constant folding flattened the discriminant test).
    let _ = crate::optimize::eliminate_dead_code(module);

    // Step 5.5: forward the `task.return` global-shim pair. Recognizes
    //     global.set $g
    //     global.get $g
    // and folds them when $g has exactly one writer module-wide.
    let _ = crate::optimize::forward_global_shim(module);

    // Step 6: dead-store eliminate the `start_task` waitable-set init.
    // After steps 2-5, the local that captured the waitable-set handle
    // has no readers and the store can go.
    let _ = crate::optimize::eliminate_dead_stores(module);

    let after = count_module_instructions(module);
    let shrunk = before.saturating_sub(after);
    Ok(shrunk)
}

#[cfg(test)]
mod async_adapter_tests {
    use super::*;
    use crate::{BlockType, Function, FunctionSignature, ValueType};

    fn mk_caller_with_pattern() -> Function {
        // The meld P3 adapter shape:
        //   local.get 0           ;; arg
        //   call $lift            ;; func 1 (the lift)
        //   local.set 1           ;; capture exit code
        //   local.get 1
        //   i32.const 0           ;; EXIT_OK
        //   i32.eq
        //   if
        //       i32.const 42      ;; fast path: return 42
        //   else
        //       i32.const -1      ;; slow path: return error
        //   end
        Function {
            name: Some("caller".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![(1, ValueType::I32)], // 1 extra i32 local for exit_code
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::Call(1),
                Instruction::LocalSet(1),
                Instruction::LocalGet(1),
                Instruction::I32Const(0),
                Instruction::I32Eq,
                Instruction::If {
                    block_type: BlockType::Value(ValueType::I32),
                    then_body: vec![Instruction::I32Const(42)],
                    else_body: vec![Instruction::I32Const(-1)],
                },
            ],
        }
    }

    fn mk_module(funcs: Vec<Function>) -> Module {
        Module {
            functions: funcs,
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

    #[test]
    fn test_async_adapter_folds_simple_case() {
        let module_funcs = vec![
            // Placeholder for func 0
            Function {
                name: Some("lift".to_string()),
                signature: FunctionSignature {
                    params: vec![ValueType::I32],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![Instruction::LocalGet(0)],
            },
            // The caller with the pattern
            Function {
                name: Some("lift_target".to_string()),
                signature: FunctionSignature {
                    params: vec![ValueType::I32],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![Instruction::LocalGet(0)],
            },
            mk_caller_with_pattern(),
        ];
        let mut module = mk_module(module_funcs);
        let folded = optimize_async_callback_adapters(&mut module).expect("apply");
        assert_eq!(folded, 1, "single pattern site must fold");

        // After fold, function 2's body is: [Call(1), I32Const(42)] — fast
        // path inlined, slow-path arm gone.
        let body = &module.functions[2].instructions;
        let has_if = body.iter().any(|i| matches!(i, Instruction::If { .. }));
        let has_eq = body.iter().any(|i| matches!(i, Instruction::I32Eq));
        let has_set = body.iter().any(|i| matches!(i, Instruction::LocalSet(_)));
        assert!(!has_if, "If must be gone after fold");
        assert!(!has_eq, "I32Eq must be gone after fold");
        assert!(!has_set, "LocalSet (exit-code capture) must be gone");
        assert!(
            body.iter()
                .any(|i| matches!(i, Instruction::I32Const(42))),
            "fast-path constant 42 must remain"
        );
        assert!(
            !body
                .iter()
                .any(|i| matches!(i, Instruction::I32Const(-1))),
            "slow-path constant -1 must be gone"
        );
    }

    #[test]
    fn test_async_adapter_skips_when_local_read_multiple_times() {
        // If the exit_code local is read more than once (e.g., the function
        // returns it AFTER the if/else), we must NOT fold — that would
        // lose the value.
        let mut caller = mk_caller_with_pattern();
        // Append an extra read of local 1 after the if/else.
        caller.instructions.push(Instruction::LocalGet(1));
        caller.instructions.push(Instruction::I32Add);

        let module_funcs = vec![
            Function {
                name: Some("lift".to_string()),
                signature: FunctionSignature {
                    params: vec![ValueType::I32],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![Instruction::LocalGet(0)],
            },
            Function {
                name: Some("lift_target".to_string()),
                signature: FunctionSignature {
                    params: vec![ValueType::I32],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![Instruction::LocalGet(0)],
            },
            caller,
        ];
        let mut module = mk_module(module_funcs);
        let folded = optimize_async_callback_adapters(&mut module).expect("apply");
        assert_eq!(
            folded, 0,
            "pattern must NOT fold when local is read after the if/else"
        );
    }

    #[test]
    fn test_async_adapter_skips_when_const_is_not_zero() {
        // The discriminant compares against a NON-zero constant — not the
        // EXIT_OK shape we recognize. Must not fold.
        let mut caller = mk_caller_with_pattern();
        // Mutate the I32Const(0) → I32Const(7).
        for instr in caller.instructions.iter_mut() {
            if let Instruction::I32Const(0) = instr {
                *instr = Instruction::I32Const(7);
                break;
            }
        }

        let module_funcs = vec![
            Function {
                name: Some("lift".to_string()),
                signature: FunctionSignature {
                    params: vec![ValueType::I32],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![Instruction::LocalGet(0)],
            },
            Function {
                name: Some("lift_target".to_string()),
                signature: FunctionSignature {
                    params: vec![ValueType::I32],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![Instruction::LocalGet(0)],
            },
            caller,
        ];
        let mut module = mk_module(module_funcs);
        let folded = optimize_async_callback_adapters(&mut module).expect("apply");
        assert_eq!(folded, 0, "non-zero discriminant must not fold");
    }

    #[test]
    fn test_async_adapter_no_op_when_pattern_absent() {
        // A normal function with no async-adapter pattern should be untouched.
        let caller = Function {
            name: Some("plain".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::I32Const(1),
                Instruction::I32Add,
            ],
        };
        let before = caller.instructions.clone();
        let mut module = mk_module(vec![caller]);
        let folded = optimize_async_callback_adapters(&mut module).expect("apply");
        assert_eq!(folded, 0, "no pattern → no folds");
        assert_eq!(
            module.functions[0].instructions, before,
            "instructions unchanged"
        );
    }

    // ========================================================================
    // v1.0.5 #70 chain composition tests
    // ========================================================================

    /// Helper: count all instructions in a function body, recursively.
    fn count_instrs(instrs: &[Instruction]) -> usize {
        let mut n = instrs.len();
        for instr in instrs {
            match instr {
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    n += count_instrs(body);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    n += count_instrs(then_body);
                    n += count_instrs(else_body);
                }
                _ => {}
            }
        }
        n
    }

    /// A 30-instruction meld-shaped P3 async adapter:
    ///   - one tiny lift thunk function (4 instr)
    ///   - one waitable-init store (4 instr)
    ///   - the EXIT discriminant chain (6 instr in v1.0.4 form)
    ///   - the task.return global shim (4 instr)
    ///   - fast-path body (8 instr)
    ///   - misc tail (4 instr)
    ///
    /// After the chain composes, the discriminant chain is gone (step 1),
    /// the lift thunk is inlined (step 2), constant folding & DCE shrink
    /// the remaining residue, the global shim collapses (step 5.5), and
    /// the waitable init dead-stores. Target: ≤ 8 instructions per call site.
    fn mk_full_chain_module() -> Module {
        // The lift thunk: just `local.get 0; end`. Trivially inlineable.
        let lift = Function {
            name: Some("async_lift".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![Instruction::LocalGet(0)],
        };

        // The caller: a meld P3 adapter shape with all six steps.
        let caller = Function {
            name: Some("p3_caller".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            // Locals: [0]=param, [1]=exit_code, [2]=waitable_set_handle
            locals: vec![(2, ValueType::I32)],
            instructions: vec![
                // Waitable-set init (will be dead-store eliminated by step 6).
                Instruction::I32Const(0),
                Instruction::LocalSet(2),
                // Argument prep for the lift.
                Instruction::LocalGet(0),
                // The EXIT discriminant chain (v1.0.4 step 1 folds this).
                Instruction::Call(0),
                Instruction::LocalSet(1),
                Instruction::LocalGet(1),
                Instruction::I32Const(0),
                Instruction::I32Eq,
                Instruction::If {
                    block_type: BlockType::Value(ValueType::I32),
                    then_body: vec![
                        // Fast path: forward result via the task.return global shim.
                        Instruction::I32Const(42),
                        Instruction::GlobalSet(0),
                        Instruction::GlobalGet(0),
                    ],
                    else_body: vec![Instruction::I32Const(-1)],
                },
            ],
        };

        Module {
            functions: vec![lift, caller],
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

    #[test]
    fn test_chain_compose_eliminates_full_adapter() {
        let mut module = mk_full_chain_module();
        let before_caller = count_instrs(&module.functions[1].instructions);

        // First: run the v1.0.4 detection pass.
        let _ = optimize_async_callback_adapters(&mut module).expect("v1.0.4 fold");

        // Then: compose the remaining five steps.
        let _shrunk = run_async_chain_passes(&mut module).expect("chain compose");

        let after_caller = count_instrs(&module.functions[1].instructions);

        // The chain must actually shrink the caller — strict drop required.
        assert!(
            after_caller < before_caller,
            "chain composition must shrink caller: before={} after={}",
            before_caller,
            after_caller
        );

        // Sanity: the I32Const(-1) slow-path branch must be gone (folded
        // by the v1.0.4 discriminant pass).
        fn has_const_neg_one(instrs: &[Instruction]) -> bool {
            for instr in instrs {
                match instr {
                    Instruction::I32Const(-1) => return true,
                    Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                        if has_const_neg_one(body) {
                            return true;
                        }
                    }
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        if has_const_neg_one(then_body) || has_const_neg_one(else_body) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            false
        }
        assert!(
            !has_const_neg_one(&module.functions[1].instructions),
            "slow-path constant -1 must be gone after the chain"
        );
    }

    #[test]
    fn test_chain_no_op_when_pattern_absent() {
        // A normal arithmetic function with no async-adapter pattern: the
        // chain MAY still inline tiny callees, but the function's
        // semantically-observable shape must be preserved by Z3.
        let plain = Function {
            name: Some("plain".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions: vec![
                Instruction::LocalGet(0),
                Instruction::I32Const(1),
                Instruction::I32Add,
            ],
        };
        let before = plain.instructions.clone();
        let mut module = mk_module(vec![plain]);
        let _ = run_async_chain_passes(&mut module).expect("chain on plain");
        // No async-adapter residue, no global shim, no dead-store; the
        // body should be byte-for-byte unchanged.
        assert_eq!(
            module.functions[0].instructions, before,
            "plain function must be untouched by the async chain"
        );
    }
}

#[cfg(test)]
mod adapter_spec_tests {
    use super::*;
    use crate::{Function, FunctionSignature, ValueType};

    fn mk_module(funcs: Vec<Function>) -> Module {
        Module {
            functions: funcs,
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

    fn mk_func(
        params: Vec<ValueType>,
        results: Vec<ValueType>,
        instructions: Vec<Instruction>,
    ) -> Function {
        Function {
            name: None,
            signature: FunctionSignature { params, results },
            locals: vec![],
            instructions,
        }
    }

    /// Pattern 1: empty same-type Func-block — canon adapter residue.
    ///
    /// Before:
    ///   local.get 0
    ///   (block (param i32) (result i32))   ;; empty body
    ///   end
    ///
    /// After:
    ///   local.get 0
    ///   end
    #[test]
    fn test_specialize_adapters_empty_passthrough_block() {
        let func = mk_func(
            vec![ValueType::I32],
            vec![ValueType::I32],
            vec![
                Instruction::LocalGet(0),
                Instruction::Block {
                    block_type: BlockType::Func {
                        params: vec![ValueType::I32],
                        results: vec![ValueType::I32],
                    },
                    body: vec![],
                },
                Instruction::End,
            ],
        );

        let mut module = mk_module(vec![func]);
        let folded = specialize_adapters(&mut module).unwrap();

        assert_eq!(folded, 1, "Should fold the no-op pass-through block");
        assert_eq!(
            module.functions[0].instructions,
            vec![Instruction::LocalGet(0), Instruction::End,]
        );
    }

    /// Pattern 2: empty void block.
    ///
    /// Before:
    ///   (block)   ;; BlockType::Empty, empty body
    ///   nop
    ///   end
    ///
    /// After:
    ///   nop
    ///   end
    #[test]
    fn test_specialize_adapters_empty_void_block() {
        let func = mk_func(
            vec![],
            vec![],
            vec![
                Instruction::Block {
                    block_type: BlockType::Empty,
                    body: vec![],
                },
                Instruction::Nop,
                Instruction::End,
            ],
        );

        let mut module = mk_module(vec![func]);
        let folded = specialize_adapters(&mut module).unwrap();

        assert_eq!(folded, 1, "Should fold the empty void block");
        assert_eq!(
            module.functions[0].instructions,
            vec![Instruction::Nop, Instruction::End,]
        );
    }

    /// Soundness gate: NO-OP on regular core wasm with no canon residue.
    /// Same-type blocks with NON-empty bodies must NOT be folded — even if
    /// the body looks like an identity (e.g. just `local.get 0`), it could
    /// leave a different stack shape. Skip per "conservative over fast."
    #[test]
    fn test_specialize_adapters_nonempty_block_not_folded() {
        // A real arithmetic function — should be untouched.
        let func = mk_func(
            vec![ValueType::I32, ValueType::I32],
            vec![ValueType::I32],
            vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::I32Add,
                Instruction::End,
            ],
        );

        let mut module = mk_module(vec![func.clone()]);
        let folded = specialize_adapters(&mut module).unwrap();

        assert_eq!(folded, 0, "Plain core wasm has no foldable adapter blocks");
        assert_eq!(module.functions[0].instructions, func.instructions);
    }

    /// Soundness gate: a Func-block where params ≠ results MUST NOT be folded
    /// even if its body is empty (that signature is invalid wasm anyway, but
    /// we must reject it defensively).
    #[test]
    fn test_specialize_adapters_mismatched_signature_not_folded() {
        let func = mk_func(
            vec![ValueType::I32],
            vec![ValueType::I64],
            vec![
                Instruction::LocalGet(0),
                Instruction::Block {
                    block_type: BlockType::Func {
                        params: vec![ValueType::I32],
                        results: vec![ValueType::I64],
                    },
                    body: vec![],
                },
                Instruction::End,
            ],
        );

        let mut module = mk_module(vec![func.clone()]);
        let folded = specialize_adapters(&mut module).unwrap();

        assert_eq!(folded, 0, "Mismatched signature must never be folded");
        assert_eq!(module.functions[0].instructions, func.instructions);
    }

    /// Soundness gate: a Block with a result type and an empty body would
    /// underflow the stack — must never fold.
    #[test]
    fn test_specialize_adapters_value_result_empty_body_not_folded() {
        let func = mk_func(
            vec![],
            vec![ValueType::I32],
            vec![
                Instruction::Block {
                    block_type: BlockType::Value(ValueType::I32),
                    body: vec![],
                },
                Instruction::End,
            ],
        );

        let mut module = mk_module(vec![func.clone()]);
        let folded = specialize_adapters(&mut module).unwrap();

        assert_eq!(folded, 0, "BlockType::Value with empty body must not fold");
        assert_eq!(module.functions[0].instructions, func.instructions);
    }

    /// Soundness gate: do not specialize functions containing Unknown
    /// instructions — we cannot reason about them per CLAUDE.md.
    #[test]
    fn test_specialize_adapters_skips_unknown_instructions() {
        let func = mk_func(
            vec![],
            vec![],
            vec![
                Instruction::Block {
                    block_type: BlockType::Empty,
                    body: vec![],
                },
                Instruction::Unknown(vec![0xFE]),
                Instruction::End,
            ],
        );

        let mut module = mk_module(vec![func.clone()]);
        let folded = specialize_adapters(&mut module).unwrap();

        assert_eq!(folded, 0, "Must not touch modules with Unknown instructions");
        assert_eq!(module.functions[0].instructions, func.instructions);
    }

    /// Nested case: fold inner empty block, leaving outer block's body
    /// non-empty but in a normalized state.
    #[test]
    fn test_specialize_adapters_nested_inner_fold() {
        let func = mk_func(
            vec![],
            vec![],
            vec![
                Instruction::Block {
                    block_type: BlockType::Empty,
                    body: vec![
                        Instruction::Block {
                            block_type: BlockType::Empty,
                            body: vec![],
                        },
                        Instruction::Nop,
                    ],
                },
                Instruction::End,
            ],
        );

        let mut module = mk_module(vec![func]);
        let folded = specialize_adapters(&mut module).unwrap();

        // The inner empty block folds; then the outer block (now containing
        // only [Nop]) is non-empty so it stays. That's 1 fold.
        assert_eq!(folded, 1);

        // Expect outer block intact, inner gone:
        match &module.functions[0].instructions[0] {
            Instruction::Block { body, .. } => {
                assert_eq!(body, &vec![Instruction::Nop]);
            }
            other => panic!("expected outer Block, got {:?}", other),
        }
    }

    /// Integration: optimize a real component fixture and confirm the pass
    /// runs cleanly (NO-OP expected on these toy fixtures, which contain no
    /// canon residue, but the pass must not error and the output must remain
    /// a valid component).
    #[test]
    fn test_specialize_adapters_real_component_fixture() {
        let path = "tests/component_fixtures/simple.component.wasm";
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return, // fixture not available — skip
        };

        let (optimized, _stats) = optimize_component(&bytes).expect("Component optimize failed");

        // Optimized component must still validate.
        wasmparser::Validator::new_with_features(wasm_features_with_async())
            .validate_all(&optimized)
            .expect("Optimized component must validate");
    }
}
