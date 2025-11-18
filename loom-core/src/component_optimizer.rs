//! Component Model Optimization
//!
//! This module provides world-class WebAssembly Component Model optimization.
//! LOOM is the first optimizer to support the Component Model specification.
//!
//! ## Architecture
//!
//! 1. **Parse**: Parse component and extract core modules
//! 2. **Optimize**: Apply LOOM's 12-phase pipeline to each core module
//! 3. **Reconstruct**: Rebuild component with optimized modules, preserving all sections
//! 4. **Validate**: Ensure correctness with wasmparser validation
//!
//! ## Component Structure
//!
//! Components contain:
//! - Core modules (embedded WASM modules)
//! - Type definitions (component-level types)
//! - Instances (module/component instantiations)
//! - Canonical functions (lift/lower operations)
//! - Imports/Exports (component interface)
//! - Aliases (export projections)
//!
//! ## Index Spaces
//!
//! Components maintain separate index spaces for different entities.
//! During reconstruction, we must remap indices carefully:
//! - Module indices (core modules)
//! - Instance indices (instantiations)
//! - Function indices (both core and component)
//! - Type indices (both core and component)
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

use anyhow::{anyhow, Context, Result};
use wasm_encoder::{Component, ComponentSectionId, RawSection};
use wasmparser::{Encoding, Parser, Payload};

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
    for core_module in &mut core_modules {
        match optimize_core_module(&core_module.original_bytes) {
            Ok(optimized_bytes) => {
                core_module.optimized_bytes = Some(optimized_bytes);
                optimized_count += 1;
            }
            Err(e) => {
                eprintln!("Warning: Failed to optimize module: {}", e);
                // Keep original bytes
            }
        }
    }

    // Step 3: Reconstruct component with optimized modules
    let optimized_component = reconstruct_component(component_bytes, &core_modules)?;

    // Step 4: Validate
    wasmparser::validate(&optimized_component).context("Optimized component validation failed")?;

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
fn optimize_core_module(module_bytes: &[u8]) -> Result<Vec<u8>> {
    // Parse the module
    let mut module = crate::parse::parse_wasm(module_bytes)?;

    // Optimize with LOOM's 12-phase pipeline
    crate::optimize::optimize_module(&mut module)?;

    // Encode
    let optimized_bytes = crate::encode::encode_wasm(&module)?;

    // Validate before accepting
    wasmparser::validate(&optimized_bytes).context("Optimized module validation failed")?;

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
    let mut component = Component::new();
    let parser = Parser::new(0);
    let mut module_index = 0;

    // Iterate through all sections and reconstruct them
    for payload in parser.parse_all(original_bytes) {
        let payload = payload.context("Failed to parse component during reconstruction")?;

        match payload {
            // Version/header handled automatically
            Payload::Version { .. } => {}

            // Replace module sections with optimized modules
            Payload::ModuleSection {
                unchecked_range, ..
            } => {
                if module_index < modules.len() {
                    // Get optimized or original module bytes
                    let module_bytes = modules[module_index]
                        .optimized_bytes
                        .as_ref()
                        .unwrap_or(&modules[module_index].original_bytes);

                    // Add as raw section
                    let section = RawSection {
                        id: ComponentSectionId::CoreModule.into(),
                        data: module_bytes,
                    };
                    component.section(&section);

                    module_index += 1;
                } else {
                    // Shouldn't happen, but preserve original if it does
                    let original_module =
                        &original_bytes[unchecked_range.start..unchecked_range.end];
                    let section = RawSection {
                        id: ComponentSectionId::CoreModule.into(),
                        data: original_module,
                    };
                    component.section(&section);
                }
            }

            // Preserve all other component sections as raw bytes
            Payload::CustomSection(custom) => {
                // Custom sections (id=0 for core, component uses different encoding)
                let section = RawSection {
                    id: ComponentSectionId::CoreCustom.into(),
                    data: custom.data(),
                };
                component.section(&section);
            }

            Payload::CoreTypeSection(range) => {
                let section = RawSection {
                    id: ComponentSectionId::CoreType.into(),
                    data: &original_bytes[range.range().start..range.range().end],
                };
                component.section(&section);
            }

            Payload::ComponentTypeSection(range) => {
                let section = RawSection {
                    id: ComponentSectionId::Type.into(),
                    data: &original_bytes[range.range().start..range.range().end],
                };
                component.section(&section);
            }

            Payload::ComponentImportSection(range) => {
                let section = RawSection {
                    id: ComponentSectionId::Import.into(),
                    data: &original_bytes[range.range().start..range.range().end],
                };
                component.section(&section);
            }

            Payload::ComponentExportSection(range) => {
                let section = RawSection {
                    id: ComponentSectionId::Export.into(),
                    data: &original_bytes[range.range().start..range.range().end],
                };
                component.section(&section);
            }

            Payload::ComponentCanonicalSection(range) => {
                let section = RawSection {
                    id: ComponentSectionId::CanonicalFunction.into(),
                    data: &original_bytes[range.range().start..range.range().end],
                };
                component.section(&section);
            }

            Payload::InstanceSection(range) => {
                let section = RawSection {
                    id: ComponentSectionId::CoreInstance.into(),
                    data: &original_bytes[range.range().start..range.range().end],
                };
                component.section(&section);
            }

            Payload::ComponentInstanceSection(range) => {
                let section = RawSection {
                    id: ComponentSectionId::Instance.into(),
                    data: &original_bytes[range.range().start..range.range().end],
                };
                component.section(&section);
            }

            Payload::ComponentAliasSection(range) => {
                let section = RawSection {
                    id: ComponentSectionId::Alias.into(),
                    data: &original_bytes[range.range().start..range.range().end],
                };
                component.section(&section);
            }

            Payload::ComponentStartSection { start: _, range } => {
                let section = RawSection {
                    id: ComponentSectionId::Start.into(),
                    data: &original_bytes[range.start..range.end],
                };
                component.section(&section);
            }

            Payload::ComponentSection {
                unchecked_range, ..
            } => {
                // Nested component
                let section = RawSection {
                    id: ComponentSectionId::Component.into(),
                    data: &original_bytes[unchecked_range.start..unchecked_range.end],
                };
                component.section(&section);
            }

            // Skip payloads that don't need preservation
            // These are internal parser events, not actual sections to write
            Payload::End(_) => {}
            Payload::CodeSectionStart { .. } => {}
            Payload::CodeSectionEntry(_) => {}
            Payload::UnknownSection { .. } => {}

            // Silently skip other payload types that are parser events
            _ => {
                // Most payload types are parser events that don't correspond
                // to sections we need to write. We've handled all the important
                // component sections above.
            }
        }
    }

    Ok(component.finish())
}
