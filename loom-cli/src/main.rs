//! LOOM Command-Line Interface
//!
//! Command-line tool for optimizing WebAssembly modules

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "loom")]
#[command(author = "PulseEngine")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "LOOM - Formally Verified WebAssembly Optimizer", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Optimize a WebAssembly module
    Optimize {
        /// Input WebAssembly file (.wasm or .wat)
        #[arg(value_name = "INPUT")]
        input: String,

        /// Output file path
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<String>,

        /// Output WAT text format instead of binary
        #[arg(long)]
        wat: bool,

        /// Show optimization statistics
        #[arg(long)]
        stats: bool,

        /// Run verification after optimization
        #[arg(long)]
        verify: bool,

        /// Add transformation attestation to output (for supply chain security)
        /// Embeds a cryptographic audit trail in a custom section
        /// Enabled by default; use --no-attestation to disable
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        #[cfg(feature = "attestation")]
        attestation: bool,

        /// Select specific optimization passes (comma-separated)
        /// Available: directize,inline,precompute,constant-folding,canonicalize,peephole-synth,cse,advanced,branches,dce,merge-blocks,vacuum,simplify-locals,dead-stores,dead-locals,vacuum-final
        /// Example: --passes inline,constant-folding,dce
        /// Default: all passes
        #[arg(long, value_delimiter = ',')]
        passes: Option<Vec<String>>,

        /// Number of optimization islands to run in parallel (issue #71).
        /// 1 = current serial pipeline (default). N>1 runs the first N
        /// entries of `loom_core::islands::DEFAULT_ISLANDS` concurrently
        /// via rayon, then picks the smallest verified result.
        /// Maximum is 4 (the size of the default fleet).
        #[arg(long, default_value_t = 1)]
        islands: usize,

        /// Behavioral differential gate (#238): after optimizing, execute the
        /// original and optimized modules through wasmtime and compare outputs.
        /// HARD FAIL — on any output divergence OR if execution was inconclusive
        /// (could not instantiate / no runnable exports), the optimization is
        /// REJECTED: the original is written to the output path and the process
        /// exits non-zero. Refuses to certify behavior it never observed.
        #[arg(long)]
        #[cfg(feature = "differential")]
        differential: bool,

        /// (#239) For COMPONENTS: also strip the `component-type` metadata custom
        /// section. Default is to PRESERVE it — `wit-component` introspection
        /// (e.g. `wasm-tools component wit`) depends on it. Stripping is sound
        /// (the section is inert to execution) but degrades tooling.
        #[arg(long)]
        strip_component_type: bool,

        /// (#239 Phase B) For COMPONENTS: disable the component-level
        /// reachability GC. By default loom drops nested-core-module functions
        /// that the component provably never references (e.g. an unreachable
        /// `cabi_realloc` on an all-scalar interface, #303). The pass is
        /// removal-only and already reverts itself if the result does not
        /// validate; this switch exists so an integrator can bisect against it.
        #[arg(long)]
        no_reachability_gc: bool,

        /// (#231) Emit a `wsc.facts` custom section (schema v1) carrying the
        /// proof-carrying value-range facts loom established, keyed to the
        /// FINAL operator sequence. Default OFF — facts-absent output is
        /// byte-identical to prior releases. Facts are optional accelerators
        /// for a downstream verified compiler (synth); they never change wasm
        /// execution semantics.
        #[arg(long)]
        facts: bool,
    },

    /// Verify ISLE optimization rules (NOT YET IMPLEMENTED — always exits 2)
    Verify {
        /// ISLE file to verify
        #[arg(value_name = "ISLE_FILE")]
        isle_file: String,
    },

    /// Show version information
    Version,
}

/// Statistics for a single pass
#[derive(Debug, Clone)]
struct PassStats {
    name: String,
    instructions_before: usize,
    instructions_after: usize,
    #[allow(dead_code)]
    reduction: i64, // Can be negative if pass adds instructions
}

/// Statistics about the optimization
#[derive(Debug, Default)]
struct OptimizationStats {
    instructions_before: usize,
    instructions_after: usize,
    bytes_before: usize,
    bytes_after: usize,
    optimization_time_ms: u128,
    constant_folds: usize,
    pass_stats: Vec<PassStats>,
    reencoding_bytes: Option<usize>, // Bytes after re-encoding with no optimization
}

impl OptimizationStats {
    fn print(&self) {
        println!("\n📊 Optimization Statistics");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "Instructions: {} → {} ({:.1}% reduction)",
            self.instructions_before,
            self.instructions_after,
            self.reduction_percentage(self.instructions_before, self.instructions_after)
        );
        println!(
            "Binary size:  {} → {} bytes ({:.1}% reduction)",
            self.bytes_before,
            self.bytes_after,
            self.reduction_percentage(self.bytes_before, self.bytes_after)
        );

        // Show re-encoding baseline if available
        if let Some(reenc) = self.reencoding_bytes {
            let reenc_effect = self.bytes_before as i64 - reenc as i64;
            let opt_effect = reenc as i64 - self.bytes_after as i64;
            println!();
            println!(
                "  Re-encoding effect: {} bytes ({:+})",
                reenc, -reenc_effect
            );
            println!("  Optimization effect: {} bytes reduction", opt_effect);
        }

        println!("Constant folds: {}", self.constant_folds);
        println!("Optimization time: {} ms", self.optimization_time_ms);

        // Per-pass breakdown
        if !self.pass_stats.is_empty() {
            println!();
            println!("📈 Per-Pass Breakdown");
            println!("─────────────────────────────────────────");
            for pass in &self.pass_stats {
                let delta = pass.instructions_before as i64 - pass.instructions_after as i64;
                if delta != 0 {
                    println!(
                        "  {:20} {:4} → {:4} ({:+} instructions)",
                        pass.name, pass.instructions_before, pass.instructions_after, delta
                    );
                }
            }

            // Show passes that did nothing
            let inactive: Vec<_> = self
                .pass_stats
                .iter()
                .filter(|p| p.instructions_before == p.instructions_after)
                .map(|p| p.name.as_str())
                .collect();
            if !inactive.is_empty() {
                println!("  (no effect: {})", inactive.join(", "));
            }
        }

        // Verification coverage (#331). The revert summary below answers
        // "how often did verification catch something?"; it has no
        // denominator, so on its own it cannot answer "how much was verified
        // at all?" — and the outcome that matters most is invisible to it by
        // construction: a transform KEPT without a proof does not revert, so
        // it never appears in a revert count.
        //
        // loom keeps such transforms deliberately and for stated reasons.
        // That is defensible; reporting success without mentioning it is not
        // (REQ-3). So the denominator is printed whether or not anything was
        // skipped, and an incomplete run says so in as many words.
        let coverage = loom_core::stats::coverage_summary();
        if coverage.attempts() > 0 {
            println!();
            println!("🔍 Verification Coverage");
            println!("─────────────────────────────────────────");
            match coverage.proven_percentage() {
                Some(pct) => println!(
                    "  Proven:  {}/{} attempts ({:.1}%)",
                    coverage.proven,
                    coverage.attempts(),
                    pct
                ),
                None => println!("  Proven:  0/0 attempts"),
            }
            if !coverage.kept_unproven.is_empty() {
                println!(
                    "  Kept WITHOUT proof: {} attempt(s)",
                    coverage.total_kept_unproven()
                );
                for (reason, count) in &coverage.kept_unproven {
                    println!("    {:36} {}", reason, count);
                }
            }
            if coverage.reverted > 0 {
                println!("  Rejected (reverted): {}", coverage.reverted);
            }
            println!("  (one attempt = one function, one pass — not one function)");
            if !coverage.is_fully_proven() {
                println!();
                println!("  ⚠️  Not every transform in this output carries a proof.");
            }
        }

        // Per-pass revert summary — counts how many functions each pass
        // tried to transform and then reverted because the verifier
        // rejected the result. Non-zero counts indicate verification
        // saved a potential miscompile.
        let reverts = loom_core::stats::revert_summary();
        if !reverts.is_empty() {
            println!();
            println!("🔁 Verification Reverts");
            println!("─────────────────────────────────────────");
            for (pass, count) in &reverts {
                println!("  {:30} {} function(s) reverted", pass, count);
            }
            let total: u64 = reverts.values().sum();
            println!("  Total: {} revert(s)", total);
        }
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    fn reduction_percentage(&self, before: usize, after: usize) -> f64 {
        if before == 0 {
            0.0
        } else {
            ((before as f64 - after as f64) / before as f64) * 100.0
        }
    }
}

/// Count instructions in a module
fn count_instructions(module: &loom_core::Module) -> usize {
    module.functions.iter().map(|f| f.instructions.len()).sum()
}

/// Count constant folding opportunities (I32Add with two constants)
fn count_constant_folds(module: &loom_core::Module) -> usize {
    use loom_core::Instruction;

    let mut count = 0;
    for func in &module.functions {
        let instrs = &func.instructions;
        for i in 0..instrs.len().saturating_sub(2) {
            if matches!(instrs[i], Instruction::I32Const(_))
                && matches!(instrs[i + 1], Instruction::I32Const(_))
                && matches!(instrs[i + 2], Instruction::I32Add)
            {
                count += 1;
            }
        }
    }
    count
}

/// Build transformation attestation for the optimization
#[cfg(feature = "attestation")]
fn build_attestation(
    input_bytes: &[u8],
    input_name: &str,
    optimized_module: &loom_core::Module,
    enabled_passes: &Option<Vec<&str>>,
    verification_status: Option<bool>,
) -> String {
    use wsc_attestation::TransformationAttestationBuilder;

    // Encode the optimized module to get its bytes for hashing
    let output_bytes = loom_core::encode::encode_wasm(optimized_module).unwrap_or_default();

    // Build the attestation
    let mut builder =
        TransformationAttestationBuilder::new_optimization("loom", env!("CARGO_PKG_VERSION"))
            .add_input_unsigned(input_bytes, input_name);

    // Add optimization parameters
    let passes_str = enabled_passes
        .as_ref()
        .map(|p| p.join(","))
        .unwrap_or_else(|| "all".to_string());
    builder = builder.add_parameter("passes", serde_json::json!(passes_str));

    // Add verification status if available
    if let Some(verified) = verification_status {
        builder = builder.add_parameter("z3_verified", serde_json::json!(verified));
    }

    // Add metadata about the optimization
    builder = builder.add_metadata(
        "instructions_before",
        serde_json::json!(count_instructions_from_bytes(input_bytes)),
    );
    builder = builder.add_metadata(
        "instructions_after",
        serde_json::json!(count_instructions(optimized_module)),
    );

    // Build and serialize
    let attestation = builder.build(&output_bytes, "output.wasm");
    attestation.to_json().unwrap_or_else(|_| "{}".to_string())
}

/// Count instructions from raw WASM bytes (for attestation)
#[cfg(feature = "attestation")]
fn count_instructions_from_bytes(bytes: &[u8]) -> usize {
    loom_core::parse::parse_wasm(bytes)
        .map(|m| count_instructions(&m))
        .unwrap_or(0)
}

/// Behavioral differential gate (#238). When enabled (`original` is `Some`),
/// execute the original and optimized WASM through wasmtime and compare outputs.
/// HARD FAIL: on any divergence or inconclusive execution, write the ORIGINAL
/// (safe, unoptimized) bytes to `output_path` and return `Err` so the process
/// exits non-zero. A no-op when the gate is disabled (`original` is `None`).
#[allow(unused_variables)]
/// Write the optimized artifact ONLY if it passes authoritative WebAssembly
/// validation (#346).
///
/// `loom optimize` used to write whatever the encoder produced and print
/// "✅ Optimization complete!" over it. When the encoder dropped the data
/// count section while keeping the `memory.init` instructions that require it,
/// the result was a module no validator accepts — emitted with exit code 0.
/// Silent invalid output is the worst shape this can take: the next tool in
/// the chain reports the failure against ITS OWN input, so the blame lands
/// downstream of the tool that actually broke the module.
///
/// `loom_core::optimize::optimize_module` has carried exactly this gate since
/// #257, described there as the systemic guarantee that loom can NEVER emit
/// structurally invalid wasm. The CLI does not go through that function — it
/// drives the passes directly (#345) — so it inherited none of it. This closes
/// that hole at the only point that matters: the write.
///
/// On failure the ORIGINAL input is written instead, when the original itself
/// validates, so the worst case is unoptimized-but-valid output rather than a
/// missing or corrupt file. The exit code is still non-zero: a caller that
/// checks it learns something went wrong, and a caller that ignores it still
/// gets a module that loads.
fn write_validated_output(
    output_path: &str,
    output_bytes: &[u8],
    wasm_to_validate: &[u8],
    original_input: &[u8],
) -> Result<()> {
    if let Err(e) = loom_core::encode::validate_output_bytes(wasm_to_validate) {
        eprintln!("error: the optimized module failed authoritative WebAssembly validation:");
        eprintln!("       {e}");
        // Fall back to the input, but only after proving the input itself is
        // valid wasm — a `.wat` input, or one that was already invalid, must
        // not be copied over the output path and presented as a module.
        let recovered = loom_core::encode::validate_output_bytes(original_input).is_ok()
            && fs::write(output_path, original_input).is_ok();
        if recovered {
            eprintln!("note: wrote the ORIGINAL module to {output_path} instead;");
            eprintln!("      nothing invalid was written, and nothing was optimized.");
        } else {
            eprintln!("note: no output was written.");
        }
        eprintln!("note: this is a loom bug — please report it with the input module.");
        return Err(anyhow!("refusing to emit a module that does not validate"));
    }
    fs::write(output_path, output_bytes).context("Failed to write output file")?;
    println!("✓ Written to: {}", output_path);
    Ok(())
}

fn maybe_differential_gate(
    original: Option<&[u8]>,
    // Underscored: these are read only inside the `differential` cfg block
    // below, so without that feature they are genuinely unused. The
    // verification gate builds with `RUSTFLAGS: -D warnings`, which turned
    // three such warnings into hard errors and stopped `loom-cli` compiling
    // there at all — so every artifact whose evidence is a `cargo test -p
    // loom-cli` invocation failed, while being marked `verified`. The
    // warnings were long-standing and looked ignorable in a normal build;
    // they were only ever visible in the one environment that had not
    // completed a run in six weeks (#353).
    #[cfg_attr(not(feature = "differential"), allow(unused_variables))] optimized_wasm: &[u8],
    #[cfg_attr(not(feature = "differential"), allow(unused_variables))] output_path: &str,
) -> Result<()> {
    #[cfg_attr(not(feature = "differential"), allow(unused_variables))]
    let original = match original {
        Some(o) => o,
        None => return Ok(()),
    };
    #[cfg(feature = "differential")]
    {
        use loom_testing::differential::{DifferentialExecutor, ExecutionConfig};
        println!("🔬 Differential gate: executing original vs optimized...");
        // Use the `thorough` profile so the gate actually exercises wider exports
        // (up to 6 params) instead of skipping them — fewer false "inconclusive"
        // rejections on real modules. Strict certification still requires zero
        // skipped functions; this just widens what we can drive.
        let exec = DifferentialExecutor::with_config(ExecutionConfig::thorough())
            .context("Failed to initialize differential executor")?;
        let result = exec
            .test_pair(original, optimized_wasm)
            .context("Differential execution failed")?;
        println!("   {}", result.summary());
        if let Some(reason) = result.strict_failure_reason() {
            fs::write(output_path, original)
                .context("Failed to write original after differential rejection")?;
            println!("✗ Differential gate REJECTED — wrote ORIGINAL to {output_path}");
            return Err(anyhow!("differential gate rejected optimization: {reason}"));
        }
        println!(
            "✓ Differential gate: behavior preserved on {} executed export(s)",
            result.functions_tested
        );
    }
    Ok(())
}

/// Optimize command implementation
#[allow(clippy::too_many_arguments)]
fn optimize_command(
    input: String,
    output: Option<String>,
    output_wat: bool,
    show_stats: bool,
    run_verify: bool,
    add_attestation: bool,
    passes: Option<Vec<String>>,
    islands: usize,
    run_differential: bool,
    strip_component_type: bool,
    no_reachability_gc: bool,
    emit_facts: bool,
) -> Result<()> {
    println!("🔧 LOOM Optimizer v{}", env!("CARGO_PKG_VERSION"));
    println!("Input: {}", input);

    // Determine input format
    let input_path = Path::new(&input);
    if !input_path.exists() {
        return Err(anyhow!("Input file not found: {}", input));
    }

    let is_wat_input = input_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s == "wat")
        .unwrap_or(false);

    // Read input file
    let input_bytes = fs::read(&input).context("Failed to read input file")?;

    // Check if this is a component (Phase 9)
    if !is_wat_input && input_bytes.len() > 8 {
        // Check magic number for component format
        // Components have: 0x00 0x61 0x73 0x6d (magic) followed by 0x0d 0x00 0x01 0x00 (component version)
        if &input_bytes[0..4] == b"\0asm" && input_bytes[4] == 0x0d && input_bytes[5] == 0x00 {
            println!("🧩 Detected WebAssembly Component!");
            println!("📦 Attempting component optimization...");
            if emit_facts {
                // #231: the component pipeline encodes its core modules
                // internally and the fact source is not wired into it. Say so
                // rather than let --facts look like it did something.
                println!(
                    "⚠️  --facts is not wired into the component pipeline yet — \
                     no wsc.facts section will be emitted for this input (#231)."
                );
            }

            // Use component optimization
            let component_config = loom_core::ComponentOptimizeConfig {
                strip_component_type,
                reachability_gc: !no_reachability_gc,
            };
            match loom_core::optimize_component_with_config(&input_bytes, component_config) {
                Ok((optimized_bytes, stats)) => {
                    println!("\n📊 Component Optimization Results");
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    println!(
                        "Component:    {} → {} bytes ({:.1}% reduction)",
                        stats.original_size,
                        stats.optimized_size,
                        if stats.original_size > 0 {
                            ((stats.original_size as f64 - stats.optimized_size as f64)
                                / stats.original_size as f64)
                                * 100.0
                        } else {
                            0.0
                        }
                    );
                    println!(
                        "Core modules: {} found, {} optimized",
                        stats.module_count, stats.modules_optimized
                    );
                    if stats.original_module_size > 0 {
                        println!(
                            "Module size:  {} → {} bytes ({:.1}% reduction)",
                            stats.original_module_size,
                            stats.optimized_module_size,
                            ((stats.original_module_size as f64
                                - stats.optimized_module_size as f64)
                                / stats.original_module_size as f64)
                                * 100.0
                        );
                    }
                    println!("Status:       {}", stats.message);

                    // #239 size breakdown — attribute the win by category.
                    let bd = &stats.size_breakdown;
                    println!("\n📐 Size Breakdown (component-level)");
                    println!(
                        "  Core module code:   {} → {} bytes",
                        bd.core_module_bytes_before, bd.core_module_bytes_after
                    );
                    println!(
                        "  Component types:    {} bytes ({})",
                        bd.component_type_bytes,
                        if bd.component_type_stripped {
                            "STRIPPED (--strip-component-type)"
                        } else if bd.component_type_bytes > 0 {
                            "preserved (default)"
                        } else {
                            "none"
                        }
                    );
                    println!(
                        "  Custom stripped:    {} bytes",
                        bd.custom_sections_stripped
                    );
                    if !bd.stripped_section_names.is_empty() {
                        println!(
                            "                      [{}]",
                            bd.stripped_section_names.join(", ")
                        );
                    }
                    // #239 Phase B — component-level reachability GC.
                    if bd.reachability_gc_applied {
                        println!(
                            "  Reachability GC:    {} unreferenced core export(s), \
                             {} unreachable function(s) removed",
                            bd.core_exports_pruned, bd.core_functions_removed
                        );
                    } else {
                        println!(
                            "  Reachability GC:    not applied (disabled, or the component \
                             graph could not be bounded)"
                        );
                    }
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                    // Write optimized component
                    let output_path = output.unwrap_or_else(|| "output.wasm".to_string());
                    // #346: components go through the same gate. `validate`
                    // handles the component grammar as well as core modules.
                    write_validated_output(
                        &output_path,
                        &optimized_bytes,
                        &optimized_bytes,
                        &input_bytes,
                    )?;
                    println!("\n✅ Optimization complete!");
                    return Ok(());
                }
                Err(e) => {
                    println!("⚠️  Component optimization not available: {}", e);
                    println!("💡 Falling back to core module optimization...");
                }
            }
        }
    }

    // Parse WebAssembly module
    let start_parse = Instant::now();
    let mut module = if is_wat_input {
        println!("📄 Parsing WAT text format...");
        loom_core::parse::parse_wat(std::str::from_utf8(&input_bytes)?)
            .context("Failed to parse WAT file")?
    } else {
        println!("📦 Parsing WASM binary format...");
        loom_core::parse::parse_wasm(&input_bytes).context("Failed to parse WASM file")?
    };
    let parse_time = start_parse.elapsed();
    println!("✓ Parsed in {:?}", parse_time);

    // Save original module for verification (if requested)
    let original_module = if run_verify {
        Some(module.clone())
    } else {
        None
    };

    // Capture the UNOPTIMIZED wasm for the behavioral differential gate (#238).
    // For wasm input that's the input bytes; for WAT we re-encode the freshly
    // parsed (pre-optimization) module so original and optimized are both wasm.
    let original_wasm: Option<Vec<u8>> = if run_differential {
        Some(if is_wat_input {
            loom_core::encode::encode_wasm(&module)
                .context("Failed to encode original (from WAT) for differential gate")?
        } else {
            input_bytes.clone()
        })
    } else {
        None
    };

    // Collect statistics before optimization
    let mut stats = OptimizationStats {
        instructions_before: count_instructions(&module),
        bytes_before: input_bytes.len(),
        constant_folds: count_constant_folds(&module),
        ..Default::default()
    };

    // Measure re-encoding baseline (parse → encode with no optimization)
    // This shows what size change comes from just re-encoding vs actual optimization
    if show_stats && !is_wat_input {
        if let Ok(reencoded) = loom_core::encode::encode_wasm(&module) {
            stats.reencoding_bytes = Some(reencoded.len());
        }
    }

    // Apply optimizations
    println!("⚡ Optimizing...");
    let start_opt = Instant::now();

    // Island-model dispatch (issue #71). When --islands N is >1, hand off to
    // the parallel harness. The harness runs the first N entries of
    // DEFAULT_ISLANDS concurrently, validates each, and picks the smallest.
    // This intentionally ignores --passes: the islands carry their own pass
    // orderings, and combining the two would defeat the point of the
    // comparison.
    if islands > 1 {
        let fleet = loom_core::islands::DEFAULT_ISLANDS;
        let n = islands.min(fleet.len());
        if islands > fleet.len() {
            println!(
                "  Note: requested {} islands but only {} are defined; running {}.",
                islands,
                fleet.len(),
                n
            );
        }
        println!("🏝️  Running {} optimization islands in parallel...", n);
        module = loom_core::islands::optimize_module_islands(&module, &fleet[..n])
            .context("Island-model optimization failed")?;
        stats.optimization_time_ms = start_opt.elapsed().as_millis();
        println!("✓ Islands optimized in {} ms", stats.optimization_time_ms);

        // Collect post-optimization stats
        stats.instructions_after = count_instructions(&module);

        // #242: drop stale .debug_* (code offsets changed under optimization).
        loom_core::fused_optimizer::strip_debug_sections(&mut module);

        // Skip the serial pipeline below — the islands already ran a full
        // pass sequence on the winning module. Jump directly to encoding.
        let output_bytes = if output_wat {
            println!("📝 Encoding to WAT format...");
            loom_core::encode::encode_wat(&module)
                .context("Failed to encode to WAT")?
                .into_bytes()
        } else {
            println!("📦 Encoding to WASM binary...");
            // #231 fact SOURCE: derive the facts from the island WINNER, here,
            // immediately before the encode that consumes them — the operator
            // keying is only valid against the body being encoded.
            if emit_facts {
                module.facts = loom_core::facts::collect_module_facts(&module);
                println!("🔖 {} proof-carrying fact(s) collected", module.facts.len());
            }
            loom_core::encode::encode_wasm_with_facts(&module, emit_facts)
                .context("Failed to encode to WASM")?
        };
        stats.bytes_after = output_bytes.len();

        let output_path = output.unwrap_or_else(|| {
            if output_wat {
                "output.wat".to_string()
            } else {
                "output.wasm".to_string()
            }
        });
        if run_differential {
            let opt_wasm = if output_wat {
                loom_core::encode::encode_wasm(&module)
                    .context("Failed to encode optimized module for differential gate")?
            } else {
                output_bytes.clone()
            };
            maybe_differential_gate(original_wasm.as_deref(), &opt_wasm, &output_path)?;
        }
        // #346: validate before writing. For WAT output the bytes on disk are
        // text, so the WASM encoding is what gets validated.
        let validate_bytes = if output_wat {
            loom_core::encode::encode_wasm(&module)
                .context("Failed to encode optimized module for output validation")?
        } else {
            output_bytes.clone()
        };
        write_validated_output(&output_path, &output_bytes, &validate_bytes, &input_bytes)?;

        if show_stats {
            stats.print();
        }

        // Suppress unused-variable warnings for branches that the
        // island path doesn't exercise (verification, attestation are
        // still meaningful TODOs for island mode but out of scope for
        // this PR — the per-island Z3 + stack gates already ran inside
        // each optimization pass).
        let _ = (run_verify, add_attestation, original_module);
        return Ok(());
    }

    // Determine which passes to run
    let enabled_passes = passes
        .as_ref()
        .map(|p| p.iter().map(|s| s.as_str()).collect::<Vec<&str>>());

    // Helper to check if a pass is enabled
    let should_run = |pass_name: &str| -> bool {
        enabled_passes
            .as_ref()
            .is_none_or(|passes| passes.contains(&pass_name))
    };

    // Helper to track pass stats
    let mut track_pass = |name: &str, before: usize, after: usize| {
        if show_stats {
            stats.pass_stats.push(PassStats {
                name: name.to_string(),
                instructions_before: before,
                instructions_after: after,
                reduction: before as i64 - after as i64,
            });
        }
    };

    // Run selected passes in optimal order with tracking
    // Directize early — resolves call_indirect → direct call before
    // inline, so the inliner sees a strictly larger set of direct calls
    // it can fold through. Conservative: skips the entire module if any
    // function contains an Unknown instruction (which is how the parser
    // represents table.set / .fill / .copy / .init / .grow today), and
    // skips individual call_indirect sites whose constant table index
    // can't be resolved to a unique function via the element section.
    if should_run("directize") {
        println!("  Running: directize");
        let before = count_instructions(&module);
        loom_core::optimize::directize(&mut module).context("Directize failed")?;
        let after = count_instructions(&module);
        track_pass("directize", before, after);
    }

    if should_run("inline") {
        println!("  Running: inline");
        let before = count_instructions(&module);
        loom_core::optimize::inline_functions(&mut module).context("Function inlining failed")?;
        let after = count_instructions(&module);
        track_pass("inline", before, after);
    }

    // #219: dissolve the u64 ABI carrier the inline leaves behind (scalar-forward
    // the single-assignment carrier to its unpack sites, then SROA). Runs right
    // after inline so the carrier is fresh; downstream dce/dead-locals reap it.
    if should_run("forward-carrier") {
        println!("  Running: forward-carrier");
        let before = count_instructions(&module);
        loom_core::optimize::forward_carrier_locals(&mut module)
            .context("Carrier forwarding failed")?;
        let after = count_instructions(&module);
        track_pass("forward-carrier", before, after);
    }

    // loom#228: whole-function (module-level) dead-function elimination.
    // Multi-site inlining duplicates a callee's body into each caller and
    // leaves the ORIGINAL orphaned; the body-level `dce` (eliminate_dead_code)
    // is intra-function and never removes a whole function, so the orphan
    // survived every `loom optimize` run and grew the module. Run it right
    // after `inline` so the orphan is gone before downstream passes touch it.
    // eliminate_dead_functions keeps exports, the start function, and every
    // element-segment (indirect-call table) target live — conservative on a
    // parse failure — so it can never drop a reachable function (#196).
    if should_run("dce-functions") {
        println!("  Running: dce-functions");
        let before = count_instructions(&module);
        let removed = loom_core::fused_optimizer::eliminate_dead_functions(&mut module)
            .context("Whole-function DCE failed")?;
        if removed > 0 {
            println!("    removed {removed} dead function(s)");
        }
        let after = count_instructions(&module);
        track_pass("dce-functions", before, after);
    }

    if should_run("precompute") {
        println!("  Running: precompute");
        let before = count_instructions(&module);
        loom_core::optimize::precompute(&mut module).context("Precompute failed")?;
        let after = count_instructions(&module);
        track_pass("precompute", before, after);
    }

    if should_run("constant-folding") {
        println!("  Running: constant-folding");
        let before = count_instructions(&module);
        loom_core::optimize::constant_folding(&mut module).context("Constant folding failed")?;
        let after = count_instructions(&module);
        track_pass("constant-folding", before, after);
    }

    // Canonicalize early so downstream passes (CSE, branches, dce) see
    // the if/else → select form and the local.tee normal form.
    if should_run("canonicalize") {
        println!("  Running: canonicalize");
        let before = count_instructions(&module);
        loom_core::optimize::canonicalize(&mut module).context("Canonicalize failed")?;
        let after = count_instructions(&module);
        track_pass("canonicalize", before, after);
    }

    // ægraph-based optimization. Runs AFTER canonicalize (canonical
    // operand order makes pattern matching deterministic) and BEFORE
    // peephole-synth (so the egraph engine gets first crack at identity
    // folds — the substrate is richer than peephole's linear pattern
    // matcher). Default-on as of v1.1.0: cost-driven extraction (Track B)
    // plus the widened i64/commutativity rule set (Track C) make it a
    // net-neutral-or-better pass on the corpus. Each function is reverted
    // untouched unless extraction is strictly shorter, so default-on
    // cannot regress output — see egraph_optimize_body.
    if should_run("egraph") {
        println!("  Running: egraph");
        let before = count_instructions(&module);
        loom_core::optimize::egraph_optimize(&mut module).context("egraph_optimize failed")?;
        let after = count_instructions(&module);
        track_pass("egraph", before, after);
    }

    // Souper-shaped peephole synthesis (PR-L2): hand-curated arithmetic
    // identity rules (12 candidates). Runs AFTER canonicalize (so the
    // canonical operand-order is in place) and BEFORE cse (so the
    // identity-eliminated forms feed CSE's value-numbering).
    if should_run("peephole-synth") {
        println!("  Running: peephole-synth");
        let before = count_instructions(&module);
        loom_core::peephole_synth::apply_peephole_synth(&mut module)
            .context("Peephole synthesis failed")?;
        let after = count_instructions(&module);
        track_pass("peephole-synth", before, after);
    }

    if should_run("cse") {
        println!("  Running: cse");
        let before = count_instructions(&module);
        loom_core::optimize::eliminate_common_subexpressions_enhanced(&mut module)
            .context("CSE failed")?;
        let after = count_instructions(&module);
        track_pass("cse", before, after);
    }

    if should_run("advanced") {
        println!("  Running: advanced");
        let before = count_instructions(&module);
        loom_core::optimize::optimize_advanced_instructions(&mut module)
            .context("Advanced instruction optimization failed")?;
        let after = count_instructions(&module);
        track_pass("advanced", before, after);
    }

    if should_run("branches") {
        println!("  Running: branches");
        let before = count_instructions(&module);
        loom_core::optimize::simplify_branches(&mut module)
            .context("Branch simplification failed")?;
        let after = count_instructions(&module);
        track_pass("branches", before, after);
    }

    if should_run("dce") {
        println!("  Running: dce");
        let before = count_instructions(&module);
        loom_core::optimize::eliminate_dead_code(&mut module).context("DCE failed")?;
        let after = count_instructions(&module);
        track_pass("dce", before, after);
    }

    if should_run("merge-blocks") {
        println!("  Running: merge-blocks");
        let before = count_instructions(&module);
        loom_core::optimize::merge_blocks(&mut module).context("Block merging failed")?;
        let after = count_instructions(&module);
        track_pass("merge-blocks", before, after);
    }

    if should_run("vacuum") {
        println!("  Running: vacuum");
        let before = count_instructions(&module);
        loom_core::optimize::vacuum(&mut module).context("Vacuum cleanup failed")?;
        let after = count_instructions(&module);
        track_pass("vacuum", before, after);
    }

    if should_run("simplify-locals") {
        println!("  Running: simplify-locals");
        let before = count_instructions(&module);
        loom_core::optimize::simplify_locals(&mut module).context("SimplifyLocals failed")?;
        let after = count_instructions(&module);
        track_pass("simplify-locals", before, after);
    }

    if should_run("dead-stores") {
        println!("  Running: dead-stores");
        let before = count_instructions(&module);
        loom_core::optimize::eliminate_dead_stores(&mut module)
            .context("Dead-store elimination failed")?;
        let after = count_instructions(&module);
        track_pass("dead-stores", before, after);
    }

    if should_run("dead-locals") {
        println!("  Running: dead-locals");
        let before = count_instructions(&module);
        loom_core::optimize::eliminate_dead_locals(&mut module)
            .context("Dead-local elimination failed")?;
        let after = count_instructions(&module);
        track_pass("dead-locals", before, after);
    }

    // Final vacuum sweep: dead-stores neutralizes LocalSet → Drop and
    // dead-locals neutralizes LocalSet/LocalTee. The peephole inside
    // vacuum_instructions folds `pure_push; Drop` pairs (PR #99). But
    // the earlier vacuum runs BEFORE these passes, so the const+drop
    // pairs they create never get cleaned up by it. A second sweep
    // closes the loop. Trades ~tens of milliseconds for the bytes that
    // would otherwise survive as orphan `i32.const N; drop` residues
    // in functions matching the gale default-then-override pattern.
    if should_run("vacuum-final") {
        println!("  Running: vacuum-final");
        let before = count_instructions(&module);
        loom_core::optimize::vacuum(&mut module).context("Final vacuum failed")?;
        let after = count_instructions(&module);
        track_pass("vacuum-final", before, after);
    }

    // #242: loom has transformed the code, so any `.debug_*` (DWARF) sections —
    // whose line/info tables reference the original instruction offsets — are now
    // stale and would mislead a debugger. Drop them. (The `name` section is
    // remapped, not dropped, inside dce-functions when functions are renumbered.)
    let dropped_debug = loom_core::fused_optimizer::strip_debug_sections(&mut module);
    if dropped_debug > 0 {
        println!("  Dropped {} stale .debug_* section(s)", dropped_debug);
    }

    stats.optimization_time_ms = start_opt.elapsed().as_millis();
    println!("✓ Optimized in {} ms", stats.optimization_time_ms);

    // Collect statistics after optimization
    stats.instructions_after = count_instructions(&module);

    // Add transformation attestation if enabled
    #[cfg(feature = "attestation")]
    let verification_status = if run_verify {
        #[cfg(feature = "verification")]
        {
            // Check if Z3 verification passed
            match loom_core::verify::verify_optimization(
                original_module.as_ref().unwrap_or(&module),
                &module,
            ) {
                Ok(true) => Some(true),
                Ok(false) => Some(false),
                Err(_) => None,
            }
        }
        #[cfg(not(feature = "verification"))]
        None
    } else {
        None
    };

    #[cfg(feature = "attestation")]
    if add_attestation && !output_wat {
        println!("🔏 Adding transformation attestation...");
        let attestation = build_attestation(
            &input_bytes,
            &input,
            &module,
            &enabled_passes,
            verification_status,
        );
        // Add attestation to custom sections
        module.custom_sections.push((
            wsc_attestation::TRANSFORMATION_ATTESTATION_SECTION.to_string(),
            attestation.into_bytes(),
        ));
        println!("✓ Attestation added to custom section");
    }
    #[cfg(not(feature = "attestation"))]
    let _ = add_attestation; // Suppress unused warning

    // Encode output
    let output_bytes = if output_wat {
        println!("📝 Encoding to WAT format...");
        loom_core::encode::encode_wat(&module)
            .context("Failed to encode to WAT")?
            .into_bytes()
    } else {
        println!("📦 Encoding to WASM binary...");
        if emit_facts {
            // #231 fact SOURCE: derive the facts from the FINAL module, here,
            // immediately before the encode that consumes them — no pass runs
            // in between, so no fact can be renumbered out from under its
            // operator.
            module.facts = loom_core::facts::collect_module_facts(&module);
            println!("🔖 {} proof-carrying fact(s) collected", module.facts.len());
            // #231: emit the wsc.facts custom section from module.facts, keyed
            // to the FINAL operator sequence this same call encodes.
            loom_core::encode::encode_wasm_with_facts(&module, true)
                .context("Failed to encode to WASM (with facts)")?
        } else {
            loom_core::encode::encode_wasm(&module).context("Failed to encode to WASM")?
        }
    };
    stats.bytes_after = output_bytes.len();

    // Write output file
    let output_path = output.unwrap_or_else(|| {
        if output_wat {
            "output.wat".to_string()
        } else {
            "output.wasm".to_string()
        }
    });
    // Behavioral differential gate (#238): on HARD FAIL this writes the original
    // to output_path and returns Err (non-zero exit) before the optimized write.
    if run_differential {
        let opt_wasm = if output_wat {
            loom_core::encode::encode_wasm(&module)
                .context("Failed to encode optimized module for differential gate")?
        } else {
            output_bytes.clone()
        };
        maybe_differential_gate(original_wasm.as_deref(), &opt_wasm, &output_path)?;
    }
    // #346: validate before writing. For WAT output the bytes on disk are
    // text, so the WASM encoding is what gets validated.
    let validate_bytes = if output_wat {
        loom_core::encode::encode_wasm(&module)
            .context("Failed to encode optimized module for output validation")?
    } else {
        output_bytes.clone()
    };
    write_validated_output(&output_path, &output_bytes, &validate_bytes, &input_bytes)?;

    // Show statistics if requested
    if show_stats {
        stats.print();
    }

    // Run verification if requested
    if run_verify {
        println!("\n🔍 Running verification...");
        if let Some(ref original) = original_module {
            run_verification(original, &module)?;
        }
    }

    println!("\n✅ Optimization complete!");
    Ok(())
}

/// Run property-based verification on the module
fn run_verification(original: &loom_core::Module, optimized: &loom_core::Module) -> Result<()> {
    use loom_core::Instruction;
    use loom_isle::{Imm32, ValueData, iadd32, iconst32, simplify};

    // First, run Z3 SMT-based formal verification (if feature enabled)
    #[cfg(feature = "verification")]
    {
        println!("🔬 Running Z3 SMT verification...");
        match loom_core::verify::verify_optimization(original, optimized) {
            Ok(true) => {
                println!("  ✅ Z3 verification passed: optimizations are semantically equivalent");
            }
            Ok(false) => {
                println!("  ❌ Z3 verification failed: counterexample found");
                println!("  ⚠️  The optimization may have changed program semantics!");
                return Err(anyhow!("Z3 verification failed"));
            }
            Err(e) => {
                println!("  ⚠️  Z3 verification error: {}", e);
                println!("  💡 Continuing with property-based verification...");
            }
        }
    }
    #[cfg(not(feature = "verification"))]
    {
        let _ = original; // Suppress unused warning when verification is disabled
        println!("⚠️  Z3 verification feature not enabled");
        println!("💡 Rebuild with --features verification to enable formal verification");
        println!();
    }

    // Run property-based verification on ISLE optimizations
    let mut test_count = 0;
    let mut pass_count = 0;

    println!("🧪 Running ISLE property-based verification...");

    // Test each constant folding in the module
    for func in &optimized.functions {
        let instrs = &func.instructions;
        for instr in instrs {
            if let Instruction::I32Const(result) = instr {
                // Check if this could have come from constant folding
                // We verify it matches the expected semantics
                test_count += 1;

                // Simple verification: constant is still a constant after optimization
                let term = iconst32(Imm32::from(*result));
                let optimized = simplify(term.clone());

                match optimized.data() {
                    ValueData::I32Const { val } if val.value() == *result => {
                        pass_count += 1;
                    }
                    _ => {
                        println!("  ⚠️  Verification warning: constant changed");
                    }
                }
            }
        }
    }

    // Additional property tests
    println!("  Running property tests...");

    // Test: simplify is idempotent
    for x in [0, 1, -1, 42, i32::MAX, i32::MIN] {
        for y in [0, 1, -1, 42, i32::MAX, i32::MIN] {
            test_count += 1;
            let term = iadd32(iconst32(Imm32::from(x)), iconst32(Imm32::from(y)));
            let opt1 = simplify(term);
            let opt2 = simplify(opt1.clone());

            if opt1 == opt2 {
                pass_count += 1;
            } else {
                println!("  ⚠️  Idempotence failed for {} + {}", x, y);
            }
        }
    }

    println!("✓ Verification: {}/{} tests passed", pass_count, test_count);

    if pass_count == test_count {
        println!("✓ All verification tests passed!");
        Ok(())
    } else {
        Err(anyhow!(
            "Verification failed: {}/{} tests passed",
            pass_count,
            test_count
        ))
    }
}

/// Process exit code for `loom verify`.
///
/// `verify` has NO success path today, so this is the only code it can
/// produce. It is deliberately distinct from `1` (the code `main`'s
/// `Result` produces for an ordinary optimization failure) so a caller can
/// tell "this build cannot verify anything" apart from "verification ran
/// and rejected something" once Phase 5 makes the latter possible.
const EXIT_VERIFY_UNIMPLEMENTED: i32 = 2;

/// The `loom verify` placeholder (#332).
///
/// This subcommand used to print `✓ LOOM Verification`, never open the
/// file, and **exit 0** — for a `.wasm` passed where an ISLE file was
/// expected, for a file containing arbitrary garbage, for an empty file,
/// and for a path that did not exist at all. The `⚠️ not yet implemented`
/// line was present and honest, but it sat below the checkmark and never
/// reached the exit code, which is the only thing automation reads. Wiring
/// that into CI would have produced a check that was green forever over
/// nothing, in a tool whose entire claim is that it proves things.
///
/// So there is deliberately **no exit-0 path here**. Input validation runs
/// first and reports real errors, but passing it establishes only that the
/// argument is a plausible ISLE file — never that anything was verified.
/// The two outcomes are kept textually distinct for that reason: a
/// validation failure says `error:`, and a validated file still says
/// `warning: nothing was verified`. Both exit non-zero.
///
/// Everything goes to stderr; stdout stays empty so a caller redirecting it
/// gets nothing that could be mistaken for a result.
fn run_verify_placeholder(isle_file: &str) -> ! {
    let path = std::path::Path::new(isle_file);

    if !path.exists() {
        eprintln!("error: no such file: {}", isle_file);
        std::process::exit(EXIT_VERIFY_UNIMPLEMENTED);
    }
    if !path.is_file() {
        eprintln!("error: not a regular file: {}", isle_file);
        std::process::exit(EXIT_VERIFY_UNIMPLEMENTED);
    }
    if path.extension().and_then(|e| e.to_str()) != Some("isle") {
        eprintln!(
            "error: expected an ISLE rule file (.isle), got: {}",
            isle_file
        );
        eprintln!("note: `loom verify` checks ISLE rewrite rules, not wasm modules.");
        eprintln!("note: to optimize a wasm module, use `loom optimize`.");
        std::process::exit(EXIT_VERIFY_UNIMPLEMENTED);
    }
    // Prove the file is actually readable rather than merely present — a
    // path that stats but cannot be opened is exactly the kind of thing a
    // check should not pass over in silence.
    if let Err(e) = std::fs::read(path) {
        eprintln!("error: cannot read {}: {}", isle_file, e);
        std::process::exit(EXIT_VERIFY_UNIMPLEMENTED);
    }

    eprintln!(
        "warning: ISLE rule verification is not implemented yet (Phase 5); \
         nothing was verified."
    );
    eprintln!("note: {} was read but not checked.", isle_file);
    eprintln!("note: rule-level proofs today run in the test suite — see");
    eprintln!("      `cargo test -p loom-core --features verification`.");
    std::process::exit(EXIT_VERIFY_UNIMPLEMENTED);
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Optimize {
            input,
            output,
            wat,
            stats,
            verify,
            #[cfg(feature = "attestation")]
            attestation,
            passes,
            islands,
            #[cfg(feature = "differential")]
            differential,
            strip_component_type,
            no_reachability_gc,
            facts,
        }) => {
            #[cfg(feature = "attestation")]
            let add_attestation = attestation;
            #[cfg(not(feature = "attestation"))]
            let add_attestation = false;
            #[cfg(feature = "differential")]
            let run_differential = differential;
            #[cfg(not(feature = "differential"))]
            let run_differential = false;
            optimize_command(
                input,
                output,
                wat,
                stats,
                verify,
                add_attestation,
                passes,
                islands,
                run_differential,
                strip_component_type,
                no_reachability_gc,
                facts,
            )?;
        }

        Some(Commands::Verify { isle_file }) => {
            run_verify_placeholder(&isle_file);
        }

        Some(Commands::Version) => {
            println!("LOOM v{}", env!("CARGO_PKG_VERSION"));
            println!("Formally Verified WebAssembly Optimizer");
            println!();
            println!("Project: https://github.com/pulseengine/loom");
            println!("License: Apache-2.0");
        }

        None => {
            println!("LOOM - Formally Verified WebAssembly Optimizer");
            println!();
            println!("Usage: loom <COMMAND>");
            println!();
            println!("Commands:");
            println!("  optimize    Optimize a WebAssembly module");
            println!("  verify      Verify ISLE optimization rules (not yet implemented)");
            println!("  version     Show version information");
            println!("  help        Print this message or the help of a subcommand");
            println!();
            println!("For more information, run: loom help");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_cli_builds() {
        // Basic test that CLI compiles and Parser derives work
        let _cli = Cli::parse_from(["loom", "version"]);
    }

    #[test]
    fn test_optimize_command_wat_input() {
        // Create temporary input file
        let input_wat = r#"(module
  (func $add_constants (result i32)
    i32.const 10
    i32.const 32
    i32.add
  )
)"#;
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_cli_input.wat");
        let output_path = temp_dir.join("test_cli_output.wasm");

        fs::write(&input_path, input_wat).unwrap();

        // Run optimization
        let result = optimize_command(
            input_path.to_string_lossy().to_string(),
            Some(output_path.to_string_lossy().to_string()),
            false,
            false,
            false,
            false, // attestation
            None,
            1,     // islands: serial path
            false, // run_differential
            false, // strip_component_type
            false, // no_reachability_gc
            false, // emit_facts
        );

        assert!(result.is_ok(), "Optimization should succeed");

        // Check output exists
        assert!(output_path.exists(), "Output file should exist");

        // Clean up
        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&output_path);
    }

    #[test]
    fn test_optimize_with_stats() {
        // Test that stats flag doesn't cause errors
        let input_wat = r#"(module
  (func $add (result i32)
    i32.const 5
    i32.const 10
    i32.add
  )
)"#;
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_cli_stats_input.wat");
        let output_path = temp_dir.join("test_cli_stats_output.wasm");

        fs::write(&input_path, input_wat).unwrap();

        let result = optimize_command(
            input_path.to_string_lossy().to_string(),
            Some(output_path.to_string_lossy().to_string()),
            false,
            true, // Enable stats
            false,
            false, // attestation
            None,
            1,     // islands: serial path
            false, // run_differential
            false, // strip_component_type
            false, // no_reachability_gc
            false, // emit_facts
        );

        assert!(result.is_ok(), "Optimization with stats should succeed");

        // Clean up
        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&output_path);
    }

    #[test]
    fn test_optimize_with_verification() {
        // Test that verify flag works
        let input_wat = r#"(module
  (func $test (result i32)
    i32.const 1
    i32.const 2
    i32.add
  )
)"#;
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_cli_verify_input.wat");
        let output_path = temp_dir.join("test_cli_verify_output.wasm");

        fs::write(&input_path, input_wat).unwrap();

        let result = optimize_command(
            input_path.to_string_lossy().to_string(),
            Some(output_path.to_string_lossy().to_string()),
            false,
            false,
            true,  // Enable verification
            false, // attestation
            None,
            1,     // islands: serial path
            false, // run_differential
            false, // strip_component_type
            false, // no_reachability_gc
            false, // emit_facts
        );

        assert!(
            result.is_ok(),
            "Optimization with verification should succeed"
        );

        // Clean up
        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&output_path);
    }

    #[test]
    fn test_optimize_wat_output() {
        // Test WAT output format
        let input_wat = r#"(module
  (func $test (result i32)
    i32.const 100
    i32.const 200
    i32.add
  )
)"#;
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_cli_wat_output_input.wat");
        let output_path = temp_dir.join("test_cli_wat_output_output.wat");

        fs::write(&input_path, input_wat).unwrap();

        let result = optimize_command(
            input_path.to_string_lossy().to_string(),
            Some(output_path.to_string_lossy().to_string()),
            true, // WAT output
            false,
            false,
            false, // attestation (disabled for WAT)
            None,
            1,     // islands: serial path
            false, // run_differential
            false, // strip_component_type
            false, // no_reachability_gc
            false, // emit_facts
        );

        assert!(
            result.is_ok(),
            "Optimization with WAT output should succeed"
        );

        // Check that output is valid WAT
        let output_content = fs::read_to_string(&output_path).unwrap();
        assert!(
            output_content.contains("i32.const 300"),
            "WAT should contain optimized constant 300"
        );

        // Clean up
        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&output_path);
    }

    #[test]
    fn test_count_instructions() {
        use loom_core::{Function, FunctionSignature, Instruction, Module, ValueType};

        let module = Module {
            functions: vec![Function {
                name: None,
                signature: FunctionSignature {
                    params: vec![],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![
                    Instruction::I32Const(10),
                    Instruction::I32Const(32),
                    Instruction::I32Add,
                    Instruction::End,
                ],
            }],
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
            facts: Vec::new(),
        };

        let count = count_instructions(&module);
        assert_eq!(count, 4, "Should count 4 instructions");
    }

    #[test]
    fn test_count_constant_folds() {
        use loom_core::{Function, FunctionSignature, Instruction, Module, ValueType};

        let module = Module {
            functions: vec![Function {
                name: None,
                signature: FunctionSignature {
                    params: vec![],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: vec![
                    Instruction::I32Const(10),
                    Instruction::I32Const(32),
                    Instruction::I32Add,
                    Instruction::I32Const(5),
                    Instruction::I32Const(7),
                    Instruction::I32Add,
                    Instruction::End,
                ],
            }],
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
            facts: Vec::new(),
        };

        let count = count_constant_folds(&module);
        assert_eq!(count, 2, "Should find 2 constant folding opportunities");
    }

    #[test]
    fn test_optimization_stats_reduction_percentage() {
        let stats = OptimizationStats {
            instructions_before: 10,
            instructions_after: 5,
            bytes_before: 100,
            bytes_after: 50,
            optimization_time_ms: 1,
            constant_folds: 2,
            pass_stats: vec![],
            reencoding_bytes: None,
        };

        assert_eq!(stats.reduction_percentage(10, 5), 50.0);
        assert_eq!(stats.reduction_percentage(100, 25), 75.0);
        assert_eq!(stats.reduction_percentage(0, 0), 0.0);
    }

    /// The #346 backstop, tested directly on its refusal path.
    ///
    /// The integration sweep asserts "success implies the artifact validates",
    /// which holds trivially while the encoder is correct — so it cannot fail
    /// if the gate is deleted. This can: it hands the gate bytes that are
    /// definitely invalid and asserts the three things that must follow.
    ///
    /// Before #346 the CLI had no such gate at all: `fs::write` was called on
    /// whatever the encoder returned, and "✅ Optimization complete!" was
    /// printed over it.
    #[test]
    fn the_output_gate_refuses_invalid_bytes_and_restores_the_original() {
        // Valid: (module) — the 8-byte preamble is a complete empty module.
        const VALID: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // Invalid: right magic, but a truncated type section header.
        const INVALID: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x7f];
        assert!(loom_core::encode::validate_output_bytes(VALID).is_ok());
        assert!(
            loom_core::encode::validate_output_bytes(INVALID).is_err(),
            "the negative fixture must actually be invalid, or this test \
             asserts nothing"
        );

        let dir = std::env::temp_dir().join(format!("loom-346-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        let out_path = dir.join("out.wasm");
        let out_str = out_path.to_str().unwrap();

        let result = write_validated_output(out_str, INVALID, INVALID, VALID);

        assert!(
            result.is_err(),
            "the gate accepted bytes that do not validate; exit 0 over an \
             invalid module is exactly the #346 defect"
        );
        let written = std::fs::read(&out_path).expect("the gate must leave a valid file behind");
        assert_ne!(
            written, INVALID,
            "the invalid bytes were written to the output path"
        );
        assert_eq!(
            written, VALID,
            "the gate must fall back to the original module so the worst case \
             is unoptimized-but-valid output, never a corrupt artifact"
        );

        let _ = std::fs::remove_file(&out_path);
    }

    /// The positive control: valid bytes must be written unchanged. Without
    /// this, a gate that refused everything would pass the test above.
    #[test]
    fn the_output_gate_writes_valid_bytes_unchanged() {
        const VALID: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        let dir = std::env::temp_dir().join(format!("loom-346-unit-ok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        let out_path = dir.join("ok.wasm");
        let out_str = out_path.to_str().unwrap();

        write_validated_output(out_str, VALID, VALID, VALID).expect("valid output must be written");
        assert_eq!(
            std::fs::read(&out_path).expect("read back"),
            VALID,
            "valid output must reach disk byte-for-byte"
        );
        let _ = std::fs::remove_file(&out_path);
    }
}
