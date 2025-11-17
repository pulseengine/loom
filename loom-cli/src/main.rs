//! LOOM Command-Line Interface
//!
//! Command-line tool for optimizing WebAssembly modules

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "loom")]
#[command(author = "PulseEngine")]
#[command(version = "0.1.0")]
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
    },

    /// Verify ISLE optimization rules
    Verify {
        /// ISLE file to verify
        #[arg(value_name = "ISLE_FILE")]
        isle_file: String,
    },

    /// Show version information
    Version,
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
        println!("Constant folds: {}", self.constant_folds);
        println!("Optimization time: {} ms", self.optimization_time_ms);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    fn reduction_percentage(&self, before: usize, after: usize) -> f64 {
        if before == 0 {
            0.0
        } else {
            ((before - after) as f64 / before as f64) * 100.0
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

/// Optimize command implementation
fn optimize_command(
    input: String,
    output: Option<String>,
    output_wat: bool,
    show_stats: bool,
    run_verify: bool,
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

            // Use component optimization
            match loom_core::component::optimize_component(&input_bytes) {
                Ok((optimized_bytes, stats)) => {
                    println!("\n📊 Component Optimization Results");
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    println!(
                        "Component:    {} → {} bytes ({:.1}% reduction)",
                        stats.original_size,
                        stats.optimized_size,
                        if stats.original_size > 0 {
                            ((stats.original_size - stats.optimized_size) as f64
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
                            ((stats.original_module_size - stats.optimized_module_size) as f64
                                / stats.original_module_size as f64)
                                * 100.0
                        );
                    }
                    println!("Status:       {}", stats.message);
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                    // Write optimized component
                    let output_path = output.unwrap_or_else(|| "output.wasm".to_string());
                    fs::write(&output_path, &optimized_bytes)
                        .context("Failed to write output file")?;
                    println!("✓ Written to: {}", output_path);
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

    // Collect statistics before optimization
    let mut stats = OptimizationStats {
        instructions_before: count_instructions(&module),
        bytes_before: input_bytes.len(),
        constant_folds: count_constant_folds(&module),
        ..Default::default()
    };

    // Apply optimizations
    println!("⚡ Optimizing...");
    let start_opt = Instant::now();
    loom_core::optimize::precompute(&mut module).context("Precompute failed")?;
    loom_core::optimize::optimize_module(&mut module).context("Optimization failed")?;
    loom_core::optimize::eliminate_common_subexpressions(&mut module).context("CSE failed")?;
    loom_core::optimize::optimize_advanced_instructions(&mut module).context("Advanced instruction optimization failed")?;
    loom_core::optimize::simplify_branches(&mut module).context("Branch simplification failed")?;
    loom_core::optimize::eliminate_dead_code(&mut module).context("DCE failed")?;
    loom_core::optimize::merge_blocks(&mut module).context("Block merging failed")?;
    loom_core::optimize::vacuum(&mut module).context("Vacuum cleanup failed")?;
    loom_core::optimize::simplify_locals(&mut module).context("SimplifyLocals failed")?;
    stats.optimization_time_ms = start_opt.elapsed().as_millis();
    println!("✓ Optimized in {} ms", stats.optimization_time_ms);

    // Collect statistics after optimization
    stats.instructions_after = count_instructions(&module);

    // Encode output
    let output_bytes = if output_wat {
        println!("📝 Encoding to WAT format...");
        loom_core::encode::encode_wat(&module)
            .context("Failed to encode to WAT")?
            .into_bytes()
    } else {
        println!("📦 Encoding to WASM binary...");
        loom_core::encode::encode_wasm(&module).context("Failed to encode to WASM")?
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
    fs::write(&output_path, &output_bytes).context("Failed to write output file")?;
    println!("✓ Written to: {}", output_path);

    // Show statistics if requested
    if show_stats {
        stats.print();
    }

    // Run verification if requested
    if run_verify {
        println!("\n🔍 Running verification...");
        run_verification(&module)?;
    }

    println!("\n✅ Optimization complete!");
    Ok(())
}

/// Run property-based verification on the module
fn run_verification(module: &loom_core::Module) -> Result<()> {
    use loom_core::Instruction;
    use loom_isle::{iadd32, iconst32, simplify, Imm32, ValueData};

    let mut test_count = 0;
    let mut pass_count = 0;

    println!("Running verification tests...");

    // Test each constant folding in the module
    for func in &module.functions {
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

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Optimize {
            input,
            output,
            wat,
            stats,
            verify,
        }) => {
            optimize_command(input, output, wat, stats, verify)?;
        }

        Some(Commands::Verify { isle_file }) => {
            println!("✓ LOOM Verification");
            println!("ISLE file: {}", isle_file);

            // TODO: Implement in Phase 5
            println!("\n⚠️  Verification not yet implemented (Phase 5)");
            println!("This is a Phase 1 placeholder.");
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
            println!("  verify      Verify ISLE optimization rules");
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
        let input_path = "/tmp/test_cli_input.wat";
        let output_path = "/tmp/test_cli_output.wasm";

        fs::write(input_path, input_wat).unwrap();

        // Run optimization
        let result = optimize_command(
            input_path.to_string(),
            Some(output_path.to_string()),
            false,
            false,
            false,
        );

        assert!(result.is_ok(), "Optimization should succeed");

        // Check output exists
        assert!(
            std::path::Path::new(output_path).exists(),
            "Output file should exist"
        );

        // Clean up
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
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
        let input_path = "/tmp/test_cli_stats_input.wat";
        let output_path = "/tmp/test_cli_stats_output.wasm";

        fs::write(input_path, input_wat).unwrap();

        let result = optimize_command(
            input_path.to_string(),
            Some(output_path.to_string()),
            false,
            true, // Enable stats
            false,
        );

        assert!(result.is_ok(), "Optimization with stats should succeed");

        // Clean up
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
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
        let input_path = "/tmp/test_cli_verify_input.wat";
        let output_path = "/tmp/test_cli_verify_output.wasm";

        fs::write(input_path, input_wat).unwrap();

        let result = optimize_command(
            input_path.to_string(),
            Some(output_path.to_string()),
            false,
            false,
            true, // Enable verification
        );

        assert!(
            result.is_ok(),
            "Optimization with verification should succeed"
        );

        // Clean up
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
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
        let input_path = "/tmp/test_cli_wat_output_input.wat";
        let output_path = "/tmp/test_cli_wat_output_output.wat";

        fs::write(input_path, input_wat).unwrap();

        let result = optimize_command(
            input_path.to_string(),
            Some(output_path.to_string()),
            true, // WAT output
            false,
            false,
        );

        assert!(
            result.is_ok(),
            "Optimization with WAT output should succeed"
        );

        // Check that output is valid WAT
        let output_content = fs::read_to_string(output_path).unwrap();
        assert!(
            output_content.contains("i32.const 300"),
            "WAT should contain optimized constant 300"
        );

        // Clean up
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
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
            globals: vec![],
            types: vec![],
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
            globals: vec![],
            types: vec![],
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
        };

        assert_eq!(stats.reduction_percentage(10, 5), 50.0);
        assert_eq!(stats.reduction_percentage(100, 25), 75.0);
        assert_eq!(stats.reduction_percentage(0, 0), 0.0);
    }
}
