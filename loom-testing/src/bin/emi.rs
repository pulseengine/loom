//! EMI (Equivalence Modulo Inputs) testing binary
//!
//! This tool finds miscompilation bugs by mutating dead code regions
//! and verifying that outputs remain unchanged after optimization.

use anyhow::{Context, Result};
use loom_testing::emi::{analyze_dead_code, emi_test, EmiConfig};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "analyze" => {
            if args.len() < 3 {
                eprintln!("Usage: {} analyze <file.wasm|file.wat>", args[0]);
                std::process::exit(1);
            }
            analyze_command(&args[2])
        }
        "test" => {
            if args.len() < 3 {
                eprintln!(
                    "Usage: {} test <file.wasm|file.wat> [--iterations N]",
                    args[0]
                );
                std::process::exit(1);
            }
            let iterations = parse_iterations(&args);
            test_command(&args[2], iterations)
        }
        "corpus" => {
            if args.len() < 3 {
                eprintln!("Usage: {} corpus <directory> [--iterations N]", args[0]);
                std::process::exit(1);
            }
            let iterations = parse_iterations(&args);
            corpus_command(&args[2], iterations)
        }
        "--help" | "-h" => {
            print_usage(&args[0]);
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage(&args[0]);
            std::process::exit(1);
        }
    }
}

fn print_usage(program: &str) {
    println!("EMI Testing for LOOM WebAssembly Optimizer");
    println!();
    println!("USAGE:");
    println!("    {} <COMMAND> [OPTIONS]", program);
    println!();
    println!("COMMANDS:");
    println!("    analyze <file>       Analyze a file for dead code regions");
    println!("    test <file>          Run EMI testing on a single file");
    println!("    corpus <directory>   Run EMI testing on all .wasm/.wat files");
    println!();
    println!("OPTIONS:");
    println!("    --iterations N       Number of mutation variants to test (default: 100)");
    println!();
    println!("EXAMPLES:");
    println!(
        "    {} analyze tests/fixtures/branch_simplification_test.wat",
        program
    );
    println!("    {} test module.wasm --iterations 200", program);
    println!("    {} corpus tests/fixtures/", program);
}

fn parse_iterations(args: &[String]) -> usize {
    for (i, arg) in args.iter().enumerate() {
        if arg == "--iterations" && i + 1 < args.len() {
            return args[i + 1].parse().unwrap_or(100);
        }
    }
    100
}

fn load_wasm(path: &str) -> Result<Vec<u8>> {
    let path = PathBuf::from(path);
    let content =
        std::fs::read(&path).with_context(|| format!("Failed to read file: {}", path.display()))?;

    // If it's a .wat file, parse it to wasm
    if path.extension().is_some_and(|ext| ext == "wat") {
        wat::parse_bytes(&content)
            .map(|cow| cow.into_owned())
            .with_context(|| format!("Failed to parse WAT: {}", path.display()))
    } else {
        Ok(content)
    }
}

fn analyze_command(path: &str) -> Result<()> {
    println!("Analyzing dead code regions in: {}", path);
    println!();

    let wasm = load_wasm(path)?;
    let regions = analyze_dead_code(&wasm)?;

    if regions.is_empty() {
        println!("No dead code regions found.");
        return Ok(());
    }

    println!("Found {} dead code region(s):", regions.len());
    println!();

    for (i, region) in regions.iter().enumerate() {
        println!("Region {}:", i + 1);
        println!("  Function index: {}", region.func_idx);
        println!(
            "  Byte range: {} - {}",
            region.start_offset, region.end_offset
        );
        println!("  Type: {:?}", region.region_type);
        println!("  Description: {}", region.description);
        println!();
    }

    Ok(())
}

fn test_command(path: &str, iterations: usize) -> Result<()> {
    println!("EMI Testing: {}", path);
    println!("Iterations: {}", iterations);
    println!();

    let wasm = load_wasm(path)?;

    let config = EmiConfig {
        iterations,
        ..EmiConfig::default()
    };

    let result = emi_test(&wasm, config)?;

    println!("Results:");
    println!("  Dead regions found: {}", result.dead_regions_found);
    println!("  Variants tested: {}", result.variants_tested);
    println!("  Bugs found: {}", result.bugs_found.len());
    println!();

    if result.bugs_found.is_empty() {
        println!("No bugs found.");
    } else {
        println!("BUGS DETECTED:");
        for (i, bug) in result.bugs_found.iter().enumerate() {
            println!();
            println!("Bug {}:", i + 1);
            println!("  Type: {}", bug.bug_type.short_description());
            println!("  Mutation: {}", bug.mutation_strategy.name());
            println!(
                "  Region: func {} @ {}-{}",
                bug.dead_region.func_idx, bug.dead_region.start_offset, bug.dead_region.end_offset
            );
            if let Some(ref expected) = bug.expected {
                println!("  Expected: {}", expected);
            }
            if let Some(ref actual) = bug.actual {
                println!("  Actual: {}", actual);
            }
        }
    }

    if result.has_bugs() {
        std::process::exit(1);
    }

    Ok(())
}

fn corpus_command(dir: &str, iterations: usize) -> Result<()> {
    println!("EMI Testing corpus: {}", dir);
    println!("Iterations per file: {}", iterations);
    println!();

    let dir_path = PathBuf::from(dir);
    if !dir_path.exists() {
        anyhow::bail!("Directory not found: {}", dir);
    }

    // Find all .wasm and .wat files
    let patterns = [format!("{}/**/*.wasm", dir), format!("{}/**/*.wat", dir)];

    let mut files: Vec<PathBuf> = Vec::new();
    for pattern in &patterns {
        for path in glob::glob(pattern)?.flatten() {
            files.push(path);
        }
    }

    if files.is_empty() {
        println!("No .wasm or .wat files found in {}", dir);
        return Ok(());
    }

    println!("Found {} files to test", files.len());
    println!();

    let mut total_bugs = 0;
    let mut total_regions = 0;
    let mut total_variants = 0;

    for (i, path) in files.iter().enumerate() {
        let filename = path.file_name().unwrap().to_string_lossy();
        print!("[{:3}/{}] {:40} ", i + 1, files.len(), filename);

        let wasm = match load_wasm(path.to_str().unwrap()) {
            Ok(w) => w,
            Err(e) => {
                println!("SKIP ({})", e);
                continue;
            }
        };

        let config = EmiConfig {
            iterations,
            ..EmiConfig::default()
        };

        match emi_test(&wasm, config) {
            Ok(result) => {
                total_regions += result.dead_regions_found;
                total_variants += result.variants_tested;
                total_bugs += result.bugs_found.len();

                if result.bugs_found.is_empty() {
                    println!(
                        "OK ({} regions, {} variants)",
                        result.dead_regions_found, result.variants_tested
                    );
                } else {
                    println!("BUGS FOUND: {}", result.bugs_found.len());
                }
            }
            Err(e) => {
                println!("ERROR: {}", e);
            }
        }
    }

    println!();
    println!("Summary:");
    println!("  Files tested: {}", files.len());
    println!("  Dead regions found: {}", total_regions);
    println!("  Variants tested: {}", total_variants);
    println!("  Total bugs: {}", total_bugs);

    if total_bugs > 0 {
        std::process::exit(1);
    }

    Ok(())
}
