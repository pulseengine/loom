//! Differential testing binary - compare LOOM vs wasm-opt
//!
//! This tool runs LOOM and wasm-opt on a corpus of WASM files and
//! compares the results to validate correctness and find optimization gaps.

use anyhow::Result;
use loom_testing::{DifferentialTester, TestResult};
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("🔬 LOOM Differential Testing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create tester
    let tester = match DifferentialTester::new() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            eprintln!("\n💡 Make sure:");
            eprintln!("   1. LOOM is built: cargo build --release");
            eprintln!("   2. wasm-opt is installed: brew install binaryen");
            eprintln!("   3. Both binaries are in PATH");
            std::process::exit(1);
        }
    };

    // Get test corpus
    let corpus_dir = PathBuf::from("tests/corpus");
    if !corpus_dir.exists() {
        eprintln!("❌ Corpus directory not found: {}", corpus_dir.display());
        eprintln!("\n💡 Run: bash scripts/collect_corpus.sh");
        std::process::exit(1);
    }

    // Find all WASM files
    let pattern = format!("{}/**/*.wasm", corpus_dir.display());
    let wasm_files: Vec<_> = glob::glob(&pattern)?.filter_map(|p| p.ok()).collect();

    if wasm_files.is_empty() {
        eprintln!("❌ No WASM files found in {}", corpus_dir.display());
        eprintln!("\n💡 Run: bash scripts/collect_corpus.sh");
        std::process::exit(1);
    }

    println!("📦 Testing {} WASM files...\n", wasm_files.len());

    // Run tests
    let mut results: Vec<(PathBuf, TestResult)> = Vec::new();
    let mut loom_wins = 0;
    let mut wasm_opt_wins = 0;
    let mut ties = 0;
    let mut errors = 0;

    for (i, path) in wasm_files.iter().enumerate() {
        let filename = path.file_name().unwrap().to_string_lossy();
        print!("[{:3}/{}] {:40} ", i + 1, wasm_files.len(), filename);

        let input = match std::fs::read(path) {
            Ok(data) => data,
            Err(e) => {
                println!("❌ Read error: {}", e);
                errors += 1;
                continue;
            }
        };

        let result = match tester.test(&input) {
            Ok(r) => r,
            Err(e) => {
                println!("❌ Test error: {}", e);
                errors += 1;
                continue;
            }
        };

        match result.winner() {
            "LOOM" => {
                loom_wins += 1;
                println!(
                    "✅ LOOM ({} bytes, {:.1}% reduction)",
                    result.loom_size,
                    result.loom_reduction_pct()
                );
            }
            "wasm-opt" => {
                wasm_opt_wins += 1;
                println!(
                    "⚠️  wasm-opt ({} vs {} bytes, {} byte gap)",
                    result.wasm_opt_size,
                    result.loom_size,
                    result.size_delta()
                );
            }
            "TIE" => {
                ties += 1;
                println!("🤝 Tie ({} bytes)", result.loom_size);
            }
            "LOOM_INVALID" => {
                errors += 1;
                println!("❌ LOOM produced invalid WASM");
            }
            "WASM_OPT_INVALID" => {
                loom_wins += 1; // Count as LOOM win if wasm-opt failed
                println!("⚡ wasm-opt failed, LOOM succeeded");
            }
            other => {
                errors += 1;
                println!("❌ {}", other);
            }
        }

        results.push((path.clone(), result));
    }

    // Print summary
    let total = wasm_files.len();
    let success_count = loom_wins + ties;
    let success_rate = if total > 0 {
        success_count as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    println!("\n📊 Differential Testing Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Total tests:     {}", total);
    println!(
        "LOOM wins:       {} ({:.1}%)",
        loom_wins,
        loom_wins as f64 / total as f64 * 100.0
    );
    println!(
        "wasm-opt wins:   {} ({:.1}%)",
        wasm_opt_wins,
        wasm_opt_wins as f64 / total as f64 * 100.0
    );
    println!(
        "Ties:            {} ({:.1}%)",
        ties,
        ties as f64 / total as f64 * 100.0
    );
    println!("Errors:          {}", errors);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n🎯 LOOM success rate: {:.1}%", success_rate);

    // Calculate average reductions
    let valid_results: Vec<_> = results
        .iter()
        .filter(|(_, r)| r.loom_valid && r.wasm_opt_valid)
        .collect();

    if !valid_results.is_empty() {
        let avg_loom_reduction: f64 = valid_results
            .iter()
            .map(|(_, r)| r.loom_reduction_pct())
            .sum::<f64>()
            / valid_results.len() as f64;

        let avg_wasm_opt_reduction: f64 = valid_results
            .iter()
            .map(|(_, r)| r.wasm_opt_reduction_pct())
            .sum::<f64>()
            / valid_results.len() as f64;

        println!("\n📉 Average Size Reductions");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("LOOM:        {:.1}%", avg_loom_reduction);
        println!("wasm-opt:    {:.1}%", avg_wasm_opt_reduction);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    // Show worst cases (where wasm-opt won by most)
    if wasm_opt_wins > 0 {
        println!("\n⚠️  Top 5 Cases Where wasm-opt Won:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let mut wasm_opt_gaps: Vec<_> = results.iter().filter(|(_, r)| r.wasm_opt_wins()).collect();

        wasm_opt_gaps.sort_by_key(|(_, r)| r.size_delta());

        for (path, result) in wasm_opt_gaps.iter().take(5) {
            let filename = path.file_name().unwrap().to_string_lossy();
            println!(
                "  {} - {} byte gap ({} vs {})",
                filename,
                result.size_delta(),
                result.loom_size,
                result.wasm_opt_size
            );
        }
    }

    // Show best cases (where LOOM won by most)
    if loom_wins > 0 {
        println!("\n✅ Top 5 Cases Where LOOM Won:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let mut loom_gaps: Vec<_> = results.iter().filter(|(_, r)| r.loom_wins()).collect();

        loom_gaps.sort_by_key(|(_, r)| r.size_delta());

        for (path, result) in loom_gaps.iter().take(5) {
            let filename = path.file_name().unwrap().to_string_lossy();
            println!(
                "  {} - {} byte advantage ({} vs {})",
                filename,
                -result.size_delta(),
                result.loom_size,
                result.wasm_opt_size
            );
        }
    }

    println!("\n✅ Differential testing complete!");

    Ok(())
}
