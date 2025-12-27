//! Demonstration of Z3-based optimization rule verification
//!
//! This shows how LOOM uses Z3 to PROVE optimization rules are correct
//! for ALL possible inputs, rather than just testing specific cases.
//!
//! Run with: cargo run --example verify_rules_demo --features verification

use loom_core::verify_rules::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       LOOM Z3 Optimization Rule Verification                     ║");
    println!("║       Proving Correctness for ALL Inputs                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Each proof below covers ALL possible values (2³² for i32, 2⁶⁴ for i64)");
    println!("This is equivalent to running billions of test cases instantly.\n");

    // Run COMPLETE verification suite (includes control flow + composition + ISLE rules)
    let results = verify_all_with_isle();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                         VERIFICATION RESULTS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Group proofs by category
    let mut strength_reduction = Vec::new();
    let mut algebraic = Vec::new();
    let mut bitwise = Vec::new();
    let mut comparison = Vec::new();
    let mut control_flow = Vec::new();
    let mut composition = Vec::new();
    let mut other = Vec::new();

    for (name, proof) in &results.proofs {
        if name.contains("strength_reduction") {
            strength_reduction.push((name, proof));
        } else if name.contains("add_zero")
            || name.contains("mul_zero")
            || name.contains("mul_one")
            || name.contains("sub_self")
        {
            algebraic.push((name, proof));
        } else if name.contains("xor") || name.contains("and") || name.contains("or") {
            bitwise.push((name, proof));
        } else if name.contains("eq_self") || name.contains("ne_self") {
            comparison.push((name, proof));
        } else if name.contains("if_true") || name.contains("if_false") || name.contains("select") {
            control_flow.push((name, proof));
        } else if name.contains("composition") || name.contains("pipeline") {
            composition.push((name, proof));
        } else {
            other.push((name, proof));
        }
    }

    // ISLE rules category
    let mut isle_rules = Vec::new();
    for (name, proof) in &results.proofs {
        if name.starts_with("isle_") {
            isle_rules.push((name, proof));
        }
    }
    // Remove ISLE rules from other
    let other: Vec<_> = other
        .into_iter()
        .filter(|(name, _)| !name.starts_with("isle_"))
        .collect();

    print_category(
        "STRENGTH REDUCTION (mul/div/rem → shift/and)",
        &strength_reduction,
    );
    print_category("ALGEBRAIC IDENTITIES", &algebraic);
    print_category("BITWISE IDENTITIES", &bitwise);
    print_category("COMPARISON IDENTITIES", &comparison);
    print_category("CONTROL FLOW (branch/select simplification)", &control_flow);
    print_category("PASS COMPOSITION (pipeline correctness)", &composition);
    print_category("ISLE RULES (DSL rewrite verification)", &isle_rules);
    print_category("OTHER PROOFS", &other);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                           SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let status = if results.failed_rules == 0 {
        "✓"
    } else {
        "✗"
    };
    println!("  {} {}", status, results.summary());

    let total_cases_i32 = results
        .proofs
        .iter()
        .filter(|(n, _)| n.contains("i32"))
        .count() as u128
        * (1u128 << 32);
    let total_cases_i64 = results
        .proofs
        .iter()
        .filter(|(n, _)| n.contains("i64"))
        .count() as u128
        * (1u128 << 64);

    println!("\n  Equivalent test coverage:");
    println!(
        "    - i32 rules: {} proofs × 2³² values = {} cases",
        results
            .proofs
            .iter()
            .filter(|(n, _)| n.contains("i32"))
            .count(),
        format_big_number(total_cases_i32)
    );
    println!(
        "    - i64 rules: {} proofs × 2⁶⁴ values = {} cases",
        results
            .proofs
            .iter()
            .filter(|(n, _)| n.contains("i64"))
            .count(),
        format_big_number(total_cases_i64)
    );

    println!(
        "\n  Total verification time: {}ms",
        results
            .proofs
            .values()
            .map(|p| p.verification_time_ms)
            .sum::<u64>()
    );

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  Z3 proves these rules correct for all inputs, but rule proofs");
    println!("  don't cover implementation bugs. See verify_e2e.rs for gaps.");
    println!("═══════════════════════════════════════════════════════════════════\n");
}

fn print_category(title: &str, proofs: &[(&String, &RuleProof)]) {
    if proofs.is_empty() {
        return;
    }

    println!("┌─ {} ─", title);
    for (_name, proof) in proofs {
        let status = if proof.proven { "✓" } else { "✗" };
        println!(
            "│  {} {} ({}ms)",
            status, proof.details, proof.verification_time_ms
        );
    }
    println!("└─\n");
}

fn format_big_number(n: u128) -> String {
    if n >= 1_000_000_000_000_000_000 {
        format!("{:.1}×10¹⁸+", n as f64 / 1e18)
    } else if n >= 1_000_000_000_000 {
        format!("{:.1} trillion", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.1} billion", n as f64 / 1e9)
    } else {
        format!("{}", n)
    }
}
