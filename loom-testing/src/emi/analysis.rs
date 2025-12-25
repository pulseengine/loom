// Allow index-based loops - needed for accessing operators[j] by index
#![allow(clippy::needless_range_loop)]

//! Static analysis for dead code detection
//!
//! This module identifies dead code regions in WebAssembly modules
//! using static analysis techniques. It finds:
//!
//! 1. Dead branches in if/else with constant conditions
//! 2. Code after unconditional branches (br, return, unreachable)
//! 3. br_if with constant conditions

use super::types::{DeadRegion, DeadRegionType};
use anyhow::{Context, Result};
use wasmparser::{FunctionBody, Operator, Parser, Payload};

/// Analyze a WebAssembly module for dead code regions
///
/// This performs static analysis to find code that is provably dead:
/// - Branches with constant conditions (if/else, br_if)
/// - Code after terminators (return, br, unreachable)
pub fn analyze_dead_code(wasm_bytes: &[u8]) -> Result<Vec<DeadRegion>> {
    let mut dead_regions = Vec::new();
    let mut func_idx = 0u32;

    let parser = Parser::new(0);

    for payload in parser.parse_all(wasm_bytes) {
        let payload = payload.context("Failed to parse Wasm payload")?;

        if let Payload::CodeSectionEntry(body) = payload {
            // Analyze this function for dead code
            let regions = analyze_function_body(func_idx, &body)?;
            dead_regions.extend(regions);
            func_idx += 1;
        }
    }

    Ok(dead_regions)
}

/// Analyze a single function body for dead code regions
fn analyze_function_body(func_idx: u32, body: &FunctionBody) -> Result<Vec<DeadRegion>> {
    let mut dead_regions = Vec::new();

    // Parse operators and track state
    let mut reader = body.get_operators_reader()?;
    let mut operators: Vec<(Operator, usize)> = Vec::new();

    // Collect all operators with their offsets
    while !reader.eof() {
        let offset = reader.original_position();
        let op = reader.read()?;
        operators.push((op, offset));
    }

    // Now analyze the operator sequence for dead code patterns
    let mut i = 0;
    while i < operators.len() {
        let (ref op, _offset) = operators[i];

        match op {
            // Pattern 1: i32.const followed by if
            // If const is 0, then branch is dead
            // If const is non-zero, else branch is dead
            Operator::I32Const { value } => {
                if i + 1 < operators.len() {
                    if let (Operator::If { .. }, _) = &operators[i + 1] {
                        // Found constant condition if
                        if let Some(region) =
                            find_dead_if_branch(&operators, i + 1, *value != 0, func_idx)
                        {
                            dead_regions.push(region);
                        }
                    } else if let (Operator::BrIf { .. }, _) = &operators[i + 1] {
                        // Found constant condition br_if
                        if *value != 0 {
                            // br_if always taken - code after is dead
                            if let Some(region) = find_dead_after_br_if(&operators, i + 1, func_idx)
                            {
                                dead_regions.push(region);
                            }
                        }
                        // If value == 0, br_if never taken - the br_if itself could be removed
                        // but that's an optimization, not dead code detection
                    }
                }
            }

            // Pattern 2: Code after unconditional terminators
            Operator::Return | Operator::Unreachable => {
                if let Some(region) = find_dead_after_terminator(&operators, i, func_idx, op) {
                    dead_regions.push(region);
                }
            }

            Operator::Br { .. } => {
                // br is unconditional - code after it (until end of block) is dead
                if let Some(region) = find_dead_after_br(&operators, i, func_idx) {
                    dead_regions.push(region);
                }
            }

            _ => {}
        }

        i += 1;
    }

    Ok(dead_regions)
}

/// Find the dead branch of an if/else with constant condition
fn find_dead_if_branch(
    operators: &[(Operator, usize)],
    if_idx: usize,
    condition_is_true: bool,
    func_idx: u32,
) -> Option<DeadRegion> {
    // We need to find the structure: if ... else ... end
    // If condition is true, else branch is dead
    // If condition is false, then branch is dead

    let if_offset = operators[if_idx].1;

    // Track nesting to find matching else/end
    let mut depth = 1;
    let mut else_offset: Option<usize> = None;
    let mut end_offset: Option<usize> = None;
    let then_start = if_idx + 1;

    for j in (if_idx + 1)..operators.len() {
        match &operators[j].0 {
            Operator::If { .. } | Operator::Block { .. } | Operator::Loop { .. } => {
                depth += 1;
            }
            Operator::Else => {
                if depth == 1 {
                    else_offset = Some(operators[j].1);
                }
            }
            Operator::End => {
                depth -= 1;
                if depth == 0 {
                    end_offset = Some(operators[j].1);
                    break;
                }
            }
            _ => {}
        }
    }

    let end_offset = end_offset?;

    if condition_is_true {
        // Else branch is dead
        else_offset.map(|else_off| DeadRegion {
            func_idx,
            start_offset: else_off,
            end_offset,
            region_type: DeadRegionType::ConstantBranchIf {
                is_then_dead: false,
            },
            description: format!(
                "Else branch is dead (condition is always true) at offset {}",
                if_offset
            ),
        })
    } else {
        // Then branch is dead (from after if to else or end)
        let dead_end = else_offset.unwrap_or(end_offset);
        Some(DeadRegion {
            func_idx,
            start_offset: operators[then_start].1,
            end_offset: dead_end,
            region_type: DeadRegionType::ConstantBranchIf { is_then_dead: true },
            description: format!(
                "Then branch is dead (condition is always false) at offset {}",
                if_offset
            ),
        })
    }
}

/// Find dead code after br_if that's always taken
fn find_dead_after_br_if(
    operators: &[(Operator, usize)],
    br_if_idx: usize,
    func_idx: u32,
) -> Option<DeadRegion> {
    let br_if_offset = operators[br_if_idx].1;

    // Find the end of the current block
    let mut depth = 1; // We're inside some block
    let mut end_idx = None;

    for j in (br_if_idx + 1)..operators.len() {
        match &operators[j].0 {
            Operator::If { .. } | Operator::Block { .. } | Operator::Loop { .. } => {
                depth += 1;
            }
            Operator::End => {
                depth -= 1;
                if depth == 0 {
                    end_idx = Some(j);
                    break;
                }
            }
            _ => {}
        }
    }

    // Check if there's code between br_if and end
    if let Some(end_j) = end_idx {
        if br_if_idx + 1 < end_j {
            // There's code after br_if before the end - it's dead!
            return Some(DeadRegion {
                func_idx,
                start_offset: operators[br_if_idx + 1].1,
                end_offset: operators[end_j].1,
                region_type: DeadRegionType::ConstantBrIf { always_taken: true },
                description: format!(
                    "Code after br_if is dead (branch always taken) at offset {}",
                    br_if_offset
                ),
            });
        }
    }

    None
}

/// Find dead code after a terminator (return, unreachable)
fn find_dead_after_terminator(
    operators: &[(Operator, usize)],
    term_idx: usize,
    func_idx: u32,
    terminator: &Operator,
) -> Option<DeadRegion> {
    let term_offset = operators[term_idx].1;

    // Find the end of the current block
    let mut depth = 1;

    for j in (term_idx + 1)..operators.len() {
        match &operators[j].0 {
            Operator::If { .. } | Operator::Block { .. } | Operator::Loop { .. } => {
                depth += 1;
            }
            Operator::End => {
                depth -= 1;
                if depth == 0 {
                    // Check if there's code between terminator and end
                    if term_idx + 1 < j {
                        let region_type = match terminator {
                            Operator::Return => DeadRegionType::AfterReturn,
                            Operator::Unreachable => DeadRegionType::AfterUnreachable,
                            _ => return None,
                        };

                        return Some(DeadRegion {
                            func_idx,
                            start_offset: operators[term_idx + 1].1,
                            end_offset: operators[j].1,
                            region_type,
                            description: format!(
                                "Code after {:?} is dead at offset {}",
                                terminator, term_offset
                            ),
                        });
                    }
                    break;
                }
            }
            _ => {}
        }
    }

    None
}

/// Find dead code after unconditional br
fn find_dead_after_br(
    operators: &[(Operator, usize)],
    br_idx: usize,
    func_idx: u32,
) -> Option<DeadRegion> {
    let br_offset = operators[br_idx].1;

    // Find the end of the current block
    let mut depth = 1;

    for j in (br_idx + 1)..operators.len() {
        match &operators[j].0 {
            Operator::If { .. } | Operator::Block { .. } | Operator::Loop { .. } => {
                depth += 1;
            }
            Operator::End | Operator::Else => {
                if depth == 1 {
                    // Check if there's code between br and end/else
                    if br_idx + 1 < j {
                        return Some(DeadRegion {
                            func_idx,
                            start_offset: operators[br_idx + 1].1,
                            end_offset: operators[j].1,
                            region_type: DeadRegionType::AfterUnconditionalBranch,
                            description: format!(
                                "Code after unconditional br is dead at offset {}",
                                br_offset
                            ),
                        });
                    }
                    break;
                }
                if matches!(&operators[j].0, Operator::End) {
                    depth -= 1;
                }
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_true_if() {
        let wat = r#"
            (module
              (func (result i32)
                (if (result i32) (i32.const 1)
                  (then (i32.const 42))
                  (else (i32.const 99))))
            )
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let regions = analyze_dead_code(&wasm).expect("Analysis failed");

        println!("Found {} dead regions", regions.len());
        for r in &regions {
            println!("  {:?}", r);
        }

        assert!(!regions.is_empty(), "Should find dead else branch");

        let else_dead = regions.iter().any(|r| {
            matches!(
                r.region_type,
                DeadRegionType::ConstantBranchIf {
                    is_then_dead: false
                }
            )
        });
        assert!(else_dead, "Should identify else branch as dead");
    }

    #[test]
    fn test_constant_false_if() {
        let wat = r#"
            (module
              (func (result i32)
                (if (result i32) (i32.const 0)
                  (then (i32.const 99))
                  (else (i32.const 42))))
            )
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let regions = analyze_dead_code(&wasm).expect("Analysis failed");

        println!("Found {} dead regions", regions.len());
        for r in &regions {
            println!("  {:?}", r);
        }

        let then_dead = regions.iter().any(|r| {
            matches!(
                r.region_type,
                DeadRegionType::ConstantBranchIf { is_then_dead: true }
            )
        });
        assert!(then_dead, "Should identify then branch as dead");
    }

    #[test]
    fn test_dead_after_return() {
        let wat = r#"
            (module
              (func (result i32)
                (return (i32.const 42))
                (i32.const 99)))
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let regions = analyze_dead_code(&wasm).expect("Analysis failed");

        println!("Found {} dead regions", regions.len());
        for r in &regions {
            println!("  {:?}", r);
        }

        let after_return = regions
            .iter()
            .any(|r| matches!(r.region_type, DeadRegionType::AfterReturn));
        assert!(after_return, "Should find dead code after return");
    }

    #[test]
    fn test_dead_after_br() {
        let wat = r#"
            (module
              (func (result i32)
                (block (result i32)
                  (br 0 (i32.const 42))
                  (i32.const 99))))
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let regions = analyze_dead_code(&wasm).expect("Analysis failed");

        println!("Found {} dead regions", regions.len());
        for r in &regions {
            println!("  {:?}", r);
        }

        let after_br = regions
            .iter()
            .any(|r| matches!(r.region_type, DeadRegionType::AfterUnconditionalBranch));
        assert!(after_br, "Should find dead code after br");
    }

    #[test]
    fn test_branch_simplification_fixture() {
        let wat = include_str!("../../../tests/fixtures/branch_simplification_test.wat");
        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let regions = analyze_dead_code(&wasm).expect("Analysis failed");

        println!(
            "Found {} dead regions in branch_simplification_test.wat",
            regions.len()
        );
        for r in &regions {
            println!(
                "  func {}: {:?} - {}",
                r.func_idx, r.region_type, r.description
            );
        }

        // Should find multiple dead regions in this fixture
        assert!(regions.len() >= 3, "Should find at least 3 dead regions");
    }
}
