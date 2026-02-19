# EMI Testing for LOOM: Implementation Proposal

**Date**: 2025-12-09
**Status**: Proposal
**Author**: Research Analysis
**Target Fixture**: `branch_simplification_test.wat`

---

## Executive Summary

This proposal outlines an implementation of **Equivalence Modulo Inputs (EMI)** testing for LOOM's WebAssembly optimizer. EMI is the most effective technique for finding miscompilation bugs, having discovered 1,600+ bugs in GCC/LLVM. No EMI implementation exists for WebAssembly optimizers, making this a novel contribution.

**Key insight**: LOOM already has the infrastructure (wasmtime, differential testing) - we just need to add dead code profiling and mutation.

---

## 1. What is EMI Testing?

### Core Principle

```
If code doesn't execute for input I, modifying it shouldn't change the output.
```

```
┌──────────────────────────────────────────────────────────────────┐
│  Program P with input I=5                                        │
│  ────────────────────────                                        │
│  func test(x: i32) -> i32 {                                     │
│      if (x > 10) {           // NOT EXECUTED when x=5           │
│          return expensive(); // ← Can delete/modify this!       │
│      }                                                           │
│      return x + 1;           // EXECUTED → returns 6            │
│  }                                                               │
│                                                                  │
│  EMI Guarantee:                                                  │
│    optimize(P)(5) == optimize(P')(5) == 6                       │
│    where P' = P with dead branch modified                        │
│                                                                  │
│  If outputs differ → MISCOMPILATION BUG                         │
└──────────────────────────────────────────────────────────────────┘
```

### Why EMI Finds Bugs

| Property | Why It Matters |
|----------|----------------|
| **Guaranteed Oracle** | Original output IS the expected output |
| **Targets Optimizer Edge Cases** | Dead code triggers special optimizer paths |
| **Finds Miscompilations** | Not just crashes - wrong code generation |
| **Real Programs** | Uses actual code, not random generation |

### Bug-Finding Record

- **147 bugs** in GCC/LLVM (original PLDI 2014 paper)
- **1,600+ bugs** total across follow-up tools
- **~550 miscompilations** (wrong code, not crashes)
- **Affected compilers**: GCC, Clang, ICC, Rust, Scala, JVMs

---

## 2. LOOM's Existing Infrastructure

### What We Already Have

```
loom-testing/
├── Cargo.toml           # wasmtime = "17.0" ✓
├── src/
│   ├── lib.rs           # DifferentialTester, TestResult ✓
│   └── bin/
│       └── differential.rs  # CLI for diff testing ✓
```

**Key existing capabilities:**

1. **wasmtime execution** - `loom-testing/src/lib.rs:9`
2. **Module loading** - `Module::new(&engine, wasm)`
3. **Semantic comparison** - `check_semantic_equivalence()`
4. **Validation** - `wasmparser::validate()`
5. **Differential testing** - `DifferentialTester::test()`

### What We Need to Add

1. **Execution profiling** - Track which instructions execute
2. **Dead code identification** - Find non-executed regions
3. **Mutation operators** - Modify dead code safely
4. **EMI test harness** - Run variants and compare

---

## 3. Target Fixture Analysis

### `branch_simplification_test.wat`

This fixture is ideal for EMI because it has **statically determinable dead code**:

```wat
;; Test 1: br_if with constant true - else branch is DEAD
(func $test_br_if_always_taken (result i32)
  (block $exit (result i32)
    (i32.const 42)
    (i32.const 1)        ;; condition = true
    (br_if $exit)        ;; ALWAYS taken
    (i32.const 99)       ;; ← DEAD CODE (never executes)
  )
)

;; Test 3: if with constant true - else branch is DEAD
(func $test_if_constant_true (result i32)
  (if (result i32) (i32.const 1)  ;; condition = true
    (then (i32.const 42))         ;; ALWAYS executed
    (else (i32.const 99))         ;; ← DEAD CODE
  )
)

;; Test 4: if with constant false - then branch is DEAD
(func $test_if_constant_false (result i32)
  (if (result i32) (i32.const 0)  ;; condition = false
    (then (i32.const 99))         ;; ← DEAD CODE
    (else (i32.const 42))         ;; ALWAYS executed
  )
)
```

### EMI Variants We Can Generate

For `$test_if_constant_true`:

```wat
;; Original
(if (result i32) (i32.const 1)
  (then (i32.const 42))
  (else (i32.const 99)))   ;; dead

;; Variant 1: Delete dead branch body
(if (result i32) (i32.const 1)
  (then (i32.const 42))
  (else (unreachable)))    ;; replaced with unreachable

;; Variant 2: Modify dead branch value
(if (result i32) (i32.const 1)
  (then (i32.const 42))
  (else (i32.const 0)))    ;; changed 99 → 0

;; Variant 3: Insert dead code in dead branch
(if (result i32) (i32.const 1)
  (then (i32.const 42))
  (else
    (i32.const 1)
    (drop)                 ;; inserted garbage
    (i32.const 99)))

;; ALL variants must return 42 after optimization!
```

---

## 4. Implementation Design

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     EMI Testing Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Input   │───▶│ Profile  │───▶│ Mutate   │───▶│ Compare  │ │
│  │  .wat    │    │ Execution│    │ Dead Code│    │ Outputs  │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │               │               │               │         │
│       ▼               ▼               ▼               ▼         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Parse to │    │ wasmtime │    │ Generate │    │ Report   │ │
│  │ Module   │    │ + trace  │    │ Variants │    │ Bugs     │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Core Data Structures

```rust
// loom-testing/src/emi.rs

use std::collections::{HashMap, HashSet};
use wasmtime::{Engine, Module, Store, Instance, Val};

/// Execution profile for a WebAssembly module
#[derive(Debug, Clone)]
pub struct ExecutionProfile {
    /// Which functions were called
    pub called_functions: HashSet<u32>,

    /// For each function, which basic blocks executed
    /// Key: (func_idx, block_idx), Value: execution count
    pub executed_blocks: HashMap<(u32, u32), u64>,

    /// Instructions that definitely did NOT execute
    pub dead_regions: Vec<DeadRegion>,
}

/// A region of code that did not execute
#[derive(Debug, Clone)]
pub struct DeadRegion {
    pub func_idx: u32,
    pub start_offset: usize,
    pub end_offset: usize,
    pub region_type: DeadRegionType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeadRegionType {
    /// Dead branch of if/else
    IfBranch { is_then: bool },
    /// Code after unconditional branch
    AfterBranch,
    /// Code after return
    AfterReturn,
    /// Code after unreachable
    AfterUnreachable,
    /// Entire function never called
    UnusedFunction,
}

/// EMI mutation strategies
#[derive(Debug, Clone, Copy)]
pub enum MutationStrategy {
    /// Replace dead code with unreachable
    ReplaceWithUnreachable,
    /// Replace dead code with nop
    ReplaceWithNop,
    /// Delete dead instructions entirely
    Delete,
    /// Modify constants in dead code
    ModifyConstants,
    /// Insert additional dead code
    InsertDeadCode,
}

/// Result of EMI testing
#[derive(Debug)]
pub struct EmiTestResult {
    pub original_output: Vec<Val>,
    pub variants_tested: usize,
    pub bugs_found: Vec<EmiBug>,
    pub profile: ExecutionProfile,
}

/// A bug found by EMI testing
#[derive(Debug)]
pub struct EmiBug {
    pub variant_id: usize,
    pub mutation: MutationStrategy,
    pub dead_region: DeadRegion,
    pub expected_output: Vec<Val>,
    pub actual_output: Result<Vec<Val>, String>,
    pub original_wat: String,
    pub variant_wat: String,
}
```

### 4.3 Profiling Strategy

For the initial implementation, we'll use **static dead code analysis** rather than runtime profiling. This is simpler and works well for `branch_simplification_test.wat`:

```rust
/// Statically analyze a module for dead code regions
///
/// This finds code that is provably dead without execution:
/// - Branches with constant conditions
/// - Code after terminators (return, br, unreachable)
pub fn analyze_dead_code_static(module: &Module) -> Vec<DeadRegion> {
    let mut dead_regions = Vec::new();

    for (func_idx, func) in module.functions.iter().enumerate() {
        let mut i = 0;
        let instructions = &func.instructions;

        while i < instructions.len() {
            match &instructions[i] {
                // Pattern: if (i32.const 0) ... else ... end
                // The "then" branch is dead
                Instruction::I32Const(0) if matches_if_next(instructions, i) => {
                    if let Some(region) = find_then_branch(instructions, i) {
                        dead_regions.push(DeadRegion {
                            func_idx: func_idx as u32,
                            start_offset: region.0,
                            end_offset: region.1,
                            region_type: DeadRegionType::IfBranch { is_then: true },
                        });
                    }
                }

                // Pattern: if (i32.const 1) ... else ... end
                // The "else" branch is dead
                Instruction::I32Const(1) if matches_if_next(instructions, i) => {
                    if let Some(region) = find_else_branch(instructions, i) {
                        dead_regions.push(DeadRegion {
                            func_idx: func_idx as u32,
                            start_offset: region.0,
                            end_offset: region.1,
                            region_type: DeadRegionType::IfBranch { is_then: false },
                        });
                    }
                }

                // Pattern: return/br/unreachable followed by more code
                Instruction::Return | Instruction::Br(_) | Instruction::Unreachable => {
                    if let Some(region) = find_code_after_terminator(instructions, i) {
                        dead_regions.push(DeadRegion {
                            func_idx: func_idx as u32,
                            start_offset: region.0,
                            end_offset: region.1,
                            region_type: DeadRegionType::AfterReturn,
                        });
                    }
                }

                _ => {}
            }
            i += 1;
        }
    }

    dead_regions
}
```

### 4.4 Mutation Operators

```rust
/// Apply a mutation to a dead region
pub fn apply_mutation(
    module: &mut Module,
    region: &DeadRegion,
    strategy: MutationStrategy,
) -> Result<()> {
    let func = &mut module.functions[region.func_idx as usize];

    match strategy {
        MutationStrategy::ReplaceWithUnreachable => {
            // Replace all instructions in dead region with single unreachable
            // Must preserve stack type!
            let result_type = infer_block_result_type(func, region)?;

            func.instructions.splice(
                region.start_offset..region.end_offset,
                std::iter::once(Instruction::Unreachable)
            );
        }

        MutationStrategy::ReplaceWithNop => {
            // Replace with nops (stack-neutral)
            let count = region.end_offset - region.start_offset;
            func.instructions.splice(
                region.start_offset..region.end_offset,
                std::iter::repeat(Instruction::Nop).take(count)
            );
        }

        MutationStrategy::ModifyConstants => {
            // Change constant values in dead region
            for i in region.start_offset..region.end_offset {
                match &mut func.instructions[i] {
                    Instruction::I32Const(v) => *v = v.wrapping_add(1),
                    Instruction::I64Const(v) => *v = v.wrapping_add(1),
                    _ => {}
                }
            }
        }

        MutationStrategy::Delete => {
            // Delete dead instructions (must ensure valid wasm)
            func.instructions.drain(region.start_offset..region.end_offset);
        }

        MutationStrategy::InsertDeadCode => {
            // Insert harmless dead code
            let dead_code = vec![
                Instruction::I32Const(42),
                Instruction::Drop,
            ];
            func.instructions.splice(
                region.start_offset..region.start_offset,
                dead_code
            );
        }
    }

    Ok(())
}
```

### 4.5 Main EMI Test Loop

```rust
/// Run EMI testing on a WebAssembly module
pub fn emi_test(
    wasm_bytes: &[u8],
    test_inputs: &[Vec<Val>],
    iterations: usize,
) -> Result<EmiTestResult> {
    let engine = Engine::default();
    let mut rng = rand::thread_rng();

    // 1. Parse and analyze for dead code
    let module = parse_wasm(wasm_bytes)?;
    let dead_regions = analyze_dead_code_static(&module);

    if dead_regions.is_empty() {
        return Ok(EmiTestResult {
            original_output: vec![],
            variants_tested: 0,
            bugs_found: vec![],
            profile: ExecutionProfile::empty(),
        });
    }

    // 2. Get original output for each test input
    let original_outputs: Vec<Vec<Val>> = test_inputs
        .iter()
        .map(|input| execute_module(&engine, wasm_bytes, input))
        .collect::<Result<_>>()?;

    // 3. Generate and test variants
    let mut bugs_found = Vec::new();
    let strategies = [
        MutationStrategy::ReplaceWithUnreachable,
        MutationStrategy::ModifyConstants,
        MutationStrategy::InsertDeadCode,
    ];

    for variant_id in 0..iterations {
        // Pick random dead region and mutation strategy
        let region = dead_regions.choose(&mut rng).unwrap();
        let strategy = strategies.choose(&mut rng).unwrap();

        // Clone module and apply mutation
        let mut variant_module = module.clone();
        apply_mutation(&mut variant_module, region, *strategy)?;

        // Encode variant back to bytes
        let variant_bytes = encode_wasm(&variant_module)?;

        // 4. Optimize the variant with LOOM
        let optimized_bytes = loom_optimize(&variant_bytes)?;

        // 5. Execute optimized variant and compare
        for (input_idx, input) in test_inputs.iter().enumerate() {
            let result = execute_module(&engine, &optimized_bytes, input);

            match result {
                Ok(output) if output != original_outputs[input_idx] => {
                    // BUG FOUND: Output differs!
                    bugs_found.push(EmiBug {
                        variant_id,
                        mutation: *strategy,
                        dead_region: region.clone(),
                        expected_output: original_outputs[input_idx].clone(),
                        actual_output: Ok(output),
                        original_wat: module_to_wat(&module),
                        variant_wat: module_to_wat(&variant_module),
                    });
                }
                Err(e) => {
                    // BUG FOUND: Variant crashed/failed validation!
                    bugs_found.push(EmiBug {
                        variant_id,
                        mutation: *strategy,
                        dead_region: region.clone(),
                        expected_output: original_outputs[input_idx].clone(),
                        actual_output: Err(e.to_string()),
                        original_wat: module_to_wat(&module),
                        variant_wat: module_to_wat(&variant_module),
                    });
                }
                Ok(_) => {} // Same output, no bug
            }
        }
    }

    Ok(EmiTestResult {
        original_output: original_outputs.into_iter().flatten().collect(),
        variants_tested: iterations,
        bugs_found,
        profile: ExecutionProfile::from_regions(dead_regions),
    })
}
```

---

## 5. Concrete Implementation for `branch_simplification_test.wat`

### 5.1 Test Harness

```rust
// loom-testing/src/bin/emi.rs

use loom_testing::emi::{emi_test, EmiTestResult};
use wasmtime::Val;

fn main() -> anyhow::Result<()> {
    println!("🔬 LOOM EMI Testing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Load test fixture
    let wat = include_str!("../../../tests/fixtures/branch_simplification_test.wat");
    let wasm = wat::parse_str(wat)?;

    // Define test inputs (empty for functions with no params)
    let test_inputs = vec![
        vec![],  // For functions with no params
    ];

    // Run EMI testing
    let result = emi_test(&wasm, &test_inputs, 100)?;

    // Report results
    println!("📊 EMI Test Results");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Dead regions found: {}", result.profile.dead_regions.len());
    println!("Variants tested: {}", result.variants_tested);
    println!("Bugs found: {}", result.bugs_found.len());

    if !result.bugs_found.is_empty() {
        println!("\n🐛 Bugs Found:");
        for (i, bug) in result.bugs_found.iter().enumerate() {
            println!("\n[Bug #{}]", i + 1);
            println!("  Mutation: {:?}", bug.mutation);
            println!("  Dead region: func {}, offsets {}..{}",
                bug.dead_region.func_idx,
                bug.dead_region.start_offset,
                bug.dead_region.end_offset);
            println!("  Expected: {:?}", bug.expected_output);
            println!("  Actual: {:?}", bug.actual_output);
            println!("  Variant WAT:\n{}", indent(&bug.variant_wat, "    "));
        }
    } else {
        println!("\n✅ No bugs found!");
    }

    Ok(())
}

fn indent(s: &str, prefix: &str) -> String {
    s.lines().map(|line| format!("{}{}", prefix, line)).collect::<Vec<_>>().join("\n")
}
```

### 5.2 Expected Dead Regions in `branch_simplification_test.wat`

```
Function                      Dead Region Type           Location
─────────────────────────────────────────────────────────────────
$test_br_if_always_taken      AfterBranch               i32.const 99
$test_br_if_never_taken       (none - br_if removed)
$test_if_constant_true        IfBranch(else)            i32.const 99
$test_if_constant_false       IfBranch(then)            i32.const 99
$test_if_identical_arms       (none - runtime dependent)
$test_nested_constant_ifs     IfBranch(else) outer      i32.const 30
                              IfBranch(then) inner      i32.const 10
$test_nop_removal             (none - nops not dead code)
$test_complex                 IfBranch(else)            i32.const 99
```

---

## 6. File Structure

```
loom-testing/
├── Cargo.toml                    # Add rand dependency
├── src/
│   ├── lib.rs                    # Re-export emi module
│   ├── emi.rs                    # NEW: EMI core implementation
│   ├── emi/
│   │   ├── mod.rs               # Module organization
│   │   ├── profile.rs           # Execution profiling
│   │   ├── mutation.rs          # Mutation operators
│   │   ├── analysis.rs          # Dead code analysis
│   │   └── report.rs            # Bug reporting
│   └── bin/
│       ├── differential.rs       # Existing
│       └── emi.rs               # NEW: EMI test binary
```

---

## 7. Implementation Phases

### Phase 1: Static Analysis (Week 1)
- [ ] Implement `analyze_dead_code_static()`
- [ ] Parse `branch_simplification_test.wat`
- [ ] Identify dead regions from constant conditions
- [ ] Unit tests for dead region detection

### Phase 2: Mutation Operators (Week 1)
- [ ] Implement `ReplaceWithUnreachable`
- [ ] Implement `ModifyConstants`
- [ ] Ensure mutations produce valid Wasm
- [ ] Unit tests for each mutation

### Phase 3: Test Harness (Week 2)
- [ ] Implement `emi_test()` main loop
- [ ] Integration with LOOM optimizer
- [ ] Wasmtime execution comparison
- [ ] Bug reporting format

### Phase 4: CLI & Integration (Week 2)
- [ ] Create `emi` binary
- [ ] Integration tests with fixtures
- [ ] Documentation
- [ ] CI integration

---

## 8. Success Criteria

1. **Detects existing bugs**: Should flag the 2 known failing fixtures
2. **No false positives**: Variants of valid code should pass
3. **Performance**: Test 100 variants in < 10 seconds
4. **Coverage**: Find dead regions in 80%+ of fixtures with branches

---

## 9. Future Extensions

### Runtime Profiling (Phase 2)
Add actual execution profiling for programs with runtime-dependent branches:

```rust
/// Profile execution using wasmtime's fuel metering
pub fn profile_execution_runtime(
    wasm: &[u8],
    inputs: &[Vec<Val>],
) -> Result<ExecutionProfile> {
    // Use wasmtime's epoch interruption for tracing
    let mut config = Config::new();
    config.epoch_interruption(true);

    // Execute and collect coverage
    // ...
}
```

### Wapplique-Style Mutations (Phase 3)
Integrate real-world code fragments:

```rust
/// Insert code fragments from other Wasm modules
pub fn applique_mutation(
    module: &mut Module,
    fragment_corpus: &[WasmFragment],
) -> Result<()> {
    // Insert fragments at dead code locations
    // Ensure type compatibility
}
```

### Integration with wasm-smith (Phase 4)
Generate random valid Wasm for broader testing:

```rust
/// Generate EMI variants using wasm-smith
pub fn generate_emi_variants_random(
    seed_module: &Module,
    count: usize,
) -> Vec<Module> {
    // Use wasm-smith to generate valid mutations
}
```

---

## 10. References

### Academic Papers
- [Compiler Validation via Equivalence Modulo Inputs](https://dl.acm.org/doi/10.1145/2594334) (PLDI 2014)
- [Boosting Compiler Testing by Injecting Real-World Code](https://dl.acm.org/doi/10.1145/3656386) (PLDI 2024)
- [Wapplique: Testing WebAssembly Runtime](https://dl.acm.org/doi/abs/10.1145/3650212.3680340) (ISSTA 2024)

### Tools
- [EMI Project Page](https://web.cs.ucdavis.edu/~su/emi-project/)
- [wasmut - Wasm Mutation Testing](https://github.com/lwagner94/wasmut)
- [wasm-smith - Wasm Fuzzer](https://github.com/bytecodealliance/wasm-tools/tree/main/crates/wasm-smith)

### LOOM Infrastructure
- `loom-testing/src/lib.rs` - Existing differential testing
- `loom-core/tests/component_execution_tests.rs` - Wasmtime integration examples

---

## 11. Decision Point: Your Input

Before I implement this, I'd like your input on one key design choice:

**Mutation Strategy Priority**

Which mutation strategies should we prioritize?

1. **Conservative** (safer, fewer false positives):
   - Only `ModifyConstants` in dead branches
   - Preserves Wasm structure completely

2. **Aggressive** (finds more bugs, may have false positives):
   - `ReplaceWithUnreachable`
   - `Delete` dead code entirely
   - `InsertDeadCode`

3. **Hybrid** (recommended):
   - Start conservative
   - Escalate to aggressive if no bugs found
   - Track which strategies find bugs

**My recommendation**: Start with **Hybrid** approach. The conservative mutations will validate the infrastructure works, then aggressive mutations will stress-test the optimizer.

---

Want me to proceed with the implementation?
