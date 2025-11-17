# CRITICAL BUG: quicksort.wat Produces Invalid WASM

## Summary

**Severity**: CRITICAL
**Status**: Identified, not yet fixed
**Discovered**: November 17, 2025
**Affects**: Complex control flow with nested blocks, loops, and recursion

LOOM optimization of `tests/fixtures/quicksort.wat` produces invalid WebAssembly that fails validation with the error:

```
[parse exception: block cannot pop from outside (at 0:192)]
Fatal: error parsing wasm
```

## Reproduction

```bash
# Compile and optimize quicksort
./target/release/loom optimize tests/fixtures/quicksort.wat -o /tmp/quicksort_loom.wasm --stats

# Try to validate with wasm-opt
/tmp/binaryen-version_118/bin/wasm-opt /tmp/quicksort_loom.wasm -O0 -o /tmp/test.wasm
# Error: [parse exception: block cannot pop from outside (at 0:192)]

# Try to disassemble
/tmp/binaryen-version_118/bin/wasm-dis /tmp/quicksort_loom.wasm
# Error: [parse exception: block cannot pop from outside (at 0:192)]
```

**100% reproducible**

## Symptoms

1. **Binary Size**: Grows from 256 bytes → 257 bytes (adds 1 byte)
2. **Instruction Count**: Grows from 51 → 53 (adds 2 instructions)
3. **Validation**: WASM binary fails all standard validators
4. **Parse Error**: "block cannot pop from outside" at offset 192

## Root Cause Analysis

### Stack Type Error

The error "block cannot pop from outside" indicates a **stack type mismatch**:
- A `block`, `loop`, or `if` is trying to access values not on the operand stack
- The block signature doesn't match the actual stack state
- Could be caused by:
  - Removing a value-producing instruction without updating consumers
  - Incorrectly merging blocks with different stack states
  - Breaking block nesting invariants

### Offset 192

The error occurs at byte offset 192 in the binary:
- quicksort.wasm is 257 bytes total
- Offset 192 is about 75% through the file
- Likely in the $quicksort or $partition function body

### Possible Culprits

Given the optimization pipeline order:

```
1. Precompute (global constant propagation)
2. ISLE Folding (algebraic simplifications)
3. Advanced Instructions (strength reduction)
4. CSE (common subexpression elimination)
5. Inline Functions
6. ISLE Folding (again)
7. Code Folding
8. LICM (loop-invariant code motion)
9. Branch Simplify
10. DCE (dead code elimination)
11. Block Merge
12. Vacuum (remove no-ops)
13. Simplify Locals
```

**Most Likely Suspects**:

1. **Block Merge (Phase 11)** - Most likely
   - Merges consecutive blocks without verifying stack invariants
   - Could be merging blocks with incompatible signatures
   - quicksort has complex nested blocks (block, loop, if)

2. **DCE (Phase 10)**
   - Might remove instructions that produce stack values
   - Consumers of those values would cause stack underflow

3. **ISLE Folding (Phases 2 & 6)**
   - terms_to_instructions could be generating invalid block structure
   - End instruction handling may be wrong

4. **Branch Simplify (Phase 9)**
   - Might be incorrectly rewriting control flow
   - Could break block nesting

### Why Other Tests Don't Catch This

The 20 optimization tests all pass because they test simpler control flow:
- Single functions
- No deep recursion
- Simpler block nesting
- Smaller code size

quicksort is unique in having:
- **3 functions** with complex interactions
- **Recursive calls** (quicksort calls itself)
- **Nested control flow**: if inside loop inside block
- **Multiple branches** in partition function

## Investigation Steps

### 1. Binary Inspection

```bash
# Hexdump at offset 192
xxd -s 192 -l 65 /tmp/quicksort_loom.wasm
```

Look for block opcodes:
- `0x02` = block
- `0x03` = loop
- `0x04` = if
- `0x0b` = end
- `0x0c` = br
- `0x0d` = br_if

### 2. Phase-by-Phase Testing

Modify `loom-core/src/lib.rs` to disable phases one at a time:

```rust
// Comment out phases to isolate the bug
pub fn optimize_module(module: &mut Module) -> Result<()> {
    precompute(module)?;
    // ... ISLE ...
    optimize_advanced_instructions(module)?;
    eliminate_common_subexpressions(module)?;
    inline_functions(module)?;
    // ... second ISLE ...
    fold_code(module)?;
    licm(module)?;
    simplify_branches(module)?;
    // eliminate_dead_code(module)?;  // Try disabling DCE
    // merge_consecutive_blocks(module)?;  // Try disabling block merge
    vacuum(module)?;
    simplify_locals(module)?;
    Ok(())
}
```

Run after each phase:
```bash
cargo build --release && ./target/release/loom optimize tests/fixtures/quicksort.wat -o /tmp/test.wasm
wasm-opt /tmp/test.wasm -O0 -o /tmp/validate.wasm
```

### 3. Add Validation Pass

Add WASM validation after each optimization phase:

```rust
fn validate_wasm(module: &Module) -> Result<()> {
    // Encode to binary
    let binary = encode::module_to_wasm_binary(module)?;

    // Try to parse it back (catches obvious errors)
    let reparsed = parse::parse_wasm_binary(&binary)?;

    // Check stack consistency (need to implement)
    // validate_stack_types(module)?;

    Ok(())
}

pub fn optimize_module(module: &mut Module) -> Result<()> {
    precompute(module)?;
    validate_wasm(module)?;  // Validate after each phase

    // ... rest of phases ...
}
```

### 4. Inspect Block Merge Logic

Check `merge_consecutive_blocks()` in loom-core/src/lib.rs:

Key questions:
- Does it verify block signatures match?
- Does it check stack heights?
- Does it handle loops correctly?
- Does it preserve br/br_if targets?

### 5. Check ISLE Conversion

The ISLE conversion (instructions → terms → instructions) might be breaking:

```rust
// In optimize_module, around ISLE phases
if let Ok(terms) = super::terms::instructions_to_terms(&func.instructions) {
    if !terms.is_empty() {
        let mut env = LocalEnv::new();
        let optimized_terms: Vec<Value> = terms
            .into_iter()
            .map(|term| simplify_with_env(term, &mut env))
            .collect();
        if let Ok(mut new_instrs) = super::terms::terms_to_instructions(&optimized_terms) {
            // Are we preserving block structure correctly?
            func.instructions = new_instrs;
        }
    }
}
```

## Proposed Fixes

### Fix 1: Add Stack Type Validation (Immediate)

```rust
// Add to loom-core/src/lib.rs
fn validate_stack_consistency(module: &Module) -> Result<()> {
    for func in &module.functions {
        let mut stack_height = 0;
        for instr in &func.instructions {
            match instr {
                Instruction::Block { .. } => { /* verify block signature */ }
                Instruction::Loop { .. } => { /* verify loop signature */ }
                Instruction::If { .. } => { /* verify if signature */ }
                Instruction::End => { /* verify stack matches block type */ }
                // ... check all instructions update stack correctly
                _ => {}
            }
        }
    }
    Ok(())
}
```

### Fix 2: Disable Block Merge for Complex Functions (Quick Workaround)

```rust
fn merge_consecutive_blocks(module: &mut Module) -> Result<()> {
    for func in &mut module.functions {
        // Skip if function is too complex
        let complexity = calculate_complexity(func);
        if complexity > THRESHOLD {
            continue;  // Don't merge blocks in complex functions
        }

        // ... existing merge logic ...
    }
    Ok(())
}
```

### Fix 3: Fix Block Merge to Preserve Invariants (Proper Fix)

```rust
fn can_merge_blocks(block1: &Block, block2: &Block) -> bool {
    // Check:
    // 1. Block signatures are compatible
    // 2. No branches into block2 from outside
    // 3. Stack height matches at boundary
    // 4. Block types are compatible (block/loop/if)

    // Only merge if ALL conditions met
    true
}

fn merge_consecutive_blocks(module: &mut Module) -> Result<()> {
    for func in &mut module.functions {
        // Use proper CFG analysis
        let cfg = build_control_flow_graph(func);

        // Only merge blocks that are safe to merge
        for (b1, b2) in find_mergeable_block_pairs(&cfg) {
            if can_merge_blocks(b1, b2) {
                merge_blocks(b1, b2);
            }
        }
    }
    Ok(())
}
```

### Fix 4: Improve ISLE Terms Conversion

```rust
// In terms.rs
pub fn terms_to_instructions(terms: &[Value]) -> Result<Vec<Instruction>> {
    let mut converter = TermConverter::new();
    let instrs = converter.convert(terms)?;

    // CRITICAL: Validate block structure before returning
    validate_block_nesting(&instrs)?;
    validate_stack_consistency(&instrs)?;

    Ok(instrs)
}
```

## Testing Strategy

### 1. Add Regression Test

```rust
// In loom-core/tests/optimization_tests.rs
#[test]
fn test_quicksort_optimization() {
    let input = include_str!("../tests/fixtures/quicksort.wat");
    let mut module = parse::parse_wat(input).unwrap();

    // Optimize
    optimize::optimize_module(&mut module).unwrap();

    // Encode to binary
    let binary = encode::module_to_wasm_binary(&module).unwrap();

    // MUST be valid WASM
    assert!(parse::parse_wasm_binary(&binary).is_ok());

    // Validate with external tool (if available)
    // std::fs::write("/tmp/test_quicksort.wasm", &binary).unwrap();
    // let output = std::process::Command::new("wasm-validate")
    //     .arg("/tmp/test_quicksort.wasm")
    //     .output();
    // assert!(output.unwrap().status.success());
}
```

### 2. Add Fuzzing

```rust
// Use cargo-fuzz to find more cases
#[cfg(fuzzing)]
fn fuzz_optimize(data: &[u8]) {
    if let Ok(module) = parse::parse_wasm_binary(data) {
        let mut module = module.clone();
        if optimize::optimize_module(&mut module).is_ok() {
            // Must produce valid WASM
            let binary = encode::module_to_wasm_binary(&module).unwrap();
            parse::parse_wasm_binary(&binary).unwrap();
        }
    }
}
```

### 3. Test All Fixtures

Create a comprehensive test that optimizes ALL fixtures and validates:

```rust
#[test]
fn test_all_fixtures_produce_valid_wasm() {
    let fixtures = [
        "advanced_math.wat",
        "bench_bitops.wat",
        "bench_locals.wat",
        "crypto_utils.wat",
        "fibonacci.wat",
        "matrix_multiply.wat",
        "quicksort.wat",  // This currently fails!
        "simple_game_logic.wat",
        "test_input.wat",
    ];

    for fixture in &fixtures {
        let path = format!("tests/fixtures/{}", fixture);
        let input = std::fs::read_to_string(&path).unwrap();
        let mut module = parse::parse_wat(&input).unwrap();

        // Optimize
        optimize::optimize_module(&mut module).unwrap();

        // Encode
        let binary = encode::module_to_wasm_binary(&module).unwrap();

        // MUST be valid
        let result = parse::parse_wasm_binary(&binary);
        assert!(result.is_ok(), "Fixture {} produced invalid WASM: {:?}", fixture, result.err());
    }
}
```

## Impact Assessment

### Severity: CRITICAL

This bug:
- ✓ **Produces invalid WASM** that fails all validators
- ✓ **Cannot be executed** by any WASM runtime
- ✓ **Silent failure** - no error during optimization
- ✓ **Data loss** - output is unusable
- ✓ **Affects real code** - quicksort is a common algorithm

### Workaround

Until fixed, users should:
1. **Validate all LOOM output** with `wasm-opt -O0` or `wasm-validate`
2. **Compare file sizes** - if output grew, likely broken
3. **Avoid optimizing complex recursive code** with LOOM
4. **Use wasm-opt instead** for production builds

## Priority Ranking

### P0 (Immediate)
1. Add validation to catch this bug
2. Disable problematic optimization phase
3. Add regression test
4. Warn users in README

### P1 (This Week)
1. Isolate which phase causes the bug
2. Implement proper fix
3. Add comprehensive fixture validation tests
4. Document control flow handling

### P2 (Next Sprint)
1. Add CFG-based analysis for block merging
2. Implement fuzzing
3. Add stack type checker
4. Improve ISLE conversion validation

## References

- **WebAssembly Validation**: https://webassembly.github.io/spec/core/valid/
- **Stack Polymorphism**: https://webassembly.github.io/spec/core/valid/instructions.html#polymorphism
- **Binaryen Source**: https://github.com/WebAssembly/binaryen
- **LOOM Issue**: (create GitHub issue)

## Notes

- This bug was discovered during performance comparison with wasm-opt
- The benchmark script showed quicksort growing by 1 byte
- All 20 existing optimization tests pass (they don't test complex control flow)
- Other complex fixtures (crypto_utils, matrix_multiply) work fine
- Suggests the issue is specific to **recursive functions with complex control flow**

---

**Last Updated**: November 17, 2025
**Reporter**: Claude (LOOM Development)
**Priority**: P0 - CRITICAL
**Status**: Investigation in progress
