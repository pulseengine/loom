# LOOM Encoder Bug Analysis - Detailed Investigation

**Date:** 2025-11-18
**Status:** CRITICAL BUGS IDENTIFIED
**Priority:** P0 - Blocks all optimization work

---

## Executive Summary

After recompiling test fixtures with wasm-tools (not LOOM), differential testing confirmed **critical encoder bugs** in LOOM's optimization pipeline. The root cause has been identified: **incorrect loop condition hoisting and return value corruption**.

**Key Insight:** The original test corpus was compiled with LOOM itself, creating a circular validation problem. After recompiling with wasm-tools, we now have valid inputs that expose LOOM's encoder bugs.

---

## Root Cause Analysis - fibonacci.wasm

### Bug Location

The bug occurs in one of LOOM's optimization phases that restructures control flow. Specifically:
- Loop conditions are incorrectly hoisted outside the loop
- Return values are corrupted (changed to incorrect values)
- Stack depth becomes misaligned

### Detailed Comparison

**ORIGINAL (Valid WASM):**
```wat
(func $fib_iterative (param $n i32) (result i32)
  (local $a i32) (local $b i32) (local $temp i32) (local $i i32)

  ;; Initialize
  i32.const 0
  local.set $a
  i32.const 1
  local.set $b
  i32.const 0
  local.set $i

  ;; Loop structure
  block $break
    loop $continue
      local.get $i    ; ← Get loop counter
      local.get $n    ; ← Get loop limit
      i32.ge_u        ; ← Compare (needs 2 values)
      br_if $break    ; ← Break if i >= n

      ;; Loop body...
      local.get $a
      local.set $temp
      local.get $b
      local.set $a
      local.get $temp
      local.get $b
      i32.add
      local.set $b
      local.get $i
      i32.const 1
      i32.add
      local.set $i

      br $continue
    end
  end

  local.get $a      ; ← Return result
)
```

**LOOM OUTPUT (Invalid WASM):**
```wat
(func (param i32) (result i32)
  (local i32 i32 i32 i32)

  ;; Initialize
  i32.const 0
  local.set 1
  i32.const 1
  local.set 2
  i32.const 0
  local.set 4

  ;; BUGGY loop structure
  block ;; label = @1
    local.get 0       ; ← BUG #1: Hoisted outside loop!
    i32.const 1       ; ← BUG #2: Hoisted outside loop!
    loop ;; label = @2
      local.get 4     ; ← Gets loop counter
      i32.ge_u        ; ← BUG #3: Only 1 value on stack, needs 2!
      br_if 1 (;@1;)

      ;; Loop body (partially correct)...
      local.get 2
      local.set 3
      local.get 2
      local.set 1
      local.get 2
      local.get 2
      i32.add
      local.set 2
      local.get 4
      i32.add
      local.set 4

      br 0 (;@2;)
    end
  end

  i32.const 0         ; ← BUG #4: Returns 0 instead of result!
)
```

### Identified Bugs

#### Bug #1 & #2: Incorrect Loop Condition Hoisting
**What:** `local.get $n` and `i32.const 1` were moved outside the loop
**Impact:** Loop condition comparison receives wrong values
**Severity:** CRITICAL - Breaks loop semantics

#### Bug #3: Stack Corruption
**What:** `i32.ge_u` expects 2 values but only gets 1 (`local.get 4`)
**Impact:** Type mismatch error "expected i32 but nothing on stack"
**Severity:** CRITICAL - Produces invalid WASM

#### Bug #4: Return Value Corruption
**What:** Function returns `i32.const 0` instead of `local.get $a` (the computed result)
**Impact:** Function always returns 0, completely incorrect behavior
**Severity:** CRITICAL - Wrong output

---

## Suspected Optimization Pass

Based on the bug pattern, the likely culprits are:

### 1. **Loop Invariant Code Motion (LICM)** - Most Likely
If LOOM has a pass that hoists loop-invariant expressions, it may be incorrectly:
- Treating loop condition operands as invariant
- Hoisting them outside the loop
- Not accounting for loop-dependent values

### 2. **Dead Code Elimination** - Possible
DCE may be incorrectly determining that:
- The actual return value (`local.get $a`) is dead
- Replacing it with a default value (`i32.const 0`)

### 3. **Control Flow Simplification** - Possible
A pass that restructures blocks/loops may be:
- Incorrectly reordering instructions
- Breaking the association between loop conditions and loop bodies

---

## Investigation Plan

### Step 1: Disable Optimization Passes One by One

Test with each pass disabled to identify the culprit:

```rust
// In loom-core/src/lib.rs optimize() function
pub fn optimize(module: &mut Module) -> Result<()> {
    // Phase 1-10: Keep these (basic optimizations)
    constant_fold(module)?;
    propagate(module)?;
    simplify(module)?;
    peephole(module)?;
    dead_code_eliminate(module)?;
    inline_tiny_functions(module)?;
    deduplicate_blocks(module)?;
    hoist_invariants(module)?;   // ← SUSPECT! Disable this first
    merge_blocks(module)?;
    vacuum(module)?;

    // Phase 11.5: CoalesceLocals (disabled, not the cause)
    // coalesce_locals(module)?;

    // Phase 12: SimplifyLocals
    simplify_locals(module)?;

    Ok(())
}
```

### Step 2: Add Validation After Each Pass

Insert wasm-tools validation after each optimization pass:

```rust
pub fn optimize(module: &mut Module) -> Result<()> {
    constant_fold(module)?;
    validate_module(module, "after constant_fold")?;

    propagate(module)?;
    validate_module(module, "after propagate")?;

    // ... etc
}
```

### Step 3: Create Minimal Reproduction

Create the simplest possible WASM that triggers the bug:

```wat
(module
  (func (export "test") (param $n i32) (result i32)
    (local $i i32)
    i32.const 0
    local.set $i

    block $break
      loop $continue
        local.get $i
        local.get $n
        i32.ge_u
        br_if $break

        local.get $i
        i32.const 1
        i32.add
        local.set $i

        br $continue
      end
    end

    local.get $i
  )
)
```

---

## Other Affected Files

The same bug pattern likely affects:
- **crypto_utils.wasm** - "block cannot pop from outside"
- **matrix_multiply.wasm** - "block cannot pop from outside"
- **quicksort.wasm** - "block cannot pop from outside"
- **simple_game_logic.wasm** - Multi-value/tuple errors

All show similar stack corruption symptoms.

---

## Recommended Fix Strategy

### Option A: Disable Problematic Pass (Quick Fix)
1. Identify the pass causing the bug
2. Disable it temporarily
3. Re-run differential tests
4. Document the disabled optimization

**Pros:** Fast, safe
**Cons:** Loses optimization opportunity

### Option B: Fix the Pass (Proper Fix)
1. Identify the specific condition causing incorrect hoisting
2. Add constraints to prevent hoisting loop-dependent values
3. Add test cases for loop structures
4. Validate fix with all test files

**Pros:** Preserves optimization
**Cons:** Takes longer, higher risk

---

## Success Criteria

Fix is complete when:
1. ✅ All 9 WAT fixtures compile to valid WASM
2. ✅ LOOM optimizes all 9 without validation errors
3. ✅ wasm-tools validates all LOOM outputs
4. ✅ Differential tests show no errors
5. ✅ All existing tests continue to pass

---

## Next Steps

1. **IMMEDIATE:** Disable `hoist_invariants()` and test
2. **SHORT TERM:** Add validation after each optimization pass
3. **MEDIUM TERM:** Fix the buggy optimization pass
4. **LONG TERM:** Add regression tests for control flow patterns

---

## Appendix: Test Corpus Status

### Valid Inputs (Compiled with wasm-tools)
All 9 WAT files now compile to valid WASM:
- ✅ advanced_math.wasm (602 bytes)
- ✅ bench_bitops.wasm (208 bytes)
- ✅ bench_locals.wasm (194 bytes)
- ✅ crypto_utils.wasm (906 bytes)
- ✅ fibonacci.wasm (222 bytes)
- ✅ matrix_multiply.wasm (348 bytes)
- ✅ quicksort.wasm (458 bytes)
- ✅ simple_game_logic.wasm (460 bytes)
- ✅ test_input.wasm (120 bytes)

### Component Model Files
- calc.component.wasm - Not tested (wasm-opt doesn't support)
- simple.component.wasm - Not tested (wasm-opt doesn't support)

### LOOM Output Status
- ❌ 5/9 files produce invalid WASM
- ⚠️ 4/9 files produce valid but oversized WASM (6-12x larger than wasm-opt)
