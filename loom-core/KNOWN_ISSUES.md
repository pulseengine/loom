# Known Optimizer Correctness Issues

## Status Summary
- **Total test fixtures**: 29
- **Passing validation**: 28/29 (96.6%)
- **Known failing**: 2/29 (3.4%)

## Fixed Issues

### ✅ Dead Code Elimination (DCE) - FIXED
- **Commit**: 5228555
- **Issue**: DCE was producing invalid WASM when removing code after terminators
- **Root Cause**:
  - Blocks with result types but unreachable bodies left no value on stack
  - Blocks containing terminators didn't mark following code as unreachable
- **Solution**: Added unreachable instruction injection and proper reachability tracking
- **Tests Fixed**: dce_test.wat

## Known Issues (Pending Fix)

### ❌ Simplify_locals Pass - Stack Type Mismatch
- **Fixture**: tests/fixtures/simplify_locals_test.wat (func 4 - $nested_blocks)
- **Error**: Type mismatch: values remaining on stack at end of block
- **Problem**: Values left on block stack when block expects empty type
- **Root Cause**: simplify_instructions doesn't validate stack balance in nested blocks
- **Impact**: Local variable optimization can break type system invariants

### ❌ Vacuum Pass - Block Unwrapping Issues
- **Fixture**: tests/fixtures/vacuum_test.wat (func 6 - $mixed_patterns)
- **Error**: Type mismatch: expected i32 but nothing on stack
- **Problem**: Block unwrapping removes necessary type conversions
- **Root Cause**: is_trivial_block unwrapping doesn't account for outer stack interaction
- **Impact**: Dead code removal can create stack imbalances

## Technical Details

Both remaining bugs stem from lacking proper **stack analysis**. The optimizer processes:

```wasm
block (result i32)        ;; expects i32 on stack
  local.get $0            ;; produces i32 - good
  local.set $1            ;; consumes i32, produces nothing - ERROR!
  local.get $0            ;; produces i32 - leaves on stack
end
```

Nested blocks compound the issue:

```wasm
block (result i32)        ;; outer - expects i32
  block (result i32)      ;; inner - expects i32
    return (i32.const 10)  ;; returns from function!
  end                      ;; inner block never produces value
  i32.const 30             ;; left on outer block's stack - ERROR!
end
```

## Next Steps

To fix remaining bugs, implement:

1. **Stack Depth Tracking**
   - Track how each instruction affects the value stack
   - Validate block transformations preserve stack invariants

2. **Block Analysis**
   - Before unwrapping blocks, verify outer stack requirements
   - Check that body produces/consumes correct stack values

3. **Testing**
   - Add stack validation tests for each pass
   - Validate before/after optimization preserves stack properties

## References

- WASM Spec: https://webassembly.org/specs/core/exec/instructions.html
- Stack Machine Model: WebAssembly operates on implicit value stack
- Block Typing: All blocks must satisfy type constraints per spec
