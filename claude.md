# Development Notes

## Current Session Work

### Completed
1. ✅ Component Model Execution Verification
   - Implemented ComponentExecutor for structural validation
   - Added canonical function preservation checks
   - Created differential testing framework

2. ✅ CI Validation Pipeline
   - Fixed "Validate WebAssembly Output" job
   - Improved error reporting for fixture validation
   - 28/29 test fixtures now pass validation

3. ✅ Dead Code Elimination (DCE) Bug Fix
   - Fixed type mismatch when removing unreachable code
   - Blocks with result types now get `unreachable` instruction
   - Terminators in blocks properly mark following code unreachable

### Known Issues (Need Investigation)

**Issue #?: Optimizer Stack Analysis Bugs**

Two test fixtures fail WASM validation due to stack type mismatches:
- `simplify_locals_test.wat` - func $nested_blocks
- `vacuum_test.wat` - func $mixed_patterns

**Problem Statement:**
The simplify_locals and vacuum passes process nested blocks without validating stack balance. 
Values get left on block stacks that expect empty types, or removed when needed.

**Solution Approach (Research Needed):**

To solve properly, we need to implement a **stack analysis pass** that:
1. Tracks how each instruction affects the value stack
2. Validates block transformations preserve stack invariants
3. Ensures all control flow paths produce correct stack types

**Questions for Implementation:**
- Should stack analysis be a separate pass that validates before/after optimization?
- Should we add stack checks into each optimization pass?
- Can we use Z3 SMT solver to verify stack properties are preserved?
- What's the minimum viable stack analysis for correct optimization?

**Recommended Research:**
1. Study Binaryen's stack analysis approach
2. Investigate WebAssembly validator's stack tracking
3. Evaluate Z3 for stack property verification
4. Check if we can extend ISLE rules for stack validation

This is a foundational issue - fixing it properly will make all future optimizations more robust.

## Proposed GitHub Issue

**Title**: Implement stack analysis validation for optimizer passes

**Description**:
The optimizer's simplify_locals and vacuum passes create invalid WASM by not validating stack balance when processing nested blocks. This causes type mismatches in 2 test fixtures.

**What needs to happen**:
Implement a proper stack analysis framework that validates all optimization passes preserve the invariant that blocks/instructions have correct stack depth before and after transformation.

**Why this matters**:
- 28/29 test fixtures currently pass (DCE fix improved from 26/29)
- Stack analysis is foundational - every optimizer pass depends on it
- Enables use of Z3 for formal verification of stack properties

**Suggested approach** (needs research):
1. Add stack depth calculation to instruction analysis
2. Create pre/post validation for block transformations
3. Investigate if ISLE rules can encode stack properties
4. Consider Z3 solver for stack property preservation proofs

**Resources for research**:
- Binaryen's stack analysis (C++)
- WASM validator source code
- Z3 arithmetic/bit-vector theories for stack modeling

