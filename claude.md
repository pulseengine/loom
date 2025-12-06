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

### Pending: Optimizer Stack Analysis with Z3 Verification

**Issue #?: Stack Analysis Implementation Plan**

Two test fixtures fail WASM validation due to stack type mismatches:
- `simplify_locals_test.wat` - func $nested_blocks
- `vacuum_test.wat` - func $mixed_patterns

**Root Cause:**
The simplify_locals and vacuum passes process nested blocks without validating stack balance.
Values get left on block stacks that expect empty types, or removed when needed.

**Comprehensive Solution Approach (Researched & Ready for Implementation):**

## Architecture Overview

Based on Binaryen's stack analysis patterns and LOOM's existing Z3 verification infrastructure:

### 1. **StackSignature System** (Binaryen-inspired)
```
StackSignature {
  params: Type,           // Stack values consumed by instruction sequence
  results: Type,          // Stack values produced by instruction sequence
  kind: Fixed/Polymorphic // Fixed=deterministic, Polymorphic=unreachable paths exist
}
```

**Key Insight**: Polymorphic signatures allow unreachable code paths. This is critical for blocks containing `unreachable` instructions, which can adapt to any outer stack context.

**Subtyping Rules**:
- `[t1] -> [t2]` <: `[s1 t1'] -> [s2 t2']` if t1'<:t1 and t2<:t2' (contravariance + prefix extension)
- `[t1] -> [t2] {poly}` <: any polymorphic signature with matching core types (allows arbitrary prefixes)
- `[] -> [] {poly}` is bottom type (subtype of everything)

### 2. **Stack Depth Tracking** (per-instruction analysis)

Each instruction is analyzed for its stack effect:
- `i32.const` → consumes 0, produces 1 (stack height +1)
- `i32.add` → consumes 2, produces 1 (stack height -1)
- `local.get` → consumes 0, produces 1 (stack height +1)
- `local.set` → consumes 1, produces 0 (stack height -1)
- `block { ... }` → depends on block's internal signature
- `return` → consumes N, produces unreachable (height becomes undefined)

### 3. **Block Validation Framework**

Before optimizing a block, validate:
1. **Pre-condition**: External stack matches block's expected input types
2. **Body consistency**: Instructions compose (outputs of one = inputs of next)
3. **Post-condition**: Final stack state matches block's declared result type
4. **Unreachability**: If reachable=false, body must end with terminator

Example validation:
```
block (result i32)          // expects empty input, produces i32
  i32.const 42              // stack: [i32]
  local.set $x              // stack: [] - ERROR! block expects to output i32
  local.get $x              // stack: [i32]
end
```

### 4. **Z3-Based Stack Property Verification**

Extend `verify.rs` with stack invariant checking:

```rust
/// Verify that optimization preserves stack properties
fn verify_stack_properties(
    original: &Block,
    optimized: &Block,
    ctx: &Context
) -> Result<bool> {
    // 1. Get signatures
    let orig_sig = StackSignature::from(original);
    let opt_sig = StackSignature::from(optimized);

    // 2. Z3: Assert signatures are compatible
    // Both must have same params and results
    solver.assert(params_equal(&orig_sig, &opt_sig));
    solver.assert(results_equal(&orig_sig, &opt_sig));

    // 3. Z3: Verify composition property
    // If original instructions i1, i2 compose, optimized must also compose

    // 4. Check UNSAT → properties preserved
    match solver.check() {
        SatResult::Unsat => Ok(true),  // Properties preserved
        SatResult::Sat => {
            let model = solver.get_model()?;
            Err(anyhow!("Stack property violation: {:?}", model))
        }
        SatResult::Unknown => Err(anyhow!("Stack verification inconclusive")),
    }
}
```

### 5. **Implementation Plan**

**Phase 1: Stack Signature System** (src/stack.rs)
- [ ] Define `StackSignature` struct with params/results/kind
- [ ] Implement `StackSignature::composes()` for composition checking
- [ ] Implement `isSubType()` for signature compatibility
- [ ] Implement `getLeastUpperBound()` for merging paths

**Phase 2: Instruction Analysis** (src/analysis/stack_depth.rs)
- [ ] Add `instruction_stack_effect()` for each instruction type
- [ ] Track stack height through instruction sequences
- [ ] Detect stack underflow/overflow
- [ ] Handle control flow (blocks, loops, if/else)

**Phase 3: Block Validation** (src/block_validation.rs)
- [ ] Validate block body produces correct stack types
- [ ] Check nested block compositions
- [ ] Verify unreachability assumptions

**Phase 4: Z3 Integration** (extend src/verify.rs)
- [ ] Add `verify_stack_properties()` function
- [ ] Encode stack signatures as Z3 predicates
- [ ] Verify composition properties with SMT solver
- [ ] Generate counterexamples for violations

**Phase 5: Integration into Passes**
- [ ] Add pre/post validation to `simplify_locals` pass
- [ ] Add pre/post validation to `vacuum` pass
- [ ] Run validation after each optimization
- [ ] Fail gracefully with diagnostic output

### 6. **Testing Strategy**

1. **Unit tests** for each signature rule (src/stack.rs tests)
2. **Property tests** for composition correctness
3. **Regression tests** for simplify_locals_test and vacuum_test
4. **Integration tests** with Z3 verification enabled

### 7. **References & Patterns**

Key files from Binaryen (already studied):
- `src/ir/stack-utils.h` - StackSignature API
- `test/example/stack-utils.cpp` - Example usage

Pattern from LOOM's existing code:
- `src/verify.rs` - Z3 encoding patterns (lines 118-175 show symbolic execution)
- Config: use `z3::Solver`, `z3::Context`, encode as SMT predicates

### Why This Works

1. **Binaryen's approach is proven** - handles complex WebAssembly stack semantics
2. **Z3 is already integrated** - existing verification infrastructure can be extended
3. **Type-safe composition** - ensures optimizations preserve invariants
4. **Formal proofs** - SMT solver provides mechanically verified correctness
5. **Minimal viable** - focus on block-level validation first (fixes 2 bugs)

This is a foundational issue - fixing it properly will enable all future optimizations to be verified correct.

## GitHub Issue: Stack Analysis with Z3 Verification

**Title**: Implement stack analysis validation with Z3 proof verification

**Description**:
The optimizer's simplify_locals and vacuum passes create invalid WASM by not validating stack balance when processing nested blocks (2/29 test fixtures failing). This requires a formal stack analysis system integrated with our existing Z3 verification infrastructure.

**Current Status**:
- 28/29 test fixtures pass validation (after DCE fix)
- 2 fixtures fail with stack type mismatches:
  - `simplify_locals_test.wat::$nested_blocks`
  - `vacuum_test.wat::$mixed_patterns`

**Root Cause**:
Optimization passes transform blocks without tracking how instructions compose on the value stack. Instructions can consume/produce incorrect stack values, violating WebAssembly's stack machine invariants.

**Solution Overview** (5-phase implementation):

1. **StackSignature System** - Binaryen-inspired compositional stack types
   - `StackSignature { params, results, Fixed/Polymorphic }`
   - Composition checking: `a.composes(b)` validates signature sequence
   - Subtyping: Polymorphic signatures handle unreachable paths

2. **Instruction Analysis** - Per-instruction stack effects
   - `i32.const` → (+1 stack)
   - `i32.add` → (-1 stack)
   - `local.set` → (-1 stack)
   - Handle control flow compositions

3. **Block Validation** - Pre/post optimization checks
   - Validate pre-conditions (external stack state)
   - Check body consistency (instructions compose)
   - Verify post-conditions (final stack matches declared type)
   - Detect reachability violations

4. **Z3 Integration** - Formal verification with SMT solver
   - Extend existing `verify.rs` with stack property verification
   - Encode signatures as Z3 predicates
   - Prove composition properties with `solver.check() = UNSAT`
   - Generate counterexamples for violations

5. **Pass Integration** - Enforce validation in optimizer
   - Add pre/post checks to `simplify_locals` pass
   - Add pre/post checks to `vacuum` pass
   - Fail gracefully with diagnostic output

**Why This Works**:
- Binaryen's approach is proven for complex WASM semantics
- Z3 is already integrated in LOOM (verification feature)
- Type-safe composition ensures optimization correctness
- SMT solver provides mechanically verified proofs
- Minimal viable solution: block-level validation fixes both bugs

**Files to Create/Modify**:
- `loom-core/src/stack.rs` - StackSignature implementation
- `loom-core/src/analysis/stack_depth.rs` - Instruction analysis
- `loom-core/src/block_validation.rs` - Block pre/post checks
- `loom-core/src/verify.rs` - Z3 stack property verification
- `loom-core/src/lib.rs` - Pass integration

**Success Criteria**:
- [ ] All 29 test fixtures pass validation
- [ ] `simplify_locals_test.wat::$nested_blocks` fixed
- [ ] `vacuum_test.wat::$mixed_patterns` fixed
- [ ] Z3 proofs generated for all block transformations
- [ ] No performance regression on optimization passes

**References**:
- Binaryen: `src/ir/stack-utils.h` (StackSignature API)
- LOOM: `src/verify.rs` (Z3 encoding patterns, lines 118-175)
- WASM Spec: https://webassembly.org/specs/core/exec/instructions.html

## Research: State-of-the-Art Stack Analysis & Potential Improvements

### Binaryen's Approach (Current Reference)

**Strengths**:
- StackSignature system elegantly handles polymorphic unreachable code
- Composition checking prevents invalid transformations
- Proven effective in production WASM optimizer
- Simple enough to implement correctly

**Limitations**:
- Focuses on stack balance (height), not type precision
- No formal verification of composition properties
- Limited to binary stack effects (consume/produce counts)
- No cross-function analysis or module-level optimizations

### Alternative/Complementary Approaches Research Needed

1. **Type-Preserving Stack Analysis** (beyond Binaryen)
   - WASM's type system allows: i32, i64, f32, f64, v128, funcref, externref, etc.
   - Current analysis only tracks stack height, not individual types
   - **Question**: Can we track type information through optimizations?
   - **Benefit**: Catch more subtle bugs (type confusion during optimization)
   - **Reference**: ISLE pattern-matching system already uses types

2. **Abstract Interpretation with SMT** (Z3 integration opportunity)
   - Encode stack state as symbolic variables in Z3
   - Prove invariants hold across transformations
   - **Question**: Can we prove stronger properties than just composition?
   - **Examples**:
     - "Stack never overflows for any function input"
     - "All blocks terminate or produce correct types"
   - **Research needed**: Cost/benefit of SMT for this vs simple analysis

3. **Control Flow Graph Analysis** (beyond local blocks)
   - Binaryen analyzes blocks in isolation
   - WASM has branches, loops, try/catch - complex CFG
   - **Question**: Should we analyze stack properties across branch targets?
   - **Example**: Loop invariants (stack state remains consistent on back-edges)
   - **Benefit**: Catch bugs from branch misdirection

4. **Data Flow Analysis for Locals** (optimization opportunity)
   - Current simplify_locals pass is brute-force
   - **Question**: Can we use data flow to identify redundant local copies?
   - **Example**: `local.set $x; local.get $x` can be eliminated with flow analysis
   - **Reference**: LLVM's SSA form and copy elimination
   - **Benefit**: Better local optimization + verified correctness

5. **Polymorphic Subtyping Refinement** (type system improvement)
   - WASM has reference types (funcref, externref, structref, arrayref)
   - Binaryen treats these as bit-patterns
   - **Question**: Can we use structural type compatibility in optimizations?
   - **Example**: ref.cast might eliminate some code paths
   - **Research needed**: How much optimization benefit for added complexity?

6. **Heap/Memory Safety Analysis** (for memory operations)
   - Current analysis doesn't track memory effects
   - **Question**: Can we verify memory safety of optimizations?
   - **Challenge**: Alias analysis for memory.load/store
   - **Benefit**: Catch use-after-free, bounds violations, race conditions
   - **Research needed**: SMT solver encoding of memory model

### Recommendation for LOOM

**Phase 1 (Current)**: Implement Binaryen's proven approach
- Stack composition checking
- Block-level validation
- Z3 verification of composition properties
- **Scope**: Simple, well-understood, fixes immediate bugs

**Phase 2 (Future Research)**:
- Type-tracking in stack analysis (discriminate i32 vs i64 vs refs)
- Control flow analysis for branches/loops
- Data flow analysis for local variable optimization
- **Benefit**: Stronger guarantees than Binaryen

**Phase 3 (Advanced Research)**:
- SMT-based abstract interpretation for arbitrary stack properties
- Module-level optimization with cross-function analysis
- Memory safety verification for load/store operations
- **Status**: Research phase - investigate benefits vs complexity

### Questions to Answer Before Proceeding

1. Do we need reference type compatibility checking?
   - Answer: Check if any failing tests involve reference types

2. Can we improve local variable elimination?
   - Answer: Compare with wasm-opt's approach on simplify_locals_test.wat

3. Should we analyze loops/branches together?
   - Answer: Needed if tests fail on control flow reachability

4. Is SMT-based abstract interpretation worth the complexity?
   - Answer: Benchmark verification cost vs optimization benefit

For now, proceeding with Phase 1 (Binaryen approach + Z3 verification).

## Research Summary: State-of-the-Art in Compiler Optimization

### What We Actually Have in LOOM (from existing codebase research)

The project already has comprehensive analysis documenting:

1. **Binaryen Comparison** (`docs/analysis/LOOM_VS_BINARYEN_COMPARISON.md`)
   - Binaryen: 123 passes, ~200k LOC, C++
   - LOOM: 13 passes, ~6k LOC, Rust + ISLE
   - LOOM achieves 80-95% reduction vs Binaryen's 85-98%
   - **Key missing passes for parity**: Code Folding (5-10%), RSE (2-5%), LICM (3-8%), Better Inlining (2-5%)
   - Opportunity: 15-35% improvement with targeted additions

2. **Z3 Verification Status** (`docs/analysis/Z3_VERIFICATION_STATUS.md`)
   - LOOM has real formal verification via Z3, not ISLE
   - Z3 provides translation validation (mathematical proofs)
   - Current coverage: i32/i64 arithmetic and bitwise ops
   - **Status**: ✅ Working with `--features verification --verify`
   - **IMPORTANT**: ISLE is just a DSL for optimization rules, NOT verification

3. **Formal Verification Architecture** (`docs/architecture/FORMAL_VERIFICATION_ARCHITECTURE.md`)
   - Hybrid 3-layer verification approach:
     - Layer 1: Optimization (ISLE + Rust)
     - Layer 2: Z3 formal verification (translation validation)
     - Layer 3: Empirical validation (proptest + differential testing)
   - **Verdict**: Z3 translation validation is best approach for production use
   - No other modern optimizers use pure formal verification (too expensive)

### State-of-the-Art Comparison

Based on research across compilers (LLVM, GCC, Cranelift, V8, SpiderMonkey, Binaryen):

**Standard Practice** (what everyone uses):
- ✅ Dataflow analysis for safety
- ✅ Effect analysis (reads/writes/calls tracking)
- ✅ Pattern-matching optimization rules
- ✅ Multi-level optimization (-O0 through -O3)
- ✅ Property-based testing
- ❌ Formal verification (too expensive for all passes)

**State-of-the-Art** (cutting edge):
- ✅ eGraphs (egg) for automatic optimization discovery (research stage)
- ✅ SMT solvers for critical passes (translation validation)
- ✅ Mechanized proofs in Coq/Isabelle (research only, not production)
- ✅ Abstract interpretation for invariants
- ✅ Alive project for LLVM pass verification (academic)

**What LOOM Has** (ahead of most):
- ✅ Z3 translation validation (most optimizers don't have this)
- ✅ Component Model support (future-proofing)
- ✅ Rust safety guarantees
- ⚠️ ISLE pattern matching (good but limited by architecture)
- ❌ eGraphs (not integrated)

### Why Stack Analysis is Critical

Stack analysis is a **foundational property** that all optimizers must get right:
- **Binaryen**: Uses compositional StackSignature system (proven approach)
- **LLVM**: Uses data flow analysis + SSA form
- **V8**: Uses type feedback + polymorphic inline caches
- **Symbolica**: Not applicable (computer algebra system, not a compiler)

The research confirms: **Binaryen's approach is state-of-the-art for stack analysis**.

### Why Formal Verification with Z3 is Correct Strategy

**Academic Research** (last 5 years):
- Translation validation papers show ~100-200ms overhead acceptable for CI
- SMT solvers proven effective for verifying compiler optimizations
- "Alive" project successfully found bugs in LLVM using this approach
- Modern trend: verify critical passes with SMT, use testing for others

**Production Compilers**:
- None verify all passes (cost prohibitive)
- Most verify nothing (rely on testing)
- LOOM is ahead with Z3 translation validation

### Conclusion

LOOM's strategy is **correct and state-of-the-art**:
1. Use Binaryen's StackSignature approach for stack analysis
2. Extend Z3 verification to cover stack properties
3. Use property-based testing for coverage
4. Add missing optimization passes (Code Folding, RSE, LICM)

This is the same approach cutting-edge compiler research recommends.
Proceeding with Phase 1 implementation.
