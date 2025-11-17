# LOOM Implementation Plan

## Executive Summary

Based on comprehensive research of the LOOM codebase and GitHub issues, this document outlines a strategic implementation plan to close the gap with wasm-opt.

## Current Status Summary

### ✅ What's Working (46/47 tests pass)

**Control Flow** (~90% complete):
- Parser, encoder, and instruction representation for Block/Loop/If/Br/Call/Return
- Round-trip conversion working
- Dead code elimination, branch simplification, block merging all handle control flow
- **Missing**: ISLE term representation for control flow

**CSE (Common Subexpression Elimination)** (MVP):
- Hash-based duplicate detection for constants
- Local variable caching with local.tee/local.get
- **Missing**: Full expression CSE, commutative matching, cross-block

**Optimization Pipeline** (8 passes):
1. Precompute (global const prop)
2. ISLE optimization (constant folding)
3. CSE (constants only)
4. Branch simplification
5. Dead code elimination
6. Block merging
7. Vacuum (cleanup)
8. SimplifyLocals

**Testing**: Property-based (256 cases), round-trip, idempotence, validity

## Priority Implementation Roadmap

### Phase 1: Quick Wins (2-3 weeks) - Issue #21

**Advanced Instruction Optimization** - Well-defined patterns, high ROI

Strength Reduction:
```rust
// x * 4 → x << 2 (if 4 is power of 2)
I32Mul(x, I32Const(4)) => I32Shl(x, I32Const(2))

// x / 4 → x >> 2 (unsigned, power of 2)
I32DivU(x, I32Const(4)) => I32ShrU(x, I32Const(2))

// x % 4 → x & 3 (power of 2)
I32RemU(x, I32Const(4)) => I32And(x, I32Const(3))
```

Bitwise Tricks:
```rust
// x ^ x → 0
I32Xor(x, x) => I32Const(0)

// x & x → x
I32And(x, x) => x

// x | x → x
I32Or(x, x) => x

// x & 0 → 0
I32And(x, I32Const(0)) => I32Const(0)

// x | 0xFFFFFFFF → 0xFFFFFFFF
I32Or(x, I32Const(-1)) => I32Const(-1)
```

**Impact**: 30-40% of code has these patterns
**Difficulty**: Low - pure pattern matching on Instructions
**Tests**: Add to existing property-based framework

### Phase 2: CSE Enhancement (2-3 weeks) - Issue #19

**Extend CSE Beyond Constants**

Hash Function for All Expressions:
```rust
fn hash_instruction(inst: &Instruction) -> u64 {
    match inst {
        I32Add => hash("i32.add" + hash(left) + hash(right)),
        I32Mul => hash("i32.mul" + hash(left) + hash(right)),
        // ... all operations
    }
}
```

Commutative Matching:
```rust
// Recognize (a + b) == (b + a)
fn is_commutative_match(inst1, inst2) -> bool {
    match (inst1, inst2) {
        (I32Add(a, b), I32Add(c, d)) =>
            (a == c && b == d) || (a == d && b == c),
        // ... other commutative ops
    }
}
```

**Impact**: 20-30% of code has duplicate computations
**Difficulty**: Medium - need expression hashing, dominance
**Tests**: CSE on arithmetic, bitwise, comparisons, cross-block

### Phase 3: Function Inlining (3-4 weeks) - Issue #14 **CRITICAL**

**40-50% of code benefits from inlining**

Call Graph Construction:
```rust
struct CallGraph {
    functions: Vec<FunctionInfo>,
    call_sites: HashMap<u32, Vec<CallSite>>,
}

fn build_call_graph(module: &Module) -> CallGraph {
    // Scan all functions for Call instructions
    // Track who calls whom
}
```

Inlining Heuristics:
```rust
fn should_inline(func: &Function, call_sites: &[CallSite]) -> bool {
    // Always inline if:
    // - Single call site
    // - Size < threshold (e.g., 50 instructions)
    // - No recursion

    // Never inline if:
    // - Recursive or mutually recursive
    // - Indirect calls only
    // - Too large (>200 instructions)
}
```

Parameter Substitution:
```rust
fn inline_function(
    call_site: &CallSite,
    callee: &Function,
    args: &[Instruction]
) -> Vec<Instruction> {
    // 1. Clone callee body
    // 2. Remap local indices (avoid conflicts)
    // 3. Replace LocalGet(param_i) with args[i]
    // 4. Handle Return → branch to end
    // 5. Insert inlined body at call site
}
```

**Impact**: Enables constant prop across functions, removes call overhead
**Difficulty**: High - complex transformation, many edge cases
**Tests**: Simple inline, params, locals, returns, recursion detection

### Phase 4: Code Folding (2 weeks) - Issue #22

**Linearize Nested Expressions**

Use-Count Analysis:
```rust
fn analyze_local_usage(func: &Function) -> HashMap<u32, UsageInfo> {
    // Track how many times each local is used
    // Identify single-use temporaries
}
```

Fold Single-Use Temps:
```rust
// Before:
// local.set $tmp (i32.add $x $y)
// call $f (local.get $tmp)

// After:
// call $f (i32.add $x $y)
```

Block Flattening:
```rust
// Remove unnecessary nesting
// Merge blocks with compatible types
// Reduce stack depth requirements
```

**Impact**: 20-25% of code, improves readability
**Difficulty**: Medium - dataflow analysis required
**Tests**: Folding, flattening, nested blocks

### Phase 5: ISLE Control Flow (1-2 weeks) - Issue #12

**Add Control Flow to ISLE Terms**

Extend wasm_terms.isle:
```isle
;; Block type
(type BlockType (enum
  (Empty)
  (Value (ty ValType))
  (Func (params (list ValType)) (results (list ValType)))))

;; Control flow constructs
(decl block (label (option String)) (ty BlockType) (body (list Value)) Value)
(decl loop_construct (label (option String)) (ty BlockType) (body (list Value)) Value)
(decl if_then_else (cond Value) (ty BlockType) (then_body (list Value)) (else_body (list Value)) Value)

;; Branches
(decl br (depth u32) (val (option Value)) Value)
(decl br_if (cond Value) (depth u32) (val (option Value)) Value)
```

Optimization Rules:
```isle
;; Constant condition folding
(rule (if_then_else (iconst32 0) ty then else)
      else) ; false → take else

(rule (if_then_else (iconst32 ?n) ty then else)
      then  ; non-zero → take then
      (if (!= ?n 0)))
```

**Impact**: Enables future CFG-based optimizations
**Difficulty**: Medium - integration with existing ISLE
**Tests**: Control flow round-trip with ISLE

### Phase 6: Loop Optimizations (3-4 weeks) - Issue #23

**High-Value CFG-Based Passes**

Loop Invariant Code Motion (LICM):
```rust
// Move computations that don't change in loop outside
for (loop $L
  (local.set $a (i32.add (local.get $x) (i32.const 10)))  // Invariant!
  (local.set $i (i32.add (local.get $i) (i32.const 1)))
  (br_if $L ...))

// After LICM:
(local.set $a (i32.add (local.get $x) (i32.const 10)))  // Hoisted
(loop $L
  (local.set $i (i32.add (local.get $i) (i32.const 1)))
  (br_if $L ...))
```

Loop Unrolling:
```rust
// Small, known-count loops → unroll
(loop $L (result i32)
  ... body ...
  (br_if $L (i32.lt_u (local.get $i) (i32.const 4))))

// → Unroll 4 times, remove loop
```

**Impact**: Critical for numerical code
**Difficulty**: High - needs loop detection, dominance
**Tests**: LICM correctness, unrolling limits

## Verification Strategy

### Phase V1: Z3 SMT Integration (parallel with main development)

Translation Validation:
```rust
pub fn verify_optimization(
    before: &Module,
    after: &Module
) -> Result<bool> {
    let ctx = z3::Context::new(&z3::Config::new());
    let solver = z3::Solver::new(&ctx);

    // Encode both modules to SMT
    let before_formula = encode_to_smt(before, &ctx);
    let after_formula = encode_to_smt(after, &ctx);

    // Assert they're NOT equal (looking for counterexample)
    solver.assert(&before_formula._eq(&after_formula).not());

    // UNSAT = equivalent (no counterexample exists)
    match solver.check() {
        SatResult::Unsat => Ok(true),
        SatResult::Sat => {
            let model = solver.get_model();
            eprintln!("Counterexample: {:?}", model);
            Ok(false)
        }
        SatResult::Unknown => Err(anyhow!("SMT timeout")),
    }
}
```

**Integration**: Add `--verify` flag to CLI
**Scope**: Run on CI for regression tests
**Timeline**: 2-3 weeks

### Phase V2: egg E-Graphs POC (research)

Equality Saturation:
```rust
use egg::*;

// Explore ALL equivalent programs
let mut egraph = EGraph::default();
let expr_id = egraph.add_expr(/* input program */);

let runner = Runner::default()
    .with_egraph(egraph)
    .run(&make_wasm_rules());

// Extract optimal program
let (cost, best) = Extractor::new(&runner.egraph, AstSize)
    .find_best(expr_id);
```

**Goal**: Discover optimizations humans might miss
**Timeline**: 2-3 weeks
**Deliverable**: Comparison with current optimizer

## Testing Strategy

### Unit Tests
- One test per optimization pattern
- Edge cases (overflow, signed/unsigned, etc.)

### Integration Tests
- WAT fixtures for each optimization pass
- Round-trip validation
- wasm-tools validate output

### Property-Based Tests (existing)
- Idempotence: optimize(optimize(x)) = optimize(x)
- Validity: output is valid WASM
- Determinism: same input → same output
- Equivalence: semantics preserved (with verification)

### Benchmarks
- Real-world WASM modules
- Measure size reduction
- Measure speed improvement
- Compare with wasm-opt

## Timeline Summary

| Phase | Duration | Cumulative | Impact | Priority |
|-------|----------|------------|--------|----------|
| #21 Advanced Opts | 2-3 weeks | 3 weeks | 30-40% | HIGH |
| #19 CSE Full | 2-3 weeks | 6 weeks | 20-30% | HIGH |
| #14 Inlining | 3-4 weeks | 10 weeks | 40-50% | CRITICAL |
| #22 Folding | 2 weeks | 12 weeks | 20-25% | MEDIUM |
| #12 ISLE CF | 1-2 weeks | 14 weeks | Future | MEDIUM |
| #23 Loops | 3-4 weeks | 18 weeks | High | HIGH |
| V1 Z3 | 2-3 weeks | (parallel) | Correctness | RESEARCH |
| V2 egg | 2-3 weeks | (parallel) | Discovery | RESEARCH |

**Total Core Development**: ~18 weeks (4.5 months)
**Target**: 80-90% of wasm-opt effectiveness

## Success Metrics

1. **Code Size**: Reduce WASM binary size by 30-50%
2. **Test Coverage**: All optimizations property-tested
3. **Correctness**: Z3 verification passes on all test cases
4. **Performance**: Compilation time < 2x wasm-opt
5. **Effectiveness**: 80-90% of wasm-opt optimization impact

## Risks and Mitigation

**Risk**: ISLE integration complexity
- **Mitigation**: Keep optimizations in Rust initially, port to ISLE later

**Risk**: Verification overhead
- **Mitigation**: Only run with `--verify` flag, not in production

**Risk**: Test case explosion
- **Mitigation**: Property-based testing covers large input space automatically

**Risk**: Performance regression
- **Mitigation**: Benchmark suite, CI performance tracking

## Next Actions

1. **Immediate**: Implement Issue #21 (Advanced Opts) - 2-3 weeks
2. **Next**: Enhance Issue #19 (CSE Full) - 2-3 weeks
3. **Then**: Critical Issue #14 (Inlining) - 3-4 weeks
4. **Parallel**: Research Z3 integration - 2-3 weeks

**First Milestone**: Phase 1 + Phase 2 = 6 weeks, ~50-70% effectiveness increase
