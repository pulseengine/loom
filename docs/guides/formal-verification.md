# Formal Verification Guide for LOOM

## Overview

This document outlines formal verification approaches for the LOOM WebAssembly optimizer, going beyond ISLE to ensure correctness of optimization passes.

## Current State

LOOM currently uses:
- **ISLE DSL**: Term rewriting with pattern matching
- **Property-Based Testing**: 256 test cases per property using proptest
- **Round-trip Testing**: Parse-optimize-encode-validate cycles

## Verification Approaches for LOOM

### 1. Translation Validation (Recommended for LOOM)

**Approach**: Validate each individual optimization run rather than proving the optimizer correct for all inputs.

**Benefits**:
- Works with existing optimizer code
- No need to verify ISLE compiler
- Catches bugs in actual transformations
- Incremental - can add validation per-pass

**Implementation**:
```rust
pub fn verify_optimization(before: &Module, after: &Module) -> Result<bool> {
    // 1. Encode both modules to SMT
    let before_smt = encode_module_to_smt(before);
    let after_smt = encode_module_to_smt(after);

    // 2. Assert equivalence
    let equiv = smt_equivalent(before_smt, after_smt)?;

    Ok(equiv)
}
```

**SMT Encoding for WebAssembly**:
- i32/i64 → BitVec[32]/BitVec[64]
- Operations map directly to SMT-LIB2 bitvector ops
- Control flow → SSA with phi nodes
- Memory → Array theory

### 2. Z3 SMT Solver Integration

**Use Cases**:
1. Verify constant folding: `i32.add(10, 32) ≡ i32.const(42)`
2. Verify algebraic laws: `x + 0 ≡ x`
3. Verify strength reduction: `x * 4 ≡ x << 2`
4. Verify CSE: Duplicate expressions eliminated safely

**Example**:
```rust
use z3::*;

fn verify_constant_fold() {
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let solver = Solver::new(&ctx);

    // Before: i32.add(i32.const 10, i32.const 32)
    let before = BV::from_i64(&ctx, 10, 32).bvadd(&BV::from_i64(&ctx, 32, 32));

    // After: i32.const 42
    let after = BV::from_i64(&ctx, 42, 32);

    // Assert they're NOT equal (looking for counterexample)
    solver.assert(&before._eq(&after).not());

    // If UNSAT, they're equivalent
    assert_eq!(solver.check(), SatResult::Unsat);
}
```

### 3. egg E-Graphs (Equality Saturation)

**Approach**: Use equality graphs to explore all equivalent programs, extract optimal.

**Benefits**:
- Finds optimizations humans might miss
- Provably optimal for given rewrite rules
- Compositional - rules combine automatically

**Implementation**:
```rust
use egg::*;

define_language! {
    enum Wasm {
        // Constants
        "i32.const" = I32Const(i32),

        // Binary ops
        "i32.add" = I32Add([Id; 2]),
        "i32.mul" = I32Mul([Id; 2]),
        "i32.shl" = I32Shl([Id; 2]),

        // Variables
        Symbol(Symbol),
    }
}

fn make_rules() -> Vec<Rewrite<Wasm, ()>> {
    vec![
        rewrite!("add-zero"; "(i32.add ?x (i32.const 0))" => "?x"),
        rewrite!("mul-one"; "(i32.mul ?x (i32.const 1))" => "?x"),
        rewrite!("mul-pow2"; "(i32.mul ?x (i32.const 4))" =>
                             "(i32.shl ?x (i32.const 2))"),
        rewrite!("commute-add"; "(i32.add ?a ?b)" => "(i32.add ?b ?a)"),
    ]
}
```

### 4. Crocus-Style Lightweight Verification

**Approach**: Verify ISLE rules themselves using SMT, not runtime behavior.

**Process**:
1. Extract ISLE rule LHS and RHS
2. Convert to SMT formulas
3. Prove `∀ inputs. eval(LHS) = eval(RHS)`

**Example Rule**:
```isle
;; Rule: (i32.add (iconst32 x) (iconst32 y)) => (iconst32 (imm32_add x y))

;; SMT encoding:
(declare-const x (_ BitVec 32))
(declare-const y (_ BitVec 32))

;; LHS eval
(define-fun lhs () (_ BitVec 32) (bvadd x y))

;; RHS eval
(define-fun rhs () (_ BitVec 32) (bvadd x y))

;; Prove equivalence
(assert (not (= lhs rhs)))
(check-sat) ; Should be UNSAT
```

### 5. Isabelle/HOL - Full Mechanized Proof

**Approach**: Formalize WebAssembly semantics and prove optimizer correct.

**Scope**: Too heavyweight for LOOM initially, but interesting for future.

**Prior Art**:
- WebAssembly mechanized spec exists in Isabelle
- Could build on this foundation
- Months of effort required

### 6. Coq/CompCert Style

**Approach**: Write optimizer in Gallina, extract to Rust/OCaml.

**Challenges**:
- ISLE integration unclear
- Performance overhead
- Large verification burden

## Recommended Implementation Plan for LOOM

### Phase 1: SMT Translation Validation (Weeks 1-2)

1. Add `z3` crate dependency
2. Implement SMT encoding for:
   - i32/i64 constants and arithmetic
   - Bitwise operations
   - Comparisons
3. Create `verify_pass()` function
4. Add `--verify` flag to CLI
5. Test on constant folding pass

**Deliverable**: Can verify simple optimizations

### Phase 2: Comprehensive SMT Support (Weeks 3-4)

1. Extend to all WASM instructions
2. Handle control flow (SSA conversion)
3. Add memory operations
4. Implement timeout handling
5. Add counterexample reporting

**Deliverable**: Can verify all current optimizations

### Phase 3: egg Integration (Weeks 5-6)

1. Define WASM e-graph language
2. Port ISLE rules to egg rewrites
3. Implement extraction with cost function
4. Benchmark vs current approach
5. Integrate as alternative optimizer

**Deliverable**: POC equality saturation optimizer

### Phase 4: Crocus for ISLE (Weeks 7-8)

1. Parse ISLE files
2. Extract rule LHS/RHS
3. Generate SMT verification conditions
4. Verify all rules at build time
5. Produce verification report

**Deliverable**: Build-time rule verification

### Phase 5: Property-Based Equivalence (Week 9)

1. Generate random WASM programs
2. Optimize with LOOM
3. Compare execution with reference interpreter
4. Report any divergence

**Deliverable**: Fuzzing-based correctness testing

## Verification Condition Examples

### Constant Folding

```smt2
; Verify: i32.add(10, 32) = 42
(declare-const result (_ BitVec 32))
(assert (= result (bvadd #x0000000a #x00000020)))
(assert (not (= result #x0000002a)))
(check-sat) ; UNSAT = correct
```

### Strength Reduction

```smt2
; Verify: x * 4 = x << 2
(declare-const x (_ BitVec 32))
(define-fun mul-version () (_ BitVec 32)
    (bvmul x #x00000004))
(define-fun shift-version () (_ BitVec 32)
    (bvshl x #x00000002))
(assert (not (= mul-version shift-version)))
(check-sat) ; UNSAT = equivalent
```

### CSE Safety

```smt2
; Verify: Two identical pure expressions can share result
(declare-const a (_ BitVec 32))
(declare-const b (_ BitVec 32))

; First computation
(define-fun expr1 () (_ BitVec 32) (bvadd a b))

; Second computation
(define-fun expr2 () (_ BitVec 32) (bvadd a b))

; Must be equal
(assert (not (= expr1 expr2)))
(check-sat) ; UNSAT = safe to deduplicate
```

## Performance Considerations

**SMT Solver Overhead**:
- Verification slower than optimization
- Only run in development/CI, not production
- Timeout after reasonable period (e.g., 10s per function)

**egg Performance**:
- E-graph construction: O(n log n)
- Saturation: Can be exponential but usually practical
- Extraction: O(n) with greedy algorithm

## Integration with LOOM Pipeline

```rust
pub fn optimize_with_verification(module: &mut Module) -> Result<()> {
    let original = module.clone();

    // 1. Run optimizations
    optimize_module(module)?;

    // 2. Verify (if enabled)
    if cfg!(feature = "verify") {
        if !verify_optimization(&original, module)? {
            return Err(anyhow!("Optimization verification failed"));
        }
    }

    Ok(())
}
```

## Testing Strategy

1. **Unit Tests**: Verify individual rules
2. **Integration Tests**: Verify full optimization passes
3. **Regression Tests**: Historical bugs stay fixed
4. **Fuzz Tests**: Random programs
5. **Differential Testing**: Compare with wasm-opt

## References

- **CompCert**: https://compcert.org/
- **CakeML**: https://cakeml.org/
- **egg**: https://egraphs-good.github.io/
- **Crocus**: ASPLOS 2024 paper
- **Z3**: https://github.com/Z3Prover/z3
- **Isabelle WebAssembly**: https://www.cl.cam.ac.uk/~caw77/

## Conclusion

For LOOM, **translation validation with Z3** provides the best balance of:
- Ease of implementation
- Effectiveness at catching bugs
- Integration with existing code

With egg as a research direction for discovering new optimizations, and Crocus-style verification for ISLE rules at build time.
