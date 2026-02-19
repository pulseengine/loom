# Verification Tools Landscape for Compiler Optimization

**Date**: 2025-12-09
**Status**: Comprehensive analysis of formal verification tools and research directions
**Purpose**: Guide LOOM's verification strategy and identify research opportunities

---

## Executive Summary

This document analyzes the landscape of formal verification tools for compiler optimization, comparing their approaches to LOOM's existing Z3-based translation validation. The analysis covers:

1. **Tool Comparison**: Alive2, CBMC, Frama-C WP, SMACK/Boogie, Rosette
2. **Research Directions**: Novel verification techniques and open problems
3. **Academic Venues**: Conferences, workshops, and journals for publication
4. **Integration Opportunities**: How each tool could enhance LOOM

**Key Finding**: LOOM's Z3 translation validation approach is aligned with state-of-the-art (Alive2). The most promising enhancements are **Rosette for rule synthesis** and **EMI testing for empirical validation**.

---

## 1. Tool Comparison Matrix

### 1.1 Overview

| Tool | Approach | Verification Type | LOOM Fit | Integration Effort |
|------|----------|-------------------|----------|-------------------|
| **Alive2** | Translation validation | Individual passes/rewrites | ✅ Already similar | Reference only |
| **CBMC** | Bounded model checking | C/C++ programs | ⚠️ Different domain | Medium |
| **Frama-C WP** | Deductive verification | C code with contracts | ⚠️ Heavy annotation | High |
| **SMACK/Boogie** | Verification pipeline | Program assertions | ⚠️ Indirect | High |
| **Rosette** | Solver-aided DSL | Relational specs, synthesis | ✅ Rule synthesis | Medium |

### 1.2 Detailed Analysis

---

## 2. Alive2: LLVM Optimization Validation

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Alive2 Architecture                                        │
│                                                              │
│  LLVM IR (before) → SMT Encoding → Z3 → UNSAT = verified   │
│  LLVM IR (after)  → SMT Encoding →─────┘                   │
│                                                              │
│  LOOM Architecture (identical pattern!)                     │
│                                                              │
│  WASM (before) → SMT Encoding → Z3 → UNSAT = verified      │
│  WASM (after)  → SMT Encoding →─────┘                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Features

**What Alive2 Does Well**:
1. **Counterexample generation**: Human-readable inputs that break the optimization
2. **Precondition inference**: Discovers when an optimization is valid (e.g., "only when x ≠ 0")
3. **Batch verification**: Verifies all InstCombine patterns at once
4. **Poison/undef semantics**: Models LLVM's undefined behavior precisely
5. **Memory model**: Supports load/store with aliasing analysis

**Bug-Finding Success**:
- Found **47+ bugs** in LLVM's InstCombine
- Verified correctness of **334 hand-coded transforms**
- 2.4% bug rate in mature, well-tested code

### 2.3 What LOOM Could Learn

**Already Implemented in LOOM**:
- ✅ Translation validation via Z3 (`verify.rs`)
- ✅ Counterexample generation (`verify.rs:266-268`)
- ✅ Function-by-function comparison

**Could Add**:
- [ ] Precondition inference (e.g., "valid when divisor ≠ 0")
- [ ] Build-time ISLE rule verification
- [ ] Poison value semantics for WebAssembly (limited applicability)

### 2.4 References

- **Paper**: [Alive2: Bounded Translation Validation for LLVM](https://dl.acm.org/doi/10.1145/3453483.3454030) (PLDI 2021)
- **Code**: https://github.com/AliveToolkit/alive2
- **Online Tool**: https://alive2.llvm.org/ce/

---

## 3. CBMC: Bounded Model Checking

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  CBMC Workflow                                               │
│                                                              │
│  C/C++ Source → Parse → Unroll Loops → SAT Formula → Solve │
│       ↓                                      ↓               │
│  Assertions           ←──────────────── Counterexample      │
│                                                              │
│  Key Insight: Bounded checking = exhaustive within bound    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Key Features

**What CBMC Does**:
1. **Loop unrolling**: Unrolls all loops to a user-specified bound
2. **Assertion checking**: Verifies assert() statements hold for all executions
3. **Memory safety**: Checks buffer overflows, null dereferences
4. **Concurrency**: Supports pthread verification (bounded threads)
5. **SAT-based**: Encodes entire program as SAT, very fast solving

**Strengths**:
- Fully automatic (no annotations required)
- Fast for bounded verification
- Excellent error traces
- Mature tool (20+ years of development)

**Limitations**:
- Only handles C/C++/Java (not Rust directly)
- Bounded: misses bugs beyond loop bound
- Not for verifying compiler transformations

### 3.3 Relevance to LOOM

**Where CBMC Fits**:
- Verifying **the optimizer implementation itself** (if we had C++ optimizer)
- NOT for verifying WebAssembly transformations (wrong abstraction level)

**LOOM Already Has This**:
- ✅ Bounded loop handling in Z3 (`MAX_LOOP_UNROLL = 3` in `verify.rs:40`)

**Could Add**:
- [ ] Use Kani (Rust model checker based on CBMC) to verify optimizer code
- [ ] Property-based assertions in optimizer passes

### 3.4 References

- **Paper**: [A Tool for Checking ANSI-C Programs](https://link.springer.com/chapter/10.1007/978-3-540-24730-2_15) (TACAS 2004)
- **Code**: https://github.com/diffblue/cbmc
- **Kani (Rust)**: https://github.com/model-checking/kani

---

## 4. Frama-C WP: Deductive Verification

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frama-C Workflow                                            │
│                                                              │
│  C Code + ACSL Annotations → WP Plugin → Z3/Alt-Ergo/CVC4  │
│                                                              │
│  /*@ requires n >= 0;                                       │
│      ensures \result == n * (n + 1) / 2; */                │
│  int sum(int n) { ... }                                     │
│                                                              │
│  WP generates: ∀n ≥ 0. sum(n) = n(n+1)/2                   │
│  Z3 proves: VALID (or finds counterexample)                 │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Key Features

**What Frama-C Does**:
1. **Contract specification**: Pre/post conditions in ACSL language
2. **Loop invariants**: User specifies what holds at each iteration
3. **Separation logic**: Memory ownership tracking
4. **Multiple provers**: Z3, Alt-Ergo, CVC4, Coq
5. **Modular verification**: Prove functions independently

**Strengths**:
- Full functional correctness (not just type safety)
- Once proven, guaranteed forever
- Generates proof obligations automatically

**Limitations**:
- Heavy annotation burden (~2-5× code size)
- Manual loop invariants required
- C-only (though variants exist for other languages)
- Steep learning curve

### 4.3 Relevance to LOOM

**Where Frama-C's Approach Fits**:
- Annotating optimizer passes with contracts
- Proving optimization rules correct once
- NOT practical for all transformations (too much effort)

**Hypothetical LOOM Integration**:
```rust
// Hypothetical: Annotated optimizer pass
/// @requires valid_wasm(original)
/// @ensures semantically_equivalent(original, optimized)
/// @ensures instruction_count(optimized) <= instruction_count(original)
fn optimize_function(original: &Function) -> Function {
    // ...
}
```

**Why We Don't Recommend This**:
- Translation validation (Z3) gives equivalent guarantees with less effort
- Annotation burden scales with code size
- LOOM's approach already provides per-run verification

### 4.4 References

- **Website**: https://frama-c.com/
- **ACSL Reference**: https://frama-c.com/html/acsl.html
- **WP Plugin**: https://frama-c.com/fc-plugins/wp.html

---

## 5. SMACK/Boogie: Verification Pipeline

### 5.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  SMACK/Boogie Pipeline                                       │
│                                                              │
│  C/C++ → LLVM IR → SMACK → Boogie IR → Z3/CVC4/Corral     │
│                                                              │
│  Boogie IR: Intermediate verification language              │
│  - Simple imperative language                               │
│  - goto, assume, assert statements                          │
│  - Weakest precondition calculus                            │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Key Features

**What SMACK Does**:
1. **LLVM to Boogie**: Translates LLVM IR to verification language
2. **Memory modeling**: Multiple memory models (flat, region-based)
3. **Concurrency**: pthread verification support
4. **Assertion checking**: Verifies user assertions

**What Boogie Provides**:
1. **Intermediate language**: Designed for verification
2. **Weakest precondition**: Automatic VC generation
3. **Multiple backends**: Z3, CVC4, Boolector
4. **Dafny**: High-level verified language compiles to Boogie

### 5.3 Relevance to LOOM

**Where SMACK/Boogie Fits**:
- LOOM already has a direct WASM → Z3 path (simpler)
- Boogie adds abstraction layer without clear benefit
- Could theoretically express WASM semantics in Boogie

**Why We Don't Recommend**:
- Extra translation step adds complexity
- No WebAssembly frontend exists
- Direct Z3 integration is more efficient

### 5.4 References

- **SMACK**: https://github.com/smackers/smack
- **Boogie**: https://github.com/boogie-org/boogie
- **Dafny**: https://dafny.org/

---

## 6. Rosette: Solver-Aided Programming ⭐

### 6.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Rosette Capabilities                                        │
│                                                              │
│  1. VERIFICATION: Prove properties for ALL inputs           │
│     (verify (assert (= (f x) (g x))))                       │
│                                                              │
│  2. SYNTHESIS: Generate code that satisfies spec           │
│     (synthesize (λ (x) (??)) spec)                          │
│                                                              │
│  3. ANGELIC EXECUTION: Find inputs satisfying condition    │
│     (solve (assert (> (f x) 10)))                          │
│                                                              │
│  Backend: Lifts Racket to SMT via symbolic execution       │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Key Features

**What Makes Rosette Special**:
1. **Solver-aided DSL**: Write programs that use SMT "for free"
2. **Symbolic execution**: Values can be symbolic, SMT figures out rest
3. **Program synthesis (CEGIS)**: CounterExample-Guided Inductive Synthesis
4. **Verification built-in**: `verify` macro proves properties
5. **Multiple solvers**: Z3, Boolector, CVC4

**Proven Applications**:
- **Herbie**: Floating-point accuracy improvement (2015)
- **Serval**: Security verification for operating systems
- **Neutrons**: Verified radiation therapy controller
- **Bonsai**: Verified BPF compiler

### 6.3 Relevance to LOOM ⭐ HIGH POTENTIAL

**Application 1: Rule Synthesis (CEGIS)**

```racket
; Use Rosette to synthesize optimization rules
(define-sketch strength-reduction
  (λ (x k)
    (if (power-of-2? k)
        (shl x (??))  ; Synthesize the shift amount
        (* x k))))

; Rosette automatically discovers:
; k=4 → shift by 2
; k=8 → shift by 3
; k=16 → shift by 4
; etc.
```

**Application 2: Optimization Discovery**

```racket
; Define WebAssembly semantics
(define (wasm-eval expr)
  (match expr
    [`(i32.add ,a ,b) (bvadd (wasm-eval a) (wasm-eval b))]
    [`(i32.mul ,a ,b) (bvmul (wasm-eval a) (wasm-eval b))]
    [`(i32.shl ,a ,b) (bvshl (wasm-eval a) (wasm-eval b))]
    [`(i32.const ,n) (bv n 32)]
    [x x]))

; Synthesize equivalent expressions
(define (find-equivalent expr)
  (define sketch `(,(choose 'i32.add 'i32.mul 'i32.shl)
                   ,(??)
                   (i32.const ,(??))))
  (synthesize
    (assert (bveq (wasm-eval expr) (wasm-eval sketch)))))

; Input: (i32.mul x (i32.const 4))
; Output: (i32.shl x (i32.const 2))
```

**Application 3: Rule Verification**

```racket
; Verify that our optimization rules are correct
(define (verify-strength-reduction)
  (define-symbolic x (bitvector 32))
  (define-symbolic k (bitvector 32))

  ; For all x and k where k is power of 2...
  (verify
    (assume (is-power-of-2 k))
    (assert (bveq (bvmul x k)
                  (bvshl x (log2 k))))))
; Returns: verified (or counterexample)
```

### 6.4 Integration Strategy for LOOM

**Phase 1: Rule Discovery (Offline)**
1. Encode WebAssembly semantics in Rosette
2. Use synthesis to discover new optimization rules
3. Verify discovered rules with Z3
4. Generate ISLE rules from verified patterns

**Phase 2: Verification DSL (Optional)**
1. Write verification specs in Rosette (more ergonomic than raw Z3)
2. Generate Z3 queries from Rosette programs
3. Use Rosette's counterexample generation for debugging

**Practical Considerations**:
- Requires learning Racket/Rosette (~1-2 weeks)
- Separate toolchain from LOOM's Rust
- Best used offline for rule discovery
- Could automate with build scripts

### 6.5 References

- **Website**: https://emina.github.io/rosette/
- **Paper**: [Growing Solver-Aided Languages with Rosette](https://homes.cs.washington.edu/~emina/doc/rosette.onward13.pdf) (Onward! 2013)
- **Code**: https://github.com/emina/rosette
- **Tutorial**: https://docs.racket-lang.org/rosette-guide/

---

## 7. Other Notable Tools

### 7.1 Kani (Rust Model Checker)

```rust
#[kani::proof]
fn verify_optimization() {
    let x: i32 = kani::any();
    let y: i32 = kani::any();

    // Verify strength reduction
    kani::assume(y == 4);
    assert_eq!(x.wrapping_mul(y), x << 2);
}
```

**Relevance**: Could verify LOOM's optimizer implementation (Rust code)
**Status**: Production-ready, used by AWS
**Link**: https://github.com/model-checking/kani

### 7.2 Prusti (Rust Verifier)

```rust
#[requires(divisor != 0)]
#[ensures(result == dividend / divisor)]
fn safe_div(dividend: i32, divisor: i32) -> i32 {
    dividend / divisor
}
```

**Relevance**: Contract-based verification for Rust
**Status**: Research prototype
**Link**: https://www.pm.inf.ethz.ch/research/prusti.html

### 7.3 Verus (Rust Verification)

```rust
proof fn strength_reduction_correct(x: u32)
    ensures x * 4 == x << 2
{
    // SMT proves this automatically
}
```

**Relevance**: Proof-carrying Rust code
**Status**: Active development
**Link**: https://verus-lang.github.io/verus/

---

## 8. Research Directions

### 8.1 Open Problems in Compiler Verification

| Problem | Description | Potential LOOM Contribution |
|---------|-------------|---------------------------|
| **Floating-point verification** | FP semantics are complex (NaN, rounding) | Extend Z3 to handle f32/f64 |
| **Memory model verification** | Alias analysis, load/store reordering | Add memory to SMT encoding |
| **Cross-function optimization** | Inlining, interprocedural analysis | Module-level verification |
| **Termination proofs** | Loop optimizations may affect termination | Bounded verification |
| **Incremental verification** | Verify only changed code | Delta verification |

### 8.2 Novel Techniques to Explore

**1. Equivalence Modulo Inputs (EMI)**
- Most effective miscompilation detector (147 bugs in GCC/LLVM)
- Generate program variants by pruning unexecuted code
- All variants should produce same output
- **LOOM could implement this for WebAssembly**

**2. Metamorphic Testing**
- Test optimization properties directly
- E.g., "optimize(optimize(P)) = optimize(P)" (idempotence)
- Found 5 new bugs in GCC/LLVM in 2024
- **LOOM has partial implementation, could expand**

**3. Guided Equality Saturation**
- POPL 2024: Use sketches to guide e-graph saturation
- Faster than full saturation, better than greedy
- **Could integrate with egglog**

**4. Counterexample-Guided Abstraction Refinement (CEGAR)**
- Start with coarse abstraction
- Refine based on spurious counterexamples
- **Could speed up verification**

**5. Neural-Guided Synthesis**
- Use ML to guide synthesis search
- DeepCoder, AlphaCode techniques
- **Discover optimization rules with neural guidance**

### 8.3 WebAssembly-Specific Research

**1. Component Model Verification**
- No existing work on verifying Component Model
- LOOM could be first to formally verify component optimization
- **High novelty for publication**

**2. Reference Type Verification**
- funcref, externref, GC types
- Complex subtyping rules
- **Extend Z3 encoding to handle references**

**3. SIMD Verification**
- v128 operations
- Complex lane operations
- **Bitvector encoding challenges**

**4. Multi-Memory Verification**
- Multiple memory spaces
- Cross-memory aliasing
- **Extension of memory model**

---

## 9. Academic Venues

### 9.1 Top Conferences for Compiler Verification

| Conference | Focus | Deadline | Acceptance Rate | Relevance |
|------------|-------|----------|-----------------|-----------|
| **PLDI** | Programming Language Design & Implementation | Nov | ~20% | ⭐⭐⭐ Core venue |
| **OOPSLA** | Object-Oriented Programming, Systems | Apr | ~25% | ⭐⭐⭐ Core venue |
| **POPL** | Principles of Programming Languages | Jul | ~25% | ⭐⭐⭐ Theory |
| **CAV** | Computer Aided Verification | Jan | ~25% | ⭐⭐⭐ Verification |
| **ICSE** | Software Engineering | Sep | ~20% | ⭐⭐ Applied |
| **ASE** | Automated Software Engineering | Apr | ~20% | ⭐⭐ Tools |
| **CC** | Compiler Construction | Nov | ~30% | ⭐⭐⭐ Compilers |
| **CGO** | Code Generation and Optimization | Aug | ~30% | ⭐⭐⭐ Optimization |

### 9.2 Relevant Workshops

| Workshop | Co-located With | Focus | LOOM Fit |
|----------|-----------------|-------|----------|
| **EGRAPHS** | PLDI | E-graphs & equality saturation | ⭐⭐⭐ If using egglog |
| **WASM Research Day** | PLDI | WebAssembly research | ⭐⭐⭐ Perfect fit |
| **SOAP** | PLDI | Static analysis | ⭐⭐ Analysis work |
| **TAPAS** | SAS | Testing and proofs | ⭐⭐ Verification |
| **NSV** | CAV | Numerical software | ⭐ Float work |

### 9.3 Journals

| Journal | Publisher | Impact | Turnaround |
|---------|-----------|--------|------------|
| **TOPLAS** | ACM | Very High | 6-12 months |
| **JSS** | Elsevier | High | 3-6 months |
| **SPE** | Wiley | Medium | 3-6 months |
| **SCP** | Elsevier | Medium | 4-8 months |

### 9.4 Publication Strategy for LOOM

**Paper 1: Verified WebAssembly Optimization** (Target: PLDI/CC)
- First formally verified WebAssembly optimizer
- Z3 translation validation approach
- Comparison with wasm-opt
- Empirical evaluation on real binaries

**Paper 2: Stack Analysis with SMT** (Target: CAV/CGO)
- Novel stack composition verification
- Binaryen-inspired signatures + Z3 proofs
- Application to optimization passes

**Paper 3: Component Model Optimization** (Target: OOPSLA)
- First optimizer for WASM Component Model
- Formal verification of component transformations
- Real-world evaluation on WASI components

---

## 10. Research Communities & Resources

### 10.1 Research Groups to Follow

| Group | Institution | Focus | Key Researchers |
|-------|-------------|-------|-----------------|
| **UW PLSE** | U. Washington | Rosette, Ruler, Herbie | Emina Torlak, Zach Tatlock |
| **MIT CSAIL** | MIT | Coq, Bedrock, Kami | Adam Chlipala |
| **CMU POP** | Carnegie Mellon | VeriISLE, Cranelift | Ranjit Jhala |
| **INRIA** | France | CompCert, Coq | Xavier Leroy |
| **Max Planck** | Germany | Alive2, Z3 | Nuno Lopes |
| **ETH Zurich** | Switzerland | Prusti, Verus | Peter Müller |

### 10.2 Mailing Lists & Forums

- **PL Perspectives** (ACM SIGPLAN blog)
- **Types Forum** (types-announce@lists.seas.upenn.edu)
- **Verification Corner** (Microsoft Research)
- **WebAssembly CG** (WebAssembly Community Group)

### 10.3 Summer Schools & Tutorials

| Event | Topic | Frequency |
|-------|-------|-----------|
| **Oregon PL Summer School** | PL foundations | Annual (June) |
| **Marktoberdorf** | Verification | Annual (August) |
| **DeepSpec Summer School** | Verified systems | Annual |
| **POPL Tutorials** | Various | Annual (January) |

---

## 11. Recommended Next Steps

### 11.1 Immediate (Next 4 Weeks)

1. **Implement EMI Testing** for LOOM
   - Profile-guided variant generation
   - Detect miscompilations via execution differences
   - Low effort, high bug-finding potential

2. **Experiment with Rosette**
   - Encode WebAssembly subset in Rosette
   - Try synthesizing 3-5 optimization rules
   - Evaluate effort vs. benefit

3. **Prepare Publication Plan**
   - Draft outline for PLDI submission
   - Identify key contributions
   - Plan evaluation methodology

### 11.2 Medium-Term (Next 3 Months)

1. **Extend Z3 Coverage**
   - Add floating-point operations
   - Add memory operations
   - Add control flow verification

2. **Integrate Kani**
   - Verify optimizer implementation
   - Catch implementation bugs early

3. **Submit to Workshop**
   - EGRAPHS or WASM Research Day
   - Get feedback from community

### 11.3 Long-Term (Next Year)

1. **Full PLDI/OOPSLA Paper**
   - Comprehensive evaluation
   - Comparison with state-of-the-art
   - Open-source release

2. **Community Building**
   - Present at WebAssembly CG
   - Engage with verification community
   - Collaborate with research groups

---

## 12. Conclusion

LOOM's verification approach is well-aligned with state-of-the-art (Alive2-style translation validation). The most impactful enhancements are:

1. ⭐ **Rosette for rule synthesis** - Discover new optimizations automatically
2. ⭐ **EMI testing** - Most effective miscompilation detector
3. ⭐ **Publication** - First verified WebAssembly optimizer

The academic landscape offers multiple venues (PLDI, OOPSLA, CC) and communities (UW PLSE, CMU POP) interested in this work.

**Bottom Line**: LOOM has a strong foundation. Strategic enhancements and publication can establish it as the reference implementation for verified WebAssembly optimization.

---

## References

### Tools
- [Alive2](https://github.com/AliveToolkit/alive2)
- [CBMC](https://github.com/diffblue/cbmc)
- [Frama-C](https://frama-c.com/)
- [Boogie](https://github.com/boogie-org/boogie)
- [Rosette](https://emina.github.io/rosette/)
- [Kani](https://github.com/model-checking/kani)
- [Verus](https://verus-lang.github.io/verus/)

### Papers
- [Alive2: Bounded Translation Validation](https://dl.acm.org/doi/10.1145/3453483.3454030) (PLDI 2021)
- [Growing Solver-Aided Languages](https://homes.cs.washington.edu/~emina/doc/rosette.onward13.pdf) (Onward! 2013)
- [Compiler Validation via EMI](https://dl.acm.org/doi/10.1145/2594291.2594334) (PLDI 2014)
- [VeriISLE](http://reports-archive.adm.cs.cmu.edu/anon/2023/CMU-CS-23-126.pdf) (CMU 2023)
- [Ruler](https://arxiv.org/pdf/2108.10436) (OOPSLA 2021)

### Academic Venues
- [PLDI](https://conf.researchr.org/series/pldi)
- [OOPSLA](https://conf.researchr.org/series/oopsla)
- [CAV](https://i-cav.org/)
- [CC](https://conf.researchr.org/series/CC)
- [CGO](https://conf.researchr.org/series/cgo)
