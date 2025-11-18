# LOOM Optimization & Testing Roadmap

**Date:** 2025-11-18
**Status:** Active Planning
**Goal:** Transform LOOM into the world's most robust and thoroughly-tested WebAssembly optimizer

---

## Executive Summary

LOOM is already the **first and only Component Model optimizer**. This roadmap outlines:
1. **20+ new optimizations** to match and exceed wasm-opt
2. **Comprehensive testing framework** with differential testing, fuzzing, and benchmarks
3. **Proof/validation strategy** to demonstrate correctness and superiority

**Key Insight from Research:** The DITWO framework found **1,293 missed optimizations** in wasm-opt, proving even mature optimizers have significant gaps. LOOM can learn from these findings.

---

## Part 1: Optimization Opportunities

### Quick Wins (1-2 weeks)

#### 1. Register Allocation & Local Management
**Impact:** 5-15% binary size reduction through better LEB encoding

- **CoalesceLocals** - Minimize local count through live range analysis
  - Combine non-overlapping locals to reduce local declarations
  - Priority: HIGH (wasm-opt calls this "key register allocation pass")

- **ReorderLocals** - Prioritize frequently-used locals
  - Lower-index locals encode to smaller LEB128 values
  - Move hot locals to indices 0-127 (1 byte encoding)
  - Priority: MEDIUM

- **RedundantSetElimination** - Remove redundant `local.set` operations
  - Track which locals are read before being overwritten
  - Priority: MEDIUM

**Example:**
```wat
;; Before
(local $temp1 i32) (local $temp2 i32) (local $temp3 i32)
local.set $temp1
local.set $temp1  ;; redundant
local.get $temp1

;; After
(local $temp i32)  ;; coalesced temp1+temp2+temp3
local.set $temp
local.get $temp
```

#### 2. Memory & Data Optimization
**Impact:** 10-30% reduction for data-heavy modules

- **MemoryPacking** - Combine and prune data segments
  - Merge adjacent segments
  - Remove unreferenced data
  - Align to reduce padding
  - Priority: HIGH (wasm-opt calls this "key optimize data segments pass")

- **OptimizeAddedConstants** - Convert load/store patterns to constant offsets
  - `(i32.load (i32.add $base (i32.const 4)))` → `(i32.load offset=4 $base)`
  - Priority: MEDIUM

**Example:**
```wat
;; Before
(data (i32.const 0) "hello")
(data (i32.const 8) "world")  ;; gap wastes space
i32.const 1024
i32.const 4
i32.add
i32.load  ;; complex offset

;; After
(data (i32.const 0) "helloworld")  ;; packed
i32.const 1024
i32.load offset=4  ;; constant offset
```

#### 3. Sign Extension Optimization
**Impact:** 5-10% instruction reduction for int8/int16 heavy code

- **PickLoadSigns** - Optimize sign extension patterns
  - Detect unnecessary sign extensions after signed loads
  - Choose optimal load instruction (i32.load8_s vs i32.load8_u)
  - Priority: MEDIUM

**Example:**
```wat
;; Before
i32.load8_u
i32.const 0xff
i32.and  ;; redundant mask after unsigned load

;; After
i32.load8_u  ;; mask removed
```

### Medium-Term Features (3-4 weeks)

#### 4. Advanced Inlining & Call Optimization
**Impact:** 10-30% performance improvement for call-heavy code

- **Directize** - Convert indirect calls to direct when index is constant
  - Analyze `call_indirect` with constant table indices
  - Replace with `call` for better optimization and performance
  - Priority: HIGH (enables further optimizations)

- **Enhanced Inlining Heuristics**
  - Consider function call frequency (if profiling available)
  - Inline across module boundaries for LTO-style optimization
  - Avoid code bloat with size budgets
  - Priority: MEDIUM

**Example:**
```wat
;; Before
(table funcref (elem $add $sub))
i32.const 0
call_indirect (param i32 i32) (result i32)  ;; index 0 is always $add

;; After
call $add  ;; direct call is faster and optimizable
```

#### 5. Dead Code Elimination Enhancements
**Impact:** 5-20% code size reduction

- **Dead Argument Elimination** - Remove unused function parameters
  - Analyze which parameters are never read
  - Create specialized versions without those parameters
  - Update all call sites
  - Priority: HIGH (wasm-opt has this, LOOM doesn't)

- **Unused Import/Export Elimination**
  - Remove imports that are never called
  - Remove exports that are never referenced (with opt-in flag)
  - Priority: MEDIUM

**Example:**
```wat
;; Before
(func $process (param $x i32) (param $unused i32) (result i32)
  local.get $x
  i32.const 10
  i32.add
)
(call $process (i32.const 5) (i32.const 999))

;; After
(func $process (param $x i32) (result i32)  ;; $unused removed
  local.get $x
  i32.const 10
  i32.add
)
(call $process (i32.const 5))
```

#### 6. Function & Code Deduplication
**Impact:** 10-50% for codegen with duplicate patterns

- **Duplicate Function Elimination** - Merge identical functions
  - Hash function bodies to find duplicates
  - Keep one canonical version, redirect all calls
  - Priority: HIGH

- **Code Folding** - Merge duplicate code sequences
  - Detect identical instruction sequences across functions
  - Extract to helper function or merge blocks
  - Priority: MEDIUM (LOOM has basic code folding, enhance it)

**Example:**
```wat
;; Before
(func $add1 (param i32) (result i32) local.get 0 i32.const 1 i32.add)
(func $add2 (param i32) (result i32) local.get 0 i32.const 1 i32.add)

;; After
(func $add (param i32) (result i32) local.get 0 i32.const 1 i32.add)
;; Redirect all $add1 and $add2 calls to $add
```

### Advanced Features (5-8 weeks)

#### 7. SIMD-Specific Optimizations
**Impact:** 2-5x speedup for SIMD code

- **SIMD Constant Folding** - Fold v128.const operations
- **SIMD Strength Reduction** - Optimize common SIMD patterns
- **SIMD Vectorization** - Convert scalar loops to SIMD where possible
- Priority: LOW (nice-to-have, not many SIMD modules yet)

#### 8. Global Value Numbering (GVN)
**Impact:** 10-30% for complex expressions

- **Global CSE** - Extend CSE across basic blocks using dominator tree
- **Value Numbering** - Assign numbers to equivalent expressions globally
- Priority: MEDIUM (more powerful than current local CSE)

**Example:**
```wat
;; Before
block $b1
  local.get $x
  i32.const 4
  i32.mul
  ;; ... use result
end
block $b2
  local.get $x
  i32.const 4
  i32.mul  ;; same expression but different block
end

;; After
local.get $x
i32.const 4
i32.mul
local.tee $temp
block $b1
  local.get $temp
end
block $b2
  local.get $temp  ;; reuse
end
```

#### 9. Profile-Guided Optimization (PGO)
**Impact:** 20-50% performance for production workloads

- **Instrumentation Mode** - Insert counters to collect runtime data
- **Profile-Driven Inlining** - Inline based on actual call frequency
- **Branch Prediction Hints** - Optimize for hot paths
- **Data Layout** - Reorder based on access patterns
- Priority: LOW (requires profiling infrastructure)

#### 10. Polyhedral Loop Optimization
**Impact:** 2-10x for nested loops

- **Loop Interchange** - Reorder nested loops for cache efficiency
- **Loop Tiling** - Improve cache locality
- **Loop Fusion/Fission** - Merge or split loops optimally
- Priority: LOW (complex, high ROI only for scientific computing)

---

## Part 2: Comprehensive Testing Framework

### Phase 1: Differential Testing (Highest Priority)

**Goal:** Prove LOOM produces valid and optimal results compared to wasm-opt

#### A. DITWO-Style Framework
Build differential testing infrastructure inspired by DITWO research:

1. **Test Case Generation**
   - Use existing WASM corpus (Wasm-R3-Bench, real-world apps)
   - Generate synthetic tests with wasm-smith
   - Start with 10,000 modules

2. **Dual Optimization**
   ```bash
   loom optimize input.wasm -o loom.wasm
   wasm-opt input.wasm -O3 -o wasm-opt.wasm
   ```

3. **Comparison Metrics**
   - Binary size: `loom.wasm` vs `wasm-opt.wasm`
   - Instruction count
   - Validation: both must pass `wasmparser::validate()`
   - Semantic equivalence: execute both with wasmtime, compare outputs

4. **Missed Optimization Detection**
   - If `wasm-opt.wasm` < `loom.wasm`: analyze the delta to find what LOOM missed
   - If `loom.wasm` < `wasm-opt.wasm`: we found a case where LOOM is better!
   - Build database of patterns

**Implementation:**
```rust
// loom-testing/src/differential.rs
pub struct DifferentialTest {
    input: Vec<u8>,
    loom_output: Vec<u8>,
    wasm_opt_output: Vec<u8>,
}

impl DifferentialTest {
    pub fn compare(&self) -> DifferentialResult {
        // Size comparison
        let loom_size = self.loom_output.len();
        let wasm_opt_size = self.wasm_opt_output.len();

        // Semantic equivalence
        let loom_result = execute_wasm(&self.loom_output);
        let wasm_opt_result = execute_wasm(&self.wasm_opt_output);

        DifferentialResult {
            size_delta: loom_size as i64 - wasm_opt_size as i64,
            semantically_equivalent: loom_result == wasm_opt_result,
            // ... more metrics
        }
    }
}
```

**Deliverable:** Report showing:
- "LOOM matches or beats wasm-opt on 95% of test cases"
- Specific patterns where LOOM excels
- Areas for improvement

**Timeline:** 2 weeks

---

### Phase 2: Fuzzing Infrastructure

**Goal:** Find bugs and edge cases automatically

#### A. Wasm-Smith Integration

1. **Add wasm-smith dependency**
   ```toml
   [dev-dependencies]
   wasm-smith = "0.214"
   arbitrary = "1.3"
   ```

2. **Continuous Fuzzing**
   ```rust
   // loom-core/fuzz/fuzz_targets/optimize.rs
   #![no_main]
   use libfuzzer_sys::fuzz_target;
   use wasm_smith::Module;

   fuzz_target!(|module: Module| {
       let wasm = module.to_bytes();

       // Try to optimize
       if let Ok(optimized) = loom_core::optimize(&wasm) {
           // Must validate
           assert!(wasmparser::validate(&optimized).is_ok());

           // Must be semantically equivalent
           assert_semantic_equivalence(&wasm, &optimized);
       }
   });
   ```

3. **Run 24/7 fuzzing**
   ```bash
   cargo fuzz run optimize -- -max_total_time=86400  # 24 hours
   ```

**Deliverable:**
- Fuzz targets for all optimization passes
- CI integration (fuzz for 1 hour on every PR)
- Crash/bug database

**Timeline:** 1 week

#### B. Component Model Fuzzing

Since LOOM is the ONLY Component Model optimizer, we need extensive fuzzing:

```rust
// Generate random components
use wasm_smith::ConfiguredModule;

let config = Config::default();
config.component_model_enabled(true);
let component = Module::new(config, &mut arbitrary)?;

// Fuzz component optimization
loom_core::optimize_component(&component.to_bytes())?;
```

**Timeline:** 1 week (after wasm-smith adds Component support, or build custom generator)

---

### Phase 3: Benchmark Suite

**Goal:** Demonstrate performance on realistic workloads

#### A. Real-World Benchmark Collection

1. **Gather Test Cases**
   - WasmScore suite (Bytecode Alliance)
   - Wasm-R3-Bench (27 real-world apps)
   - Nutrient PDF benchmark
   - libsodium crypto
   - Game engines (e.g., Unity WASM output)
   - Total target: 100+ real-world modules

2. **Automated Benchmarking**
   ```bash
   # loom-benchmarks/run_all.sh
   for wasm in benchmarks/*.wasm; do
       echo "Testing $wasm..."

       # Optimize with LOOM
       time loom optimize $wasm -o loom.wasm

       # Optimize with wasm-opt
       time wasm-opt $wasm -O3 -o wasm-opt.wasm

       # Compare
       ls -lh loom.wasm wasm-opt.wasm

       # Execute and measure
       wasmtime run loom.wasm --invoke benchmark
       wasmtime run wasm-opt.wasm --invoke benchmark
   done
   ```

3. **Continuous Tracking**
   - Track results over time
   - Detect regressions in CI
   - Generate comparison charts

**Deliverable:**
- Benchmark dashboard: `docs/benchmarks/index.html`
- Comparison table: LOOM vs wasm-opt on 100+ real modules
- Performance claims backed by data

**Timeline:** 2 weeks

---

### Phase 4: Property-Based Testing

**Goal:** Verify correctness properties mathematically

#### A. Idempotence Testing (Already Have)
Enhance existing property tests:

```rust
#[test]
fn property_optimization_is_idempotent() {
    for _ in 0..1000 {
        let module = generate_random_module();

        let opt1 = optimize(&module)?;
        let opt2 = optimize(&opt1)?;

        // Must be identical after second pass
        assert_eq!(opt1, opt2);
    }
}
```

#### B. Semantic Preservation
```rust
#[test]
fn property_semantics_preserved() {
    for _ in 0..1000 {
        let module = generate_random_module();
        let inputs = generate_random_inputs();

        let original_output = execute(&module, &inputs);
        let optimized = optimize(&module)?;
        let optimized_output = execute(&optimized, &inputs);

        assert_eq!(original_output, optimized_output);
    }
}
```

#### C. Size Monotonicity
```rust
#[test]
fn property_size_decreases_or_stays_same() {
    for _ in 0..1000 {
        let module = generate_random_module();
        let optimized = optimize(&module)?;

        // Optimization should never increase size
        assert!(optimized.len() <= module.len());
    }
}
```

**Timeline:** 1 week

---

### Phase 5: Regression Test Suite

**Goal:** Never break what already works

1. **Capture Current Behavior**
   ```bash
   # Generate golden outputs for all test cases
   for test in tests/fixtures/*.wasm; do
       loom optimize $test -o tests/golden/$(basename $test)
   done
   ```

2. **Regression Detection**
   ```rust
   #[test]
   fn regression_all_fixtures() {
       for fixture in read_dir("tests/fixtures")? {
           let optimized = optimize(&read(fixture)?)?;
           let golden = read(golden_path(fixture))?;

           // Must match exactly
           assert_eq!(optimized, golden, "Regression in {}", fixture);
       }
   }
   ```

3. **CI Integration**
   - Run on every commit
   - Block PRs that cause regressions
   - Manual review required if golden files change

**Timeline:** 3 days

---

## Part 3: Proof & Validation Strategy

### How to Prove LOOM is Best-in-Class

#### 1. Correctness Proof
✅ **Already Have:**
- Z3 SMT formal verification
- Property-based testing

📋 **Add:**
- Differential testing vs wasm-opt (semantically equivalent)
- Fuzzing with 0 crashes for 1M+ inputs
- Formal specification document

**Claim:** "LOOM produces provably correct optimizations verified by Z3 SMT solver"

#### 2. Performance Proof
📋 **Need:**
- Benchmark results on 100+ real-world modules
- Head-to-head comparison with wasm-opt
- Statistical significance testing

**Claim:** "LOOM matches or exceeds wasm-opt on 95% of real-world WebAssembly modules"

**Metrics:**
- Binary size reduction: LOOM avg 85%, wasm-opt avg 82% (example)
- Optimization time: LOOM avg 50µs, wasm-opt avg 2ms
- Runtime performance: (need execution benchmarks)

#### 3. Uniqueness Proof
✅ **Already Have:**
- Component Model support (WORLD FIRST!)

📋 **Add:**
- Documentation of Component Model optimizations
- Component Model benchmark suite
- Comparison showing wasm-opt fails on components

**Claim:** "LOOM is the only optimizer supporting the WebAssembly Component Model"

---

## Part 4: Implementation Plan

### Sprint 1 (Week 1-2): Testing Infrastructure
- [ ] Set up differential testing framework
- [ ] Integrate wasm-smith for fuzzing
- [ ] Create initial benchmark harness
- [ ] Collect 100 real-world WASM modules

**Deliverable:** Working test infrastructure

### Sprint 2 (Week 3-4): Quick Win Optimizations
- [ ] Implement CoalesceLocals (register allocation)
- [ ] Implement MemoryPacking (data segment optimization)
- [ ] Implement RedundantSetElimination
- [ ] Add comprehensive tests for each

**Deliverable:** 3 new optimizations with 10-20% additional gains

### Sprint 3 (Week 5-6): Advanced Optimizations
- [ ] Implement Directize (indirect→direct calls)
- [ ] Implement Dead Argument Elimination
- [ ] Implement Duplicate Function Elimination
- [ ] Enhance existing CSE to GVN

**Deliverable:** 4 major optimizations

### Sprint 4 (Week 7-8): Benchmarking & Documentation
- [ ] Run full benchmark suite
- [ ] Generate comparison reports
- [ ] Write optimization guide documentation
- [ ] Create performance dashboard
- [ ] Publish results

**Deliverable:** Public proof of LOOM's capabilities

---

## Part 5: Success Metrics

### Quantitative Goals
1. **Test Coverage:** 100+ real-world WASM modules tested
2. **Correctness:** 0 validation failures on 1M+ fuzzed inputs
3. **Performance:** Match or beat wasm-opt on 95% of benchmarks
4. **Size Reduction:** Average 85%+ binary size reduction
5. **Speed:** Optimize in <100µs for typical modules

### Qualitative Goals
1. **Documentation:** Comprehensive optimization guide
2. **Reputation:** Recognized as best-in-class by WebAssembly community
3. **Adoption:** Used in production by major projects
4. **Research:** Publish findings (DITWO found 1,293 wasm-opt issues, we can do similar analysis)

---

## Part 6: Tooling & Infrastructure

### Tools to Build/Integrate

#### 1. `loom-test` - Testing CLI
```bash
loom-test differential input.wasm     # Compare with wasm-opt
loom-test fuzz --time 1h               # Fuzz for 1 hour
loom-test benchmark suite/*.wasm       # Run benchmarks
loom-test regression                   # Check for regressions
```

#### 2. `loom-analyze` - Analysis Tool
```bash
loom-analyze missed input.wasm         # Find missed optimizations
loom-analyze compare loom.wasm wasm-opt.wasm  # Detailed comparison
loom-analyze stats benchmark/*.wasm    # Aggregate statistics
```

#### 3. CI/CD Integration
```yaml
# .github/workflows/test.yml
- name: Differential Testing
  run: cargo test --test differential

- name: Fuzzing (1 hour)
  run: cargo fuzz run optimize -- -max_total_time=3600

- name: Benchmark Regression
  run: cargo bench --bench all -- --baseline main
```

#### 4. Dashboard
- Real-time benchmark results
- Comparison charts (LOOM vs wasm-opt)
- Fuzzing status (crashes, coverage)
- Test pass rates

---

## Part 7: Research Opportunities

### Publications & Recognition

1. **"LOOM: Component Model Optimization"**
   - First optimizer for Component Model
   - Novel techniques for component-level optimization
   - Target: PLDI, OOPSLA, or WebAssembly Workshop

2. **"Differential Analysis of WebAssembly Optimizers"**
   - DITWO-style analysis comparing LOOM vs wasm-opt
   - Catalog of missed optimizations
   - Target: ISSTA (where DITWO was published)

3. **"Verified WebAssembly Optimization"**
   - Z3 SMT-based correctness proofs
   - Translation validation approach
   - Target: CAV, TACAS (verification conferences)

---

## Conclusion

This roadmap transforms LOOM from "feature-complete optimizer" to **"world's most robust and thoroughly-tested WebAssembly optimizer"**.

**Key Advantages:**
1. ✅ **Only Component Model optimizer** (already true)
2. 🚧 **Most thoroughly tested** (via differential testing + fuzzing)
3. 🚧 **Proven correctness** (Z3 + property testing + differential)
4. 🚧 **Best performance** (benchmarks on 100+ real modules)

**Next Immediate Steps:**
1. Set up differential testing framework (Week 1)
2. Integrate wasm-smith fuzzing (Week 1)
3. Implement CoalesceLocals optimization (Week 2)
4. Collect 100 real-world WASM modules (Week 2)

**Timeline:** 8 weeks to complete core items, 6 months for advanced features

---

**Questions to resolve:**
1. Should we prioritize size reduction or execution speed?
2. Which 3 optimizations give the best ROI?
3. Do we want to publish research papers?
4. Should we build a public benchmark dashboard?
