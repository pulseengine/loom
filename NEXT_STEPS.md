# LOOM: Next Steps - Prioritized Action Plan

**Created:** 2025-11-18
**Purpose:** Clear, prioritized next steps based on comprehensive research

---

## TL;DR - Do This First

1. **Differential Testing** (1 week) - Prove LOOM matches/beats wasm-opt
2. **CoalesceLocals Optimization** (3 days) - Quick 10-15% size win
3. **Fuzzing** (3 days) - Find bugs automatically
4. **Real-World Benchmarks** (1 week) - Proof of performance

**Total:** ~3 weeks to transform LOOM from "feature-complete" to "proven best-in-class"

---

## Background: What We Learned

### Research Findings

1. **DITWO Study (ISSTA 2023)** found **1,293 missed optimizations** in wasm-opt
   - Even mature optimizers have significant gaps
   - Differential testing is the gold standard for validation
   - We can use the same approach

2. **wasm-opt has 40+ optimization passes**, LOOM has 12
   - We're missing: register allocation, memory packing, deduplication
   - But we have: Component Model (WORLD FIRST!)
   - Gap analysis shows where to focus

3. **Testing Tools Available:**
   - **wasm-smith**: Generate random valid WASM for fuzzing
   - **WasmScore**: Real-world benchmark suite (Bytecode Alliance)
   - **Wasm-R3-Bench**: 27 real-world applications
   - **wasmtime**: Execute WASM for semantic testing

### LOOM's Current State

✅ **Strengths:**
- Only Component Model optimizer (no competition!)
- 135/135 tests passing
- Z3 formal verification
- Fast (10-30µs typical optimization)
- Clean codebase (zero clippy warnings)

⚠️ **Gaps vs wasm-opt:**
- Missing register allocation (CoalesceLocals, MergeLocals)
- Missing memory optimization (MemoryPacking)
- Missing deduplication (function, code)
- No differential testing yet
- No fuzzing infrastructure
- Limited real-world benchmarks

---

## Priority 1: Differential Testing (HIGHEST VALUE)

**Why:** Proves correctness and finds optimization gaps
**Time:** 1 week
**ROI:** Extremely high - enables all other work

### Tasks

#### Day 1-2: Infrastructure
```bash
# Create testing crate
cargo new --lib loom-testing
cd loom-testing

# Add dependencies
cat >> Cargo.toml <<EOF
[dependencies]
anyhow = "1.0"
wasmparser = "0.214"
tempfile = "3.8"
glob = "0.3"

[dev-dependencies]
criterion = "0.5"
EOF
```

Copy implementation from `docs/TESTING_FRAMEWORK.md` → "Phase 1: Differential Testing"

#### Day 3-4: Collect Test Corpus
```bash
# Install wasm-opt
brew install binaryen  # macOS
# OR
wget https://github.com/WebAssembly/binaryen/releases/latest/download/binaryen-linux-x64.tar.gz
tar xzf binaryen-linux-x64.tar.gz
sudo cp binaryen-*/bin/wasm-opt /usr/local/bin/

# Collect corpus
bash scripts/collect_corpus.sh

# Goal: 100+ real WASM files
find tests/corpus -name '*.wasm' | wc -l
```

#### Day 5-7: Run Tests & Analyze
```bash
# Build differential tester
cargo build --release --bin differential

# Run tests
./target/release/differential

# Expected output:
# 📊 Differential Testing Summary
# Total tests:     142
# LOOM wins:       67 (47.2%)
# wasm-opt wins:   52 (36.6%)
# Ties:            23 (16.2%)
# 🎯 LOOM success rate: 63.4%
```

**Analyze gaps:**
- Save all "wasm-opt wins" cases to `analysis/wasm-opt-wins/`
- Use `wasm-objdump -d` to compare optimized output
- Identify patterns: "wasm-opt did X, LOOM didn't"
- Prioritize missing optimizations

### Deliverable

- Working differential test suite
- Report: "LOOM vs wasm-opt on 100+ Modules"
- Gap analysis: "Top 5 Missing Optimizations"

---

## Priority 2: CoalesceLocals Optimization (QUICK WIN)

**Why:** wasm-opt calls this "key register allocation pass"
**Time:** 3 days
**ROI:** High - 10-15% size reduction expected

### Background

WebAssembly locals are indexed (0, 1, 2, ...). Lower indices encode smaller in LEB128:
- Indices 0-127: 1 byte
- Indices 128-16383: 2 bytes
- etc.

**CoalesceLocals** combines non-overlapping locals to reduce total local count.

### Example

```wat
;; Before (4 locals)
(local $temp1 i32)  ;; used lines 10-20
(local $temp2 i32)  ;; used lines 30-40  (doesn't overlap with $temp1!)
(local $temp3 i32)  ;; used lines 50-60
(local $x i32)      ;; used throughout

;; After (2 locals)
(local $temp i32)   ;; reused for temp1, temp2, temp3 (non-overlapping)
(local $x i32)
```

### Implementation

```rust
// loom-core/src/lib.rs

/// Coalesce locals to minimize count (register allocation)
///
/// Performs live range analysis to find non-overlapping locals,
/// then merges them to reduce total local count.
fn coalesce_locals(functions: &mut [Function]) {
    for func in functions {
        // Step 1: Build live ranges for each local
        let live_ranges = compute_live_ranges(&func.instructions);

        // Step 2: Build interference graph
        // Two locals interfere if their live ranges overlap
        let interference_graph = build_interference_graph(&live_ranges);

        // Step 3: Graph coloring to find minimal coloring
        // Each color = one coalesced local
        let coloring = color_graph(&interference_graph);

        // Step 4: Remap local indices
        let mapping = build_local_mapping(&coloring, &func.locals);

        // Step 5: Rewrite instructions with new indices
        rewrite_local_references(&mut func.instructions, &mapping);

        // Step 6: Update local declarations
        func.locals = compute_new_locals(&mapping);
    }
}

struct LiveRange {
    start: usize,  // instruction index
    end: usize,
}

fn compute_live_ranges(instructions: &[Instruction]) -> Vec<LiveRange> {
    // TODO: Data flow analysis
    // Track where each local is defined (local.set, local.tee)
    // Track where each local is used (local.get)
    // Live range = first def to last use
    vec![]
}
```

### Tasks

**Day 1:** Implement live range analysis
**Day 2:** Implement graph coloring (greedy algorithm is fine)
**Day 3:** Test and integrate into pipeline

### Testing

```rust
#[test]
fn test_coalesce_locals() {
    let wat = r#"
    (module
      (func $test (result i32)
        (local $a i32)
        (local $b i32)
        (local $c i32)
        (i32.const 1)
        (local.set $a)
        (local.get $a)  ;; $a dies here
        (i32.const 2)
        (local.set $b)  ;; $b can reuse $a's slot!
        (local.get $b)
        (i32.add)
      )
    )
    "#;

    let wasm = wat::parse_str(wat).unwrap();
    let optimized = optimize(&wasm).unwrap();

    // Should have fewer locals after coalescing
    // Original: 3 locals, After: 2 locals (a and b coalesced)
    // TODO: Verify local count reduced
}
```

### Deliverable

- CoalesceLocals pass implemented
- 10-15% additional binary size reduction
- Tests verifying correctness

---

## Priority 3: Fuzzing Infrastructure (ROBUSTNESS)

**Why:** Find bugs automatically, build confidence
**Time:** 3 days
**ROI:** High - prevents production bugs

### Setup

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Initialize
cargo fuzz init

# This creates:
# fuzz/
#   ├── Cargo.toml
#   └── fuzz_targets/
#       └── fuzz_target_1.rs
```

### Create Fuzz Targets

Copy from `docs/TESTING_FRAMEWORK.md` → "Phase 2: Fuzzing"

**Targets needed:**
1. `optimize.rs` - Fuzz core optimization
2. `component.rs` - Fuzz component optimization
3. `semantic.rs` - Fuzz with execution validation

### Run Fuzzing

```bash
# Quick test (10 minutes)
cargo fuzz run optimize -- -max_total_time=600

# Overnight (8 hours)
nohup cargo fuzz run optimize -- -max_total_time=28800 &

# Check for crashes
ls fuzz/artifacts/optimize/
# Should be empty (no crashes)
```

### CI Integration

```yaml
# .github/workflows/fuzz.yml
name: Fuzzing

on: [push, pull_request]

jobs:
  fuzz:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: cargo install cargo-fuzz
      - run: cargo fuzz run optimize -- -max_total_time=600
      - name: Check for crashes
        run: |
          if [ -d fuzz/artifacts/optimize ]; then
            echo "❌ Fuzzing found crashes!"
            ls -la fuzz/artifacts/optimize/
            exit 1
          fi
```

### Deliverable

- Fuzzing running in CI (10 min per PR)
- 0 crashes on 1M+ inputs
- Confidence in robustness

---

## Priority 4: Real-World Benchmarks (PROOF)

**Why:** Concrete proof LOOM performs well
**Time:** 1 week
**ROI:** High - marketing/adoption

### Collect Benchmarks

```bash
# WasmScore (Bytecode Alliance official benchmarks)
git clone https://github.com/bytecodealliance/wasm-score
cd wasm-score
# Follow build instructions to get WASM files

# Wasm-R3-Bench (27 real-world apps)
# Download from https://doi.org/10.1145/3689787

# Build popular projects to WASM
cargo new --lib example-rust
# ... compile to wasm32-unknown-unknown

# Goal: 50+ diverse real-world modules
```

### Run Benchmarks

```bash
# Create benchmark script (copy from docs/TESTING_FRAMEWORK.md)
bash scripts/benchmark_comparison.sh

# Output: benchmarks/results/YYYYMMDD_HHMMSS.md
```

### Example Output

```markdown
# LOOM vs wasm-opt Benchmark Results
**Date:** 2025-11-18

| Module | Input | LOOM | wasm-opt | Winner | LOOM Time | wasm-opt Time |
|--------|-------|------|----------|--------|-----------|---------------|
| pdf-render | 2.4MB | 142KB | 156KB | ✅ LOOM | 12ms | 450ms |
| libsodium | 512KB | 98KB | 101KB | ✅ LOOM | 3ms | 89ms |
| image-proc | 1.1MB | 234KB | 229KB | ⚠️ wasm-opt | 8ms | 312ms |
| game-logic | 3.2MB | 389KB | 402KB | ✅ LOOM | 18ms | 1.2s |

**Summary:**
- Win rate: 73% (LOOM wins or ties)
- Avg size: LOOM 87.2% reduction, wasm-opt 84.1%
- Avg time: LOOM 45x faster than wasm-opt
```

### Deliverable

- Benchmark suite with 50+ real modules
- Comparison report
- Automated tracking in CI

---

## Priority 5: Missing Optimizations (FEATURE PARITY)

**Why:** Close gap with wasm-opt
**Time:** 2-4 weeks (ongoing)
**ROI:** Medium-High

Based on gap analysis from differential testing, implement in this order:

### Week 1: Memory & Data
1. **MemoryPacking** - Combine data segments (wasm-opt's "key data optimization")
2. **OptimizeAddedConstants** - Convert `(i32.load (i32.add $x (i32.const 4)))` to `(i32.load offset=4 $x)`

### Week 2: Deduplication
3. **Duplicate Function Elimination** - Hash function bodies, merge identical
4. **Enhanced Code Folding** - Detect duplicate sequences across functions

### Week 3: Call Optimization
5. **Directize** - Convert `call_indirect` with constant index to `call`
6. **Dead Argument Elimination** - Remove unused function parameters

### Week 4: Advanced
7. **Global Value Numbering** - Extend CSE across basic blocks
8. **Enhanced LICM** - Hoist more operations (currently only loads)

Each optimization:
- Research wasm-opt implementation (Binaryen source)
- Implement in LOOM
- Add tests
- Verify with differential testing

---

## Timeline Summary

```
Week 1:  Differential Testing Infrastructure ✅
Week 2:  CoalesceLocals + Fuzzing ✅
Week 3:  Real-World Benchmarks ✅
Week 4:  MemoryPacking + OptimizeAddedConstants
Week 5:  Deduplication (functions, code)
Week 6:  Call Optimization (Directize, Dead Args)
Week 7:  Advanced (GVN, Enhanced LICM)
Week 8:  Documentation, Polish, Release
```

---

## Success Criteria

After 8 weeks, LOOM should have:

### Correctness Proof ✅
- [x] 135/135 tests passing
- [ ] Differential testing: 95%+ match/beat wasm-opt
- [ ] Fuzzing: 0 crashes on 1M+ inputs
- [ ] Z3 verification (already have)

### Performance Proof ✅
- [ ] Benchmarks on 50+ real-world modules
- [ ] 85%+ average binary size reduction
- [ ] 10-50x faster than wasm-opt
- [ ] Win rate: 70%+ (LOOM wins or ties)

### Feature Completeness ✅
- [x] Component Model support (UNIQUE!)
- [ ] Register allocation (CoalesceLocals)
- [ ] Memory optimization (MemoryPacking)
- [ ] Deduplication (functions, code)
- [ ] 20+ optimization passes (vs current 12)

### Documentation & Adoption ✅
- [ ] Benchmark dashboard (public)
- [ ] Research paper draft
- [ ] Blog post: "LOOM: First Component Model Optimizer"
- [ ] Promotion to WebAssembly community

---

## Measuring Success

Track these metrics weekly:

```bash
# Differential testing win rate
./scripts/measure_win_rate.sh
# Target: Start at ~50%, reach 95% by week 8

# Binary size reduction
./scripts/measure_avg_reduction.sh
# Target: Start at 85%, reach 90% by week 8

# Fuzzing crashes
ls fuzz/artifacts/optimize/ | wc -l
# Target: 0 (always)

# Test coverage
cargo tarpaulin
# Target: >90%
```

---

## Quick Commands Reference

```bash
# Differential testing
cargo run --bin differential

# Fuzzing (10 min)
cargo fuzz run optimize -- -max_total_time=600

# Benchmarks
bash scripts/benchmark_comparison.sh

# Run all tests
cargo test --all

# Check quality
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check

# Build docs
cargo doc --no-deps --open
```

---

## Questions to Answer This Week

Before starting implementation, decide:

1. **Primary goal: size or speed?**
   - Size reduction: Focus on register allocation, memory packing
   - Execution speed: Focus on inlining, LICM, GVN
   - **Recommendation:** Size first (easier to measure, more visible)

2. **Target win rate vs wasm-opt?**
   - 70%: Good enough to claim "competitive"
   - 85%: Can claim "matches wasm-opt"
   - 95%: Can claim "beats wasm-opt"
   - **Recommendation:** Start at 70%, aim for 85%

3. **When to publish?**
   - After differential testing: "LOOM validated"
   - After benchmarks: "LOOM proven"
   - After 20+ optimizations: "LOOM complete"
   - **Recommendation:** Publish early, update often

4. **Research paper?**
   - Component Model paper (PLDI, OOPSLA)
   - Differential testing paper (ISSTA)
   - Both?
   - **Recommendation:** Start with blog post, consider paper if results compelling

---

## Get Started Now

```bash
# Step 1: Read the docs
cat docs/OPTIMIZATION_ROADMAP.md
cat docs/TESTING_FRAMEWORK.md

# Step 2: Install wasm-opt
brew install binaryen

# Step 3: Create testing infrastructure
cargo new --lib loom-testing

# Step 4: Start differential testing
# (Follow docs/TESTING_FRAMEWORK.md Phase 1)

# Step 5: Report results
./target/release/differential > results/week1.txt
```

**First milestone: Working differential test in 1 week!**

---

## Resources

### Documentation
- [OPTIMIZATION_ROADMAP.md](docs/OPTIMIZATION_ROADMAP.md) - Comprehensive 6-month plan
- [TESTING_FRAMEWORK.md](docs/TESTING_FRAMEWORK.md) - Detailed testing implementation

### Research Papers
- DITWO (ISSTA 2023): Differential Testing of wasm-opt
- WASMaker (ISSTA 2024): Differential Testing of Runtimes
- Wasm-R3 (OOPSLA 2024): Real-world Benchmarks

### Tools
- wasm-opt: https://github.com/WebAssembly/binaryen
- wasm-smith: https://github.com/bytecodealliance/wasm-tools/tree/main/crates/wasm-smith
- WasmScore: https://github.com/bytecodealliance/wasm-score
- wasmtime: https://github.com/bytecodealliance/wasmtime

### LOOM Docs
- README.md - Overview
- docs/ARCHITECTURE.md - 12-phase pipeline
- docs/USAGE_GUIDE.md - CLI reference

---

**Ready to make LOOM the world's best WebAssembly optimizer? Start with differential testing!**
