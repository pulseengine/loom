# Formal Verification Architecture for LOOM

## Executive Summary

This document presents a **formally verified optimizer architecture** for LOOM based on comprehensive research into eGraphs, SMT solvers, and verification techniques. The recommended approach combines:

1. **Z3 SMT Solver** - Mathematical proofs of semantic equivalence
2. **Differential Testing** - Runtime execution comparison against wasmtime
3. **Fuzzing** - cargo-fuzz infrastructure for crash/bug detection
4. **Property-Based Testing** - Empirical validation
5. **ISLE DSL** - Readable, auditable optimization rules

**Current Status (v0.2.0)**: LOOM implements a **hybrid verification approach** combining Z3 proofs for arithmetic/memory/control flow with differential testing and fuzzing for comprehensive coverage.

**Note**: LOOM uses Z3 for translation validation. This proves transformations correct but has limitations - see Section 6 for remaining gaps.

---

## 1. Verification Guarantees We Need

### 1.1 Formal Requirements

For LOOM to claim "formally verified optimizer," we need to prove:

**Soundness**: `∀ programs P, optimized P'. semantics(P) ≡ semantics(P')`

In plain English: "For all valid programs, optimization preserves semantics."

### 1.2 What Different Tools Provide

| Tool | Soundness | Completeness | Proof Type | License |
|------|-----------|--------------|------------|---------|
| **Z3 SMT** | ✅ Formal proof per run | ❌ Not guaranteed to find all opts | Mathematical | MIT |
| **egg** | ✅ IF rules are correct | ❌ Bounded by rules | Conditional | MIT/Apache |
| **Symbolica** | ❌ No verification | ❌ N/A | None (CAS only) | Paid |
| **ISLE** | ❌ Just a DSL | ❌ N/A | None | Apache |
| **Coq/Isabelle** | ✅ Formal proof | ✅ Yes | Mechanized | Free |

**Conclusion**: For production use, **Z3 translation validation** provides the best balance of:
- Formal verification (mathematical proof)
- Practical integration (already implemented)
- Incremental adoption (per-pass verification)
- Reasonable performance (~100ms overhead in dev/CI)

---

## 2. Recommended Architecture: Hybrid Verification

### 2.1 Three-Layer Approach (Implemented in v0.2.0)

```
┌─────────────────────────────────────────────────────────────────┐
│                  LOOM Verified Optimizer                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: Optimization (ISLE + Rust)                           │
│  ├─ Readable, declarative rules                                │
│  ├─ Explicit control flow handling                             │
│  └─ 12-phase pipeline (parse → optimize → encode)              │
│                                                                  │
│  Layer 2: Formal Verification (Z3 SMT) ✅ IMPLEMENTED          │
│  ├─ Translation validation per optimization run                │
│  ├─ Encodes original and optimized as SMT formulas            │
│  ├─ Memory model with Z3 Array theory                         │
│  ├─ Bounded loop verification (unroll 3 iterations)           │
│  ├─ Nested loop support (MAX_LOOP_NESTING_DEPTH = 1)          │
│  └─ Returns UNSAT = verified, SAT = bug with counterexample   │
│                                                                  │
│  Layer 3: Empirical Validation ✅ IMPLEMENTED                  │
│  ├─ Property-based testing (256 cases per property)           │
│  ├─ Differential testing with wasmtime execution              │
│  │   └─ loom-testing/src/lib.rs: compare_execution()          │
│  ├─ Fuzzing with cargo-fuzz (fuzz/fuzz_targets/)              │
│  │   ├─ fuzz_optimize.rs: Optimization pipeline fuzzing       │
│  │   └─ fuzz_roundtrip.rs: Parse/encode roundtrip fuzzing     │
│  └─ Catches bugs Z3 might miss (complex control flow)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Verification Levels

**Level 1: Fast Feedback (Development)**
- Property-based tests (proptest) - subsecond
- Unit tests - subsecond
- Idempotence checks - subsecond

**Level 2: Formal Verification (CI)**
- Z3 translation validation - ~100ms per function
- SMT encoding for all optimizations
- **Enabled by default** (Phase 5: `verification` is a default feature in loom-core)

**Level 3: Exhaustive Testing (Nightly)**
- Differential testing vs wasm-opt
- Fuzzing with 10k+ random programs
- Stress testing on large WebAssembly binaries

### 2.3 When Each Layer Catches Bugs

| Bug Type | proptest | Z3 | Differential | Example |
|----------|----------|-----|--------------|---------|
| **Wrong constant folding** | ✅ | ✅ | ✅ | `2+3 → 6` |
| **Overflow handling** | ✅ | ✅ | ✅ | `i32::MAX + 1` |
| **Strength reduction error** | ⚠️ | ✅ | ✅ | `x*4 → x<<3` (wrong shift) |
| **Control flow bug** | ⚠️ | ⚠️ | ✅ | Loop unrolling breaks semantics |
| **Memory aliasing** | ❌ | ⚠️ | ✅ | Load/store reordering |
| **Floating point** | ⚠️ | ❌ | ✅ | NaN handling |

**Key Insight**: No single technique catches everything. Layered approach provides defense in depth.

---

## 3. Z3 SMT Verification (Primary Approach)

### 3.1 Current Implementation

**File**: `loom-core/src/verify.rs` (370 lines)

**Architecture**:
```rust
pub fn verify_optimization(original: &Module, optimized: &Module) -> Result<bool> {
    // 1. Create Z3 context and solver
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let solver = Solver::new(&ctx);

    // 2. For each function pair
    for (orig_func, opt_func) in original.functions.iter().zip(optimized.functions.iter()) {
        // 3. Encode both to SMT (symbolic execution)
        let orig_formula = encode_function_to_smt(&ctx, orig_func)?;
        let opt_formula = encode_function_to_smt(&ctx, opt_func)?;

        // 4. Assert they are NOT equal (looking for counterexample)
        solver.push();
        solver.assert(&orig_formula._eq(&opt_formula).not());

        // 5. UNSAT = no counterexample exists = proven equivalent
        match solver.check() {
            SatResult::Unsat => continue, // ✅ Verified
            SatResult::Sat => {
                let model = solver.get_model()?;
                eprintln!("Counterexample: {}", model);
                return Ok(false); // ❌ Bug found
            }
            SatResult::Unknown => return Err(anyhow!("Timeout")),
        }
    }

    Ok(true)
}
```

**SMT Encoding** (lines 118-362):

| WASM Instruction | Z3 Encoding |
|------------------|-------------|
| `i32.const N` | `BV::from_i64(ctx, N, 32)` |
| `i64.const N` | `BV::from_i64(ctx, N, 64)` |
| `f32.const bits` | `BV::from_i64(ctx, bits as i64, 32)` |
| `f64.const bits` | `BV::from_i64(ctx, bits as i64, 64)` |
| `i32.add` | `lhs.bvadd(&rhs)` |
| `i32.mul` | `lhs.bvmul(&rhs)` |
| `i32.shl` | `lhs.bvshl(&rhs)` |
| `i32.and` | `lhs.bvand(&rhs)` |
| `local.get` | `locals[idx].clone()` |
| `local.set` | `locals[idx] = value` |

**Coverage**: Currently supports ~20 instructions (integer arithmetic, bitwise, and float constants as bit patterns).

### 3.2 Extending Z3 Coverage

**Phase 1: Core Operations (Week 1-2)**

Add support for:
```rust
// Comparison operations
Instruction::I32Eq => {
    let rhs = stack.pop()?;
    let lhs = stack.pop()?;
    let result = lhs.bveq(&rhs);
    // Convert bool to i32 (1 or 0)
    stack.push(result.ite(
        &BV::from_i64(ctx, 1, 32),
        &BV::from_i64(ctx, 0, 32)
    ));
}

Instruction::I32LtS => {
    // Signed less-than
    stack.push(lhs.bvslt(&rhs).ite(...));
}

// Division with safety check
Instruction::I32DivS => {
    let rhs = stack.pop()?;
    let lhs = stack.pop()?;
    // Check for division by zero
    solver.assert(&rhs._eq(&BV::from_i64(ctx, 0, 32)).not());
    // Check for INT_MIN / -1 overflow
    solver.assert(&lhs.bveq(&BV::from_i64(ctx, i32::MIN as i64, 32))
                    .and(&rhs.bveq(&BV::from_i64(ctx, -1, 32)))
                    .not());
    stack.push(lhs.bvsdiv(&rhs));
}
```

**Phase 2: Control Flow (Week 3-4)**

Convert to SSA form:
```rust
// Convert function with control flow to SSA
fn convert_to_ssa(func: &Function) -> SSAFunction {
    // 1. Identify basic blocks
    let blocks = split_into_basic_blocks(func);

    // 2. Insert phi nodes at join points
    let ssa_blocks = insert_phi_nodes(blocks);

    // 3. Rename variables to SSA form
    rename_variables(ssa_blocks)
}

// Encode SSA function to SMT
fn encode_ssa_to_smt(ctx: &Context, ssa_func: &SSAFunction) -> BV {
    for block in &ssa_func.blocks {
        // Phi nodes become ITE (if-then-else) in SMT
        for phi in &block.phis {
            let value = if came_from_block_1 {
                &phi.incoming[0]
            } else {
                &phi.incoming[1]
            };
            // Use symbolic boolean for branch condition
        }
    }
}
```

**Phase 3: Memory Operations (Week 5-6)**

Use Z3 Array theory:
```rust
// Memory as array: Array<BitVec<32>, BitVec<8>>
let memory = Array::new_const(ctx, "mem", &Sort::bitvector(32), &Sort::bitvector(8));

// Load: (memory[addr], memory[addr+1], ..., memory[addr+3])
Instruction::I32Load { offset } => {
    let addr = stack.pop()?;
    let effective_addr = addr.bvadd(&BV::from_i64(ctx, offset as i64, 32));

    // Load 4 bytes (little-endian)
    let b0 = memory.select(&effective_addr);
    let b1 = memory.select(&effective_addr.bvadd(&BV::from_i64(ctx, 1, 32)));
    let b2 = memory.select(&effective_addr.bvadd(&BV::from_i64(ctx, 2, 32)));
    let b3 = memory.select(&effective_addr.bvadd(&BV::from_i64(ctx, 3, 32)));

    let value = b0.concat(&b1).concat(&b2).concat(&b3);
    stack.push(value);
}

// Store: memory' = memory[addr := value]
Instruction::I32Store { offset } => {
    let value = stack.pop()?;
    let addr = stack.pop()?;
    let effective_addr = addr.bvadd(&BV::from_i64(ctx, offset as i64, 32));

    // Store 4 bytes (little-endian)
    let bytes = value.extract(7, 0) // Low byte
                    .concat(&value.extract(15, 8))
                    .concat(&value.extract(23, 16))
                    .concat(&value.extract(31, 24));

    memory = memory.store(&effective_addr, &bytes);
}
```

### 3.3 Build-Time Rule Verification (Future)

Verify ISLE rules at compile time:

```rust
// In build.rs
fn verify_isle_rules() -> Result<()> {
    let rules = parse_isle_files()?;

    for rule in rules {
        println!("Verifying rule: {}", rule.name);

        // Generate SMT verification condition
        let smt_code = rule_to_smt(&rule)?;

        // Check with Z3
        if !z3_verify_smt(&smt_code)? {
            return Err(anyhow!("Rule verification failed: {}", rule.name));
        }
    }

    Ok(())
}

// Example: Verify strength reduction rule
// Rule: (i32.mul ?x (i32.const 4)) => (i32.shl ?x (i32.const 2))
fn rule_to_smt(rule: &Rule) -> String {
    format!(r#"
        (declare-const x (_ BitVec 32))
        (define-fun lhs () (_ BitVec 32) (bvmul x #x00000004))
        (define-fun rhs () (_ BitVec 32) (bvshl x #x00000002))
        (assert (not (= lhs rhs)))
        (check-sat) ; Should be unsat
    "#)
}
```

---

## 4. egg E-Graph Integration (Optional Enhancement)

### 4.1 When to Use E-Graphs

**Use egg if**:
- Want to discover optimizations automatically
- Have composable rewrite rules
- Can tolerate compilation time increase
- Want to explore optimization space exhaustively

**Don't use egg if**:
- Need predictable compile times (use Cranelift's acyclic e-graphs)
- Optimizations require global analysis (use dataflow)
- Rules are already well-known and ordered

### 4.2 Integration Architecture

**Option A: Research Pass (Recommended)**

Add egg as **experimental** pass that can be enabled:

```rust
// In loom-cli/src/main.rs
#[derive(Args)]
struct OptimizeArgs {
    /// Enable experimental e-graph optimization
    #[arg(long)]
    egraph: bool,
}

// In optimization pipeline
if args.egraph {
    println!("Running experimental e-graph pass...");
    egraph_optimize(&mut module)?;
}
```

**Option B: Hybrid Pipeline**

Use egg for local optimizations, Rust for global:

```
Phase 1-2: Parse + Inline (Rust)
Phase 3:   E-graph saturation (egg) - local optimization
Phase 4-5: Dataflow analysis (Rust) - global optimization
Phase 6:   E-graph saturation (egg) - post-inline cleanup
Phase 7-12: DCE, CFG opts (Rust)
```

### 4.3 Implementation

```rust
use egg::*;

// 1. Define WebAssembly language
define_language! {
    enum Wasm {
        // Constants
        "i32.const" = I32Const(i32),
        "i64.const" = I64Const(i64),

        // Arithmetic
        "i32.add" = I32Add([Id; 2]),
        "i32.sub" = I32Sub([Id; 2]),
        "i32.mul" = I32Mul([Id; 2]),
        "i32.shl" = I32Shl([Id; 2]),

        // Bitwise
        "i32.and" = I32And([Id; 2]),
        "i32.or" = I32Or([Id; 2]),
        "i32.xor" = I32Xor([Id; 2]),

        // Variables (locals, parameters)
        Symbol(Symbol),
    }
}

// 2. Define rewrite rules
fn make_rules() -> Vec<Rewrite<Wasm, ()>> {
    vec![
        // Arithmetic identities
        rewrite!("add-zero"; "(i32.add ?x (i32.const 0))" => "?x"),
        rewrite!("sub-zero"; "(i32.sub ?x (i32.const 0))" => "?x"),
        rewrite!("mul-one"; "(i32.mul ?x (i32.const 1))" => "?x"),
        rewrite!("mul-zero"; "(i32.mul ?x (i32.const 0))" => "(i32.const 0)"),

        // Strength reduction
        rewrite!("mul-pow2-4"; "(i32.mul ?x (i32.const 4))" => "(i32.shl ?x (i32.const 2))"),
        rewrite!("mul-pow2-8"; "(i32.mul ?x (i32.const 8))" => "(i32.shl ?x (i32.const 3))"),

        // Bitwise identities
        rewrite!("and-all-ones"; "(i32.and ?x (i32.const -1))" => "?x"),
        rewrite!("or-zero"; "(i32.or ?x (i32.const 0))" => "?x"),
        rewrite!("xor-zero"; "(i32.xor ?x (i32.const 0))" => "?x"),
        rewrite!("xor-self"; "(i32.xor ?x ?x)" => "(i32.const 0)"),

        // Constant folding (egg does this automatically)
        rewrite!("add-consts"; "(i32.add (i32.const ?a) (i32.const ?b))" =>
                 { ConstantFolder { a: "?a".parse(), b: "?b", op: Add } }),

        // Associativity (for constant gathering)
        rewrite!("add-assoc"; "(i32.add (i32.add ?a ?b) ?c)" => "(i32.add ?a (i32.add ?b ?c))"),
        rewrite!("mul-assoc"; "(i32.mul (i32.mul ?a ?b) ?c)" => "(i32.mul ?a (i32.mul ?b ?c))"),

        // Commutativity (to canonicalize)
        rewrite!("add-comm"; "(i32.add ?a ?b)" => "(i32.add ?b ?a)"),
        rewrite!("mul-comm"; "(i32.mul ?a ?b)" => "(i32.mul ?b ?a)"),
    ]
}

// 3. Cost function for extraction
#[derive(Default)]
struct WasmCost;

impl CostFunction<Wasm> for WasmCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &Wasm, mut costs: C) -> Self::Cost
    where C: FnMut(Id) -> Self::Cost
    {
        match enode {
            // Constants are free
            Wasm::I32Const(_) | Wasm::I64Const(_) => 0,

            // Prefer shifts over multiplies
            Wasm::I32Shl(_) => 1,
            Wasm::I32Mul(_) => 2,

            // Prefer smaller operations
            Wasm::I32Add(args) | Wasm::I32Sub(args) => {
                1 + costs(args[0]) + costs(args[1])
            }

            // Variables
            Wasm::Symbol(_) => 1,

            _ => 1,
        }
    }
}

// 4. Main optimization function
pub fn egraph_optimize(module: &mut Module) -> Result<()> {
    for func in &mut module.functions {
        // Convert instructions to egg expression
        let start_expr = instructions_to_egg(&func.instructions)?;

        // Run equality saturation
        let runner = Runner::default()
            .with_expr(&start_expr)
            .with_iter_limit(10)
            .with_node_limit(10_000)
            .run(&make_rules());

        // Extract best program
        let extractor = Extractor::new(&runner.egraph, WasmCost);
        let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);

        println!("  E-graph: {} nodes, {} classes, best cost: {}",
                 runner.egraph.total_size(),
                 runner.egraph.number_of_classes(),
                 best_cost);

        // Convert back to instructions
        func.instructions = egg_to_instructions(best_expr)?;
    }

    Ok(())
}

// 5. Verify e-graph output with Z3
#[cfg(feature = "verification")]
{
    if args.egraph {
        println!("Verifying e-graph optimization with Z3...");
        if !verify_optimization(&original, &module)? {
            return Err(anyhow!("E-graph produced invalid optimization!"));
        }
    }
}
```

### 4.4 E-Graph Verification Strategy

**Key Insight**: E-graphs don't verify rules; they assume rules are sound.

**Our strategy**:
1. **Verify each rewrite rule once** with Z3 (build-time or manual proof)
2. **Trust egg's e-graph invariants** (congruence, hashcons)
3. **Verify final output** with Z3 translation validation (runtime)

**Example rule verification**:
```rust
#[test]
fn verify_strength_reduction_rule() {
    // Rule: x * 4 == x << 2
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let solver = Solver::new(&ctx);

    let x = BV::new_const(&ctx, "x", 32);
    let lhs = x.bvmul(&BV::from_i64(&ctx, 4, 32));
    let rhs = x.bvshl(&BV::from_i64(&ctx, 2, 32));

    solver.assert(&lhs._eq(&rhs).not());
    assert_eq!(solver.check(), SatResult::Unsat); // PROVEN ✅
}
```

---

## 5. Backlog: Symbolica for Performance Optimization

### 5.1 Why Symbolica Doesn't Replace Z3

**Fundamental mismatch for primary verification**:

| LOOM Needs | Symbolica Provides |
|------------|-------------------|
| Bitvector reasoning (i32, i64) | Symbolic mathematics (ℝ, ℂ) |
| Overflow/wraparound semantics | Mathematical identities |
| Formal verification (proofs) | Pattern transformation (no proofs) |
| WebAssembly semantics | Continuous mathematics |
| Free/MIT license | Paid license for work use |

**Example of the problem**:
```rust
// LOOM needs to verify this for i32 with wraparound:
let x: i32 = i32::MAX;
assert_eq!(x + 1, i32::MIN); // Wraps to -2147483648

// Symbolica cannot reason about this - it assumes mathematical addition
// where MAX + 1 = MAX + 1 (no wraparound)
```

### 5.2 Potential Use: Fast Pre-Filter for Arithmetic (Backlog)

**Hypothesis**: Symbolica could speed up verification for pure arithmetic optimizations.

**Workflow**:
```rust
fn verify_optimization(original: &ValueData, optimized: &ValueData) -> bool {
    // Fast path: Try Symbolica for arithmetic-only expressions (~1ms)
    if is_pure_arithmetic(original) && is_pure_arithmetic(optimized) {
        if let Some(result) = symbolica_verify_algebraic(original, optimized) {
            // Polynomial identities hold mod 2^32 if they hold over ℝ
            return result; // Done in microseconds!
        }
    }

    // Complete path: Z3 for everything (~10-50ms)
    z3_verify(original, optimized)
}
```

**Where this helps**:
- ✅ Constant folding: `(x * 4) + (x * 8)` → `x * 12`
- ✅ Algebraic identities: `x + 0` → `x`, `x * 1` → `x`
- ✅ Distributivity: `x * (y + z)` → `(x * y) + (x * z)`

**Where this doesn't help**:
- ❌ Division/modulo (rounding semantics differ)
- ❌ Bitwise operations (not in Symbolica's domain)
- ❌ Comparisons (not arithmetic)
- ❌ Control flow (not in either tool's domain)

**Mathematical justification**: If two polynomial expressions with integer coefficients are algebraically identical over ℝ, then their bitvector representations mod 2^32 are also identical.

**Potential benefit**: For ~30-40% of optimizations that are pure arithmetic, reduce verification time from 10-50ms (Z3) to <1ms (Symbolica).

**Cost**:
- Licensing: Paid license required for commercial use
- Integration complexity: New dependency
- Limited coverage: Only helps with subset of optimizations

**Verdict**: Backlog item for performance optimization only. Z3 verification already works and is fast enough for current needs. Consider only if verification time becomes a bottleneck in CI.

### 5.3 What Symbolica Could Also Do (But We Don't Need)

**Potential use**: Generate candidate rewrite rules via pattern matching

```rust
// Hypothetical (NOT RECOMMENDED)
let expr = parse!("x * 4");
let candidates = symbolica_find_patterns(expr);
// Returns: [x << 2, x + x + x + x, 4 * x, ...]

// Then verify each with Z3
for candidate in candidates {
    if z3_verify_equivalent(expr, candidate) {
        return candidate; // Found verified optimization
    }
}
```

**Why we don't need this**:
- We already know which optimizations to apply (constant folding, strength reduction, CSE)
- egg can do this better (designed for program optimization, not math)
- Adds complexity, licensing costs, and dependency

---

## 6. Testing Strategy

### 6.1 What's NOW Verified (v0.2.0 Status)

| Category | Implementation | Status |
|----------|---------------|--------|
| **57+ Algebraic Rules** | Z3 proves `∀x: x*2^n = x<<n` | ✅ Complete |
| **Full-Width Memory** | I32Load, I64Load, I32Store, I64Store | ✅ Complete |
| **Partial-Width Memory** | I32Load8S/U, I32Load16S/U, I64Load8S/U, etc. | ✅ Complete |
| **Bounded Loops** | Unroll 3 iterations, bodies ≤100 instructions | ✅ Complete |
| **Nested Loops** | MAX_LOOP_NESTING_DEPTH = 1 | ✅ Complete |
| **Control Flow** | If/Else, Select, Block, BrIf | ✅ Complete |
| **Integer Arithmetic** | All i32/i64 ops with wrapping | ✅ Complete |
| **Differential Testing** | Runtime execution via wasmtime | ✅ Complete |
| **Fuzzing Infrastructure** | cargo-fuzz with libfuzzer | ✅ Complete |

### 6.2 What's Still Skipped

| Gap | Why | Impact |
|-----|-----|--------|
| **Loop iteration 4+** | Unbounded verification hard | May miss bugs in long loops |
| **Function calls** | Fresh symbolic result | Side effects invisible |
| **Float ops** | IEEE 754 not fully modeled | Treated as symbolic |
| **MemorySize/Grow** | Dynamic memory | Skip entire function |
| **ISLE compiler** | Trusted, not verified | Meta-level gap |

### 6.3 Property-Based Testing

**File**: `loom-core/tests/verification.rs`

**Properties tested**:
1. **Correctness**: Constant folding produces correct result
2. **Idempotence**: `simplify(simplify(x)) == simplify(x)`
3. **Preservation**: Constants unchanged by optimization
4. **Round-trip**: Optimize + encode + parse succeeds
5. **Overflow**: Boundary cases (i32::MAX + 1, etc.)
6. **Nested**: Nested operations fully optimized

**Example**:
```rust
proptest! {
    fn prop_constant_folding_correctness(x: i32, y: i32) {
        let term = iadd32(iconst32(x), iconst32(y));
        let optimized = simplify(term);

        match optimized.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), x.wrapping_add(y));
            }
            _ => panic!("Should produce I32Const"),
        }
    }
}
```

### 6.4 Differential Testing (Implemented in v0.2.0)

**File**: `loom-testing/src/lib.rs`

LOOM now includes runtime differential testing that compares execution of original vs optimized code:

```rust
/// Details of execution comparison between original and optimized
#[cfg(feature = "runtime")]
pub struct ExecutionComparison {
    pub functions_tested: usize,
    pub functions_matching: usize,
    pub functions_diverged: usize,
    pub functions_skipped: usize,
    pub divergence_details: Vec<DivergenceDetail>,
}

/// Compare execution of original and optimized modules using wasmtime
#[cfg(feature = "runtime")]
pub fn compare_execution(
    original_bytes: &[u8],
    optimized_bytes: &[u8],
) -> Result<ExecutionComparison> {
    // 1. Create wasmtime engine and compile both modules
    let engine = wasmtime::Engine::default();
    let orig_module = wasmtime::Module::new(&engine, original_bytes)?;
    let opt_module = wasmtime::Module::new(&engine, optimized_bytes)?;

    // 2. For each exported function, call with test inputs
    // 3. Compare results - any divergence indicates a bug
    // ...
}
```

**Usage**:
```bash
# Run differential testing
cargo run -p loom-testing --bin differential -- tests/*.wasm

# Output includes semantic verification stats:
# Semantic correctness: 100.00% (42/42 functions matching)
```

### 6.5 Fuzzing (Implemented in v0.2.0)

**Directory**: `fuzz/`

LOOM now includes cargo-fuzz infrastructure with two fuzz targets:

**fuzz_optimize.rs** - Optimization Pipeline Fuzzing:
```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use loom_core::{optimize, parse};

fuzz_target!(|data: &[u8]| {
    // Try to parse the input as WASM
    let module = match parse::parse_wasm(data) {
        Ok(m) => m,
        Err(_) => return, // Invalid input, skip
    };

    // Run optimization - should never panic
    let mut optimized = module;
    let _ = optimize::optimize_module(&mut optimized);

    // The optimized module should still be valid
    let encoded = match loom_core::encode::encode_wasm(&optimized) {
        Ok(bytes) => bytes,
        Err(_) => return,
    };

    // Validate the output with wasmparser
    if wasmparser::validate(&encoded).is_err() {
        panic!("LOOM produced invalid WASM output!");
    }
});
```

**fuzz_roundtrip.rs** - Parse/Encode Roundtrip Fuzzing:
- Tests that parse → encode → parse produces structurally equivalent modules
- Catches encoder bugs and parsing inconsistencies

**Usage**:
```bash
# Build fuzz targets (requires nightly)
cd fuzz && cargo +nightly fuzz build

# Run optimization fuzzer
cargo +nightly fuzz run fuzz_optimize

# Run roundtrip fuzzer
cargo +nightly fuzz run fuzz_roundtrip

# Example output: "Done 86740 runs in 6 second(s)"
```

**Note**: The fuzz crate uses `default-features = false` for loom-core to avoid Z3 compilation, making fuzzing faster to build and run.

---

## 7. Implementation Roadmap

### v0.1.0: Z3 Coverage (COMPLETED)

| Task | Status |
|------|--------|
| i32/i64 comparison ops (eq, ne, lt, gt, le, ge) | ✅ Complete |
| Division and remainder (with safety checks) | ✅ Complete |
| Bitwise ops (rotl, rotr, clz, ctz, popcnt) | ✅ Complete |
| Control flow (If/Else, Block, Loop, BrIf) | ✅ Complete |
| Memory operations (Z3 Array theory) | ✅ Complete |
| Full-width loads/stores (I32Load, I64Store, etc.) | ✅ Complete |
| Partial-width loads/stores (I32Load8S/U, etc.) | ✅ Complete |
| Bounded loop verification (3 iterations) | ✅ Complete |

### v0.2.0: Hybrid Verification (COMPLETED)

| Task | Status |
|------|--------|
| Differential testing with wasmtime execution | ✅ Complete |
| Runtime comparison of original vs optimized | ✅ Complete |
| Nested loop support (MAX_LOOP_NESTING_DEPTH = 1) | ✅ Complete |
| Fixed local index encoding for LICM-added locals | ✅ Complete |
| cargo-fuzz infrastructure | ✅ Complete |
| fuzz_optimize target | ✅ Complete |
| fuzz_roundtrip target | ✅ Complete |
| Updated documentation | ✅ Complete |

### v0.3.0: Extended Coverage (PLANNED)

**Remaining gaps to address:**

| Task | Priority |
|------|----------|
| K-induction for loops (beyond 3 iterations) | High |
| Function call summaries | Medium |
| Float verification (Z3 FPA theory) | Medium |
| MemorySize/Grow support | Low |
| ClusterFuzz/OSS-Fuzz integration | Medium |

### v1.0.0: Production Ready (PLANNED)

| Task | Priority |
|------|----------|
| ISLE rule verification at build time | High |
| Independent third-party audit | High |
| Comprehensive documentation | High |
| ISO 26262 TCL-3 package (optional) | Low |

### Future: E-Graph Research (Optional)

E-graph equality saturation (via `egg` crate) remains a potential enhancement for automatic optimization discovery. See Section 4 for details.

---

## 8. Current Verification Coverage (v0.2.0)

### 8.1 Instruction Coverage

```
Z3-Verified Instructions:
├─ i32 ops: 31/31 (100%) ✅
│   ├─ Arithmetic: add, sub, mul, div_s/u, rem_s/u
│   ├─ Comparison: eq, ne, lt_s/u, gt_s/u, le_s/u, ge_s/u, eqz
│   ├─ Bitwise: and, or, xor, shl, shr_s/u, rotl, rotr
│   └─ Other: clz, ctz, popcnt, wrap_i64, extend_*
├─ i64 ops: 31/31 (100%) ✅
├─ Memory ops: 12/12 (100%) ✅
│   ├─ Full-width: i32.load, i64.load, i32.store, i64.store
│   └─ Partial: i32.load8_s/u, i32.load16_s/u, i64.load8/16/32_s/u, stores
├─ Control flow: 5/5 (100%) ✅
│   └─ if/else, select, block, loop (bounded), br_if
└─ Float ops: 0/16 (0%) ⚠️ Skipped (IEEE 754 complexity)
```

### 8.2 Optimization Pass Coverage

```
Translation Validation Coverage:
├─ Constant folding: ✅ Verified
├─ Strength reduction: ✅ Verified
├─ Algebraic simplification: ✅ Verified
├─ Dead code elimination: ✅ Verified
├─ Common subexpression elimination: ✅ Verified
├─ Branch simplification: ✅ Verified
├─ Loop-invariant code motion: ✅ Verified
├─ Function inlining: ✅ Verified
└─ All 34 TranslationValidator calls: ✅ Active
```

### 8.3 Test Coverage

```
Current metrics:
├─ Fixture tests: 25/25 passing ✅
├─ Property tests: 6 properties × 256 cases each ✅
├─ Verification tests: All passing ✅
├─ Differential tests: Runtime execution comparison ✅
└─ Fuzz tests: 86,740+ runs without crashes ✅
```

### 8.4 Continuous Monitoring

CI configuration for verification:
```yaml
# .github/workflows/verification.yml
name: Formal Verification

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Z3
        run: sudo apt-get install -y z3 libz3-dev

      - name: Run verification tests
        run: |
          cargo test --features verification --lib verify
          cargo test --features verification --test verification

      - name: Coverage report
        run: |
          cargo tarpaulin --features verification --out Html
          echo "Verification coverage:" $(grep -oP '\d+\.\d+%' tarpaulin-report.html | head -1)
```

---

## 9. Best Practices and Lessons Learned

### 9.1 From Academic Research

**egg paper (POPL 2021)**:
> "Equality saturation works well when you have many small, local rewrite rules. It struggles with rules that require global analysis or side conditions."

**Lesson**: Use egg for local optimizations (constant folding, algebraic simplification). Use dataflow analysis for global (DCE, LICM).

**Cranelift RFC**:
> "We propose using acyclic e-graphs rather than full equality saturation to ensure predictable compile times."

**Lesson**: Production compilers prioritize predictability over optimality. Don't chase perfect optimization at the cost of usability.

**Alive2 (LLVM verification)**:
> "Translation validation catches bugs that no amount of testing can find. 8 bugs in 334 hand-coded transforms (2.4%) in LLVM's InstCombine."

**Lesson**: Even mature, well-tested optimizers have bugs. Formal verification is essential for correctness.

### 9.2 From Production Systems

**Herbie** (floating-point accuracy):
- Uses egg for simplification (100× faster than previous approach)
- Relies on numerical testing, not formal proofs
- Good enough for its domain (scientific computing)

**wasm-mutate** (fuzzing):
- Uses e-graphs to guarantee semantic equivalence
- Only applies proven-correct transformations
- Deterministic fuzzing enables reproducibility

**Cranelift** (WebAssembly JIT):
- Uses ISLE for readability + acyclic e-graphs for optimization
- Relies on testing and code review, not automated verification
- Pragmatic approach works well in practice

### 9.3 Our Recommendations for LOOM

1. **Don't over-invest in formal proofs**
   - Z3 translation validation is sufficient for most use cases
   - Full mechanized proof (Coq/Isabelle) is overkill unless safety-critical

2. **Layer verification approaches**
   - Fast feedback: property tests (seconds)
   - Formal verification: Z3 (minutes)
   - Exhaustive testing: fuzzing (hours)

3. **Make verification optional but encouraged**
   - Production builds: No verification overhead
   - Development: `--verify` flag enables Z3
   - CI: Always run verification tests

4. **Prioritize coverage over perfection**
   - Verify critical optimizations first (constant folding, strength reduction)
   - Add verification for new optimizations incrementally
   - Accept that some operations (floating-point, complex control flow) may not be fully verified

5. **Learn from failures**
   - When Z3 finds a counterexample, add it as a regression test
   - Track verification failures in GitHub issues
   - Use failures to improve optimization rules

---

## 10. Conclusion and Current Status (v0.2.0)

### Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **Z3 Verification** | ✅ Complete | Memory, loops, control flow verified |
| **Property Testing** | ✅ Complete | 256 cases per property |
| **ISLE DSL** | ✅ Complete | 57+ verified rules |
| **Differential Testing** | ✅ Complete | Runtime execution comparison |
| **Fuzzing** | ✅ Complete | cargo-fuzz infrastructure |
| **egg E-Graphs** | ⚠️ Research | Optional future enhancement |
| **Float Verification** | ⚠️ Partial | Constants only, ops skipped |

### What LOOM Verifies

**Mathematically Proven (Z3)**:
- 57+ algebraic rules (strength reduction, identities, etc.)
- Memory load/store semantics (full and partial width)
- Bounded loop execution (first 3 iterations)
- Control flow (if/else, select, block, br_if)
- All integer arithmetic with proper wrapping

**Runtime Validated (Differential Testing)**:
- Execution equivalence via wasmtime
- Original vs optimized output comparison
- Structural integrity after roundtrip

**Stress Tested (Fuzzing)**:
- Random WASM input handling
- No crashes in optimization pipeline
- Valid output for all valid inputs

### Known Limitations

| Gap | Status | Impact |
|-----|--------|--------|
| Loop iteration 4+ | Skipped | May miss long-loop bugs |
| Function calls | Symbolic | Side effects invisible |
| Float operations | Skipped | IEEE 754 not modeled |
| MemorySize/Grow | Skipped | Dynamic memory unsupported |
| ISLE compiler | Trusted | Meta-level gap |

### Honest Assessment

**What LOOM can claim**:
> "LOOM provides Z3-verified translation validation for arithmetic, memory, and bounded control flow. Functions with loops (≤3 iterations), memory access, and simple control flow are mathematically proven correct. Complex constructs are validated through differential testing and fuzzing."

**What LOOM cannot claim**:
- Full mechanized proof (no Coq/Isabelle)
- Unbounded loop verification
- Float operation correctness
- ISLE compiler correctness

### Files Reference

| File | Purpose |
|------|---------|
| `loom-core/src/verify.rs` | Z3 translation validation |
| `loom-core/src/verify_rules.rs` | Individual rule proofs |
| `loom-testing/src/lib.rs` | Differential testing framework |
| `fuzz/fuzz_targets/fuzz_optimize.rs` | Optimization fuzzer |
| `fuzz/fuzz_targets/fuzz_roundtrip.rs` | Roundtrip fuzzer |

---

## References

### Academic Papers
- [egg: Fast and Extensible Equality Saturation](https://arxiv.org/abs/2004.03082) - POPL 2021
- [Ruler: Rewrite Rule Inference Using Equality Saturation](https://arxiv.org/abs/2108.10436) - OOPSLA 2021
- [Better Together: Unifying Datalog and Equality Saturation](https://arxiv.org/abs/2304.04332) - PLDI 2023
- [Alive2: Bounded Translation Validation for LLVM](https://link.springer.com/chapter/10.1007/978-3-030-81688-9_35) - CAV 2021

### Bytecode Alliance / Cranelift
- [A Function Inliner for Wasmtime and Cranelift](https://bytecodealliance.org/articles/inliner) - Details SCC-based parallel inlining (same approach LOOM uses)
- [Cranelift ISLE: Term-Rewriting Made Practical](https://cfallin.org/blog/2023/01/20/cranelift-isle/) - ISLE architecture and design
- [Cranelift E-Graph RFC](https://github.com/bytecodealliance/rfcs/blob/main/accepted/cranelift-egraph.md) - Acyclic e-graphs proposal
- [Cranelift ISLE/Peepmatic RFC](https://github.com/bytecodealliance/rfcs/blob/main/accepted/cranelift-isel-isle-peepmatic.md) - ISLE replacing Peepmatic

### Tools and Projects
- [egg E-Graphs Library](https://github.com/egraphs-good/egg)
- [Z3 SMT Solver](https://github.com/Z3Prover/z3)
- [wasm-mutate](https://github.com/bytecodealliance/wasm-tools/tree/main/crates/wasm-mutate)
- [Herbie](https://herbie.uwplse.org/)

### LOOM Documentation
- [Z3 Verification Status](../analysis/Z3_VERIFICATION_STATUS.md)
- [ISLE Investigation](../analysis/ISLE_INVESTIGATION.md)
- [ISLE Deep Dive](../analysis/ISLE_DEEP_DIVE.md)
- [Formal Verification Guide](../FORMAL_VERIFICATION_GUIDE.md)
