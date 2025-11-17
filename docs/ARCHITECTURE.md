# LOOM Architecture

Deep dive into LOOM's internal architecture, optimization pipeline, and implementation details.

## Table of Contents
- [Overview](#overview)
- [Component Architecture](#component-architecture)
- [Optimization Pipeline](#optimization-pipeline)
- [ISLE Integration](#isle-integration)
- [Verification System](#verification-system)
- [Data Structures](#data-structures)
- [Performance Considerations](#performance-considerations)
- [Extension Points](#extension-points)

## Overview

LOOM is a WebAssembly optimizer built around three core principles:

1. **Correctness**: Formal verification ensures optimizations preserve semantics
2. **Performance**: Ultra-fast optimization (10-30 µs) with excellent results
3. **Maintainability**: Declarative ISLE rules make optimizations easy to understand and extend

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                         User Input                              │
│                    (WAT or WASM file)                          │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────┐
│                    Parser Module                                │
│  ┌──────────────┐        ┌─────────────────┐                  │
│  │ WAT Parser   │──────▶ │ wasmparser      │                  │
│  │ (wat crate)  │        │ Validation      │                  │
│  └──────────────┘        └─────────────────┘                  │
│                                 │                               │
│                                 ▼                               │
│                          ┌──────────────┐                      │
│                          │  AST Builder │                      │
│                          └──────────────┘                      │
└──────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                  Internal Representation                        │
│                                                                 │
│   Module {                                                      │
│     types: Vec<FuncType>,                                      │
│     functions: Vec<Function>,                                  │
│     tables: Vec<Table>,                                        │
│     memories: Vec<Memory>,                                     │
│     globals: Vec<Global>,                                      │
│     exports: Vec<Export>,                                      │
│   }                                                            │
│                                                                 │
│   Function {                                                    │
│     signature: FuncType,                                       │
│     locals: Vec<(count, ValueType)>,                          │
│     instructions: Vec<Instruction>,                            │
│   }                                                            │
└──────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│              12-Phase Optimization Pipeline                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 1: Precompute                                    │  │
│  │  - Global constant propagation                          │  │
│  │  - Replace immutable global.get with constants          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 2: ISLE Constant Folding (Pre-CSE)              │  │
│  │  - Convert to ISLE terms                                │  │
│  │  - Apply pattern matching rules                         │  │
│  │  - Fold constants: 10 + 20 → 30                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 3: Strength Reduction                            │  │
│  │  - x * 8 → x << 3  (2-3x speedup)                      │  │
│  │  - x / 4 → x >> 2  (2-3x speedup)                      │  │
│  │  - x % 32 → x & 31 (2-3x speedup)                      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 4: Common Subexpression Elimination (CSE)       │  │
│  │  - Hash expressions for duplicate detection             │  │
│  │  - Cache expensive computations in locals               │  │
│  │  - Skip caching simple constants                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 5: Function Inlining                             │  │
│  │  - Build call graph                                      │  │
│  │  - Inline small, frequently-called functions            │  │
│  │  - Substitute parameters with arguments                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 6: ISLE Constant Folding (Post-Inline)          │  │
│  │  - Fold constants exposed by inlining                   │  │
│  │  - Cross-function optimization                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 7: Code Folding                                  │  │
│  │  - Flatten nested blocks                                │  │
│  │  - Remove redundant control flow                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 8: Loop-Invariant Code Motion (LICM)            │  │
│  │  - Detect modified locals in loops                      │  │
│  │  - Hoist constants and unmodified locals                │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 9: Branch Simplification                         │  │
│  │  - Simplify constant conditions                         │  │
│  │  - Remove redundant branches                            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 10: Dead Code Elimination (DCE)                 │  │
│  │  - Remove unreachable code after terminators            │  │
│  │  - Clean up code following return/br/unreachable        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 11: Block Merging                                │  │
│  │  - Merge consecutive blocks                             │  │
│  │  - Reduce control flow overhead                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Phase 12: Vacuum & Simplify Locals                    │  │
│  │  - Remove empty blocks                                   │  │
│  │  - Eliminate unused local variables                     │  │
│  │  - Final cleanup pass                                    │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                  Verification (Optional)                        │
│                                                                 │
│  if --verify flag:                                             │
│                                                                 │
│    ┌──────────────────────────────────────────────┐           │
│    │  Z3 SMT Verification (if feature enabled)    │           │
│    │  - Encode original function to SMT           │           │
│    │  - Encode optimized function to SMT          │           │
│    │  - Prove equivalence via SAT solver          │           │
│    │  - Generate counterexample if failed         │           │
│    └──────────────────────────────────────────────┘           │
│                         │                                       │
│                         ▼                                       │
│    ┌──────────────────────────────────────────────┐           │
│    │  ISLE Property-Based Verification            │           │
│    │  - Check idempotence                         │           │
│    │  - Validate constant folding                 │           │
│    │  - Test algebraic properties                 │           │
│    └──────────────────────────────────────────────┘           │
└──────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                      Encoder Module                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  WAT Encoder (if --wat flag)                            │  │
│  │  - Convert AST to text format                           │  │
│  │  - Pretty-print with indentation                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  WASM Binary Encoder (default)                          │  │
│  │  - Encode to binary format                              │  │
│  │  - Ultra-fast: ~183 nanoseconds!                        │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                      Output File                                │
│                  (Optimized WASM/WAT)                          │
└────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. loom-core

The core optimization library. Contains all optimization passes and the main pipeline.

**Key Modules:**
- `parse`: WAT/WASM parsing using `wat` and `wasmparser` crates
- `optimize`: 12-phase optimization pipeline
- `encode`: WAT/WASM encoding using `wasmprinter` and `wasm-encoder`
- `verify`: Z3 SMT-based formal verification
- `component`: WebAssembly Component Model support
- `terms`: Conversion between instructions and ISLE terms

**File Structure:**
```
loom-core/
├── src/
│   ├── lib.rs              # Main module, optimization pipeline
│   └── verify.rs           # Z3 verification module
├── tests/
│   └── optimization_tests.rs  # Comprehensive test suite (20 tests)
└── benches/
    └── optimization_benchmarks.rs  # Criterion benchmarks
```

### 2. loom-isle

ISLE (Instruction Selection and Lowering Engine) integration. Defines term types and optimization rules.

**Key Files:**
- `isle/wasm_terms.isle`: Term type definitions (I32Add, I32Const, etc.)
- `isle/wasm_opts.isle`: Optimization rules (constant folding, algebraic simplification)
- `src/lib.rs`: Rust bindings and term conversion

**Term System:**
```rust
pub enum ValueData {
    I32Const { val: Imm32 },
    I32Add { lhs: Value, rhs: Value },
    I32Sub { lhs: Value, rhs: Value },
    LocalGet { idx: u32 },
    LocalSet { idx: u32, value: Value },
    // ... 40+ operations
}
```

### 3. loom-cli

Command-line interface. Provides user-friendly access to optimization and verification.

**Commands:**
```bash
loom optimize <input> [options]
  --output, -o <file>   # Output path
  --wat                 # Output as WAT text
  --stats               # Show statistics
  --verify              # Run verification
```

## Optimization Pipeline

### Phase Order Rationale

The 12-phase pipeline is carefully ordered to maximize optimization opportunities:

#### Why Constant Folding Before CSE (Phases 2 & 3)?

**Problem**: If CSE runs before constant folding:
```wasm
i32.const 42
i32.const 42
i32.add
```

CSE would cache the constant 42 in a local:
```wasm
i32.const 42
local.tee $0
local.get $0
i32.add
```

Now constant folding can't work because it's no longer two constants!

**Solution**: Run constant folding first:
```wasm
i32.const 42
i32.const 42
i32.add
→ i32.const 84  # Folded before CSE sees it
```

#### Why ISLE Twice (Phases 2 & 6)?

**Phase 2 (Pre-CSE)**: Fold obvious constants
**Phase 6 (Post-Inline)**: Fold constants exposed by inlining

Example:
```wasm
(func $add_ten (param $x i32) (result i32)
  local.get $x
  i32.const 10
  i32.add
)

(func $main (result i32)
  i32.const 5
  call $add_ten  # After inlining: 5 + 10, foldable!
)
```

#### Why Strength Reduction After Constant Folding?

Constants might create optimization opportunities:
```wasm
i32.const 2
i32.const 3
i32.add       # Folds to: i32.const 5
i32.const 8
i32.mul       # Now: 5 * 8, but 8 is power of 2!
              # Becomes: 5 << 3
```

### Optimization Pass Details

#### Strength Reduction

Replaces expensive operations with cheaper equivalents.

**Implementation** (`optimize_advanced_instructions`):
```rust
match (op, const_val) {
    // Multiplication by power of 2 → Shift left
    (Instruction::I32Mul, n) if n.is_power_of_two() => {
        // x * 8 → x << 3
        instructions.push(Instruction::I32Const(n.trailing_zeros() as i32));
        instructions.push(Instruction::I32Shl);
    }

    // Division by power of 2 → Shift right
    (Instruction::I32DivU, n) if n.is_power_of_two() => {
        // x / 4 → x >> 2
        instructions.push(Instruction::I32Const(n.trailing_zeros() as i32));
        instructions.push(Instruction::I32ShrU);
    }

    // Modulo by power of 2 → Bitwise AND
    (Instruction::I32RemU, n) if n.is_power_of_two() => {
        // x % 32 → x & 31
        instructions.push(Instruction::I32Const(n - 1));
        instructions.push(Instruction::I32And);
    }
}
```

**Speedup**: 2-3x for power-of-2 operations

#### Common Subexpression Elimination (CSE)

Caches duplicate expensive computations.

**Implementation**:
1. Hash all expressions to detect duplicates
2. Filter out simple constants (they should be folded, not cached)
3. Allocate new locals for cached values
4. Replace first occurrence with `local.tee $temp`
5. Replace duplicates with `local.get $temp`

**Example**:
```wasm
# Before
local.get $x
i32.const 4
i32.mul
local.get $x
i32.const 4
i32.mul  # Duplicate!

# After
local.get $x
i32.const 4
i32.mul
local.tee $temp
local.get $temp  # Reuse
```

**Key Design Decision**: Don't cache constants!
```rust
let const_duplicates: Vec<_> = duplicates
    .iter()
    .filter(|(orig_pos, _dup_pos, _type)| {
        // Skip simple constants - they're cheap and prevent constant folding
        !matches!(func.instructions.get(*orig_pos),
            Some(Instruction::I32Const(_)) | Some(Instruction::I64Const(_)))
    })
    .collect();
```

#### Function Inlining

Inlines small, frequently-called functions to reduce call overhead and enable cross-function optimizations.

**Algorithm**:
1. Build call graph
2. Identify inlining candidates (small functions, multiple call sites)
3. For each call site:
   - Substitute parameters with arguments
   - Insert function body
   - Update local indices
4. Remove inlined functions if no remaining callers

**Heuristics**:
- Inline if: `body_size < 20 instructions && call_count > 2`
- Or: `body_size < 5 instructions` (always inline tiny functions)

## ISLE Integration

### Term Conversion

**Instructions to Terms**:
```rust
pub fn instructions_to_terms(instructions: &[Instruction]) -> Result<Vec<Value>> {
    let mut stack: Vec<Value> = Vec::new();

    for instr in instructions {
        match instr {
            Instruction::I32Const(val) => {
                stack.push(iconst32(Imm32::from(*val)));
            }
            Instruction::I32Add => {
                let rhs = stack.pop().ok_or(...)?;
                let lhs = stack.pop().ok_or(...)?;
                stack.push(iadd32(lhs, rhs));
            }
            // ...
        }
    }

    Ok(stack)
}
```

**Terms to Instructions**:
```rust
pub fn terms_to_instructions(terms: &[Value]) -> Result<Vec<Instruction>> {
    let mut instructions = Vec::new();

    for term in terms {
        term_to_instructions_recursive(term, &mut instructions)?;
    }

    Ok(instructions)
}

fn term_to_instructions_recursive(term: &Value, out: &mut Vec<Instruction>) {
    match term.data() {
        ValueData::I32Const { val } => {
            out.push(Instruction::I32Const(val.value()));
        }
        ValueData::I32Add { lhs, rhs } => {
            term_to_instructions_recursive(lhs, out);  # Stack-based!
            term_to_instructions_recursive(rhs, out);
            out.push(Instruction::I32Add);
        }
        // ...
    }
}
```

### ISLE Rules

**Constant Folding**:
```lisp
;; From wasm_opts.isle
(rule (simplify (I32Add (I32Const a) (I32Const b)))
      (I32Const (iadd_imm32 a b)))

(rule (simplify (I32Mul (I32Const a) (I32Const b)))
      (I32Const (imul_imm32 a b)))
```

**Algebraic Simplification**:
```lisp
;; x + 0 → x
(rule (simplify (I32Add x (I32Const (imm32_zero))))
      x)

;; x * 1 → x
(rule (simplify (I32Mul x (I32Const (imm32_one))))
      x)

;; x | 0 → x
(rule (simplify (I32Or x (I32Const (imm32_zero))))
      x)
```

## Verification System

### Z3 SMT Encoding

**Translation Validation Approach**:
1. Encode original function as SMT formula
2. Encode optimized function as SMT formula
3. Assert: `original(inputs) ≠ optimized(inputs)`
4. Ask Z3: Is this SAT (satisfiable)?
   - If UNSAT: Functions are equivalent! ✅
   - If SAT: Found counterexample! ❌
   - If UNKNOWN: Too complex or timeout

**Example Encoding**:
```rust
fn encode_function_to_smt<'ctx>(ctx: &'ctx Context, func: &Function) -> Result<BV<'ctx>> {
    let mut stack: Vec<BV<'ctx>> = Vec::new();
    let mut locals: Vec<BV<'ctx>> = Vec::new();

    // Parameters as symbolic inputs
    for (idx, param_type) in func.signature.params.iter().enumerate() {
        let width = match param_type {
            ValueType::I32 => 32,
            ValueType::I64 => 64,
            _ => bail!("Unsupported type"),
        };
        let param = BV::new_const(ctx, format!("param{}", idx), width);
        locals.push(param);
    }

    // Symbolically execute
    for instr in &func.instructions {
        match instr {
            Instruction::I32Add => {
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvadd(&rhs));  // SMT bit-vector addition
            }
            Instruction::I32Mul => {
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(lhs.bvmul(&rhs));  // SMT bit-vector multiplication
            }
            // ...
        }
    }

    Ok(stack.pop().unwrap())
}
```

**Verification Query**:
```rust
pub fn verify_optimization(original: &Module, optimized: &Module) -> Result<bool> {
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let solver = Solver::new(&ctx);

    for (orig_func, opt_func) in original.functions.iter().zip(optimized.functions.iter()) {
        let orig_formula = encode_function_to_smt(&ctx, orig_func)?;
        let opt_formula = encode_function_to_smt(&ctx, opt_func)?;

        // Assert: original ≠ optimized
        solver.assert(&orig_formula._eq(&opt_formula).not());

        match solver.check() {
            SatResult::Unsat => continue,  // Equivalent! ✅
            SatResult::Sat => {
                // Counterexample found!
                let model = solver.get_model()?;
                eprintln!("Counterexample: {}", model);
                return Ok(false);
            }
            SatResult::Unknown => bail!("SMT solver timeout"),
        }
    }

    Ok(true)  // All functions verified!
}
```

## Data Structures

### Module Representation

```rust
pub struct Module {
    pub types: Vec<FuncType>,           // Function signatures
    pub functions: Vec<Function>,       // Function definitions
    pub tables: Vec<Table>,             // Indirect function tables
    pub memories: Vec<Memory>,          // Linear memory definitions
    pub globals: Vec<Global>,           // Global variables
    pub exports: Vec<Export>,           // Exported functions/memory
    pub start: Option<u32>,             // Start function
    pub elements: Vec<Element>,         // Table initializers
    pub code: Vec<FunctionBody>,        // Function bodies
    pub data: Vec<Data>,                // Data segments
}
```

### Instruction Enum

```rust
pub enum Instruction {
    // Constants
    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),

    // Arithmetic
    I32Add, I32Sub, I32Mul, I32DivS, I32DivU, I32RemS, I32RemU,
    I64Add, I64Sub, I64Mul, I64DivS, I64DivU, I64RemS, I64RemU,

    // Bitwise
    I32And, I32Or, I32Xor, I32Shl, I32ShrS, I32ShrU, I32Rotl, I32Rotr,
    I64And, I64Or, I64Xor, I64Shl, I64ShrS, I64ShrU, I64Rotl, I64Rotr,

    // Comparison
    I32Eqz, I32Eq, I32Ne, I32LtS, I32LtU, I32GtS, I32GtU, I32LeS, I32LeU, I32GeS, I32GeU,
    I64Eqz, I64Eq, I64Ne, I64LtS, I64LtU, I64GtS, I64GtU, I64LeS, I64LeU, I64GeS, I64GeU,

    // Local variables
    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),

    // Global variables
    GlobalGet(u32),
    GlobalSet(u32),

    // Memory
    I32Load { offset: u32, align: u32 },
    I64Load { offset: u32, align: u32 },
    I32Store { offset: u32, align: u32 },
    I64Store { offset: u32, align: u32 },
    MemorySize,
    MemoryGrow,

    // Control flow
    Block { block_type: BlockType, body: Vec<Instruction> },
    Loop { block_type: BlockType, body: Vec<Instruction> },
    If { block_type: BlockType, then_body: Vec<Instruction>, else_body: Vec<Instruction> },
    Br(u32),
    BrIf(u32),
    BrTable { targets: Vec<u32>, default: u32 },
    Return,
    Call(u32),
    CallIndirect { type_idx: u32 },

    // Other
    Drop,
    Select,
    Unreachable,
    Nop,
    End,
}
```

## Performance Considerations

### Why So Fast?

**Benchmark Results**: 10-30 µs for most modules

**Key Optimizations**:
1. **Minimal allocations**: Reuse vectors where possible
2. **Single-pass parsing**: No intermediate representations
3. **Lazy evaluation**: Only convert to ISLE terms when needed
4. **Efficient encoding**: wasm-encoder is extremely fast (183ns!)
5. **No file I/O in hot path**: Read once, optimize in memory

### Bottlenecks

**Measured via profiling**:
- Parsing: ~7 µs (fast!)
- ISLE conversion: ~2-3 µs per function
- Optimization passes: ~5-15 µs total
- Encoding: ~0.18 µs (183ns - negligible!)

**Slowest phases**:
1. CSE (expression hashing)
2. Function inlining (call graph analysis)
3. Loop optimization (modified local tracking)

### Scalability

**Tested on**:
- Small modules (10 functions): <10 µs
- Medium modules (100 functions): ~50-100 µs
- Large modules (1000 functions): ~500 µs - 1ms

**Linear scaling**: O(n) where n = number of instructions

## Extension Points

### Adding New Optimizations

1. **Add to pipeline** (`optimize_module` in `lib.rs`):
```rust
pub fn optimize_module(module: &mut Module) -> Result<()> {
    precompute(module)?;
    // ... existing phases ...
    my_new_optimization(module)?;  // Add here!
    vacuum(module)?;
    Ok(())
}
```

2. **Implement the pass**:
```rust
pub fn my_new_optimization(module: &mut Module) -> Result<()> {
    for func in &mut module.functions {
        func.instructions = optimize_instructions(&func.instructions);
    }
    Ok(())
}

fn optimize_instructions(instructions: &[Instruction]) -> Vec<Instruction> {
    // Your optimization logic here
}
```

### Adding New ISLE Rules

1. **Add rule to `wasm_opts.isle`**:
```lisp
;; New optimization: x - x → 0
(rule (simplify (I32Sub x x))
      (I32Const (imm32_zero)))
```

2. **Rebuild**: ISLE compiler regenerates Rust code automatically

### Adding New Verification Checks

1. **Add to `run_verification`** in `loom-cli/src/main.rs`:
```rust
fn run_verification(original: &Module, optimized: &Module) -> Result<()> {
    // Z3 verification ...
    // ISLE verification ...

    // Your new check:
    println!("🔍 Running custom verification...");
    my_verification_check(original, optimized)?;

    Ok(())
}
```

## Future Directions

### Planned Improvements

1. **More aggressive LICM**:
   - Hoist pure arithmetic operations
   - Hoist global.get of immutable globals
   - Track memory dependencies

2. **Profile-guided optimization**:
   - Collect runtime profiles
   - Inline hot functions
   - Specialize for common inputs

3. **SIMD optimizations**:
   - Auto-vectorize loops
   - SIMD-specific patterns

4. **Inter-procedural analysis**:
   - Global CSE across functions
   - Whole-program optimization

5. **Custom optimization passes**:
   - User-defined ISLE rules
   - Plugin architecture

### Research Directions

1. **Machine learning for optimization ordering**:
   - Learn optimal phase ordering
   - Predict which optimizations will help

2. **Probabilistic verification**:
   - Random testing with counterexample generation
   - Property-based testing framework

3. **Component Model integration**:
   - Optimize at component boundaries
   - Cross-component inlining

---

**For more information**:
- [Usage Guide](USAGE_GUIDE.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [Formal Verification Guide](FORMAL_VERIFICATION_GUIDE.md)
- [WASM Build Guide](WASM_BUILD.md)
