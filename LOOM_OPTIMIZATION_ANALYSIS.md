# LOOM Optimization Passes: Comprehensive Analysis

## Overview
Loom implements a sophisticated 12-phase WebAssembly optimization pipeline with ISLE-based term rewriting, advanced instruction optimizations, and control flow analysis. This document provides a detailed breakdown of each optimization pass with line numbers, pattern matching rules, and transformation examples.

---

## PART 1: CORE OPTIMIZATION PIPELINE

The main optimization pipeline is defined in `/home/user/loom/loom-cli/src/main.rs` (lines 237-246) and executed in the following order:

### 1. **Precompute / Global Constant Propagation (Phase 19)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 4070-4120)

**What it does:**
Propagates immutable global constants throughout the module by replacing `global.get $x` with constant values when:
- The global is immutable (!mutable)
- The global has a constant initializer

**Pattern Matching:**
- Detects: `global.get(idx)` where `globals[idx]` has constant initializer
- Replaces with: The constant value itself

**Example Transformations:**
```wasm
Before:
  (global $FLAG (mut i32) (i32.const 1))
  (func
    global.get $FLAG
    if ...
  )

After:
  (func
    i32.const 1
    if ...
  )
```

**Benefits:**
- Enables branch simplification when globals are boolean flags
- Enables constant folding when globals are numeric constants
- Enables dead code elimination when paths become unreachable

**Limitations/TODOs:**
- Only handles single constant instruction initializers
- Requires immutability constraint

---

### 2. **Constant Folding (Phase 12 - ISLE-Based)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 2974-3007)

**What it does:**
Uses ISLE pattern matching rules to fold constants at optimization time. Converts expressions with constant operands into single constant results.

**Pattern Matching Rules (ISLE):**
ISLE rules defined in `/home/user/loom/loom-shared/isle/wasm_terms.isle` (lines 508-549):

The ISLE file defines:
- Helper functions: `imm32_add`, `imm32_sub`, `imm32_mul`, `imm32_and`, `imm32_or`, `imm32_xor`, `imm32_shl`, `imm32_shr_s`, `imm32_shr_u` (lines 436-481)
- Equivalent 64-bit versions: `imm64_*` (lines 447-505)
- Default fallback rule: `(rule (simplify val) val)` (line 549)

**Example Transformations:**
```wasm
i32.const 100
i32.const 200
i32.add
→ i32.const 300

i64.const 0x1000000000
i64.const 0x2
i64.mul
→ i64.const 0x2000000000

i32.const 16
i32.const 3
i32.shl
→ i32.const 128
```

**Implementation:**
Lines 2985-3003: Converts instructions to ISLE Value terms, applies simplification with environment tracking, converts back to instructions.

**Note:** Comments at lines 532-545 of wasm_terms.isle explain that constant folding is currently implemented as Rust peephole optimizations rather than ISLE rules, pending proper extractor setup.

---

### 3. **Common Subexpression Elimination (CSE - Phase 20)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 4115-4270)

**State:** Conservative/MVP implementation

**What it does:**
Basic duplicate detection for simple expressions (constants only). Full expression-tree CSE is incomplete.

**Pattern Matching:**
- Current: Only caches self-contained constants
- Disabled: Operations like `i32.add` cannot be cached without operands

**Current Limitations (Comments at lines 4117-4135):**
- Stack-based WASM operations consume values (can't cache `i32.add` alone)
- Previous implementation incorrectly used `local.tee`, leaving extra stack values
- Requires proper stack simulation and dependency tracking

**Future Implementation:**
The enhanced version (lines 4283-4633) provides expression-tree framework:
- Phase 1: Build expression trees from stack simulation
- Phase 2: Find duplicates with stable hashing
- Phase 3: Allocate local variables
- Phase 4: Transform instructions (conservative for simple expressions only)

**Example (Working for Enhanced CSE):**
```wasm
Before:
  local.get $x
  local.get $y
  i32.add
  (use result)
  local.get $x
  local.get $y
  i32.add  ;; duplicate
  (use result)

After:
  local.get $x
  local.get $y
  i32.add
  local.tee $tmp
  (use result)
  local.get $tmp
  (use result)
```

---

### 4. **Optimize Advanced Instructions (Issue #21)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 4645-5178)

**What it does:**
Peephole optimizations including:
- Strength reduction (mul/div/rem by power of 2)
- Bitwise tricks (x^x→0, x&x→x, etc.)
- Algebraic simplifications
- Memory operation patterns

**Strength Reduction Examples:**
```wasm
// Multiplication by power of 2
x * 8  → x << 3
x * 16 → x << 4

// Division by power of 2 (unsigned)
x / 16 → x >> 4
x / 32 → x >> 5

// Remainder by power of 2 (unsigned)
x % 32 → x & 31
x % 64 → x & 63

// 64-bit equivalents
x * 0x1000 → x << 12 (if power of 2)
```

**Algebraic Simplifications:**
```wasm
x * 0  → 0
x * 1  → x (identity)
x + 0  → x (identity)
x - 0  → x (identity)
x / 1  → x (identity)
x % 1  → 0 (always zero)

// Bitwise
x & 0       → 0 (absorption)
x | 0xFFFF  → 0xFFFF (absorption)
x | 0       → x (identity)
x & 0xFFFF  → x (identity)
x ^ 0       → x (identity)
x << 0      → x (identity)
x >> 0      → x (identity)
```

**Self-Operation Patterns (3+ instruction sequences):**
```wasm
local.get $a
local.get $a
i32.xor  → i32.const 0

local.get $a
local.get $a
i32.and  → local.get $a

local.get $a
local.get $a
i32.or   → local.get $a

local.get $a
local.get $a
i32.sub  → i32.const 0
```

**Helper Functions:**
- `is_power_of_two(n: i32)`: Checks `n > 0 && (n & (n-1)) == 0`
- `log2_i32(n: i32)`: Bit-shift loop to find log base 2
- `is_power_of_two_u32(n: u32)`: Unsigned version
- `log2_u32(n: u32)`: Unsigned version

**Testing (from optimization_tests.rs):**
Lines 29-105 contain tests for:
- Strength reduction: mul/8, div/16, rem/32
- Bitwise: x^x, x&x, x|x, x-x
- Algebraic: x+0, x*0, x*1

---

### 5. **Branch Simplification (Phase 15 - Issue #16)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 3127-3240)

**What it does:**
Removes redundant branches and folds constant conditions in if/br_if statements.

**Pattern Matching:**
```wasm
// Pattern 1: constant followed by br_if
(i32.const 0) (br_if $label)  → remove (condition false, never taken)
(i32.const N) (br_if $label)  → br $label (condition true, always taken)

// Pattern 2: constant followed by if
(i32.const 0) (if ...)  → take else branch
(i32.const N) (if ...)  → take then branch

// Pattern 3: Nop elimination
nop  → remove
```

**Example:**
```wasm
Before:
  i32.const 0
  br_if $exit

After:
  (removed - condition never true)

Before:
  i32.const 1
  if $type
    (then_body)
  else
    (else_body)
  end

After:
  (then_body)  ;; takes then branch always
```

**Implementation Details (lines 3135-3240):**
- Two-instruction lookahead for patterns
- Recursive processing of nested blocks
- Early elimination with continue statement to skip pattern match

---

### 6. **Dead Code Elimination (Phase 14 - Issue #13)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 3008-3125)

**What it does:**
Removes unreachable code that follows terminators (return, br, unreachable).

**Pattern Matching:**
```wasm
terminator instruction → mark rest of block unreachable
```

**Terminators Recognized:**
- `return` - function exit
- `br $label` - unconditional branch
- `unreachable` - trap instruction

**Example:**
```wasm
Before:
  i32.const 1
  return
  i32.const 2
  i32.add
  (more unreachable code)

After:
  i32.const 1
  return
  ;; unreachable code removed
```

**Recursion:**
- Processes blocks, loops, and if statements recursively
- Cleans nested structures independently

---

### 7. **Block Merging (Phase 16 - Issue #17)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 3241-3407)

**What it does:**
Merges nested blocks with compatible types to reduce CFG complexity.

**Type Compatibility Check (lines 3374-3407):**
```wasm
compatible if:
- Both Empty
- Both same ValueType (I32/I64)
- Both same function signature (params and results match)

NOT compatible:
- Empty + Value mismatch
- Mixed function/non-function types
```

**Critical Safety Check (lines 3305-3320):**
⚠️ **CRITICAL**: Blocks containing branch instructions (Br, BrIf, BrTable) are never merged because merging invalidates branch depths.

**Example:**
```wasm
Before:
  block $outer
    block $inner
      i32.const 1
      i32.const 2
      i32.add
    end  ;; $inner
  end  ;; $outer

After:
  block $outer
    i32.const 1
    i32.const 2
    i32.add
  end  ;; merged
```

---

### 8. **Vacuum Cleanup Pass (Phase 17 - Issue #20)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 3408-3548)

**What it does:**
Final cleanup pass that:
- Removes nops
- Unwraps trivial blocks
- Simplifies degenerate patterns

**Trivial Block Detection (lines 3492-3548):**
- Empty blocks: unwrap
- Single-instruction blocks: unwrap if type-compatible
  - Special case: NEVER unwrap Loop blocks (may contain br_if targeting outer block)
  - For Value types: only unwrap known value-producing instructions
  - For Empty blocks: generally safe

**Safe-to-unwrap Instructions:**
```wasm
I32Const(_), I64Const(_), arithmetic, LocalGet(_), Block/If
```

**Loop Safety Constraint:**
Loops are special because they contain `br_if` targeting the loop's outer block. Unwrapping removes the target label.

---

### 9. **SimplifyLocals (Phase 18 - Issue #15)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 3549-3764)

**What it does:**
Optimizes local variable usage by:
1. Detecting redundant copies (local.get → local.set)
2. Tracking equivalent locals
3. Eliminating dead stores
4. Simplifying tees of unused locals

**Implementation:**
- Iterative process (max 10 iterations) until fixed point
- Analyzes usage patterns from position tracking
- Canonicalizes locals based on equivalence relations

**Pattern Matching (Equivalence Detection):**
```wasm
local.get $src
local.set $dst
→ $dst ≡ $src (equivalence established)
```

**Canonicalization:**
All subsequent `local.get $dst` are replaced with `local.get $src` (the canonical representative).

**Dead Store Detection:**
```wasm
local.set $x → marked as dead if $x has no gets
(but currently kept for stack balance)
```

**Limitations (Line 3640):**
```rust
// TODO: Add proper dead store elimination with stack analysis
```

---

### 10. **CoalesceLocals - Register Allocation (Phase 12.5)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 3765-3996)

**What it does:**
Merges non-overlapping local variables using graph coloring algorithm.

**Algorithm Steps:**

**Step 1: Compute Live Ranges (lines 3828-3912)**
- Track first definition and last use for each local
- Recursively scan all blocks/loops/if statements
- Parameters treated as always-live (never coalesced)

**Step 2: Build Interference Graph (lines 3913-3934)**
- Create nodes for each local with live range
- Add edges between locals with overlapping ranges
- Two-phase check: `start < other.end && other.start < end`

**Step 3: Graph Coloring - Greedy Algorithm (lines 3935-3983)**
```rust
Greedy Algorithm:
1. Sort nodes by degree (highest first)
2. For each node:
   - Find colors used by neighbors
   - Assign smallest available color not in use
```

**Example:**
```
Locals: a(1-5), b(3-7), c(6-10)
Live ranges: a overlaps b, b overlaps c, a doesn't overlap c

Interference edges: (a,b), (b,c)

Coloring:
  a → color 0
  b → color 1 (conflicts with a's color 0)
  c → color 0 (doesn't conflict with b)

Result: 2 registers instead of 3
```

**Step 4: Remap Locals**
- Update all local references with new indices
- Rebuild locals vector with new type declarations

**Benefits:**
- 10-15% binary size reduction expected
- Lower indices use smaller LEB128 encoding
- Matches wasm-opt's register allocation

**Constraints:**
- Skips if dead locals exist (SimplifyLocals must run first)
- Only coalesces when actual reduction achieved

---

### 11. **Function Inlining (Issue #14 - CRITICAL)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 5179-5400)

**What it does:**
Inlines small functions and single-call-site functions to:
- Enable constant propagation across function boundaries
- Reduce call overhead
- Eliminate parameter passing overhead

**Inlining Heuristics (lines 5207-5221):**
```rust
Inline if:
1. Single call site, OR
2. Small function (< 10 instructions)

But don't inline if:
3. Function larger than 50 instructions
```

**Implementation (lines 5225-5400):**
- Phase 1: Count instructions and calls for each function
- Phase 2: Identify candidates meeting heuristics
- Phase 3: Perform inlining with local remapping

**Local Remapping:**
When inlining, callee's locals are appended to caller with index offset to avoid conflicts.

**Example:**
```wasm
Before:
  (func $add (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add
  )
  (func
    i32.const 5
    i32.const 3
    call $add  ;; single call site, < 10 instructions
  )

After:
  (func
    i32.const 5
    i32.const 3
    i32.add  ;; inlined
  )
```

**Limitations (Lines 5288-5300):**
- MVP implementation: doesn't substitute parameters properly
- Full implementation would require:
  - Stack value tracking
  - Parameter substitution
  - Return handling via branching

---

### 12. **Code Folding and Flattening (Issue #22)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 5401-5480)

**What it does:**
Eliminates single-use temporary variables and flattens nested blocks.

**Phase 1: Analyze Local Usage**
Counts gets/sets for each local variable.

**Phase 2: Identify Single-Use Locals**
Filters locals with usage count == 1.

**Phase 3: Flatten Nested Blocks**
```wasm
// Empty blocks
block {} → remove

// Single-block nesting with type match
block { block { X } } → block { X }
```

**Phase 4: Fold Single-Use Temporaries (TODO - line 5481)**
```rust
// TODO: Phase 4: Fold single-use temporaries
// This requires tracking the expression assigned to each temporary
// and substituting it at the use site (complex for stack-based code)
```

**Example:**
```wasm
Before:
  block
    block
      i32.const 1
      local.set $tmp
      local.get $tmp
    end
  end

After:
  i32.const 1
```

---

### 13. **Loop Optimizations (Issue #23)**
**File:** `/home/user/loom/loom-core/src/lib.rs` (lines 5481-5741)

**What it does:**
Optimizes loops through loop-invariant code motion (LICM).

**Loop-Invariant Code Motion:**
Moves invariant computations outside loop to reduce iteration overhead.

**Pattern Detection (lines 5714-5741):**
- Constants: always invariant
- LocalGet: invariant if local not modified in loop
- Complex operations: conservatively non-invariant

**Limitations:**
Very conservative implementation - most operations marked as non-invariant.

**Example:**
```wasm
Before:
  loop $l
    i32.const 100  ;; invariant
    local.get $i
    i32.add
    local.tee $sum
    local.get $n
    i32.lt_u
    br_if $l
  end

After (ideally):
  i32.const 100
  loop $l
    local.get $i
    i32.add
    local.tee $sum
    local.get $n
    i32.lt_u
    br_if $l
  end
```

---

## PART 2: ISLE OPTIMIZATION RULES

**File:** `/home/user/loom/loom-shared/isle/wasm_terms.isle`

### ISLE Type System Definition (lines 1-159)

**Primitive Types:**
```
Imm32  - 32-bit immediate values
Imm64  - 64-bit immediate values
Value  - Boxed pointer to ValueData (recursive structures)
u32    - Local variable indices
```

**ValueData Enum (lines 40-155):**
Defines all representable WebAssembly operations:

**Arithmetic Operations:**
- I32Add, I32Sub, I32Mul (32-bit)
- I64Add, I64Sub, I64Mul (64-bit)
- Analogous division/remainder variants

**Bitwise Operations:**
- I32And, I32Or, I32Xor, I32Shl, I32ShrS, I32ShrU (32-bit)
- I64And, I64Or, I64Xor, I64Shl, I64ShrS, I64ShrU (64-bit)

**Comparisons:**
- I32Eq, I32Ne, I32LtS, I32LtU, I32GtS, I32GtU, I32LeS, I32LeU, I32GeS, I32GeU (32-bit)
- I64Eq, I64Ne, ... (64-bit - same pattern)

**Unary Operations:**
- I32Eqz, I32Clz, I32Ctz, I32Popcnt (32-bit)
- I64Eqz, I64Clz, I64Ctz, I64Popcnt (64-bit)

**Control Flow:**
- Block, Loop, If (with nested instruction lists)
- Br (unconditional branch)
- BrIf (conditional branch)
- Return
- Call

**Local Variables (Phase 12):**
- LocalGet(idx)
- LocalSet(idx, val)
- LocalTee(idx, val)

**Memory Operations (Phase 13):**
- I32Load(addr, offset, align)
- I32Store(addr, value, offset, align)
- I64Load, I64Store variants

---

### ISLE Constructors (lines 164-410)

**Terminology:**
- `(decl name (arg_types) return_type)` - declares constructor
- `(extern constructor name name)` - links to Rust implementation

**Constructor Categories:**

**1. Arithmetic Constructors:**
- `iconst32(Imm32) → Value`
- `iadd32, isub32, imul32(Value, Value) → Value`
- Equivalent 64-bit: `iconst64, iadd64, isub64, imul64`

**2. Bitwise Constructors:**
- `iand32, ior32, ixor32(Value, Value) → Value`
- `ishl32, ishrs32, ishru32(Value, Value) → Value`
- Equivalent 64-bit versions

**3. Comparison Constructors (Return i32: 0 or 1):**
- `ieq32, ine32, ilts32, iltu32, igts32, igtu32, iles32, ileu32, iges32, igeu32`
- Equivalent 64-bit versions

**4. Division/Remainder Constructors:**
- `idivs32, idivu32, irems32, iremu32`
- Equivalent 64-bit versions

**5. Unary Constructors:**
- `ieqz32, iclz32, ictz32, ipopcnt32(Value) → Value`
- Equivalent 64-bit versions

**6. Helper Functions (Immediate Arithmetic):**
- `imm32_add, imm32_sub, imm32_mul(Imm32, Imm32) → Imm32`
- `imm32_and, imm32_or, imm32_xor`
- `imm32_shl, imm32_shr_s, imm32_shr_u`
- Equivalent 64-bit versions

---

### ISLE Optimization Rules (lines 508-583)

**Main Rule Declaration (line 526):**
```lisp
(decl simplify (Value) Value)
```

**Default Rule (line 549):**
```lisp
(rule (simplify val) val)
```
Returns value unchanged if no specific rule matches.

**Comment Block (lines 529-545):**
Explains that specific optimization rules (constant folding, algebraic simplifications) are currently:
1. Implemented as Rust peephole optimizations in `optimize_advanced_instructions`
2. Not yet implemented in ISLE due to:
   - Complexity of pattern matching with ValueData variants
   - Difficulty with nested extractor chains
   - Rust implementation being more efficient and maintainable

**Future ISLE Rules (lines 563-582):**
Planned control flow optimizations (pending ISLE list support):
- Constant condition folding: `if (const 0) ... else B → B`
- BrIf constant folding: `br_if (const 0) → nop`; `br_if (const 1) → br`
- Empty block elimination: `block {} → nop`
- Block flattening: `block { block { X } } → block { X }`
- Dead code after unconditional br/return

---

## PART 3: COMPONENT MODEL OPTIMIZATION

**File:** `/home/user/loom/loom-core/src/component_optimizer.rs`

### Phase 1: Core Module Optimization (lines 1-207)

**ComponentStats (lines 47-82):**
Tracks optimization metrics:
- original_size, optimized_size (component bytes)
- module_count, modules_optimized
- original_module_size, optimized_module_size (before/after for core modules)

**optimize_component() (lines 93-181):**

**Step 1: Parse Component (lines 95-127)**
- Extracts all core modules from component
- Validates this is a component (not a core module)

**Step 2: Optimize Each Module (lines 135-146)**
- Applies LOOM's 12-phase pipeline to each core module
- 80-95% size reduction on module code expected
- Errors logged but don't fail entire component

**Step 3: Reconstruct Component (lines 149)**
- Rebuilds component structure with optimized modules
- Preserves all sections verbatim

**Step 4: Validate (line 152)**
- Runs wasmparser validation on result

---

### Phase 1.5: Full Section Preservation (lines 209-382)

**reconstruct_component() (lines 228-382):**

Rebuilds component section-by-section, replacing only core modules:

**Sections Preserved Verbatim:**
- CustomSection (names, producers, etc.)
- CoreTypeSection, ComponentTypeSection
- Import/Export sections
- Instance sections
- Alias sections
- Canonical sections (lift/lower)
- Start section
- Nested component sections

**Modules Replaced:**
- ModuleSection → replaced with optimized bytes

**Example:**
```
Original Component:
  [Header]
  [Type Section] ← preserved
  [Module 1] ← REPLACED with optimized version
  [Module 2] ← REPLACED with optimized version
  [Export Section] ← preserved
  [Canonical Section] ← preserved

Optimized Component:
  [Header]
  [Type Section] (same bytes)
  [Module 1 optimized]
  [Module 2 optimized]
  [Export Section] (same bytes)
  [Canonical Section] (same bytes)
```

---

### Phase 2: Component-Level Optimizations (lines 384-480)

**Infrastructure Defined (Currently Inactive)**

**apply_component_optimizations() (lines 412-429):**
Planned optimizations (not yet fully implemented):
1. Type deduplication across component
2. Unused import/export elimination
3. Canonical function optimization

**deduplicate_component_types() (lines 444-480):**
Analyzes for duplicate component type definitions (MVP - doesn't modify yet).

**ComponentAnalysis (lines 488-569):**
Analyzes component structure without modification:
- Counts modules, types, imports/exports
- Estimates optimization potential
- Used for metrics and guidance

---

## PART 4: EXPERIMENTAL/COMMENTED-OUT OPTIMIZATIONS

### TODOs Found in Code:

1. **Line 870 (parse.rs):**
   ```rust
   // TODO: Add F32Const and F64Const to Instruction enum
   ```
   Floating-point constant support not yet implemented.

2. **Line 2300 (lib.rs):**
   ```rust
   // TODO: Proper call handling requires knowing function signature
   ```
   Function signature tracking needed for better call analysis.

3. **Line 2313 (lib.rs):**
   ```rust
   // TODO: Proper handling requires knowing type signature for arguments
   ```
   Type-directed optimizations require more sophisticated tracking.

4. **Line 3640 (lib.rs - simplify_locals):**
   ```rust
   // TODO: Add proper dead store elimination with stack analysis
   ```
   Current dead store detection conservatively keeps values for stack balance.

5. **Line 5481 (lib.rs - fold_code):**
   ```rust
   // TODO: Phase 4: Fold single-use temporaries
   // This requires tracking the expression assigned to each temporary
   // and substituting it at the use site (complex for stack-based code)
   ```
   Expression substitution for single-use locals not yet implemented.

### Disabled/Incomplete Features:

1. **Full CSE Implementation (lines 4117-4135):**
   Current MVP disabled complex expression caching due to stack safety concerns.

2. **Loop Unrolling (Issue #23):**
   Not implemented - only loop-invariant code motion partially implemented.

3. **ISLE-Based Optimization Rules (wasm_terms.isle, lines 529-545):**
   Helper functions defined but rules not implemented pending better ISLE support.

4. **Phase 2 Component-Level Optimizations:**
   Infrastructure in place but optimizations marked `#[allow(dead_code)]` - deferred to follow-up.

---

## PART 5: OPTIMIZATION STATISTICS & TESTING

### Test Coverage
File: `/home/user/loom/loom-core/tests/optimization_tests.rs`

**Test Categories:**
1. Strength reduction (mul/div/rem by power of 2)
2. Bitwise operations (self-operations: x^x, x&x, x|x, x-x)
3. Algebraic identities (x+0, x*0, x*1, x/1, x%1)
4. Dead code elimination
5. Branch simplification
6. Block merging safety

**Example Tests (lines 29-105):**
- `test_strength_reduction_mul_power_of_2`: Verifies `x * 8 → x << 3`
- `test_bitwise_xor_self`: Verifies `x ^ x → 0`
- `test_algebraic_add_zero`: Verifies `x + 0 → x`

---

## PART 6: OPTIMIZATION PIPELINE EXECUTION ORDER

Complete pipeline from `/home/user/loom/loom-cli/src/main.rs` lines 237-246:

```
1. precompute()  →  Global constant propagation
2. constant_folding()  →  ISLE-based constant folding
3. eliminate_common_subexpressions()  →  CSE (MVP)
4. optimize_advanced_instructions()  →  Strength reduction, algebraic simplifications
5. simplify_branches()  →  Constant condition folding
6. eliminate_dead_code()  →  Remove unreachable code
7. merge_blocks()  →  Reduce CFG complexity
8. vacuum()  →  Final cleanup (nop removal, trivial block unwrapping)
9. simplify_locals()  →  Equivalence canonicalization (+ future dead stores)
10. (coalesce_locals not in main pipeline currently)
11. (fold_code not in main pipeline currently)
12. (optimize_loops not in main pipeline currently)
13. (inline_functions not in main pipeline currently)
```

---

## SUMMARY: CAPABILITIES & LIMITATIONS

### Strong Areas:
1. Constant folding with ISLE framework
2. Advanced instruction optimizations (strength reduction)
3. Bitwise algebraic rules (x^x→0, etc.)
4. Dead code elimination
5. Control flow simplification
6. Block structure optimization
7. Component model support

### Limitations & TODOs:
1. CSE restricted to MVP (constants only)
2. Loop optimizations incomplete (LICM only)
3. Function inlining MVP (no parameter substitution)
4. Dead store elimination incomplete
5. ISLE rules not yet used (infrastructure only)
6. Floating-point support missing
7. Component-level optimizations deferred
8. Memory operation optimizations incomplete

### Optimization Benefits Expected:
- Constant folding: 5-10% reduction
- Dead code elimination: 3-8%
- Strength reduction & algebraic: 2-5%
- Local coalescing: 10-15%
- Function inlining: 5-10%
- **Total: 25-50%+ combined reduction**

