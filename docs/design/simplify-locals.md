# SimplifyLocals / CoalesceLocals Design

## Overview

This pass optimizes local variable usage through three main techniques:
1. **Redundant Get/Set Elimination**: Remove unnecessary local.set followed by local.get
2. **Equivalent Locals**: Track and canonicalize equivalent locals (local.set from local.get)
3. **Local Coalescing**: Merge locals with non-overlapping lifetimes (like register allocation)

## Phase 1: SimplifyLocals (This Implementation)

### Key Optimizations

#### 1. Redundant Get/Set Elimination

**Pattern**: `local.set $x (local.get $y)` followed by `local.get $x`
**Optimization**: Replace with just `local.get $y`, remove the set

Example:
```wasm
;; Before
local.get $0
local.set $1
local.get $1  ;; redundant
i32.add

;; After
local.get $0
local.get $0  ;; direct use
i32.add
```

#### 2. Equivalent Local Tracking

**Pattern**: `local.set $x (local.get $y)` creates equivalence $x ≡ $y
**Optimization**: All subsequent `local.get $x` can use `local.get $y` instead

Example:
```wasm
;; Before
local.get $0
local.set $1    ;; $1 ≡ $0
local.get $1    ;; can use $0
local.get $1    ;; can use $0

;; After
local.get $0
nop             ;; set removed
local.get $0
local.get $0
```

#### 3. Dead Store Elimination

**Pattern**: `local.set $x (...)` with no subsequent `local.get $x`
**Optimization**: Remove the set (keep value if it has side effects)

Example:
```wasm
;; Before
i32.const 42
local.set $0    ;; never read
i32.const 10

;; After
i32.const 42
drop            ;; preserve stack depth
i32.const 10
```

#### 4. Tee Simplification

**Pattern**: `local.tee $x (local.get $y)` where $x is never read
**Optimization**: Remove tee, just keep the get

Example:
```wasm
;; Before
local.get $0
local.tee $1    ;; $1 never used
i32.add

;; After
local.get $0
i32.add
```

### Algorithm

```
1. For each function:
   a. Build local usage information (gets and sets)
   b. Track equivalences (local.set from local.get creates equivalence)
   c. For each local.set:
      - Check if it's a copy (set from get)
      - Check if the local is never read after this point
      - Mark for removal if redundant
   d. For each local.get:
      - Canonicalize to equivalent local if available
   e. Remove dead stores and simplify tees

2. Iterate until fixed point (no more changes)
```

### Implementation Strategy

```rust
// Main data structures
struct LocalUsage {
    gets: Vec<usize>,  // positions of local.get
    sets: Vec<usize>,  // positions of local.set
}

struct EquivalenceSet {
    // Maps local index to its equivalent local
    equivalents: HashMap<u32, u32>,
}

pub fn simplify_locals(module: &mut Module) -> Result<()> {
    for func in &mut module.functions {
        let mut changed = true;
        while changed {
            changed = false;

            // 1. Analyze local usage
            let usage = analyze_local_usage(&func.instructions);

            // 2. Build equivalence sets
            let equivalences = build_equivalences(&func.instructions);

            // 3. Remove redundant copies
            changed |= remove_redundant_copies(
                &mut func.instructions,
                &usage,
                &equivalences
            );

            // 4. Canonicalize gets
            changed |= canonicalize_gets(
                &mut func.instructions,
                &equivalences
            );

            // 5. Remove dead stores
            changed |= remove_dead_stores(
                &mut func.instructions,
                &usage
            );
        }
    }
    Ok(())
}
```

### Safety Considerations

1. **Side Effects**: When removing a local.set, must keep value if it has side effects
2. **Control Flow**: Equivalences are invalidated at control flow merge points
3. **Type Safety**: Ensure replacement maintains correct types
4. **Stack Depth**: Removed sets may need `drop` to maintain stack balance

### Test Cases

1. **Redundant Copy**:
   ```wasm
   (local.get $0)
   (local.set $1)
   (local.get $1)  ;; Should become (local.get $0)
   ```

2. **Dead Store**:
   ```wasm
   (i32.const 42)
   (local.set $0)  ;; Never read, should become (drop)
   (i32.const 10)
   ```

3. **Equivalence Chain**:
   ```wasm
   (local.get $0)
   (local.set $1)  ;; $1 ≡ $0
   (local.get $1)
   (local.set $2)  ;; $2 ≡ $1 ≡ $0
   (local.get $2)  ;; Should become (local.get $0)
   ```

4. **Tee Optimization**:
   ```wasm
   (local.get $0)
   (local.tee $1)  ;; $1 never used
   ;; Should become just (local.get $0)
   ```

5. **Control Flow Invalidation**:
   ```wasm
   (local.get $0)
   (local.set $1)  ;; $1 ≡ $0
   (if (result i32)
     (then (local.set $1 (i32.const 42)))  ;; Invalidates $1 ≡ $0
     (else (nop))
   )
   (local.get $1)  ;; Cannot optimize
   ```

6. **Nested Blocks**:
   ```wasm
   (block
     (local.get $0)
     (local.set $1)
     (block
       (local.get $1)  ;; Should use $0
     )
   )
   ```

## Phase 2: CoalesceLocals (Future Work)

### Liveness Analysis

Track where each local is "live" (its value may be used later):
- **Live-in**: locals live at block entry
- **Live-out**: locals live at block exit
- **Interference**: two locals interfere if their live ranges overlap

### Coalescing Algorithm

```
1. Build liveness information for all locals
2. Build interference graph:
   - Nodes = locals
   - Edge = two locals interfere (overlapping lifetimes)
3. Graph coloring / greedy allocation:
   - Assign same index to non-interfering locals
   - Prioritize locals with many copies between them
4. Rewrite all local indices
```

### Example

```wasm
;; Before (4 locals)
(local $0 i32)
(local $1 i32)
(local $2 i32)
(local $3 i32)

(local.set $0 (i32.const 1))
(local.set $1 (i32.const 2))
(i32.add (local.get $0) (local.get $1))  ;; $0 and $1 dead after this
(local.set $2 (i32.const 3))
(local.set $3 (i32.const 4))
(i32.add (local.get $2) (local.get $3))

;; After (2 locals) - $0≡$2, $1≡$3 (non-overlapping)
(local $0 i32)
(local $1 i32)

(local.set $0 (i32.const 1))
(local.set $1 (i32.const 2))
(i32.add (local.get $0) (local.get $1))
(local.set $0 (i32.const 3))  ;; reuse $0
(local.set $1 (i32.const 4))  ;; reuse $1
(i32.add (local.get $0) (local.get $1))
```

## Integration

**Pipeline Position**: After Vacuum cleanup
```
Constant Folding → Branch Simplification → DCE → Block Merging → Vacuum → SimplifyLocals
```

**Rationale**:
- Other optimizations may create redundant locals
- SimplifyLocals cleans them up
- Can enable further DCE in next iteration

## Performance Impact

Based on Binaryen benchmarks:
- **50-60% of code** has redundant local operations
- **Memory savings** from reduced stack frame size
- **Better register allocation** in VMs (fewer locals = more efficient)

## Complexity

- **Time**: O(n × i) where n = instructions, i = iterations (usually 1-3)
- **Space**: O(locals × instructions) for usage tracking
- **Implementation**: Medium complexity (simpler than full coalescing)

## References

- Binaryen SimplifyLocals.cpp: Sinking, teeing, block return values
- Binaryen CoalesceLocals.cpp: Liveness-based coalescing
- wasm-opt passes: `simplify-locals`, `coalesce-locals`, `merge-locals`
