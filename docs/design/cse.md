# Common Subexpression Elimination (CSE) Design

## Overview

Common Subexpression Elimination identifies and eliminates redundant computations by:
1. **Detecting** duplicate expressions within a basic block
2. **Caching** the first occurrence in a local variable
3. **Reusing** the cached value instead of recomputing

## Example

```wasm
;; Before
(i32.add
  (i32.mul (local.get $x) (i32.const 4))
  (i32.mul (local.get $x) (i32.const 4))  ;; Duplicate!
)

;; After
(i32.add
  (local.tee $temp (i32.mul (local.get $x) (i32.const 4)))
  (local.get $temp)
)
```

## Algorithm (Based on Binaryen LocalCSE)

### Phase 1: Scan

```
For each expression in the basic block:
1. Hash the expression (based on operation + children)
2. Check if we've seen this exact expression before
3. If YES (duplicate found):
   a. Mark first occurrence as "requested for reuse"
   b. Link duplicate to original
   c. Decrement reuse requests for duplicate's children
      (we'll reuse parent, so don't need to cache children)
```

### Phase 2: Check Effects

```
For each requested reuse:
1. Check if side effects invalidate the optimization
2. Examples that prevent CSE:
   - Memory load before/after a store (value might change)
   - Load before/after a call (memory might change)
   - Operations with side effects between duplicates
```

### Phase 3: Apply

```
For each expression with valid reuse requests:
1. Replace first occurrence with (local.tee $new_local ...)
2. Replace duplicate occurrences with (local.get $new_local)
3. Allocate new local variables as needed
```

## Expression Hashing

Hash expressions based on:
- Operation type (i32.add, i32.mul, etc.)
- Children (recursively)
- Constants (if any)

```rust
fn hash_expression(expr: &Instruction) -> u64 {
    let mut hasher = DefaultHasher::new();

    match expr {
        Instruction::I32Add => {
            hasher.write_u8(OP_I32_ADD);
            // Children hashed by context
        }
        Instruction::I32Const(val) => {
            hasher.write_u8(OP_I32_CONST);
            hasher.write_i32(*val);
        }
        Instruction::LocalGet(idx) => {
            hasher.write_u8(OP_LOCAL_GET);
            hasher.write_u32(*idx);
        }
        // ... etc
    }

    hasher.finish()
}
```

## Side Effects Analysis

**Safe to CSE** (no side effects):
- `i32.add`, `i32.sub`, `i32.mul`, `i32.and`, etc. (pure arithmetic)
- `local.get` (reading local variable)
- `i32.const`, `i64.const` (constants)

**Unsafe to CSE** (have side effects or invalidated by effects):
- `i32.load`, `i64.load` (memory read - invalidated by stores)
- `i32.store`, `i64.store` (memory write - side effect)
- `call` (function call - unknown effects)
- `local.set` (writes to local)

**Invalidation Rules**:
```
If between first and second occurrence there is:
- A store → invalidates prior loads
- A call → invalidates all loads (conservative)
- A set to a local → invalidates gets of that local
```

## WebAssembly-Specific Considerations

### Stack-Based Representation

In our representation, expressions are instructions in sequence. We need to:
1. **Identify expression boundaries** (which instructions form a complete expression)
2. **Track stack depth** to understand dependencies
3. **Handle nested expressions** properly

Example:
```wasm
;; Stack representation
(local.get $x)    ;; Push x
(i32.const 4)     ;; Push 4
(i32.mul)         ;; Pop 2, push result
(local.get $x)    ;; Push x
(i32.const 4)     ;; Push 4
(i32.mul)         ;; Pop 2, push result (DUPLICATE!)
(i32.add)         ;; Pop 2, push result
```

### Local Allocation

Need to:
1. **Count required locals** for CSE temps
2. **Extend function locals** with new variables
3. **Choose appropriate types** (i32, i64, etc.)

## Implementation Strategy

### MVP: Simple Pattern Matching

Start with simple, common patterns:

```wasm
;; Pattern 1: Duplicate binary operations with constants
(i32.mul (local.get $x) (i32.const N))
(i32.mul (local.get $x) (i32.const N))

;; Pattern 2: Duplicate local.get sequences
(local.get $a)
(local.get $b)
(i32.add)
...
(local.get $a)
(local.get $b)
(i32.add)

;; Pattern 3: Duplicate constant computations
(i32.const 10)
(i32.const 20)
(i32.mul)
...
(i32.const 10)
(i32.const 20)
(i32.mul)
```

### Algorithm Simplification

For MVP, use simplified approach:

```rust
pub fn eliminate_common_subexpressions(module: &mut Module) -> Result<()> {
    for func in &mut module.functions {
        // Phase 1: Find duplicate expression sequences
        let duplicates = find_duplicates(&func.instructions);

        // Phase 2: Check if safe to optimize (conservative check)
        let safe_duplicates = filter_safe(duplicates);

        // Phase 3: Apply optimization
        if !safe_duplicates.is_empty() {
            func.instructions = apply_cse(
                &func.instructions,
                &safe_duplicates,
                &mut func.locals
            );
        }
    }
    Ok(())
}
```

## Test Cases

### 1. Simple Duplicate Expression
```wasm
(func $test (param $x i32) (result i32)
  (local.get $x)
  (local.get $x)
  (i32.mul)     ;; x * x
  (local.get $x)
  (local.get $x)
  (i32.mul)     ;; x * x (duplicate!)
  (i32.add)
)

;; After CSE:
(func $test (param $x i32) (result i32)
  (local $temp i32)
  (local.tee $temp
    (i32.mul (local.get $x) (local.get $x))
  )
  (local.get $temp)
  (i32.add)
)
```

### 2. Multiple Occurrences
```wasm
(func $test (result i32)
  (i32.const 10)
  (i32.const 20)
  (i32.add)    ;; 30
  (i32.const 10)
  (i32.const 20)
  (i32.add)    ;; 30 (duplicate)
  (i32.const 10)
  (i32.const 20)
  (i32.add)    ;; 30 (duplicate)
  (i32.add)
  (i32.add)
)

;; After CSE + constant folding:
;; Should become just (i32.const 90)
```

### 3. Load Invalidation (Don't Optimize)
```wasm
(func $test (param $addr i32) (result i32)
  (i32.load (local.get $addr))
  (i32.const 42)
  (i32.store (local.get $addr))  ;; Invalidates prior load!
  (i32.load (local.get $addr))   ;; Different value now
  (i32.add)
)

;; Should NOT optimize - values are different
```

### 4. Safe Arithmetic Duplication
```wasm
(func $test (param $a i32) (param $b i32) (result i32)
  (i32.add (local.get $a) (local.get $b))
  (i32.add (local.get $a) (local.get $b))
  (i32.mul)
)

;; After CSE:
(func $test (param $a i32) (param $b i32) (result i32)
  (local $temp i32)
  (local.tee $temp (i32.add (local.get $a) (local.get $b)))
  (local.get $temp)
  (i32.mul)
)
```

## Integration

**Pipeline Position**: After SimplifyLocals, before final passes

```
... → SimplifyLocals → CSE → [Re-run optimizations] → Output
```

**Rationale**:
- SimplifyLocals may create more CSE opportunities by canonicalizing locals
- CSE creates new locals that may benefit from another SimplifyLocals pass
- New constant folding opportunities may arise from CSE

## Performance Impact

Based on compiler benchmarks:
- **20-30% of code** has duplicate expressions
- Particularly common in:
  - Array indexing: `base + (i * stride)` computed multiple times
  - Macro expansions creating duplicate calculations
  - Hand-written code with repeated computations
  - Loop invariant code

## Complexity

- **Time**: O(n²) worst case (comparing all pairs), O(n) average with hashing
- **Space**: O(n) for hash table
- **Implementation**: Medium complexity

## Success Metrics

- Duplicate pure expressions eliminated
- New locals created for cached values
- No incorrect optimizations (side effects respected)
- All optimized code produces valid WASM
- Code size reduction for duplicate-heavy code
- Enables further constant folding

## Future Enhancements

1. **Global CSE**: Across basic blocks (requires dominator analysis)
2. **Loop Invariant Code Motion**: Hoist computations out of loops
3. **Partial Redundancy Elimination**: Insert computations to enable CSE
4. **Better side effect analysis**: Track which loads/stores alias
