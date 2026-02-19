# Global Constant Propagation (Precompute) Design

## Status

**Phase 1: Infrastructure Complete ✅**
- Global metadata capture implemented
- Global initializer parsing added
- Global initializer encoding added
- Constant value tracking structure in place

**Blocked On**: GlobalGet/GlobalSet instruction support
- Our current Instruction enum doesn't include global.get/global.set
- Current benchmarks don't use global variables
- Full implementation requires adding these instructions first

## Overview

This pass will extend constant propagation beyond expression-level folding to:
1. **Global variable constants**: Track immutable globals and propagate their values
2. **Inter-block propagation**: Propagate constants across basic blocks within functions
3. **Local dataflow**: Track local.set → local.get constant flows

## Current State vs Target

### Current (LOOM)
- ✅ Expression-level constant folding (i32.add (i32.const 1) (i32.const 2) → i32.const 3)
- ✅ Algebraic simplifications (i32.mul x 0 → i32.const 0)
- ❌ No inter-block propagation
- ❌ No global constant tracking
- ❌ No local dataflow analysis

### Target (Precompute)
- ✅ All existing optimizations
- ✅ Global immutable constant propagation
- ✅ Local constant propagation across blocks
- ✅ Dataflow-based constant tracking

## Phase 1: Global Variable Constants (This Implementation)

### Key Concept

WebAssembly globals can be:
- **Immutable** (`global $x (i32) (i32.const 42)`) - value never changes
- **Mutable** (`global (mut i32) (i32.const 0)`) - value can change

For immutable globals with constant initializers, we can replace all `global.get` with the constant value.

### Example

```wasm
;; Before
(global $PI (f64) (f64.const 3.14159))
(global $DEBUG (i32) (i32.const 0))
(global $counter (mut i32) (i32.const 0))  ;; mutable!

(func $calculate (result f64)
  (global.get $PI)     ;; Can be replaced with 3.14159
  (f64.const 2.0)
  (f64.mul)
)

(func $check (result i32)
  (global.get $DEBUG)  ;; Can be replaced with 0
  (if (result i32)
    (then (i32.const 100))
    (else (i32.const 200))
  )
)

(func $increment
  (global.get $counter)  ;; CANNOT optimize - mutable
  (i32.const 1)
  (i32.add)
  (global.set $counter)
)

;; After
(global $PI (f64) (f64.const 3.14159))
(global $DEBUG (i32) (i32.const 0))
(global $counter (mut i32) (i32.const 0))

(func $calculate (result f64)
  (f64.const 3.14159)   ;; Propagated!
  (f64.const 2.0)
  (f64.mul)
)

(func $check (result i32)
  (i32.const 0)         ;; Propagated!
  (if (result i32)
    (then (i32.const 100))
    (else (i32.const 200))
  )
)

;; Note: $check can be further optimized by branch simplification to just return 200
```

### Algorithm

```rust
1. Analyze Module Globals:
   For each global:
   - Check if immutable (!global.mutable)
   - Check if initializer is constant expression
   - If both true: record global_index → constant_value

2. Transform Functions:
   For each instruction:
   - If GlobalGet(idx):
     - If idx in constant_globals:
       - Replace with constant instruction
       - Mark as changed

3. Iterate until fixed point:
   - Run global propagation
   - Run existing constant folding
   - Run branch simplification
   - Run DCE
   - Repeat if anything changed
```

### Safety Considerations

1. **Mutability Check**: ONLY optimize immutable globals
2. **Constant Initializer**: ONLY optimize globals with constant init expressions
3. **Type Safety**: Ensure replacement constant matches expected type
4. **No Side Effects**: global.get has no side effects, safe to replace

### Data Structures

```rust
struct GlobalConstants {
    // Maps global index to its constant value (if immutable + constant init)
    constants: HashMap<u32, ConstantValue>,
}

enum ConstantValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}
```

## Phase 2: Local Dataflow (Future Work)

### Concept

Track `local.set` → `local.get` constant flows within functions.

Example:
```wasm
(local $x i32)
(local.set $x (i32.const 42))
...no other sets to $x...
(local.get $x)  ;; Can be replaced with (i32.const 42)
```

### Algorithm

1. **Build Local Graph**: Track all sets and gets for each local
2. **Dataflow Analysis**: For each local.get, find reaching definitions
3. **Constant Check**: If all reaching defs are same constant, propagate
4. **Replace**: Substitute local.get with constant

### Complexity

More complex than globals because:
- Locals can have multiple sets
- Need reachability analysis
- Control flow merges require phi-like logic

## Phase 3: Inter-Block Propagation (Future Work)

### Concept

Propagate constants across basic block boundaries.

Example:
```wasm
(block $outer
  (local.set $x (i32.const 10))
  (block $inner
    (local.get $x)  ;; Can infer $x = 10 from outer block
  )
)
```

### Algorithm

1. **Build CFG**: Represent control flow as graph of basic blocks
2. **Dataflow Analysis**: Compute constant state at each block entry/exit
3. **Meet Operation**: When paths merge, only propagate if all paths agree
4. **Worklist Algorithm**: Iterate until fixed point

## Integration with Existing Passes

**Pipeline Position**: After existing constant folding, before branch simplification

```
Current: Constant Folding → Branch Simplification → DCE → Block Merging → Vacuum → SimplifyLocals

With Precompute:
Constant Folding → Precompute (Globals) → Branch Simplification → DCE →
Block Merging → Vacuum → SimplifyLocals
```

**Rationale**:
- Precompute enables more branch simplification opportunities
- Branch simplification + DCE can remove code unlocked by precompute
- Creates virtuous cycle when iterated

## Performance Impact

Based on Binaryen benchmarks:
- **30-40% impact** on codebases with many module-level constants
- Particularly effective for:
  - Configuration constants
  - Debug flags
  - Math constants (PI, E, etc.)
  - Enum values
  - Feature flags

## Test Cases

### 1. Immutable Global Propagation
```wasm
(global $MAX (i32) (i32.const 100))
(func $test (result i32)
  (global.get $MAX)
)
;; Should become: (i32.const 100)
```

### 2. Mutable Global NOT Propagated
```wasm
(global $count (mut i32) (i32.const 0))
(func $test (result i32)
  (global.get $count)  ;; Cannot optimize
)
```

### 3. Multiple Uses
```wasm
(global $X (i32) (i32.const 42))
(func $test (result i32)
  (global.get $X)
  (global.get $X)
  (i32.add)
)
;; Should become: (i32.const 84) after folding
```

### 4. Type Preservation
```wasm
(global $F (f64) (f64.const 1.5))
(func $test (result f64)
  (global.get $F)
)
;; Should become: (f64.const 1.5)
```

### 5. Enables Branch Simplification
```wasm
(global $DEBUG (i32) (i32.const 0))
(func $check (result i32)
  (if (result i32)
    (global.get $DEBUG)
    (then (i32.const 1))
    (else (i32.const 2))
  )
)
;; After precompute: (if (result i32) (i32.const 0) ...)
;; After branch simplification: (i32.const 2)
```

### 6. No Constant Initializer
```wasm
(import "env" "value" (global $imported i32))
(func $test (result i32)
  (global.get $imported)  ;; Cannot optimize - no constant init
)
```

## Implementation Strategy

```rust
// Phase 1: Analyze globals
pub fn analyze_global_constants(module: &Module) -> GlobalConstants {
    let mut constants = HashMap::new();

    for (idx, global) in module.globals.iter().enumerate() {
        // Only immutable globals with constant values
        if !global.mutable && is_constant_init(&global.init) {
            if let Some(value) = extract_constant(&global.init) {
                constants.insert(idx as u32, value);
            }
        }
    }

    GlobalConstants { constants }
}

// Phase 2: Transform instructions
pub fn propagate_global_constants(module: &mut Module) -> Result<()> {
    let constants = analyze_global_constants(module);

    for func in &mut module.functions {
        func.instructions = propagate_in_instructions(
            &func.instructions,
            &constants
        );
    }

    Ok(())
}

fn propagate_in_instructions(
    instructions: &[Instruction],
    constants: &GlobalConstants,
) -> Vec<Instruction> {
    instructions.iter().map(|instr| {
        match instr {
            Instruction::GlobalGet(idx) => {
                if let Some(value) = constants.get(idx) {
                    value_to_instruction(value)
                } else {
                    instr.clone()
                }
            }
            // Recurse into control flow
            _ => recurse(instr, constants)
        }
    }).collect()
}
```

## References

- Binaryen Precompute.cpp: Global and local constant propagation
- wasm-opt `--precompute` pass
- Standard compiler dataflow analysis techniques

## Success Metrics

- All immutable global constants propagated
- Enables additional branch simplification
- Maintains 100% test pass rate
- All benchmarks produce valid WASM
