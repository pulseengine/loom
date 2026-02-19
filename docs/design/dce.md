# Dead Code Elimination (DCE) Design

## Overview

Dead Code Elimination removes unreachable and unused code that doesn't affect program output. This is one of the highest-impact optimizations, affecting 60-70% of real-world code.

## Types of Dead Code in WebAssembly

### 1. **Unreachable Code**
Code that follows control flow operations that never return:
```wasm
(func $example
  (return (i32.const 42))
  ;; Everything below is unreachable
  (i32.const 10)
  (drop)
)
```

### 2. **Unused Values**
Instructions whose results are never used:
```wasm
(func $example
  (i32.const 10)  ;; Computed but never used
  (i32.const 20)  ;; Computed but never used
  (i32.const 30)  ;; This one is used
)
```

### 3. **Dead Branches**
Blocks/branches with no effect:
```wasm
(func $example
  (if (i32.const 0)  ;; Always false
    (then
      (i32.const 100)  ;; Never executed
      (drop)
    )
  )
)
```

## DCE Strategy

### Phase 1: Control Flow Reachability Analysis

Track which instructions are reachable based on control flow:

1. **Start from function entry** - mark as reachable
2. **Follow control flow**:
   - All instructions before terminator → reachable
   - After `return`, `br`, `unreachable` → unreachable
   - After unconditional branch → unreachable
3. **Mark unreachable blocks**

**Terminators** (make following code unreachable):
- `Return` - exits function
- `Br` (unconditional) - jumps away
- `Unreachable` - traps
- `BrTable` (if all targets same) - deterministic jump

### Phase 2: Liveness Analysis (Value Usage)

Track which values are actually used:

1. **Mark instructions with side effects** as live:
   - Memory stores (`I32Store`, `I64Store`)
   - Calls (`Call`, `CallIndirect`)
   - `Return` (uses return values)
   - Control flow with effects

2. **Backward propagation**:
   - If instruction is used → its operands are live
   - If instruction has no uses + no side effects → dead

3. **Special cases**:
   - `LocalSet` without corresponding `LocalGet` → dead
   - `LocalTee` returns value, so check if return is used
   - Pure arithmetic with no uses → dead

### Phase 3: Elimination

1. **Remove unreachable instructions**
2. **Remove unused pure instructions**
3. **Simplify blocks with only dead code**
4. **Clean up empty blocks**

## Implementation Plan

### Data Structures

```rust
/// Track reachability for each instruction
struct ReachabilityInfo {
    reachable: HashSet<InstructionId>,
}

/// Track which instructions are used
struct LivenessInfo {
    live: HashSet<InstructionId>,
    used_locals: HashSet<u32>,
}
```

### Algorithm

```rust
fn eliminate_dead_code(instructions: &mut Vec<Instruction>) {
    // Phase 1: Mark reachable code
    let reachability = analyze_reachability(instructions);

    // Phase 2: Mark live values
    let liveness = analyze_liveness(instructions, &reachability);

    // Phase 3: Remove dead code
    remove_dead_instructions(instructions, &reachability, &liveness);
}
```

### Reachability Analysis

```rust
fn analyze_reachability(instrs: &[Instruction]) -> HashSet<usize> {
    let mut reachable = HashSet::new();
    let mut is_reachable = true;

    for (idx, instr) in instrs.enumerate() {
        if is_reachable {
            reachable.insert(idx);
        }

        match instr {
            Instruction::Return => is_reachable = false,
            Instruction::Br(_) => is_reachable = false,
            Instruction::Unreachable => is_reachable = false,
            Instruction::Block { body, .. } => {
                // Recursively analyze block body
                let block_reachable = analyze_reachability(body);
                // ... merge results
            }
            Instruction::If { then_body, else_body, .. } => {
                // Both branches might be reachable
                analyze_reachability(then_body);
                analyze_reachability(else_body);
            }
            Instruction::Loop { body, .. } => {
                // Loop body is reachable
                analyze_reachability(body);
            }
            _ => {}
        }
    }

    reachable
}
```

### Liveness Analysis

```rust
fn analyze_liveness(instrs: &[Instruction]) -> HashSet<usize> {
    let mut live = HashSet::new();
    let mut used_values = HashSet::new();

    // Backward pass: mark live instructions
    for (idx, instr) in instrs.iter().enumerate().rev() {
        let is_live = match instr {
            // Side effects are always live
            Instruction::I32Store { .. } => true,
            Instruction::I64Store { .. } => true,
            Instruction::Call(_) => true,
            Instruction::CallIndirect { .. } => true,
            Instruction::Return => true,

            // Control flow is live
            Instruction::If { .. } => true,
            Instruction::Loop { .. } => true,
            Instruction::Block { .. } => true,
            Instruction::Br(_) => true,
            Instruction::BrIf(_) => true,
            Instruction::BrTable { .. } => true,

            // LocalGet/LocalSet need special handling
            Instruction::LocalGet(idx) => used_values.contains(idx),
            Instruction::LocalSet(idx) => {
                // Live if the local is used later
                used_values.contains(idx)
            }

            // Pure operations - live only if result is used
            _ => used_values.contains(&idx),
        };

        if is_live {
            live.insert(idx);
            // Mark operands as used
            mark_operands_used(instr, &mut used_values);
        }
    }

    live
}
```

## Testing Strategy

### Test Cases

1. **Unreachable after return**
```wasm
(func $test (result i32)
  (return (i32.const 42))
  (i32.const 99)  ;; Dead - should be removed
)
```

2. **Unused computation**
```wasm
(func $test (result i32)
  (i32.const 10)
  (i32.const 20)
  (i32.add)  ;; Dead - result not used
  (i32.const 42)
)
```

3. **Dead local**
```wasm
(func $test (result i32)
  (local $unused i32)
  (local.set $unused (i32.const 100))  ;; Dead - never read
  (i32.const 42)
)
```

4. **Dead branch**
```wasm
(func $test (result i32)
  (block $exit (result i32)
    (br $exit (i32.const 42))
    (i32.const 99)  ;; Dead - after unconditional branch
  )
)
```

5. **Side effects preserved**
```wasm
(func $test (param $addr i32)
  (i32.const 10)
  (i32.const 20)
  (i32.add)  ;; Dead
  (local.get $addr)
  (i32.const 42)
  (i32.store)  ;; Preserved - has side effect
)
```

## Integration with LOOM

### Optimization Pass

DCE should run:
1. **After constant propagation** - exposes more dead code
2. **After inlining** - creates unreachable code
3. **Before code generation** - final cleanup

### ISLE Integration

DCE will work on the `Instruction` level (not ISLE terms) because:
- Operates on control flow structure
- Needs to analyze entire function body
- Works on post-optimization instructions

Can add ISLE-level DCE later for expression-level optimization.

## Performance Considerations

- **O(n) reachability** - single forward pass
- **O(n) liveness** - single backward pass
- **O(n) elimination** - filter pass
- **Total: O(n)** - very efficient

## Future Enhancements

1. **Function-level DCE** - remove unused functions
2. **Global DCE** - remove unused globals
3. **Aggressive DCE** - more sophisticated value tracking
4. **Inter-procedural DCE** - cross-function analysis

## References

- Binaryen DeadCodeElimination.cpp
- LLVM DCE pass
- "Engineering a Compiler" (Cooper & Torczon), Chapter 10
