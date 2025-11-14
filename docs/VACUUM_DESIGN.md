# Vacuum Cleanup Pass Design

## Overview

The Vacuum pass is a final cleanup optimization that removes artifacts left by other optimizations, simplifies trivial patterns, and normalizes code structure. It's a "polishing" pass that ensures clean, minimal output.

## Why Vacuum Matters

- **100% of code benefits** - every optimization creates cleanup opportunities
- Removes nop instructions created by other passes
- Unwraps trivial blocks (single-instruction wrappers)
- Simplifies degenerate control flow
- Very cheap to run (O(n) single pass)
- High value-per-cost ratio

## Types of Cleanup

### 1. **Nop Removal**

Remove explicit nop instructions:

```wasm
;; Before
(func $test (result i32)
  (nop)
  (i32.const 42)
  (nop)
)

;; After
(func $test (result i32)
  (i32.const 42)
)
```

**Already handled by branch simplification**, but Vacuum catches any remaining nops.

### 2. **Trivial Block Unwrapping**

Remove blocks that just wrap a single expression:

```wasm
;; Before
(func $test (result i32)
  (block (result i32)
    (i32.const 42)
  )
)

;; After
(func $test (result i32)
  (i32.const 42)
)
```

**Most Common Pattern** - created by block merging and branch simplification.

### 3. **Empty Block Removal**

```wasm
;; Before
(func $test
  (block)
)

;; After
(func $test)
```

### 4. **Trivial If Simplification**

**Constant Conditions (already handled by branch simplification):**
```wasm
;; Before
(if (i32.const 1)
  (then (i32.const 42))
  (else (i32.const 99))
)

;; After
(i32.const 42)
```

**Empty Else Branch:**
```wasm
;; Before
(if (local.get $x)
  (then (call $foo))
  (else)
)

;; After
(if (local.get $x)
  (then (call $foo))
)
```

**Both Branches Empty:**
```wasm
;; Before
(if (local.get $x)
  (then)
  (else)
)

;; After
(drop (local.get $x))
```

### 5. **Trivial Loop Unwrapping**

```wasm
;; Before
(loop (result i32)
  (i32.const 42)
)

;; After
(i32.const 42)
```

**Only if loop body doesn't branch back** (no recursive execution).

### 6. **Identity Operation Removal**

**Local Set-Get Identity:**
```wasm
;; Before
(local.set $x (local.get $x))

;; After
(drop (local.get $x))
```

**Or remove entirely if** `local.get` is pure.

### 7. **Unreachable Code Removal**

**After Unreachable:**
```wasm
;; Before
(unreachable)
(i32.const 42)
(drop)

;; After
(unreachable)
```

**Already handled by DCE**, but Vacuum can catch stragglers.

## Implementation Strategy

### Phase 1: Simple Cleanup (Current Focus)

Focus on the highest-value, lowest-complexity patterns:

1. **Remove nops** (already done, but verify)
2. **Unwrap trivial blocks**
3. **Remove empty blocks**
4. **Simplify empty if branches**
5. **Remove identity local operations**

**Algorithm:**
```rust
fn vacuum_cleanup(instructions: &[Instruction]) -> Vec<Instruction> {
    let mut result = Vec::new();

    for instr in instructions {
        match instr {
            // Skip nops
            Instruction::Nop => continue,

            // Unwrap trivial blocks
            Instruction::Block { block_type, body } => {
                let cleaned_body = vacuum_cleanup(body);

                // If block has only one instruction, unwrap it
                if is_trivial_block(&cleaned_body, block_type) {
                    result.extend(cleaned_body);
                } else {
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: cleaned_body,
                    });
                }
            }

            // Clean other control flow recursively
            Instruction::Loop { block_type, body } => {
                let cleaned_body = vacuum_cleanup(body);

                if !cleaned_body.is_empty() {
                    result.push(Instruction::Loop {
                        block_type: block_type.clone(),
                        body: cleaned_body,
                    });
                }
            }

            Instruction::If { block_type, then_body, else_body } => {
                let cleaned_then = vacuum_cleanup(then_body);
                let cleaned_else = vacuum_cleanup(else_body);

                // Simplify based on branch emptiness
                if cleaned_then.is_empty() && cleaned_else.is_empty() {
                    // Both empty - just drop the condition
                    result.push(Instruction::Drop);
                } else if cleaned_else.is_empty() {
                    // Empty else - remove it
                    result.push(Instruction::If {
                        block_type: block_type.clone(),
                        then_body: cleaned_then,
                        else_body: vec![],
                    });
                } else {
                    result.push(Instruction::If {
                        block_type: block_type.clone(),
                        then_body: cleaned_then,
                        else_body: cleaned_else,
                    });
                }
            }

            _ => result.push(instr.clone()),
        }
    }

    result
}

fn is_trivial_block(body: &[Instruction], block_type: &BlockType) -> bool {
    // Block is trivial if it has exactly one non-End instruction
    // and that instruction produces the expected type
    if body.is_empty() {
        return true; // Empty block is trivial
    }

    // Single instruction that matches block type
    if body.len() == 1 {
        match block_type {
            BlockType::Empty => true,
            BlockType::Value(_) => {
                // Check if instruction produces a value
                instruction_produces_value(&body[0])
            }
            _ => false,
        }
    } else {
        false
    }
}
```

### Phase 2: Advanced Cleanup (Future)

1. **Constant folding cleanup** - remove trivial arithmetic
2. **Drop optimization** - combine multiple drops
3. **Local coalescing hints** - mark redundant locals for later removal
4. **Dead store elimination** - detect and remove unused local.set

## Integration with Optimization Pipeline

Vacuum should run **last** in the pipeline:

```
Constant Folding → Branch Simplification → DCE → Block Merging → Vacuum
```

**Why last?**
- Every optimization can create cleanup opportunities
- Vacuum polishes the final output
- Cheap enough to run at the end

**Multiple passes?**
- One pass usually sufficient
- Could run iteratively until no changes (fixpoint)
- For Phase 1, single pass is enough

## Trivial Block Detection Rules

**Safe to unwrap when:**

1. **Single instruction in body:**
   ```wasm
   (block (result i32)
     (i32.const 42)
   )
   ```
   → Can unwrap if instruction produces correct type

2. **Empty body:**
   ```wasm
   (block)
   ```
   → Can remove entirely

3. **Multiple instructions with no branches:**
   ```wasm
   (block
     (i32.const 10)
     (drop)
   )
   ```
   → Can unwrap if no branches target this block

**NOT safe to unwrap when:**

1. **Block is branch target** (has label that's referenced)
2. **Type mismatch** between block type and body result
3. **Multiple instructions** where block enforces execution order

## Type Safety

Vacuum must preserve WebAssembly type correctness:

```wasm
;; INCORRECT unwrapping:
(block (result i32)
  (call $void_func)
  (i32.const 42)
)
→ Cannot unwrap - multiple instructions

;; CORRECT unwrapping:
(block (result i32)
  (i32.const 42)
)
→ Can unwrap - single instruction, matching type
```

## Edge Cases

### 1. **Blocks with Branches**

```wasm
(block $label
  (br $label)
)
```

**Cannot unwrap** - $label is a branch target.

**Solution:** Phase 1 only unwraps blocks with no branches. Phase 2 would track branch targets.

### 2. **Empty If with Side Effects**

```wasm
(if (call $has_side_effect)
  (then)
  (else)
)
```

**Cannot remove** - condition has side effects.

**Solution:** Check for side effects before dropping condition.

### 3. **Nested Trivial Blocks**

```wasm
(block (result i32)
  (block (result i32)
    (i32.const 42)
  )
)
```

**Should fully unwrap** to `(i32.const 42)`.

**Solution:** Recursive processing handles this naturally.

### 4. **Loop with Single Iteration**

```wasm
(loop (result i32)
  (i32.const 42)
)
```

**Can unwrap** - no backward branches.

**Solution:** Only unwrap loops with no `br` to loop label.

## Testing Strategy

### Test Cases

1. **Remove nops**
```wasm
(func $test (result i32)
  (nop)
  (i32.const 42)
)
```
Should remove nop.

2. **Unwrap trivial block**
```wasm
(func $test (result i32)
  (block (result i32)
    (i32.const 42)
  )
)
```
Should unwrap to `(i32.const 42)`.

3. **Unwrap nested trivial blocks**
```wasm
(func $test (result i32)
  (block (result i32)
    (block (result i32)
      (i32.const 42)
    )
  )
)
```
Should fully unwrap.

4. **Remove empty if branches**
```wasm
(func $test (param $x i32)
  (if (local.get $x)
    (then)
    (else)
  )
)
```
Should become `(drop (local.get $x))`.

5. **Preserve blocks with multiple instructions**
```wasm
(func $test (result i32)
  (block (result i32)
    (i32.const 10)
    (i32.const 32)
    (i32.add)
  )
)
```
Should keep block (multiple instructions).

6. **After block merging artifacts**

Test on output from block merging to ensure cleanup.

## Performance Expectations

- **Time Complexity**: O(n) - single pass
- **Space Complexity**: O(n) - rebuilding instruction tree
- **Expected Reduction**: 5-10% instruction count reduction
- **Impact**: Cleaner output, easier debugging, smaller binaries

## Implementation Notes

### Simple vs. Complex Blocks

**Simple block** (can unwrap):
- Empty body OR
- Single instruction that matches type

**Complex block** (keep):
- Multiple instructions
- Branch target (future: requires CFG)
- Type mismatch

### Recursive Processing

Process from **innermost to outermost**:
1. Clean inner blocks first
2. Then check if outer block is now trivial
3. Unwrap if trivial

This handles nested trivial blocks correctly.

## Future Enhancements

1. **CFG-based analysis** - detect unused labels
2. **Side effect analysis** - more aggressive cleanup
3. **Drop merging** - combine adjacent drops
4. **Local identity removal** - full analysis
5. **Fixpoint iteration** - run until no changes

## References

- Binaryen Vacuum.cpp
- LLVM InstCombine pass (similar spirit)
- WebAssembly structured control flow spec
