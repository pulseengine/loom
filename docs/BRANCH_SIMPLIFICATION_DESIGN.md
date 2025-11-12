# Branch Simplification Design

## Overview

Branch simplification optimizes control flow by removing redundant branches, folding constant conditions, and simplifying branch patterns. This is a high-impact optimization affecting 30-40% of real-world WebAssembly code.

## Types of Branch Optimizations

### 1. **Constant Branch Folding**

Replace `br_if` with constant conditions:

```wasm
;; Before: Always taken
(br_if $label (i32.const 1))  ;; Condition always true

;; After: Unconditional branch
(br $label)
```

```wasm
;; Before: Never taken
(br_if $label (i32.const 0))  ;; Condition always false

;; After: Remove entirely
;; (removed)
```

### 2. **Redundant Branch Removal**

Remove branches that jump to where execution would go anyway:

```wasm
;; Before: Branch to block end
(block $exit
  (br $exit)  ;; Jumps to end of block (redundant)
)

;; After: Remove branch
(block $exit
  ;; (removed)
)
```

### 3. **If-Else Simplification**

**Identical Arms**:
```wasm
;; Before: Both branches identical
(if (local.get $x)
  (then (i32.const 42))
  (else (i32.const 42))
)

;; After: Remove condition (if no side effects)
(drop (local.get $x))
(i32.const 42)
```

**EqZ Flipping**:
```wasm
;; Before: Condition is eqz
(if (i32.eqz (local.get $x))
  (then (i32.const 1))
  (else (i32.const 2))
)

;; After: Flip branches and remove eqz
(if (local.get $x)
  (then (i32.const 2))
  (else (i32.const 1))
)
```

### 4. **Branch Table Simplification**

**Single Target**:
```wasm
;; Before: All targets same
(br_table $label $label $label (local.get $x))

;; After: Unconditional branch
(drop (local.get $x))
(br $label)
```

**Constant Index**:
```wasm
;; Before: Index is constant
(br_table $l0 $l1 $l2 (i32.const 1))

;; After: Direct branch
(br $l1)
```

## Implementation Strategy

### Phase 1: Constant Folding (Current Focus)

1. **br_if with constant condition**
   - `(br_if $label (i32.const 1))` → `(br $label)`
   - `(br_if $label (i32.const 0))` → remove

2. **If with constant condition**
   - `(if (i32.const 1) (then ...) (else ...))` → `then` branch
   - `(if (i32.const 0) (then ...) (else ...))` → `else` branch

3. **If with identical arms**
   - Drop condition (if no side effects) and keep one arm

### Phase 2: Structural Simplification (Future)

1. **Redundant branch removal** - detect branches to natural flow
2. **Branch merging** - combine adjacent br_if to same target
3. **EqZ optimization** - flip if/else to remove eqz

### Phase 3: Advanced Optimizations (Future)

1. **Branch table optimization** - simplify switch patterns
2. **Loop specialization** - optimize loop breaks
3. **Nested condition flattening** - combine nested ifs

## Algorithm

### Constant Branch Folding

```rust
fn simplify_branches(instructions: &[Instruction]) -> Vec<Instruction> {
    let mut result = Vec::new();

    for instr in instructions {
        let simplified = match instr {
            // br_if with constant condition
            Instruction::BrIf(label) if has_const_condition() => {
                match get_const_value() {
                    Some(0) => None,  // Never taken - remove
                    Some(_) => Some(Instruction::Br(label)),  // Always taken
                    None => Some(instr.clone()),  // Keep original
                }
            }

            // If with constant condition
            Instruction::If { block_type, then_body, else_body }
                if has_const_condition() => {
                match get_const_value() {
                    Some(0) => simplify_branches(else_body),  // Take else
                    Some(_) => simplify_branches(then_body),  // Take then
                    None => {
                        // Check for identical arms
                        if arms_identical(then_body, else_body) {
                            // Drop condition and keep one arm
                            drop_condition();
                            simplify_branches(then_body)
                        } else {
                            // Recurse into both branches
                            simplify_if(block_type, then_body, else_body)
                        }
                    }
                }
            }

            // Recursively process nested structures
            Instruction::Block { block_type, body } => {
                Instruction::Block {
                    block_type,
                    body: simplify_branches(body),
                }
            }

            Instruction::Loop { block_type, body } => {
                Instruction::Loop {
                    block_type,
                    body: simplify_branches(body),
                }
            }

            _ => instr.clone(),
        };

        if let Some(s) = simplified {
            result.push(s);
        }
    }

    result
}
```

### Detecting Constant Conditions

The challenge: WebAssembly is stack-based, so we need to track what's on the stack before the branch.

**Approach**:
1. Track the last instruction that produces a value
2. If it's `I32Const`, `I64Const`, etc., we know the condition
3. Remove the constant instruction when simplifying

**Example**:
```wasm
(i32.const 1)
(br_if $label)
```
→ The br_if consumes the i32.const, so we can detect it's constant 1

## Integration with LOOM Pipeline

Branch simplification should run:
1. **After constant propagation** - exposes constant conditions
2. **Before DCE** - creates dead code for DCE to remove
3. **Iteratively** - changes can expose more opportunities

Optimization order:
```
Constant Folding → Branch Simplification → DCE → Block Merging
```

## Test Strategy

### Test Cases

1. **Constant br_if - always taken**
```wasm
(func $test
  (block $exit
    (i32.const 1)
    (br_if $exit)
  )
)
```
Should become: `(br $exit)`

2. **Constant br_if - never taken**
```wasm
(func $test
  (block $exit
    (i32.const 0)
    (br_if $exit)
  )
)
```
Should remove br_if entirely

3. **If with constant condition - true**
```wasm
(func $test (result i32)
  (if (result i32) (i32.const 1)
    (then (i32.const 42))
    (else (i32.const 99))
  )
)
```
Should become: `(i32.const 42)`

4. **If with constant condition - false**
```wasm
(func $test (result i32)
  (if (result i32) (i32.const 0)
    (then (i32.const 42))
    (else (i32.const 99))
  )
)
```
Should become: `(i32.const 99)`

5. **If with identical arms**
```wasm
(func $test (param $x i32) (result i32)
  (if (result i32) (local.get $x)
    (then (i32.const 42))
    (else (i32.const 42))
  )
)
```
Should become: `(drop (local.get $x)) (i32.const 42)`

6. **Nested ifs with constant conditions**
```wasm
(func $test (result i32)
  (if (result i32) (i32.const 1)
    (then
      (if (result i32) (i32.const 0)
        (then (i32.const 10))
        (else (i32.const 20))
      )
    )
    (else (i32.const 30))
  )
)
```
Should become: `(i32.const 20)`

## Implementation Challenges

### 1. Stack-Based Condition Detection

WebAssembly is stack-based, so conditions aren't explicit parameters. We need to:
- Track instruction sequences
- Look back at previous instruction to detect constants
- Handle complex condition expressions

**Solution**: Process instructions in pairs/sequences, recognizing patterns like:
- `I32Const(n)` followed by `BrIf` → constant condition
- `I32Const(n)` followed by `If` → constant condition

### 2. Side Effect Preservation

When removing branches, we must preserve side effects:

```wasm
(call $side_effect_func)
(br_if $label)  ;; Constant 0, never taken
```

We can't remove the call, only the br_if.

**Solution**: Only remove the branch instruction itself, not its condition if it has side effects.

### 3. Type Preservation

Removing if/else must preserve stack types:

```wasm
(if (result i32) ...
  (then (i32.const 42))
  (else (i32.const 99))
)
```

If we select one branch, we must ensure it produces the correct type.

**Solution**: Verify BlockType matches selected branch.

## Performance Expectations

- **Time Complexity**: O(n) - single pass through instructions
- **Space Complexity**: O(n) - rebuilding instruction list
- **Expected Reduction**: 10-15% instruction count in typical code
- **Cascading Effect**: Enables 20-30% more DCE opportunities

## Future Enhancements

1. **Redundant branch detection** - requires control flow graph
2. **Branch merging** - combine multiple br_if to same target
3. **Switch optimization** - simplify br_table patterns
4. **EqZ optimization** - recognize and flip patterns
5. **Inter-block analysis** - optimize across block boundaries

## References

- Binaryen RemoveUnusedBrs.cpp
- Binaryen OptimizeInstructions.cpp (if simplification)
- LLVM SimplifyCFG pass
- "Engineering a Compiler" (Cooper & Torczon), Chapter 10
