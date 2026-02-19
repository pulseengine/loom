# Block Merging Design

## Overview

Block merging combines sequential basic blocks that have no branches between them, reducing CFG complexity and improving code locality. This is a cleanup optimization that runs after DCE and branch simplification to consolidate the resulting linear code sequences.

## Why Block Merging Matters

- **25-35%** of code has mergeable blocks (especially after DCE/branch simplification)
- Reduces CFG complexity → faster interpreter execution
- Improves instruction cache locality
- Reduces block overhead in WebAssembly engines
- Cleanup after optimizations that create sequential blocks

## WebAssembly Block Structure

WebAssembly uses **structured control flow** (not goto-based CFG):

```wasm
(block $outer (result i32)
  (block $inner (result i32)
    (i32.const 42)
  )
  (i32.const 10)
  (i32.add)
)
```

Unlike traditional CFG, WebAssembly blocks:
- **Nest hierarchically** (tree structure)
- Have **no explicit predecessors** (structure defines flow)
- End with implicit "fall-through" to next instruction
- Can be **branch targets** (via label names)

## Types of Mergeable Blocks

### 1. **Nested Blocks** (Most Common)

Sequential blocks where inner block immediately precedes outer block end:

```wasm
;; Before
(block $outer (result i32)
  (block $inner (result i32)
    (i32.const 42)
  )
  (i32.const 10)
  (i32.add)
)

;; After
(block $outer (result i32)
  (i32.const 42)
  (i32.const 10)
  (i32.add)
)
```

**Conditions:**
- Inner block is last (or only significant) instruction in outer
- No branches to `$inner` label
- Types compatible (inner result type = outer input expectation)

### 2. **Sequential Blocks** (After Branch Simplification)

```wasm
;; Before (after constant if folding)
(block $a (result i32)
  (i32.const 42)
)
(block $b (result i32)
  (i32.const 10)
  (i32.add)
)

;; After
(block $merged (result i32)
  (i32.const 42)
  (i32.const 10)
  (i32.add)
)
```

**Conditions:**
- Blocks are sequential in instruction stream
- No branches target intermediate blocks
- Compatible types

### 3. **Single-Instruction Blocks** (Degenerate Case)

```wasm
;; Before
(block $wrapper (result i32)
  (i32.const 42)
)

;; After
(i32.const 42)
```

Remove trivial wrapper blocks entirely.

## Merging Strategy

### Phase 1: Simple Nested Block Merging (Current Implementation)

Focus on the most common case: **nested blocks with no branches**.

**Algorithm:**
1. **Identify mergeable nested blocks**:
   - Find `Block { body }` where last instruction in `body` is another `Block`
   - Check: no branches to inner block label
   - Check: compatible types

2. **Inline inner block contents**:
   - Remove inner `Block` wrapper
   - Append inner instructions to outer block
   - Preserve other instructions

3. **Recursive processing**:
   - Process from innermost to outermost
   - One pass may expose more opportunities

**Example Implementation:**
```rust
fn merge_blocks_in_block(instructions: &[Instruction]) -> Vec<Instruction> {
    let mut result = Vec::new();

    for instr in instructions {
        match instr {
            Instruction::Block { block_type, body } => {
                // Recursively process inner blocks first
                let merged_body = merge_blocks_in_block(body);

                // Check if body ends with a block we can inline
                if can_merge_last_block(&merged_body) {
                    let inlined = inline_last_block(merged_body, block_type);
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: inlined,
                    });
                } else {
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: merged_body,
                    });
                }
            }

            // Recursively process other control flow
            Instruction::Loop { block_type, body } => {
                result.push(Instruction::Loop {
                    block_type: block_type.clone(),
                    body: merge_blocks_in_block(body),
                });
            }

            Instruction::If { block_type, then_body, else_body } => {
                result.push(Instruction::If {
                    block_type: block_type.clone(),
                    then_body: merge_blocks_in_block(then_body),
                    else_body: merge_blocks_in_block(else_body),
                });
            }

            _ => result.push(instr.clone()),
        }
    }

    result
}

fn can_merge_last_block(body: &[Instruction]) -> bool {
    if body.is_empty() {
        return false;
    }

    // Check if last instruction is a block with compatible type
    match body.last() {
        Some(Instruction::Block { block_type, body: inner_body }) => {
            // Only merge unnamed blocks (no branches target them)
            // Check type compatibility
            true // Simplified - needs proper checking
        }
        _ => false,
    }
}

fn inline_last_block(body: Vec<Instruction>, outer_type: &BlockType) -> Vec<Instruction> {
    let mut result = Vec::new();
    let mut body_iter = body.into_iter();

    // Add all but last instruction
    while let Some(instr) = body_iter.next() {
        if body_iter.len() == 0 {
            // This is the last instruction - inline it
            if let Instruction::Block { body: inner_body, .. } = instr {
                result.extend(inner_body);
            }
        } else {
            result.push(instr);
        }
    }

    result
}
```

### Phase 2: Branch Analysis (Future Enhancement)

For more aggressive merging:

1. **Build CFG** - track all branch targets
2. **Identify unreferenced labels** - blocks with no branches
3. **Merge unreferenced blocks** - safe to inline
4. **Update branch targets** - if block was referenced but merged

**CFG Data Structure:**
```rust
struct BlockCFG {
    /// Map from block label to its definition
    blocks: HashMap<String, BlockInfo>,
    /// Map from block label to all branches targeting it
    branch_targets: HashMap<String, Vec<BranchSite>>,
}

struct BlockInfo {
    instructions: Vec<Instruction>,
    predecessors: Vec<String>,  // blocks that can reach this
    successors: Vec<String>,    // blocks this can reach
}
```

### Phase 3: Sequential Block Merging (Future Enhancement)

Merge sequential blocks at function level:

```rust
fn merge_sequential_blocks(instructions: &[Instruction]) -> Vec<Instruction> {
    // Collect consecutive Block instructions
    // If no branches between them, merge into single block
    // Requires full CFG analysis
}
```

## Integration with Optimization Pipeline

Block merging should run:

1. **After DCE** - removes dead code, creating sequential blocks
2. **After branch simplification** - constant folding creates linear flow
3. **Before final output** - cleanup pass

**Pipeline Order:**
```
Constant Folding → Branch Simplification → DCE → Block Merging → Output
```

**Why this order?**
- Branch simplification creates sequential code (constant if → single branch)
- DCE removes unreachable branches (creates linear blocks)
- Block merging consolidates the resulting structure

## Type Safety

Block merging must preserve WebAssembly type correctness:

```wasm
;; INCORRECT merging:
(block (result i32)      ;; Expects i32 result
  (block (result i64)    ;; Produces i64 - TYPE MISMATCH!
    (i64.const 42)
  )
)

;; Only merge if types align:
(block (result i32)
  (block (result i32)    ;; ✓ Compatible
    (i32.const 42)
  )
)
```

**Type Checking:**
- Inner block result type must match outer block's expectation
- Empty blocks (type: `Empty`) can be merged freely
- Multi-value returns require careful analysis

## Edge Cases

### 1. **Blocks with Branches**

```wasm
(block $outer
  (block $inner
    (br $inner)  ;; Branch to inner!
  )
)
```

**Cannot merge** - `$inner` is a branch target.

**Solution:** Phase 1 only merges blocks with no branches. Phase 2 tracks branch targets.

### 2. **Loops**

```wasm
(loop $L
  (block $B
    (i32.const 42)
  )
  (br $L)
)
```

**Can merge** block $B into loop if no branches target it.

### 3. **Nested Control Flow**

```wasm
(block $outer
  (if (local.get 0)
    (then
      (block $inner
        (i32.const 42)
      )
    )
  )
)
```

**Can merge** $inner independently within its context.

### 4. **Empty Blocks**

```wasm
(block $empty)
```

**Remove entirely** - degenerate case.

## Testing Strategy

### Test Cases

1. **Simple nested blocks**
```wasm
(func $test (result i32)
  (block (result i32)
    (block (result i32)
      (i32.const 42)
    )
  )
)
```
Should merge to: `(block (result i32) (i32.const 42))`

2. **Blocks with branches - should NOT merge**
```wasm
(func $test (result i32)
  (block $outer (result i32)
    (block $inner (result i32)
      (br $inner (i32.const 42))
    )
  )
)
```
Should remain unchanged (Phase 1).

3. **Triple nesting**
```wasm
(func $test (result i32)
  (block (result i32)
    (block (result i32)
      (block (result i32)
        (i32.const 42)
      )
    )
  )
)
```
Should fully merge to: `(block (result i32) (i32.const 42))`

4. **Mixed content**
```wasm
(func $test (result i32)
  (block (result i32)
    (i32.const 10)
    (block (result i32)
      (i32.const 32)
      (i32.add)
    )
  )
)
```
Should merge to:
```wasm
(block (result i32)
  (i32.const 10)
  (i32.const 32)
  (i32.add)
)
```

5. **After branch simplification**

Input (after constant if folding):
```wasm
(func $test (result i32)
  (block (result i32)
    (i32.const 42)
  )
  (i32.const 10)
  (i32.add)
)
```

Currently this has block at top level. Phase 1 won't merge.
Phase 2/3 would handle this.

## Performance Expectations

- **Time Complexity**: O(n) - single recursive pass
- **Space Complexity**: O(n) - rebuilding instruction tree
- **Expected Reduction**: 10-20% reduction in block count
- **Cascading Effect**: Enables further optimizations by reducing structure

## Future Enhancements

1. **Full CFG construction** - track all branch targets
2. **Cross-block merging** - merge sequential function-level blocks
3. **Loop tail merging** - extract trailing instructions from loops
4. **Try-catch block handling** - exception handler awareness
5. **Block reordering** - rearrange to create merge opportunities

## References

- Binaryen MergeBlocks.cpp
- WebAssembly structured control flow specification
- LLVM SimplifyCFG pass
- "Engineering a Compiler" (Cooper & Torczon), Chapter 8
