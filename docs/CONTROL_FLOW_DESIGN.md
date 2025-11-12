# Control Flow Representation in ISLE

## Overview

This document describes the design for adding WebAssembly control flow constructs to LOOM's ISLE term representation.

## Design Principles

1. **Structured Control Flow**: WebAssembly uses structured control flow (blocks must be properly nested)
2. **Type Safety**: All blocks have block types that describe stack input/output
3. **Label Targeting**: Branches use relative label depths (0 = innermost block)
4. **Value Semantics**: Control flow constructs are expressions that produce values

## WebAssembly Control Flow Model

### Block Types

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BlockType {
    Empty,           // [] -> []
    Value(ValueType), // [] -> [type]
    Func(Vec<ValueType>, Vec<ValueType>), // [params] -> [results]
}
```

### Label Semantics

- Label depth 0 = current (innermost) block
- Label depth 1 = parent block
- Label depth n = n-th ancestor block
- `block` branches jump PAST the end (forward)
- `loop` branches jump TO the start (backward)

## ISLE Term Design

### Core Principle: Sequence vs Expression

WebAssembly has two contexts:
1. **Expression context**: Single value on stack (e.g., `i32.add`)
2. **Sequence context**: Multiple instructions, side effects (e.g., `block`, `if`)

**Decision**: Extend `ValueData` with control flow that can contain sequences.

### Representation

```rust
pub enum ValueData {
    // ... existing variants ...

    /// Block: structured control that can be branched to
    /// Semantics: Execute body, branches to label jump past end
    Block {
        /// Optional label for debugging
        label: Option<String>,
        /// Block type (input/output signature)
        block_type: BlockType,
        /// Body instructions (sequence)
        body: Vec<Value>,
    },

    /// Loop: structured control where branches restart
    /// Semantics: Execute body, branches to label jump to start
    Loop {
        label: Option<String>,
        block_type: BlockType,
        body: Vec<Value>,
    },

    /// If-then-else conditional
    /// Semantics: Pop condition, execute then or else branch
    If {
        label: Option<String>,
        block_type: BlockType,
        condition: Value,
        then_body: Vec<Value>,
        else_body: Vec<Value>, // empty for if without else
    },

    /// Unconditional branch to label
    /// Semantics: Jump to target, unwind stack to block entry
    Br {
        /// Relative label depth (0 = innermost)
        depth: u32,
        /// Value to leave on stack (if block expects result)
        /// None for blocks with no result
        value: Option<Value>,
    },

    /// Conditional branch
    /// Semantics: Pop i32 condition, if non-zero branch
    BrIf {
        depth: u32,
        condition: Value,
        value: Option<Value>,
    },

    /// Branch table (switch/case)
    /// Semantics: Pop i32 index, branch to targets[index] or default
    BrTable {
        /// List of target label depths
        targets: Vec<u32>,
        /// Default label depth
        default: u32,
        /// Index to select target
        index: Value,
        /// Value to pass (if blocks expect results)
        value: Option<Value>,
    },

    /// Return from function
    /// Semantics: Return from function with value(s)
    Return {
        /// Return values (match function signature)
        values: Vec<Value>,
    },

    /// Function call (direct)
    Call {
        /// Function index
        func_idx: u32,
        /// Arguments
        args: Vec<Value>,
    },

    /// Function call (indirect through table)
    CallIndirect {
        /// Table index
        table_idx: u32,
        /// Type index (for signature checking)
        type_idx: u32,
        /// Table offset (which function in table)
        table_offset: Value,
        /// Arguments
        args: Vec<Value>,
    },

    /// Unreachable - traps execution
    Unreachable,

    /// Nop - no operation
    Nop,
}
```

## Sequence Representation

**Key Design Decision**: `Vec<Value>` for instruction sequences.

This allows:
- Multiple instructions in block bodies
- Side effects (stores, calls) in sequences
- Empty sequences for empty blocks

Example:
```rust
Block {
    label: None,
    block_type: BlockType::Value(ValueType::I32),
    body: vec![
        iconst32(10),      // First instruction
        iconst32(20),      // Second instruction
        iadd32(..),        // Third instruction (result)
    ],
}
```

## Type System Integration

### BlockType in loom-isle

```rust
// Add to loom-isle/src/lib.rs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BlockType {
    /// No parameters, no results
    Empty,
    /// No parameters, single result
    Value(ValueType),
    /// Full function signature (for multi-value blocks)
    Func {
        params: Vec<ValueType>,
        results: Vec<ValueType>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    I32,
    I64,
    F32,
    F64,
}
```

### Integration with loom-core

loom-core already has `ValueType` - we'll unify these types.

## ISLE DSL Terms

Add to `loom-isle/isle/wasm_terms.isle`:

```isle
;; Control flow block types
(type BlockType (enum
  (Empty)
  (Value (val ValueType))
  (Func (params (list ValueType)) (results (list ValueType)))
))

;; Control flow constructs
(decl block (label (option String)) (block_type BlockType) (body (list Value)) Value)
(decl loop (label (option String)) (block_type BlockType) (body (list Value)) Value)
(decl if_then_else (label (option String)) (block_type BlockType)
                   (condition Value) (then_body (list Value)) (else_body (list Value)) Value)

(decl br (depth u32) (value (option Value)) Value)
(decl br_if (depth u32) (condition Value) (value (option Value)) Value)
(decl br_table (targets (list u32)) (default u32) (index Value) (value (option Value)) Value)

(decl return_val (values (list Value)) Value)
(decl call (func_idx u32) (args (list Value)) Value)
(decl call_indirect (table_idx u32) (type_idx u32) (offset Value) (args (list Value)) Value)

(decl unreachable () Value)
(decl nop () Value)

;; Extern constructors
(extern constructor block block)
(extern constructor loop loop)
(extern constructor if_then_else if_then_else)
(extern constructor br br)
(extern constructor br_if br_if)
(extern constructor br_table br_table)
(extern constructor return_val return_val)
(extern constructor call call)
(extern constructor call_indirect call_indirect)
(extern constructor unreachable unreachable)
(extern constructor nop nop)
```

## Optimization Rules (Future)

Once representation is in place, we can add optimization rules:

```isle
;; Constant condition folding
(rule (if_then_else label block_type (iconst32 0) then_body else_body)
      (else_body))  ;; Condition is false, take else

(rule (if_then_else label block_type (iconst32 n) then_body else_body)
      (then_body))  ;; Condition is non-zero, take then (when n != 0)

;; Branch to depth 0 at end of block = remove (fall through)
(rule (block label block_type (append prefix (br 0 value)))
      (block label block_type prefix))  ;; Remove redundant branch

;; Unreachable code after unconditional branch
(rule (append (br depth value) rest)
      (br depth value))  ;; Remove rest (unreachable)
```

## Parser Changes

### loom-core Instruction enum

Extend to include control flow:

```rust
pub enum Instruction {
    // ... existing variants ...

    /// Block with typed body
    Block {
        block_type: BlockType,
        body: Vec<Instruction>,
    },

    /// Loop with typed body
    Loop {
        block_type: BlockType,
        body: Vec<Instruction>,
    },

    /// If-then-else
    If {
        block_type: BlockType,
        then_body: Vec<Instruction>,
        else_body: Vec<Instruction>,
    },

    /// Unconditional branch
    Br(u32),

    /// Conditional branch
    BrIf(u32),

    /// Branch table
    BrTable {
        targets: Vec<u32>,
        default: u32,
    },

    /// Return from function
    Return,

    /// Direct call
    Call(u32),

    /// Indirect call
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
    },

    /// Trap
    Unreachable,

    /// No-op
    Nop,
}
```

### Parser Implementation

Key challenge: **Nested structure parsing**

WebAssembly binary format encodes blocks as:
```
block <blocktype>
  <instructions>
end
```

The parser must:
1. Track nesting depth
2. Accumulate instructions in current block
3. Handle `end` to close blocks
4. Handle `else` to switch from then to else

```rust
fn parse_control_flow(reader: &mut OperatorsReader) -> Result<Vec<Instruction>> {
    let mut instrs = Vec::new();
    let mut depth = 0;

    for op in reader {
        match op? {
            Operator::Block { blockty } => {
                // Start new block, increment depth
                depth += 1;
                // Parse nested instructions recursively
            }
            Operator::End => {
                if depth == 0 {
                    break; // End of this block
                }
                depth -= 1;
            }
            // ... handle other operators
        }
    }

    Ok(instrs)
}
```

## Encoder Changes

Emit control flow instructions:

```rust
fn encode_instruction(instr: &Instruction, encoder: &mut Function) {
    match instr {
        Instruction::Block { block_type, body } => {
            encoder.instruction(&EncoderInstruction::Block(
                convert_block_type(block_type)
            ));
            for nested in body {
                encode_instruction(nested, encoder);
            }
            encoder.instruction(&EncoderInstruction::End);
        }
        // ... similar for Loop, If, etc.
    }
}
```

## Testing Strategy

### Unit Tests

1. **Round-trip tests**: Parse then encode control flow
2. **Nesting tests**: Deeply nested blocks
3. **Branch target tests**: Correct label depths
4. **Type tests**: Block types preserved

### Integration Tests

WAT examples:
```wat
;; Simple block
(func $test_block (result i32)
  (block (result i32)
    i32.const 42
  )
)

;; Nested blocks with branches
(func $test_nested (param $x i32) (result i32)
  (block (result i32)
    (block
      (br_if 0 (local.get $x))  ;; branch to inner block end
      (br 1 (i32.const 1))       ;; branch to outer block end
    )
    i32.const 0
  )
)

;; Loop
(func $test_loop (param $n i32) (result i32)
  (local $i i32)
  (loop (result i32)
    (local.set $i (i32.add (local.get $i) (i32.const 1)))
    (br_if 0 (i32.lt_u (local.get $i) (local.get $n)))
    (local.get $i)
  )
)
```

## Implementation Phases

### Phase 1: Data Structures (Week 1)
- [ ] Add `BlockType` to loom-isle
- [ ] Extend `ValueData` with control flow variants
- [ ] Add ISLE terms to wasm_terms.isle
- [ ] Implement constructor functions

### Phase 2: Parser (Week 1-2)
- [ ] Extend `Instruction` enum in loom-core
- [ ] Implement nested block parsing
- [ ] Handle block types from wasmparser
- [ ] Add parser tests

### Phase 3: Encoder (Week 2)
- [ ] Implement control flow encoding
- [ ] Handle nested structure emission
- [ ] Add encoder tests

### Phase 4: Term Conversion (Week 2-3)
- [ ] instructions_to_terms for control flow
- [ ] terms_to_instructions for control flow
- [ ] Handle label depth tracking
- [ ] Add round-trip tests

### Phase 5: Integration Testing (Week 3)
- [ ] End-to-end tests with WAT files
- [ ] Validation with wasm-tools
- [ ] Performance testing

## Future Work

Once control flow is in place:
- Dead code elimination (#13)
- Branch simplification (#16)
- Block merging (#17)
- Loop invariant code motion
- Inlining (#14)

## References

- [WebAssembly Spec: Control Instructions](https://webassembly.github.io/spec/core/syntax/instructions.html#control-instructions)
- [wasmparser documentation](https://docs.rs/wasmparser/)
- [wasm-encoder documentation](https://docs.rs/wasm-encoder/)
