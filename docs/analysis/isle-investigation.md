# ISLE Integration Investigation

## Executive Summary

After extensive investigation, **ISLE cannot be used for LOOM's optimization rules** due to fundamental architectural mismatches.

**IMPORTANT CLARIFICATION**: ISLE is **NOT** a formal verification tool. It's a pattern matching DSL for implementing optimizations. For actual formal verification, LOOM uses **Z3 SMT solver** (see docs/analysis/Z3_VERIFICATION_STATUS.md).

## Summary

ISLE is a term rewriting DSL, not a verification system. LOOM already has Z3-based formal verification that proves optimizations are semantically correct.

## What We Tried

1. **Modular ISLE Structure** ✅
   - Successfully created separate files for types, constructors, and rules
   - Build system correctly compiles multiple ISLE files in sequence
   - Files:
     - `isle/types.isle` - Type definitions (ValueData enum, primitives)
     - `isle/constructors.isle` - Constructor/extractor declarations
     - `isle/rules/default.isle` - Fallback rule
     - `isle/rules/constant_folding.isle` - Attempted optimization rules (FAILED)

2. **Pattern Matching Approaches Attempted** ❌
   - Direct enum variant matching: `(ValueData.I32Add lhs rhs)` - Type errors
   - `if-let` with `value_data` extractor - ISLE compiler panic
   - Multi-value extractors - Parse errors / wrong semantics
   - Nested extractors (`i32_const`, `i64_const`) - ISLE compiler panic

## Root Cause

ISLE **cannot handle recursive primitive/enum structures**:

```rust
// This pattern doesn't work with ISLE:
pub enum ValueData {
    I32Add { lhs: Value, rhs: Value },  // ValueData contains Value
    I32Const { val: Imm32 },
    // ...
}
pub type Value = Box<ValueData>;  // Value is opaque primitive to ISLE
```

### Why This Fails

1. **Value is primitive** - ISLE can't see inside it
2. **ValueData is enum** - Contains Value fields (recursive)
3. **Pattern matching requires unwrapping** - Need to go `Value → ValueData → Value → ValueData`
4. **ISLE can't do multi-level extraction** - The `value_data` extractor + enum variant matching causes internal compiler panics

### What Works in Cranelift

Cranelift's ISLE usage works because:
```rust
// Cranelift pattern - instruction enum is directly accessible
enum HighLevelInst {
    Add(Value, Value),  // Can pattern match directly
    Load(Value),
}
```

They don't have the recursive wrapping we do.

## ISLE Compiler Bugs Encountered

1. **Internal panic**: `Should have been caught by typechecking` at `sema.rs:789`
   - Triggered by: `(if-let (ValueData.I32Add lhs rhs) (value_data val))`
   - This should be a type error, not a panic

2. **Unclear error messages** when mixing primitives and enums in patterns

## Conclusion

**ISLE is NOT suitable for LOOM's optimization passes.**

### Why ISLE Exists

ISLE is designed for:
- **Instruction selection**: High-level IR → Low-level machine instructions
- **Pattern matching on accessible enums**: Not opaque primitive wrappers
- **One-level deep patterns**: Not recursive structures

### What We Should Use Instead

**Manual Rust pattern matching** (what we already have):

```rust
pub fn simplify_stateless(val: &ValueData, env: &impl SimplifyEnv) -> Option<ValueData> {
    match val {
        ValueData::I32Add { lhs, rhs } => {
            if let (ValueData::I32Const { val: x }, ValueData::I32Const { val: y }) =
                (env.get_value_data(lhs), env.get_value_data(rhs)) {
                Some(ValueData::I32Const { val: x.wrapping_add(*y) })
            } else {
                None
            }
        }
        // ... more rules
    }
}
```

This is:
- ✅ **More readable** than ISLE for Rust developers
- ✅ **Type-safe** with compiler checking
- ✅ **Debuggable** with standard Rust tooling
- ✅ **Flexible** - can handle any pattern
- ✅ **No external DSL** to learn
- ✅ **Verified by Z3** - LOOM uses Z3 SMT solver to formally prove optimizations are correct

## What We Achieved

Despite ISLE not working for rules, we successfully:

1. ✅ Created modular ISLE file structure
2. ✅ Defined complete ValueData type system in ISLE
3. ✅ Declared all constructors and extractors
4. ✅ Proved the build system works
5. ✅ Identified ISLE's limitations definitively

## Recommendation

**Keep the manual Rust optimization code.** Do not attempt to migrate to ISLE.

## Formal Verification Status

**LOOM already has formal verification via Z3 SMT solver** (see docs/analysis/Z3_VERIFICATION_STATUS.md):

✅ **What we have**:
- Z3 translation validation in loom-core/src/verify.rs
- Proves optimizations are semantically equivalent for all inputs
- 4 passing verification tests
- CLI integration with `--verify` flag
- Working today with `--features verification`

✅ **Currently verified operations**:
- i32/i64 constants and arithmetic (add, sub, mul)
- i32/i64 bitwise operations (and, or, xor, shl, shr)
- Local variables (get, set, tee)

❌ **Not a verification tool**: ISLE is just a pattern matching DSL, not a formal verification system

## Files Status

- `loom-shared/isle/types.isle` - Complete type definitions ✅
- `loom-shared/isle/constructors.isle` - All constructors declared ✅
- `loom-shared/isle/rules/default.isle` - Identity fallback rule ✅
- `loom-shared/isle/rules/constant_folding.isle` - NOT USED (causes ISLE panic) ❌
- `loom-shared/build.rs` - Modular ISLE compilation setup ✅

The infrastructure is there if we ever find a use case that ISLE can handle, but optimization rules are not it.
