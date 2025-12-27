# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Standard build
cargo build --release

# Build with Z3 verification (requires Z3 installed)
export Z3_SYS_Z3_HEADER=/opt/homebrew/include/z3.h  # macOS with Homebrew
LIBRARY_PATH=/opt/homebrew/lib cargo build --release

# Run all tests
cargo test --release

# Run a single test
cargo test --release test_name

# Run the optimizer
./target/release/loom optimize input.wasm -o output.wasm

# Validate output
wasm-tools validate output.wasm
```

## Architecture

LOOM is a workspace with five crates:

- **loom-shared**: ISLE term definitions and WebAssembly IR (Module, Function, Instruction types)
- **loom-isle**: ISLE pattern matching rules for optimizations, auto-generated from `.isle` files
- **loom-core**: Parser, encoder, 12-phase optimization pipeline, Z3 verification
- **loom-cli**: Command-line interface
- **loom-testing**: Differential testing framework

### Optimization Pipeline Flow

1. Parse WASM → `Module` struct with `Function`s containing `Instruction` vectors
2. Convert instructions to ISLE `Value` terms via `instructions_to_terms()`
3. Apply ISLE rewrite rules via `simplify_with_env()`
4. Convert back to instructions via `terms_to_instructions()`
5. Run optimization passes (DCE, code folding, branch simplification, etc.)
6. Validate with stack checking and optionally Z3 verification
7. Encode back to WASM binary

### Key Files

- `loom-core/src/lib.rs`: Main optimizer with all passes, instruction definitions, parser, encoder
- `loom-core/src/stack.rs`: Stack validation and type checking
- `loom-core/src/verify.rs`: Z3 SMT verification
- `loom-core/src/component_optimizer.rs`: WebAssembly Component Model support
- `loom-shared/isle/wasm_terms.isle`: ISLE term definitions and rewrite rules

### Adding New Instructions

When adding support for a new WebAssembly instruction, update ALL of these locations:
1. `Instruction` enum in `lib.rs`
2. Parser (wasmparser `Operator` → `Instruction`)
3. Encoder (two locations: `encode_wasm` function bodies)
4. `apply_instruction_to_stack` in `stack.rs`
5. `instruction_signature` in `stack.rs`
6. `instructions_to_ir` in `lib.rs` (ISLE term conversion)
7. `instruction_stack_io` in `lib.rs`
8. Z3 verification in `verify.rs`
9. Add to `has_unsupported_isle_instructions` if not ISLE-optimizable

## Core Philosophy: Provably Correct Optimization

LOOM's mission is to be a **provably correct** WebAssembly optimizer. This is not aspirational - it is the fundamental constraint that governs every decision.

### Absolute Requirements

**NOTHING may be skipped, temporarily fixed, or worked around.** There are no exceptions:

1. **No "temporary" fixes** - If something doesn't work correctly, fix it properly or don't implement it at all. There is no "we'll fix it later."
2. **No silent failures** - Every error condition must be handled explicitly with clear error messages
3. **No assumptions** - If we cannot prove something is correct, we must not do it
4. **No "good enough"** - Either an optimization is provably correct, or it is not included
5. **No shortcuts** - Every optimization must have its corresponding proof before it is considered complete

### Constant Vigilance

Before **every action**, ask yourself:
- Does this maintain provable correctness?
- Can we formally verify this transformation preserves semantics?
- Am I cutting any corners?
- Would this pass rigorous formal verification?

Think not twice, but **constantly** - every line of code, every transformation, every decision must be evaluated against the standard of provable correctness.

### Proof-First Development

Optimizations are added **step by step**, always providing the proof **immediately afterwards**:

1. **Design** - Define the transformation formally
2. **Implement** - Write the code
3. **Prove** - Immediately provide Z3 verification or formal proof
4. **Validate** - Run the proof, ensure it passes
5. **Only then** - Consider the optimization complete

**An optimization without its proof is not an optimization - it is a bug waiting to happen.**

An optimization that works "most of the time" is worthless. We need optimizations that work **all of the time**, proven mathematically.

### Conservative Over Fast

When in doubt:
- Skip the function rather than risk incorrect optimization
- Return an error rather than produce potentially wrong output
- Keep the original code rather than apply an unverified transformation

**A correct optimizer that handles 50% of cases is infinitely better than a fast optimizer that corrupts 1% of cases.**

Functions with unsupported instructions are skipped entirely - we do not optimize what we cannot prove. This is the correct behavior, not a limitation to be worked around.

## Testing Philosophy: Real-World Validation

Beyond formal verification, LOOM employs a comprehensive real-world testing strategy:

### Real-World Test Corpus

We maintain a collection of real-world WebAssembly files for validation:
- **User-provided files**: Actual production WASM from users
- **Component Model files**: Files using the Component Model specification
- **Large-scale modules**: Complex modules like `loom.wasm` itself

Every optimization pass must work correctly on ALL test files. If an optimization fails on any real-world file, it is rejected.

### Dogfooding: Self-Optimization

The ultimate validation is **dogfooding** - LOOM optimizes itself:

1. **Build LOOM as WebAssembly Component**: Compile LOOM to a .wasm component
2. **Optimize with LOOM**: Run LOOM to optimize its own WebAssembly binary
3. **Execute the optimized LOOM**: Run the optimized version
4. **Compare outputs**: The optimized LOOM must produce identical optimization results as the original

If the optimized LOOM produces different results than the original, something is wrong. This is the most rigorous test possible - we are literally betting the correctness of our output on our own optimization.

### Test File Locations

- `tests/*.wasm`: User-provided and generated test files
- `loom-testing/`: Differential testing framework
- Real components are tested via `loom optimize <file>.wasm`

### Validation Requirements

For any optimization to be considered complete:
1. ✅ Unit tests pass
2. ✅ Z3 verification passes (for supported instructions)
3. ✅ All real-world test files optimize correctly
4. ✅ Dogfooding produces identical results
