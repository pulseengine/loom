# LOOM Usage Guide

Complete guide to using the LOOM WebAssembly optimizer.

## Table of Contents
- [Quick Start](#quick-start)
- [Command-Line Interface](#command-line-interface)
- [Optimization Features](#optimization-features)
- [Verification](#verification)
- [Performance Tips](#performance-tips)
- [Examples](#examples)

## Quick Start

### Installation

```bash
# Build from source
cargo build --release

# The binary will be at target/release/loom
./target/release/loom --help
```

### Basic Usage

```bash
# Optimize a WAT file
loom optimize input.wat --output output.wasm

# Optimize a WASM binary
loom optimize input.wasm --output optimized.wasm

# Show optimization statistics
loom optimize input.wat --output output.wasm --stats

# Enable verification
loom optimize input.wat --output output.wasm --verify

# Output WAT text format instead of binary
loom optimize input.wasm --output output.wat --wat
```

## Command-Line Interface

### Optimize Command

```bash
loom optimize <INPUT> [OPTIONS]

Arguments:
  <INPUT>          Input WebAssembly file (.wat or .wasm)

Options:
  -o, --output <OUTPUT>    Output file path (default: output.wasm)
  --wat                    Output WAT text format instead of binary
  --stats                  Show detailed optimization statistics
  --verify                 Run verification checks after optimization
```

### Example Commands

```bash
# Optimize and show stats
loom optimize my_module.wat --output optimized.wasm --stats

# Optimize and verify correctness
loom optimize input.wasm -o output.wasm --verify

# Convert WASM to optimized WAT
loom optimize input.wasm --output output.wat --wat
```

## Optimization Features

LOOM applies a comprehensive 12-phase optimization pipeline:

### Phase 1: Precompute (Global Constant Propagation)
- Replaces immutable global.get with constants
- Enables further constant folding

### Phase 2: ISLE-Based Constant Folding
- Folds constant arithmetic at compile time
- Example: `i32.const 10 + i32.const 20` → `i32.const 30`

### Phase 3: Advanced Instruction Optimizations
- **Strength Reduction**: Replaces expensive operations with cheaper equivalents
  - `x * 8` → `x << 3` (multiply to shift)
  - `x / 4` → `x >> 2` (divide to shift)
  - `x % 32` → `x & 31` (modulo to bitwise AND)
- **Bitwise Optimizations**:
  - `x | 0` → `x`
  - `x & -1` → `x`
  - `x ^ 0` → `x`

### Phase 4: Common Subexpression Elimination (CSE)
- Caches duplicate computations in local variables
- Skips simple constants (they're cheap)
- Example:
  ```wasm
  ;; Before
  local.get $x
  i32.const 4
  i32.mul
  local.get $x
  i32.const 4
  i32.mul  ;; Duplicate!

  ;; After
  local.get $x
  i32.const 4
  i32.mul
  local.tee $temp
  local.get $temp  ;; Reuse cached result
  ```

### Phase 5: Function Inlining
- Inlines small, frequently-called functions
- Reduces call overhead
- Enables further optimizations across function boundaries

### Phase 6: ISLE Optimization (Post-Inlining)
- Second pass of constant folding
- Optimizes code exposed by inlining

### Phase 7: Code Folding
- Flattens nested blocks
- Removes unnecessary control flow structures

### Phase 8: Loop-Invariant Code Motion (LICM)
- Hoists loop-invariant computations outside loops
- Currently handles:
  - Constants
  - Unmodified local variables

### Phase 9: Branch Simplification
- Simplifies conditional branches
- Removes redundant conditions

### Phase 10: Dead Code Elimination (DCE)
- Removes unreachable code after terminators
- Cleans up code that never executes

### Phase 11: Block Merging
- Merges consecutive blocks
- Reduces control flow overhead

### Phase 12: Vacuum & Simplify Locals
- Removes empty blocks
- Eliminates unused local variables

## Verification

LOOM supports two types of verification:

### 1. ISLE Property-Based Verification (Always Available)
- Checks idempotence (running optimization twice gives same result)
- Validates constant folding correctness
- Fast and lightweight

### 2. Z3 SMT-Based Formal Verification (Optional)
- Proves optimization correctness via translation validation
- Requires Z3 to be installed

#### Installing Z3

**macOS (Homebrew):**
```bash
brew install z3
```

**Ubuntu/Debian:**
```bash
sudo apt-get install z3
```

**From source:**
```bash
git clone https://github.com/Z3Prover/z3
cd z3
python scripts/mk_make.py
cd build
make
sudo make install
```

#### Using Z3 Verification

```bash
# Build LOOM with verification support
cargo build --release --features verification

# Run with verification
./target/release/loom optimize input.wat -o output.wasm --verify
```

**Expected output:**
```
🔍 Running verification...
🔬 Running Z3 SMT verification...
  ✅ Z3 verification passed: optimizations are semantically equivalent
🧪 Running ISLE property-based verification...
  Running property tests...
✓ Verification: 38/38 tests passed
✓ All verification tests passed!
```

If verification fails, LOOM will show:
```
  ❌ Z3 verification failed: counterexample found
  ⚠️  The optimization may have changed program semantics!
```

## Performance Tips

### 1. Use Statistics to Guide Optimization

```bash
loom optimize input.wasm -o output.wasm --stats
```

Look for:
- **Instruction count reduction**: Fewer instructions = faster execution
- **Binary size reduction**: Smaller = faster download/load
- **Constant folds**: More folds = more optimization opportunities

### 2. Verify Critical Code

```bash
# Always verify safety-critical or production code
loom optimize production.wasm -o prod.opt.wasm --verify --stats
```

### 3. Benchmark Before and After

```bash
# Optimize
loom optimize app.wasm -o app.opt.wasm --stats

# Compare sizes
ls -lh app.wasm app.opt.wasm

# Test performance with your WebAssembly runtime
```

### 4. Iterative Optimization

LOOM's optimizations are idempotent - running optimization multiple times produces the same result:

```bash
loom optimize input.wasm -o pass1.wasm
loom optimize pass1.wasm -o pass2.wasm
# pass1.wasm and pass2.wasm should be identical
```

## Examples

### Example 1: Optimizing Math-Heavy Code

**Input (math.wat):**
```wasm
(module
  (func $calculate (param $x i32) (result i32)
    ;; Multiply by 8 (expensive)
    local.get $x
    i32.const 8
    i32.mul

    ;; Divide by 4 (expensive)
    i32.const 4
    i32.div_u

    ;; Add constants (can be folded)
    i32.const 10
    i32.const 20
    i32.add
    i32.add
  )
)
```

**Optimize:**
```bash
loom optimize math.wat -o math.opt.wasm --stats
```

**Output:**
```
Instructions: 9 → 5 (44.4% reduction)
Binary size:  156 → 98 bytes (37.2% reduction)
Constant folds: 1
Optimization time: 0 ms
```

**What happened:**
- `x * 8` → `x << 3` (strength reduction)
- `/ 4` → `>> 2` (strength reduction)
- `10 + 20` → `30` (constant folding)

### Example 2: Optimizing with Verification

**Input (critical.wat):**
```wasm
(module
  (func $process (param $data i32) (result i32)
    local.get $data
    i32.const 16
    i32.div_u
    i32.const 100
    i32.add
  )
)
```

**Optimize with verification:**
```bash
cargo build --release --features verification
./target/release/loom optimize critical.wat -o critical.opt.wasm --verify --stats
```

**Output:**
```
✓ Parsed in 234µs
⚡ Optimizing...
✓ Optimized in 0 ms

🔍 Running verification...
🔬 Running Z3 SMT verification...
  ✅ Z3 verification passed: optimizations are semantically equivalent
🧪 Running ISLE property-based verification...
✓ Verification: 38/38 tests passed

Instructions: 5 → 4 (20.0% reduction)
Binary size:  120 → 95 bytes (20.8% reduction)
```

### Example 3: Batch Optimization

**Optimize multiple files:**
```bash
#!/bin/bash
for file in modules/*.wasm; do
  echo "Optimizing $file..."
  loom optimize "$file" -o "optimized/$(basename $file)" --stats
done
```

### Example 4: CI/CD Integration

**GitHub Actions workflow:**
```yaml
name: Optimize WebAssembly

on: [push]

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Build LOOM
        run: cargo build --release

      - name: Optimize WASM modules
        run: |
          for wasm in dist/*.wasm; do
            ./target/release/loom optimize "$wasm" -o "$wasm" --stats --verify
          done

      - name: Upload optimized modules
        uses: actions/upload-artifact@v2
        with:
          name: optimized-wasm
          path: dist/*.wasm
```

## Benchmark Results

Based on comprehensive benchmarking:

| Optimization Phase | Average Time | Description |
|-------------------|-------------|-------------|
| Constant Folding | 8-11 µs | Fast and effective |
| Strength Reduction | 10 µs | Consistent performance |
| CSE | 9-14 µs | Scales with complexity |
| Function Inlining | 16-18 µs | Moderate overhead |
| Full Pipeline | 19-28 µs | Complete optimization |
| Parser | 6.8 µs | Very fast |
| Encoder | 183 ns | Extremely fast |

### Real-World Results

| Fixture | Instruction Reduction | Binary Size Reduction |
|---------|---------------------|---------------------|
| bench_bitops | 16.7% | 88.5% |
| test_input | 22.2% | 81.6% |
| fibonacci | 0% | 92.6% |
| quicksort | varies | 92.5% |
| game_logic | varies | 92.5% |

**Key Takeaway:** Binary size reductions are consistently excellent (80-93%), while instruction count improvements vary by code complexity.

## Troubleshooting

### "Optimization failed"

Check that your input is valid WebAssembly:
```bash
# Validate with wabt tools
wasm-validate input.wasm

# Or try parsing without optimization
loom optimize input.wasm -o test.wasm
```

### Z3 verification not available

If you see "Z3 verification feature not enabled":
```bash
# Rebuild with verification feature
cargo clean
cargo build --release --features verification
```

### Performance degradation

If optimized code is slower:
1. Check instruction count - it should decrease or stay same
2. Profile with your WebAssembly runtime
3. Some optimizations trade instructions for code size
4. File an issue with your test case

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: https://github.com/pulseengine/loom/issues
- **Examples**: See `tests/fixtures/` for sample WASM files

## Next Steps

- Read [formal-verification.md](formal-verification.md) for detailed verification info
- Check [wasm-build.md](wasm-build.md) for building loom to WebAssembly
- Explore the fixtures in `tests/fixtures/` for real-world examples
