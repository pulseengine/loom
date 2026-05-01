# LOOM Quick Reference

## Command Cheat Sheet

```bash
# Basic optimization
loom optimize input.wat -o output.wasm

# With statistics
loom optimize input.wasm -o output.wasm --stats

# With verification
loom optimize input.wasm -o output.wasm --verify

# Output as WAT
loom optimize input.wasm -o output.wat --wat

# Everything
loom optimize input.wat -o output.wasm --stats --verify
```

## Optimization Pipeline

Library `optimize_module()` (loom-core) runs the production pipeline.
CLI `loom optimize --passes` exposes selectable pass names. Library
order:

| Pass | Optimization | Example Transformation |
|------|-------------|----------------------|
| Function Inlining | Inline small / single-call functions | |
| ISLE Constant Folding | `10 + 20` → `30` | |
| Advanced Instructions | `x * 8` → `x << 3` (strength reduction) | |
| Simplify Locals | Eliminate redundant local sets (RSE) | |
| Dead Code Elimination | Remove unreachable / unused code | |
| Code Folding | Flatten blocks, tail-merge common suffixes | |
| LICM | Hoist loop invariants | |
| Remove Unused Branches | Strip dead branches | |
| Optimize Added Constants | Fold `(x + c1) + c2` → `x + (c1+c2)` | |
| Dead Code Elimination (2nd) | Clean up after LICM/branch removal | |
| Coalesce Locals | Merge non-interfering locals (must run last) | |

Plus a fused-component prelude (memory-import dedup, dead-function elim,
type dedup) when input is a fused Component-Model module.

## Strength Reduction Patterns

| Before | After | Speedup |
|--------|-------|---------|
| `x * 2^n` | `x << n` | ~2-3x |
| `x / 2^n` | `x >> n` | ~2-3x |
| `x % 2^n` | `x & (2^n - 1)` | ~2-3x |
| `x | 0` | `x` | - |
| `x & -1` | `x` | - |
| `x ^ 0` | `x` | - |

## Verification

### Default (Z3 verification on)
```bash
cargo build --release
loom optimize input.wasm -o output.wasm --verify
```
- Z3 SMT translation validation per pass per function
- Reverts a single function on counterexample; reports overall stats
- Z3 verification feature is on by default

### Without Z3 (faster, no semantic proofs)
```bash
cargo build --release --no-default-features
loom optimize input.wasm -o output.wasm --verify
```
- Skips Z3 (verification feature off)
- `--verify` runs ISLE-rule property checks only

If installing Z3 from scratch:
```bash
brew install z3  # macOS
sudo apt install z3  # Linux

# Build with verification
cargo build --release --features verification

# Verify
./target/release/loom optimize input.wasm -o out.wasm --verify
```
- Formal proof via SMT
- Translation validation
- Counterexample generation
- Slower (~100-500ms)

## Performance Metrics

### Typical Results
- **Instruction reduction**: 0-40% (varies by code)
- **Binary size reduction**: 80-95% (consistent)
- **Optimization time**: 10-30 µs (very fast)
- **Parse time**: ~7 µs
- **Encode time**: ~180 ns

### Best Cases
- Math-heavy code: 20-40% instruction reduction
- Lots of constants: High constant folding
- Repeated computations: CSE shines
- Small functions: Inlining effective

### Worst Cases
- Already optimized code: 0% change
- Complex control flow: Limited LICM
- Dynamic computation: Few constant folds

## File Format Support

| Input | Output | Command |
|-------|--------|---------|
| WAT → WASM | Binary | `loom optimize in.wat -o out.wasm` |
| WASM → WASM | Binary | `loom optimize in.wasm -o out.wasm` |
| WAT → WAT | Text | `loom optimize in.wat -o out.wat --wat` |
| WASM → WAT | Text | `loom optimize in.wasm -o out.wat --wat` |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Failed to parse" | Validate with `wasm-validate` |
| No Z3 verification | Build with `--features verification` |
| Slow optimization | Normal for large modules (still <1ms usually) |
| Binary got bigger | Rare; file issue with test case |
| Verification failed | Possible optimizer bug; file issue |

## Integration Examples

### Makefile
```makefile
optimize:
\t@for f in dist/*.wasm; do \\
\t\tloom optimize $$f -o $$f --stats; \\
\tdone
```

### NPM Script
```json
{
  "scripts": {
    "optimize": "loom optimize build/app.wasm -o build/app.wasm --stats"
  }
}
```

### Rust Build Script
```rust
use std::process::Command;

fn main() {
    Command::new("loom")
        .args(&["optimize", "target/wasm.wasm", "-o", "dist/optimized.wasm"])
        .status()
        .expect("LOOM optimization failed");
}
```

## Statistics Explained

```
Instructions: 24 → 20 (16.7% reduction)
```
- Fewer instructions = faster execution
- 10-20% is good, 20%+ is excellent

```
Binary size: 797 → 92 bytes (88.5% reduction)
```
- Smaller = faster download/parse
- 80%+ reduction is typical

```
Constant folds: 3
```
- Number of compile-time evaluations
- More is better (enables other optimizations)

```
Optimization time: 0 ms
```
- Total optimization duration
- Should be <1ms for most modules

## Advanced Usage

### Custom Pipeline (Future)
```bash
# Not yet implemented
loom optimize input.wasm -o out.wasm --passes=cse,inline,dce
```

### Profile-Guided Optimization (Future)
```bash
# Not yet implemented
loom optimize input.wasm -o out.wasm --profile=usage.prof
```

### Optimization Levels (Future)
```bash
# Not yet implemented
loom optimize input.wasm -o out.wasm --level=3
```

## Limitations

**Current limitations:**
- LICM only hoists constants and unmodified locals
- No inter-procedural analysis (yet)
- Limited floating-point optimizations
- No SIMD-specific optimizations
- Component model support is basic

**Coming soon:**
- More aggressive LICM
- Profile-guided optimization
- Custom optimization passes
- SIMD optimizations
- Enhanced component support

## Resources

- **Full Guide**: [usage.md](usage.md)
- **Verification**: [formal-verification.md](formal-verification.md)
- **WASM Build**: [wasm-build.md](wasm-build.md)

## Quick Tips

1. **Always use --stats** to see what was optimized
2. **Use --verify for production** to ensure correctness
3. **WAT format is slower** to parse than binary
4. **Optimization is idempotent** - safe to run multiple times
5. **Binary size reduction is more consistent** than instruction count
6. **File issues with test cases** if something seems wrong

## Version Info

- LOOM Version: 0.1.0
- ISLE Integration: Yes
- Z3 Support: Optional (feature flag)
- Component Model: Basic support
- Bazel Build: Yes
- wasm32-wasip2: Yes
